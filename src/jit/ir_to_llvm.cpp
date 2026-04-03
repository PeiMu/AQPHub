/**
 * ir_to_llvm.cpp — AQP IR → LLVM IR → native machine code compiler.
 *
 * Walks SimplestFilter/SimplestExpr IR trees and emits LLVM IR that
 * evaluates the filter over an AQPChunkView (batch of 2048 rows) and
 * writes matching row indices to an AQPSelView.
 *
 * Compiled function signature (C ABI):
 *   uint64_t aqp_expr_<hash>(AQPChunkView* chunk, AQPSelView* sel);
 *
 * Returns: number of selected rows (written to sel->indices[0..ret-1]).
 */

#include "jit/ir_to_llvm.h"
#include "jit/aqp_jit_abi.h"

// LLVM headers — only included in this TU
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Constants.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>

#include "simplest_ir.h"

// Forward-declare C-linkage runtime helpers (defined in aqp_jit_runtime.cpp).
// Must come before Impl::Impl() which takes their addresses.
extern "C" {
    int aqp_like_match(const char *str, int32_t slen, const char *pat, int32_t plen);
    int aqp_ilike_match(const char *str, int32_t slen, const char *pat, int32_t plen);
    int aqp_str_eq(const char *a, int32_t alen, const char *b, int32_t blen);
    int aqp_in_set_i32(int32_t val, const int32_t *values, int32_t n);
    int aqp_in_set_i64(int64_t val, const int64_t *values, int32_t n);
    int aqp_in_set_str(const char *str, int32_t slen, const char **ptrs,
                       const int32_t *lens, int32_t n);
}

#include <cstdint>
#include <functional>
#include <atomic>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Monotonically-increasing counter used to generate unique function names.
// Using pointer/hash produced duplicate names when two filters in the same
// sub-query happened to hash to the same value within the same LLJIT dylib.
static std::atomic<uint64_t> s_filter_counter{0};

using namespace llvm;
using namespace llvm::orc;
using namespace ir_sql_converter;

namespace aqp_jit {

// ---------------------------------------------------------------------------
// LLVM initialisation (done once per process)
// ---------------------------------------------------------------------------
static bool llvm_initialized = false;
static void EnsureLLVMInit() {
    if (llvm_initialized) return;
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();
    llvm_initialized = true;
}

// ---------------------------------------------------------------------------
// Helper: FNV-1a hash of a string — used to generate unique function names
// ---------------------------------------------------------------------------
static uint64_t FNV1a(const std::string &s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

// ---------------------------------------------------------------------------
// Pimpl implementation
// ---------------------------------------------------------------------------
struct IrToLlvmCompiler::Impl {
    std::unique_ptr<LLJIT> jit;

    // Compiled modules kept alive so their symbols remain valid
    // (function pointers remain valid for the lifetime of Impl)

    Impl() {
        EnsureLLVMInit();
        auto jit_or = LLJITBuilder().create();
        if (!jit_or) {
            std::string msg;
            raw_string_ostream ss(msg);
            ss << "Failed to create ORC LLJIT: ";
            logAllUnhandledErrors(jit_or.takeError(), ss);
            throw std::runtime_error(ss.str());
        }
        jit = std::move(*jit_or);

        // Make runtime helper symbols (aqp_like_match etc.) visible to JIT.
        // LLVM 14 uses JITEvaluatedSymbol(JITTargetAddress, JITSymbolFlags).
        auto &es = jit->getExecutionSession();
        auto &jd = jit->getMainJITDylib();
        (void)jd.define(absoluteSymbols({
            {es.intern("aqp_like_match"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_like_match),
                                JITSymbolFlags::Exported)},
            {es.intern("aqp_ilike_match"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_ilike_match),
                                JITSymbolFlags::Exported)},
            {es.intern("aqp_str_eq"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_str_eq),
                                JITSymbolFlags::Exported)},
            {es.intern("aqp_in_set_i32"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_in_set_i32),
                                JITSymbolFlags::Exported)},
            {es.intern("aqp_in_set_i64"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_in_set_i64),
                                JITSymbolFlags::Exported)},
            {es.intern("aqp_in_set_str"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_in_set_str),
                                JITSymbolFlags::Exported)},
        }));
    }
};

// ---------------------------------------------------------------------------
// Per-compilation context — holds the LLVM module and IR builder state
// ---------------------------------------------------------------------------
struct CompileCtx {
    LLVMContext           &llctx;
    Module                &mod;
    IRBuilder<>            b;
    const std::vector<ColSchema> &schema;

    // LLVM struct types matching aqp_jit_abi.h
    StructType *AQPColViewTy;   // { i8*, i64*, i32, i32 }
    StructType *AQPChunkViewTy; // { AQPColView*, i64, i64 }
    StructType *AQPSelViewTy;   // { i32*, i32 }

    // Function arguments
    Value *chunk_arg;  // AQPChunkView*
    Value *sel_arg;    // AQPSelView*

    // Per-row loop variables (set inside the loop body)
    Value *row_idx    = nullptr; // i64 — current row index

    // Column data and validity pointers (loaded once before loop)
    std::vector<Value*> col_data;     // one void* per column
    std::vector<Value*> col_validity; // one i64* per column (may be null ptr)

    CompileCtx(LLVMContext &ctx, Module &m, const std::vector<ColSchema> &s,
               Value *chunk, Value *sel)
        : llctx(ctx), mod(m), b(ctx), schema(s),
          chunk_arg(chunk), sel_arg(sel) {

        // Build struct types
        Type *i8p  = PointerType::getUnqual(Type::getInt8Ty(ctx));
        Type *i32  = Type::getInt32Ty(ctx);
        Type *i64  = Type::getInt64Ty(ctx);
        Type *i64p = PointerType::getUnqual(i64);

        AQPColViewTy   = StructType::get(ctx, {i8p, i64p, i32, i32});
        AQPChunkViewTy = StructType::get(ctx, {
            PointerType::getUnqual(AQPColViewTy), i64, i64});
        // sel.indices is sel_t* = uint32_t* in DuckDB (typedefs.hpp: typedef uint32_t sel_t)
        AQPSelViewTy   = StructType::get(ctx, {
            PointerType::getUnqual(i32), i32});
    }

    Type *i8p()  { return PointerType::getUnqual(Type::getInt8Ty(llctx)); }
    Type *i32()  { return Type::getInt32Ty(llctx); }
    Type *i64()  { return Type::getInt64Ty(llctx); }
    Type *f32()  { return Type::getFloatTy(llctx); }
    Type *f64()  { return Type::getDoubleTy(llctx); }
    Type *i1()   { return Type::getInt1Ty(llctx); }

    ConstantInt *c32(int32_t v)  { return ConstantInt::get(llctx, APInt(32, (uint64_t)v, true)); }
    ConstantInt *c64(int64_t v)  { return ConstantInt::get(llctx, APInt(64, (uint64_t)v, true)); }
    ConstantInt *ci(int v, int bits) { return ConstantInt::get(llctx, APInt(bits, v)); }

    // Load col data pointer for column col_chunk_idx (index into AQPChunkView)
    Value *LoadColData(unsigned col_chunk_idx) {
        // AQPChunkView.cols[col_chunk_idx].data
        Value *cols_ptr = b.CreateStructGEP(AQPChunkViewTy, chunk_arg, 0);
        Value *cols     = b.CreateLoad(
            PointerType::getUnqual(AQPColViewTy), cols_ptr, "cols");
        Value *col_i    = b.CreateGEP(AQPColViewTy, cols,
            c64((int64_t)col_chunk_idx), "col");
        Value *data_pp  = b.CreateStructGEP(AQPColViewTy, col_i, 0);
        return b.CreateLoad(i8p(), data_pp, "data");
    }

    Value *LoadColValidity(unsigned col_chunk_idx) {
        Value *cols_ptr = b.CreateStructGEP(AQPChunkViewTy, chunk_arg, 0);
        Value *cols     = b.CreateLoad(
            PointerType::getUnqual(AQPColViewTy), cols_ptr);
        Value *col_i    = b.CreateGEP(AQPColViewTy, cols, c64((int64_t)col_chunk_idx));
        Value *val_pp   = b.CreateStructGEP(AQPColViewTy, col_i, 1);
        return b.CreateLoad(
            PointerType::getUnqual(i64()), val_pp, "validity");
    }

    // Find the AQPChunkView column index for the given IR attribute.
    // Returns -1 if not found.
    int FindColIdx(const SimplestAttr &attr) const {
        for (int i = 0; i < (int)schema.size(); i++) {
            if (schema[i].table_idx == attr.GetTableIndex() &&
                schema[i].col_idx  == attr.GetColumnIndex())
                return i;
        }
        return -1;
    }
};

// ---------------------------------------------------------------------------
// Expression emission (forward declaration)
// ---------------------------------------------------------------------------
static Value *EmitExpr(CompileCtx &cc, const AQPExpr *expr);

// Check validity bit for row cc.row_idx in the validity array ptr.
// Returns i1: true = valid (not null), false = null.
static Value *EmitValidityCheck(CompileCtx &cc, Value *validity_ptr) {
    // validity_ptr may be nullptr (all-valid) — checked by caller
    Value *word_idx = cc.b.CreateLShr(cc.row_idx, cc.c64(6), "word_idx");
    Value *bit_idx  = cc.b.CreateAnd(cc.row_idx, cc.c64(63), "bit_idx");
    Value *word_ptr = cc.b.CreateGEP(cc.i64(), validity_ptr, word_idx, "word_ptr");
    Value *word     = cc.b.CreateLoad(cc.i64(), word_ptr, "word");
    Value *shifted  = cc.b.CreateLShr(word, bit_idx, "shifted");
    Value *bit      = cc.b.CreateAnd(shifted, cc.c64(1), "bit");
    return cc.b.CreateICmpNE(bit, cc.c64(0), "valid");
}

// Load an INT32 value from a flat column array at row index row_idx
static Value *LoadI32(CompileCtx &cc, Value *data_ptr) {
    Value *p32  = cc.b.CreateBitCast(data_ptr, PointerType::getUnqual(cc.i32()));
    Value *elem = cc.b.CreateGEP(cc.i32(), p32, cc.row_idx, "elem_ptr");
    return cc.b.CreateLoad(cc.i32(), elem, "val_i32");
}
static Value *LoadI64(CompileCtx &cc, Value *data_ptr) {
    Value *p64  = cc.b.CreateBitCast(data_ptr, PointerType::getUnqual(cc.i64()));
    Value *elem = cc.b.CreateGEP(cc.i64(), p64, cc.row_idx, "elem_ptr");
    return cc.b.CreateLoad(cc.i64(), elem, "val_i64");
}
static Value *LoadF32(CompileCtx &cc, Value *data_ptr) {
    Value *pf   = cc.b.CreateBitCast(data_ptr, PointerType::getUnqual(cc.f32()));
    Value *elem = cc.b.CreateGEP(cc.f32(), pf, cc.row_idx, "elem_ptr");
    return cc.b.CreateLoad(cc.f32(), elem, "val_f32");
}
static Value *LoadF64(CompileCtx &cc, Value *data_ptr) {
    Value *pf   = cc.b.CreateBitCast(data_ptr, PointerType::getUnqual(cc.f64()));
    Value *elem = cc.b.CreateGEP(cc.f64(), pf, cc.row_idx, "elem_ptr");
    return cc.b.CreateLoad(cc.f64(), elem, "val_f64");
}

// Emit comparison for VarConstComparison node.
// Returns i1 (true = row matches).
static Value *EmitVarConst(CompileCtx &cc, const SimplestVarConstComparison *cmp) {
    int col_idx = cc.FindColIdx(*cmp->attr);
    if (col_idx < 0) return ConstantInt::getTrue(cc.llctx); // fallback: accept

    const ColSchema &cs = cc.schema[col_idx];
    Value *data = cc.col_data[col_idx];

    const SimplestConstVar *cv = cmp->const_var.get();
    SimplestExprType op = cmp->GetSimplestExprType();

    // ---- Integer types ----
    if (cs.dtype == AQP_DTYPE_INT32 || cs.dtype == AQP_DTYPE_DATE ||
        cs.dtype == AQP_DTYPE_BOOL || cs.dtype == AQP_DTYPE_INT8 ||
        cs.dtype == AQP_DTYPE_INT16) {

        Value *lhs;
        if (cs.dtype == AQP_DTYPE_INT8 || cs.dtype == AQP_DTYPE_BOOL) {
            Value *p8 = cc.b.CreateBitCast(data, PointerType::getUnqual(Type::getInt8Ty(cc.llctx)));
            Value *ep = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), p8, cc.row_idx);
            lhs = cc.b.CreateLoad(Type::getInt8Ty(cc.llctx), ep);
            lhs = cc.b.CreateSExt(lhs, cc.i32());
        } else if (cs.dtype == AQP_DTYPE_INT16) {
            Value *p16 = cc.b.CreateBitCast(data, PointerType::getUnqual(Type::getInt16Ty(cc.llctx)));
            Value *ep  = cc.b.CreateGEP(Type::getInt16Ty(cc.llctx), p16, cc.row_idx);
            lhs = cc.b.CreateLoad(Type::getInt16Ty(cc.llctx), ep);
            lhs = cc.b.CreateSExt(lhs, cc.i32());
        } else {
            lhs = LoadI32(cc, data);
        }

        int64_t rhs_raw = 0;
        if (cv->GetType() == SimplestVarType::IntVar || cv->GetType() == SimplestVarType::Date)
            rhs_raw = (int64_t)cv->GetIntValue();
        else if (cv->GetType() == SimplestVarType::FloatVar)
            rhs_raw = (int64_t)cv->GetFloatValue();
        Value *rhs = cc.c32((int32_t)rhs_raw);

        switch (op) {
        case SimplestExprType::Equal:        return cc.b.CreateICmpEQ(lhs, rhs);
        case SimplestExprType::NotEqual:     return cc.b.CreateICmpNE(lhs, rhs);
        case SimplestExprType::LessThan:     return cc.b.CreateICmpSLT(lhs, rhs);
        case SimplestExprType::GreaterThan:  return cc.b.CreateICmpSGT(lhs, rhs);
        case SimplestExprType::LessEqual:    return cc.b.CreateICmpSLE(lhs, rhs);
        case SimplestExprType::GreaterEqual: return cc.b.CreateICmpSGE(lhs, rhs);
        default: return ConstantInt::getTrue(cc.llctx);
        }
    }

    if (cs.dtype == AQP_DTYPE_INT64) {
        Value *lhs = LoadI64(cc, data);
        int64_t rhs_raw = 0;
        if (cv->GetType() == SimplestVarType::IntVar || cv->GetType() == SimplestVarType::Date)
            rhs_raw = (int64_t)cv->GetIntValue();
        else if (cv->GetType() == SimplestVarType::FloatVar)
            rhs_raw = (int64_t)cv->GetFloatValue();
        Value *rhs = cc.c64(rhs_raw);
        switch (op) {
        case SimplestExprType::Equal:        return cc.b.CreateICmpEQ(lhs, rhs);
        case SimplestExprType::NotEqual:     return cc.b.CreateICmpNE(lhs, rhs);
        case SimplestExprType::LessThan:     return cc.b.CreateICmpSLT(lhs, rhs);
        case SimplestExprType::GreaterThan:  return cc.b.CreateICmpSGT(lhs, rhs);
        case SimplestExprType::LessEqual:    return cc.b.CreateICmpSLE(lhs, rhs);
        case SimplestExprType::GreaterEqual: return cc.b.CreateICmpSGE(lhs, rhs);
        default: return ConstantInt::getTrue(cc.llctx);
        }
    }

    if (cs.dtype == AQP_DTYPE_FLOAT) {
        Value *lhs = LoadF32(cc, data);
        float rhs_raw = (cv->GetType() == SimplestVarType::FloatVar) ? (float)cv->GetFloatValue() : 0.0f;
        Value *rhs = ConstantFP::get(cc.f32(), rhs_raw);
        switch (op) {
        case SimplestExprType::Equal:        return cc.b.CreateFCmpOEQ(lhs, rhs);
        case SimplestExprType::NotEqual:     return cc.b.CreateFCmpONE(lhs, rhs);
        case SimplestExprType::LessThan:     return cc.b.CreateFCmpOLT(lhs, rhs);
        case SimplestExprType::GreaterThan:  return cc.b.CreateFCmpOGT(lhs, rhs);
        case SimplestExprType::LessEqual:    return cc.b.CreateFCmpOLE(lhs, rhs);
        case SimplestExprType::GreaterEqual: return cc.b.CreateFCmpOGE(lhs, rhs);
        default: return ConstantInt::getTrue(cc.llctx);
        }
    }

    if (cs.dtype == AQP_DTYPE_DOUBLE) {
        Value *lhs = LoadF64(cc, data);
        double rhs_raw = (cv->GetType() == SimplestVarType::FloatVar) ? (double)cv->GetFloatValue() : 0.0;
        Value *rhs = ConstantFP::get(cc.f64(), rhs_raw);
        switch (op) {
        case SimplestExprType::Equal:        return cc.b.CreateFCmpOEQ(lhs, rhs);
        case SimplestExprType::NotEqual:     return cc.b.CreateFCmpONE(lhs, rhs);
        case SimplestExprType::LessThan:     return cc.b.CreateFCmpOLT(lhs, rhs);
        case SimplestExprType::GreaterThan:  return cc.b.CreateFCmpOGT(lhs, rhs);
        case SimplestExprType::LessEqual:    return cc.b.CreateFCmpOLE(lhs, rhs);
        case SimplestExprType::GreaterEqual: return cc.b.CreateFCmpOGE(lhs, rhs);
        default: return ConstantInt::getTrue(cc.llctx);
        }
    }

    if (cs.dtype == AQP_DTYPE_VARCHAR) {
        // DuckDB string_t layout: 4-byte length, then either 12 inline bytes
        // (length ≤ 12) or 4-byte prefix + 8-byte heap pointer.
        // Emit: extract char* and len, then call aqp_like_match / strcmp helper.
        Value *str_base = cc.b.CreateBitCast(data, cc.i8p());
        // Each string_t is 16 bytes
        Value *str_offset = cc.b.CreateMul(cc.row_idx, cc.c64(16));
        Value *str_ptr    = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_base,
                                           str_offset, "str_ptr");
        // Load length (first 4 bytes)
        Value *len_ptr = cc.b.CreateBitCast(str_ptr,
            PointerType::getUnqual(cc.i32()));
        Value *slen = cc.b.CreateLoad(cc.i32(), len_ptr, "slen");
        // Determine if inline: len <= 12
        Value *is_inline = cc.b.CreateICmpSLE(slen, cc.c32(12), "is_inline");
        // Inline data starts at str_ptr + 4
        Value *inline_ptr = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_ptr,
                                            cc.c64(4), "inline_ptr");
        // Heap pointer at str_ptr + 8 (skip 4-byte length + 4-byte prefix)
        Value *heap_pp  = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_ptr,
                                          cc.c64(8), "heap_pp_raw");
        Value *heap_ppc = cc.b.CreateBitCast(heap_pp,
            PointerType::getUnqual(cc.i8p()));
        Value *heap_ptr = cc.b.CreateLoad(cc.i8p(), heap_ppc, "heap_ptr");
        Value *char_ptr = cc.b.CreateSelect(is_inline, inline_ptr, heap_ptr,
                                             "char_ptr");

        if (cv->GetType() == SimplestVarType::StringVar) {
            const std::string &pat = cv->GetStringValue();
            // Store pattern as a global constant
            Constant *pat_const = ConstantDataArray::getString(
                cc.llctx, pat, /*AddNull=*/false);
            GlobalVariable *pat_gv = new GlobalVariable(
                cc.mod, pat_const->getType(), /*isConst=*/true,
                GlobalValue::PrivateLinkage, pat_const, "pat");
            Value *pat_ptr = cc.b.CreateBitCast(pat_gv, cc.i8p());
            Value *pat_len = cc.c32((int32_t)pat.size());

            FunctionType *ft4 = FunctionType::get(cc.i32(),
                {cc.i8p(), cc.i32(), cc.i8p(), cc.i32()}, false);
            if (op == SimplestExprType::Equal || op == SimplestExprType::NotEqual) {
                // Exact byte equality: length check + memcmp (NOT LIKE semantics).
                // aqp_like_match would misinterpret '%'/'_' in literal values.
                FunctionCallee callee = cc.mod.getOrInsertFunction("aqp_str_eq", ft4);
                Value *result = cc.b.CreateCall(callee, {char_ptr, slen, pat_ptr, pat_len});
                Value *match  = cc.b.CreateICmpNE(result, cc.c32(0));
                return (op == SimplestExprType::NotEqual)
                    ? cc.b.CreateNot(match)
                    : match;
            } else {
                // TextLike / Text_Not_Like: use LIKE pattern matching
                FunctionCallee callee = cc.mod.getOrInsertFunction("aqp_like_match", ft4);
                Value *result = cc.b.CreateCall(callee, {char_ptr, slen, pat_ptr, pat_len});
                Value *match  = cc.b.CreateICmpNE(result, cc.c32(0));
                return (op == SimplestExprType::Text_Not_Like)
                    ? cc.b.CreateNot(match)
                    : match;
            }
        }
    }

    return ConstantInt::getTrue(cc.llctx); // unknown type: pass all
}

// Emit IS NULL / IS NOT NULL check
static Value *EmitIsNull(CompileCtx &cc, const SimplestIsNullExpr *expr) {
    int col_idx = cc.FindColIdx(*expr->attr);
    if (col_idx < 0) return ConstantInt::getFalse(cc.llctx);

    Value *validity_ptr = cc.col_validity[col_idx];
    // If validity_ptr is null (all-valid), IS NULL is always false
    Value *null_ptr = ConstantPointerNull::get(
        cast<PointerType>(validity_ptr->getType()));
    Value *has_nulls = cc.b.CreateICmpNE(validity_ptr, null_ptr, "has_nulls");

    // Compute the validity bit
    Function *fn  = cc.b.GetInsertBlock()->getParent();
    BasicBlock *pre_bb    = cc.b.GetInsertBlock(); // block before the branch
    BasicBlock *check_bb  = BasicBlock::Create(cc.llctx, "check_valid", fn);
    BasicBlock *merge_bb  = BasicBlock::Create(cc.llctx, "merge_valid", fn);

    cc.b.CreateCondBr(has_nulls, check_bb, merge_bb);

    cc.b.SetInsertPoint(check_bb);
    Value *is_valid_inner = EmitValidityCheck(cc, validity_ptr);
    BasicBlock *check_end = cc.b.GetInsertBlock();
    cc.b.CreateBr(merge_bb);

    cc.b.SetInsertPoint(merge_bb);
    PHINode *is_valid = cc.b.CreatePHI(cc.i1(), 2, "is_valid");
    // When has_nulls==false (all-valid), jump from pre_bb → merge_bb: row is valid
    is_valid->addIncoming(ConstantInt::getTrue(cc.llctx), pre_bb);
    is_valid->addIncoming(is_valid_inner, check_end);

    // IS NULL = !is_valid; IS NOT NULL = is_valid
    bool is_null_check = (expr->GetSimplestExprType() == SimplestExprType::NullType);
    return is_null_check ? cc.b.CreateNot(is_valid) : is_valid;
}

// Emit AND/OR/NOT
static Value *EmitLogical(CompileCtx &cc, const SimplestLogicalExpr *expr) {
    using Op = SimplestLogicalOp;
    Op op = expr->GetLogicalOp();

    if (op == Op::LogicalNot) {
        // LogicalNot: left_expr is nullptr; operand is in right_expr
        Value *child = EmitExpr(cc, expr->right_expr.get());
        return cc.b.CreateNot(child, "not");
    }

    // Short-circuit AND / OR using basic blocks
    Function *fn = cc.b.GetInsertBlock()->getParent();
    BasicBlock *lhs_bb   = cc.b.GetInsertBlock();
    BasicBlock *rhs_bb   = BasicBlock::Create(cc.llctx, "logical_rhs", fn);
    BasicBlock *merge_bb = BasicBlock::Create(cc.llctx, "logical_merge", fn);

    Value *lhs = EmitExpr(cc, expr->left_expr.get());
    BasicBlock *lhs_end = cc.b.GetInsertBlock();

    if (op == Op::LogicalAnd) {
        // If lhs false → skip rhs (short-circuit false)
        cc.b.CreateCondBr(lhs, rhs_bb, merge_bb);
    } else {
        // LogicalOr: if lhs true → skip rhs (short-circuit true)
        cc.b.CreateCondBr(lhs, merge_bb, rhs_bb);
    }

    cc.b.SetInsertPoint(rhs_bb);
    Value *rhs = EmitExpr(cc, expr->right_expr.get());
    BasicBlock *rhs_end = cc.b.GetInsertBlock();
    cc.b.CreateBr(merge_bb);

    cc.b.SetInsertPoint(merge_bb);
    PHINode *phi = cc.b.CreatePHI(cc.i1(), 2, "logical_result");
    if (op == Op::LogicalAnd) {
        phi->addIncoming(ConstantInt::getFalse(cc.llctx), lhs_end);
        phi->addIncoming(rhs, rhs_end);
    } else {
        phi->addIncoming(ConstantInt::getTrue(cc.llctx), lhs_end);
        phi->addIncoming(rhs, rhs_end);
    }
    return phi;
}

// Emit IN expression: col IN (v1, v2, ...)
static Value *EmitIn(CompileCtx &cc, const SimplestInExpr *expr) {
    int col_idx = cc.FindColIdx(*expr->attr);
    if (col_idx < 0)
        return expr->negated ? ConstantInt::getFalse(cc.llctx)
                             : ConstantInt::getTrue(cc.llctx);

    const ColSchema &cs = cc.schema[col_idx];
    Value *data = cc.col_data[col_idx];
    const auto &vals = expr->values;

    if (cs.dtype == AQP_DTYPE_INT32 || cs.dtype == AQP_DTYPE_INT16 ||
        cs.dtype == AQP_DTYPE_INT8  || cs.dtype == AQP_DTYPE_DATE) {

        Value *lhs;
        if (cs.dtype == AQP_DTYPE_INT8) {
            Value *p8 = cc.b.CreateBitCast(data, PointerType::getUnqual(Type::getInt8Ty(cc.llctx)));
            lhs = cc.b.CreateLoad(Type::getInt8Ty(cc.llctx),
                cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), p8, cc.row_idx));
            lhs = cc.b.CreateSExt(lhs, cc.i32());
        } else {
            lhs = LoadI32(cc, data);
        }

        if (vals.size() <= 8) {
            // Unroll: lhs == v1 || lhs == v2 || ...
            Value *any = ConstantInt::getFalse(cc.llctx);
            for (const auto &v : vals) {
                int32_t iv = (int32_t)v->GetIntValue();
                Value *match = cc.b.CreateICmpEQ(lhs, cc.c32(iv));
                any = cc.b.CreateOr(any, match);
            }
            return expr->negated ? cc.b.CreateNot(any) : any;
        } else {
            // Call aqp_in_set_i32 with a global constant array
            std::vector<Constant*> consts;
            for (const auto &v : vals) {
                int32_t iv = (int32_t)v->GetIntValue();
                consts.push_back(cc.c32(iv));
            }
            ArrayType *arr_ty = ArrayType::get(cc.i32(), consts.size());
            Constant *arr_init = ConstantArray::get(arr_ty, consts);
            GlobalVariable *gv = new GlobalVariable(
                cc.mod, arr_ty, true, GlobalValue::PrivateLinkage, arr_init, "in_vals");
            Value *arr_ptr = cc.b.CreateBitCast(gv,
                PointerType::getUnqual(cc.i32()));
            FunctionType *ft = FunctionType::get(cc.i32(),
                {cc.i32(), PointerType::getUnqual(cc.i32()), cc.i32()}, false);
            FunctionCallee callee = cc.mod.getOrInsertFunction("aqp_in_set_i32", ft);
            Value *result = cc.b.CreateCall(callee,
                {lhs, arr_ptr, cc.c32((int32_t)consts.size())});
            Value *match = cc.b.CreateICmpNE(result, cc.c32(0));
            return expr->negated ? cc.b.CreateNot(match) : match;
        }
    }

    if (cs.dtype == AQP_DTYPE_INT64) {
        Value *lhs = LoadI64(cc, data);
        if (vals.size() <= 8) {
            Value *any = ConstantInt::getFalse(cc.llctx);
            for (const auto &v : vals) {
                int64_t iv = (int64_t)v->GetIntValue();
                Value *match = cc.b.CreateICmpEQ(lhs, cc.c64(iv));
                any = cc.b.CreateOr(any, match);
            }
            return expr->negated ? cc.b.CreateNot(any) : any;
        }
        // Fallback: accept all (TODO: i64 runtime helper)
        return expr->negated ? ConstantInt::getFalse(cc.llctx)
                             : ConstantInt::getTrue(cc.llctx);
    }

    // VARCHAR IN — emit aqp_in_set_str call (simplified: only for small sets)
    // TODO: implement runtime set for large VARCHAR IN lists
    return expr->negated ? ConstantInt::getFalse(cc.llctx)
                         : ConstantInt::getTrue(cc.llctx);
}

// Main expression dispatch
static Value *EmitExpr(CompileCtx &cc, const AQPExpr *expr) {
    if (!expr) return ConstantInt::getTrue(cc.llctx);

    switch (expr->GetNodeType()) {
    case VarConstComparisonNode:
        return EmitVarConst(cc, static_cast<const SimplestVarConstComparison*>(expr));
    case IsNullExprNode:
        return EmitIsNull(cc, static_cast<const SimplestIsNullExpr*>(expr));
    case LogicalExprNode:
        return EmitLogical(cc, static_cast<const SimplestLogicalExpr*>(expr));
    case InExprNode:
        return EmitIn(cc, static_cast<const SimplestInExpr*>(expr));
    default:
        return ConstantInt::getTrue(cc.llctx); // unknown: pass all rows
    }
}

// ---------------------------------------------------------------------------
// Build the outer loop function:
//   uint64_t aqp_expr_<hash>(AQPChunkView* chunk, AQPSelView* sel)
// ---------------------------------------------------------------------------
static Function *BuildFilterFunction(LLVMContext &llctx, Module &mod,
                                     const std::string &fn_name,
                                     const std::vector<const AQPExpr*> &exprs,
                                     const std::vector<ColSchema> &schema) {
    Type *i8p  = PointerType::getUnqual(Type::getInt8Ty(llctx));
    Type *i32  = Type::getInt32Ty(llctx);
    Type *i64  = Type::getInt64Ty(llctx);
    Type *i64p = PointerType::getUnqual(i64);

    // Struct types for the ABI
    StructType *ColViewTy   = StructType::get(llctx, {i8p, i64p, i32, i32});
    StructType *ChunkViewTy = StructType::get(llctx, {
        PointerType::getUnqual(ColViewTy), i64, i64});
    // AQPSelView.indices is sel_t* = uint32_t* (DuckDB typedefs.hpp: typedef uint32_t sel_t)
    StructType *SelViewTy   = StructType::get(llctx, {
        PointerType::getUnqual(i32), i32});

    FunctionType *fn_ty = FunctionType::get(
        i64, {PointerType::getUnqual(ChunkViewTy),
              PointerType::getUnqual(SelViewTy)}, false);
    Function *fn = Function::Create(fn_ty, Function::ExternalLinkage,
                                    fn_name, &mod);

    Value *chunk_arg = fn->getArg(0);
    Value *sel_arg   = fn->getArg(1);
    chunk_arg->setName("chunk");
    sel_arg->setName("sel");

    BasicBlock *entry_bb  = BasicBlock::Create(llctx, "entry", fn);
    BasicBlock *loop_bb   = BasicBlock::Create(llctx, "loop", fn);
    BasicBlock *body_bb   = BasicBlock::Create(llctx, "body", fn);
    BasicBlock *store_bb  = BasicBlock::Create(llctx, "store", fn);
    BasicBlock *next_bb   = BasicBlock::Create(llctx, "next", fn);
    BasicBlock *exit_bb   = BasicBlock::Create(llctx, "exit", fn);

    CompileCtx cc(llctx, mod, schema, chunk_arg, sel_arg);
    cc.b.SetInsertPoint(entry_bb);

    // Load nrows from chunk->nrows (field index 1)
    Value *nrows_ptr = cc.b.CreateStructGEP(ChunkViewTy, chunk_arg, 1);
    Value *nrows     = cc.b.CreateLoad(i64, nrows_ptr, "nrows");

    // Load col data + validity pointers (once, before the loop)
    cc.col_data.resize(schema.size());
    cc.col_validity.resize(schema.size());
    for (size_t i = 0; i < schema.size(); i++) {
        cc.col_data[i]     = cc.LoadColData((unsigned)i);
        cc.col_validity[i] = cc.LoadColValidity((unsigned)i);
    }

    // Load sel->indices pointer (sel_t* = uint32_t*)
    Value *sel_idx_ptr_ptr = cc.b.CreateStructGEP(SelViewTy, sel_arg, 0);
    Value *sel_idx_ptr     = cc.b.CreateLoad(
        PointerType::getUnqual(i32), sel_idx_ptr_ptr, "sel_indices");

    cc.b.CreateBr(loop_bb);

    // Loop header — i = 0, out_count = 0
    cc.b.SetInsertPoint(loop_bb);
    PHINode *i         = cc.b.CreatePHI(i64, 2, "i");
    PHINode *out_count = cc.b.CreatePHI(i64, 2, "out_count");
    i->addIncoming(ConstantInt::get(llctx, APInt(64, 0)), entry_bb);
    out_count->addIncoming(ConstantInt::get(llctx, APInt(64, 0)), entry_bb);

    // Check loop condition
    Value *done = cc.b.CreateICmpEQ(i, nrows, "done");
    cc.b.CreateCondBr(done, exit_bb, body_bb);

    // Loop body — evaluate all expressions (AND them together)
    cc.b.SetInsertPoint(body_bb);
    cc.row_idx = i;

    Value *match = ConstantInt::getTrue(llctx);
    for (const AQPExpr *e : exprs) {
        Value *res = EmitExpr(cc, e);
        match = cc.b.CreateAnd(match, res);
    }
    // After EmitExpr, the insert point may be a different block (e.g. merge_bb
    // created by EmitLogical for OR/AND).  Capture it now — this is the block
    // that actually branches to store_bb / next_bb.
    BasicBlock *condBr_bb = cc.b.GetInsertBlock();
    cc.b.CreateCondBr(match, store_bb, next_bb);

    // Store matching row index as sel_t = uint32_t (4-byte stride)
    cc.b.SetInsertPoint(store_bb);
    Value *dst   = cc.b.CreateGEP(i32, sel_idx_ptr, out_count, "dst");
    Value *i_i32 = cc.b.CreateTrunc(i, i32, "i_i32");
    cc.b.CreateStore(i_i32, dst);
    Value *out_count1 = cc.b.CreateAdd(out_count, ConstantInt::get(llctx, APInt(64, 1)));
    cc.b.CreateBr(next_bb);

    // Increment i
    cc.b.SetInsertPoint(next_bb);
    PHINode *out_count_next = cc.b.CreatePHI(i64, 2, "out_count_next");
    // condBr_bb is the actual predecessor of next_bb on the "no match" path.
    // (It may differ from body_bb when EmitLogical created intermediate blocks.)
    out_count_next->addIncoming(out_count, condBr_bb);
    out_count_next->addIncoming(out_count1, store_bb);
    Value *i1 = cc.b.CreateAdd(i, ConstantInt::get(llctx, APInt(64, 1)), "i_next");
    i->addIncoming(i1, next_bb);
    out_count->addIncoming(out_count_next, next_bb);
    cc.b.CreateBr(loop_bb);

    // Exit — store count to sel->count, return out_count
    cc.b.SetInsertPoint(exit_bb);
    Value *sel_cnt_ptr = cc.b.CreateStructGEP(SelViewTy, sel_arg, 1);
    Value *final_count = out_count; // comes from loop PHI via exit edge
    // Patch PHI incoming for exit (from loop_bb when done==true)
    // The exit_bb is entered from loop_bb when done is true; at that point
    // out_count still holds its current value.
    cc.b.CreateStore(cc.b.CreateTrunc(out_count, i32), sel_cnt_ptr);
    cc.b.CreateRet(out_count);
    (void)final_count;

    return fn;
}

// ---------------------------------------------------------------------------
// Optimise the module with O0 (mem2reg only) or O3
// ---------------------------------------------------------------------------
static void OptimiseModule(Module &mod, bool use_o3) {
    PassBuilder pb;
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm = use_o3
        ? pb.buildPerModuleDefaultPipeline(OptimizationLevel::O3)
        : pb.buildPerModuleDefaultPipeline(OptimizationLevel::O1);
    mpm.run(mod, mam);
}

// ---------------------------------------------------------------------------
// IrToLlvmCompiler public API
// ---------------------------------------------------------------------------
IrToLlvmCompiler::IrToLlvmCompiler(bool use_o3)
    : use_o3_(use_o3), impl_(std::make_unique<Impl>()) {}

IrToLlvmCompiler::~IrToLlvmCompiler() = default;

AQPExprFn IrToLlvmCompiler::CompileExpr(
        const AQPExpr &expr,
        const std::vector<ColSchema> &schema) {

    auto ctx = std::make_unique<LLVMContext>();
    auto mod = std::make_unique<Module>("aqp_expr_mod", *ctx);

    // Serialise expression to a string for hashing
    std::ostringstream oss;
    const_cast<AQPExpr&>(expr).Print(false); // Print to string (side-effect: oss via cout redirect)
    uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
    std::string fn_name = "aqp_expr_" + std::to_string(fn_id);

    std::vector<const AQPExpr*> exprs = {&expr};
    Function *fn = BuildFilterFunction(*ctx, *mod, fn_name, exprs, schema);
    if (!fn) return nullptr;

    // Verify
    std::string err;
    raw_string_ostream es(err);
    if (verifyFunction(*fn, &es)) {
        // Verification failed — fall back to interpreter
        return nullptr;
    }

    OptimiseModule(*mod, use_o3_);

    // Add module to ORC JIT
    auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
    if (auto err2 = impl_->jit->addIRModule(std::move(tsm))) {
        logAllUnhandledErrors(std::move(err2), errs());
        return nullptr;
    }

    // Look up the compiled function
    auto sym = impl_->jit->lookup(fn_name);
    if (!sym) {
        logAllUnhandledErrors(sym.takeError(), errs());
        return nullptr;
    }

    return jitTargetAddressToFunction<AQPExprFn>(sym->getAddress());
}

AQPExprFn IrToLlvmCompiler::CompileFilter(
        const AQPStmt &filter_node,
        const std::vector<ColSchema> &schema) {

    // Collect the qual_vec expressions from the filter node
    std::vector<const AQPExpr*> exprs;
    for (const auto &qe : filter_node.qual_vec) {
        exprs.push_back(qe.get());
    }
    if (exprs.empty()) return nullptr;

    // Use a monotonic counter to guarantee a unique function name within
    // this LLJIT instance (hash collisions caused "Duplicate definition").
    uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
    std::string fn_name = "aqp_expr_" + std::to_string(fn_id);

    auto ctx = std::make_unique<LLVMContext>();
    auto mod = std::make_unique<Module>("aqp_filter_mod", *ctx);

    Function *fn = BuildFilterFunction(*ctx, *mod, fn_name, exprs, schema);
    if (!fn) return nullptr;

    std::string err;
    raw_string_ostream es(err);
    if (verifyFunction(*fn, &es)) {
        std::cerr << "[AQP-JIT] verifyFunction failed: " << es.str() << "\n";
        return nullptr;
    }

    OptimiseModule(*mod, use_o3_);

    auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
    if (auto e = impl_->jit->addIRModule(std::move(tsm))) {
        std::cerr << "[AQP-JIT] addIRModule failed\n";
        logAllUnhandledErrors(std::move(e), errs());
        return nullptr;
    }

    auto sym = impl_->jit->lookup(fn_name);
    if (!sym) {
        std::cerr << "[AQP-JIT] lookup failed for " << fn_name << "\n";
        logAllUnhandledErrors(sym.takeError(), errs());
        return nullptr;
    }

    return jitTargetAddressToFunction<AQPExprFn>(sym->getAddress());
}

} // namespace aqp_jit
