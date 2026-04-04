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

// Forward-declare C-linkage runtime helpers (defined in aqp_jit_runtime.cpp
// and aqp_jit_hashtable.cpp).  Must come before Impl::Impl() which takes
// their addresses.
extern "C" {
    int aqp_like_match(const char *str, int32_t slen, const char *pat, int32_t plen);
    int aqp_ilike_match(const char *str, int32_t slen, const char *pat, int32_t plen);
    int aqp_str_eq(const char *a, int32_t alen, const char *b, int32_t blen);
    int aqp_str_cmp(const char *a, int32_t alen, const char *b, int32_t blen);
    int aqp_in_set_i32(int32_t val, const int32_t *values, int32_t n);
    int aqp_in_set_i64(int64_t val, const int64_t *values, int32_t n);
    int aqp_in_set_str(const char *str, int32_t slen, const char **ptrs,
                       const int32_t *lens, int32_t n);
    // Hash table (aqp_jit_hashtable.cpp)
    struct AQPHashTable;
    AQPHashTable *aqp_ht_create(uint32_t key_width, uint32_t payload_width, uint64_t est_rows);
    void  aqp_ht_destroy(AQPHashTable *ht);
    void *aqp_ht_insert(AQPHashTable *ht, const void *key);
    void *aqp_ht_probe(const AQPHashTable *ht, const void *key);
    void  aqp_ht_iter_reset(AQPHashTable *ht);
    int   aqp_ht_next(AQPHashTable *ht, void **key_out, void **payload_out);
    uint64_t aqp_ht_size(const AQPHashTable *ht);
    uint64_t aqp_hash(const void *key, uint32_t len);
}

#include <cstdint>
#include <functional>
#include <atomic>
#include <map>
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
            {es.intern("aqp_str_cmp"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_str_cmp),
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
            // Hash table runtime
            {es.intern("aqp_ht_create"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_ht_create),
                                JITSymbolFlags::Exported)},
            {es.intern("aqp_ht_destroy"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_ht_destroy),
                                JITSymbolFlags::Exported)},
            {es.intern("aqp_ht_insert"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_ht_insert),
                                JITSymbolFlags::Exported)},
            {es.intern("aqp_ht_probe"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_ht_probe),
                                JITSymbolFlags::Exported)},
            {es.intern("aqp_ht_iter_reset"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_ht_iter_reset),
                                JITSymbolFlags::Exported)},
            {es.intern("aqp_ht_next"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_ht_next),
                                JITSymbolFlags::Exported)},
            {es.intern("aqp_ht_size"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_ht_size),
                                JITSymbolFlags::Exported)},
            {es.intern("aqp_hash"),
             JITEvaluatedSymbol(pointerToJITTargetAddress(::aqp_hash),
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
        if (cv->GetType() != SimplestVarType::StringVar)
            return ConstantInt::getTrue(cc.llctx);

        // ---- NULL guard ------------------------------------------------
        // DuckDB does NOT zero-initialise string_t at NULL positions.
        // Accessing such data yields a garbage length / heap-pointer → crash.
        // SQL semantics: NULL compared to anything yields NULL → excluded from
        // WHERE → return false for this row.
        Value *validity_ptr = cc.col_validity[col_idx];
        Function *fn = cc.b.GetInsertBlock()->getParent();
        BasicBlock *chk_bb   = BasicBlock::Create(cc.llctx, "str_chk_null", fn);
        BasicBlock *cmp_bb   = BasicBlock::Create(cc.llctx, "str_cmp",      fn);
        BasicBlock *after_bb = BasicBlock::Create(cc.llctx, "str_after",    fn);

        Value *null_vp    = ConstantPointerNull::get(
            cast<PointerType>(validity_ptr->getType()));
        Value *has_valvec = cc.b.CreateICmpNE(validity_ptr, null_vp, "has_valvec");
        // has_valvec=true  → chk_bb (validity vector exists, check individual bit)
        // has_valvec=false → cmp_bb  (all rows valid, no null check needed)
        cc.b.CreateCondBr(has_valvec, chk_bb, cmp_bb);

        cc.b.SetInsertPoint(chk_bb);
        Value *is_valid = EmitValidityCheck(cc, validity_ptr);
        // is_valid=true  → cmp_bb  (row is not NULL)
        // is_valid=false → after_bb (row is NULL → result = false)
        cc.b.CreateCondBr(is_valid, cmp_bb, after_bb);
        BasicBlock *chk_end = cc.b.GetInsertBlock();

        // ---- string_t extraction + comparison --------------------------
        // DuckDB string_t layout (16 bytes):
        //   [0..3]  uint32_t length
        //   [4..7]  char prefix[4]       (non-inline: first 4 bytes of heap data)
        //   [8..15] char* ptr            (heap pointer, only valid when length > 12)
        //   OR (inline, length <= 12):
        //   [4..15] char inlined[12]
        cc.b.SetInsertPoint(cmp_bb);
        Value *str_base   = cc.b.CreateBitCast(data, cc.i8p());
        Value *str_offset = cc.b.CreateMul(cc.row_idx, cc.c64(16));
        Value *str_ptr    = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_base,
                                           str_offset, "str_ptr");
        Value *len_ptr    = cc.b.CreateBitCast(str_ptr,
                                PointerType::getUnqual(cc.i32()));
        Value *slen       = cc.b.CreateLoad(cc.i32(), len_ptr, "slen");
        Value *is_inline  = cc.b.CreateICmpSLE(slen, cc.c32(12), "is_inline");
        Value *inline_ptr = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_ptr,
                                           cc.c64(4), "inline_ptr");
        Value *heap_pp    = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_ptr,
                                           cc.c64(8), "heap_pp_raw");
        Value *heap_ppc   = cc.b.CreateBitCast(heap_pp,
                                PointerType::getUnqual(cc.i8p()));
        Value *heap_ptr   = cc.b.CreateLoad(cc.i8p(), heap_ppc, "heap_ptr");
        Value *char_ptr   = cc.b.CreateSelect(is_inline, inline_ptr, heap_ptr,
                                              "char_ptr");

        const std::string &pat = cv->GetStringValue();
        Constant *pat_const = ConstantDataArray::getString(
            cc.llctx, pat, /*AddNull=*/false);
        GlobalVariable *pat_gv = new GlobalVariable(
            cc.mod, pat_const->getType(), /*isConst=*/true,
            GlobalValue::PrivateLinkage, pat_const, "pat");
        Value *pat_ptr = cc.b.CreateBitCast(pat_gv, cc.i8p());
        Value *pat_len = cc.c32((int32_t)pat.size());

        FunctionType *ft4 = FunctionType::get(cc.i32(),
            {cc.i8p(), cc.i32(), cc.i8p(), cc.i32()}, false);
        Value *cmp_result;
        if (op == SimplestExprType::Equal || op == SimplestExprType::NotEqual) {
            FunctionCallee callee = cc.mod.getOrInsertFunction("aqp_str_eq", ft4);
            Value *r = cc.b.CreateCall(callee, {char_ptr, slen, pat_ptr, pat_len});
            Value *m = cc.b.CreateICmpNE(r, cc.c32(0));
            cmp_result = (op == SimplestExprType::NotEqual)
                         ? cc.b.CreateNot(m) : m;
        } else if (op == SimplestExprType::TextLike ||
                   op == SimplestExprType::Text_Not_Like) {
            FunctionCallee callee = cc.mod.getOrInsertFunction("aqp_like_match", ft4);
            Value *r = cc.b.CreateCall(callee, {char_ptr, slen, pat_ptr, pat_len});
            Value *m = cc.b.CreateICmpNE(r, cc.c32(0));
            cmp_result = (op == SimplestExprType::Text_Not_Like)
                         ? cc.b.CreateNot(m) : m;
        } else if (op == SimplestExprType::GreaterEqual ||
                   op == SimplestExprType::LessEqual    ||
                   op == SimplestExprType::GreaterThan  ||
                   op == SimplestExprType::LessThan) {
            // Lexicographic ordering via aqp_str_cmp (returns <0, 0, or >0)
            FunctionType *ft_cmp = FunctionType::get(cc.i32(),
                {cc.i8p(), cc.i32(), cc.i8p(), cc.i32()}, false);
            FunctionCallee callee = cc.mod.getOrInsertFunction("aqp_str_cmp", ft_cmp);
            Value *r    = cc.b.CreateCall(callee, {char_ptr, slen, pat_ptr, pat_len});
            Value *zero = cc.c32(0);
            switch (op) {
            case SimplestExprType::GreaterEqual:
                cmp_result = cc.b.CreateICmpSGE(r, zero); break;
            case SimplestExprType::LessEqual:
                cmp_result = cc.b.CreateICmpSLE(r, zero); break;
            case SimplestExprType::GreaterThan:
                cmp_result = cc.b.CreateICmpSGT(r, zero); break;
            case SimplestExprType::LessThan:
                cmp_result = cc.b.CreateICmpSLT(r, zero); break;
            default:
                cmp_result = ConstantInt::getTrue(cc.llctx); break;
            }
        } else {
            // Unknown VARCHAR operator: pass all rows
            cmp_result = ConstantInt::getTrue(cc.llctx);
        }
        cc.b.CreateBr(after_bb);
        BasicBlock *cmp_end = cc.b.GetInsertBlock();

        // ---- merge: PHI for null-guard result --------------------------
        cc.b.SetInsertPoint(after_bb);
        PHINode *phi = cc.b.CreatePHI(cc.i1(), 2, "str_result");
        // NULL row: excluded from WHERE (false)
        phi->addIncoming(ConstantInt::getFalse(cc.llctx), chk_end);
        // non-NULL row: actual comparison result
        phi->addIncoming(cmp_result, cmp_end);
        return phi;
    }

    return ConstantInt::getTrue(cc.llctx); // unknown type: pass all
}

// Emit IS NULL / IS NOT NULL check
static Value *EmitIsNull(CompileCtx &cc, const SimplestIsNullExpr *expr) {
    int col_idx = cc.FindColIdx(*expr->attr);
    // Column not in schema: predicate doesn't apply to this filter → pass all rows.
    // (same semantics as EmitVarConst when col_idx < 0)
    if (col_idx < 0) return ConstantInt::getTrue(cc.llctx);

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

// Returns true if `expr` references at least one column present in the schema.
// Used to detect pass-through expressions so that NOT(pass-through) stays pass-through.
static bool ExprInvolvesSchema(const CompileCtx &cc, const AQPExpr *expr) {
    if (!expr) return false;
    switch (expr->GetNodeType()) {
    case VarConstComparisonNode: {
        auto *cmp = static_cast<const SimplestVarConstComparison *>(expr);
        return cc.FindColIdx(*cmp->attr) >= 0;
    }
    case IsNullExprNode: {
        auto *isnull = static_cast<const SimplestIsNullExpr *>(expr);
        return cc.FindColIdx(*isnull->attr) >= 0;
    }
    case LogicalExprNode: {
        auto *log = static_cast<const SimplestLogicalExpr *>(expr);
        return ExprInvolvesSchema(cc, log->left_expr.get()) ||
               ExprInvolvesSchema(cc, log->right_expr.get());
    }
    case InExprNode: {
        auto *in_expr = static_cast<const SimplestInExpr *>(expr);
        return cc.FindColIdx(*in_expr->attr) >= 0;
    }
    default:
        return false;
    }
}

// Emit AND/OR/NOT
static Value *EmitLogical(CompileCtx &cc, const SimplestLogicalExpr *expr) {
    using Op = SimplestLogicalOp;
    Op op = expr->GetLogicalOp();

    if (op == Op::LogicalNot) {
        // If the operand references no column from this schema, the NOT predicate
        // is not applicable here → pass-through true (don't filter any rows).
        // Without this guard, NOT(pass-through-true) = false → all rows rejected.
        if (!ExprInvolvesSchema(cc, expr->right_expr.get()))
            return ConstantInt::getTrue(cc.llctx);
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

// Element size in bytes for each AQP dtype.
static unsigned DtypeElemSize(int32_t dtype) {
    switch (dtype) {
    case AQP_DTYPE_BOOL:  case AQP_DTYPE_INT8:    return 1;
    case AQP_DTYPE_INT16:                          return 2;
    case AQP_DTYPE_INT32: case AQP_DTYPE_FLOAT:
    case AQP_DTYPE_DATE:                           return 4;
    case AQP_DTYPE_INT64: case AQP_DTYPE_DOUBLE:   return 8;
    case AQP_DTYPE_VARCHAR:                        return 16; // DuckDB string_t
    default:                                       return 0;
    }
}

// ---------------------------------------------------------------------------
// Build a projection function:
//   int32_t aqp_proj_<id>(AQPChunkView* in, AQPChunkView* out)
//
// Copies actual column DATA (memcpy) from input to output for each mapped
// column.  This is portable — works with any engine that provides flat
// columnar buffers via AQPChunkView.
//
// col_mapping[i]  = input column index for output column i (-1 = skip)
// col_dtypes[i]   = AQP_DTYPE_* of output column i (determines element size)
// ---------------------------------------------------------------------------
static Function *BuildProjectionFunction(LLVMContext &llctx, Module &mod,
                                         const std::string &fn_name,
                                         const std::vector<int> &col_mapping,
                                         const std::vector<int32_t> &col_dtypes) {
    Type *i8p  = PointerType::getUnqual(Type::getInt8Ty(llctx));
    Type *i32  = Type::getInt32Ty(llctx);
    Type *i64  = Type::getInt64Ty(llctx);
    Type *i64p = PointerType::getUnqual(i64);
    Type *i1   = Type::getInt1Ty(llctx);

    StructType *ColViewTy   = StructType::get(llctx, {i8p, i64p, i32, i32});
    StructType *ChunkViewTy = StructType::get(llctx, {
        PointerType::getUnqual(ColViewTy), i64, i64});

    // int32_t fn(AQPChunkView *in, AQPChunkView *out)
    FunctionType *fn_ty = FunctionType::get(
        i32, {PointerType::getUnqual(ChunkViewTy),
              PointerType::getUnqual(ChunkViewTy)}, false);
    Function *fn = Function::Create(fn_ty, Function::ExternalLinkage,
                                    fn_name, &mod);

    Value *in_arg  = fn->getArg(0);  in_arg->setName("in");
    Value *out_arg = fn->getArg(1);  out_arg->setName("out");

    BasicBlock *entry = BasicBlock::Create(llctx, "entry", fn);
    IRBuilder<> b(entry);

    // Load in->nrows
    Value *in_nrows_ptr  = b.CreateStructGEP(ChunkViewTy, in_arg, 1);
    Value *in_nrows      = b.CreateLoad(i64, in_nrows_ptr, "in_nrows");

    // Load in->cols and out->cols base pointers
    Value *in_cols_pp  = b.CreateStructGEP(ChunkViewTy, in_arg, 0);
    Value *in_cols     = b.CreateLoad(PointerType::getUnqual(ColViewTy), in_cols_pp, "in_cols");
    Value *out_cols_pp = b.CreateStructGEP(ChunkViewTy, out_arg, 0);
    Value *out_cols    = b.CreateLoad(PointerType::getUnqual(ColViewTy), out_cols_pp, "out_cols");

    // Declare llvm.memcpy intrinsic
    Function *memcpy_fn = Intrinsic::getDeclaration(&mod, Intrinsic::memcpy,
        {i8p, i8p, i64});

    // Validity size in bytes: ceil(nrows / 64) * 8
    Value *nrows_plus_63 = b.CreateAdd(in_nrows, ConstantInt::get(i64, 63));
    Value *nwords = b.CreateLShr(nrows_plus_63, ConstantInt::get(i64, 6));
    Value *val_bytes = b.CreateMul(nwords, ConstantInt::get(i64, 8), "val_bytes");

    // For each output column, memcpy actual data from input column
    for (size_t out_i = 0; out_i < col_mapping.size(); out_i++) {
        int in_i = col_mapping[out_i];
        if (in_i < 0) continue;

        unsigned elem_size = DtypeElemSize(col_dtypes[out_i]);
        if (elem_size == 0) continue;  // unknown dtype, skip

        Value *src_col = b.CreateGEP(ColViewTy, in_cols,
            ConstantInt::get(i64, (uint64_t)in_i), "src_col");
        Value *dst_col = b.CreateGEP(ColViewTy, out_cols,
            ConstantInt::get(i64, (uint64_t)out_i), "dst_col");

        // Load source and dest data pointers
        Value *src_data = b.CreateLoad(i8p, b.CreateStructGEP(ColViewTy, src_col, 0), "src_data");
        Value *dst_data = b.CreateLoad(i8p, b.CreateStructGEP(ColViewTy, dst_col, 0), "dst_data");

        // memcpy(dst_data, src_data, nrows * elem_size)
        Value *data_bytes = b.CreateMul(in_nrows,
            ConstantInt::get(i64, (uint64_t)elem_size), "data_bytes");
        b.CreateCall(memcpy_fn, {dst_data, src_data, data_bytes,
                                 ConstantInt::getFalse(llctx)});

        // Copy validity: if src validity is not null, memcpy it
        Value *src_val = b.CreateLoad(i64p,
            b.CreateStructGEP(ColViewTy, src_col, 1), "src_val");
        Value *dst_val = b.CreateLoad(i64p,
            b.CreateStructGEP(ColViewTy, dst_col, 1), "dst_val");
        Value *src_val_nonnull = b.CreateICmpNE(
            b.CreatePtrToInt(src_val, i64), ConstantInt::get(i64, 0), "val_nonnull");

        // Conditional memcpy for validity
        BasicBlock *copy_val_bb = BasicBlock::Create(llctx,
            "copy_val_" + std::to_string(out_i), fn);
        BasicBlock *next_col_bb = BasicBlock::Create(llctx,
            "next_col_" + std::to_string(out_i), fn);
        b.CreateCondBr(src_val_nonnull, copy_val_bb, next_col_bb);

        b.SetInsertPoint(copy_val_bb);
        Value *src_val_i8 = b.CreateBitCast(src_val, i8p);
        Value *dst_val_i8 = b.CreateBitCast(dst_val, i8p);
        b.CreateCall(memcpy_fn, {dst_val_i8, src_val_i8, val_bytes,
                                 ConstantInt::getFalse(llctx)});
        b.CreateBr(next_col_bb);

        b.SetInsertPoint(next_col_bb);
    }

    // Return 0 (NEED_MORE_INPUT)
    b.CreateRet(ConstantInt::get(i32, 0));

    return fn;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Build a fused pipeline function (Filter → Projection):
//   int64_t aqp_pipe_<id>(AQPChunkView* in, AQPChunkView* out, i8* state)
//
// Single row loop: for each input row, evaluate filter predicates (AND).
// If match: for each output column, copy element from input to output.
// Returns count of output rows. No intermediate DataChunk materialization.
//
// filter_exprs:   list of filter expressions (AND'd), may be empty
// col_mapping:    out_col_i → in_col_i (projection mapping)
// col_dtypes:     dtype per output column (for element size)
// ---------------------------------------------------------------------------
static Function *BuildPipelineFunction(LLVMContext &llctx, Module &mod,
                                       const std::string &fn_name,
                                       const std::vector<const AQPExpr*> &filter_exprs,
                                       const std::vector<int> &col_mapping,
                                       const std::vector<int32_t> &col_dtypes,
                                       const std::vector<ColSchema> &schema) {
    Type *i8p  = PointerType::getUnqual(Type::getInt8Ty(llctx));
    Type *i32  = Type::getInt32Ty(llctx);
    Type *i64  = Type::getInt64Ty(llctx);
    Type *i64p = PointerType::getUnqual(i64);

    StructType *ColViewTy   = StructType::get(llctx, {i8p, i64p, i32, i32});
    StructType *ChunkViewTy = StructType::get(llctx, {
        PointerType::getUnqual(ColViewTy), i64, i64});

    // int64_t fn(AQPChunkView *in, AQPChunkView *out, i8 *state)
    FunctionType *fn_ty = FunctionType::get(
        i64, {PointerType::getUnqual(ChunkViewTy),
              PointerType::getUnqual(ChunkViewTy), i8p}, false);
    Function *fn = Function::Create(fn_ty, Function::ExternalLinkage,
                                    fn_name, &mod);

    Value *in_arg    = fn->getArg(0); in_arg->setName("in");
    Value *out_arg   = fn->getArg(1); out_arg->setName("out");
    Value *state_arg = fn->getArg(2); state_arg->setName("state");
    (void)state_arg; // reserved for future use (hash tables, etc.)

    BasicBlock *entry_bb = BasicBlock::Create(llctx, "entry", fn);
    BasicBlock *loop_bb  = BasicBlock::Create(llctx, "loop", fn);
    BasicBlock *body_bb  = BasicBlock::Create(llctx, "body", fn);
    BasicBlock *write_bb = BasicBlock::Create(llctx, "write", fn);
    BasicBlock *next_bb  = BasicBlock::Create(llctx, "next", fn);
    BasicBlock *exit_bb  = BasicBlock::Create(llctx, "exit", fn);

    // Use CompileCtx for filter expression emission (reuses EmitExpr infrastructure)
    // We use a dummy AQPSelView since CompileCtx requires it, but we don't use it
    CompileCtx cc(llctx, mod, schema, in_arg, out_arg /* repurposed as sel_arg placeholder */);
    cc.b.SetInsertPoint(entry_bb);

    // Load in->nrows
    Value *nrows = cc.b.CreateLoad(i64, cc.b.CreateStructGEP(ChunkViewTy, in_arg, 1), "nrows");

    // Load input column data + validity (for filter expression evaluation)
    cc.col_data.resize(schema.size());
    cc.col_validity.resize(schema.size());
    for (size_t i = 0; i < schema.size(); i++) {
        cc.col_data[i]     = cc.LoadColData((unsigned)i);
        cc.col_validity[i] = cc.LoadColValidity((unsigned)i);
    }

    // Load output column data pointers
    Value *out_cols_pp = cc.b.CreateStructGEP(ChunkViewTy, out_arg, 0);
    Value *out_cols    = cc.b.CreateLoad(PointerType::getUnqual(ColViewTy), out_cols_pp, "out_cols");

    std::vector<Value*> out_data_ptrs;
    for (size_t oi = 0; oi < col_mapping.size(); oi++) {
        Value *col_i = cc.b.CreateGEP(ColViewTy, out_cols, ConstantInt::get(i64, oi));
        out_data_ptrs.push_back(
            cc.b.CreateLoad(i8p, cc.b.CreateStructGEP(ColViewTy, col_i, 0),
                            "out_data_" + std::to_string(oi)));
    }

    cc.b.CreateBr(loop_bb);

    // Loop header
    cc.b.SetInsertPoint(loop_bb);
    PHINode *row_i     = cc.b.CreatePHI(i64, 2, "i");
    PHINode *out_count = cc.b.CreatePHI(i64, 2, "out_count");
    row_i->addIncoming(ConstantInt::get(i64, 0), entry_bb);
    out_count->addIncoming(ConstantInt::get(i64, 0), entry_bb);
    cc.b.CreateCondBr(cc.b.CreateICmpEQ(row_i, nrows), exit_bb, body_bb);

    // Body: evaluate filter expressions
    cc.b.SetInsertPoint(body_bb);
    cc.row_idx = row_i;

    Value *match = ConstantInt::getTrue(llctx);
    if (!filter_exprs.empty()) {
        for (const AQPExpr *e : filter_exprs) {
            Value *res = EmitExpr(cc, e);
            match = cc.b.CreateAnd(match, res);
        }
    }
    BasicBlock *condBr_bb = cc.b.GetInsertBlock();
    cc.b.CreateCondBr(match, write_bb, next_bb);

    // Write: copy projected columns for this row to output
    cc.b.SetInsertPoint(write_bb);
    for (size_t oi = 0; oi < col_mapping.size(); oi++) {
        int in_i = col_mapping[oi];
        if (in_i < 0) continue;
        unsigned elem_size = DtypeElemSize(col_dtypes[oi]);
        if (elem_size == 0) continue;

        // src = in_cols[in_i].data + row_i * elem_size
        Value *src = cc.b.CreateGEP(Type::getInt8Ty(llctx), cc.col_data[in_i],
            cc.b.CreateMul(row_i, ConstantInt::get(i64, elem_size)));
        // dst = out_cols[oi].data + out_count * elem_size
        Value *dst = cc.b.CreateGEP(Type::getInt8Ty(llctx), out_data_ptrs[oi],
            cc.b.CreateMul(out_count, ConstantInt::get(i64, elem_size)));
        cc.b.CreateMemCpy(dst, MaybeAlign(1), src, MaybeAlign(1),
                          ConstantInt::get(i64, elem_size));
    }
    Value *out_count1 = cc.b.CreateAdd(out_count, ConstantInt::get(i64, 1));
    cc.b.CreateBr(next_bb);

    // Next
    cc.b.SetInsertPoint(next_bb);
    PHINode *oc_next = cc.b.CreatePHI(i64, 2, "oc_next");
    oc_next->addIncoming(out_count, condBr_bb);
    oc_next->addIncoming(out_count1, write_bb);
    Value *i_next = cc.b.CreateAdd(row_i, ConstantInt::get(i64, 1));
    row_i->addIncoming(i_next, next_bb);
    out_count->addIncoming(oc_next, next_bb);
    cc.b.CreateBr(loop_bb);

    // Exit: store output nrows and return count
    cc.b.SetInsertPoint(exit_bb);
    Value *out_nrows_ptr = cc.b.CreateStructGEP(ChunkViewTy, out_arg, 1);
    cc.b.CreateStore(out_count, out_nrows_ptr);
    cc.b.CreateRet(out_count);

    return fn;
}

// ---------------------------------------------------------------------------
// Build a hash join build function:
//   void aqp_hbuild_<id>(AQPChunkView* in, i8* hash_table)
//
// For each row: extracts key columns into a stack buffer, calls aqp_ht_insert,
// then copies payload columns into the returned payload slot.
//
// key_col_indices[i] = input chunk column index for key column i
// key_elem_sizes[i]  = byte size of key column i
// payload_col_indices/sizes = same for payload columns
// ---------------------------------------------------------------------------
struct HashColDesc {
    int      col_idx;   // chunk column index
    unsigned elem_size; // bytes per element
    int32_t  dtype;
};

static Function *BuildHashBuildFunction(LLVMContext &llctx, Module &mod,
                                        const std::string &fn_name,
                                        const std::vector<HashColDesc> &key_cols,
                                        unsigned key_width,
                                        const std::vector<HashColDesc> &payload_cols,
                                        unsigned payload_width,
                                        const std::vector<ColSchema> &schema) {
    Type *i8    = Type::getInt8Ty(llctx);
    Type *i8p   = PointerType::getUnqual(i8);
    Type *i32   = Type::getInt32Ty(llctx);
    Type *i64   = Type::getInt64Ty(llctx);
    Type *i64p  = PointerType::getUnqual(i64);
    Type *voidTy = Type::getVoidTy(llctx);

    StructType *ColViewTy   = StructType::get(llctx, {i8p, i64p, i32, i32});
    StructType *ChunkViewTy = StructType::get(llctx, {
        PointerType::getUnqual(ColViewTy), i64, i64});

    // void fn(AQPChunkView *in, i8 *hash_table)
    FunctionType *fn_ty = FunctionType::get(
        voidTy, {PointerType::getUnqual(ChunkViewTy), i8p}, false);
    Function *fn = Function::Create(fn_ty, Function::ExternalLinkage,
                                    fn_name, &mod);

    Value *in_arg = fn->getArg(0); in_arg->setName("in");
    Value *ht_arg = fn->getArg(1); ht_arg->setName("ht");

    BasicBlock *entry_bb = BasicBlock::Create(llctx, "entry", fn);
    BasicBlock *loop_bb  = BasicBlock::Create(llctx, "loop", fn);
    BasicBlock *body_bb  = BasicBlock::Create(llctx, "body", fn);
    BasicBlock *next_bb  = BasicBlock::Create(llctx, "next", fn);
    BasicBlock *exit_bb  = BasicBlock::Create(llctx, "exit", fn);

    IRBuilder<> b(entry_bb);

    // Load nrows
    Value *nrows = b.CreateLoad(i64, b.CreateStructGEP(ChunkViewTy, in_arg, 1), "nrows");

    // Load column data pointers
    Value *cols = b.CreateLoad(PointerType::getUnqual(ColViewTy),
        b.CreateStructGEP(ChunkViewTy, in_arg, 0), "cols");

    std::map<int, Value*> col_data;
    auto load_col = [&](int ci) {
        if (col_data.find(ci) == col_data.end()) {
            Value *col_i = b.CreateGEP(ColViewTy, cols, ConstantInt::get(i64, (uint64_t)ci));
            col_data[ci] = b.CreateLoad(i8p, b.CreateStructGEP(ColViewTy, col_i, 0),
                "data_" + std::to_string(ci));
        }
    };
    for (auto &kc : key_cols) load_col(kc.col_idx);
    for (auto &pc : payload_cols) load_col(pc.col_idx);

    // Alloca for key buffer (stack, fixed size)
    Value *key_buf = b.CreateAlloca(i8, ConstantInt::get(i32, key_width), "key_buf");

    // Declare aqp_ht_insert: void* aqp_ht_insert(void* ht, void* key)
    FunctionType *insert_ft = FunctionType::get(i8p, {i8p, i8p}, false);
    FunctionCallee insert_fn = mod.getOrInsertFunction("aqp_ht_insert", insert_ft);

    b.CreateBr(loop_bb);

    // Loop
    b.SetInsertPoint(loop_bb);
    PHINode *row_i = b.CreatePHI(i64, 2, "i");
    row_i->addIncoming(ConstantInt::get(i64, 0), entry_bb);
    b.CreateCondBr(b.CreateICmpEQ(row_i, nrows), exit_bb, body_bb);

    // Body: build key, insert, copy payload
    b.SetInsertPoint(body_bb);

    // Extract key columns into key_buf
    unsigned key_off = 0;
    for (auto &kc : key_cols) {
        Value *src = col_data[kc.col_idx];
        Value *elem_ptr = b.CreateGEP(i8, src,
            b.CreateMul(row_i, ConstantInt::get(i64, kc.elem_size)));
        Value *dst = b.CreateGEP(i8, key_buf, ConstantInt::get(i32, key_off));
        b.CreateMemCpy(dst, MaybeAlign(1), elem_ptr, MaybeAlign(1),
                        ConstantInt::get(i64, kc.elem_size));
        key_off += kc.elem_size;
    }

    // Insert into hash table → get payload pointer
    Value *payload_ptr = b.CreateCall(insert_fn, {ht_arg, key_buf});

    // Copy payload columns into payload slot
    unsigned pay_off = 0;
    for (auto &pc : payload_cols) {
        Value *src = col_data[pc.col_idx];
        Value *elem_ptr = b.CreateGEP(i8, src,
            b.CreateMul(row_i, ConstantInt::get(i64, pc.elem_size)));
        Value *dst = b.CreateGEP(i8, payload_ptr, ConstantInt::get(i64, pay_off));
        b.CreateMemCpy(dst, MaybeAlign(1), elem_ptr, MaybeAlign(1),
                        ConstantInt::get(i64, pc.elem_size));
        pay_off += pc.elem_size;
    }

    b.CreateBr(next_bb);

    // Next
    b.SetInsertPoint(next_bb);
    Value *i_next = b.CreateAdd(row_i, ConstantInt::get(i64, 1));
    row_i->addIncoming(i_next, next_bb);
    b.CreateBr(loop_bb);

    // Exit
    b.SetInsertPoint(exit_bb);
    b.CreateRetVoid();

    return fn;
}

// ---------------------------------------------------------------------------
// Build a hash join probe function:
//   uint64_t aqp_hprobe_<id>(AQPChunkView* probe, i8* hash_table, AQPSelView* sel)
//
// For each probe row: extracts key, calls aqp_ht_probe; if found, writes
// row index to selection vector.  Returns count of matching rows.
// ---------------------------------------------------------------------------
static Function *BuildHashProbeFunction(LLVMContext &llctx, Module &mod,
                                        const std::string &fn_name,
                                        const std::vector<HashColDesc> &probe_key_cols,
                                        unsigned key_width) {
    Type *i8    = Type::getInt8Ty(llctx);
    Type *i8p   = PointerType::getUnqual(i8);
    Type *i32   = Type::getInt32Ty(llctx);
    Type *i64   = Type::getInt64Ty(llctx);
    Type *i64p  = PointerType::getUnqual(i64);

    StructType *ColViewTy   = StructType::get(llctx, {i8p, i64p, i32, i32});
    StructType *ChunkViewTy = StructType::get(llctx, {
        PointerType::getUnqual(ColViewTy), i64, i64});
    StructType *SelViewTy   = StructType::get(llctx, {
        PointerType::getUnqual(i32), i32});

    // uint64_t fn(AQPChunkView *probe, i8 *ht, AQPSelView *sel)
    FunctionType *fn_ty = FunctionType::get(
        i64, {PointerType::getUnqual(ChunkViewTy), i8p,
              PointerType::getUnqual(SelViewTy)}, false);
    Function *fn = Function::Create(fn_ty, Function::ExternalLinkage,
                                    fn_name, &mod);

    Value *probe_arg = fn->getArg(0); probe_arg->setName("probe");
    Value *ht_arg    = fn->getArg(1); ht_arg->setName("ht");
    Value *sel_arg   = fn->getArg(2); sel_arg->setName("sel");

    BasicBlock *entry_bb = BasicBlock::Create(llctx, "entry", fn);
    BasicBlock *loop_bb  = BasicBlock::Create(llctx, "loop", fn);
    BasicBlock *body_bb  = BasicBlock::Create(llctx, "body", fn);
    BasicBlock *found_bb = BasicBlock::Create(llctx, "found", fn);
    BasicBlock *next_bb  = BasicBlock::Create(llctx, "next", fn);
    BasicBlock *exit_bb  = BasicBlock::Create(llctx, "exit", fn);

    IRBuilder<> b(entry_bb);

    Value *nrows = b.CreateLoad(i64, b.CreateStructGEP(ChunkViewTy, probe_arg, 1), "nrows");
    Value *cols = b.CreateLoad(PointerType::getUnqual(ColViewTy),
        b.CreateStructGEP(ChunkViewTy, probe_arg, 0), "cols");

    // Load probe column data
    std::map<int, Value*> col_data;
    for (auto &kc : probe_key_cols) {
        if (col_data.find(kc.col_idx) == col_data.end()) {
            Value *col_i = b.CreateGEP(ColViewTy, cols, ConstantInt::get(i64, (uint64_t)kc.col_idx));
            col_data[kc.col_idx] = b.CreateLoad(i8p,
                b.CreateStructGEP(ColViewTy, col_i, 0), "data_" + std::to_string(kc.col_idx));
        }
    }

    Value *key_buf = b.CreateAlloca(i8, ConstantInt::get(i32, key_width), "key_buf");

    // Load sel->indices
    Value *sel_idx_ptr = b.CreateLoad(PointerType::getUnqual(i32),
        b.CreateStructGEP(SelViewTy, sel_arg, 0), "sel_indices");

    // Declare aqp_ht_probe: void* aqp_ht_probe(void* ht, void* key)
    FunctionType *probe_ft = FunctionType::get(i8p, {i8p, i8p}, false);
    FunctionCallee probe_fn = mod.getOrInsertFunction("aqp_ht_probe", probe_ft);

    b.CreateBr(loop_bb);

    // Loop
    b.SetInsertPoint(loop_bb);
    PHINode *row_i     = b.CreatePHI(i64, 2, "i");
    PHINode *out_count = b.CreatePHI(i64, 2, "out_count");
    row_i->addIncoming(ConstantInt::get(i64, 0), entry_bb);
    out_count->addIncoming(ConstantInt::get(i64, 0), entry_bb);
    b.CreateCondBr(b.CreateICmpEQ(row_i, nrows), exit_bb, body_bb);

    // Body: extract key, probe
    b.SetInsertPoint(body_bb);
    unsigned key_off = 0;
    for (auto &kc : probe_key_cols) {
        Value *src = col_data[kc.col_idx];
        Value *elem_ptr = b.CreateGEP(i8, src,
            b.CreateMul(row_i, ConstantInt::get(i64, kc.elem_size)));
        Value *dst = b.CreateGEP(i8, key_buf, ConstantInt::get(i32, key_off));
        b.CreateMemCpy(dst, MaybeAlign(1), elem_ptr, MaybeAlign(1),
                        ConstantInt::get(i64, kc.elem_size));
        key_off += kc.elem_size;
    }

    Value *result = b.CreateCall(probe_fn, {ht_arg, key_buf});
    Value *is_found = b.CreateICmpNE(
        b.CreatePtrToInt(result, i64), ConstantInt::get(i64, 0));
    b.CreateCondBr(is_found, found_bb, next_bb);

    // Found: store row index in selection vector
    b.SetInsertPoint(found_bb);
    Value *dst = b.CreateGEP(i32, sel_idx_ptr, out_count);
    b.CreateStore(b.CreateTrunc(row_i, i32), dst);
    Value *out_count1 = b.CreateAdd(out_count, ConstantInt::get(i64, 1));
    b.CreateBr(next_bb);

    // Next
    b.SetInsertPoint(next_bb);
    PHINode *oc_next = b.CreatePHI(i64, 2, "oc_next");
    oc_next->addIncoming(out_count, body_bb);
    oc_next->addIncoming(out_count1, found_bb);
    Value *i_next = b.CreateAdd(row_i, ConstantInt::get(i64, 1));
    row_i->addIncoming(i_next, next_bb);
    out_count->addIncoming(oc_next, next_bb);
    b.CreateBr(loop_bb);

    // Exit
    b.SetInsertPoint(exit_bb);
    b.CreateStore(b.CreateTrunc(out_count, i32),
        b.CreateStructGEP(SelViewTy, sel_arg, 1));
    b.CreateRet(out_count);

    return fn;
}

// ---------------------------------------------------------------------------
// Build an ungrouped aggregate update function:
//   void aqp_agg_<id>(AQPChunkView* in, i8* agg_state)
//
// Loops over input rows and updates accumulator state.
// agg_state layout: 8 bytes per aggregate (16 for AVG: sum + count).
//
// agg_ops[i] = { input_col_idx, agg_type, state_offset, dtype }
// ---------------------------------------------------------------------------
struct AggOp {
    int           col_idx;       // input chunk column index (-1 for COUNT*)
    int32_t       agg_type;      // SimplestAggFnType cast to int32_t
    unsigned      state_offset;  // byte offset in agg_state
    int32_t       dtype;         // AQP_DTYPE_* of the column
};

static Function *BuildAggUpdateFunction(LLVMContext &llctx, Module &mod,
                                        const std::string &fn_name,
                                        const std::vector<AggOp> &agg_ops,
                                        unsigned total_state_size,
                                        const std::vector<ColSchema> &schema) {
    Type *i8p  = PointerType::getUnqual(Type::getInt8Ty(llctx));
    Type *i32  = Type::getInt32Ty(llctx);
    Type *i64  = Type::getInt64Ty(llctx);
    Type *i64p = PointerType::getUnqual(i64);
    Type *f64  = Type::getDoubleTy(llctx);
    Type *voidTy = Type::getVoidTy(llctx);

    StructType *ColViewTy   = StructType::get(llctx, {i8p, i64p, i32, i32});
    StructType *ChunkViewTy = StructType::get(llctx, {
        PointerType::getUnqual(ColViewTy), i64, i64});

    // void fn(AQPChunkView *in, i8 *agg_state)
    FunctionType *fn_ty = FunctionType::get(
        voidTy, {PointerType::getUnqual(ChunkViewTy), i8p}, false);
    Function *fn = Function::Create(fn_ty, Function::ExternalLinkage,
                                    fn_name, &mod);

    Value *in_arg    = fn->getArg(0); in_arg->setName("in");
    Value *state_arg = fn->getArg(1); state_arg->setName("state");

    BasicBlock *entry_bb = BasicBlock::Create(llctx, "entry", fn);
    BasicBlock *loop_bb  = BasicBlock::Create(llctx, "loop", fn);
    BasicBlock *body_bb  = BasicBlock::Create(llctx, "body", fn);
    BasicBlock *next_bb  = BasicBlock::Create(llctx, "next", fn);
    BasicBlock *exit_bb  = BasicBlock::Create(llctx, "exit", fn);

    IRBuilder<> b(entry_bb);

    // Load nrows
    Value *nrows_ptr = b.CreateStructGEP(ChunkViewTy, in_arg, 1);
    Value *nrows     = b.CreateLoad(i64, nrows_ptr, "nrows");

    // Load column data pointers
    Value *cols_pp = b.CreateStructGEP(ChunkViewTy, in_arg, 0);
    Value *cols    = b.CreateLoad(PointerType::getUnqual(ColViewTy), cols_pp, "cols");

    // Pre-load data pointers for columns used by agg functions
    std::map<int, Value*> col_data_ptrs;
    std::map<int, Value*> col_validity_ptrs;
    for (const auto &op : agg_ops) {
        if (op.col_idx >= 0 && col_data_ptrs.find(op.col_idx) == col_data_ptrs.end()) {
            Value *col_i = b.CreateGEP(ColViewTy, cols,
                ConstantInt::get(i64, (uint64_t)op.col_idx));
            col_data_ptrs[op.col_idx] = b.CreateLoad(i8p,
                b.CreateStructGEP(ColViewTy, col_i, 0), "data_" + std::to_string(op.col_idx));
            col_validity_ptrs[op.col_idx] = b.CreateLoad(i64p,
                b.CreateStructGEP(ColViewTy, col_i, 1), "val_" + std::to_string(op.col_idx));
        }
    }

    b.CreateBr(loop_bb);

    // Loop header
    b.SetInsertPoint(loop_bb);
    PHINode *row_i = b.CreatePHI(i64, 2, "i");
    row_i->addIncoming(ConstantInt::get(i64, 0), entry_bb);
    Value *done = b.CreateICmpEQ(row_i, nrows, "done");
    b.CreateCondBr(done, exit_bb, body_bb);

    // Loop body — update accumulators
    b.SetInsertPoint(body_bb);

    for (const auto &op : agg_ops) {
        // CountStar: always increment, no column needed
        if (op.agg_type == 6 /* CountStar */) {
            Value *acc_ptr = b.CreateBitCast(
                b.CreateGEP(Type::getInt8Ty(llctx), state_arg,
                    ConstantInt::get(i64, op.state_offset)),
                PointerType::getUnqual(i64));
            Value *acc = b.CreateLoad(i64, acc_ptr, "count_star");
            b.CreateStore(b.CreateAdd(acc, ConstantInt::get(i64, 1)), acc_ptr);
            continue;
        }

        if (op.col_idx < 0) continue;

        // Check validity (skip NULL rows)
        Value *validity = col_validity_ptrs[op.col_idx];

        // Capture current block — may be body_bb (first op) or previous op's cont_bb
        BasicBlock *pre_check_bb = b.GetInsertBlock();

        BasicBlock *check_bb = BasicBlock::Create(llctx, "check_val", fn);
        BasicBlock *valid_bb = BasicBlock::Create(llctx, "valid", fn);
        BasicBlock *cont_bb  = BasicBlock::Create(llctx, "cont", fn);

        // If validity pointer is non-null, check the bit; else all valid
        Value *val_nonnull = b.CreateICmpNE(
            b.CreatePtrToInt(validity, i64), ConstantInt::get(i64, 0));
        b.CreateCondBr(val_nonnull, check_bb, valid_bb);

        b.SetInsertPoint(check_bb);
        Value *word_idx = b.CreateLShr(row_i, ConstantInt::get(i64, 6));
        Value *bit_idx  = b.CreateAnd(row_i, ConstantInt::get(i64, 63));
        Value *word     = b.CreateLoad(i64, b.CreateGEP(i64, validity, word_idx));
        Value *bit      = b.CreateAnd(b.CreateLShr(word, bit_idx), ConstantInt::get(i64, 1));
        Value *bit_valid = b.CreateICmpNE(bit, ConstantInt::get(i64, 0));
        b.CreateCondBr(bit_valid, valid_bb, cont_bb);

        // Valid path: update accumulator
        b.SetInsertPoint(valid_bb);
        PHINode *came_from = b.CreatePHI(Type::getInt1Ty(llctx), 2, "from_valid");
        came_from->addIncoming(ConstantInt::getTrue(llctx), pre_check_bb); // all-valid path
        came_from->addIncoming(ConstantInt::getTrue(llctx), check_bb);     // bit-valid path
        (void)came_from;

        Value *data_ptr = col_data_ptrs[op.col_idx];
        Value *acc_ptr = b.CreateBitCast(
            b.CreateGEP(Type::getInt8Ty(llctx), state_arg,
                ConstantInt::get(i64, op.state_offset)),
            PointerType::getUnqual(i64));

        bool is_float = (op.dtype == AQP_DTYPE_FLOAT || op.dtype == AQP_DTYPE_DOUBLE);

        // Load current row value
        Value *val = nullptr;
        if (op.dtype == AQP_DTYPE_INT32 || op.dtype == AQP_DTYPE_DATE) {
            Value *typed_ptr = b.CreateBitCast(data_ptr, PointerType::getUnqual(i32));
            Value *elem = b.CreateLoad(i32, b.CreateGEP(i32, typed_ptr, row_i));
            val = b.CreateSExt(elem, i64); // extend to i64 for accumulator
        } else if (op.dtype == AQP_DTYPE_INT64) {
            Value *typed_ptr = b.CreateBitCast(data_ptr, PointerType::getUnqual(i64));
            val = b.CreateLoad(i64, b.CreateGEP(i64, typed_ptr, row_i));
        } else if (op.dtype == AQP_DTYPE_INT8) {
            Value *elem = b.CreateLoad(Type::getInt8Ty(llctx),
                b.CreateGEP(Type::getInt8Ty(llctx), data_ptr, row_i));
            val = b.CreateSExt(elem, i64);
        } else if (op.dtype == AQP_DTYPE_INT16) {
            Value *typed_ptr = b.CreateBitCast(data_ptr, PointerType::getUnqual(Type::getInt16Ty(llctx)));
            Value *elem = b.CreateLoad(Type::getInt16Ty(llctx), b.CreateGEP(Type::getInt16Ty(llctx), typed_ptr, row_i));
            val = b.CreateSExt(elem, i64);
        } else if (op.dtype == AQP_DTYPE_DOUBLE) {
            acc_ptr = b.CreateBitCast(
                b.CreateGEP(Type::getInt8Ty(llctx), state_arg,
                    ConstantInt::get(i64, op.state_offset)),
                PointerType::getUnqual(f64));
            Value *typed_ptr = b.CreateBitCast(data_ptr, PointerType::getUnqual(f64));
            val = b.CreateLoad(f64, b.CreateGEP(f64, typed_ptr, row_i));
        } else if (op.dtype == AQP_DTYPE_FLOAT) {
            acc_ptr = b.CreateBitCast(
                b.CreateGEP(Type::getInt8Ty(llctx), state_arg,
                    ConstantInt::get(i64, op.state_offset)),
                PointerType::getUnqual(f64));
            Value *typed_ptr = b.CreateBitCast(data_ptr, PointerType::getUnqual(Type::getFloatTy(llctx)));
            Value *fval = b.CreateLoad(Type::getFloatTy(llctx),
                b.CreateGEP(Type::getFloatTy(llctx), typed_ptr, row_i));
            val = b.CreateFPExt(fval, f64); // promote to double for accumulator
        } else {
            // Unsupported dtype — skip
            b.CreateBr(cont_bb);
            b.SetInsertPoint(cont_bb);
            continue;
        }

        // Update accumulator based on agg type
        // SimplestAggFnType: Min=1, Max=2, Sum=3, Average=4, Count=5, CountStar=6
        Value *acc = is_float ? b.CreateLoad(f64, acc_ptr, "acc_f")
                              : b.CreateLoad(i64, acc_ptr, "acc_i");

        switch (op.agg_type) {
        case 3: /* Sum */ {
            Value *new_acc = is_float ? b.CreateFAdd(acc, val) : b.CreateAdd(acc, val);
            b.CreateStore(new_acc, acc_ptr);
            break;
        }
        case 5: /* Count */ {
            // Count non-null: just increment i64 accumulator
            Value *cnt_ptr = b.CreateBitCast(
                b.CreateGEP(Type::getInt8Ty(llctx), state_arg,
                    ConstantInt::get(i64, op.state_offset)),
                PointerType::getUnqual(i64));
            Value *cnt = b.CreateLoad(i64, cnt_ptr);
            b.CreateStore(b.CreateAdd(cnt, ConstantInt::get(i64, 1)), cnt_ptr);
            break;
        }
        case 1: /* Min */ {
            Value *cmp = is_float ? b.CreateFCmpOLT(val, acc) : b.CreateICmpSLT(val, acc);
            Value *new_acc = b.CreateSelect(cmp, val, acc);
            b.CreateStore(new_acc, acc_ptr);
            break;
        }
        case 2: /* Max */ {
            Value *cmp = is_float ? b.CreateFCmpOGT(val, acc) : b.CreateICmpSGT(val, acc);
            Value *new_acc = b.CreateSelect(cmp, val, acc);
            b.CreateStore(new_acc, acc_ptr);
            break;
        }
        case 4: /* Average */ {
            // AVG uses 16 bytes: [sum:8, count:8]
            Value *sum_ptr = acc_ptr; // already points to state_offset
            Value *cnt_ptr = b.CreateBitCast(
                b.CreateGEP(Type::getInt8Ty(llctx), state_arg,
                    ConstantInt::get(i64, op.state_offset + 8)),
                PointerType::getUnqual(i64));
            if (is_float) {
                Value *sum = b.CreateLoad(f64, sum_ptr);
                b.CreateStore(b.CreateFAdd(sum, val), sum_ptr);
            } else {
                // For integer AVG, accumulate as i64 then divide at finalize
                Value *sum_i = b.CreateLoad(i64, b.CreateBitCast(sum_ptr, PointerType::getUnqual(i64)));
                b.CreateStore(b.CreateAdd(sum_i, val), b.CreateBitCast(sum_ptr, PointerType::getUnqual(i64)));
            }
            Value *cnt = b.CreateLoad(i64, cnt_ptr);
            b.CreateStore(b.CreateAdd(cnt, ConstantInt::get(i64, 1)), cnt_ptr);
            break;
        }
        default:
            break;
        }

        b.CreateBr(cont_bb);
        b.SetInsertPoint(cont_bb);
    }

    // After processing all agg ops for this row, go to next
    // (cont_bb from the last agg op, or body_bb if no ops ran)
    BasicBlock *cur_bb = b.GetInsertBlock();
    if (cur_bb != next_bb) {
        b.CreateBr(next_bb);
        b.SetInsertPoint(next_bb);
    }

    Value *i_next = b.CreateAdd(row_i, ConstantInt::get(i64, 1), "i_next");
    row_i->addIncoming(i_next, next_bb);
    b.CreateBr(loop_bb);

    // Exit
    b.SetInsertPoint(exit_bb);
    b.CreateRetVoid();

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

AQPOperatorFn IrToLlvmCompiler::CompileProjection(
        const AQPStmt &proj_node,
        const std::vector<ColSchema> &in_schema) {

    // Build column mapping and dtype list:
    // col_mapping[i] = input column index for output column i
    // col_dtypes[i]  = AQP_DTYPE_* for output column i (determines element size for memcpy)
    std::vector<int> col_mapping;
    std::vector<int32_t> col_dtypes;
    for (const auto &attr : proj_node.target_list) {
        int found = -1;
        int32_t dtype = AQP_DTYPE_OTHER;
        for (int i = 0; i < (int)in_schema.size(); i++) {
            if (in_schema[i].table_idx == attr->GetTableIndex() &&
                in_schema[i].col_idx  == attr->GetColumnIndex()) {
                found = i;
                dtype = in_schema[i].dtype;
                break;
            }
        }
        col_mapping.push_back(found);
        col_dtypes.push_back(dtype);
    }
    if (col_mapping.empty()) return nullptr;

    uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
    std::string fn_name = "aqp_proj_" + std::to_string(fn_id);

    auto ctx = std::make_unique<LLVMContext>();
    auto mod = std::make_unique<Module>("aqp_proj_mod", *ctx);

    Function *fn = BuildProjectionFunction(*ctx, *mod, fn_name, col_mapping, col_dtypes);
    if (!fn) return nullptr;

    std::string err;
    raw_string_ostream es(err);
    if (verifyFunction(*fn, &es)) {
        std::cerr << "[AQP-JIT] verifyFunction failed (proj): " << es.str() << "\n";
        return nullptr;
    }

    OptimiseModule(*mod, use_o3_);

    auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
    if (auto e = impl_->jit->addIRModule(std::move(tsm))) {
        std::cerr << "[AQP-JIT] addIRModule failed (proj)\n";
        logAllUnhandledErrors(std::move(e), errs());
        return nullptr;
    }

    auto sym = impl_->jit->lookup(fn_name);
    if (!sym) {
        std::cerr << "[AQP-JIT] lookup failed for " << fn_name << "\n";
        logAllUnhandledErrors(sym.takeError(), errs());
        return nullptr;
    }

    return jitTargetAddressToFunction<AQPOperatorFn>(sym->getAddress());
}

void *IrToLlvmCompiler::CompileAggUpdate(
        const AQPStmt &agg_node,
        const std::vector<ColSchema> &in_schema) {

    auto *agg = dynamic_cast<const ir_sql_converter::SimplestAggregate *>(&agg_node);
    if (!agg || agg->agg_fns.empty()) return nullptr;

    // For now: only ungrouped aggregates (no GROUP BY).
    // Grouped aggregates use the hash table and will be added in Phase 2C.
    if (!agg->groups.empty()) {
        std::cerr << "[AQP-JIT] grouped aggregate not yet supported → interpreter\n";
        return nullptr;
    }

    // Build AggOp descriptors and compute state layout
    std::vector<AggOp> agg_ops;
    unsigned state_offset = 0;
    for (const auto &fn_pair : agg->agg_fns) {
        AggOp op;
        op.agg_type = static_cast<int32_t>(fn_pair.second);
        op.state_offset = state_offset;

        if (fn_pair.second == ir_sql_converter::SimplestAggFnType::CountStar) {
            op.col_idx = -1;
            op.dtype   = AQP_DTYPE_INT64;
            state_offset += 8;
        } else {
            // Find column in input schema
            op.col_idx = -1;
            op.dtype   = AQP_DTYPE_OTHER;
            for (int i = 0; i < (int)in_schema.size(); i++) {
                if (in_schema[i].table_idx == fn_pair.first->GetTableIndex() &&
                    in_schema[i].col_idx  == fn_pair.first->GetColumnIndex()) {
                    op.col_idx = i;
                    op.dtype   = in_schema[i].dtype;
                    break;
                }
            }
            if (op.col_idx < 0) {
                std::cerr << "[AQP-JIT] agg col not in schema: table="
                          << fn_pair.first->GetTableIndex()
                          << " col=" << fn_pair.first->GetColumnIndex() << "\n";
                continue;  // skip this agg function
            }
            if (fn_pair.second == ir_sql_converter::SimplestAggFnType::Average)
                state_offset += 16;  // sum + count
            else
                state_offset += 8;
        }
        agg_ops.push_back(op);
    }
    if (agg_ops.empty()) return nullptr;

    uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
    std::string fn_name = "aqp_agg_" + std::to_string(fn_id);

    auto ctx = std::make_unique<LLVMContext>();
    auto mod = std::make_unique<Module>("aqp_agg_mod", *ctx);

    Function *fn = BuildAggUpdateFunction(*ctx, *mod, fn_name, agg_ops, state_offset, in_schema);
    if (!fn) return nullptr;

    std::string err;
    raw_string_ostream es(err);
    if (verifyFunction(*fn, &es)) {
        std::cerr << "[AQP-JIT] verifyFunction failed (agg): " << es.str() << "\n";
        return nullptr;
    }

    OptimiseModule(*mod, use_o3_);

    auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
    if (auto e = impl_->jit->addIRModule(std::move(tsm))) {
        std::cerr << "[AQP-JIT] addIRModule failed (agg)\n";
        logAllUnhandledErrors(std::move(e), errs());
        return nullptr;
    }

    auto sym = impl_->jit->lookup(fn_name);
    if (!sym) {
        std::cerr << "[AQP-JIT] lookup failed for " << fn_name << "\n";
        logAllUnhandledErrors(sym.takeError(), errs());
        return nullptr;
    }

    std::cerr << "[AQP-JIT] compiled agg fn=" << fn_name
              << "  ops=" << agg_ops.size()
              << "  state_bytes=" << state_offset << "\n";

    return reinterpret_cast<void*>(sym->getAddress());
}

void *IrToLlvmCompiler::CompileHashBuild(
        const AQPStmt &hash_node,
        const std::vector<ColSchema> &in_schema) {

    auto *hash = dynamic_cast<const ir_sql_converter::SimplestHash *>(&hash_node);
    if (!hash || hash->hash_keys.empty()) return nullptr;

    // Build key column descriptors from hash_keys
    std::vector<HashColDesc> key_cols;
    unsigned key_width = 0;
    for (const auto &hk : hash->hash_keys) {
        HashColDesc kc;
        kc.col_idx = -1;
        kc.dtype = AQP_DTYPE_OTHER;
        for (int i = 0; i < (int)in_schema.size(); i++) {
            if (in_schema[i].table_idx == hk->GetTableIndex() &&
                in_schema[i].col_idx  == hk->GetColumnIndex()) {
                kc.col_idx = i;
                kc.dtype   = in_schema[i].dtype;
                break;
            }
        }
        if (kc.col_idx < 0) {
            std::cerr << "[AQP-JIT] hash key not in schema: table="
                      << hk->GetTableIndex() << " col=" << hk->GetColumnIndex() << "\n";
            return nullptr;
        }
        kc.elem_size = DtypeElemSize(kc.dtype);
        if (kc.elem_size == 0) return nullptr;
        key_width += kc.elem_size;
        key_cols.push_back(kc);
    }

    // Payload = all input columns (for now; could be optimized to needed cols only)
    std::vector<HashColDesc> payload_cols;
    unsigned payload_width = 0;
    for (int i = 0; i < (int)in_schema.size(); i++) {
        HashColDesc pc;
        pc.col_idx = i;
        pc.dtype = in_schema[i].dtype;
        pc.elem_size = DtypeElemSize(pc.dtype);
        if (pc.elem_size == 0) continue;
        payload_width += pc.elem_size;
        payload_cols.push_back(pc);
    }

    uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
    std::string fn_name = "aqp_hbuild_" + std::to_string(fn_id);

    auto ctx = std::make_unique<LLVMContext>();
    auto mod = std::make_unique<Module>("aqp_hbuild_mod", *ctx);

    Function *fn = BuildHashBuildFunction(*ctx, *mod, fn_name,
                                          key_cols, key_width,
                                          payload_cols, payload_width, in_schema);
    if (!fn) return nullptr;

    std::string err;
    raw_string_ostream es(err);
    if (verifyFunction(*fn, &es)) {
        std::cerr << "[AQP-JIT] verifyFunction failed (hbuild): " << es.str() << "\n";
        return nullptr;
    }

    OptimiseModule(*mod, use_o3_);

    auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
    if (auto e = impl_->jit->addIRModule(std::move(tsm))) {
        logAllUnhandledErrors(std::move(e), errs());
        return nullptr;
    }

    auto sym = impl_->jit->lookup(fn_name);
    if (!sym) {
        logAllUnhandledErrors(sym.takeError(), errs());
        return nullptr;
    }

    std::cerr << "[AQP-JIT] compiled hash build fn=" << fn_name
              << "  key_width=" << key_width
              << "  payload_width=" << payload_width << "\n";

    return reinterpret_cast<void*>(sym->getAddress());
}

void *IrToLlvmCompiler::CompileHashProbe(
        const AQPStmt &join_node,
        const std::vector<ColSchema> &probe_schema) {

    auto *join = dynamic_cast<const ir_sql_converter::SimplestJoin *>(&join_node);
    if (!join || join->join_conditions.empty()) return nullptr;

    // For inner/equi-join: extract probe-side key columns from join conditions.
    // Each condition is attr_left OP attr_right; the probe side is the one
    // that matches the probe_schema.
    std::vector<HashColDesc> probe_key_cols;
    unsigned key_width = 0;
    for (const auto &cond : join->join_conditions) {
        // Try left attr first
        HashColDesc kc;
        kc.col_idx = -1;
        for (int i = 0; i < (int)probe_schema.size(); i++) {
            if (probe_schema[i].table_idx == cond->left_attr->GetTableIndex() &&
                probe_schema[i].col_idx  == cond->left_attr->GetColumnIndex()) {
                kc.col_idx = i;
                kc.dtype = probe_schema[i].dtype;
                break;
            }
        }
        // Try right attr if left didn't match
        if (kc.col_idx < 0) {
            for (int i = 0; i < (int)probe_schema.size(); i++) {
                if (probe_schema[i].table_idx == cond->right_attr->GetTableIndex() &&
                    probe_schema[i].col_idx  == cond->right_attr->GetColumnIndex()) {
                    kc.col_idx = i;
                    kc.dtype = probe_schema[i].dtype;
                    break;
                }
            }
        }
        if (kc.col_idx < 0) {
            std::cerr << "[AQP-JIT] probe key not in schema\n";
            return nullptr;
        }
        kc.elem_size = DtypeElemSize(kc.dtype);
        if (kc.elem_size == 0) return nullptr;
        key_width += kc.elem_size;
        probe_key_cols.push_back(kc);
    }

    uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
    std::string fn_name = "aqp_hprobe_" + std::to_string(fn_id);

    auto ctx = std::make_unique<LLVMContext>();
    auto mod = std::make_unique<Module>("aqp_hprobe_mod", *ctx);

    Function *fn = BuildHashProbeFunction(*ctx, *mod, fn_name,
                                           probe_key_cols, key_width);
    if (!fn) return nullptr;

    std::string err;
    raw_string_ostream es(err);
    if (verifyFunction(*fn, &es)) {
        std::cerr << "[AQP-JIT] verifyFunction failed (hprobe): " << es.str() << "\n";
        return nullptr;
    }

    OptimiseModule(*mod, use_o3_);

    auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
    if (auto e = impl_->jit->addIRModule(std::move(tsm))) {
        logAllUnhandledErrors(std::move(e), errs());
        return nullptr;
    }

    auto sym = impl_->jit->lookup(fn_name);
    if (!sym) {
        logAllUnhandledErrors(sym.takeError(), errs());
        return nullptr;
    }

    std::cerr << "[AQP-JIT] compiled hash probe fn=" << fn_name
              << "  key_width=" << key_width
              << "  probe_cols=" << probe_key_cols.size() << "\n";

    return reinterpret_cast<void*>(sym->getAddress());
}

AQPPipelineFn IrToLlvmCompiler::CompilePipeline(
        const AQPStmt *filter_node,
        const AQPStmt *proj_node,
        const std::vector<ColSchema> &in_schema) {

    // Collect filter expressions
    std::vector<const AQPExpr*> filter_exprs;
    if (filter_node) {
        for (const auto &qe : filter_node->qual_vec)
            filter_exprs.push_back(qe.get());
    }

    // Build projection column mapping
    std::vector<int> col_mapping;
    std::vector<int32_t> col_dtypes;
    if (proj_node && !proj_node->target_list.empty()) {
        for (const auto &attr : proj_node->target_list) {
            int found = -1;
            int32_t dtype = AQP_DTYPE_OTHER;
            for (int i = 0; i < (int)in_schema.size(); i++) {
                if (in_schema[i].table_idx == attr->GetTableIndex() &&
                    in_schema[i].col_idx  == attr->GetColumnIndex()) {
                    found = i;
                    dtype = in_schema[i].dtype;
                    break;
                }
            }
            col_mapping.push_back(found);
            col_dtypes.push_back(dtype);
        }
    } else {
        // No projection: pass-through all input columns
        for (int i = 0; i < (int)in_schema.size(); i++) {
            col_mapping.push_back(i);
            col_dtypes.push_back(in_schema[i].dtype);
        }
    }

    if (filter_exprs.empty() && !proj_node) return nullptr; // nothing to fuse

    uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
    std::string fn_name = "aqp_pipe_" + std::to_string(fn_id);

    auto ctx = std::make_unique<LLVMContext>();
    auto mod = std::make_unique<Module>("aqp_pipe_mod", *ctx);

    Function *fn = BuildPipelineFunction(*ctx, *mod, fn_name,
                                          filter_exprs, col_mapping,
                                          col_dtypes, in_schema);
    if (!fn) return nullptr;

    std::string err;
    raw_string_ostream es(err);
    if (verifyFunction(*fn, &es)) {
        std::cerr << "[AQP-JIT] verifyFunction failed (pipeline): " << es.str() << "\n";
        return nullptr;
    }

    OptimiseModule(*mod, use_o3_);

    auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
    if (auto e = impl_->jit->addIRModule(std::move(tsm))) {
        std::cerr << "[AQP-JIT] addIRModule failed (pipeline)\n";
        logAllUnhandledErrors(std::move(e), errs());
        return nullptr;
    }

    auto sym = impl_->jit->lookup(fn_name);
    if (!sym) {
        std::cerr << "[AQP-JIT] lookup failed for " << fn_name << "\n";
        logAllUnhandledErrors(sym.takeError(), errs());
        return nullptr;
    }

    std::cerr << "[AQP-JIT] compiled pipeline fn=" << fn_name
              << "  filter_exprs=" << filter_exprs.size()
              << "  out_cols=" << col_mapping.size() << "\n";

    return jitTargetAddressToFunction<AQPPipelineFn>(sym->getAddress());
}

} // namespace aqp_jit
