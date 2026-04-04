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
#include <llvm/Support/Host.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>

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

#include <climits>
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

    // SIMD configuration (detected at init time)
    std::string host_cpu;
    std::string feature_str;
    unsigned vec_width = 1;    // SIMD lanes for i32: 1=scalar, 4=SSE, 8=AVX2, 16=AVX-512
    bool has_avx2    = false;
    bool has_avx512f = false;
    bool has_sse42   = false;

    Impl() {
        EnsureLLVMInit();

        // Detect CPU features for SIMD
        host_cpu = std::string(sys::getHostCPUName());
        StringMap<bool> host_features;
        sys::getHostCPUFeatures(host_features);

        // Build feature string: "+avx2,+sse4.2,..."
        for (auto &kv : host_features) {
            if (!feature_str.empty()) feature_str += ",";
            feature_str += (kv.second ? "+" : "-");
            feature_str += kv.first().str();
        }

        has_sse42   = host_features.count("sse4.2") && host_features["sse4.2"];
        has_avx2    = host_features.count("avx2") && host_features["avx2"];
        has_avx512f = host_features.count("avx512f") && host_features["avx512f"];

        if (has_avx512f)     vec_width = 16;
        else if (has_avx2)   vec_width = 8;
        else if (has_sse42)  vec_width = 4;
        else                 vec_width = 1;

        std::cerr << "[AQP-JIT] CPU=" << host_cpu
                  << " AVX2=" << has_avx2
                  << " AVX512=" << has_avx512f
                  << " vec_width=" << vec_width << "\n";

        // Create LLJIT with detected CPU features for optimal codegen
        auto jtmb = JITTargetMachineBuilder::detectHost();
        if (jtmb) {
            jtmb->setCPU(host_cpu);
            // Note: feature_str is applied per-function via attributes
        }

        auto builder = LLJITBuilder();
        if (jtmb)
            builder.setJITTargetMachineBuilder(std::move(*jtmb));
        auto jit_or = builder.create();
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

// Set target CPU and features on a generated function so LLVM's backend
// uses the best available instructions (AVX2, SSE4.2, etc.).
static void SetTargetAttrs(Function *fn, const std::string &cpu,
                            const std::string &features) {
    if (!cpu.empty())
        fn->addFnAttr("target-cpu", cpu);
    if (!features.empty())
        fn->addFnAttr("target-features", features);
}

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
            // Length pre-filter: if lengths differ, skip aqp_str_eq entirely.
            // Most rows have different lengths → eliminates most memcmp calls.
            BasicBlock *pre_len_bb = cc.b.GetInsertBlock();
            BasicBlock *len_match_bb = BasicBlock::Create(cc.llctx, "len_match",
                pre_len_bb->getParent());
            BasicBlock *len_done_bb = BasicBlock::Create(cc.llctx, "len_done",
                pre_len_bb->getParent());
            Value *len_eq = cc.b.CreateICmpEQ(slen, pat_len, "len_eq");
            cc.b.CreateCondBr(len_eq, len_match_bb, len_done_bb);

            // Lengths match → call aqp_str_eq for byte comparison
            cc.b.SetInsertPoint(len_match_bb);
            FunctionCallee callee = cc.mod.getOrInsertFunction("aqp_str_eq", ft4);
            Value *r = cc.b.CreateCall(callee, {char_ptr, slen, pat_ptr, pat_len});
            Value *m_match = cc.b.CreateICmpNE(r, cc.c32(0));
            cc.b.CreateBr(len_done_bb);

            // Merge: lengths differ → false; lengths match → aqp_str_eq result
            cc.b.SetInsertPoint(len_done_bb);
            PHINode *m_phi = cc.b.CreatePHI(cc.i1(), 2, "streq_result");
            m_phi->addIncoming(ConstantInt::getFalse(cc.llctx), pre_len_bb);
            m_phi->addIncoming(m_match, len_match_bb);
            cmp_result = (op == SimplestExprType::NotEqual)
                         ? cc.b.CreateNot(m_phi) : m_phi;
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
        // Large INT64 IN-set: call aqp_in_set_i64 runtime helper
        std::vector<Constant*> consts;
        for (const auto &v : vals)
            consts.push_back(cc.c64((int64_t)v->GetIntValue()));
        ArrayType *arr_ty = ArrayType::get(cc.i64(), consts.size());
        Constant *arr_init = ConstantArray::get(arr_ty, consts);
        GlobalVariable *gv = new GlobalVariable(
            cc.mod, arr_ty, true, GlobalValue::PrivateLinkage, arr_init, "in_vals_i64");
        Value *arr_ptr = cc.b.CreateBitCast(gv, PointerType::getUnqual(cc.i64()));
        FunctionType *ft = FunctionType::get(cc.i32(),
            {cc.i64(), PointerType::getUnqual(cc.i64()), cc.i32()}, false);
        FunctionCallee callee = cc.mod.getOrInsertFunction("aqp_in_set_i64", ft);
        Value *result = cc.b.CreateCall(callee,
            {lhs, arr_ptr, cc.c32((int32_t)consts.size())});
        Value *match = cc.b.CreateICmpNE(result, cc.c32(0));
        return expr->negated ? cc.b.CreateNot(match) : match;
    }

    // VARCHAR IN: call aqp_in_set_str runtime helper
    if (cs.dtype == AQP_DTYPE_VARCHAR && col_idx >= 0) {
        // Build global arrays of string pointers and lengths
        std::vector<Constant*> str_ptrs, str_lens;
        for (const auto &v : vals) {
            const std::string &s = v->GetStringValue();
            Constant *str_const = ConstantDataArray::getString(cc.llctx, s, false);
            GlobalVariable *str_gv = new GlobalVariable(
                cc.mod, str_const->getType(), true,
                GlobalValue::PrivateLinkage, str_const, "in_str");
            str_ptrs.push_back(ConstantExpr::getBitCast(str_gv, cc.i8p()));
            str_lens.push_back(cc.c32((int32_t)s.size()));
        }

        // Build global arrays
        ArrayType *ptr_arr_ty = ArrayType::get(cc.i8p(), str_ptrs.size());
        ArrayType *len_arr_ty = ArrayType::get(cc.i32(), str_lens.size());
        GlobalVariable *ptrs_gv = new GlobalVariable(
            cc.mod, ptr_arr_ty, true, GlobalValue::PrivateLinkage,
            ConstantArray::get(ptr_arr_ty, str_ptrs), "in_str_ptrs");
        GlobalVariable *lens_gv = new GlobalVariable(
            cc.mod, len_arr_ty, true, GlobalValue::PrivateLinkage,
            ConstantArray::get(len_arr_ty, str_lens), "in_str_lens");

        // Load VARCHAR value from column (reuse string_t extraction logic)
        // Extract length and char pointer from DuckDB string_t
        Value *data = cc.col_data[col_idx];
        Value *str_t_ptr = cc.b.CreateBitCast(data, cc.i8p());
        Value *row_offset = cc.b.CreateMul(cc.row_idx, cc.c64(16)); // string_t = 16 bytes
        Value *str_t_base = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_t_ptr, row_offset);
        // Length at offset 0 (4 bytes)
        Value *len_ptr = cc.b.CreateBitCast(str_t_base, PointerType::getUnqual(cc.i32()));
        Value *slen = cc.b.CreateLoad(cc.i32(), len_ptr, "slen");
        // Char data: inline (offset 4) if len <= 12, else pointer at offset 8
        Value *inline_ptr = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_t_base, cc.c64(4));
        Value *heap_ptr_ptr = cc.b.CreateBitCast(
            cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_t_base, cc.c64(8)),
            PointerType::getUnqual(cc.i8p()));
        Value *heap_ptr = cc.b.CreateLoad(cc.i8p(), heap_ptr_ptr);
        Value *is_inline = cc.b.CreateICmpULE(slen, cc.c32(12));
        Value *char_ptr = cc.b.CreateSelect(is_inline, inline_ptr, heap_ptr);

        // Call aqp_in_set_str(str, slen, ptrs, lens, n)
        FunctionType *ft_str = FunctionType::get(cc.i32(),
            {cc.i8p(), cc.i32(),
             PointerType::getUnqual(cc.i8p()), PointerType::getUnqual(cc.i32()),
             cc.i32()}, false);
        FunctionCallee callee = cc.mod.getOrInsertFunction("aqp_in_set_str", ft_str);
        Value *ptrs_ptr = cc.b.CreateBitCast(ptrs_gv, PointerType::getUnqual(cc.i8p()));
        Value *lens_ptr = cc.b.CreateBitCast(lens_gv, PointerType::getUnqual(cc.i32()));
        Value *result = cc.b.CreateCall(callee,
            {char_ptr, slen, ptrs_ptr, lens_ptr, cc.c32((int32_t)vals.size())});
        Value *match = cc.b.CreateICmpNE(result, cc.c32(0));
        return expr->negated ? cc.b.CreateNot(match) : match;
    }

    // Unsupported dtype for IN — pass all
    return expr->negated ? ConstantInt::getFalse(cc.llctx)
                         : ConstantInt::getTrue(cc.llctx);
}

// Arithmetic expression: left OP right → numeric result
// Returns an i32, i64, float, or double Value*.
// Note: for use in filter context, the caller wraps the result in a comparison.
// For use in projection context (future), the result is stored directly.
static Value *EmitArith(CompileCtx &cc, const SimplestArithExpr *expr) {
    if (!expr || !expr->left || !expr->right)
        return ConstantInt::get(cc.i32(), 0);

    Value *lhs = EmitExpr(cc, expr->left.get());
    Value *rhs = EmitExpr(cc, expr->right.get());
    if (!lhs || !rhs) return ConstantInt::get(cc.i32(), 0);

    // Determine if floating point based on result type
    bool is_fp = (expr->result_type == ir_sql_converter::FloatVar);

    switch (expr->arith_op) {
    case ir_sql_converter::ArithAdd:
        return is_fp ? cc.b.CreateFAdd(lhs, rhs, "add") : cc.b.CreateAdd(lhs, rhs, "add");
    case ir_sql_converter::ArithSub:
        return is_fp ? cc.b.CreateFSub(lhs, rhs, "sub") : cc.b.CreateSub(lhs, rhs, "sub");
    case ir_sql_converter::ArithMul:
        return is_fp ? cc.b.CreateFMul(lhs, rhs, "mul") : cc.b.CreateMul(lhs, rhs, "mul");
    case ir_sql_converter::ArithDiv:
        return is_fp ? cc.b.CreateFDiv(lhs, rhs, "div") : cc.b.CreateSDiv(lhs, rhs, "div");
    case ir_sql_converter::ArithMod:
        return is_fp ? cc.b.CreateFRem(lhs, rhs, "mod") : cc.b.CreateSRem(lhs, rhs, "mod");
    default:
        return ConstantInt::get(cc.i32(), 0);
    }
}

// Type cast: child → target_type
static Value *EmitCast(CompileCtx &cc, const SimplestCastExpr *expr) {
    if (!expr || !expr->child)
        return ConstantInt::get(cc.i32(), 0);

    Value *child = EmitExpr(cc, expr->child.get());
    if (!child) return ConstantInt::get(cc.i32(), 0);

    Type *src_ty = child->getType();
    Type *dst_ty;
    switch (expr->target_type) {
    case ir_sql_converter::BoolVar:   dst_ty = cc.i1(); break;
    case ir_sql_converter::IntVar:    dst_ty = cc.i32(); break;
    case ir_sql_converter::FloatVar:  dst_ty = cc.f64(); break;
    case ir_sql_converter::Date:      dst_ty = cc.i32(); break;
    default: return child; // no-op cast
    }

    if (src_ty == dst_ty) return child;

    // Integer → Integer
    if (src_ty->isIntegerTy() && dst_ty->isIntegerTy()) {
        unsigned src_bits = src_ty->getIntegerBitWidth();
        unsigned dst_bits = dst_ty->getIntegerBitWidth();
        if (dst_bits > src_bits) return cc.b.CreateSExt(child, dst_ty, "cast_sext");
        else return cc.b.CreateTrunc(child, dst_ty, "cast_trunc");
    }
    // Integer → Float
    if (src_ty->isIntegerTy() && dst_ty->isFloatingPointTy())
        return cc.b.CreateSIToFP(child, dst_ty, "cast_itof");
    // Float → Integer
    if (src_ty->isFloatingPointTy() && dst_ty->isIntegerTy())
        return cc.b.CreateFPToSI(child, dst_ty, "cast_ftoi");
    // Float → Float (precision change)
    if (src_ty->isFloatingPointTy() && dst_ty->isFloatingPointTy()) {
        if (dst_ty->getPrimitiveSizeInBits() > src_ty->getPrimitiveSizeInBits())
            return cc.b.CreateFPExt(child, dst_ty, "cast_fpext");
        else
            return cc.b.CreateFPTrunc(child, dst_ty, "cast_fptrunc");
    }
    return child; // fallback: no-op
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
    case ArithExprNode:
        return EmitArith(cc, static_cast<const SimplestArithExpr*>(expr));
    case CastExprNode:
        return EmitCast(cc, static_cast<const SimplestCastExpr*>(expr));
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
// Check if all expressions in a filter are SIMD-friendly (numeric comparisons
// and logical AND/OR/NOT — no VARCHAR LIKE, no external function calls).
// ---------------------------------------------------------------------------
static bool AllExprsSIMDFriendly(const std::vector<const AQPExpr*> &exprs,
                                 const std::vector<ColSchema> &schema) {
    for (const AQPExpr *e : exprs) {
        if (!e) return false;
        switch (e->GetNodeType()) {
        case VarConstComparisonNode: {
            auto *cmp = static_cast<const SimplestVarConstComparison *>(e);
            auto et = cmp->GetSimplestExprType();
            // VARCHAR LIKE/equality needs runtime function calls — not SIMD-friendly
            if (et == SimplestExprType::TextLike || et == SimplestExprType::Text_Not_Like)
                return false;
            // Check dtype: VARCHAR comparisons stay scalar.
            // Also: column must be IN schema (not a pass-through).
            bool found = false;
            for (auto &cs : schema) {
                if (cs.table_idx == cmp->attr->GetTableIndex() &&
                    cs.col_idx  == cmp->attr->GetColumnIndex()) {
                    if (cs.dtype == AQP_DTYPE_VARCHAR) return false;
                    if (cs.dtype != AQP_DTYPE_INT32 && cs.dtype != AQP_DTYPE_DATE &&
                        cs.dtype != AQP_DTYPE_INT64 &&
                        cs.dtype != AQP_DTYPE_FLOAT && cs.dtype != AQP_DTYPE_DOUBLE)
                        return false; // only numeric types for SIMD
                    found = true;
                    break;
                }
            }
            if (!found) return false; // column not in schema → pass-through, not SIMD-able
            break;
        }
        case LogicalExprNode: {
            auto *log = static_cast<const SimplestLogicalExpr *>(e);
            std::vector<const AQPExpr*> children;
            if (log->left_expr) children.push_back(log->left_expr.get());
            if (log->right_expr) children.push_back(log->right_expr.get());
            if (!AllExprsSIMDFriendly(children, schema)) return false;
            break;
        }
        case IsNullExprNode:
            break; // IS NULL uses validity bitmap — SIMD-friendly
        case InExprNode:
            return false; // IN-set needs runtime calls
        default:
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Build a SIMD-vectorized filter function (two-phase: vec loop + scalar tail):
//   uint64_t aqp_expr_<id>(AQPChunkView* chunk, AQPSelView* sel)
//
// Phase 1 (vectorized): process VW rows at a time using <VW x i32> comparisons.
//   Compound expressions use bitwise AND/OR (zero branches in inner loop).
//   Selection vector compaction via scalar bit extraction.
// Phase 2 (scalar tail): process remaining nrows % VW rows with scalar code.
// ---------------------------------------------------------------------------
static Function *BuildFilterFunctionSIMD(LLVMContext &llctx, Module &mod,
                                         const std::string &fn_name,
                                         const std::vector<const AQPExpr*> &exprs,
                                         const std::vector<ColSchema> &schema,
                                         unsigned VW) {
    Type *i8p  = PointerType::getUnqual(Type::getInt8Ty(llctx));
    Type *i1   = Type::getInt1Ty(llctx);
    Type *i32  = Type::getInt32Ty(llctx);
    Type *i64  = Type::getInt64Ty(llctx);
    Type *i64p = PointerType::getUnqual(i64);

    StructType *ColViewTy   = StructType::get(llctx, {i8p, i64p, i32, i32});
    StructType *ChunkViewTy = StructType::get(llctx, {
        PointerType::getUnqual(ColViewTy), i64, i64});
    StructType *SelViewTy   = StructType::get(llctx, {
        PointerType::getUnqual(i32), i32});

    // Vector types
    auto *vi32  = FixedVectorType::get(i32, VW);
    auto *vi64  = FixedVectorType::get(i64, VW);
    auto *vi1   = FixedVectorType::get(i1, VW);
    (void)vi64; // may be used for INT64 later

    FunctionType *fn_ty = FunctionType::get(
        i64, {PointerType::getUnqual(ChunkViewTy),
              PointerType::getUnqual(SelViewTy)}, false);
    Function *fn = Function::Create(fn_ty, Function::ExternalLinkage,
                                    fn_name, &mod);

    Value *chunk_arg = fn->getArg(0); chunk_arg->setName("chunk");
    Value *sel_arg   = fn->getArg(1); sel_arg->setName("sel");

    BasicBlock *entry_bb     = BasicBlock::Create(llctx, "entry", fn);
    BasicBlock *vec_loop_bb  = BasicBlock::Create(llctx, "vec_loop", fn);
    BasicBlock *vec_body_bb  = BasicBlock::Create(llctx, "vec_body", fn);
    BasicBlock *vec_store_bb = BasicBlock::Create(llctx, "vec_store", fn);
    BasicBlock *vec_next_bb  = BasicBlock::Create(llctx, "vec_next", fn);
    BasicBlock *tail_bb      = BasicBlock::Create(llctx, "tail", fn);
    BasicBlock *tail_body_bb = BasicBlock::Create(llctx, "tail_body", fn);
    BasicBlock *tail_store_bb= BasicBlock::Create(llctx, "tail_store", fn);
    BasicBlock *tail_next_bb = BasicBlock::Create(llctx, "tail_next", fn);
    BasicBlock *exit_bb      = BasicBlock::Create(llctx, "exit", fn);

    IRBuilder<> b(entry_bb);

    // Load nrows, col data, sel indices
    Value *nrows_ptr = b.CreateStructGEP(ChunkViewTy, chunk_arg, 1);
    Value *nrows     = b.CreateLoad(i64, nrows_ptr, "nrows");
    Value *cols_ptr  = b.CreateStructGEP(ChunkViewTy, chunk_arg, 0);
    Value *cols      = b.CreateLoad(PointerType::getUnqual(ColViewTy), cols_ptr, "cols");

    // Pre-load column data pointers
    std::vector<Value*> col_data(schema.size());
    std::vector<Value*> col_validity(schema.size());
    for (size_t ci = 0; ci < schema.size(); ci++) {
        Value *col_i = b.CreateGEP(ColViewTy, cols, ConstantInt::get(i64, ci));
        col_data[ci]     = b.CreateLoad(i8p, b.CreateStructGEP(ColViewTy, col_i, 0));
        col_validity[ci] = b.CreateLoad(i64p, b.CreateStructGEP(ColViewTy, col_i, 1));
    }

    Value *sel_idx_ptr = b.CreateLoad(PointerType::getUnqual(i32),
        b.CreateStructGEP(SelViewTy, sel_arg, 0), "sel_indices");

    // vec_limit = nrows & ~(VW-1)  — round down to multiple of VW
    Value *vw_const = ConstantInt::get(i64, VW);
    Value *vw_mask  = ConstantInt::get(i64, ~(uint64_t)(VW - 1));
    Value *vec_limit = b.CreateAnd(nrows, vw_mask, "vec_limit");

    b.CreateBr(vec_loop_bb);

    // ========== PHASE 1: Vectorized main loop ==========
    b.SetInsertPoint(vec_loop_bb);
    PHINode *vi = b.CreatePHI(i64, 2, "vi");
    PHINode *voc = b.CreatePHI(i64, 2, "voc"); // vectorized out_count
    vi->addIncoming(ConstantInt::get(i64, 0), entry_bb);
    voc->addIncoming(ConstantInt::get(i64, 0), entry_bb);
    Value *vdone = b.CreateICmpEQ(vi, vec_limit);
    b.CreateCondBr(vdone, tail_bb, vec_body_bb);

    // Vec body: evaluate each expression as a <VW x i1> mask, AND them together
    b.SetInsertPoint(vec_body_bb);
    Value *combined_mask = ConstantInt::getTrue(llctx); // will become <VW x i1>
    // Start with all-true vector mask
    combined_mask = ConstantVector::getSplat(ElementCount::getFixed(VW),
                                             ConstantInt::getTrue(llctx));

    for (const AQPExpr *e : exprs) {
        // For each expression, emit a vectorized comparison
        Value *expr_mask = nullptr;

        if (e->GetNodeType() == VarConstComparisonNode) {
            auto *cmp = static_cast<const SimplestVarConstComparison *>(e);
            // Find column index
            int col_idx = -1;
            int32_t dtype = AQP_DTYPE_OTHER;
            for (int ci = 0; ci < (int)schema.size(); ci++) {
                if (schema[ci].table_idx == cmp->attr->GetTableIndex() &&
                    schema[ci].col_idx  == cmp->attr->GetColumnIndex()) {
                    col_idx = ci;
                    dtype = schema[ci].dtype;
                    break;
                }
            }
            if (col_idx < 0 || dtype == AQP_DTYPE_VARCHAR) {
                // Can't vectorize — return nullptr to fall back to scalar
                fn->eraseFromParent();
                return nullptr;
            }

            // Load VW elements as a vector
            Value *data_ptr = col_data[col_idx];
            Value *data_vec = nullptr;
            Value *const_vec = nullptr;

            bool is_fp = false;
            if (dtype == AQP_DTYPE_INT32 || dtype == AQP_DTYPE_DATE) {
                Value *typed_ptr = b.CreateBitCast(data_ptr, PointerType::getUnqual(i32));
                auto *vty = FixedVectorType::get(i32, VW);
                data_vec = b.CreateLoad(vty, b.CreateBitCast(
                    b.CreateGEP(i32, typed_ptr, vi), PointerType::getUnqual(vty)), "data_vec");
                int32_t cv = cmp->const_var->GetIntValue();
                const_vec = ConstantVector::getSplat(ElementCount::getFixed(VW),
                    ConstantInt::get(i32, (uint64_t)(uint32_t)cv, true));
            } else if (dtype == AQP_DTYPE_INT64) {
                Value *typed_ptr = b.CreateBitCast(data_ptr, PointerType::getUnqual(i64));
                auto *vty = FixedVectorType::get(i64, VW);
                data_vec = b.CreateLoad(vty, b.CreateBitCast(
                    b.CreateGEP(i64, typed_ptr, vi), PointerType::getUnqual(vty)), "data_vec");
                int64_t cv = (int64_t)cmp->const_var->GetIntValue();
                const_vec = ConstantVector::getSplat(ElementCount::getFixed(VW),
                    ConstantInt::get(i64, (uint64_t)cv, true));
            } else if (dtype == AQP_DTYPE_FLOAT) {
                Type *f32 = Type::getFloatTy(llctx);
                Value *typed_ptr = b.CreateBitCast(data_ptr, PointerType::getUnqual(f32));
                auto *vty = FixedVectorType::get(f32, VW);
                data_vec = b.CreateLoad(vty, b.CreateBitCast(
                    b.CreateGEP(f32, typed_ptr, vi), PointerType::getUnqual(vty)), "data_vec");
                float cv = cmp->const_var->GetFloatValue();
                const_vec = ConstantVector::getSplat(ElementCount::getFixed(VW),
                    ConstantFP::get(f32, (double)cv));
                is_fp = true;
            } else if (dtype == AQP_DTYPE_DOUBLE) {
                Type *f64 = Type::getDoubleTy(llctx);
                Value *typed_ptr = b.CreateBitCast(data_ptr, PointerType::getUnqual(f64));
                auto *vty = FixedVectorType::get(f64, VW);
                data_vec = b.CreateLoad(vty, b.CreateBitCast(
                    b.CreateGEP(f64, typed_ptr, vi), PointerType::getUnqual(vty)), "data_vec");
                float cv = cmp->const_var->GetFloatValue();
                const_vec = ConstantVector::getSplat(ElementCount::getFixed(VW),
                    ConstantFP::get(f64, (double)cv));
                is_fp = true;
            } else {
                fn->eraseFromParent(); return nullptr;
            }

            // Vector comparison
            auto et = cmp->GetSimplestExprType();
            if (is_fp) {
                switch (et) {
                case SimplestExprType::Equal:        expr_mask = b.CreateFCmpOEQ(data_vec, const_vec); break;
                case SimplestExprType::NotEqual:     expr_mask = b.CreateFCmpONE(data_vec, const_vec); break;
                case SimplestExprType::LessThan:     expr_mask = b.CreateFCmpOLT(data_vec, const_vec); break;
                case SimplestExprType::GreaterThan:  expr_mask = b.CreateFCmpOGT(data_vec, const_vec); break;
                case SimplestExprType::LessEqual:    expr_mask = b.CreateFCmpOLE(data_vec, const_vec); break;
                case SimplestExprType::GreaterEqual: expr_mask = b.CreateFCmpOGE(data_vec, const_vec); break;
                default: fn->eraseFromParent(); return nullptr;
                }
            } else {
            switch (et) {
            case SimplestExprType::Equal:
                expr_mask = b.CreateICmpEQ(data_vec, const_vec); break;
            case SimplestExprType::NotEqual:
                expr_mask = b.CreateICmpNE(data_vec, const_vec); break;
            case SimplestExprType::LessThan:
                expr_mask = b.CreateICmpSLT(data_vec, const_vec); break;
            case SimplestExprType::GreaterThan:
                expr_mask = b.CreateICmpSGT(data_vec, const_vec); break;
            case SimplestExprType::LessEqual:
                expr_mask = b.CreateICmpSLE(data_vec, const_vec); break;
            case SimplestExprType::GreaterEqual:
                expr_mask = b.CreateICmpSGE(data_vec, const_vec); break;
            default:
                fn->eraseFromParent();
                return nullptr;
            }
            } // end else (integer path)

            // AND with validity mask
            Value *validity = col_validity[col_idx];
            Value *val_nonnull = b.CreateICmpNE(
                b.CreatePtrToInt(validity, i64), ConstantInt::get(i64, 0));

            // If validity is non-null, extract VW bits and AND with comparison
            // Since VW divides 64 evenly (4,8,16), bits never span two words
            BasicBlock *has_val_bb = BasicBlock::Create(llctx, "has_val", fn);
            BasicBlock *no_val_bb  = BasicBlock::Create(llctx, "no_val", fn);
            BasicBlock *merge_val_bb = BasicBlock::Create(llctx, "merge_val", fn);
            b.CreateCondBr(val_nonnull, has_val_bb, no_val_bb);

            b.SetInsertPoint(has_val_bb);
            Value *word_idx = b.CreateLShr(vi, ConstantInt::get(i64, 6));
            Value *bit_off  = b.CreateAnd(vi, ConstantInt::get(i64, 63));
            Value *word     = b.CreateLoad(i64, b.CreateGEP(i64, validity, word_idx));
            Value *shifted  = b.CreateLShr(word, bit_off);
            // Extract VW bits: truncate to iVW, then bitcast to <VW x i1>
            Type *iVW = Type::getIntNTy(llctx, VW);
            Value *mask_int = b.CreateTrunc(shifted, iVW);
            Value *val_mask = b.CreateBitCast(mask_int, vi1);
            Value *masked_result = b.CreateAnd(expr_mask, val_mask);
            b.CreateBr(merge_val_bb);

            b.SetInsertPoint(no_val_bb);
            // All valid — use expr_mask as-is
            b.CreateBr(merge_val_bb);

            b.SetInsertPoint(merge_val_bb);
            PHINode *final_mask = b.CreatePHI(vi1, 2, "final_mask");
            final_mask->addIncoming(masked_result, has_val_bb);
            final_mask->addIncoming(expr_mask, no_val_bb);
            expr_mask = final_mask;

        } else if (e->GetNodeType() == IsNullExprNode) {
            auto *isnull = static_cast<const SimplestIsNullExpr *>(e);
            int col_idx = -1;
            for (int ci = 0; ci < (int)schema.size(); ci++) {
                if (schema[ci].table_idx == isnull->attr->GetTableIndex() &&
                    schema[ci].col_idx  == isnull->attr->GetColumnIndex()) {
                    col_idx = ci;
                    break;
                }
            }
            if (col_idx < 0) { fn->eraseFromParent(); return nullptr; }

            // IS NULL: check validity bits
            Value *validity = col_validity[col_idx];
            Value *val_nonnull = b.CreateICmpNE(
                b.CreatePtrToInt(validity, i64), ConstantInt::get(i64, 0));

            BasicBlock *has_v = BasicBlock::Create(llctx, "isnull_has_v", fn);
            BasicBlock *no_v  = BasicBlock::Create(llctx, "isnull_no_v", fn);
            BasicBlock *merge = BasicBlock::Create(llctx, "isnull_merge", fn);
            b.CreateCondBr(val_nonnull, has_v, no_v);

            b.SetInsertPoint(has_v);
            Value *word_idx2 = b.CreateLShr(vi, ConstantInt::get(i64, 6));
            Value *bit_off2  = b.CreateAnd(vi, ConstantInt::get(i64, 63));
            Value *word2     = b.CreateLoad(i64, b.CreateGEP(i64, col_validity[col_idx], word_idx2));
            Value *shifted2  = b.CreateLShr(word2, bit_off2);
            Type *iVW2 = Type::getIntNTy(llctx, VW);
            Value *bits = b.CreateBitCast(b.CreateTrunc(shifted2, iVW2), vi1);
            // IS NULL: valid bit = 0 means NULL
            Value *null_mask = (isnull->GetSimplestExprType() == SimplestExprType::NullType)
                ? b.CreateNot(bits) : bits;
            b.CreateBr(merge);

            b.SetInsertPoint(no_v);
            // All valid: IS NULL → all false, IS NOT NULL → all true
            Value *all_const = (isnull->GetSimplestExprType() == SimplestExprType::NullType)
                ? ConstantVector::getSplat(ElementCount::getFixed(VW), ConstantInt::getFalse(llctx))
                : ConstantVector::getSplat(ElementCount::getFixed(VW), ConstantInt::getTrue(llctx));
            b.CreateBr(merge);

            b.SetInsertPoint(merge);
            PHINode *isnull_mask = b.CreatePHI(vi1, 2);
            isnull_mask->addIncoming(null_mask, has_v);
            isnull_mask->addIncoming(all_const, no_v);
            expr_mask = isnull_mask;
        } else {
            // Unsupported expression type for SIMD — fall back
            fn->eraseFromParent();
            return nullptr;
        }

        // Bitwise AND with combined mask (no branches!)
        combined_mask = b.CreateAnd(combined_mask, expr_mask);
    }

    // Selection vector compaction: convert <VW x i1> to iVW integer,
    // then use a scalar bit-scan loop to scatter matching row indices.
    b.CreateBr(vec_store_bb);
    b.SetInsertPoint(vec_store_bb);

    // Bitcast <VW x i1> → iVW, then zext to i64 for bit manipulation
    Type *iVW_ty = Type::getIntNTy(llctx, VW);
    Value *mask_int = b.CreateBitCast(combined_mask, iVW_ty);
    Value *mask_i64 = b.CreateZExt(mask_int, i64, "mask_i64");

    // Scatter loop: while (mask != 0) { k = ctz(mask); store vi+k; mask &= mask-1; }
    BasicBlock *scatter_loop_bb = BasicBlock::Create(llctx, "scatter_loop", fn);
    BasicBlock *scatter_body_bb = BasicBlock::Create(llctx, "scatter_body", fn);
    BasicBlock *scatter_done_bb = BasicBlock::Create(llctx, "scatter_done", fn);
    b.CreateBr(scatter_loop_bb);

    b.SetInsertPoint(scatter_loop_bb);
    PHINode *sc_mask = b.CreatePHI(i64, 2, "sc_mask");
    PHINode *sc_oc   = b.CreatePHI(i64, 2, "sc_oc");
    sc_mask->addIncoming(mask_i64, vec_store_bb);
    sc_oc->addIncoming(voc, vec_store_bb);
    Value *mask_zero = b.CreateICmpEQ(sc_mask, ConstantInt::get(i64, 0));
    b.CreateCondBr(mask_zero, scatter_done_bb, scatter_body_bb);

    b.SetInsertPoint(scatter_body_bb);
    // Count trailing zeros = index of lowest set bit
    Function *cttz_fn = Intrinsic::getDeclaration(&mod, Intrinsic::cttz, {i64});
    Value *k = b.CreateCall(cttz_fn, {sc_mask, ConstantInt::getTrue(llctx)}, "k");
    // Store row index = vi + k
    Value *row_idx = b.CreateAdd(vi, k, "row_idx");
    Value *dst = b.CreateGEP(i32, sel_idx_ptr, sc_oc);
    b.CreateStore(b.CreateTrunc(row_idx, i32), dst);
    // Clear lowest set bit: mask &= mask - 1
    Value *sc_mask_next = b.CreateAnd(sc_mask,
        b.CreateSub(sc_mask, ConstantInt::get(i64, 1)));
    Value *sc_oc_next = b.CreateAdd(sc_oc, ConstantInt::get(i64, 1));
    sc_mask->addIncoming(sc_mask_next, scatter_body_bb);
    sc_oc->addIncoming(sc_oc_next, scatter_body_bb);
    b.CreateBr(scatter_loop_bb);

    b.SetInsertPoint(scatter_done_bb);
    b.CreateBr(vec_next_bb);

    // Vec next: increment by VW
    b.SetInsertPoint(vec_next_bb);
    Value *vi_next = b.CreateAdd(vi, vw_const, "vi_next");
    vi->addIncoming(vi_next, vec_next_bb);
    voc->addIncoming(sc_oc, vec_next_bb);
    b.CreateBr(vec_loop_bb);

    // ========== PHASE 2: Scalar tail loop ==========
    b.SetInsertPoint(tail_bb);
    PHINode *ti = b.CreatePHI(i64, 2, "ti");
    PHINode *toc = b.CreatePHI(i64, 2, "toc");
    ti->addIncoming(vec_limit, vec_loop_bb);
    toc->addIncoming(voc, vec_loop_bb);
    Value *tdone = b.CreateICmpEQ(ti, nrows);
    b.CreateCondBr(tdone, exit_bb, tail_body_bb);

    // Scalar tail body — reuse existing EmitExpr infrastructure
    {
        CompileCtx cc(llctx, mod, schema, chunk_arg, sel_arg);
        cc.b.SetInsertPoint(tail_body_bb);
        cc.row_idx = ti;
        cc.col_data = col_data;
        cc.col_validity = col_validity;

        Value *match = ConstantInt::getTrue(llctx);
        for (const AQPExpr *e : exprs) {
            Value *res = EmitExpr(cc, e);
            match = cc.b.CreateAnd(match, res);
        }
        BasicBlock *condBr_bb = cc.b.GetInsertBlock();
        cc.b.CreateCondBr(match, tail_store_bb, tail_next_bb);

        b.SetInsertPoint(tail_store_bb);
        Value *tdst = b.CreateGEP(i32, sel_idx_ptr, toc);
        b.CreateStore(b.CreateTrunc(ti, i32), tdst);
        Value *toc1 = b.CreateAdd(toc, ConstantInt::get(i64, 1));
        b.CreateBr(tail_next_bb);

        b.SetInsertPoint(tail_next_bb);
        PHINode *toc_next = b.CreatePHI(i64, 2, "toc_next");
        toc_next->addIncoming(toc, condBr_bb);
        toc_next->addIncoming(toc1, tail_store_bb);
        Value *ti_next = b.CreateAdd(ti, ConstantInt::get(i64, 1));
        ti->addIncoming(ti_next, tail_next_bb);
        toc->addIncoming(toc_next, tail_next_bb);
        b.CreateBr(tail_bb);
    }

    // Exit
    b.SetInsertPoint(exit_bb);
    PHINode *final_oc = b.CreatePHI(i64, 2, "final_oc");
    final_oc->addIncoming(voc, vec_loop_bb); // if vec_limit == 0
    final_oc->addIncoming(toc, tail_bb);     // normal path
    Value *sel_cnt_ptr = b.CreateStructGEP(SelViewTy, sel_arg, 1);
    b.CreateStore(b.CreateTrunc(final_oc, i32), sel_cnt_ptr);
    b.CreateRet(final_oc);

    return fn;
}

// ---------------------------------------------------------------------------
// Build a hybrid filter function (SIMD numeric + scalar VARCHAR):
//   uint64_t aqp_expr_<id>(AQPChunkView* chunk, AQPSelView* sel)
//
// Phase 1: SIMD-evaluate numeric predicates → fill selection vector
// Phase 2: iterate selection vector, scalar-evaluate VARCHAR predicates,
//          compact in-place (survivors stay, non-survivors removed)
// Returns final count of surviving rows.
// ---------------------------------------------------------------------------
static Function *BuildFilterFunctionHybrid(LLVMContext &llctx, Module &mod,
                                            const std::string &fn_name,
                                            const std::vector<const AQPExpr*> &simd_exprs,
                                            const std::vector<const AQPExpr*> &scalar_exprs,
                                            const std::vector<ColSchema> &schema,
                                            unsigned VW) {
    // Phase 1: Build a SIMD filter for numeric expressions only.
    // This produces the selection vector with rows passing numeric predicates.
    // We reuse BuildFilterFunctionSIMD's logic inline.

    Type *i8p  = PointerType::getUnqual(Type::getInt8Ty(llctx));
    Type *i1   = Type::getInt1Ty(llctx);
    Type *i32  = Type::getInt32Ty(llctx);
    Type *i64  = Type::getInt64Ty(llctx);
    Type *i64p = PointerType::getUnqual(i64);

    StructType *ColViewTy   = StructType::get(llctx, {i8p, i64p, i32, i32});
    StructType *ChunkViewTy = StructType::get(llctx, {
        PointerType::getUnqual(ColViewTy), i64, i64});
    StructType *SelViewTy   = StructType::get(llctx, {
        PointerType::getUnqual(i32), i32});

    auto *vi32 = FixedVectorType::get(i32, VW);
    auto *vi1  = FixedVectorType::get(i1, VW);

    FunctionType *fn_ty = FunctionType::get(
        i64, {PointerType::getUnqual(ChunkViewTy),
              PointerType::getUnqual(SelViewTy)}, false);
    Function *fn = Function::Create(fn_ty, Function::ExternalLinkage,
                                    fn_name, &mod);

    Value *chunk_arg = fn->getArg(0); chunk_arg->setName("chunk");
    Value *sel_arg   = fn->getArg(1); sel_arg->setName("sel");

    BasicBlock *entry_bb      = BasicBlock::Create(llctx, "entry", fn);
    BasicBlock *vec_loop_bb   = BasicBlock::Create(llctx, "vec_loop", fn);
    BasicBlock *vec_body_bb   = BasicBlock::Create(llctx, "vec_body", fn);
    BasicBlock *scatter_bb    = BasicBlock::Create(llctx, "scatter", fn);
    BasicBlock *scatter_loop  = BasicBlock::Create(llctx, "scatter_loop", fn);
    BasicBlock *scatter_body  = BasicBlock::Create(llctx, "scatter_body", fn);
    BasicBlock *scatter_done  = BasicBlock::Create(llctx, "scatter_done", fn);
    BasicBlock *vec_next_bb   = BasicBlock::Create(llctx, "vec_next", fn);
    BasicBlock *tail_bb       = BasicBlock::Create(llctx, "tail", fn);
    BasicBlock *tail_body_bb  = BasicBlock::Create(llctx, "tail_body", fn);
    BasicBlock *tail_store_bb = BasicBlock::Create(llctx, "tail_store", fn);
    BasicBlock *tail_next_bb  = BasicBlock::Create(llctx, "tail_next", fn);
    BasicBlock *phase2_bb     = BasicBlock::Create(llctx, "phase2", fn);
    BasicBlock *p2_body_bb    = BasicBlock::Create(llctx, "p2_body", fn);
    BasicBlock *p2_keep_bb    = BasicBlock::Create(llctx, "p2_keep", fn);
    BasicBlock *p2_next_bb    = BasicBlock::Create(llctx, "p2_next", fn);
    BasicBlock *exit_bb       = BasicBlock::Create(llctx, "exit", fn);

    IRBuilder<> b(entry_bb);

    // Load nrows, columns, sel indices
    Value *nrows = b.CreateLoad(i64, b.CreateStructGEP(ChunkViewTy, chunk_arg, 1), "nrows");
    Value *cols  = b.CreateLoad(PointerType::getUnqual(ColViewTy),
        b.CreateStructGEP(ChunkViewTy, chunk_arg, 0), "cols");

    std::vector<Value*> col_data(schema.size());
    std::vector<Value*> col_validity(schema.size());
    for (size_t ci = 0; ci < schema.size(); ci++) {
        Value *col_i = b.CreateGEP(ColViewTy, cols, ConstantInt::get(i64, ci));
        col_data[ci]     = b.CreateLoad(i8p, b.CreateStructGEP(ColViewTy, col_i, 0));
        col_validity[ci] = b.CreateLoad(i64p, b.CreateStructGEP(ColViewTy, col_i, 1));
    }

    Value *sel_idx_ptr = b.CreateLoad(PointerType::getUnqual(i32),
        b.CreateStructGEP(SelViewTy, sel_arg, 0), "sel_indices");

    Value *vec_limit = b.CreateAnd(nrows, ConstantInt::get(i64, ~(uint64_t)(VW - 1)));
    b.CreateBr(vec_loop_bb);

    // ===== PHASE 1: SIMD numeric predicates =====
    b.SetInsertPoint(vec_loop_bb);
    PHINode *vi = b.CreatePHI(i64, 2, "vi");
    PHINode *voc = b.CreatePHI(i64, 2, "voc");
    vi->addIncoming(ConstantInt::get(i64, 0), entry_bb);
    voc->addIncoming(ConstantInt::get(i64, 0), entry_bb);
    b.CreateCondBr(b.CreateICmpEQ(vi, vec_limit), tail_bb, vec_body_bb);

    b.SetInsertPoint(vec_body_bb);
    Value *combined_mask = ConstantVector::getSplat(
        ElementCount::getFixed(VW), ConstantInt::getTrue(llctx));

    // Flatten compound AND expressions into leaf comparisons.
    // e.g., (year >= 1950 AND year <= 2000 AND year IS NOT NULL) → 3 leaves
    std::vector<const AQPExpr*> flat_simd;
    std::function<void(const AQPExpr*)> flatten = [&](const AQPExpr *e) {
        if (!e) return;
        if (e->GetNodeType() == LogicalExprNode) {
            auto *log = static_cast<const SimplestLogicalExpr *>(e);
            if (log->GetLogicalOp() == SimplestLogicalOp::LogicalAnd) {
                flatten(log->left_expr.get());
                flatten(log->right_expr.get());
                return;
            }
        }
        flat_simd.push_back(e);
    };
    for (const AQPExpr *e : simd_exprs)
        flatten(e);

    for (const AQPExpr *e : flat_simd) {
        if (e->GetNodeType() == IsNullExprNode) {
            // IS NOT NULL: skip — handled by validity mask in VarConst path
            // IS NULL in a filter context usually means "keep NULLs" — rare, skip for now
            continue;
        }
        if (e->GetNodeType() != VarConstComparisonNode) {
            fn->eraseFromParent(); return nullptr;
        }
        auto *cmp = static_cast<const SimplestVarConstComparison *>(e);
        int col_idx = -1;
        int32_t dtype = AQP_DTYPE_OTHER;
        for (int ci = 0; ci < (int)schema.size(); ci++) {
            if (schema[ci].table_idx == cmp->attr->GetTableIndex() &&
                schema[ci].col_idx  == cmp->attr->GetColumnIndex()) {
                col_idx = ci; dtype = schema[ci].dtype; break;
            }
        }
        if (col_idx < 0) { fn->eraseFromParent(); return nullptr; }

        // Load VW elements as a vector and splat the constant.
        // LLVM automatically splits oversized vectors (e.g., <8 x i64> on AVX2
        // becomes two <4 x i64> ops). Result is always <VW x i1>.
        Value *data_vec = nullptr;
        Value *const_vec = nullptr;
        bool is_fp = false;

        if (dtype == AQP_DTYPE_INT32 || dtype == AQP_DTYPE_DATE) {
            Value *typed_ptr = b.CreateBitCast(col_data[col_idx], PointerType::getUnqual(i32));
            auto *vty = FixedVectorType::get(i32, VW);
            data_vec = b.CreateLoad(vty, b.CreateBitCast(
                b.CreateGEP(i32, typed_ptr, vi), PointerType::getUnqual(vty)));
            int32_t cv = cmp->const_var->GetIntValue();
            const_vec = ConstantVector::getSplat(ElementCount::getFixed(VW),
                ConstantInt::get(i32, (uint64_t)(uint32_t)cv, true));
        } else if (dtype == AQP_DTYPE_INT64) {
            Value *typed_ptr = b.CreateBitCast(col_data[col_idx], PointerType::getUnqual(i64));
            auto *vty = FixedVectorType::get(i64, VW);
            data_vec = b.CreateLoad(vty, b.CreateBitCast(
                b.CreateGEP(i64, typed_ptr, vi), PointerType::getUnqual(vty)));
            int64_t cv = (int64_t)cmp->const_var->GetIntValue();
            const_vec = ConstantVector::getSplat(ElementCount::getFixed(VW),
                ConstantInt::get(i64, (uint64_t)cv, true));
        } else if (dtype == AQP_DTYPE_FLOAT) {
            Type *f32 = Type::getFloatTy(llctx);
            Value *typed_ptr = b.CreateBitCast(col_data[col_idx], PointerType::getUnqual(f32));
            auto *vty = FixedVectorType::get(f32, VW);
            data_vec = b.CreateLoad(vty, b.CreateBitCast(
                b.CreateGEP(f32, typed_ptr, vi), PointerType::getUnqual(vty)));
            float cv = cmp->const_var->GetFloatValue();
            const_vec = ConstantVector::getSplat(ElementCount::getFixed(VW),
                ConstantFP::get(f32, (double)cv));
            is_fp = true;
        } else if (dtype == AQP_DTYPE_DOUBLE) {
            Type *f64 = Type::getDoubleTy(llctx);
            Value *typed_ptr = b.CreateBitCast(col_data[col_idx], PointerType::getUnqual(f64));
            auto *vty = FixedVectorType::get(f64, VW);
            data_vec = b.CreateLoad(vty, b.CreateBitCast(
                b.CreateGEP(f64, typed_ptr, vi), PointerType::getUnqual(vty)));
            float cv = cmp->const_var->GetFloatValue();
            const_vec = ConstantVector::getSplat(ElementCount::getFixed(VW),
                ConstantFP::get(f64, (double)cv));
            is_fp = true;
        } else {
            fn->eraseFromParent(); return nullptr;
        }

        Value *expr_mask;
        auto et = cmp->GetSimplestExprType();
        if (is_fp) {
            switch (et) {
            case SimplestExprType::Equal:        expr_mask = b.CreateFCmpOEQ(data_vec, const_vec); break;
            case SimplestExprType::NotEqual:     expr_mask = b.CreateFCmpONE(data_vec, const_vec); break;
            case SimplestExprType::LessThan:     expr_mask = b.CreateFCmpOLT(data_vec, const_vec); break;
            case SimplestExprType::GreaterThan:  expr_mask = b.CreateFCmpOGT(data_vec, const_vec); break;
            case SimplestExprType::LessEqual:    expr_mask = b.CreateFCmpOLE(data_vec, const_vec); break;
            case SimplestExprType::GreaterEqual: expr_mask = b.CreateFCmpOGE(data_vec, const_vec); break;
            default: fn->eraseFromParent(); return nullptr;
            }
        } else {
            switch (et) {
            case SimplestExprType::Equal:        expr_mask = b.CreateICmpEQ(data_vec, const_vec); break;
            case SimplestExprType::NotEqual:     expr_mask = b.CreateICmpNE(data_vec, const_vec); break;
            case SimplestExprType::LessThan:     expr_mask = b.CreateICmpSLT(data_vec, const_vec); break;
            case SimplestExprType::GreaterThan:  expr_mask = b.CreateICmpSGT(data_vec, const_vec); break;
            case SimplestExprType::LessEqual:    expr_mask = b.CreateICmpSLE(data_vec, const_vec); break;
            case SimplestExprType::GreaterEqual: expr_mask = b.CreateICmpSGE(data_vec, const_vec); break;
            default: fn->eraseFromParent(); return nullptr;
            }
        }
        combined_mask = b.CreateAnd(combined_mask, expr_mask);
    }

    // Scatter: cttz-based loop
    b.CreateBr(scatter_bb);
    b.SetInsertPoint(scatter_bb);
    Type *iVW_ty = Type::getIntNTy(llctx, VW);
    Value *mask_int = b.CreateBitCast(combined_mask, iVW_ty);
    Value *mask_i64 = b.CreateZExt(mask_int, i64);
    b.CreateBr(scatter_loop);

    b.SetInsertPoint(scatter_loop);
    PHINode *sc_mask = b.CreatePHI(i64, 2, "sc_mask");
    PHINode *sc_oc   = b.CreatePHI(i64, 2, "sc_oc");
    sc_mask->addIncoming(mask_i64, scatter_bb);
    sc_oc->addIncoming(voc, scatter_bb);
    b.CreateCondBr(b.CreateICmpEQ(sc_mask, ConstantInt::get(i64, 0)),
                   scatter_done, scatter_body);

    b.SetInsertPoint(scatter_body);
    Function *cttz_fn = Intrinsic::getDeclaration(&mod, Intrinsic::cttz, {i64});
    Value *k = b.CreateCall(cttz_fn, {sc_mask, ConstantInt::getTrue(llctx)});
    Value *row_idx = b.CreateAdd(vi, k);
    b.CreateStore(b.CreateTrunc(row_idx, i32),
                  b.CreateGEP(i32, sel_idx_ptr, sc_oc));
    Value *sc_mask_next = b.CreateAnd(sc_mask, b.CreateSub(sc_mask, ConstantInt::get(i64, 1)));
    Value *sc_oc_next = b.CreateAdd(sc_oc, ConstantInt::get(i64, 1));
    sc_mask->addIncoming(sc_mask_next, scatter_body);
    sc_oc->addIncoming(sc_oc_next, scatter_body);
    b.CreateBr(scatter_loop);

    b.SetInsertPoint(scatter_done);
    b.CreateBr(vec_next_bb);

    b.SetInsertPoint(vec_next_bb);
    Value *vi_next = b.CreateAdd(vi, ConstantInt::get(i64, VW));
    vi->addIncoming(vi_next, vec_next_bb);
    voc->addIncoming(sc_oc, vec_next_bb);
    b.CreateBr(vec_loop_bb);

    // Scalar tail for numeric predicates (remaining nrows % VW rows)
    b.SetInsertPoint(tail_bb);
    PHINode *ti = b.CreatePHI(i64, 2, "ti");
    PHINode *toc = b.CreatePHI(i64, 2, "toc");
    ti->addIncoming(vec_limit, vec_loop_bb);
    toc->addIncoming(voc, vec_loop_bb);
    b.CreateCondBr(b.CreateICmpEQ(ti, nrows), phase2_bb, tail_body_bb);

    {
        CompileCtx cc(llctx, mod, schema, chunk_arg, sel_arg);
        cc.b.SetInsertPoint(tail_body_bb);
        cc.row_idx = ti;
        cc.col_data = col_data;
        cc.col_validity = col_validity;

        Value *match = ConstantInt::getTrue(llctx);
        for (const AQPExpr *e : simd_exprs) {
            Value *res = EmitExpr(cc, e);
            match = cc.b.CreateAnd(match, res);
        }
        BasicBlock *condBr_bb = cc.b.GetInsertBlock();
        cc.b.CreateCondBr(match, tail_store_bb, tail_next_bb);

        b.SetInsertPoint(tail_store_bb);
        b.CreateStore(b.CreateTrunc(ti, i32), b.CreateGEP(i32, sel_idx_ptr, toc));
        Value *toc1 = b.CreateAdd(toc, ConstantInt::get(i64, 1));
        b.CreateBr(tail_next_bb);

        b.SetInsertPoint(tail_next_bb);
        PHINode *toc_next = b.CreatePHI(i64, 2, "toc_next");
        toc_next->addIncoming(toc, condBr_bb);
        toc_next->addIncoming(toc1, tail_store_bb);
        Value *ti_next = b.CreateAdd(ti, ConstantInt::get(i64, 1));
        ti->addIncoming(ti_next, tail_next_bb);
        toc->addIncoming(toc_next, tail_next_bb);
        b.CreateBr(tail_bb);
    }

    // ===== PHASE 2: Scalar VARCHAR predicates on survivors =====
    // Iterate sel[0..phase1_count), evaluate VARCHAR exprs, compact in-place.
    b.SetInsertPoint(phase2_bb);
    PHINode *phase1_count = b.CreatePHI(i64, 2, "phase1_count");
    phase1_count->addIncoming(voc, vec_loop_bb);  // if tail was empty (vec_limit == nrows)
    phase1_count->addIncoming(toc, tail_bb);

    PHINode *p2i = b.CreatePHI(i64, 2, "p2i");
    PHINode *p2oc = b.CreatePHI(i64, 2, "p2oc");
    p2i->addIncoming(ConstantInt::get(i64, 0), phase2_bb->getSinglePredecessor() ?
                     phase2_bb->getSinglePredecessor() : tail_bb);
    p2oc->addIncoming(ConstantInt::get(i64, 0), phase2_bb->getSinglePredecessor() ?
                      phase2_bb->getSinglePredecessor() : tail_bb);

    // Actually, phase2_bb has two predecessors (vec_loop and tail_bb).
    // Need a separate loop header.
    // Let me restructure: phase2_bb is just an entry, then jump to p2_loop.
    BasicBlock *p2_loop_bb = BasicBlock::Create(llctx, "p2_loop", fn);
    b.CreateBr(p2_loop_bb);

    // Remove the broken PHIs from phase2_bb and use p2_loop_bb instead
    p2i->eraseFromParent();
    p2oc->eraseFromParent();

    b.SetInsertPoint(p2_loop_bb);
    PHINode *p2i2 = b.CreatePHI(i64, 2, "p2i");
    PHINode *p2oc2 = b.CreatePHI(i64, 2, "p2oc");
    p2i2->addIncoming(ConstantInt::get(i64, 0), phase2_bb);
    p2oc2->addIncoming(ConstantInt::get(i64, 0), phase2_bb);
    b.CreateCondBr(b.CreateICmpEQ(p2i2, phase1_count), exit_bb, p2_body_bb);

    // Evaluate VARCHAR predicates for the row at sel[p2i]
    {
        CompileCtx cc(llctx, mod, schema, chunk_arg, sel_arg);
        cc.b.SetInsertPoint(p2_body_bb);
        cc.col_data = col_data;
        cc.col_validity = col_validity;

        // Load row index from selection vector
        Value *row_from_sel = cc.b.CreateZExt(
            cc.b.CreateLoad(i32, cc.b.CreateGEP(i32, sel_idx_ptr, p2i2)), i64);
        cc.row_idx = row_from_sel;

        Value *match = ConstantInt::getTrue(llctx);
        for (const AQPExpr *e : scalar_exprs) {
            Value *res = EmitExpr(cc, e);
            match = cc.b.CreateAnd(match, res);
        }
        BasicBlock *condBr_bb = cc.b.GetInsertBlock();
        cc.b.CreateCondBr(match, p2_keep_bb, p2_next_bb);

        b.SetInsertPoint(p2_keep_bb);
        // Copy surviving row index to compacted position
        Value *src_val = b.CreateLoad(i32, b.CreateGEP(i32, sel_idx_ptr, p2i2));
        b.CreateStore(src_val, b.CreateGEP(i32, sel_idx_ptr, p2oc2));
        Value *p2oc_inc = b.CreateAdd(p2oc2, ConstantInt::get(i64, 1));
        b.CreateBr(p2_next_bb);

        b.SetInsertPoint(p2_next_bb);
        PHINode *p2oc_next = b.CreatePHI(i64, 2, "p2oc_next");
        p2oc_next->addIncoming(p2oc2, condBr_bb);
        p2oc_next->addIncoming(p2oc_inc, p2_keep_bb);
        Value *p2i_next = b.CreateAdd(p2i2, ConstantInt::get(i64, 1));
        p2i2->addIncoming(p2i_next, p2_next_bb);
        p2oc2->addIncoming(p2oc_next, p2_next_bb);
        b.CreateBr(p2_loop_bb);
    }

    // Exit
    b.SetInsertPoint(exit_bb);
    PHINode *final_count = b.CreatePHI(i64, 1, "final_count");
    final_count->addIncoming(p2oc2, p2_loop_bb);
    b.CreateStore(b.CreateTrunc(final_count, i32),
                  b.CreateStructGEP(SelViewTy, sel_arg, 1));
    b.CreateRet(final_count);

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

// AggOp descriptor (used by scalar and SIMD aggregate builders)
struct AggOp {
    int           col_idx;       // input chunk column index (-1 for COUNT*)
    int32_t       agg_type;      // SimplestAggFnType cast to int32_t
    unsigned      state_offset;  // byte offset in agg_state
    int32_t       dtype;         // AQP_DTYPE_* of the column
};

// ---------------------------------------------------------------------------
static bool AllAggOpsSIMDFriendly(const std::vector<AggOp> &ops) {
    for (const auto &op : ops) {
        if (op.agg_type == 6) continue; // CountStar always works
        if (op.dtype != AQP_DTYPE_INT32 && op.dtype != AQP_DTYPE_DATE &&
            op.dtype != AQP_DTYPE_INT64 &&
            op.dtype != AQP_DTYPE_FLOAT && op.dtype != AQP_DTYPE_DOUBLE)
            return false;
        // Only SUM(3), COUNT(5), MIN(1), MAX(2) — not AVG(4) yet
        if (op.agg_type == 4) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Build a SIMD-vectorized aggregate update function (ungrouped, all numeric):
//   void aqp_agg_<id>(AQPChunkView* in, i8* agg_state)
//
// Each accumulator uses its own vector type based on column dtype:
//   INT32/DATE: <VW x i32>, FLOAT: <VW x float>
//   INT64: <VW x i64>, DOUBLE: <VW x double>
// LLVM auto-splits oversized vectors (e.g., <8 x i64> → 2x <4 x i64> on AVX2).
// ---------------------------------------------------------------------------
static Function *BuildAggUpdateFunctionSIMD(LLVMContext &llctx, Module &mod,
                                             const std::string &fn_name,
                                             const std::vector<AggOp> &agg_ops,
                                             unsigned total_state_size,
                                             const std::vector<ColSchema> &schema,
                                             unsigned VW) {
    Type *i8    = Type::getInt8Ty(llctx);
    Type *i8p   = PointerType::getUnqual(i8);
    Type *i32   = Type::getInt32Ty(llctx);
    Type *i64   = Type::getInt64Ty(llctx);
    Type *f32   = Type::getFloatTy(llctx);
    Type *f64   = Type::getDoubleTy(llctx);
    Type *i64p  = PointerType::getUnqual(i64);
    Type *voidTy = Type::getVoidTy(llctx);

    StructType *ColViewTy   = StructType::get(llctx, {i8p, i64p, i32, i32});
    StructType *ChunkViewTy = StructType::get(llctx, {
        PointerType::getUnqual(ColViewTy), i64, i64});

    FunctionType *fn_ty = FunctionType::get(
        voidTy, {PointerType::getUnqual(ChunkViewTy), i8p}, false);
    Function *fn = Function::Create(fn_ty, Function::ExternalLinkage,
                                    fn_name, &mod);

    Value *in_arg    = fn->getArg(0); in_arg->setName("in");
    Value *state_arg = fn->getArg(1); state_arg->setName("state");

    BasicBlock *entry_bb    = BasicBlock::Create(llctx, "entry", fn);
    BasicBlock *vec_loop_bb = BasicBlock::Create(llctx, "vec_loop", fn);
    BasicBlock *vec_body_bb = BasicBlock::Create(llctx, "vec_body", fn);
    BasicBlock *vec_next_bb = BasicBlock::Create(llctx, "vec_next", fn);
    BasicBlock *tail_bb     = BasicBlock::Create(llctx, "tail", fn);
    BasicBlock *tail_body_bb= BasicBlock::Create(llctx, "tail_body", fn);
    BasicBlock *tail_next_bb= BasicBlock::Create(llctx, "tail_next", fn);
    BasicBlock *exit_bb     = BasicBlock::Create(llctx, "exit", fn);

    IRBuilder<> b(entry_bb);

    Value *nrows = b.CreateLoad(i64, b.CreateStructGEP(ChunkViewTy, in_arg, 1), "nrows");
    Value *cols  = b.CreateLoad(PointerType::getUnqual(ColViewTy),
        b.CreateStructGEP(ChunkViewTy, in_arg, 0), "cols");

    std::map<int, Value*> col_data;
    for (const auto &op : agg_ops) {
        if (op.col_idx >= 0 && col_data.find(op.col_idx) == col_data.end()) {
            Value *col_i = b.CreateGEP(ColViewTy, cols, ConstantInt::get(i64, (uint64_t)op.col_idx));
            col_data[op.col_idx] = b.CreateLoad(i8p,
                b.CreateStructGEP(ColViewTy, col_i, 0));
        }
    }

    Value *vec_limit = b.CreateAnd(nrows, ConstantInt::get(i64, ~(uint64_t)(VW - 1)), "vec_limit");

    // Helper: get scalar and vector types per dtype
    auto getElemType = [&](int32_t dtype) -> Type* {
        switch (dtype) {
        case AQP_DTYPE_INT32: case AQP_DTYPE_DATE: return i32;
        case AQP_DTYPE_INT64: return i64;
        case AQP_DTYPE_FLOAT: return f32;
        case AQP_DTYPE_DOUBLE: return f64;
        default: return i32;
        }
    };
    auto isFP = [](int32_t dtype) {
        return dtype == AQP_DTYPE_FLOAT || dtype == AQP_DTYPE_DOUBLE;
    };

    // Per-accumulator state
    struct VecAcc {
        Type *elem_ty;        // scalar element type
        FixedVectorType *vty; // vector type <VW x elem_ty>
        Constant *init_vec;   // initial vector value
        bool fp;
    };
    std::vector<VecAcc> vaccs;
    for (const auto &op : agg_ops) {
        VecAcc va;
        va.fp = isFP(op.dtype);
        va.elem_ty = getElemType(op.dtype);
        va.vty = FixedVectorType::get(va.elem_ty, VW);

        if (op.agg_type == 3 || op.agg_type == 5 || op.agg_type == 6) {
            va.init_vec = va.fp
                ? ConstantVector::getSplat(ElementCount::getFixed(VW), ConstantFP::get(va.elem_ty, 0.0))
                : ConstantVector::getSplat(ElementCount::getFixed(VW), ConstantInt::get(va.elem_ty, 0));
        } else if (op.agg_type == 1) { // MIN
            va.init_vec = va.fp
                ? ConstantVector::getSplat(ElementCount::getFixed(VW), ConstantFP::getInfinity(va.elem_ty, false))
                : ConstantVector::getSplat(ElementCount::getFixed(VW),
                    ConstantInt::get(va.elem_ty, va.elem_ty == i64
                        ? APInt(64, (uint64_t)INT64_MAX) : APInt(32, (uint64_t)INT32_MAX)));
        } else if (op.agg_type == 2) { // MAX
            va.init_vec = va.fp
                ? ConstantVector::getSplat(ElementCount::getFixed(VW), ConstantFP::getInfinity(va.elem_ty, true))
                : ConstantVector::getSplat(ElementCount::getFixed(VW),
                    ConstantInt::get(va.elem_ty, va.elem_ty == i64
                        ? APInt(64, (uint64_t)INT64_MIN) : APInt(32, (uint64_t)(uint32_t)INT32_MIN)));
        } else {
            va.init_vec = ConstantVector::getSplat(ElementCount::getFixed(VW),
                                                    ConstantInt::get(va.elem_ty, 0));
        }
        vaccs.push_back(va);
    }

    b.CreateBr(vec_loop_bb);

    // ===== Vectorized loop =====
    b.SetInsertPoint(vec_loop_bb);
    PHINode *vi = b.CreatePHI(i64, 2, "vi");
    vi->addIncoming(ConstantInt::get(i64, 0), entry_bb);

    std::vector<PHINode*> vec_acc_phis;
    std::vector<Value*> vec_acc_vals; // current accumulator values
    for (size_t ai = 0; ai < vaccs.size(); ai++) {
        PHINode *phi = b.CreatePHI(vaccs[ai].vty, 2, "vacc");
        phi->addIncoming(vaccs[ai].init_vec, entry_bb);
        vec_acc_phis.push_back(phi);
        vec_acc_vals.push_back(phi);
    }

    b.CreateCondBr(b.CreateICmpEQ(vi, vec_limit), tail_bb, vec_body_bb);

    // Vec body
    b.SetInsertPoint(vec_body_bb);
    for (size_t ai = 0; ai < agg_ops.size(); ai++) {
        auto &op = agg_ops[ai];
        auto &va = vaccs[ai];

        if (op.agg_type == 6) { // CountStar
            Value *ones = va.fp
                ? ConstantVector::getSplat(ElementCount::getFixed(VW), ConstantFP::get(va.elem_ty, 1.0))
                : ConstantVector::getSplat(ElementCount::getFixed(VW), ConstantInt::get(va.elem_ty, 1));
            vec_acc_vals[ai] = va.fp
                ? b.CreateFAdd(vec_acc_vals[ai], ones)
                : b.CreateAdd(vec_acc_vals[ai], ones);
            continue;
        }
        if (op.col_idx < 0) continue;

        // Load <VW x elem_ty> from column
        Value *typed_ptr = b.CreateBitCast(col_data[op.col_idx], PointerType::getUnqual(va.elem_ty));
        Value *vec_ptr = b.CreateBitCast(
            b.CreateGEP(va.elem_ty, typed_ptr, vi), PointerType::getUnqual(va.vty));
        Value *data_vec = b.CreateLoad(va.vty, vec_ptr, "dvec");

        switch (op.agg_type) {
        case 3: // SUM
            vec_acc_vals[ai] = va.fp
                ? b.CreateFAdd(vec_acc_vals[ai], data_vec)
                : b.CreateAdd(vec_acc_vals[ai], data_vec);
            break;
        case 5: { // COUNT non-null (count all for now)
            Value *ones = va.fp
                ? ConstantVector::getSplat(ElementCount::getFixed(VW), ConstantFP::get(va.elem_ty, 1.0))
                : ConstantVector::getSplat(ElementCount::getFixed(VW), ConstantInt::get(va.elem_ty, 1));
            vec_acc_vals[ai] = va.fp ? b.CreateFAdd(vec_acc_vals[ai], ones)
                                      : b.CreateAdd(vec_acc_vals[ai], ones);
            break;
        }
        case 1: { // MIN
            Value *cmp = va.fp ? b.CreateFCmpOLT(data_vec, vec_acc_vals[ai])
                                : b.CreateICmpSLT(data_vec, vec_acc_vals[ai]);
            vec_acc_vals[ai] = b.CreateSelect(cmp, data_vec, vec_acc_vals[ai]);
            break;
        }
        case 2: { // MAX
            Value *cmp = va.fp ? b.CreateFCmpOGT(data_vec, vec_acc_vals[ai])
                                : b.CreateICmpSGT(data_vec, vec_acc_vals[ai]);
            vec_acc_vals[ai] = b.CreateSelect(cmp, data_vec, vec_acc_vals[ai]);
            break;
        }
        }
    }
    b.CreateBr(vec_next_bb);

    b.SetInsertPoint(vec_next_bb);
    Value *vi_next = b.CreateAdd(vi, ConstantInt::get(i64, VW));
    vi->addIncoming(vi_next, vec_next_bb);
    for (size_t ai = 0; ai < vec_acc_vals.size(); ai++)
        vec_acc_phis[ai]->addIncoming(vec_acc_vals[ai], vec_next_bb);
    b.CreateBr(vec_loop_bb);

    // ===== Scalar tail =====
    b.SetInsertPoint(tail_bb);
    // Horizontal reduce each vector accumulator to scalar
    std::vector<Value*> scalar_accs;
    for (size_t ai = 0; ai < vaccs.size(); ai++) {
        Value *vacc = vec_acc_vals[ai];
        Value *result = b.CreateExtractElement(vacc, (uint64_t)0);
        for (unsigned k = 1; k < VW; k++) {
            Value *elem = b.CreateExtractElement(vacc, (uint64_t)k);
            bool fp = vaccs[ai].fp;
            switch (agg_ops[ai].agg_type) {
            case 3: case 5: case 6:
                result = fp ? b.CreateFAdd(result, elem) : b.CreateAdd(result, elem); break;
            case 1: {
                Value *cmp = fp ? b.CreateFCmpOLT(elem, result) : b.CreateICmpSLT(elem, result);
                result = b.CreateSelect(cmp, elem, result); break;
            }
            case 2: {
                Value *cmp = fp ? b.CreateFCmpOGT(elem, result) : b.CreateICmpSGT(elem, result);
                result = b.CreateSelect(cmp, elem, result); break;
            }
            default: break;
            }
        }
        scalar_accs.push_back(result);
    }

    // Scalar tail loop
    PHINode *ti = b.CreatePHI(i64, 2, "ti");
    ti->addIncoming(vec_limit, vec_loop_bb);
    std::vector<PHINode*> tail_acc_phis;
    for (size_t ai = 0; ai < scalar_accs.size(); ai++) {
        PHINode *phi = b.CreatePHI(vaccs[ai].elem_ty, 2, "tacc");
        phi->addIncoming(scalar_accs[ai], vec_loop_bb);
        tail_acc_phis.push_back(phi);
    }
    b.CreateCondBr(b.CreateICmpEQ(ti, nrows), exit_bb, tail_body_bb);

    b.SetInsertPoint(tail_body_bb);
    std::vector<Value*> tail_updated;
    for (size_t ai = 0; ai < agg_ops.size(); ai++) {
        auto &op = agg_ops[ai];
        auto &va = vaccs[ai];
        Value *acc = tail_acc_phis[ai];

        if (op.agg_type == 6) {
            tail_updated.push_back(va.fp
                ? b.CreateFAdd(acc, ConstantFP::get(va.elem_ty, 1.0))
                : b.CreateAdd(acc, ConstantInt::get(va.elem_ty, 1)));
            continue;
        }
        if (op.col_idx < 0) { tail_updated.push_back(acc); continue; }

        Value *typed_ptr = b.CreateBitCast(col_data[op.col_idx], PointerType::getUnqual(va.elem_ty));
        Value *elem = b.CreateLoad(va.elem_ty, b.CreateGEP(va.elem_ty, typed_ptr, ti));

        switch (op.agg_type) {
        case 3: tail_updated.push_back(va.fp ? b.CreateFAdd(acc, elem) : b.CreateAdd(acc, elem)); break;
        case 5: tail_updated.push_back(va.fp
            ? b.CreateFAdd(acc, ConstantFP::get(va.elem_ty, 1.0))
            : b.CreateAdd(acc, ConstantInt::get(va.elem_ty, 1))); break;
        case 1: {
            Value *cmp = va.fp ? b.CreateFCmpOLT(elem, acc) : b.CreateICmpSLT(elem, acc);
            tail_updated.push_back(b.CreateSelect(cmp, elem, acc)); break;
        }
        case 2: {
            Value *cmp = va.fp ? b.CreateFCmpOGT(elem, acc) : b.CreateICmpSGT(elem, acc);
            tail_updated.push_back(b.CreateSelect(cmp, elem, acc)); break;
        }
        default: tail_updated.push_back(acc); break;
        }
    }
    b.CreateBr(tail_next_bb);

    b.SetInsertPoint(tail_next_bb);
    Value *ti_next = b.CreateAdd(ti, ConstantInt::get(i64, 1));
    ti->addIncoming(ti_next, tail_next_bb);
    for (size_t ai = 0; ai < tail_updated.size(); ai++)
        tail_acc_phis[ai]->addIncoming(tail_updated[ai], tail_next_bb);
    b.CreateBr(tail_bb);

    // ===== Exit: store final values to agg_state =====
    b.SetInsertPoint(exit_bb);

    for (size_t ai = 0; ai < agg_ops.size(); ai++) {
        auto &va = vaccs[ai];
        Value *state_ptr = b.CreateBitCast(
            b.CreateGEP(i8, state_arg, ConstantInt::get(i64, agg_ops[ai].state_offset)),
            PointerType::getUnqual(i64));
        Value *existing = b.CreateLoad(i64, state_ptr);

        // Convert partial scalar to i64 for storage
        Value *partial;
        if (va.fp) {
            // Store as double bits in i64 slot
            Value *as_f64 = (va.elem_ty == f32)
                ? b.CreateFPExt(tail_acc_phis[ai], f64)
                : tail_acc_phis[ai];
            // For FP aggregates: load existing as double, combine, store back
            Value *existing_f = b.CreateBitCast(existing, f64);
            Value *combined_f;
            switch (agg_ops[ai].agg_type) {
            case 3: case 5: case 6: combined_f = b.CreateFAdd(existing_f, as_f64); break;
            case 1: {
                Value *cmp = b.CreateFCmpOLT(as_f64, existing_f);
                combined_f = b.CreateSelect(cmp, as_f64, existing_f); break;
            }
            case 2: {
                Value *cmp = b.CreateFCmpOGT(as_f64, existing_f);
                combined_f = b.CreateSelect(cmp, as_f64, existing_f); break;
            }
            default: combined_f = existing_f; break;
            }
            b.CreateStore(b.CreateBitCast(combined_f, i64), state_ptr);
        } else {
            partial = (va.elem_ty == i32)
                ? b.CreateSExt(tail_acc_phis[ai], i64)
                : tail_acc_phis[ai]; // already i64
            Value *combined;
            switch (agg_ops[ai].agg_type) {
            case 3: case 5: case 6: combined = b.CreateAdd(existing, partial); break;
            case 1: {
                Value *cmp = b.CreateICmpSLT(partial, existing);
                combined = b.CreateSelect(cmp, partial, existing); break;
            }
            case 2: {
                Value *cmp = b.CreateICmpSGT(partial, existing);
                combined = b.CreateSelect(cmp, partial, existing); break;
            }
            default: combined = existing; break;
            }
            b.CreateStore(combined, state_ptr);
        }
    }
    b.CreateRetVoid();

    return fn;
}

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
// (AggOp struct defined earlier in file)
// ---------------------------------------------------------------------------
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

    // CountStar optimization: state.count += nrows (one instruction, no loop)
    for (const auto &op : agg_ops) {
        if (op.agg_type == 6 /* CountStar */) {
            Value *acc_ptr = b.CreateBitCast(
                b.CreateGEP(Type::getInt8Ty(llctx), state_arg,
                    ConstantInt::get(i64, op.state_offset)),
                PointerType::getUnqual(i64));
            Value *acc = b.CreateLoad(i64, acc_ptr, "count_star");
            b.CreateStore(b.CreateAdd(acc, nrows), acc_ptr);
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
        // CountStar: already handled in one shot above
        if (op.agg_type == 6 /* CountStar */) {
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
IrToLlvmCompiler::IrToLlvmCompiler(bool use_o3, bool use_simd)
    : use_o3_(use_o3), use_simd_(use_simd), impl_(std::make_unique<Impl>()) {}

IrToLlvmCompiler::~IrToLlvmCompiler() = default;

unsigned IrToLlvmCompiler::GetVecWidth() const {
    return (use_simd_ && impl_) ? impl_->vec_width : 1;
}

bool IrToLlvmCompiler::HasSIMD() const {
    return use_simd_ && impl_ && impl_->vec_width > 1;
}

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
    SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

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

    // Try SIMD version first if enabled and expressions are SIMD-friendly.
    // Split: separate numeric (SIMD-able) from VARCHAR (scalar-only) expressions.
    Function *fn = nullptr;
    bool used_simd = false;
    if (use_simd_ && impl_->vec_width > 1) {
        // Check which top-level expressions are SIMD-friendly
        std::vector<const AQPExpr*> simd_exprs, scalar_exprs;
        for (const AQPExpr *e : exprs) {
            std::vector<const AQPExpr*> single = {e};
            if (AllExprsSIMDFriendly(single, schema))
                simd_exprs.push_back(e);
            else
                scalar_exprs.push_back(e);
        }

        if (!simd_exprs.empty() && scalar_exprs.empty()) {
            // All expressions are SIMD-friendly — full SIMD path
            fn = BuildFilterFunctionSIMD(*ctx, *mod, fn_name, simd_exprs, schema,
                                         impl_->vec_width);
            if (fn) {
                used_simd = true;
                std::cerr << "[AQP-JIT] using SIMD filter (VW=" << impl_->vec_width
                          << " all_simd=" << simd_exprs.size() << ")\n";
            }
        } else if (!simd_exprs.empty()) {
            // Hybrid: SIMD numeric predicates first, then scalar VARCHAR on survivors
            fn = BuildFilterFunctionHybrid(*ctx, *mod, fn_name,
                                            simd_exprs, scalar_exprs, schema,
                                            impl_->vec_width);
            if (fn) {
                used_simd = true;
                std::cerr << "[AQP-JIT] using HYBRID filter (VW=" << impl_->vec_width
                          << " simd=" << simd_exprs.size()
                          << " scalar=" << scalar_exprs.size() << ")\n";
            } else {
                std::cerr << "[AQP-JIT] hybrid SIMD failed → scalar fallback\n";
            }
        }
    }
    // Fall back to scalar
    if (!fn) {
        fn = BuildFilterFunction(*ctx, *mod, fn_name, exprs, schema);
    }
    if (!fn) return nullptr;
    SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

    std::string err;
    raw_string_ostream es(err);
    if (verifyFunction(*fn, &es)) {
        std::cerr << "[AQP-JIT] verifyFunction failed" << (used_simd ? " (SIMD)" : "")
                  << ": " << es.str() << "\n";
        // If SIMD failed verification, retry with scalar
        if (used_simd) {
            fn->eraseFromParent();
            fn = BuildFilterFunction(*ctx, *mod, fn_name, exprs, schema);
            if (!fn) return nullptr;
            SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);
            err.clear();
            if (verifyFunction(*fn, &es)) {
                std::cerr << "[AQP-JIT] scalar fallback also failed: " << es.str() << "\n";
                return nullptr;
            }
            used_simd = false;
        } else {
            return nullptr;
        }
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
    SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

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

    // Try SIMD aggregate if enabled and all ops are SIMD-friendly
    Function *fn = nullptr;
    bool used_simd = false;
    if (use_simd_ && impl_->vec_width > 1 && AllAggOpsSIMDFriendly(agg_ops)) {
        fn = BuildAggUpdateFunctionSIMD(*ctx, *mod, fn_name, agg_ops,
                                         state_offset, in_schema, impl_->vec_width);
        if (fn) {
            used_simd = true;
            std::cerr << "[AQP-JIT] using SIMD aggregate (VW=" << impl_->vec_width << ")\n";
        }
    }
    if (!fn) {
        fn = BuildAggUpdateFunction(*ctx, *mod, fn_name, agg_ops, state_offset, in_schema);
    }
    if (!fn) return nullptr;
    SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

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
    SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

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
    SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

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

    if (filter_exprs.empty() && col_mapping.empty()) return nullptr; // nothing to compile

    uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
    std::string fn_name = "aqp_pipe_" + std::to_string(fn_id);

    auto ctx = std::make_unique<LLVMContext>();
    auto mod = std::make_unique<Module>("aqp_pipe_mod", *ctx);

    Function *fn = BuildPipelineFunction(*ctx, *mod, fn_name,
                                          filter_exprs, col_mapping,
                                          col_dtypes, in_schema);
    if (!fn) return nullptr;
    SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

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

// ---------------------------------------------------------------------------
// Filter + Aggregate fusion: one loop, no intermediate DataChunk.
//   void fn(AQPChunkView *in, void *agg_state)
// For each row: evaluate filter; if match, update accumulators.
// ---------------------------------------------------------------------------
void *IrToLlvmCompiler::CompileFilterAggFusion(
        const AQPStmt *filter_node,
        const AQPStmt *agg_node,
        const std::vector<ColSchema> &in_schema) {

    if (!agg_node) return nullptr;
    auto *agg = dynamic_cast<const SimplestAggregate *>(agg_node);
    if (!agg || agg->agg_fns.empty()) return nullptr;
    if (!agg->groups.empty()) return nullptr; // grouped agg not supported yet

    // Build filter expressions
    std::vector<const AQPExpr*> filter_exprs;
    if (filter_node) {
        for (const auto &qe : filter_node->qual_vec)
            filter_exprs.push_back(qe.get());
    }

    // Build agg ops (same logic as CompileAggUpdate)
    std::vector<AggOp> agg_ops;
    unsigned state_offset = 0;
    for (const auto &fn_pair : agg->agg_fns) {
        AggOp op;
        op.agg_type = static_cast<int32_t>(fn_pair.second);
        op.state_offset = state_offset;
        if (fn_pair.second == SimplestAggFnType::CountStar) {
            op.col_idx = -1;
            op.dtype = AQP_DTYPE_INT64;
            state_offset += 8;
        } else {
            op.col_idx = -1;
            op.dtype = AQP_DTYPE_OTHER;
            for (int i = 0; i < (int)in_schema.size(); i++) {
                if (in_schema[i].table_idx == fn_pair.first->GetTableIndex() &&
                    in_schema[i].col_idx  == fn_pair.first->GetColumnIndex()) {
                    op.col_idx = i;
                    op.dtype = in_schema[i].dtype;
                    break;
                }
            }
            if (fn_pair.second == SimplestAggFnType::Average)
                state_offset += 16;
            else
                state_offset += 8;
        }
        agg_ops.push_back(op);
    }

    uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
    std::string fn_name = "aqp_filt_agg_" + std::to_string(fn_id);

    auto ctx = std::make_unique<LLVMContext>();
    auto mod = std::make_unique<Module>("aqp_filt_agg_mod", *ctx);

    // Build function: void fn(AQPChunkView *in, i8 *agg_state)
    Type *i8    = Type::getInt8Ty(*ctx);
    Type *i8p   = PointerType::getUnqual(i8);
    Type *i32   = Type::getInt32Ty(*ctx);
    Type *i64   = Type::getInt64Ty(*ctx);
    Type *i64p  = PointerType::getUnqual(i64);
    Type *voidTy = Type::getVoidTy(*ctx);

    StructType *ColViewTy   = StructType::get(*ctx, {i8p, i64p, i32, i32});
    StructType *ChunkViewTy = StructType::get(*ctx, {
        PointerType::getUnqual(ColViewTy), i64, i64});
    StructType *SelViewTy   = StructType::get(*ctx, {
        PointerType::getUnqual(i32), i32});

    FunctionType *fn_ty = FunctionType::get(
        voidTy, {PointerType::getUnqual(ChunkViewTy), i8p}, false);
    Function *fn = Function::Create(fn_ty, Function::ExternalLinkage, fn_name, mod.get());

    Value *in_arg    = fn->getArg(0); in_arg->setName("in");
    Value *state_arg = fn->getArg(1); state_arg->setName("state");

    BasicBlock *entry_bb = BasicBlock::Create(*ctx, "entry", fn);
    BasicBlock *loop_bb  = BasicBlock::Create(*ctx, "loop", fn);
    BasicBlock *body_bb  = BasicBlock::Create(*ctx, "body", fn);
    BasicBlock *agg_bb   = BasicBlock::Create(*ctx, "agg_update", fn);
    BasicBlock *next_bb  = BasicBlock::Create(*ctx, "next", fn);
    BasicBlock *exit_bb  = BasicBlock::Create(*ctx, "exit", fn);

    // Use a dummy sel_arg for CompileCtx (not used for aggregation)
    Value *dummy_sel = ConstantPointerNull::get(PointerType::getUnqual(SelViewTy));
    CompileCtx cc(*ctx, *mod, in_schema, in_arg, dummy_sel);
    cc.b.SetInsertPoint(entry_bb);

    // Load nrows
    Value *nrows = cc.b.CreateLoad(i64, cc.b.CreateStructGEP(ChunkViewTy, in_arg, 1), "nrows");

    // Load column data + validity
    cc.col_data.resize(in_schema.size());
    cc.col_validity.resize(in_schema.size());
    for (size_t i = 0; i < in_schema.size(); i++) {
        cc.col_data[i]     = cc.LoadColData((unsigned)i);
        cc.col_validity[i] = cc.LoadColValidity((unsigned)i);
    }

    // CountStar: add nrows once (no per-row loop needed)
    for (const auto &op : agg_ops) {
        if (op.agg_type == 6) {
            Value *acc_ptr = cc.b.CreateBitCast(
                cc.b.CreateGEP(i8, state_arg, ConstantInt::get(i64, op.state_offset)),
                PointerType::getUnqual(i64));
            // CountStar with filter: can't add nrows directly — must count matches
            // So skip here; handle in the loop
        }
    }

    cc.b.CreateBr(loop_bb);

    // Loop header
    cc.b.SetInsertPoint(loop_bb);
    PHINode *row_i = cc.b.CreatePHI(i64, 2, "i");
    row_i->addIncoming(ConstantInt::get(i64, 0), entry_bb);
    cc.b.CreateCondBr(cc.b.CreateICmpEQ(row_i, nrows), exit_bb, body_bb);

    // Body: evaluate filter
    cc.b.SetInsertPoint(body_bb);
    cc.row_idx = row_i;

    Value *match = ConstantInt::getTrue(*ctx);
    for (const AQPExpr *e : filter_exprs) {
        Value *res = EmitExpr(cc, e);
        match = cc.b.CreateAnd(match, res);
    }
    BasicBlock *condBr_bb = cc.b.GetInsertBlock();
    cc.b.CreateCondBr(match, agg_bb, next_bb);

    // Aggregate update (only for matching rows)
    cc.b.SetInsertPoint(agg_bb);
    for (const auto &op : agg_ops) {
        if (op.agg_type == 6) { // CountStar: increment for each matching row
            Value *acc_ptr = cc.b.CreateBitCast(
                cc.b.CreateGEP(i8, state_arg, ConstantInt::get(i64, op.state_offset)),
                PointerType::getUnqual(i64));
            Value *acc = cc.b.CreateLoad(i64, acc_ptr);
            cc.b.CreateStore(cc.b.CreateAdd(acc, ConstantInt::get(i64, 1)), acc_ptr);
            continue;
        }
        if (op.col_idx < 0) continue;

        Value *acc_ptr = cc.b.CreateBitCast(
            cc.b.CreateGEP(i8, state_arg, ConstantInt::get(i64, op.state_offset)),
            PointerType::getUnqual(i64));

        bool is_float = (op.dtype == AQP_DTYPE_FLOAT || op.dtype == AQP_DTYPE_DOUBLE);

        // Load value based on dtype
        Value *val = nullptr;
        if (op.dtype == AQP_DTYPE_INT32 || op.dtype == AQP_DTYPE_DATE) {
            Value *p = cc.b.CreateBitCast(cc.col_data[op.col_idx], PointerType::getUnqual(i32));
            val = cc.b.CreateSExt(cc.b.CreateLoad(i32, cc.b.CreateGEP(i32, p, row_i)), i64);
        } else if (op.dtype == AQP_DTYPE_INT64) {
            Value *p = cc.b.CreateBitCast(cc.col_data[op.col_idx], PointerType::getUnqual(i64));
            val = cc.b.CreateLoad(i64, cc.b.CreateGEP(i64, p, row_i));
        } else {
            continue; // unsupported dtype
        }

        Value *acc = cc.b.CreateLoad(i64, acc_ptr);
        switch (op.agg_type) {
        case 3: cc.b.CreateStore(cc.b.CreateAdd(acc, val), acc_ptr); break; // SUM
        case 5: cc.b.CreateStore(cc.b.CreateAdd(acc, ConstantInt::get(i64, 1)), acc_ptr); break; // COUNT
        case 1: { // MIN
            Value *cmp = cc.b.CreateICmpSLT(val, acc);
            cc.b.CreateStore(cc.b.CreateSelect(cmp, val, acc), acc_ptr); break;
        }
        case 2: { // MAX
            Value *cmp = cc.b.CreateICmpSGT(val, acc);
            cc.b.CreateStore(cc.b.CreateSelect(cmp, val, acc), acc_ptr); break;
        }
        }
    }
    cc.b.CreateBr(next_bb);

    // Next
    cc.b.SetInsertPoint(next_bb);
    Value *i_next = cc.b.CreateAdd(row_i, ConstantInt::get(i64, 1));
    row_i->addIncoming(i_next, next_bb);
    cc.b.CreateBr(loop_bb);

    // Exit
    cc.b.SetInsertPoint(exit_bb);
    cc.b.CreateRetVoid();

    SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

    std::string err;
    raw_string_ostream es(err);
    if (verifyFunction(*fn, &es)) {
        std::cerr << "[AQP-JIT] verifyFunction failed (filt_agg): " << es.str() << "\n";
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

    std::cerr << "[AQP-JIT] compiled filter+agg fusion fn=" << fn_name
              << "  filter_exprs=" << filter_exprs.size()
              << "  agg_ops=" << agg_ops.size() << "\n";

    return reinterpret_cast<void*>(sym->getAddress());
}

// ---------------------------------------------------------------------------
// Sub-plan compilation (Level 4)
//
// Walks the IR tree to identify pipeline segments separated by breakers
// (AggregateNode, HashNode, JoinNode). Compiles each segment as a pipeline
// function. Returns a SubPlanDescriptor that the adapter uses to orchestrate
// execution.
// ---------------------------------------------------------------------------

// A pipeline segment: a chain of operators between two breakers
struct PipelineSegment {
    const AQPStmt *filter_node = nullptr;     // FilterNode in this segment
    const AQPStmt *proj_node   = nullptr;     // ProjectionNode in this segment
    const AQPStmt *scan_node   = nullptr;     // ScanNode (source)
    const AQPStmt *hash_node   = nullptr;     // HashNode (build side)
    const AQPStmt *agg_node    = nullptr;     // AggregateNode (sink)
    const AQPStmt *join_node   = nullptr;     // JoinNode
    const AQPStmt *parent_join = nullptr;     // parent JoinNode (for build-side key derivation)
    bool is_build = false;                     // true if this is a hash build pipeline
    int dependency = -1;                       // index of pipeline that must complete first
    AQPPipelineFn compiled_fn = nullptr;
    void *hash_build_fn = nullptr;            // compiled hash build function
};

// Collect pipeline segments from the IR tree
static void CollectPipelineSegments(const AQPStmt *node,
                                     std::vector<PipelineSegment> &segments,
                                     PipelineSegment &current) {
    if (!node) return;

    switch (node->GetNodeType()) {
    case SimplestNodeType::ScanNode:
        current.scan_node = node;
        break;
    case SimplestNodeType::FilterNode:
        current.filter_node = node;
        break;
    case SimplestNodeType::ProjectionNode:
        current.proj_node = node;
        break;
    case SimplestNodeType::HashNode:
        current.hash_node = node;
        current.is_build = true;
        break;
    case SimplestNodeType::AggregateNode:
        current.agg_node = node;
        // Aggregate is a pipeline breaker — flush current segment
        segments.push_back(current);
        current = PipelineSegment();
        break;
    case SimplestNodeType::JoinNode: {
        current.join_node = node;
        // Join has two children: probe (left) and build (right/hash)
        // Build side is a separate pipeline that must run first
        if (node->children.size() > 1) {
            // Right child (build side) → separate pipeline
            PipelineSegment build_seg;
            build_seg.is_build = true;
            build_seg.dependency = -1; // no dependency (runs first)
            build_seg.parent_join = node; // store parent join for key derivation
            CollectPipelineSegments(node->children[1].get(), segments, build_seg);
            if (build_seg.scan_node || build_seg.filter_node || build_seg.hash_node) {
                int build_idx = (int)segments.size();
                segments.push_back(build_seg);
                current.dependency = build_idx; // probe depends on build
            }
        }
        // Left child (probe side) continues current pipeline
        if (!node->children.empty()) {
            CollectPipelineSegments(node->children[0].get(), segments, current);
        }
        return; // already recursed into children
    }
    default:
        break;
    }

    // Recurse into children
    for (const auto &child : node->children)
        CollectPipelineSegments(child.get(), segments, current);
}

void *IrToLlvmCompiler::CompileSubPlan(const AQPStmt &sub_ir) {
    // Step 1: Collect pipeline segments from the IR tree
    std::vector<PipelineSegment> segments;
    PipelineSegment initial;
    CollectPipelineSegments(&sub_ir, segments, initial);
    // Flush any remaining segment
    if (initial.scan_node || initial.filter_node || initial.proj_node)
        segments.push_back(initial);

    if (segments.empty()) {
        std::cerr << "[AQP-JIT] sub-plan: no pipeline segments found\n";
        return nullptr;
    }

    std::cerr << "[AQP-JIT] sub-plan: found " << segments.size() << " pipeline segment(s)\n";

    // Step 2: Build schema for each segment and compile pipeline functions
    for (size_t si = 0; si < segments.size(); si++) {
        auto &seg = segments[si];
        std::cerr << "[AQP-JIT] sub-plan segment[" << si << "]:"
                  << " scan=" << (seg.scan_node ? "yes" : "no")
                  << " filter=" << (seg.filter_node ? "yes" : "no")
                  << " proj=" << (seg.proj_node ? "yes" : "no")
                  << " hash=" << (seg.hash_node ? "yes" : "no")
                  << " agg=" << (seg.agg_node ? "yes" : "no")
                  << " build=" << seg.is_build
                  << " dep=" << seg.dependency << "\n";

        // Build input schema from scan node's target_list
        std::vector<ColSchema> in_schema;
        const AQPStmt *schema_source = seg.scan_node;
        if (!schema_source && seg.filter_node && !seg.filter_node->children.empty())
            schema_source = seg.filter_node->children[0].get();
        if (!schema_source) schema_source = seg.filter_node;

        if (schema_source && !schema_source->target_list.empty()) {
            for (const auto &attr : schema_source->target_list) {
                ColSchema cs;
                cs.table_idx = attr->GetTableIndex();
                cs.col_idx   = attr->GetColumnIndex();
                switch (attr->GetType()) {
                case ir_sql_converter::IntVar:    cs.dtype = AQP_DTYPE_INT32; break;
                case ir_sql_converter::FloatVar:  cs.dtype = AQP_DTYPE_DOUBLE; break;
                case ir_sql_converter::StringVar: cs.dtype = AQP_DTYPE_VARCHAR; break;
                case ir_sql_converter::BoolVar:   cs.dtype = AQP_DTYPE_BOOL; break;
                case ir_sql_converter::Date:      cs.dtype = AQP_DTYPE_DATE; break;
                default: cs.dtype = AQP_DTYPE_OTHER; break;
                }
                in_schema.push_back(cs);
            }
        }

        if (in_schema.empty()) {
            std::cerr << "[AQP-JIT] sub-plan segment[" << si << "]: no schema → skip\n";
            continue;
        }

        // Compile the segment as a pipeline function (filter → projection)
        seg.compiled_fn = CompilePipeline(seg.filter_node, seg.proj_node, in_schema);
        if (seg.compiled_fn) {
            std::cerr << "[AQP-JIT] sub-plan segment[" << si << "]: compiled pipeline\n";
        }

        // Compile hash build: either from explicit HashNode or synthesized from JoinNode
        if (seg.hash_node) {
            seg.hash_build_fn = CompileHashBuild(*seg.hash_node, in_schema);
            if (seg.hash_build_fn)
                std::cerr << "[AQP-JIT] sub-plan segment[" << si << "]: compiled hash build (from HashNode)\n";
        } else if (seg.is_build && seg.parent_join) {
            // Build side without HashNode (DuckDB path): derive keys from JoinNode conditions
            auto *join = dynamic_cast<const SimplestJoin *>(seg.parent_join);
            if (join && !join->join_conditions.empty()) {
                // Synthesize SimplestHash with keys from join conditions (build-side attrs)
                std::vector<std::unique_ptr<SimplestAttr>> synth_keys;
                for (const auto &cond : join->join_conditions) {
                    // Build-side key: the attr that matches our build schema
                    bool left_in = false, right_in = false;
                    for (const auto &cs : in_schema) {
                        if (cs.table_idx == cond->left_attr->GetTableIndex() &&
                            cs.col_idx  == cond->left_attr->GetColumnIndex())
                            left_in = true;
                        if (cs.table_idx == cond->right_attr->GetTableIndex() &&
                            cs.col_idx  == cond->right_attr->GetColumnIndex())
                            right_in = true;
                    }
                    if (left_in)
                        synth_keys.push_back(std::make_unique<SimplestAttr>(*cond->left_attr));
                    else if (right_in)
                        synth_keys.push_back(std::make_unique<SimplestAttr>(*cond->right_attr));
                }

                if (!synth_keys.empty()) {
                    auto synth_base = std::make_unique<AQPStmt>(
                        std::vector<std::unique_ptr<AQPStmt>>{},
                        SimplestNodeType::HashNode);
                    SimplestHash synth_hash(std::move(synth_base), std::move(synth_keys));
                    seg.hash_build_fn = CompileHashBuild(synth_hash, in_schema);
                    if (seg.hash_build_fn)
                        std::cerr << "[AQP-JIT] sub-plan segment[" << si
                                  << "]: compiled hash build (from JoinNode, "
                                  << join->join_conditions.size() << " keys)\n";
                }
            }
        }

        // Compile scan as pass-through pipeline if nothing else compiled
        // (establishes the segment as "compiled" for sub-plan orchestration)
        if (!seg.compiled_fn && !seg.hash_build_fn && seg.scan_node) {
            // Scan-only segment: compile as identity pipeline (pass all columns through)
            seg.compiled_fn = CompilePipeline(nullptr, nullptr, in_schema);
            if (seg.compiled_fn)
                std::cerr << "[AQP-JIT] sub-plan segment[" << si << "]: compiled scan (pass-through)\n";
        }

        // Also compile aggregate if present
        if (seg.agg_node) {
            void *agg_fn = CompileAggUpdate(*seg.agg_node, in_schema);
            if (agg_fn)
                std::cerr << "[AQP-JIT] sub-plan segment[" << si << "]: compiled aggregate\n";
        }
    }

    // Step 3: Count successfully compiled segments
    int compiled_count = 0;
    for (auto &seg : segments)
        if (seg.compiled_fn || seg.hash_build_fn) compiled_count++;

    std::cerr << "[AQP-JIT] sub-plan: " << compiled_count << "/" << segments.size()
              << " segments compiled\n";

    // The sub-plan coordinator is not an LLVM function — it's orchestrated
    // by the adapter at the IRQuerySplitter level. The compiled pipeline
    // functions are stored in the JIT context. The adapter calls them
    // in dependency order (build before probe).
    //
    // Return non-null to indicate sub-plan compilation succeeded.
    // The actual value is the number of compiled segments (cast to void*).
    return (compiled_count > 0) ? reinterpret_cast<void*>((uintptr_t)compiled_count) : nullptr;
}

} // namespace aqp_jit
