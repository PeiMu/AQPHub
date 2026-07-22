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
#include "qjit/query_jit_steps.h"

// LLVM headers — only included in this TU
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#if LLVM_VERSION_MAJOR >= 16
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#ifdef AQP_HAVE_TPDE
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <tpde-llvm/LLVMCompiler.hpp>
#endif
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>

#include "simplest_ir.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <unistd.h>

// LLVM version compatibility (14 vs 16+)
#if LLVM_VERSION_MAJOR >= 16
#define AQP_JIT_SYM(ptr) \
  ExecutorSymbolDef(ExecutorAddr::fromPtr(ptr), JITSymbolFlags::Exported)
#define AQP_JIT_SYM_ADDR(addr) \
  ExecutorSymbolDef(ExecutorAddr(addr), JITSymbolFlags::Exported)
#define AQP_JIT_GET_ADDR(sym) reinterpret_cast<void *>((sym)->getValue())
#define AQP_JIT_GET_FN(T, sym) (sym)->toPtr<T>()
#else
#define AQP_JIT_SYM(ptr) \
  JITEvaluatedSymbol(pointerToJITTargetAddress(ptr), JITSymbolFlags::Exported)
#define AQP_JIT_SYM_ADDR(addr) \
  JITEvaluatedSymbol(static_cast<JITTargetAddress>(addr), JITSymbolFlags::Exported)
#define AQP_JIT_GET_ADDR(sym) reinterpret_cast<void *>((sym)->getAddress())
#define AQP_JIT_GET_FN(T, sym) jitTargetAddressToFunction<T>((sym)->getAddress())
#endif
#include "kernel/pipeline_kernel.h"

// Forward-declare C-linkage runtime helpers (defined in aqp_jit_runtime.cpp
// and aqp_jit_hashtable.cpp).  Must come before Impl::Impl() which takes
// their addresses.
extern "C" {
int aqp_like_match(const char *str, int32_t slen, const char *pat,
                   int32_t plen);
int aqp_ilike_match(const char *str, int32_t slen, const char *pat,
                    int32_t plen);
int aqp_str_eq(const char *a, int32_t alen, const char *b, int32_t blen);
int aqp_str_cmp(const char *a, int32_t alen, const char *b, int32_t blen);
int aqp_str_contains(const char *str, int32_t slen, const char *pat,
                     int32_t plen);
int aqp_like_match_segments(const char *str, int32_t slen,
                            const char **segs, const int32_t *seg_lens,
                            int32_t n_segs, int has_leading_pct,
                            int has_trailing_pct);
int aqp_in_set_i32(int32_t val, const int32_t *values, int32_t n);
int aqp_in_set_i64(int64_t val, const int64_t *values, int32_t n);
int aqp_in_set_str(const char *str, int32_t slen, const char **ptrs,
                   const int32_t *lens, int32_t n);
// Directory-bucketed IN-set (large sets); aqp_in_hash_mix64 is also used at
// codegen time to lay out the bucket-ordered constant arrays.
uint64_t aqp_in_hash_mix64(uint64_t v);
int aqp_in_hash_i32(int32_t val, const int32_t *vals, const int32_t *dir,
                    int32_t mask);
int aqp_in_hash_i64(int64_t val, const int64_t *vals, const int32_t *dir,
                    int32_t mask);
int aqp_in_len_str(const char *str, int32_t slen, const char **ptrs,
                   const int32_t *lens, const int32_t *dir, int32_t cap);
// Hash table (aqp_jit_hashtable.cpp)
struct AQPHashTable;
AQPHashTable *aqp_ht_create(uint32_t key_width, uint32_t payload_width,
                            uint64_t est_rows);
void aqp_ht_destroy(AQPHashTable *ht);
void *aqp_ht_insert(AQPHashTable *ht, const void *key);
void *aqp_ht_insert_prehash(AQPHashTable *ht, const void *key, uint64_t hash);
void *aqp_ht_probe(const AQPHashTable *ht, const void *key);
void *aqp_ht_probe_prehash(const AQPHashTable *ht, const void *key,
                            uint64_t hash);
void aqp_ht_iter_reset(AQPHashTable *ht);
int aqp_ht_next(AQPHashTable *ht, void **key_out, void **payload_out);
uint64_t aqp_ht_size(const AQPHashTable *ht);
uint8_t *aqp_ht_slots_base(const AQPHashTable *ht);
uint64_t aqp_ht_mask(const AQPHashTable *ht);
uint32_t aqp_ht_slot_size(const AQPHashTable *ht);
uint64_t aqp_hash(const void *key, uint32_t len);
// Safe VARCHAR copy (defined in duckdb/src/execution/aqp_jit.cpp)
void aqp_copy_string(void *dst_data, void *src_data,
                     uint64_t dst_row, uint64_t src_row,
                     void *state_ptr, uint32_t col_idx);
// Pipeline kernel builder helpers — resolved at runtime via dlsym()
// to avoid link-time dependency from aqp_jit_lib to kernel_lib.
}

#include <atomic>
#include <climits>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <dlfcn.h>
#ifndef NDEBUG
#include <fcntl.h>
#include <unistd.h>
#endif
#include <unordered_map>
#include <vector>

#include <llvm/Support/MemoryBuffer.h>

// Monotonically-increasing counter used to generate unique function names.
// Using pointer/hash produced duplicate names when two filters in the same
// sub-query happened to hash to the same value within the same LLJIT dylib.
static std::atomic<uint64_t> s_filter_counter{0};

using namespace llvm;
using namespace llvm::orc;
using namespace ir_sql_converter;

#ifdef AQP_HAVE_TPDE
namespace {
// tpde_llvm::ConcurrentOrcCompiler falls back to LLVM SimpleCompiler
// SILENTLY on unsupported IR; this wrapper does the same but logs each
// fallback so a --compile-mode=tpde measurement run cannot secretly
// degrade into an all-LLVM run. The fallback TargetMachine inherits the
// FastISel-O0 settings set on the JTMB in Impl's constructor.
class TpdeOrcCompiler : public IRCompileLayer::IRCompiler {
  JITTargetMachineBuilder jtmb_;

public:
  explicit TpdeOrcCompiler(JITTargetMachineBuilder jtmb)
      : IRCompiler({}), jtmb_(std::move(jtmb)) {}

  Expected<std::unique_ptr<MemoryBuffer>> operator()(Module &mod) override {
    std::vector<uint8_t> buf;
    auto compiler = tpde_llvm::LLVMCompiler::create(jtmb_.getTargetTriple());
    if (compiler && compiler->compile_to_elf(mod, buf)) {
      return MemoryBuffer::getMemBufferCopy(
          StringRef(reinterpret_cast<const char *>(buf.data()), buf.size()),
          mod.getModuleIdentifier());
    }
    fprintf(stderr, "[AQP-JIT] TPDE rejected module '%s' -> LLVM fallback\n",
            mod.getModuleIdentifier().c_str());
    auto tm = jtmb_.createTargetMachine();
    if (!tm)
      return tm.takeError();
    return SimpleCompiler(**tm)(mod);
  }
};
} // namespace
#endif

namespace aqp_jit {

// ---------------------------------------------------------------------------
// Filter predicate cost model (B2 in runtime_execution_optimizations.md).
// Returns a small integer; lower = evaluate first. Cheap conjuncts run before
// expensive ones so the short-circuit chain skips slow predicates (string
// compare, LIKE call, large IN list) whenever a cheap predicate already
// rejected the row.
// ---------------------------------------------------------------------------
static int EstimateFilterCost(const AQPExpr *e) {
  if (!e) return 100;
  SimplestExprType et = e->GetSimplestExprType();
  if (et == SimplestExprType::TextLike ||
      et == SimplestExprType::Text_Not_Like)
    return 30;

  SimplestNodeType nt = e->GetNodeType();
  if (nt == IsNullExprNode) return 1;
  if (nt == ArithExprNode) return 4;

  auto attr_dtype = [&]() -> SimplestVarType {
    if (nt == VarComparisonNode) {
      auto *vc = static_cast<const SimplestVarComparison *>(e);
      return vc->left_attr ? vc->left_attr->GetType() : InvalidVarType;
    }
    if (nt == VarConstComparisonNode) {
      auto *vc = static_cast<const SimplestVarConstComparison *>(e);
      return vc->attr ? vc->attr->GetType() : InvalidVarType;
    }
    if (nt == VarParamComparisonNode) {
      auto *vc = static_cast<const SimplestVarParamComparison *>(e);
      return vc->attr ? vc->attr->GetType() : InvalidVarType;
    }
    if (nt == InExprNode) {
      auto *in = static_cast<const SimplestInExpr *>(e);
      return in->attr ? in->attr->GetType() : InvalidVarType;
    }
    return InvalidVarType;
  };

  if (nt == InExprNode) {
    auto *in = static_cast<const SimplestInExpr *>(e);
    int sz = (int)in->values.size();
    SimplestVarType vt = attr_dtype();
    int per = (vt == StringVar || vt == StringVarArr) ? 10 : 1;
    return 5 + per * sz;
  }

  SimplestVarType vt = attr_dtype();
  if (vt == StringVar || vt == StringVarArr) return 10;
  if (vt == Date) return 3;
  if (vt == FloatVar) return 2;
  if (vt == IntVar || vt == BoolVar) return 1;
  return 5;
}

// Stable-sort a filter-conjunct list cheap-first. Stable so equal-cost
// predicates retain user-given order.
static void SortFiltersByCost(std::vector<const AQPExpr *> &filter_exprs) {
  std::stable_sort(filter_exprs.begin(), filter_exprs.end(),
                   [](const AQPExpr *a, const AQPExpr *b) {
                     return EstimateFilterCost(a) < EstimateFilterCost(b);
                   });
}

// ---------------------------------------------------------------------------
// LLVM initialisation (done once per process)
// ---------------------------------------------------------------------------
static std::once_flag llvm_init_flag;
static void EnsureLLVMInit() {
  std::call_once(llvm_init_flag, [] {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();
  });
}

// ---------------------------------------------------------------------------
// SIMD ISA → vector width resolution
// ---------------------------------------------------------------------------
static unsigned ResolveVecWidth(SimdISA simd,
                                const StringMap<bool> &host_features) {
  auto has = [&](const char *f) -> bool {
    auto it = host_features.find(f);
    return it != host_features.end() && it->second;
  };
  bool has_sse2 = has("sse2");
  bool has_avx = has("avx");
  bool has_avx2 = has("avx2");
  bool has_avx512f = has("avx512f");

  switch (simd) {
  case SimdISA::OFF:
    return 1;
  case SimdISA::SSE2:
    return has_sse2 ? 4 : 1;
  case SimdISA::AVX:
    return has_avx ? 8 : (has_sse2 ? 4 : 1);
  case SimdISA::AVX2:
    return has_avx2 ? 8 : (has_avx ? 8 : (has_sse2 ? 4 : 1));
  case SimdISA::AVX512:
    return has_avx512f ? 16 : (has_avx2 ? 8 : (has_sse2 ? 4 : 1));
  case SimdISA::AUTO:
    if (has_avx512f)
      return 16;
    if (has_avx2)
      return 8;
    if (has_sse2)
      return 4;
    return 1;
  }
  return 1;
}

// Build LLVM target-features string constrained to the requested ISA.
// e.g. --jit-simd=avx2 includes +avx2,+sse4.2 but NOT +avx512f
static std::string BuildFeatureStr(SimdISA simd,
                                   const StringMap<bool> &host_features) {
  if (simd == SimdISA::OFF)
    return "";

  // ISA level → maximum feature set allowed
  // SSE2: only up to SSE4.2 features
  // AVX:  up to AVX (no AVX2)
  // AVX2: up to AVX2 (no AVX-512)
  // AVX512 / AUTO: all host features
  if (simd == SimdISA::AUTO || simd == SimdISA::AVX512) {
    std::string fs;
    for (auto &kv : host_features) {
      if (!fs.empty())
        fs += ",";
      fs += (kv.second ? "+" : "-");
      fs += kv.first().str();
    }
    return fs;
  }

  // For constrained ISA levels, build feature string with only allowed features
  static const char *const sse2_only[] = {"sse2",   "sse3", "ssse3", "sse4.1",
                                          "sse4.2", "cmov", "cx8",   "mmx",
                                          "fxsr",   "cx16", "popcnt"};
  static const char *const avx_extra[] = {"avx",   "xsave", "xsaveopt",
                                          "rdrnd", "f16c",  "fsgsbase"};
  static const char *const avx2_extra[] = {"avx2", "bmi",   "bmi2",
                                           "fma",  "lzcnt", "movbe"};

  // SSE2 base features
  auto is_in_list = [](const char *name, const char *const *list,
                       size_t n) -> bool {
    for (size_t i = 0; i < n; i++)
      if (name == std::string(list[i]))
        return true;
    return false;
  };

  std::string fs;
  auto add_feature = [&](const char *name, bool enabled) {
    if (!fs.empty())
      fs += ",";
    fs += (enabled ? "+" : "-");
    fs += name;
  };

  // Always include SSE2-level features
  for (auto feat : sse2_only) {
    auto it = host_features.find(feat);
    bool enabled = it != host_features.end() && it->second;
    add_feature(feat, enabled);
  }
  if (simd == SimdISA::SSE2)
    return fs;

  // AVX-level features
  for (auto feat : avx_extra) {
    auto it = host_features.find(feat);
    bool enabled = it != host_features.end() && it->second;
    add_feature(feat, enabled);
  }
  if (simd == SimdISA::AVX)
    return fs;

  // AVX2-level features
  for (auto feat : avx2_extra) {
    auto it = host_features.find(feat);
    bool enabled = it != host_features.end() && it->second;
    add_feature(feat, enabled);
  }
  return fs;
}

// ---------------------------------------------------------------------------
// Helper: FNV-1a hash of a string — used to generate unique function names
// ---------------------------------------------------------------------------
static uint64_t FNV1a(const std::string &s) {
  uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}

// ---------------------------------------------------------------------------
// Pimpl implementation
// ---------------------------------------------------------------------------
struct IrToLlvmCompiler::Impl {
  std::unique_ptr<LLJIT> jit;
  // Per-generation resource tracker. All compiled IR modules and cached
  // object files are added with this tracker so that ResetModules() can
  // free the machine code, IR allocations and symbol-table entries for a
  // whole generation at once. The runtime helper symbols defined in the
  // ctor are intentionally NOT tracked (they live in the default tracker)
  // because they are constant for the compiler's lifetime.
  orc::ResourceTrackerSP current_tracker;

  // SIMD configuration (detected at init time)
  std::string host_cpu;
  std::string feature_str;
  unsigned vec_width =
      1; // SIMD lanes for i32: 1=scalar, 4=SSE, 8=AVX2, 16=AVX-512
  bool has_avx2 = false;
  bool has_avx512f = false;
  bool has_sse42 = false;

  // In-memory object cache (debug flag --jit-cache, default OFF).
  // Stores compiled object code bytes keyed by content hash so that
  // identical IR across queries can skip LLVM compilation. The object
  // bytes survive ResetModules() and are re-loaded via addObjectFile.
  // PROCESS-GLOBAL: the main compiler and the rotating spec-jit background
  // compilers are distinct IrToLlvmCompiler instances; identical content
  // must hit across all of them. Entries are write-once (emplace, never
  // overwritten), so a reference obtained under the lock stays valid after
  // unlock (node-based map: rehash never invalidates references).
  bool cache_enabled = false;
  std::string pending_cache_key;
  std::string last_cache_key;
  std::string last_entry_name;
  std::string disk_cache_dir;
  static std::mutex &ObjCacheMu() {
    static std::mutex mu;
    return mu;
  }
  static std::unordered_map<std::string, std::string> &ObjCache() {
    static std::unordered_map<std::string, std::string> cache;
    return cache;
  }

  static std::string ComputeCacheKey(const std::string &content) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : content) {
      h ^= c;
      h *= 1099511628211ULL;
    }
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return std::string(buf);
  }

  static std::string TryDiskLoad(const std::string &dir,
                                   const std::string &key) {
    auto path = dir + "/" + key + ".o";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
      return {};
    auto sz = f.tellg();
    if (sz <= 0)
      return {};
    std::string bytes(static_cast<size_t>(sz), '\0');
    f.seekg(0);
    f.read(bytes.data(), sz);
    return f ? bytes : std::string{};
  }

  static void WriteToDisk(const std::string &dir, const std::string &key,
                          const std::string &bytes) {
    auto final_path = dir + "/" + key + ".o";
    auto tmp_path = final_path + ".tmp." + std::to_string(getpid());
    {
      std::ofstream f(tmp_path, std::ios::binary);
      if (!f)
        return;
      f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      if (!f) {
        std::remove(tmp_path.c_str());
        return;
      }
    }
    std::rename(tmp_path.c_str(), final_path.c_str());
  }

  void *TryCacheLoad(const std::string &key, const std::string &fn_name) {
    if (!cache_enabled || key.empty())
      return nullptr;
    const std::string *bytes = nullptr;
    {
      std::lock_guard<std::mutex> lk(ObjCacheMu());
      auto it = ObjCache().find(key);
      if (it != ObjCache().end())
        bytes = &it->second;
    }
    if (!bytes && !disk_cache_dir.empty()) {
      auto disk_bytes = TryDiskLoad(disk_cache_dir, key);
      if (!disk_bytes.empty()) {
#ifndef NDEBUG
        std::cerr << "[AQP-JIT] disk cache HIT: " << key << "\n";
#endif
        std::lock_guard<std::mutex> lk(ObjCacheMu());
        auto [it2, _] = ObjCache().emplace(key, std::move(disk_bytes));
        bytes = &it2->second;
      }
    }
    if (!bytes)
      return nullptr;

    // Same content already materialized in this generation (e.g. two
    // identical fusion pipelines in one query): reuse the live symbol — a
    // second addObjectFile of the same name would be a duplicate definition.
    if (auto existing = jit->lookup(fn_name)) {
#ifndef NDEBUG
      std::cerr << "[AQP-JIT] cache HIT (live): " << fn_name << "\n";
#endif
      return AQP_JIT_GET_ADDR(existing);
    } else {
      consumeError(existing.takeError());
    }

    auto buf = MemoryBuffer::getMemBufferCopy(
        StringRef(bytes->data(), bytes->size()));
    if (auto e = jit->addObjectFile(current_tracker, std::move(buf))) {
      logAllUnhandledErrors(std::move(e), errs());
      return nullptr;
    }

    auto sym = jit->lookup(fn_name);
    if (!sym) {
      logAllUnhandledErrors(sym.takeError(), errs());
      return nullptr;
    }

#ifndef NDEBUG
    std::cerr << "[AQP-JIT] cache HIT: " << fn_name << " (" << key << ")\n";
#endif
    return AQP_JIT_GET_ADDR(sym);
  }

  void InstallCacheHook() {
    if (!cache_enabled)
      return;
    jit->getObjTransformLayer().setTransform(
        [this](std::unique_ptr<MemoryBuffer> obj)
            -> Expected<std::unique_ptr<MemoryBuffer>> {
          if (!pending_cache_key.empty()) {
            std::string obj_bytes(obj->getBufferStart(), obj->getBufferSize());
            {
              std::lock_guard<std::mutex> lk(ObjCacheMu());
              ObjCache().emplace(pending_cache_key, obj_bytes);
            }
            if (!disk_cache_dir.empty())
              WriteToDisk(disk_cache_dir, pending_cache_key, obj_bytes);
            pending_cache_key.clear();
          }
          return std::move(obj);
        });
  }

  Impl(SimdISA simd_isa, FastCompileBackend fast = FastCompileBackend::OFF) {
    EnsureLLVMInit();

    // Detect CPU features for SIMD
    host_cpu = std::string(sys::getHostCPUName());
    auto host_features = sys::getHostCPUFeatures();

    has_sse42 = host_features.count("sse4.2") && host_features["sse4.2"];
    has_avx2 = host_features.count("avx2") && host_features["avx2"];
    has_avx512f = host_features.count("avx512f") && host_features["avx512f"];

    // Resolve vector width from requested ISA level
    vec_width = ResolveVecWidth(simd_isa, host_features);

    // Build feature string constrained to the requested ISA level
    feature_str = BuildFeatureStr(simd_isa, host_features);

#ifndef NDEBUG
    std::cerr << "[AQP-JIT] CPU=" << host_cpu << " AVX2=" << has_avx2
              << " AVX512=" << has_avx512f << " vec_width=" << vec_width
              << "\n";
#endif

    // Create LLJIT with detected CPU features for optimal codegen
    auto jtmb = JITTargetMachineBuilder::detectHost();
    if (jtmb) {
      jtmb->setCPU(host_cpu);
      // Note: feature_str is applied per-function via attributes
      if (fast != FastCompileBackend::OFF) {
        // §6.6 fast tier: O0 instruction selection with FastISel (mirrors
        // lingo-db LLVMBackends.cpp:248-249). For TPDE this configures the
        // LLVM fallback used when TPDE rejects a module.
        jtmb->setCodeGenOptLevel(CodeGenOptLevel::None);
        jtmb->getOptions().EnableFastISel = true;
      }
    }

    auto builder = LLJITBuilder();
    if (jtmb)
      builder.setJITTargetMachineBuilder(std::move(*jtmb));
    if (fast == FastCompileBackend::TPDE) {
#ifdef AQP_HAVE_TPDE
      builder.setCompileFunctionCreator(
          [](JITTargetMachineBuilder JTMB)
              -> Expected<std::unique_ptr<orc::IRCompileLayer::IRCompiler>> {
            return std::make_unique<TpdeOrcCompiler>(std::move(JTMB));
          });
#else
      throw std::runtime_error(
          "--compile-mode=tpde requires a build with AQP_HAVE_TPDE "
          "(see third_party/tpde_build)");
#endif
    }
    auto jit_or = builder.create();
    if (!jit_or) {
      std::string msg;
      raw_string_ostream ss(msg);
      ss << "Failed to create ORC LLJIT: ";
      logAllUnhandledErrors(jit_or.takeError(), ss);
      throw std::runtime_error(ss.str());
    }
    jit = std::move(*jit_or);
    current_tracker = jit->getMainJITDylib().createResourceTracker();

    // Make runtime helper symbols (aqp_like_match etc.) visible to JIT.
    auto &es = jit->getExecutionSession();
    auto &jd = jit->getMainJITDylib();
    (void)jd.define(absoluteSymbols({
        {es.intern("aqp_like_match"), AQP_JIT_SYM(::aqp_like_match)},
        {es.intern("aqp_ilike_match"), AQP_JIT_SYM(::aqp_ilike_match)},
        {es.intern("aqp_str_eq"), AQP_JIT_SYM(::aqp_str_eq)},
        {es.intern("aqp_str_cmp"), AQP_JIT_SYM(::aqp_str_cmp)},
        {es.intern("aqp_str_contains"), AQP_JIT_SYM(::aqp_str_contains)},
        {es.intern("aqp_like_match_segments"), AQP_JIT_SYM(::aqp_like_match_segments)},
        {es.intern("memcmp"), AQP_JIT_SYM(::memcmp)},
        {es.intern("memchr"),
         AQP_JIT_SYM_ADDR(reinterpret_cast<uintptr_t>(
             static_cast<void *(*)(void *, int, size_t)>(::memchr)))},
        {es.intern("aqp_in_set_i32"), AQP_JIT_SYM(::aqp_in_set_i32)},
        {es.intern("aqp_in_set_i64"), AQP_JIT_SYM(::aqp_in_set_i64)},
        {es.intern("aqp_in_set_str"), AQP_JIT_SYM(::aqp_in_set_str)},
        {es.intern("aqp_in_hash_i32"), AQP_JIT_SYM(::aqp_in_hash_i32)},
        {es.intern("aqp_in_hash_i64"), AQP_JIT_SYM(::aqp_in_hash_i64)},
        {es.intern("aqp_in_len_str"), AQP_JIT_SYM(::aqp_in_len_str)},
        {es.intern("aqp_ht_create"), AQP_JIT_SYM(::aqp_ht_create)},
        {es.intern("aqp_ht_destroy"), AQP_JIT_SYM(::aqp_ht_destroy)},
        {es.intern("aqp_ht_insert"), AQP_JIT_SYM(::aqp_ht_insert)},
        {es.intern("aqp_ht_insert_prehash"), AQP_JIT_SYM(::aqp_ht_insert_prehash)},
        {es.intern("aqp_ht_probe"), AQP_JIT_SYM(::aqp_ht_probe)},
        {es.intern("aqp_ht_probe_prehash"), AQP_JIT_SYM(::aqp_ht_probe_prehash)},
        {es.intern("aqp_ht_iter_reset"), AQP_JIT_SYM(::aqp_ht_iter_reset)},
        {es.intern("aqp_ht_next"), AQP_JIT_SYM(::aqp_ht_next)},
        {es.intern("aqp_ht_size"), AQP_JIT_SYM(::aqp_ht_size)},
        {es.intern("aqp_ht_slots_base"), AQP_JIT_SYM(::aqp_ht_slots_base)},
        {es.intern("aqp_ht_mask"), AQP_JIT_SYM(::aqp_ht_mask)},
        {es.intern("aqp_ht_slot_size"), AQP_JIT_SYM(::aqp_ht_slot_size)},
        {es.intern("aqp_hash"), AQP_JIT_SYM(::aqp_hash)},
        {es.intern("aqp_copy_string"), AQP_JIT_SYM(::aqp_copy_string)},
        {es.intern("memcpy"), AQP_JIT_SYM((void *)&memcpy)},
        {es.intern("memset"), AQP_JIT_SYM((void *)&memset)},
        {es.intern("aqp_jit_get_params"), AQP_JIT_SYM(aqp_jit_get_params)},
    }));
  }

  void EnableCache() {
    if (cache_enabled)
      return;
    cache_enabled = true;
    InstallCacheHook();
  }

  // Frees all JIT machine code, IR allocations, symbol-table entries, and
  // ExecutionSession state added under the current tracker, then opens a
  // fresh tracker for subsequent additions. Caller is responsible for
  // ensuring no JIT function pointer obtained against the old tracker is
  // still in use — pair this with clearing aqp_jit_context first.
  //
  // Runtime helper symbols defined in Impl() are not tracked and survive.
  void ResetModules() {
    if (current_tracker) {
      if (auto e = current_tracker->remove()) {
        logAllUnhandledErrors(std::move(e), errs());
      }
    }
    current_tracker = jit->getMainJITDylib().createResourceTracker();
  }
};

// RAII guard: temporarily swaps the tracker slot with an isolated one,
// restoring the original on destruction. Allows isolated-tracker overloads
// to reuse the same Compile* bodies without touching every addIRModule site.
struct TrackerGuard {
  orc::ResourceTrackerSP &slot;
  orc::ResourceTrackerSP saved;
  TrackerGuard(orc::ResourceTrackerSP &tracker_slot, JITTrackerHandle &h)
      : slot(tracker_slot), saved(slot) {
    slot = *static_cast<orc::ResourceTrackerSP *>(h.ptr);
  }
  ~TrackerGuard() { slot = std::move(saved); }
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

// §7.3 single-run-template (cache mode 2): tracks constant layout during
// codegen. The same walk order at cache-hit time produces an identical buffer.
struct ParamsBuilder {
  std::vector<uint8_t> buf;
  uint32_t offset = 0;

  uint32_t Align(uint32_t alignment) {
    uint32_t r = offset % alignment;
    if (r) offset += alignment - r;
    buf.resize(offset);
    return offset;
  }

  uint32_t AllocI32(int32_t v) {
    Align(4);
    uint32_t off = offset;
    buf.resize(offset + 4);
    memcpy(buf.data() + off, &v, 4);
    offset += 4;
    return off;
  }

  uint32_t AllocI64(int64_t v) {
    Align(8);
    uint32_t off = offset;
    buf.resize(offset + 8);
    memcpy(buf.data() + off, &v, 8);
    offset += 8;
    return off;
  }

  uint32_t AllocF32(float v) {
    Align(4);
    uint32_t off = offset;
    buf.resize(offset + 4);
    memcpy(buf.data() + off, &v, 4);
    offset += 4;
    return off;
  }

  uint32_t AllocF64(double v) {
    Align(8);
    uint32_t off = offset;
    buf.resize(offset + 8);
    memcpy(buf.data() + off, &v, 8);
    offset += 8;
    return off;
  }

  // Returns offset of an 8-byte slot {uint32_t str_offset, uint32_t str_len}.
  // Actual string bytes are appended during Finalize(). This keeps the
  // fixed-offset section deterministic regardless of string length.
  uint32_t AllocString(const std::string &s) {
    Align(4);
    uint32_t off = offset;
    buf.resize(offset + 8);
    // Placeholder: str_offset patched in Finalize(), str_len written now.
    uint32_t len = (uint32_t)s.size();
    memset(buf.data() + off, 0, 4);      // str_offset placeholder
    memcpy(buf.data() + off + 4, &len, 4);
    offset += 8;
    deferred_strings_.push_back({off, s});
    return off;
  }

  void Finalize() {
    for (auto &ds : deferred_strings_) {
      uint32_t str_start = (uint32_t)buf.size();
      buf.resize(buf.size() + ds.data.size());
      if (!ds.data.empty())
        memcpy(buf.data() + str_start, ds.data.data(), ds.data.size());
      memcpy(buf.data() + ds.slot_offset, &str_start, 4);
    }
    deferred_strings_.clear();
  }

private:
  struct DeferredString { uint32_t slot_offset; std::string data; };
  std::vector<DeferredString> deferred_strings_;
};

// ---------------------------------------------------------------------------
// Per-compilation context — holds the LLVM module and IR builder state
// ---------------------------------------------------------------------------
struct CompileCtx {
  LLVMContext &llctx;
  Module &mod;
  IRBuilder<> b;
  const std::vector<ColSchema> &schema;

  // LLVM struct types matching aqp_jit_abi.h
  StructType *AQPColViewTy;   // { i8*, i64*, i32, i32 }
  StructType *AQPChunkViewTy; // { AQPColView*, i64, i64 }
  StructType *AQPSelViewTy;   // { i32*, i32 }

  // Function arguments
  Value *chunk_arg; // AQPChunkView*
  Value *sel_arg;   // AQPSelView*

  // Per-row loop variables (set inside the loop body)
  Value *row_idx = nullptr; // i64 — current row index

  // Column data and validity pointers (loaded once before loop)
  std::vector<Value *> col_data;     // one void* per column
  std::vector<Value *> col_validity; // one i64* per column (may be null ptr)

  // Query-jit strict mode: wrap NULL-unguarded leaf predicates (integer
  // VarConst, all IN) so a NULL row value compares false instead of reading
  // garbage. Only correct in monotone AND/OR trees (BuildExecutionSteps rejects
  // NOT). Default off: pipeline-jit relies on DuckDB re-checking, and the
  // extra branch would cost it for no correctness gain.
  bool strict_null_guard = false;

  // §7.3 template cache mode: load constants from runtime params buffer.
  bool template_mode = false;
  Value *params_base = nullptr; // i8* to flat params buffer (loaded once)
  ParamsBuilder *params_builder = nullptr; // offset tracker during codegen

  CompileCtx(LLVMContext &ctx, Module &m, const std::vector<ColSchema> &s,
             Value *chunk, Value *sel)
      : llctx(ctx), mod(m), b(ctx), schema(s), chunk_arg(chunk), sel_arg(sel) {

    // Build struct types
    Type *i8p = PointerType::getUnqual(Type::getInt8Ty(ctx));
    Type *i32 = Type::getInt32Ty(ctx);
    Type *i64 = Type::getInt64Ty(ctx);
    Type *i64p = PointerType::getUnqual(i64);

    AQPColViewTy = StructType::get(ctx, {i8p, i64p, i32, i32});
    AQPChunkViewTy =
        StructType::get(ctx, {PointerType::getUnqual(AQPColViewTy), i64, i64});
    // sel.indices is sel_t* = uint32_t* in DuckDB (typedefs.hpp: typedef
    // uint32_t sel_t)
    AQPSelViewTy = StructType::get(ctx, {PointerType::getUnqual(i32), i32});
  }

  Type *i8p() { return PointerType::getUnqual(Type::getInt8Ty(llctx)); }
  Type *i32() { return Type::getInt32Ty(llctx); }
  Type *i64() { return Type::getInt64Ty(llctx); }
  Type *f32() { return Type::getFloatTy(llctx); }
  Type *f64() { return Type::getDoubleTy(llctx); }
  Type *i1() { return Type::getInt1Ty(llctx); }

  ConstantInt *c32(int32_t v) {
    return ConstantInt::get(llctx, APInt(32, (uint64_t)v, true));
  }
  ConstantInt *c64(int64_t v) {
    return ConstantInt::get(llctx, APInt(64, (uint64_t)v, true));
  }
  ConstantInt *ci(int v, int bits) {
    return ConstantInt::get(llctx, APInt(bits, v));
  }

  // §7.3 template mode: emit call to aqp_jit_get_params() to load
  // params_base from the thread-local (non-query-jit only).
  void LoadParamsFromThreadLocal() {
    FunctionCallee gp = mod.getOrInsertFunction(
        "aqp_jit_get_params",
        FunctionType::get(i8p(), {}, false));
    params_base = b.CreateCall(gp, {}, "params_tl");
  }

  // Load col data pointer for column col_chunk_idx (index into AQPChunkView)
  Value *LoadColData(unsigned col_chunk_idx) {
    // AQPChunkView.cols[col_chunk_idx].data
    Value *cols_ptr = b.CreateStructGEP(AQPChunkViewTy, chunk_arg, 0);
    Value *cols =
        b.CreateLoad(PointerType::getUnqual(AQPColViewTy), cols_ptr, "cols");
    Value *col_i =
        b.CreateGEP(AQPColViewTy, cols, c64((int64_t)col_chunk_idx), "col");
    Value *data_pp = b.CreateStructGEP(AQPColViewTy, col_i, 0);
    return b.CreateLoad(i8p(), data_pp, "data");
  }

  Value *LoadColValidity(unsigned col_chunk_idx) {
    Value *cols_ptr = b.CreateStructGEP(AQPChunkViewTy, chunk_arg, 0);
    Value *cols = b.CreateLoad(PointerType::getUnqual(AQPColViewTy), cols_ptr);
    Value *col_i = b.CreateGEP(AQPColViewTy, cols, c64((int64_t)col_chunk_idx));
    Value *val_pp = b.CreateStructGEP(AQPColViewTy, col_i, 1);
    return b.CreateLoad(PointerType::getUnqual(i64()), val_pp, "validity");
  }

  // Find the AQPChunkView column index for the given IR attribute.
  // Returns -1 if not found.
  int FindColIdx(const SimplestAttr &attr) const {
#ifndef NDEBUG
    std::cerr << "[AQP-JIT-TRACE] FindColIdx: IR attr (table="
              << attr.GetTableIndex() << ", col=" << attr.GetColumnIndex()
              << ", name=\"" << attr.GetColumnName()
              << "\") searching schema:\n";
    for (int i = 0; i < (int)schema.size(); i++) {
      std::cerr << "[AQP-JIT-TRACE]   schema[" << i
                << "] (table=" << schema[i].table_idx
                << ", col=" << schema[i].col_idx
                << ", dtype=" << schema[i].dtype << ")"
                << (schema[i].table_idx == attr.GetTableIndex() &&
                            schema[i].col_idx == attr.GetColumnIndex()
                        ? " MATCH"
                        : "")
                << "\n";
    }
#endif
    for (int i = 0; i < (int)schema.size(); i++) {
      if (schema[i].table_idx == attr.GetTableIndex() &&
          schema[i].col_idx == attr.GetColumnIndex())
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
  Value *bit_idx = cc.b.CreateAnd(cc.row_idx, cc.c64(63), "bit_idx");
  Value *word_ptr =
      cc.b.CreateGEP(cc.i64(), validity_ptr, word_idx, "word_ptr");
  Value *word = cc.b.CreateLoad(cc.i64(), word_ptr, "word");
  Value *shifted = cc.b.CreateLShr(word, bit_idx, "shifted");
  Value *bit = cc.b.CreateAnd(shifted, cc.c64(1), "bit");
  return cc.b.CreateICmpNE(bit, cc.c64(0), "valid");
}

// Strict NULL guard around a leaf predicate (cc.strict_null_guard): if the
// guarded column's value is NULL for this row, yield false WITHOUT executing
// `body` (whose data slot is undefined for NULL rows). Runtime branch on the
// validity pointer (nullptr = all valid), same shape as EmitVarConst's
// internal VARCHAR guard:
//   pre:  has_valvec ? chk : cmp
//   chk:  valid ? cmp : after        (false reaches the phi from chk_end)
//   cmp:  res = body()  -> after
//   after: phi(false from chk_end, res from cmp_end)
template <typename BodyFn>
static Value *EmitNullGuardedLeaf(CompileCtx &cc, int col_idx, BodyFn body) {
  Value *validity = (col_idx >= 0 && col_idx < (int)cc.col_validity.size())
                        ? cc.col_validity[col_idx]
                        : nullptr;
  if (!validity)
    return body();
  Function *fn = cc.b.GetInsertBlock()->getParent();
  BasicBlock *bb_chk = BasicBlock::Create(cc.llctx, "ng_chk", fn);
  BasicBlock *bb_cmp = BasicBlock::Create(cc.llctx, "ng_cmp", fn);
  BasicBlock *bb_after = BasicBlock::Create(cc.llctx, "ng_after", fn);
  Value *has_valvec = cc.b.CreateIsNotNull(validity, "ng_has_valvec");
  cc.b.CreateCondBr(has_valvec, bb_chk, bb_cmp);
  cc.b.SetInsertPoint(bb_chk);
  Value *valid = EmitValidityCheck(cc, validity);
  BasicBlock *chk_end = cc.b.GetInsertBlock();
  cc.b.CreateCondBr(valid, bb_cmp, bb_after);
  cc.b.SetInsertPoint(bb_cmp);
  Value *res = body();
  BasicBlock *cmp_end = cc.b.GetInsertBlock();
  cc.b.CreateBr(bb_after);
  cc.b.SetInsertPoint(bb_after);
  PHINode *phi = cc.b.CreatePHI(Type::getInt1Ty(cc.llctx), 2, "ng_res");
  phi->addIncoming(ConstantInt::getFalse(cc.llctx), chk_end);
  phi->addIncoming(res, cmp_end);
  return phi;
}

// Load an INT32 value from a flat column array at row index row_idx
static Value *LoadI32(CompileCtx &cc, Value *data_ptr) {
  Value *p32 = cc.b.CreateBitCast(data_ptr, PointerType::getUnqual(cc.i32()));
  Value *elem = cc.b.CreateGEP(cc.i32(), p32, cc.row_idx, "elem_ptr");
  return cc.b.CreateLoad(cc.i32(), elem, "val_i32");
}
static Value *LoadI64(CompileCtx &cc, Value *data_ptr) {
  Value *p64 = cc.b.CreateBitCast(data_ptr, PointerType::getUnqual(cc.i64()));
  Value *elem = cc.b.CreateGEP(cc.i64(), p64, cc.row_idx, "elem_ptr");
  return cc.b.CreateLoad(cc.i64(), elem, "val_i64");
}
static Value *LoadF32(CompileCtx &cc, Value *data_ptr) {
  Value *pf = cc.b.CreateBitCast(data_ptr, PointerType::getUnqual(cc.f32()));
  Value *elem = cc.b.CreateGEP(cc.f32(), pf, cc.row_idx, "elem_ptr");
  return cc.b.CreateLoad(cc.f32(), elem, "val_f32");
}
static Value *LoadF64(CompileCtx &cc, Value *data_ptr) {
  Value *pf = cc.b.CreateBitCast(data_ptr, PointerType::getUnqual(cc.f64()));
  Value *elem = cc.b.CreateGEP(cc.f64(), pf, cc.row_idx, "elem_ptr");
  return cc.b.CreateLoad(cc.f64(), elem, "val_f64");
}

enum LikePatternKind {
  LIKE_COMPLEX = 0,
  LIKE_EQUALITY,
  LIKE_PREFIX,
  LIKE_SUFFIX,
  LIKE_CONTAINS,
  LIKE_MULTI_SEGMENT
};

static LikePatternKind ClassifyLikePattern(const std::string &pattern,
                                           std::string &literal_out) {
  if (pattern.empty()) {
    literal_out.clear();
    return LIKE_EQUALITY;
  }
  if (pattern.find('_') != std::string::npos)
    return LIKE_COMPLEX;

  size_t leading = 0;
  while (leading < pattern.size() && pattern[leading] == '%')
    ++leading;
  size_t trailing = 0;
  while (trailing < pattern.size() &&
         pattern[pattern.size() - 1 - trailing] == '%')
    ++trailing;

  size_t mid_start = leading, mid_end = pattern.size() - trailing;
  for (size_t i = mid_start; i < mid_end; ++i)
    if (pattern[i] == '%')
      return LIKE_COMPLEX;

  literal_out = pattern.substr(mid_start, mid_end - mid_start);

  if (leading == 0 && trailing == 0)
    return LIKE_EQUALITY;
  if (leading == 0 && trailing > 0)
    return LIKE_PREFIX;
  if (leading > 0 && trailing == 0)
    return LIKE_SUFFIX;
  return LIKE_CONTAINS;
}

struct LikeSegments {
  std::vector<std::string> segs;
  bool has_leading_pct = false;
  bool has_trailing_pct = false;
};

static LikePatternKind ClassifyLikePatternEx(const std::string &pattern,
                                             std::string &literal_out,
                                             LikeSegments &seg_out) {
  LikePatternKind k = ClassifyLikePattern(pattern, literal_out);
  if (k != LIKE_COMPLEX) return k;

  if (pattern.find('_') != std::string::npos) return LIKE_COMPLEX;

  seg_out.has_leading_pct = (!pattern.empty() && pattern[0] == '%');
  seg_out.has_trailing_pct = (!pattern.empty() && pattern.back() == '%');

  seg_out.segs.clear();
  std::string cur;
  for (char c : pattern) {
    if (c == '%') {
      if (!cur.empty()) { seg_out.segs.push_back(cur); cur.clear(); }
    } else {
      cur += c;
    }
  }
  if (!cur.empty()) seg_out.segs.push_back(cur);

  if (seg_out.segs.size() >= 2) return LIKE_MULTI_SEGMENT;
  return LIKE_COMPLEX;
}

// §7.3 template-mode helpers: emit a load from the runtime params buffer
// instead of an LLVM immediate. The ParamsBuilder tracks offsets during
// codegen; at cache-hit time, BuildParamsBuffer re-walks the same expressions
// in the same order to produce a byte-identical layout.
static Value *EmitParamI32(CompileCtx &cc, int32_t v) {
  uint32_t off = cc.params_builder->AllocI32(v);
  Value *p = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), cc.params_base,
                             cc.c64(off));
  return cc.b.CreateLoad(
      cc.i32(), cc.b.CreateBitCast(p, PointerType::getUnqual(cc.i32())),
      "pi32");
}

static Value *EmitParamI64(CompileCtx &cc, int64_t v) {
  uint32_t off = cc.params_builder->AllocI64(v);
  Value *p = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), cc.params_base,
                             cc.c64(off));
  return cc.b.CreateLoad(
      cc.i64(), cc.b.CreateBitCast(p, PointerType::getUnqual(cc.i64())),
      "pi64");
}

static Value *EmitParamF32(CompileCtx &cc, float v) {
  uint32_t off = cc.params_builder->AllocF32(v);
  Value *p = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), cc.params_base,
                             cc.c64(off));
  return cc.b.CreateLoad(
      cc.f32(), cc.b.CreateBitCast(p, PointerType::getUnqual(cc.f32())),
      "pf32");
}

static Value *EmitParamF64(CompileCtx &cc, double v) {
  uint32_t off = cc.params_builder->AllocF64(v);
  Value *p = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), cc.params_base,
                             cc.c64(off));
  return cc.b.CreateLoad(
      cc.f64(), cc.b.CreateBitCast(p, PointerType::getUnqual(cc.f64())),
      "pf64");
}

// Returns {pat_ptr (i8*), pat_len (i32)} loaded from params buffer.
// Slot layout: {uint32_t str_offset, uint32_t str_len} at fixed position.
// Actual string bytes at params_base + str_offset (filled by Finalize).
static std::pair<Value *, Value *> EmitParamString(CompileCtx &cc,
                                                   const std::string &s) {
  uint32_t off = cc.params_builder->AllocString(s);
  Value *slot = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), cc.params_base,
                                cc.c64(off));
  Value *str_off = cc.b.CreateLoad(
      cc.i32(),
      cc.b.CreateBitCast(slot, PointerType::getUnqual(cc.i32())), "pstr_off");
  Value *len = cc.b.CreateLoad(
      cc.i32(),
      cc.b.CreateBitCast(
          cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), slot, cc.c64(4)),
          PointerType::getUnqual(cc.i32())),
      "pstr_len");
  Value *ptr = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), cc.params_base,
                               cc.b.CreateZExt(str_off, cc.b.getInt64Ty()),
                               "pstr_ptr");
  return {ptr, len};
}

// Emit comparison for VarConstComparison node.
// Returns i1 (true = row matches).
static Value *EmitVarConst(CompileCtx &cc,
                           const SimplestVarConstComparison *cmp) {
  int col_idx = cc.FindColIdx(*cmp->attr);
  if (col_idx < 0)
    throw std::runtime_error(
        "AQP-JIT unsupported: filter column '" + cmp->attr->GetColumnName() +
        "' not found in chunk schema (would pass all rows)");

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
      Value *p8 = cc.b.CreateBitCast(
          data, PointerType::getUnqual(Type::getInt8Ty(cc.llctx)));
      Value *ep = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), p8, cc.row_idx);
      lhs = cc.b.CreateLoad(Type::getInt8Ty(cc.llctx), ep);
      lhs = cc.b.CreateSExt(lhs, cc.i32());
    } else if (cs.dtype == AQP_DTYPE_INT16) {
      Value *p16 = cc.b.CreateBitCast(
          data, PointerType::getUnqual(Type::getInt16Ty(cc.llctx)));
      Value *ep = cc.b.CreateGEP(Type::getInt16Ty(cc.llctx), p16, cc.row_idx);
      lhs = cc.b.CreateLoad(Type::getInt16Ty(cc.llctx), ep);
      lhs = cc.b.CreateSExt(lhs, cc.i32());
    } else {
      lhs = LoadI32(cc, data);
    }

    int64_t rhs_raw = 0;
    if (cv->GetType() == SimplestVarType::IntVar ||
        cv->GetType() == SimplestVarType::Date)
      rhs_raw = (int64_t)cv->GetIntValue();
    else if (cv->GetType() == SimplestVarType::FloatVar)
      rhs_raw = (int64_t)cv->GetFloatValue();
    Value *rhs;
    if (cc.template_mode && cc.params_builder) {
      Value *r64 = EmitParamI64(cc, rhs_raw);
      rhs = cc.b.CreateTrunc(r64, cc.i32(), "pi32t");
    } else {
      rhs = cc.c32((int32_t)rhs_raw);
    }

    switch (op) {
    case SimplestExprType::Equal:
      return cc.b.CreateICmpEQ(lhs, rhs);
    case SimplestExprType::NotEqual:
      return cc.b.CreateICmpNE(lhs, rhs);
    case SimplestExprType::LessThan:
      return cc.b.CreateICmpSLT(lhs, rhs);
    case SimplestExprType::GreaterThan:
      return cc.b.CreateICmpSGT(lhs, rhs);
    case SimplestExprType::LessEqual:
      return cc.b.CreateICmpSLE(lhs, rhs);
    case SimplestExprType::GreaterEqual:
      return cc.b.CreateICmpSGE(lhs, rhs);
    default:
      return ConstantInt::getTrue(cc.llctx);
    }
  }

  if (cs.dtype == AQP_DTYPE_INT64) {
    Value *lhs = LoadI64(cc, data);
    int64_t rhs_raw = 0;
    if (cv->GetType() == SimplestVarType::IntVar ||
        cv->GetType() == SimplestVarType::Date)
      rhs_raw = (int64_t)cv->GetIntValue();
    else if (cv->GetType() == SimplestVarType::FloatVar)
      rhs_raw = (int64_t)cv->GetFloatValue();
    Value *rhs = (cc.template_mode && cc.params_builder)
                     ? EmitParamI64(cc, rhs_raw)
                     : cc.c64(rhs_raw);
    switch (op) {
    case SimplestExprType::Equal:
      return cc.b.CreateICmpEQ(lhs, rhs);
    case SimplestExprType::NotEqual:
      return cc.b.CreateICmpNE(lhs, rhs);
    case SimplestExprType::LessThan:
      return cc.b.CreateICmpSLT(lhs, rhs);
    case SimplestExprType::GreaterThan:
      return cc.b.CreateICmpSGT(lhs, rhs);
    case SimplestExprType::LessEqual:
      return cc.b.CreateICmpSLE(lhs, rhs);
    case SimplestExprType::GreaterEqual:
      return cc.b.CreateICmpSGE(lhs, rhs);
    default:
      return ConstantInt::getTrue(cc.llctx);
    }
  }

  if (cs.dtype == AQP_DTYPE_FLOAT) {
    Value *lhs = LoadF32(cc, data);
    float rhs_raw = (cv->GetType() == SimplestVarType::FloatVar)
                        ? (float)cv->GetFloatValue()
                        : 0.0f;
    Value *rhs;
    if (cc.template_mode && cc.params_builder) {
      Value *r64 = EmitParamF64(cc, (double)rhs_raw);
      rhs = cc.b.CreateFPTrunc(r64, cc.f32(), "pf32t");
    } else {
      rhs = ConstantFP::get(cc.f32(), rhs_raw);
    }
    switch (op) {
    case SimplestExprType::Equal:
      return cc.b.CreateFCmpOEQ(lhs, rhs);
    case SimplestExprType::NotEqual:
      return cc.b.CreateFCmpONE(lhs, rhs);
    case SimplestExprType::LessThan:
      return cc.b.CreateFCmpOLT(lhs, rhs);
    case SimplestExprType::GreaterThan:
      return cc.b.CreateFCmpOGT(lhs, rhs);
    case SimplestExprType::LessEqual:
      return cc.b.CreateFCmpOLE(lhs, rhs);
    case SimplestExprType::GreaterEqual:
      return cc.b.CreateFCmpOGE(lhs, rhs);
    default:
      return ConstantInt::getTrue(cc.llctx);
    }
  }

  if (cs.dtype == AQP_DTYPE_DOUBLE) {
    Value *lhs = LoadF64(cc, data);
    double rhs_raw = (cv->GetType() == SimplestVarType::FloatVar)
                         ? (double)cv->GetFloatValue()
                         : 0.0;
    Value *rhs = (cc.template_mode && cc.params_builder)
                     ? EmitParamF64(cc, rhs_raw)
                     : ConstantFP::get(cc.f64(), rhs_raw);
    switch (op) {
    case SimplestExprType::Equal:
      return cc.b.CreateFCmpOEQ(lhs, rhs);
    case SimplestExprType::NotEqual:
      return cc.b.CreateFCmpONE(lhs, rhs);
    case SimplestExprType::LessThan:
      return cc.b.CreateFCmpOLT(lhs, rhs);
    case SimplestExprType::GreaterThan:
      return cc.b.CreateFCmpOGT(lhs, rhs);
    case SimplestExprType::LessEqual:
      return cc.b.CreateFCmpOLE(lhs, rhs);
    case SimplestExprType::GreaterEqual:
      return cc.b.CreateFCmpOGE(lhs, rhs);
    default:
      return ConstantInt::getTrue(cc.llctx);
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
    BasicBlock *chk_bb = BasicBlock::Create(cc.llctx, "str_chk_null", fn);
    BasicBlock *cmp_bb = BasicBlock::Create(cc.llctx, "str_cmp", fn);
    BasicBlock *after_bb = BasicBlock::Create(cc.llctx, "str_after", fn);

    Value *null_vp =
        ConstantPointerNull::get(cast<PointerType>(validity_ptr->getType()));
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
    //   [8..15] char* ptr            (heap pointer, only valid when length >
    //   12) OR (inline, length <= 12): [4..15] char inlined[12]
    cc.b.SetInsertPoint(cmp_bb);
    Value *str_base = cc.b.CreateBitCast(data, cc.i8p());
    Value *str_offset = cc.b.CreateMul(cc.row_idx, cc.c64(16));
    Value *str_ptr = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_base,
                                    str_offset, "str_ptr");
    Value *len_ptr =
        cc.b.CreateBitCast(str_ptr, PointerType::getUnqual(cc.i32()));
    Value *slen = cc.b.CreateLoad(cc.i32(), len_ptr, "slen");
    Value *is_inline = cc.b.CreateICmpSLE(slen, cc.c32(12), "is_inline");
    Value *inline_ptr = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_ptr,
                                       cc.c64(4), "inline_ptr");
    Value *heap_pp = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_ptr,
                                    cc.c64(8), "heap_pp_raw");
    Value *heap_ppc =
        cc.b.CreateBitCast(heap_pp, PointerType::getUnqual(cc.i8p()));
    Value *heap_ptr = cc.b.CreateLoad(cc.i8p(), heap_ppc, "heap_ptr");
    Value *char_ptr =
        cc.b.CreateSelect(is_inline, inline_ptr, heap_ptr, "char_ptr");

    const std::string &pat = cv->GetStringValue();
    const bool is_like = (op == SimplestExprType::TextLike ||
                          op == SimplestExprType::Text_Not_Like);
    Value *pat_ptr = nullptr, *pat_len = nullptr;
    if (cc.template_mode && cc.params_builder && is_like) {
      // LIKE specialized paths allocate literal/segments themselves
      // in BuildParamsFromExpr; skip the raw pattern allocation here.
      // LIKE_EQUALITY and LIKE_COMPLEX still need pat_ptr/pat_len,
      // so those branches call EmitParamString on their own.
    } else if (cc.template_mode && cc.params_builder) {
      auto [pp, pl] = EmitParamString(cc, pat);
      pat_ptr = pp;
      pat_len = pl;
    } else {
      Constant *pat_const =
          ConstantDataArray::getString(cc.llctx, pat, /*AddNull=*/false);
      GlobalVariable *pat_gv =
          new GlobalVariable(cc.mod, pat_const->getType(), /*isConst=*/true,
                             GlobalValue::PrivateLinkage, pat_const, "pat");
      pat_ptr = cc.b.CreateBitCast(pat_gv, cc.i8p());
      pat_len = cc.c32((int32_t)pat.size());
    }

    FunctionType *ft4 = FunctionType::get(
        cc.i32(), {cc.i8p(), cc.i32(), cc.i8p(), cc.i32()}, false);
    Value *cmp_result;
    if (op == SimplestExprType::Equal || op == SimplestExprType::NotEqual) {
      // Length pre-filter: if lengths differ, skip byte comparison entirely.
      BasicBlock *pre_len_bb = cc.b.GetInsertBlock();
      BasicBlock *len_match_bb =
          BasicBlock::Create(cc.llctx, "len_match", pre_len_bb->getParent());
      BasicBlock *len_done_bb =
          BasicBlock::Create(cc.llctx, "len_done", pre_len_bb->getParent());
      Value *len_eq = cc.b.CreateICmpEQ(slen, pat_len, "len_eq");
      cc.b.CreateCondBr(len_eq, len_match_bb, len_done_bb);

      // Lengths match → inline memcmp for byte comparison
      cc.b.SetInsertPoint(len_match_bb);
      FunctionType *ft_memcmp = FunctionType::get(
          cc.i32(), {cc.i8p(), cc.i8p(), cc.i64()}, false);
      FunctionCallee memcmp_fn =
          cc.mod.getOrInsertFunction("memcmp", ft_memcmp);
      Value *pat_len_ext = cc.b.CreateSExt(pat_len, cc.i64());
      Value *mcr =
          cc.b.CreateCall(memcmp_fn, {char_ptr, pat_ptr, pat_len_ext});
      Value *m_match = cc.b.CreateICmpEQ(mcr, cc.c32(0));
      cc.b.CreateBr(len_done_bb);

      // Merge: lengths differ → false; lengths match → memcmp result
      cc.b.SetInsertPoint(len_done_bb);
      PHINode *m_phi = cc.b.CreatePHI(cc.i1(), 2, "streq_result");
      m_phi->addIncoming(ConstantInt::getFalse(cc.llctx), pre_len_bb);
      m_phi->addIncoming(m_match, len_match_bb);
      cmp_result =
          (op == SimplestExprType::NotEqual) ? cc.b.CreateNot(m_phi) : m_phi;
    } else if (op == SimplestExprType::TextLike ||
               op == SimplestExprType::Text_Not_Like) {
      {
      std::string literal;
      LikeSegments seg_info;
      LikePatternKind kind = ClassifyLikePatternEx(pat, literal, seg_info);
#ifndef NDEBUG
      static const char *kind_names[] = {"COMPLEX", "EQUALITY", "PREFIX",
                                         "SUFFIX", "CONTAINS", "MULTI_SEGMENT"};
      std::cerr << "[AQP-JIT-TRACE] EmitVarConst: compiling "
                << (op == SimplestExprType::Text_Not_Like ? "Text_Not_Like"
                                                          : "TextLike")
                << " pattern=\"" << pat << "\" → " << kind_names[kind] << "\n";
#endif

      if (kind == LIKE_EQUALITY) {
        // No wildcards: reuse aqp_str_eq with length pre-filter.
        if (cc.template_mode && cc.params_builder) {
          auto [pp, pl] = EmitParamString(cc, pat);
          pat_ptr = pp;
          pat_len = pl;
        }
        BasicBlock *pre_bb = cc.b.GetInsertBlock();
        Function *fn = pre_bb->getParent();
        BasicBlock *match_bb =
            BasicBlock::Create(cc.llctx, "like_eq_match", fn);
        BasicBlock *done_bb =
            BasicBlock::Create(cc.llctx, "like_eq_done", fn);
        Value *len_eq = cc.b.CreateICmpEQ(slen, pat_len, "like_eq_len");
        cc.b.CreateCondBr(len_eq, match_bb, done_bb);

        cc.b.SetInsertPoint(match_bb);
        FunctionCallee callee =
            cc.mod.getOrInsertFunction("aqp_str_eq", ft4);
        Value *r =
            cc.b.CreateCall(callee, {char_ptr, slen, pat_ptr, pat_len});
        Value *m_match = cc.b.CreateICmpNE(r, cc.c32(0));
        cc.b.CreateBr(done_bb);

        cc.b.SetInsertPoint(done_bb);
        PHINode *phi = cc.b.CreatePHI(cc.i1(), 2, "like_eq_result");
        phi->addIncoming(ConstantInt::getFalse(cc.llctx), pre_bb);
        phi->addIncoming(m_match, match_bb);
        cmp_result = (op == SimplestExprType::Text_Not_Like)
                         ? cc.b.CreateNot(phi)
                         : phi;

      } else if (kind == LIKE_PREFIX || kind == LIKE_SUFFIX) {
        Value *lit_ptr, *lit_len;
        if (cc.template_mode && cc.params_builder) {
          auto [lp, ll] = EmitParamString(cc, literal);
          lit_ptr = lp;
          lit_len = ll;
        } else {
          Constant *lit_const = ConstantDataArray::getString(
              cc.llctx, literal, /*AddNull=*/false);
          GlobalVariable *lit_gv = new GlobalVariable(
              cc.mod, lit_const->getType(), /*isConst=*/true,
              GlobalValue::PrivateLinkage, lit_const, "like_lit");
          lit_ptr = cc.b.CreateBitCast(lit_gv, cc.i8p());
          lit_len = cc.c32((int32_t)literal.size());
        }

        BasicBlock *pre_bb = cc.b.GetInsertBlock();
        Function *fn = pre_bb->getParent();
        BasicBlock *cmp_bb_like =
            BasicBlock::Create(cc.llctx, "like_cmp", fn);
        BasicBlock *done_bb =
            BasicBlock::Create(cc.llctx, "like_done", fn);

        Value *len_ok =
            cc.b.CreateICmpSGE(slen, lit_len, "like_len_ok");
        cc.b.CreateCondBr(len_ok, cmp_bb_like, done_bb);

        cc.b.SetInsertPoint(cmp_bb_like);
        Value *cmp_ptr;
        if (kind == LIKE_PREFIX) {
          cmp_ptr = char_ptr;
        } else {
          Value *offset = cc.b.CreateSub(slen, lit_len, "suffix_off");
          Value *offset_ext =
              cc.b.CreateSExt(offset, cc.i64(), "suffix_off_ext");
          cmp_ptr = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), char_ptr,
                                   offset_ext, "suffix_ptr");
        }
        FunctionType *ft_memcmp = FunctionType::get(
            cc.i32(), {cc.i8p(), cc.i8p(), cc.i64()}, false);
        FunctionCallee memcmp_fn =
            cc.mod.getOrInsertFunction("memcmp", ft_memcmp);
        Value *lit_len_ext =
            cc.b.CreateSExt(lit_len, cc.i64(), "lit_len_ext");
        Value *mcr =
            cc.b.CreateCall(memcmp_fn, {cmp_ptr, lit_ptr, lit_len_ext});
        Value *m_match = cc.b.CreateICmpEQ(mcr, cc.c32(0), "like_match");
        cc.b.CreateBr(done_bb);

        cc.b.SetInsertPoint(done_bb);
        PHINode *phi = cc.b.CreatePHI(cc.i1(), 2, "like_result");
        phi->addIncoming(ConstantInt::getFalse(cc.llctx), pre_bb);
        phi->addIncoming(m_match, cmp_bb_like);
        cmp_result = (op == SimplestExprType::Text_Not_Like)
                         ? cc.b.CreateNot(phi)
                         : phi;

      } else if (kind == LIKE_CONTAINS) {
        Value *lit_ptr;
        Value *needle_len_v;
        Value *first_char;
        if (cc.template_mode && cc.params_builder) {
          auto [lp, ll] = EmitParamString(cc, literal);
          lit_ptr = lp;
          needle_len_v = ll;
          first_char = EmitParamI32(cc, (int32_t)(unsigned char)literal[0]);
        } else {
          Constant *lit_const = ConstantDataArray::getString(
              cc.llctx, literal, /*AddNull=*/false);
          GlobalVariable *lit_gv = new GlobalVariable(
              cc.mod, lit_const->getType(), /*isConst=*/true,
              GlobalValue::PrivateLinkage, lit_const, "like_lit");
          lit_ptr = cc.b.CreateBitCast(lit_gv, cc.i8p());
          needle_len_v = cc.c32((int32_t)literal.size());
          first_char = cc.c32((int32_t)(unsigned char)literal[0]);
        }

        BasicBlock *pre_bb = cc.b.GetInsertBlock();
        Function *fn = pre_bb->getParent();
        BasicBlock *search_bb =
            BasicBlock::Create(cc.llctx, "ct_search", fn);
        BasicBlock *found_bb =
            BasicBlock::Create(cc.llctx, "ct_found", fn);
        BasicBlock *notfound_bb =
            BasicBlock::Create(cc.llctx, "ct_notfound", fn);
        BasicBlock *done_bb =
            BasicBlock::Create(cc.llctx, "ct_done", fn);

        Value *len_ok =
            cc.b.CreateICmpSGE(slen, needle_len_v, "ct_len_ok");
        cc.b.CreateCondBr(len_ok, search_bb, notfound_bb);

        cc.b.SetInsertPoint(search_bb);
        Value *needle_minus_1 = cc.b.CreateSub(needle_len_v, cc.c32(1),
                                               "needle_m1");
        Value *search_end =
            cc.b.CreateSub(slen, needle_minus_1, "ct_end");
        Value *init_off = cc.c32(0);

        BasicBlock *loop_bb =
            BasicBlock::Create(cc.llctx, "ct_loop", fn);
        cc.b.CreateBr(loop_bb);
        cc.b.SetInsertPoint(loop_bb);
        PHINode *off_phi = cc.b.CreatePHI(cc.i32(), 2, "ct_off");
        off_phi->addIncoming(init_off, search_bb);

        Value *off_ext =
            cc.b.CreateSExt(off_phi, cc.i64(), "off_ext");
        Value *hay_ptr = cc.b.CreateGEP(
            Type::getInt8Ty(cc.llctx), char_ptr, off_ext, "hay_ptr");
        Value *remain =
            cc.b.CreateSub(search_end, off_phi, "ct_remain");
        Value *remain_ext =
            cc.b.CreateZExt(remain, cc.i64(), "remain_ext");

        FunctionType *ft_memchr = FunctionType::get(
            cc.i8p(), {cc.i8p(), cc.i32(), cc.i64()}, false);
        FunctionCallee memchr_fn =
            cc.mod.getOrInsertFunction("memchr", ft_memchr);
        Value *loc = cc.b.CreateCall(
            memchr_fn, {hay_ptr, first_char, remain_ext}, "ct_loc");

        Value *loc_null = cc.b.CreateICmpEQ(
            loc,
            ConstantPointerNull::get(
                cast<PointerType>(cc.i8p())),
            "ct_null");
        BasicBlock *check_bb =
            BasicBlock::Create(cc.llctx, "ct_check", fn);
        cc.b.CreateCondBr(loc_null, notfound_bb, check_bb);

        cc.b.SetInsertPoint(check_bb);
        Value *is_single = cc.b.CreateICmpEQ(needle_len_v, cc.c32(1),
                                             "ct_single");
        BasicBlock *single_bb =
            BasicBlock::Create(cc.llctx, "ct_single_found", fn);
        BasicBlock *multi_bb =
            BasicBlock::Create(cc.llctx, "ct_multi", fn);
        cc.b.CreateCondBr(is_single, single_bb, multi_bb);

        cc.b.SetInsertPoint(single_bb);
        cc.b.CreateBr(found_bb);

        cc.b.SetInsertPoint(multi_bb);
        {
          FunctionType *ft_mc = FunctionType::get(
              cc.i32(), {cc.i8p(), cc.i8p(), cc.i64()}, false);
          FunctionCallee mc_fn =
              cc.mod.getOrInsertFunction("memcmp", ft_mc);
          Value *needle_len_ext =
              cc.b.CreateSExt(needle_len_v, cc.i64(), "nlen_ext");
          Value *mcr = cc.b.CreateCall(
              mc_fn, {loc, lit_ptr, needle_len_ext});
          Value *eq = cc.b.CreateICmpEQ(mcr, cc.c32(0), "ct_eq");

          BasicBlock *next_bb =
              BasicBlock::Create(cc.llctx, "ct_next", fn);
          cc.b.CreateCondBr(eq, found_bb, next_bb);

          cc.b.SetInsertPoint(next_bb);
          Value *loc_int = cc.b.CreatePtrToInt(loc, cc.i64());
          Value *base_int = cc.b.CreatePtrToInt(char_ptr, cc.i64());
          Value *new_off_64 =
              cc.b.CreateSub(loc_int, base_int);
          Value *new_off =
              cc.b.CreateTrunc(new_off_64, cc.i32());
          Value *next_off =
              cc.b.CreateAdd(new_off, cc.c32(1), "ct_next_off");
          Value *in_range = cc.b.CreateICmpSLT(
              next_off, search_end, "ct_inrange");
          off_phi->addIncoming(next_off, next_bb);
          cc.b.CreateCondBr(in_range, loop_bb, notfound_bb);
        }

        cc.b.SetInsertPoint(found_bb);
        cc.b.CreateBr(done_bb);
        cc.b.SetInsertPoint(notfound_bb);
        cc.b.CreateBr(done_bb);

        cc.b.SetInsertPoint(done_bb);
        PHINode *phi = cc.b.CreatePHI(cc.i1(), 2, "ct_result");
        phi->addIncoming(ConstantInt::getTrue(cc.llctx), found_bb);
        phi->addIncoming(ConstantInt::getFalse(cc.llctx), notfound_bb);
        cmp_result = (op == SimplestExprType::Text_Not_Like)
                         ? cc.b.CreateNot(phi)
                         : phi;

      } else if (kind == LIKE_MULTI_SEGMENT) {
        int32_t n_segs = (int32_t)seg_info.segs.size();

        if (cc.template_mode && cc.params_builder) {
          // Build seg_ptrs/seg_lens arrays on the stack from params buffer.
          Value *ptrs_alloca = cc.b.CreateAlloca(cc.i8p(), cc.c32(n_segs),
                                                 "seg_ptrs_a");
          Value *lens_alloca = cc.b.CreateAlloca(cc.i32(), cc.c32(n_segs),
                                                 "seg_lens_a");
          for (int si = 0; si < n_segs; ++si) {
            auto [sp, sl] = EmitParamString(cc, seg_info.segs[si]);
            cc.b.CreateStore(
                sp, cc.b.CreateGEP(cc.i8p(), ptrs_alloca, cc.c32(si)));
            cc.b.CreateStore(
                sl, cc.b.CreateGEP(cc.i32(), lens_alloca, cc.c32(si)));
          }
          FunctionType *ft_seg = FunctionType::get(
              cc.i32(),
              {cc.i8p(), cc.i32(),
               PointerType::getUnqual(cc.i8p()),
               PointerType::getUnqual(cc.i32()),
               cc.i32(), cc.i32(), cc.i32()},
              false);
          FunctionCallee callee = cc.mod.getOrInsertFunction(
              "aqp_like_match_segments", ft_seg);
          Value *r = cc.b.CreateCall(
              callee,
              {char_ptr, slen, ptrs_alloca, lens_alloca, cc.c32(n_segs),
               cc.c32(seg_info.has_leading_pct ? 1 : 0),
               cc.c32(seg_info.has_trailing_pct ? 1 : 0)});
          Value *m = cc.b.CreateICmpNE(r, cc.c32(0));
          cmp_result = (op == SimplestExprType::Text_Not_Like)
                           ? cc.b.CreateNot(m)
                           : m;
        } else {
          std::vector<Constant *> seg_ptrs, seg_lens_vals;
          for (auto &seg : seg_info.segs) {
            Constant *sc = ConstantDataArray::getString(
                cc.llctx, seg, /*AddNull=*/false);
            GlobalVariable *sg = new GlobalVariable(
                cc.mod, sc->getType(), /*isConst=*/true,
                GlobalValue::PrivateLinkage, sc, "like_seg");
            seg_ptrs.push_back(
                ConstantExpr::getBitCast(sg, cc.i8p()));
            seg_lens_vals.push_back(
                cc.b.getInt32((int32_t)seg.size()));
          }

          ArrayType *ptr_arr_ty = ArrayType::get(cc.i8p(), n_segs);
          Constant *ptr_arr =
              ConstantArray::get(ptr_arr_ty, seg_ptrs);
          GlobalVariable *ptr_gv = new GlobalVariable(
              cc.mod, ptr_arr_ty, /*isConst=*/true,
              GlobalValue::PrivateLinkage, ptr_arr, "seg_ptrs");

          ArrayType *len_arr_ty = ArrayType::get(cc.i32(), n_segs);
          Constant *len_arr =
              ConstantArray::get(len_arr_ty, seg_lens_vals);
          GlobalVariable *len_gv = new GlobalVariable(
              cc.mod, len_arr_ty, /*isConst=*/true,
              GlobalValue::PrivateLinkage, len_arr, "seg_lens");

          Value *ptrs_ptr = cc.b.CreateBitCast(
              ptr_gv, PointerType::getUnqual(cc.i8p()));
          Value *lens_ptr = cc.b.CreateBitCast(
              len_gv, PointerType::getUnqual(cc.i32()));

          FunctionType *ft_seg = FunctionType::get(
              cc.i32(),
              {cc.i8p(), cc.i32(),
               PointerType::getUnqual(cc.i8p()),
               PointerType::getUnqual(cc.i32()),
               cc.i32(), cc.i32(), cc.i32()},
              false);
          FunctionCallee callee = cc.mod.getOrInsertFunction(
              "aqp_like_match_segments", ft_seg);
          Value *r = cc.b.CreateCall(
              callee,
              {char_ptr, slen, ptrs_ptr, lens_ptr, cc.c32(n_segs),
               cc.c32(seg_info.has_leading_pct ? 1 : 0),
               cc.c32(seg_info.has_trailing_pct ? 1 : 0)});
          Value *m = cc.b.CreateICmpNE(r, cc.c32(0));
          cmp_result = (op == SimplestExprType::Text_Not_Like)
                           ? cc.b.CreateNot(m)
                           : m;
        }

      } else {
        // LIKE_COMPLEX: fallback to generic backtracking matcher
        if (cc.template_mode && cc.params_builder) {
          auto [pp, pl] = EmitParamString(cc, pat);
          pat_ptr = pp;
          pat_len = pl;
        }
        FunctionCallee callee =
            cc.mod.getOrInsertFunction("aqp_like_match", ft4);
        Value *r =
            cc.b.CreateCall(callee, {char_ptr, slen, pat_ptr, pat_len});
        Value *m = cc.b.CreateICmpNE(r, cc.c32(0));
        cmp_result = (op == SimplestExprType::Text_Not_Like)
                         ? cc.b.CreateNot(m)
                         : m;
      }
      }
    } else if (op == SimplestExprType::GreaterEqual ||
               op == SimplestExprType::LessEqual ||
               op == SimplestExprType::GreaterThan ||
               op == SimplestExprType::LessThan) {
      // Lexicographic ordering via aqp_str_cmp (returns <0, 0, or >0)
      FunctionType *ft_cmp = FunctionType::get(
          cc.i32(), {cc.i8p(), cc.i32(), cc.i8p(), cc.i32()}, false);
      FunctionCallee callee = cc.mod.getOrInsertFunction("aqp_str_cmp", ft_cmp);
      Value *r = cc.b.CreateCall(callee, {char_ptr, slen, pat_ptr, pat_len});
      Value *zero = cc.c32(0);
      switch (op) {
      case SimplestExprType::GreaterEqual:
        cmp_result = cc.b.CreateICmpSGE(r, zero);
        break;
      case SimplestExprType::LessEqual:
        cmp_result = cc.b.CreateICmpSLE(r, zero);
        break;
      case SimplestExprType::GreaterThan:
        cmp_result = cc.b.CreateICmpSGT(r, zero);
        break;
      case SimplestExprType::LessThan:
        cmp_result = cc.b.CreateICmpSLT(r, zero);
        break;
      default:
        cmp_result = ConstantInt::getTrue(cc.llctx);
        break;
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

// Emit col-vs-col comparison on the same source (integer family only).
// Non-integer or unresolvable columns pass all rows, matching EmitVarConst's
// fallback convention; the qjit whitelist (CheckFilterStrict) guarantees
// INT32 on both sides before this is reached from CompileQuerySteps.
static Value *EmitVarVar(CompileCtx &cc, const SimplestVarComparison *cmp) {
  if (!cmp->left_attr || !cmp->right_attr)
    return ConstantInt::getTrue(cc.llctx);
  int li = cc.FindColIdx(*cmp->left_attr);
  int ri = cc.FindColIdx(*cmp->right_attr);
  if (li < 0 || ri < 0)
    return ConstantInt::getTrue(cc.llctx);

  auto load_as_i64 = [&](int ci) -> Value * {
    const ColSchema &cs = cc.schema[ci];
    Value *data = cc.col_data[ci];
    switch (cs.dtype) {
    case AQP_DTYPE_INT8:
    case AQP_DTYPE_BOOL: {
      Value *p8 = cc.b.CreateBitCast(
          data, PointerType::getUnqual(Type::getInt8Ty(cc.llctx)));
      Value *ep = cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), p8, cc.row_idx);
      return cc.b.CreateSExt(cc.b.CreateLoad(Type::getInt8Ty(cc.llctx), ep),
                             cc.i64());
    }
    case AQP_DTYPE_INT16: {
      Value *p16 = cc.b.CreateBitCast(
          data, PointerType::getUnqual(Type::getInt16Ty(cc.llctx)));
      Value *ep = cc.b.CreateGEP(Type::getInt16Ty(cc.llctx), p16, cc.row_idx);
      return cc.b.CreateSExt(cc.b.CreateLoad(Type::getInt16Ty(cc.llctx), ep),
                             cc.i64());
    }
    case AQP_DTYPE_INT32:
    case AQP_DTYPE_DATE:
      return cc.b.CreateSExt(LoadI32(cc, data), cc.i64());
    case AQP_DTYPE_INT64:
      return LoadI64(cc, data);
    default:
      return nullptr;
    }
  };

  Value *lhs = load_as_i64(li);
  Value *rhs = load_as_i64(ri);
  if (!lhs || !rhs)
    return ConstantInt::getTrue(cc.llctx);

  switch (cmp->GetSimplestExprType()) {
  case SimplestExprType::Equal:
    return cc.b.CreateICmpEQ(lhs, rhs);
  case SimplestExprType::NotEqual:
    return cc.b.CreateICmpNE(lhs, rhs);
  case SimplestExprType::LessThan:
    return cc.b.CreateICmpSLT(lhs, rhs);
  case SimplestExprType::GreaterThan:
    return cc.b.CreateICmpSGT(lhs, rhs);
  case SimplestExprType::LessEqual:
    return cc.b.CreateICmpSLE(lhs, rhs);
  case SimplestExprType::GreaterEqual:
    return cc.b.CreateICmpSGE(lhs, rhs);
  default:
    return ConstantInt::getTrue(cc.llctx);
  }
}

// Emit IS NULL / IS NOT NULL check
static Value *EmitIsNull(CompileCtx &cc, const SimplestIsNullExpr *expr) {
  int col_idx = cc.FindColIdx(*expr->attr);
  // Column not in schema: predicate doesn't apply to this filter → pass all
  // rows. (same semantics as EmitVarConst when col_idx < 0)
  if (col_idx < 0)
    return ConstantInt::getTrue(cc.llctx);

  Value *validity_ptr = cc.col_validity[col_idx];
  // If validity_ptr is null (all-valid), IS NULL is always false
  Value *null_ptr =
      ConstantPointerNull::get(cast<PointerType>(validity_ptr->getType()));
  Value *has_nulls = cc.b.CreateICmpNE(validity_ptr, null_ptr, "has_nulls");

  // Compute the validity bit
  Function *fn = cc.b.GetInsertBlock()->getParent();
  BasicBlock *pre_bb = cc.b.GetInsertBlock(); // block before the branch
  BasicBlock *check_bb = BasicBlock::Create(cc.llctx, "check_valid", fn);
  BasicBlock *merge_bb = BasicBlock::Create(cc.llctx, "merge_valid", fn);

  cc.b.CreateCondBr(has_nulls, check_bb, merge_bb);

  cc.b.SetInsertPoint(check_bb);
  Value *is_valid_inner = EmitValidityCheck(cc, validity_ptr);
  BasicBlock *check_end = cc.b.GetInsertBlock();
  cc.b.CreateBr(merge_bb);

  cc.b.SetInsertPoint(merge_bb);
  PHINode *is_valid = cc.b.CreatePHI(cc.i1(), 2, "is_valid");
  // When has_nulls==false (all-valid), jump from pre_bb → merge_bb: row is
  // valid
  is_valid->addIncoming(ConstantInt::getTrue(cc.llctx), pre_bb);
  is_valid->addIncoming(is_valid_inner, check_end);

  // IS NULL = !is_valid; IS NOT NULL = is_valid
  bool is_null_check =
      (expr->GetSimplestExprType() == SimplestExprType::NullType);
  return is_null_check ? cc.b.CreateNot(is_valid) : is_valid;
}

// Returns true if `expr` references at least one column present in the schema.
// Used to detect pass-through expressions so that NOT(pass-through) stays
// pass-through.
static bool ExprInvolvesSchema(const CompileCtx &cc, const AQPExpr *expr) {
  if (!expr)
    return false;
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

    // NOT(LIKE) → Text_Not_Like: keep the inversion INSIDE EmitVarConst's
    // NULL guard so that NULL rows stay excluded (NOT(NULL→false) must remain
    // false, not become true).  EmitVarConst applies the NOT before the
    // NULL-guard PHI merge, giving correct three-valued SQL semantics.
    if (expr->right_expr &&
        expr->right_expr->GetNodeType() ==
            ir_sql_converter::SimplestNodeType::VarConstComparisonNode) {
      auto *cmp =
          static_cast<const ir_sql_converter::SimplestVarConstComparison *>(
              expr->right_expr.get());
      if (cmp->GetSimplestExprType() == SimplestExprType::TextLike) {
        // Re-emit as Text_Not_Like (NOT applied inside NULL guard)
        auto not_like =
            std::make_unique<ir_sql_converter::SimplestVarConstComparison>(
                SimplestExprType::Text_Not_Like,
                std::make_unique<ir_sql_converter::SimplestAttr>(*cmp->attr),
                std::make_unique<ir_sql_converter::SimplestConstVar>(
                    *cmp->const_var));
        return EmitVarConst(cc, not_like.get());
      }
    }

    Value *child = EmitExpr(cc, expr->right_expr.get());
    return cc.b.CreateNot(child, "not");
  }

  // Short-circuit AND / OR using basic blocks
  Function *fn = cc.b.GetInsertBlock()->getParent();
  BasicBlock *lhs_bb = cc.b.GetInsertBlock();
  BasicBlock *rhs_bb = BasicBlock::Create(cc.llctx, "logical_rhs", fn);
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

// Build the bucket layout for a hash-directory IN-set: perm is the
// bucket-ordered permutation of value indexes, dir the (mask+2) prefix-sum
// offsets. Uses the same aqp_in_hash_* mixers the runtime probe uses.
static void BuildInHashDir(const std::vector<uint64_t> &hashes,
                           std::vector<int> &perm, std::vector<int32_t> &dir,
                           int32_t &mask) {
  size_t n = hashes.size();
  uint32_t d = 1;
  while (d < 2 * n)
    d <<= 1;
  mask = (int32_t)(d - 1);
  std::vector<std::vector<int>> buckets(d);
  for (size_t i = 0; i < n; i++)
    buckets[hashes[i] & (uint64_t)mask].push_back((int)i);
  perm.clear();
  perm.reserve(n);
  dir.assign(d + 1, 0);
  for (uint32_t b = 0; b < d; b++) {
    dir[b] = (int32_t)perm.size();
    for (int idx : buckets[b])
      perm.push_back(idx);
  }
  dir[d] = (int32_t)perm.size();
}

static GlobalVariable *EmitI32ArrayGlobal(CompileCtx &cc,
                                          const std::vector<int32_t> &vals,
                                          const char *name) {
  std::vector<Constant *> consts;
  consts.reserve(vals.size());
  for (int32_t v : vals)
    consts.push_back(cc.c32(v));
  ArrayType *ty = ArrayType::get(cc.i32(), consts.size());
  return new GlobalVariable(cc.mod, ty, true, GlobalValue::PrivateLinkage,
                            ConstantArray::get(ty, consts), name);
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
      cs.dtype == AQP_DTYPE_INT8 || cs.dtype == AQP_DTYPE_DATE) {

    Value *lhs;
    if (cs.dtype == AQP_DTYPE_INT8) {
      Value *p8 = cc.b.CreateBitCast(
          data, PointerType::getUnqual(Type::getInt8Ty(cc.llctx)));
      lhs = cc.b.CreateLoad(
          Type::getInt8Ty(cc.llctx),
          cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), p8, cc.row_idx));
      lhs = cc.b.CreateSExt(lhs, cc.i32());
    } else {
      lhs = LoadI32(cc, data);
    }

    if (vals.size() <= 8) {
      Value *any = ConstantInt::getFalse(cc.llctx);
      for (const auto &v : vals) {
        int32_t iv = (int32_t)v->GetIntValue();
        Value *rv;
        if (cc.template_mode && cc.params_builder) {
          Value *r64 = EmitParamI64(cc, (int64_t)iv);
          rv = cc.b.CreateTrunc(r64, cc.i32(), "pin_i32");
        } else {
          rv = cc.c32(iv);
        }
        Value *match = cc.b.CreateICmpEQ(lhs, rv);
        any = cc.b.CreateOr(any, match);
      }
      return expr->negated ? cc.b.CreateNot(any) : any;
    } else {
      // Large INT32 IN-set: bucket-ordered constant array + hash directory,
      // probed by aqp_in_hash_i32 (O(1) per row vs the old linear scan).
      std::vector<int32_t> raw;
      std::vector<uint64_t> hashes;
      for (const auto &v : vals) {
        int32_t iv = (int32_t)v->GetIntValue();
        raw.push_back(iv);
        hashes.push_back(::aqp_in_hash_mix64((uint64_t)(uint32_t)iv));
      }
      std::vector<int> perm;
      std::vector<int32_t> dir;
      int32_t mask = 0;
      BuildInHashDir(hashes, perm, dir, mask);
      std::vector<int32_t> ordered;
      ordered.reserve(raw.size());
      for (int idx : perm)
        ordered.push_back(raw[idx]);
      GlobalVariable *vals_gv = EmitI32ArrayGlobal(cc, ordered, "in_vals");
      GlobalVariable *dir_gv = EmitI32ArrayGlobal(cc, dir, "in_dir");
      Value *vals_ptr =
          cc.b.CreateBitCast(vals_gv, PointerType::getUnqual(cc.i32()));
      Value *dir_ptr =
          cc.b.CreateBitCast(dir_gv, PointerType::getUnqual(cc.i32()));
      FunctionType *ft = FunctionType::get(
          cc.i32(),
          {cc.i32(), PointerType::getUnqual(cc.i32()),
           PointerType::getUnqual(cc.i32()), cc.i32()},
          false);
      FunctionCallee callee = cc.mod.getOrInsertFunction("aqp_in_hash_i32", ft);
      Value *result =
          cc.b.CreateCall(callee, {lhs, vals_ptr, dir_ptr, cc.c32(mask)});
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
        Value *rv = (cc.template_mode && cc.params_builder)
                        ? EmitParamI64(cc, iv)
                        : cc.c64(iv);
        Value *match = cc.b.CreateICmpEQ(lhs, rv);
        any = cc.b.CreateOr(any, match);
      }
      return expr->negated ? cc.b.CreateNot(any) : any;
    }
    // Large INT64 IN-set: hash directory, probed by aqp_in_hash_i64
    std::vector<int64_t> raw;
    std::vector<uint64_t> hashes;
    for (const auto &v : vals) {
      int64_t iv = (int64_t)v->GetIntValue();
      raw.push_back(iv);
      hashes.push_back(::aqp_in_hash_mix64((uint64_t)iv));
    }
    std::vector<int> perm;
    std::vector<int32_t> dir;
    int32_t mask = 0;
    BuildInHashDir(hashes, perm, dir, mask);
    std::vector<Constant *> consts;
    consts.reserve(raw.size());
    for (int idx : perm)
      consts.push_back(cc.c64(raw[idx]));
    ArrayType *arr_ty = ArrayType::get(cc.i64(), consts.size());
    GlobalVariable *gv = new GlobalVariable(
        cc.mod, arr_ty, true, GlobalValue::PrivateLinkage,
        ConstantArray::get(arr_ty, consts), "in_vals_i64");
    GlobalVariable *dir_gv = EmitI32ArrayGlobal(cc, dir, "in_dir_i64");
    Value *arr_ptr = cc.b.CreateBitCast(gv, PointerType::getUnqual(cc.i64()));
    Value *dir_ptr =
        cc.b.CreateBitCast(dir_gv, PointerType::getUnqual(cc.i32()));
    FunctionType *ft = FunctionType::get(
        cc.i32(),
        {cc.i64(), PointerType::getUnqual(cc.i64()),
         PointerType::getUnqual(cc.i32()), cc.i32()},
        false);
    FunctionCallee callee = cc.mod.getOrInsertFunction("aqp_in_hash_i64", ft);
    Value *result =
        cc.b.CreateCall(callee, {lhs, arr_ptr, dir_ptr, cc.c32(mask)});
    Value *match = cc.b.CreateICmpNE(result, cc.c32(0));
    return expr->negated ? cc.b.CreateNot(match) : match;
  }

  // VARCHAR IN: inline length-check + memcmp OR-chain for small sets,
  // fall back to aqp_in_set_str for large sets.
  if (cs.dtype == AQP_DTYPE_VARCHAR && col_idx >= 0) {
    // Load VARCHAR value from column (DuckDB string_t extraction)
    Value *data = cc.col_data[col_idx];
    Value *str_t_ptr = cc.b.CreateBitCast(data, cc.i8p());
    Value *row_offset =
        cc.b.CreateMul(cc.row_idx, cc.c64(16)); // string_t = 16 bytes
    Value *str_t_base =
        cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_t_ptr, row_offset);
    Value *len_ptr =
        cc.b.CreateBitCast(str_t_base, PointerType::getUnqual(cc.i32()));
    Value *slen = cc.b.CreateLoad(cc.i32(), len_ptr, "slen");
    Value *inline_ptr =
        cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_t_base, cc.c64(4));
    Value *heap_ptr_ptr = cc.b.CreateBitCast(
        cc.b.CreateGEP(Type::getInt8Ty(cc.llctx), str_t_base, cc.c64(8)),
        PointerType::getUnqual(cc.i8p()));
    Value *heap_ptr = cc.b.CreateLoad(cc.i8p(), heap_ptr_ptr);
    Value *is_inline = cc.b.CreateICmpULE(slen, cc.c32(12));
    Value *char_ptr = cc.b.CreateSelect(is_inline, inline_ptr, heap_ptr);

    if (vals.size() <= 8) {
      FunctionType *ft_memcmp = FunctionType::get(
          cc.i32(), {cc.i8p(), cc.i8p(), cc.i64()}, false);
      FunctionCallee memcmp_fn =
          cc.mod.getOrInsertFunction("memcmp", ft_memcmp);

      Value *any = ConstantInt::getFalse(cc.llctx);
      for (size_t i = 0; i < vals.size(); ++i) {
        const std::string &s = vals[i]->GetStringValue();

        Value *str_ptr, *str_len_v;
        if (cc.template_mode && cc.params_builder) {
          auto [pp, pl] = EmitParamString(cc, s);
          str_ptr = pp;
          str_len_v = pl;
        } else {
          Constant *str_const =
              ConstantDataArray::getString(cc.llctx, s, false);
          GlobalVariable *str_gv = new GlobalVariable(
              cc.mod, str_const->getType(), true,
              GlobalValue::PrivateLinkage, str_const,
              "in_str_val" + std::to_string(i));
          str_ptr = cc.b.CreateBitCast(str_gv, cc.i8p());
          str_len_v = cc.c32((int32_t)s.size());
        }

        Value *len_eq = cc.b.CreateICmpEQ(slen, str_len_v);
        Value *match;
        if (!cc.template_mode && s.empty()) {
          match = len_eq;
        } else {
          Value *cmp_len = cc.b.CreateSelect(
              len_eq,
              cc.b.CreateSExt(str_len_v, cc.i64()),
              cc.c64(0));
          Value *mcr =
              cc.b.CreateCall(memcmp_fn, {char_ptr, str_ptr, cmp_len});
          Value *bytes_eq = cc.b.CreateICmpEQ(mcr, cc.c32(0));
          match = cc.b.CreateAnd(len_eq, bytes_eq);
        }
        any = cc.b.CreateOr(any, match);
      }
      return expr->negated ? cc.b.CreateNot(any) : any;
    }

    // Large VARCHAR IN-set: LENGTH-bucketed directory over bucket-ordered
    // (ptr, len) arrays, probed by aqp_in_len_str. Rejecting on length needs
    // two directory loads and never touches the probe string's bytes.
    constexpr int32_t kLenCap = 64; // lengths >= cap share the last bucket
    std::vector<std::vector<int>> buckets(kLenCap + 1);
    for (size_t i = 0; i < vals.size(); ++i) {
      int32_t l = (int32_t)vals[i]->GetStringValue().size();
      buckets[l < kLenCap ? l : kLenCap].push_back((int)i);
    }
    std::vector<int> perm;
    std::vector<int32_t> dir(kLenCap + 2, 0);
    for (int32_t b = 0; b <= kLenCap; b++) {
      dir[b] = (int32_t)perm.size();
      for (int idx : buckets[b])
        perm.push_back(idx);
    }
    dir[kLenCap + 1] = (int32_t)perm.size();
    std::vector<Constant *> str_ptrs, str_lens;
    for (int idx : perm) {
      const std::string &s = vals[idx]->GetStringValue();
      Constant *str_const = ConstantDataArray::getString(cc.llctx, s, false);
      GlobalVariable *str_gv =
          new GlobalVariable(cc.mod, str_const->getType(), true,
                             GlobalValue::PrivateLinkage, str_const, "in_str");
      str_ptrs.push_back(ConstantExpr::getBitCast(str_gv, cc.i8p()));
      str_lens.push_back(cc.c32((int32_t)s.size()));
    }
    ArrayType *ptr_arr_ty = ArrayType::get(cc.i8p(), str_ptrs.size());
    ArrayType *len_arr_ty = ArrayType::get(cc.i32(), str_lens.size());
    GlobalVariable *ptrs_gv = new GlobalVariable(
        cc.mod, ptr_arr_ty, true, GlobalValue::PrivateLinkage,
        ConstantArray::get(ptr_arr_ty, str_ptrs), "in_str_ptrs");
    GlobalVariable *lens_gv = new GlobalVariable(
        cc.mod, len_arr_ty, true, GlobalValue::PrivateLinkage,
        ConstantArray::get(len_arr_ty, str_lens), "in_str_lens");
    GlobalVariable *dir_gv = EmitI32ArrayGlobal(cc, dir, "in_str_dir");
    FunctionType *ft_str =
        FunctionType::get(cc.i32(),
                          {cc.i8p(), cc.i32(), PointerType::getUnqual(cc.i8p()),
                           PointerType::getUnqual(cc.i32()),
                           PointerType::getUnqual(cc.i32()), cc.i32()},
                          false);
    FunctionCallee callee =
        cc.mod.getOrInsertFunction("aqp_in_len_str", ft_str);
    Value *ptrs_ptr =
        cc.b.CreateBitCast(ptrs_gv, PointerType::getUnqual(cc.i8p()));
    Value *lens_ptr =
        cc.b.CreateBitCast(lens_gv, PointerType::getUnqual(cc.i32()));
    Value *dir_ptr =
        cc.b.CreateBitCast(dir_gv, PointerType::getUnqual(cc.i32()));
    Value *result = cc.b.CreateCall(
        callee, {char_ptr, slen, ptrs_ptr, lens_ptr, dir_ptr, cc.c32(kLenCap)});
    Value *match = cc.b.CreateICmpNE(result, cc.c32(0));
    return expr->negated ? cc.b.CreateNot(match) : match;
  }

  throw std::runtime_error(
      "AQP-JIT unsupported: IN-list over unsupported column dtype for '" +
      (expr->attr ? expr->attr->GetColumnName() : std::string("<null>")) +
      "' (would pass all rows)");
}

// Arithmetic expression: left OP right → numeric result
// Returns an i32, i64, float, or double Value*.
// Note: for use in filter context, the caller wraps the result in a comparison.
// For use in projection context (future), the result is stored directly.
static Value *EmitArith(CompileCtx &cc, const SimplestArithExpr *expr) {
  if (!expr || !expr->left || !expr->right)
    throw std::runtime_error(
        "AQP-JIT unsupported: arithmetic expression with missing operand");

  Value *lhs = EmitExpr(cc, expr->left.get());
  Value *rhs = EmitExpr(cc, expr->right.get());
  if (!lhs || !rhs)
    throw std::runtime_error(
        "AQP-JIT unsupported: arithmetic operand failed to compile");

  // Determine if floating point based on result type
  bool is_fp = (expr->result_type == ir_sql_converter::FloatVar);

  // Guard integer division/modulo against zero divisor (UB / SIGFPE on x86).
  // Float division by zero is well-defined (produces inf/nan), so no guard needed.
  if (!is_fp && (expr->arith_op == ir_sql_converter::ArithDiv ||
                 expr->arith_op == ir_sql_converter::ArithMod)) {
    Value *is_zero = cc.b.CreateICmpEQ(rhs, ConstantInt::get(rhs->getType(), 0));
    Value *safe_rhs = cc.b.CreateSelect(is_zero,
        ConstantInt::get(rhs->getType(), 1), rhs, "safe_rhs");
    Value *result;
    if (expr->arith_op == ir_sql_converter::ArithDiv)
      result = cc.b.CreateSDiv(lhs, safe_rhs, "div");
    else
      result = cc.b.CreateSRem(lhs, safe_rhs, "mod");
    // Return 0 when divisor was zero (SQL NULL semantics approximation)
    return cc.b.CreateSelect(is_zero,
        ConstantInt::get(result->getType(), 0), result, "divmod_safe");
  }

  switch (expr->arith_op) {
  case ir_sql_converter::ArithAdd:
    return is_fp ? cc.b.CreateFAdd(lhs, rhs, "add")
                 : cc.b.CreateAdd(lhs, rhs, "add");
  case ir_sql_converter::ArithSub:
    return is_fp ? cc.b.CreateFSub(lhs, rhs, "sub")
                 : cc.b.CreateSub(lhs, rhs, "sub");
  case ir_sql_converter::ArithMul:
    return is_fp ? cc.b.CreateFMul(lhs, rhs, "mul")
                 : cc.b.CreateMul(lhs, rhs, "mul");
  case ir_sql_converter::ArithDiv:
    return cc.b.CreateFDiv(lhs, rhs, "div");
  case ir_sql_converter::ArithMod:
    return cc.b.CreateFRem(lhs, rhs, "mod");
  default:
    throw std::runtime_error(
        "AQP-JIT unsupported: arithmetic operator " +
        std::to_string(static_cast<int>(expr->arith_op)));
  }
}

// Type cast: child → target_type
static Value *EmitCast(CompileCtx &cc, const SimplestCastExpr *expr) {
  if (!expr || !expr->child)
    throw std::runtime_error("AQP-JIT unsupported: cast with missing child");

  Value *child = EmitExpr(cc, expr->child.get());
  if (!child)
    throw std::runtime_error(
        "AQP-JIT unsupported: cast child failed to compile");

  Type *src_ty = child->getType();
  Type *dst_ty;
  switch (expr->target_type) {
  case ir_sql_converter::BoolVar:
    dst_ty = cc.i1();
    break;
  case ir_sql_converter::IntVar:
    dst_ty = cc.i32();
    break;
  case ir_sql_converter::FloatVar:
    dst_ty = cc.f64();
    break;
  case ir_sql_converter::Date:
    dst_ty = cc.i32();
    break;
  default:
    throw std::runtime_error(
        "AQP-JIT unsupported: cast to var type " +
        std::to_string(static_cast<int>(expr->target_type)));
  }

  if (src_ty == dst_ty)
    return child;

  // Integer → Integer
  if (src_ty->isIntegerTy() && dst_ty->isIntegerTy()) {
    unsigned src_bits = src_ty->getIntegerBitWidth();
    unsigned dst_bits = dst_ty->getIntegerBitWidth();
    if (dst_bits > src_bits)
      return cc.b.CreateSExt(child, dst_ty, "cast_sext");
    else
      return cc.b.CreateTrunc(child, dst_ty, "cast_trunc");
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
  throw std::runtime_error(
      "AQP-JIT unsupported: cast between incompatible LLVM types");
}

// Main expression dispatch
static Value *EmitExpr(CompileCtx &cc, const AQPExpr *expr) {
  if (!expr)
    throw std::runtime_error(
        "AQP-JIT unsupported: null expression in filter (would pass all "
        "rows)");

  switch (expr->GetNodeType()) {
  case VarConstComparisonNode: {
    auto *vc = static_cast<const SimplestVarConstComparison *>(expr);
    // Integer paths of EmitVarConst have no validity check (pipeline-jit
    // relies on DuckDB re-filtering). The VARCHAR path has its own internal
    // guard — wrapping it again would just add a redundant branch.
    if (cc.strict_null_guard && vc->attr) {
      int ci = cc.FindColIdx(*vc->attr);
      if (ci >= 0 && cc.schema[ci].dtype != AQP_DTYPE_VARCHAR)
        return EmitNullGuardedLeaf(cc, ci,
                                   [&] { return EmitVarConst(cc, vc); });
    }
    return EmitVarConst(cc, vc);
  }
  case VarComparisonNode: {
    auto *vv = static_cast<const SimplestVarComparison *>(expr);
    // EmitVarVar's integer loads have no validity check; NULL on either
    // side must yield false. Guard both columns.
    if (cc.strict_null_guard && vv->left_attr && vv->right_attr) {
      int li = cc.FindColIdx(*vv->left_attr);
      int ri = cc.FindColIdx(*vv->right_attr);
      if (li >= 0 && ri >= 0)
        return EmitNullGuardedLeaf(cc, li, [&] {
          return EmitNullGuardedLeaf(cc, ri,
                                     [&] { return EmitVarVar(cc, vv); });
        });
    }
    return EmitVarVar(cc, vv);
  }
  case IsNullExprNode:
    return EmitIsNull(cc, static_cast<const SimplestIsNullExpr *>(expr));
  case LogicalExprNode:
    return EmitLogical(cc, static_cast<const SimplestLogicalExpr *>(expr));
  case InExprNode: {
    auto *in = static_cast<const SimplestInExpr *>(expr);
    // EmitIn has no validity check in any dtype path; NULL NOT IN (...)
    // would wrongly yield true. Guard all dtypes.
    if (cc.strict_null_guard && in->attr) {
      int ci = cc.FindColIdx(*in->attr);
      if (ci >= 0)
        return EmitNullGuardedLeaf(cc, ci, [&] { return EmitIn(cc, in); });
    }
    return EmitIn(cc, in);
  }
  case ArithExprNode:
    return EmitArith(cc, static_cast<const SimplestArithExpr *>(expr));
  case CastExprNode:
    return EmitCast(cc, static_cast<const SimplestCastExpr *>(expr));
  default:
    throw std::runtime_error(
        "AQP-JIT unsupported: expression node type " +
        std::to_string(static_cast<int>(expr->GetNodeType())) +
        " in filter (would pass all rows)");
  }
}

// ---------------------------------------------------------------------------
// Emit a short-circuit AND chain of filter conjuncts. cc.b must be positioned
// at the start of where the first conjunct should evaluate. On exit, control
// has already branched to bb_pass (all conjuncts true) or bb_fail (any false);
// the caller should not emit any further code immediately after this call
// without first SetInsertPoint'ing somewhere meaningful.
//
// Returns the BB that holds the *final* CondBr (so callers that need PHI
// predecessor tracking can use it). Empty filter list -> unconditional br
// to bb_pass.
// ---------------------------------------------------------------------------
static BasicBlock *EmitShortCircuitFilter(
    CompileCtx &cc, Function *fn,
    const std::vector<const AQPExpr *> &filter_exprs,
    BasicBlock *bb_pass, BasicBlock *bb_fail) {
  if (filter_exprs.empty()) {
    BasicBlock *here = cc.b.GetInsertBlock();
    cc.b.CreateBr(bb_pass);
    return here;
  }
  auto &llctx = cc.b.getContext();
  for (size_t k = 0; k < filter_exprs.size(); ++k) {
    Value *res = EmitExpr(cc, filter_exprs[k]);
    if (k + 1 == filter_exprs.size()) {
      BasicBlock *last = cc.b.GetInsertBlock();
      cc.b.CreateCondBr(res, bb_pass, bb_fail);
      return last;
    }
    BasicBlock *next_chain =
        BasicBlock::Create(llctx, "filt_sc_" + std::to_string(k), fn);
    cc.b.CreateCondBr(res, next_chain, bb_fail);
    cc.b.SetInsertPoint(next_chain);
  }
  return nullptr; // unreachable
}

// ---------------------------------------------------------------------------
// Build the outer loop function:
//   uint64_t aqp_expr_<hash>(AQPChunkView* chunk, AQPSelView* sel)
// ---------------------------------------------------------------------------
static Function *BuildFilterFunction(LLVMContext &llctx, Module &mod,
                                     const std::string &fn_name,
                                     const std::vector<const AQPExpr *> &exprs,
                                     const std::vector<ColSchema> &schema,
                                     ParamsBuilder *pb = nullptr) {
  Type *i8p = PointerType::getUnqual(Type::getInt8Ty(llctx));
  Type *i32 = Type::getInt32Ty(llctx);
  Type *i64 = Type::getInt64Ty(llctx);
  Type *i64p = PointerType::getUnqual(i64);

  // Struct types for the ABI
  StructType *ColViewTy = StructType::get(llctx, {i8p, i64p, i32, i32});
  StructType *ChunkViewTy =
      StructType::get(llctx, {PointerType::getUnqual(ColViewTy), i64, i64});
  // AQPSelView.indices is sel_t* = uint32_t* (DuckDB typedefs.hpp: typedef
  // uint32_t sel_t)
  StructType *SelViewTy =
      StructType::get(llctx, {PointerType::getUnqual(i32), i32});

  FunctionType *fn_ty = FunctionType::get(
      i64,
      {PointerType::getUnqual(ChunkViewTy), PointerType::getUnqual(SelViewTy)},
      false);
  Function *fn =
      Function::Create(fn_ty, Function::ExternalLinkage, fn_name, &mod);

  Value *chunk_arg = fn->getArg(0);
  Value *sel_arg = fn->getArg(1);
  chunk_arg->setName("chunk");
  sel_arg->setName("sel");

  BasicBlock *entry_bb = BasicBlock::Create(llctx, "entry", fn);
  BasicBlock *loop_bb = BasicBlock::Create(llctx, "loop", fn);
  BasicBlock *body_bb = BasicBlock::Create(llctx, "body", fn);
  BasicBlock *store_bb = BasicBlock::Create(llctx, "store", fn);
  BasicBlock *next_bb = BasicBlock::Create(llctx, "next", fn);
  BasicBlock *exit_bb = BasicBlock::Create(llctx, "exit", fn);

  CompileCtx cc(llctx, mod, schema, chunk_arg, sel_arg);
  cc.b.SetInsertPoint(entry_bb);

  if (pb) {
    cc.template_mode = true;
    cc.params_builder = pb;
    cc.LoadParamsFromThreadLocal();
  }

  // Load nrows from chunk->nrows (field index 1)
  Value *nrows_ptr = cc.b.CreateStructGEP(ChunkViewTy, chunk_arg, 1);
  Value *nrows = cc.b.CreateLoad(i64, nrows_ptr, "nrows");

  // Load col data + validity pointers (once, before the loop)
  cc.col_data.resize(schema.size());
  cc.col_validity.resize(schema.size());
  for (size_t i = 0; i < schema.size(); i++) {
    cc.col_data[i] = cc.LoadColData((unsigned)i);
    cc.col_validity[i] = cc.LoadColValidity((unsigned)i);
  }

  // Load sel->indices pointer (sel_t* = uint32_t*)
  Value *sel_idx_ptr_ptr = cc.b.CreateStructGEP(SelViewTy, sel_arg, 0);
  Value *sel_idx_ptr = cc.b.CreateLoad(PointerType::getUnqual(i32),
                                       sel_idx_ptr_ptr, "sel_indices");

  cc.b.CreateBr(loop_bb);

  // Loop header — i = 0, out_count = 0
  cc.b.SetInsertPoint(loop_bb);
  PHINode *i = cc.b.CreatePHI(i64, 2, "i");
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
  Value *dst = cc.b.CreateGEP(i32, sel_idx_ptr, out_count, "dst");
  Value *i_i32 = cc.b.CreateTrunc(i, i32, "i_i32");
  cc.b.CreateStore(i_i32, dst);
  Value *out_count1 =
      cc.b.CreateAdd(out_count, ConstantInt::get(llctx, APInt(64, 1)));
  cc.b.CreateBr(next_bb);

  // Increment i
  cc.b.SetInsertPoint(next_bb);
  PHINode *out_count_next = cc.b.CreatePHI(i64, 2, "out_count_next");
  // condBr_bb is the actual predecessor of next_bb on the "no match" path.
  // (It may differ from body_bb when EmitLogical created intermediate blocks.)
  out_count_next->addIncoming(out_count, condBr_bb);
  out_count_next->addIncoming(out_count1, store_bb);
  Value *i1 =
      cc.b.CreateAdd(i, ConstantInt::get(llctx, APInt(64, 1)), "i_next");
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
// Check if an expression is "expensive" — involves VARCHAR LIKE, CONTAINS,
// or other string pattern matching.  Used by the two-pass scalar filter to
// separate cheap predicates (Phase 1) from expensive ones (Phase 2).
// ---------------------------------------------------------------------------
static bool IsExpensiveExpr(const AQPExpr *e) {
  if (!e) return false;
  switch (e->GetNodeType()) {
  case VarConstComparisonNode: {
    auto *cmp = static_cast<const SimplestVarConstComparison *>(e);
    auto et = cmp->GetSimplestExprType();
    if (et == SimplestExprType::TextLike ||
        et == SimplestExprType::Text_Not_Like)
      return true;
    return false;
  }
  case LogicalExprNode: {
    auto *log = static_cast<const SimplestLogicalExpr *>(e);
    if (log->left_expr && IsExpensiveExpr(log->left_expr.get()))
      return true;
    if (log->right_expr && IsExpensiveExpr(log->right_expr.get()))
      return true;
    return false;
  }
  default:
    return false;
  }
}

// ---------------------------------------------------------------------------
// Two-pass scalar filter: cheap predicates first (sequential), then expensive
// predicates (LIKE/CONTAINS) only on survivors.  No SIMD — pure scalar.
//
// Phase 1: loop all rows, evaluate cheap_exprs with && short-circuit,
//          write survivors to sel[].
// Phase 2: loop sel[0..phase1_count), evaluate expensive_exprs on each
//          survivor, compact in-place.
// ---------------------------------------------------------------------------
static Function *BuildFilterFunctionTwoPass(
    LLVMContext &llctx, Module &mod, const std::string &fn_name,
    const std::vector<const AQPExpr *> &cheap_exprs,
    const std::vector<const AQPExpr *> &expensive_exprs,
    const std::vector<ColSchema> &schema) {
  Type *i8p = PointerType::getUnqual(Type::getInt8Ty(llctx));
  Type *i32 = Type::getInt32Ty(llctx);
  Type *i64 = Type::getInt64Ty(llctx);
  Type *i64p = PointerType::getUnqual(i64);

  StructType *ColViewTy = StructType::get(llctx, {i8p, i64p, i32, i32});
  StructType *ChunkViewTy =
      StructType::get(llctx, {PointerType::getUnqual(ColViewTy), i64, i64});
  StructType *SelViewTy =
      StructType::get(llctx, {PointerType::getUnqual(i32), i32});

  FunctionType *fn_ty = FunctionType::get(
      i64,
      {PointerType::getUnqual(ChunkViewTy), PointerType::getUnqual(SelViewTy)},
      false);
  Function *fn =
      Function::Create(fn_ty, Function::ExternalLinkage, fn_name, &mod);

  Value *chunk_arg = fn->getArg(0);
  Value *sel_arg = fn->getArg(1);
  chunk_arg->setName("chunk");
  sel_arg->setName("sel");

  // ===== PHASE 1: cheap predicates on all rows =====
  BasicBlock *entry_bb = BasicBlock::Create(llctx, "entry", fn);
  BasicBlock *p1_loop_bb = BasicBlock::Create(llctx, "p1_loop", fn);
  BasicBlock *p1_body_bb = BasicBlock::Create(llctx, "p1_body", fn);
  BasicBlock *p1_store_bb = BasicBlock::Create(llctx, "p1_store", fn);
  BasicBlock *p1_next_bb = BasicBlock::Create(llctx, "p1_next", fn);
  BasicBlock *phase2_bb = BasicBlock::Create(llctx, "phase2", fn);

  CompileCtx cc1(llctx, mod, schema, chunk_arg, sel_arg);
  cc1.b.SetInsertPoint(entry_bb);

  Value *nrows_ptr = cc1.b.CreateStructGEP(ChunkViewTy, chunk_arg, 1);
  Value *nrows = cc1.b.CreateLoad(i64, nrows_ptr, "nrows");

  cc1.col_data.resize(schema.size());
  cc1.col_validity.resize(schema.size());
  for (size_t i = 0; i < schema.size(); i++) {
    cc1.col_data[i] = cc1.LoadColData((unsigned)i);
    cc1.col_validity[i] = cc1.LoadColValidity((unsigned)i);
  }

  Value *sel_idx_ptr_ptr = cc1.b.CreateStructGEP(SelViewTy, sel_arg, 0);
  Value *sel_idx_ptr = cc1.b.CreateLoad(PointerType::getUnqual(i32),
                                        sel_idx_ptr_ptr, "sel_indices");
  cc1.b.CreateBr(p1_loop_bb);

  cc1.b.SetInsertPoint(p1_loop_bb);
  PHINode *p1_i = cc1.b.CreatePHI(i64, 2, "p1_i");
  PHINode *p1_oc = cc1.b.CreatePHI(i64, 2, "p1_oc");
  p1_i->addIncoming(ConstantInt::get(i64, 0), entry_bb);
  p1_oc->addIncoming(ConstantInt::get(i64, 0), entry_bb);
  cc1.b.CreateCondBr(cc1.b.CreateICmpEQ(p1_i, nrows, "p1_done"),
                     phase2_bb, p1_body_bb);

  cc1.b.SetInsertPoint(p1_body_bb);
  cc1.row_idx = p1_i;

  Value *p1_match = ConstantInt::getTrue(llctx);
  for (const AQPExpr *e : cheap_exprs) {
    Value *res = EmitExpr(cc1, e);
    p1_match = cc1.b.CreateAnd(p1_match, res);
  }
  BasicBlock *p1_condBr_bb = cc1.b.GetInsertBlock();
  cc1.b.CreateCondBr(p1_match, p1_store_bb, p1_next_bb);

  cc1.b.SetInsertPoint(p1_store_bb);
  Value *p1_dst = cc1.b.CreateGEP(i32, sel_idx_ptr, p1_oc, "p1_dst");
  Value *p1_i32 = cc1.b.CreateTrunc(p1_i, i32, "p1_i32");
  cc1.b.CreateStore(p1_i32, p1_dst);
  Value *p1_oc1 = cc1.b.CreateAdd(p1_oc, ConstantInt::get(i64, 1));
  cc1.b.CreateBr(p1_next_bb);

  cc1.b.SetInsertPoint(p1_next_bb);
  PHINode *p1_oc_next = cc1.b.CreatePHI(i64, 2, "p1_oc_next");
  p1_oc_next->addIncoming(p1_oc, p1_condBr_bb);
  p1_oc_next->addIncoming(p1_oc1, p1_store_bb);
  Value *p1_i_next = cc1.b.CreateAdd(p1_i, ConstantInt::get(i64, 1), "p1_i_next");
  p1_i->addIncoming(p1_i_next, p1_next_bb);
  p1_oc->addIncoming(p1_oc_next, p1_next_bb);
  cc1.b.CreateBr(p1_loop_bb);

  // ===== PHASE 2: expensive predicates on survivors =====
  BasicBlock *p2_loop_bb = BasicBlock::Create(llctx, "p2_loop", fn);
  BasicBlock *p2_body_bb = BasicBlock::Create(llctx, "p2_body", fn);
  BasicBlock *p2_keep_bb = BasicBlock::Create(llctx, "p2_keep", fn);
  BasicBlock *p2_next_bb = BasicBlock::Create(llctx, "p2_next", fn);
  BasicBlock *exit_bb = BasicBlock::Create(llctx, "exit", fn);

  IRBuilder<> b2(llctx);
  b2.SetInsertPoint(phase2_bb);
  Value *phase1_count = p1_oc;
  b2.CreateBr(p2_loop_bb);

  b2.SetInsertPoint(p2_loop_bb);
  PHINode *p2_i = b2.CreatePHI(i64, 2, "p2_i");
  PHINode *p2_oc = b2.CreatePHI(i64, 2, "p2_oc");
  p2_i->addIncoming(ConstantInt::get(i64, 0), phase2_bb);
  p2_oc->addIncoming(ConstantInt::get(i64, 0), phase2_bb);
  b2.CreateCondBr(b2.CreateICmpEQ(p2_i, phase1_count), exit_bb, p2_body_bb);

  {
    CompileCtx cc2(llctx, mod, schema, chunk_arg, sel_arg);
    cc2.b.SetInsertPoint(p2_body_bb);
    cc2.col_data = cc1.col_data;
    cc2.col_validity = cc1.col_validity;

    Value *row_from_sel = cc2.b.CreateZExt(
        cc2.b.CreateLoad(i32, cc2.b.CreateGEP(i32, sel_idx_ptr, p2_i)), i64);
    cc2.row_idx = row_from_sel;

    Value *p2_match = ConstantInt::getTrue(llctx);
    for (const AQPExpr *e : expensive_exprs) {
      Value *res = EmitExpr(cc2, e);
      p2_match = cc2.b.CreateAnd(p2_match, res);
    }
    BasicBlock *p2_condBr_bb = cc2.b.GetInsertBlock();
    cc2.b.CreateCondBr(p2_match, p2_keep_bb, p2_next_bb);

    b2.SetInsertPoint(p2_keep_bb);
    Value *src_val = b2.CreateLoad(i32, b2.CreateGEP(i32, sel_idx_ptr, p2_i));
    b2.CreateStore(src_val, b2.CreateGEP(i32, sel_idx_ptr, p2_oc));
    Value *p2_oc_inc = b2.CreateAdd(p2_oc, ConstantInt::get(i64, 1));
    b2.CreateBr(p2_next_bb);

    b2.SetInsertPoint(p2_next_bb);
    PHINode *p2_oc_next = b2.CreatePHI(i64, 2, "p2_oc_next");
    p2_oc_next->addIncoming(p2_oc, p2_condBr_bb);
    p2_oc_next->addIncoming(p2_oc_inc, p2_keep_bb);
    Value *p2_i_next = b2.CreateAdd(p2_i, ConstantInt::get(i64, 1));
    p2_i->addIncoming(p2_i_next, p2_next_bb);
    p2_oc->addIncoming(p2_oc_next, p2_next_bb);
    b2.CreateBr(p2_loop_bb);
  }

  b2.SetInsertPoint(exit_bb);
  PHINode *final_count = b2.CreatePHI(i64, 1, "final_count");
  final_count->addIncoming(p2_oc, p2_loop_bb);
  b2.CreateStore(b2.CreateTrunc(final_count, i32),
                 b2.CreateStructGEP(SelViewTy, sel_arg, 1));
  b2.CreateRet(final_count);

  return fn;
}

// ---------------------------------------------------------------------------
// Check if all expressions in a filter are SIMD-friendly (numeric comparisons
// and logical AND/OR/NOT — no VARCHAR LIKE, no external function calls).
// ---------------------------------------------------------------------------
static bool AllExprsSIMDFriendly(const std::vector<const AQPExpr *> &exprs,
                                 const std::vector<ColSchema> &schema) {
  for (const AQPExpr *e : exprs) {
    if (!e)
      return false;
    switch (e->GetNodeType()) {
    case VarConstComparisonNode: {
      auto *cmp = static_cast<const SimplestVarConstComparison *>(e);
      auto et = cmp->GetSimplestExprType();
      // VARCHAR LIKE/equality needs runtime function calls — not SIMD-friendly
      if (et == SimplestExprType::TextLike ||
          et == SimplestExprType::Text_Not_Like)
        return false;
      // Check dtype: VARCHAR comparisons stay scalar.
      // Also: column must be IN schema (not a pass-through).
      bool found = false;
      for (auto &cs : schema) {
        if (cs.table_idx == cmp->attr->GetTableIndex() &&
            cs.col_idx == cmp->attr->GetColumnIndex()) {
          if (cs.dtype == AQP_DTYPE_VARCHAR)
            return false;
          if (cs.dtype != AQP_DTYPE_INT32 && cs.dtype != AQP_DTYPE_DATE &&
              cs.dtype != AQP_DTYPE_INT64 && cs.dtype != AQP_DTYPE_FLOAT &&
              cs.dtype != AQP_DTYPE_DOUBLE)
            return false; // only numeric types for SIMD
          found = true;
          break;
        }
      }
      if (!found)
        return false; // column not in schema → pass-through, not SIMD-able
      break;
    }
    case LogicalExprNode: {
      auto *log = static_cast<const SimplestLogicalExpr *>(e);
      std::vector<const AQPExpr *> children;
      if (log->left_expr)
        children.push_back(log->left_expr.get());
      if (log->right_expr)
        children.push_back(log->right_expr.get());
      if (!AllExprsSIMDFriendly(children, schema))
        return false;
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
static Function *
BuildFilterFunctionSIMD(LLVMContext &llctx, Module &mod,
                        const std::string &fn_name,
                        const std::vector<const AQPExpr *> &exprs,
                        const std::vector<ColSchema> &schema, unsigned VW) {
  Type *i8p = PointerType::getUnqual(Type::getInt8Ty(llctx));
  Type *i1 = Type::getInt1Ty(llctx);
  Type *i32 = Type::getInt32Ty(llctx);
  Type *i64 = Type::getInt64Ty(llctx);
  Type *i64p = PointerType::getUnqual(i64);

  StructType *ColViewTy = StructType::get(llctx, {i8p, i64p, i32, i32});
  StructType *ChunkViewTy =
      StructType::get(llctx, {PointerType::getUnqual(ColViewTy), i64, i64});
  StructType *SelViewTy =
      StructType::get(llctx, {PointerType::getUnqual(i32), i32});

  // Vector types
  auto *vi32 = FixedVectorType::get(i32, VW);
  auto *vi64 = FixedVectorType::get(i64, VW);
  auto *vi1 = FixedVectorType::get(i1, VW);
  (void)vi64; // may be used for INT64 later

  FunctionType *fn_ty = FunctionType::get(
      i64,
      {PointerType::getUnqual(ChunkViewTy), PointerType::getUnqual(SelViewTy)},
      false);
  Function *fn =
      Function::Create(fn_ty, Function::ExternalLinkage, fn_name, &mod);

  Value *chunk_arg = fn->getArg(0);
  chunk_arg->setName("chunk");
  Value *sel_arg = fn->getArg(1);
  sel_arg->setName("sel");

  BasicBlock *entry_bb = BasicBlock::Create(llctx, "entry", fn);
  BasicBlock *vec_loop_bb = BasicBlock::Create(llctx, "vec_loop", fn);
  BasicBlock *vec_body_bb = BasicBlock::Create(llctx, "vec_body", fn);
  BasicBlock *vec_store_bb = BasicBlock::Create(llctx, "vec_store", fn);
  BasicBlock *vec_next_bb = BasicBlock::Create(llctx, "vec_next", fn);
  BasicBlock *tail_bb = BasicBlock::Create(llctx, "tail", fn);
  BasicBlock *tail_body_bb = BasicBlock::Create(llctx, "tail_body", fn);
  BasicBlock *tail_store_bb = BasicBlock::Create(llctx, "tail_store", fn);
  BasicBlock *tail_next_bb = BasicBlock::Create(llctx, "tail_next", fn);
  BasicBlock *exit_bb = BasicBlock::Create(llctx, "exit", fn);

  IRBuilder<> b(entry_bb);

  // Load nrows, col data, sel indices
  Value *nrows_ptr = b.CreateStructGEP(ChunkViewTy, chunk_arg, 1);
  Value *nrows = b.CreateLoad(i64, nrows_ptr, "nrows");
  Value *cols_ptr = b.CreateStructGEP(ChunkViewTy, chunk_arg, 0);
  Value *cols =
      b.CreateLoad(PointerType::getUnqual(ColViewTy), cols_ptr, "cols");

  // Pre-load column data pointers
  std::vector<Value *> col_data(schema.size());
  std::vector<Value *> col_validity(schema.size());
  for (size_t ci = 0; ci < schema.size(); ci++) {
    Value *col_i = b.CreateGEP(ColViewTy, cols, ConstantInt::get(i64, ci));
    col_data[ci] = b.CreateLoad(i8p, b.CreateStructGEP(ColViewTy, col_i, 0));
    col_validity[ci] =
        b.CreateLoad(i64p, b.CreateStructGEP(ColViewTy, col_i, 1));
  }

  Value *sel_idx_ptr =
      b.CreateLoad(PointerType::getUnqual(i32),
                   b.CreateStructGEP(SelViewTy, sel_arg, 0), "sel_indices");

  // vec_limit = nrows & ~(VW-1)  — round down to multiple of VW
  Value *vw_const = ConstantInt::get(i64, VW);
  Value *vw_mask = ConstantInt::get(i64, ~(uint64_t)(VW - 1));
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
            schema[ci].col_idx == cmp->attr->GetColumnIndex()) {
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
        Value *typed_ptr =
            b.CreateBitCast(data_ptr, PointerType::getUnqual(i32));
        auto *vty = FixedVectorType::get(i32, VW);
        auto *load = b.CreateAlignedLoad(
            vty,
            b.CreateBitCast(b.CreateGEP(i32, typed_ptr, vi),
                            PointerType::getUnqual(vty)),
            Align(4), "data_vec");
        data_vec = load;
        int32_t cv = cmp->const_var->GetIntValue();
        const_vec = ConstantVector::getSplat(
            ElementCount::getFixed(VW),
            ConstantInt::get(i32, (uint64_t)(uint32_t)cv, true));
      } else if (dtype == AQP_DTYPE_INT64) {
        Value *typed_ptr =
            b.CreateBitCast(data_ptr, PointerType::getUnqual(i64));
        auto *vty = FixedVectorType::get(i64, VW);
        auto *load = b.CreateAlignedLoad(
            vty,
            b.CreateBitCast(b.CreateGEP(i64, typed_ptr, vi),
                            PointerType::getUnqual(vty)),
            Align(8), "data_vec");
        data_vec = load;
        int64_t cv = (int64_t)cmp->const_var->GetIntValue();
        const_vec =
            ConstantVector::getSplat(ElementCount::getFixed(VW),
                                     ConstantInt::get(i64, (uint64_t)cv, true));
      } else if (dtype == AQP_DTYPE_FLOAT) {
        Type *f32 = Type::getFloatTy(llctx);
        Value *typed_ptr =
            b.CreateBitCast(data_ptr, PointerType::getUnqual(f32));
        auto *vty = FixedVectorType::get(f32, VW);
        auto *load = b.CreateAlignedLoad(
            vty,
            b.CreateBitCast(b.CreateGEP(f32, typed_ptr, vi),
                            PointerType::getUnqual(vty)),
            Align(4), "data_vec");
        data_vec = load;
        float cv = cmp->const_var->GetFloatValue();
        const_vec = ConstantVector::getSplat(ElementCount::getFixed(VW),
                                             ConstantFP::get(f32, (double)cv));
        is_fp = true;
      } else if (dtype == AQP_DTYPE_DOUBLE) {
        Type *f64 = Type::getDoubleTy(llctx);
        Value *typed_ptr =
            b.CreateBitCast(data_ptr, PointerType::getUnqual(f64));
        auto *vty = FixedVectorType::get(f64, VW);
        auto *load = b.CreateAlignedLoad(
            vty,
            b.CreateBitCast(b.CreateGEP(f64, typed_ptr, vi),
                            PointerType::getUnqual(vty)),
            Align(8), "data_vec");
        data_vec = load;
        float cv = cmp->const_var->GetFloatValue();
        const_vec = ConstantVector::getSplat(ElementCount::getFixed(VW),
                                             ConstantFP::get(f64, (double)cv));
        is_fp = true;
      } else {
        fn->eraseFromParent();
        return nullptr;
      }

      // Vector comparison
      auto et = cmp->GetSimplestExprType();
      if (is_fp) {
        switch (et) {
        case SimplestExprType::Equal:
          expr_mask = b.CreateFCmpOEQ(data_vec, const_vec);
          break;
        case SimplestExprType::NotEqual:
          expr_mask = b.CreateFCmpONE(data_vec, const_vec);
          break;
        case SimplestExprType::LessThan:
          expr_mask = b.CreateFCmpOLT(data_vec, const_vec);
          break;
        case SimplestExprType::GreaterThan:
          expr_mask = b.CreateFCmpOGT(data_vec, const_vec);
          break;
        case SimplestExprType::LessEqual:
          expr_mask = b.CreateFCmpOLE(data_vec, const_vec);
          break;
        case SimplestExprType::GreaterEqual:
          expr_mask = b.CreateFCmpOGE(data_vec, const_vec);
          break;
        default:
          fn->eraseFromParent();
          return nullptr;
        }
      } else {
        switch (et) {
        case SimplestExprType::Equal:
          expr_mask = b.CreateICmpEQ(data_vec, const_vec);
          break;
        case SimplestExprType::NotEqual:
          expr_mask = b.CreateICmpNE(data_vec, const_vec);
          break;
        case SimplestExprType::LessThan:
          expr_mask = b.CreateICmpSLT(data_vec, const_vec);
          break;
        case SimplestExprType::GreaterThan:
          expr_mask = b.CreateICmpSGT(data_vec, const_vec);
          break;
        case SimplestExprType::LessEqual:
          expr_mask = b.CreateICmpSLE(data_vec, const_vec);
          break;
        case SimplestExprType::GreaterEqual:
          expr_mask = b.CreateICmpSGE(data_vec, const_vec);
          break;
        default:
          fn->eraseFromParent();
          return nullptr;
        }
      } // end else (integer path)

      // AND with validity mask
      Value *validity = col_validity[col_idx];
      Value *val_nonnull = b.CreateICmpNE(b.CreatePtrToInt(validity, i64),
                                          ConstantInt::get(i64, 0));

      // If validity is non-null, extract VW bits and AND with comparison
      // Since VW divides 64 evenly (4,8,16), bits never span two words
      BasicBlock *has_val_bb = BasicBlock::Create(llctx, "has_val", fn);
      BasicBlock *no_val_bb = BasicBlock::Create(llctx, "no_val", fn);
      BasicBlock *merge_val_bb = BasicBlock::Create(llctx, "merge_val", fn);
      b.CreateCondBr(val_nonnull, has_val_bb, no_val_bb);

      b.SetInsertPoint(has_val_bb);
      Value *word_idx = b.CreateLShr(vi, ConstantInt::get(i64, 6));
      Value *bit_off = b.CreateAnd(vi, ConstantInt::get(i64, 63));
      Value *word = b.CreateLoad(i64, b.CreateGEP(i64, validity, word_idx));
      Value *shifted = b.CreateLShr(word, bit_off);
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
            schema[ci].col_idx == isnull->attr->GetColumnIndex()) {
          col_idx = ci;
          break;
        }
      }
      if (col_idx < 0) {
        fn->eraseFromParent();
        return nullptr;
      }

      // IS NULL: check validity bits
      Value *validity = col_validity[col_idx];
      Value *val_nonnull = b.CreateICmpNE(b.CreatePtrToInt(validity, i64),
                                          ConstantInt::get(i64, 0));

      BasicBlock *has_v = BasicBlock::Create(llctx, "isnull_has_v", fn);
      BasicBlock *no_v = BasicBlock::Create(llctx, "isnull_no_v", fn);
      BasicBlock *merge = BasicBlock::Create(llctx, "isnull_merge", fn);
      b.CreateCondBr(val_nonnull, has_v, no_v);

      b.SetInsertPoint(has_v);
      Value *word_idx2 = b.CreateLShr(vi, ConstantInt::get(i64, 6));
      Value *bit_off2 = b.CreateAnd(vi, ConstantInt::get(i64, 63));
      Value *word2 =
          b.CreateLoad(i64, b.CreateGEP(i64, col_validity[col_idx], word_idx2));
      Value *shifted2 = b.CreateLShr(word2, bit_off2);
      Type *iVW2 = Type::getIntNTy(llctx, VW);
      Value *bits = b.CreateBitCast(b.CreateTrunc(shifted2, iVW2), vi1);
      // IS NULL: valid bit = 0 means NULL
      Value *null_mask =
          (isnull->GetSimplestExprType() == SimplestExprType::NullType)
              ? b.CreateNot(bits)
              : bits;
      b.CreateBr(merge);

      b.SetInsertPoint(no_v);
      // All valid: IS NULL → all false, IS NOT NULL → all true
      Value *all_const =
          (isnull->GetSimplestExprType() == SimplestExprType::NullType)
              ? ConstantVector::getSplat(ElementCount::getFixed(VW),
                                         ConstantInt::getFalse(llctx))
              : ConstantVector::getSplat(ElementCount::getFixed(VW),
                                         ConstantInt::getTrue(llctx));
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

  // Scatter loop: while (mask != 0) { k = ctz(mask); store vi+k; mask &=
  // mask-1; }
  BasicBlock *scatter_loop_bb = BasicBlock::Create(llctx, "scatter_loop", fn);
  BasicBlock *scatter_body_bb = BasicBlock::Create(llctx, "scatter_body", fn);
  BasicBlock *scatter_done_bb = BasicBlock::Create(llctx, "scatter_done", fn);
  b.CreateBr(scatter_loop_bb);

  b.SetInsertPoint(scatter_loop_bb);
  PHINode *sc_mask = b.CreatePHI(i64, 2, "sc_mask");
  PHINode *sc_oc = b.CreatePHI(i64, 2, "sc_oc");
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
  Value *sc_mask_next =
      b.CreateAnd(sc_mask, b.CreateSub(sc_mask, ConstantInt::get(i64, 1)));
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
  // exit_bb has one predecessor (tail_bb), so no PHI needed.
  // toc already carries the final row count (initialized from voc, accumulated in tail).
  Value *sel_cnt_ptr = b.CreateStructGEP(SelViewTy, sel_arg, 1);
  b.CreateStore(b.CreateTrunc(toc, i32), sel_cnt_ptr);
  b.CreateRet(toc);

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
static Function *
BuildFilterFunctionHybrid(LLVMContext &llctx, Module &mod,
                          const std::string &fn_name,
                          const std::vector<const AQPExpr *> &simd_exprs,
                          const std::vector<const AQPExpr *> &scalar_exprs,
                          const std::vector<ColSchema> &schema, unsigned VW) {
  // Phase 1: Build a SIMD filter for numeric expressions only.
  // This produces the selection vector with rows passing numeric predicates.
  // We reuse BuildFilterFunctionSIMD's logic inline.

  Type *i8p = PointerType::getUnqual(Type::getInt8Ty(llctx));
  Type *i1 = Type::getInt1Ty(llctx);
  Type *i32 = Type::getInt32Ty(llctx);
  Type *i64 = Type::getInt64Ty(llctx);
  Type *i64p = PointerType::getUnqual(i64);

  StructType *ColViewTy = StructType::get(llctx, {i8p, i64p, i32, i32});
  StructType *ChunkViewTy =
      StructType::get(llctx, {PointerType::getUnqual(ColViewTy), i64, i64});
  StructType *SelViewTy =
      StructType::get(llctx, {PointerType::getUnqual(i32), i32});

  auto *vi32 = FixedVectorType::get(i32, VW);
  auto *vi1 = FixedVectorType::get(i1, VW);

  FunctionType *fn_ty = FunctionType::get(
      i64,
      {PointerType::getUnqual(ChunkViewTy), PointerType::getUnqual(SelViewTy)},
      false);
  Function *fn =
      Function::Create(fn_ty, Function::ExternalLinkage, fn_name, &mod);

  Value *chunk_arg = fn->getArg(0);
  chunk_arg->setName("chunk");
  Value *sel_arg = fn->getArg(1);
  sel_arg->setName("sel");

  BasicBlock *entry_bb = BasicBlock::Create(llctx, "entry", fn);
  BasicBlock *vec_loop_bb = BasicBlock::Create(llctx, "vec_loop", fn);
  BasicBlock *vec_body_bb = BasicBlock::Create(llctx, "vec_body", fn);
  BasicBlock *scatter_bb = BasicBlock::Create(llctx, "scatter", fn);
  BasicBlock *scatter_loop = BasicBlock::Create(llctx, "scatter_loop", fn);
  BasicBlock *scatter_body = BasicBlock::Create(llctx, "scatter_body", fn);
  BasicBlock *scatter_done = BasicBlock::Create(llctx, "scatter_done", fn);
  BasicBlock *vec_next_bb = BasicBlock::Create(llctx, "vec_next", fn);
  BasicBlock *tail_bb = BasicBlock::Create(llctx, "tail", fn);
  BasicBlock *tail_body_bb = BasicBlock::Create(llctx, "tail_body", fn);
  BasicBlock *tail_store_bb = BasicBlock::Create(llctx, "tail_store", fn);
  BasicBlock *tail_next_bb = BasicBlock::Create(llctx, "tail_next", fn);
  BasicBlock *phase2_bb = BasicBlock::Create(llctx, "phase2", fn);
  BasicBlock *p2_body_bb = BasicBlock::Create(llctx, "p2_body", fn);
  BasicBlock *p2_keep_bb = BasicBlock::Create(llctx, "p2_keep", fn);
  BasicBlock *p2_next_bb = BasicBlock::Create(llctx, "p2_next", fn);
  BasicBlock *exit_bb = BasicBlock::Create(llctx, "exit", fn);

  IRBuilder<> b(entry_bb);

  // Load nrows, columns, sel indices
  Value *nrows =
      b.CreateLoad(i64, b.CreateStructGEP(ChunkViewTy, chunk_arg, 1), "nrows");
  Value *cols =
      b.CreateLoad(PointerType::getUnqual(ColViewTy),
                   b.CreateStructGEP(ChunkViewTy, chunk_arg, 0), "cols");

  std::vector<Value *> col_data(schema.size());
  std::vector<Value *> col_validity(schema.size());
  for (size_t ci = 0; ci < schema.size(); ci++) {
    Value *col_i = b.CreateGEP(ColViewTy, cols, ConstantInt::get(i64, ci));
    col_data[ci] = b.CreateLoad(i8p, b.CreateStructGEP(ColViewTy, col_i, 0));
    col_validity[ci] =
        b.CreateLoad(i64p, b.CreateStructGEP(ColViewTy, col_i, 1));
  }

  Value *sel_idx_ptr =
      b.CreateLoad(PointerType::getUnqual(i32),
                   b.CreateStructGEP(SelViewTy, sel_arg, 0), "sel_indices");

  Value *vec_limit =
      b.CreateAnd(nrows, ConstantInt::get(i64, ~(uint64_t)(VW - 1)));
  b.CreateBr(vec_loop_bb);

  // ===== PHASE 1: SIMD numeric predicates =====
  b.SetInsertPoint(vec_loop_bb);
  PHINode *vi = b.CreatePHI(i64, 2, "vi");
  PHINode *voc = b.CreatePHI(i64, 2, "voc");
  vi->addIncoming(ConstantInt::get(i64, 0), entry_bb);
  voc->addIncoming(ConstantInt::get(i64, 0), entry_bb);
  b.CreateCondBr(b.CreateICmpEQ(vi, vec_limit), tail_bb, vec_body_bb);

  b.SetInsertPoint(vec_body_bb);
  Value *combined_mask = ConstantVector::getSplat(ElementCount::getFixed(VW),
                                                  ConstantInt::getTrue(llctx));

  // Flatten compound AND expressions into leaf comparisons.
  // e.g., (year >= 1950 AND year <= 2000 AND year IS NOT NULL) → 3 leaves
  std::vector<const AQPExpr *> flat_simd;
  std::function<void(const AQPExpr *)> flatten = [&](const AQPExpr *e) {
    if (!e)
      return;
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
      // IS NULL in a filter context usually means "keep NULLs" — rare, skip for
      // now
      continue;
    }
    if (e->GetNodeType() != VarConstComparisonNode) {
      fn->eraseFromParent();
      return nullptr;
    }
    auto *cmp = static_cast<const SimplestVarConstComparison *>(e);
    int col_idx = -1;
    int32_t dtype = AQP_DTYPE_OTHER;
    for (int ci = 0; ci < (int)schema.size(); ci++) {
      if (schema[ci].table_idx == cmp->attr->GetTableIndex() &&
          schema[ci].col_idx == cmp->attr->GetColumnIndex()) {
        col_idx = ci;
        dtype = schema[ci].dtype;
        break;
      }
    }
    if (col_idx < 0) {
      fn->eraseFromParent();
      return nullptr;
    }

    // Load VW elements as a vector and splat the constant.
    // LLVM automatically splits oversized vectors (e.g., <8 x i64> on AVX2
    // becomes two <4 x i64> ops). Result is always <VW x i1>.
    Value *data_vec = nullptr;
    Value *const_vec = nullptr;
    bool is_fp = false;

    if (dtype == AQP_DTYPE_INT32 || dtype == AQP_DTYPE_DATE) {
      Value *typed_ptr =
          b.CreateBitCast(col_data[col_idx], PointerType::getUnqual(i32));
      auto *vty = FixedVectorType::get(i32, VW);
      data_vec = b.CreateAlignedLoad(
          vty,
          b.CreateBitCast(b.CreateGEP(i32, typed_ptr, vi),
                          PointerType::getUnqual(vty)),
          Align(4));
      int32_t cv = cmp->const_var->GetIntValue();
      const_vec = ConstantVector::getSplat(
          ElementCount::getFixed(VW),
          ConstantInt::get(i32, (uint64_t)(uint32_t)cv, true));
    } else if (dtype == AQP_DTYPE_INT64) {
      Value *typed_ptr =
          b.CreateBitCast(col_data[col_idx], PointerType::getUnqual(i64));
      auto *vty = FixedVectorType::get(i64, VW);
      data_vec = b.CreateAlignedLoad(
          vty,
          b.CreateBitCast(b.CreateGEP(i64, typed_ptr, vi),
                          PointerType::getUnqual(vty)),
          Align(8));
      int64_t cv = (int64_t)cmp->const_var->GetIntValue();
      const_vec =
          ConstantVector::getSplat(ElementCount::getFixed(VW),
                                   ConstantInt::get(i64, (uint64_t)cv, true));
    } else if (dtype == AQP_DTYPE_FLOAT) {
      Type *f32 = Type::getFloatTy(llctx);
      Value *typed_ptr =
          b.CreateBitCast(col_data[col_idx], PointerType::getUnqual(f32));
      auto *vty = FixedVectorType::get(f32, VW);
      data_vec = b.CreateAlignedLoad(
          vty,
          b.CreateBitCast(b.CreateGEP(f32, typed_ptr, vi),
                          PointerType::getUnqual(vty)),
          Align(4));
      float cv = cmp->const_var->GetFloatValue();
      const_vec = ConstantVector::getSplat(ElementCount::getFixed(VW),
                                           ConstantFP::get(f32, (double)cv));
      is_fp = true;
    } else if (dtype == AQP_DTYPE_DOUBLE) {
      Type *f64 = Type::getDoubleTy(llctx);
      Value *typed_ptr =
          b.CreateBitCast(col_data[col_idx], PointerType::getUnqual(f64));
      auto *vty = FixedVectorType::get(f64, VW);
      data_vec = b.CreateAlignedLoad(
          vty,
          b.CreateBitCast(b.CreateGEP(f64, typed_ptr, vi),
                          PointerType::getUnqual(vty)),
          Align(8));
      float cv = cmp->const_var->GetFloatValue();
      const_vec = ConstantVector::getSplat(ElementCount::getFixed(VW),
                                           ConstantFP::get(f64, (double)cv));
      is_fp = true;
    } else {
      fn->eraseFromParent();
      return nullptr;
    }

    Value *expr_mask;
    auto et = cmp->GetSimplestExprType();
    if (is_fp) {
      switch (et) {
      case SimplestExprType::Equal:
        expr_mask = b.CreateFCmpOEQ(data_vec, const_vec);
        break;
      case SimplestExprType::NotEqual:
        expr_mask = b.CreateFCmpONE(data_vec, const_vec);
        break;
      case SimplestExprType::LessThan:
        expr_mask = b.CreateFCmpOLT(data_vec, const_vec);
        break;
      case SimplestExprType::GreaterThan:
        expr_mask = b.CreateFCmpOGT(data_vec, const_vec);
        break;
      case SimplestExprType::LessEqual:
        expr_mask = b.CreateFCmpOLE(data_vec, const_vec);
        break;
      case SimplestExprType::GreaterEqual:
        expr_mask = b.CreateFCmpOGE(data_vec, const_vec);
        break;
      default:
        fn->eraseFromParent();
        return nullptr;
      }
    } else {
      switch (et) {
      case SimplestExprType::Equal:
        expr_mask = b.CreateICmpEQ(data_vec, const_vec);
        break;
      case SimplestExprType::NotEqual:
        expr_mask = b.CreateICmpNE(data_vec, const_vec);
        break;
      case SimplestExprType::LessThan:
        expr_mask = b.CreateICmpSLT(data_vec, const_vec);
        break;
      case SimplestExprType::GreaterThan:
        expr_mask = b.CreateICmpSGT(data_vec, const_vec);
        break;
      case SimplestExprType::LessEqual:
        expr_mask = b.CreateICmpSLE(data_vec, const_vec);
        break;
      case SimplestExprType::GreaterEqual:
        expr_mask = b.CreateICmpSGE(data_vec, const_vec);
        break;
      default:
        fn->eraseFromParent();
        return nullptr;
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
  PHINode *sc_oc = b.CreatePHI(i64, 2, "sc_oc");
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
  Value *sc_mask_next =
      b.CreateAnd(sc_mask, b.CreateSub(sc_mask, ConstantInt::get(i64, 1)));
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
  // phase2_bb has exactly one predecessor (tail_bb).  toc already carries the
  // correct survivor count: it is initialized from voc when entering tail_bb,
  // then accumulated through the scalar tail loop.
  b.SetInsertPoint(phase2_bb);
  Value *phase1_count = toc;

  BasicBlock *p2_loop_bb = BasicBlock::Create(llctx, "p2_loop", fn);
  b.CreateBr(p2_loop_bb);

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
  case AQP_DTYPE_BOOL:
  case AQP_DTYPE_INT8:
    return 1;
  case AQP_DTYPE_INT16:
    return 2;
  case AQP_DTYPE_INT32:
  case AQP_DTYPE_FLOAT:
  case AQP_DTYPE_DATE:
    return 4;
  case AQP_DTYPE_INT64:
  case AQP_DTYPE_DOUBLE:
    return 8;
  case AQP_DTYPE_VARCHAR:
    return 16; // DuckDB string_t
  default:
    return 0;
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
static Function *
BuildProjectionFunction(LLVMContext &llctx, Module &mod,
                        const std::string &fn_name,
                        const std::vector<int> &col_mapping,
                        const std::vector<int32_t> &col_dtypes) {
  Type *i8p = PointerType::getUnqual(Type::getInt8Ty(llctx));
  Type *i32 = Type::getInt32Ty(llctx);
  Type *i64 = Type::getInt64Ty(llctx);
  Type *i64p = PointerType::getUnqual(i64);
  Type *i1 = Type::getInt1Ty(llctx);

  StructType *ColViewTy = StructType::get(llctx, {i8p, i64p, i32, i32});
  StructType *ChunkViewTy =
      StructType::get(llctx, {PointerType::getUnqual(ColViewTy), i64, i64});

  // int32_t fn(AQPChunkView *in, AQPChunkView *out)
  FunctionType *fn_ty = FunctionType::get(i32,
                                          {PointerType::getUnqual(ChunkViewTy),
                                           PointerType::getUnqual(ChunkViewTy)},
                                          false);
  Function *fn =
      Function::Create(fn_ty, Function::ExternalLinkage, fn_name, &mod);

  Value *in_arg = fn->getArg(0);
  in_arg->setName("in");
  Value *out_arg = fn->getArg(1);
  out_arg->setName("out");

  BasicBlock *entry = BasicBlock::Create(llctx, "entry", fn);
  IRBuilder<> b(entry);

  // Load in->nrows
  Value *in_nrows_ptr = b.CreateStructGEP(ChunkViewTy, in_arg, 1);
  Value *in_nrows = b.CreateLoad(i64, in_nrows_ptr, "in_nrows");

  // Load in->cols and out->cols base pointers
  Value *in_cols_pp = b.CreateStructGEP(ChunkViewTy, in_arg, 0);
  Value *in_cols =
      b.CreateLoad(PointerType::getUnqual(ColViewTy), in_cols_pp, "in_cols");
  Value *out_cols_pp = b.CreateStructGEP(ChunkViewTy, out_arg, 0);
  Value *out_cols =
      b.CreateLoad(PointerType::getUnqual(ColViewTy), out_cols_pp, "out_cols");

  // Declare llvm.memcpy intrinsic
  Function *memcpy_fn =
      Intrinsic::getDeclaration(&mod, Intrinsic::memcpy, {i8p, i8p, i64});

  // Validity size in bytes: ceil(nrows / 64) * 8
  Value *nrows_plus_63 = b.CreateAdd(in_nrows, ConstantInt::get(i64, 63));
  Value *nwords = b.CreateLShr(nrows_plus_63, ConstantInt::get(i64, 6));
  Value *val_bytes = b.CreateMul(nwords, ConstantInt::get(i64, 8), "val_bytes");

  // For each output column, memcpy actual data from input column
  for (size_t out_i = 0; out_i < col_mapping.size(); out_i++) {
    int in_i = col_mapping[out_i];
    if (in_i < 0)
      continue;

    unsigned elem_size = DtypeElemSize(col_dtypes[out_i]);
    if (elem_size == 0)
      continue; // unknown dtype, skip

    Value *src_col = b.CreateGEP(
        ColViewTy, in_cols, ConstantInt::get(i64, (uint64_t)in_i), "src_col");
    Value *dst_col = b.CreateGEP(
        ColViewTy, out_cols, ConstantInt::get(i64, (uint64_t)out_i), "dst_col");

    // Load source and dest data pointers
    Value *src_data =
        b.CreateLoad(i8p, b.CreateStructGEP(ColViewTy, src_col, 0), "src_data");
    Value *dst_data =
        b.CreateLoad(i8p, b.CreateStructGEP(ColViewTy, dst_col, 0), "dst_data");

    // memcpy(dst_data, src_data, nrows * elem_size)
    Value *data_bytes = b.CreateMul(
        in_nrows, ConstantInt::get(i64, (uint64_t)elem_size), "data_bytes");
    b.CreateCall(memcpy_fn, {dst_data, src_data, data_bytes,
                             ConstantInt::getFalse(llctx)});

    // Copy validity: if src validity is not null, memcpy it
    Value *src_val =
        b.CreateLoad(i64p, b.CreateStructGEP(ColViewTy, src_col, 1), "src_val");
    Value *dst_val =
        b.CreateLoad(i64p, b.CreateStructGEP(ColViewTy, dst_col, 1), "dst_val");
    Value *src_val_nonnull =
        b.CreateICmpNE(b.CreatePtrToInt(src_val, i64), ConstantInt::get(i64, 0),
                       "val_nonnull");

    // Conditional memcpy for validity
    BasicBlock *copy_val_bb =
        BasicBlock::Create(llctx, "copy_val_" + std::to_string(out_i), fn);
    BasicBlock *next_col_bb =
        BasicBlock::Create(llctx, "next_col_" + std::to_string(out_i), fn);
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
// SIMD version of BuildPipelineFunction: vectorized filter + projection fusion.
// Phase 1: vector loop (VW rows at a time) — load, compare, extract matches,
//           copy projected columns for matching rows.
// Phase 2: scalar tail for remainder rows < VW.
// ---------------------------------------------------------------------------
static Function *BuildPipelineFunctionSIMD(
    LLVMContext &llctx, Module &mod, const std::string &fn_name,
    const std::vector<const AQPExpr *> &filter_exprs,
    const std::vector<int> &col_mapping, const std::vector<int32_t> &col_dtypes,
    const std::vector<ColSchema> &schema, unsigned VW) {
  Type *i8p = PointerType::getUnqual(Type::getInt8Ty(llctx));
  Type *i1 = Type::getInt1Ty(llctx);
  Type *i32 = Type::getInt32Ty(llctx);
  Type *i64 = Type::getInt64Ty(llctx);
  Type *i64p = PointerType::getUnqual(i64);

  StructType *ColViewTy = StructType::get(llctx, {i8p, i64p, i32, i32});
  StructType *ChunkViewTy =
      StructType::get(llctx, {PointerType::getUnqual(ColViewTy), i64, i64});

  FunctionType *fn_ty =
      FunctionType::get(i64,
                        {PointerType::getUnqual(ChunkViewTy),
                         PointerType::getUnqual(ChunkViewTy), i8p},
                        false);
  Function *fn =
      Function::Create(fn_ty, Function::ExternalLinkage, fn_name, &mod);

  Value *in_arg = fn->getArg(0);
  in_arg->setName("in");
  Value *out_arg = fn->getArg(1);
  out_arg->setName("out");
  Value *state_arg = fn->getArg(2);
  state_arg->setName("state");
  (void)state_arg;

  BasicBlock *entry_bb = BasicBlock::Create(llctx, "entry", fn);
  BasicBlock *vec_loop_bb = BasicBlock::Create(llctx, "vec_loop", fn);
  BasicBlock *vec_body_bb = BasicBlock::Create(llctx, "vec_body", fn);
  BasicBlock *vec_scatter_bb = BasicBlock::Create(llctx, "vec_scatter", fn);
  BasicBlock *vec_next_bb = BasicBlock::Create(llctx, "vec_next", fn);
  BasicBlock *tail_bb = BasicBlock::Create(llctx, "tail", fn);
  BasicBlock *tail_body_bb = BasicBlock::Create(llctx, "tail_body", fn);
  BasicBlock *tail_write_bb = BasicBlock::Create(llctx, "tail_write", fn);
  BasicBlock *tail_next_bb = BasicBlock::Create(llctx, "tail_next", fn);
  BasicBlock *exit_bb = BasicBlock::Create(llctx, "exit", fn);

  IRBuilder<> b(entry_bb);

  // Load nrows, input/output column pointers
  Value *nrows_ptr = b.CreateStructGEP(ChunkViewTy, in_arg, 1);
  Value *nrows = b.CreateLoad(i64, nrows_ptr, "nrows");
  Value *in_cols_pp = b.CreateStructGEP(ChunkViewTy, in_arg, 0);
  Value *in_cols =
      b.CreateLoad(PointerType::getUnqual(ColViewTy), in_cols_pp, "in_cols");
  Value *out_cols_pp = b.CreateStructGEP(ChunkViewTy, out_arg, 0);
  Value *out_cols =
      b.CreateLoad(PointerType::getUnqual(ColViewTy), out_cols_pp, "out_cols");

  // Pre-load input column data pointers
  std::vector<Value *> col_data(schema.size());
  std::vector<Value *> col_validity(schema.size());
  for (size_t ci = 0; ci < schema.size(); ci++) {
    Value *col_i = b.CreateGEP(ColViewTy, in_cols, ConstantInt::get(i64, ci));
    col_data[ci] = b.CreateLoad(i8p, b.CreateStructGEP(ColViewTy, col_i, 0),
                                "cd" + std::to_string(ci));
    col_validity[ci] =
        b.CreateLoad(i64p, b.CreateStructGEP(ColViewTy, col_i, 1),
                     "cv" + std::to_string(ci));
  }

  // Pre-load output column data pointers
  std::vector<Value *> out_data_ptrs(col_mapping.size());
  for (size_t oi = 0; oi < col_mapping.size(); oi++) {
    Value *col_i = b.CreateGEP(ColViewTy, out_cols, ConstantInt::get(i64, oi));
    out_data_ptrs[oi] = b.CreateLoad(
        i8p, b.CreateStructGEP(ColViewTy, col_i, 0), "od" + std::to_string(oi));
  }

  // vec_limit = nrows & ~(VW-1)
  Value *vw_const = ConstantInt::get(i64, VW);
  Value *vw_mask = ConstantInt::get(i64, ~(uint64_t)(VW - 1));
  Value *vec_limit = b.CreateAnd(nrows, vw_mask, "vec_limit");

  b.CreateBr(vec_loop_bb);

  // ========== PHASE 1: Vectorized main loop ==========
  b.SetInsertPoint(vec_loop_bb);
  PHINode *vi = b.CreatePHI(i64, 2, "vi");
  PHINode *voc = b.CreatePHI(i64, 2, "voc"); // vectorized out_count
  vi->addIncoming(ConstantInt::get(i64, 0), entry_bb);
  voc->addIncoming(ConstantInt::get(i64, 0), entry_bb);
  b.CreateCondBr(b.CreateICmpULT(vi, vec_limit), vec_body_bb, tail_bb);

  // vec_body: evaluate all filter expressions as vector comparisons
  b.SetInsertPoint(vec_body_bb);

  // Evaluate each filter expression — produce <VW x i1> mask
  Value *combined_mask = ConstantVector::getSplat(ElementCount::getFixed(VW),
                                                  ConstantInt::getTrue(llctx));

  for (const AQPExpr *e : filter_exprs) {
    auto *cmp = dynamic_cast<const SimplestVarConstComparison *>(e);
    if (!cmp) {
      fn->eraseFromParent();
      return nullptr;
    }

    // Find column index in schema
    int ci = -1;
    for (int j = 0; j < (int)schema.size(); j++) {
      if (schema[j].table_idx == cmp->attr->GetTableIndex() &&
          schema[j].col_idx == cmp->attr->GetColumnIndex()) {
        ci = j;
        break;
      }
    }
    if (ci < 0) {
      fn->eraseFromParent();
      return nullptr;
    }

    int32_t dtype = schema[ci].dtype;
    Value *data_ptr = col_data[ci];

    // Vector load of VW elements from the column
    Value *data_vec = nullptr;
    Value *const_vec = nullptr;
    bool is_fp = false;

    if (dtype == AQP_DTYPE_INT32 || dtype == AQP_DTYPE_DATE) {
      auto *vty = FixedVectorType::get(i32, VW);
      data_vec = b.CreateAlignedLoad(
          vty,
          b.CreateBitCast(
              b.CreateGEP(
                  i32, b.CreateBitCast(data_ptr, PointerType::getUnqual(i32)),
                  vi),
              PointerType::getUnqual(vty)),
          Align(4));
      const_vec = ConstantVector::getSplat(
          ElementCount::getFixed(VW),
          ConstantInt::get(
              i32, (uint64_t)(uint32_t)cmp->const_var->GetIntValue(), true));
    } else if (dtype == AQP_DTYPE_INT64) {
      auto *vty = FixedVectorType::get(i64, VW);
      data_vec = b.CreateAlignedLoad(
          vty,
          b.CreateBitCast(
              b.CreateGEP(
                  i64, b.CreateBitCast(data_ptr, PointerType::getUnqual(i64)),
                  vi),
              PointerType::getUnqual(vty)),
          Align(8));
      const_vec = ConstantVector::getSplat(
          ElementCount::getFixed(VW),
          ConstantInt::get(i64, (uint64_t)cmp->const_var->GetIntValue(), true));
    } else if (dtype == AQP_DTYPE_DOUBLE) {
      Type *f64 = Type::getDoubleTy(llctx);
      auto *vty = FixedVectorType::get(f64, VW);
      data_vec = b.CreateAlignedLoad(
          vty,
          b.CreateBitCast(
              b.CreateGEP(
                  f64, b.CreateBitCast(data_ptr, PointerType::getUnqual(f64)),
                  vi),
              PointerType::getUnqual(vty)),
          Align(8));
      const_vec = ConstantVector::getSplat(
          ElementCount::getFixed(VW),
          ConstantFP::get(f64, cmp->const_var->GetFloatValue()));
      is_fp = true;
    } else {
      fn->eraseFromParent();
      return nullptr;
    }

    // Vector comparison
    auto et = cmp->GetSimplestExprType();
    Value *expr_mask = nullptr;
    if (is_fp) {
      switch (et) {
      case SimplestExprType::Equal:
        expr_mask = b.CreateFCmpOEQ(data_vec, const_vec);
        break;
      case SimplestExprType::NotEqual:
        expr_mask = b.CreateFCmpONE(data_vec, const_vec);
        break;
      case SimplestExprType::LessThan:
        expr_mask = b.CreateFCmpOLT(data_vec, const_vec);
        break;
      case SimplestExprType::GreaterThan:
        expr_mask = b.CreateFCmpOGT(data_vec, const_vec);
        break;
      case SimplestExprType::LessEqual:
        expr_mask = b.CreateFCmpOLE(data_vec, const_vec);
        break;
      case SimplestExprType::GreaterEqual:
        expr_mask = b.CreateFCmpOGE(data_vec, const_vec);
        break;
      default:
        fn->eraseFromParent();
        return nullptr;
      }
    } else {
      switch (et) {
      case SimplestExprType::Equal:
        expr_mask = b.CreateICmpEQ(data_vec, const_vec);
        break;
      case SimplestExprType::NotEqual:
        expr_mask = b.CreateICmpNE(data_vec, const_vec);
        break;
      case SimplestExprType::LessThan:
        expr_mask = b.CreateICmpSLT(data_vec, const_vec);
        break;
      case SimplestExprType::GreaterThan:
        expr_mask = b.CreateICmpSGT(data_vec, const_vec);
        break;
      case SimplestExprType::LessEqual:
        expr_mask = b.CreateICmpSLE(data_vec, const_vec);
        break;
      case SimplestExprType::GreaterEqual:
        expr_mask = b.CreateICmpSGE(data_vec, const_vec);
        break;
      default:
        fn->eraseFromParent();
        return nullptr;
      }
    }

    // AND validity mask
    Value *val_ptr = col_validity[ci];
    Value *val_nonnull = b.CreateICmpNE(b.CreatePtrToInt(val_ptr, i64),
                                        ConstantInt::get(i64, 0));
    BasicBlock *has_val_bb = BasicBlock::Create(llctx, "has_val", fn);
    BasicBlock *no_val_bb = BasicBlock::Create(llctx, "no_val", fn);
    BasicBlock *val_done_bb = BasicBlock::Create(llctx, "val_done", fn);
    b.CreateCondBr(val_nonnull, has_val_bb, no_val_bb);

    b.SetInsertPoint(has_val_bb);
    Value *validity_word = b.CreateLoad(
        i64,
        b.CreateGEP(i64, val_ptr, b.CreateLShr(vi, ConstantInt::get(i64, 6))));
    Value *shifted = b.CreateLShr(
        validity_word, b.CreateAnd(vi, ConstantInt::get(i64, VW - 1)));
    unsigned mask_bits = (1u << VW) - 1;
    Value *vbits = b.CreateAnd(shifted, ConstantInt::get(i64, mask_bits));
    auto *mask_int_ty = IntegerType::get(llctx, VW);
    Value *vbits_narrow = b.CreateTrunc(vbits, mask_int_ty);
    Value *val_mask =
        b.CreateBitCast(vbits_narrow, FixedVectorType::get(i1, VW));
    Value *expr_mask_valid = b.CreateAnd(expr_mask, val_mask);
    b.CreateBr(val_done_bb);

    b.SetInsertPoint(no_val_bb);
    b.CreateBr(val_done_bb);

    b.SetInsertPoint(val_done_bb);
    PHINode *final_expr_mask = b.CreatePHI(expr_mask->getType(), 2);
    final_expr_mask->addIncoming(expr_mask_valid, has_val_bb);
    final_expr_mask->addIncoming(expr_mask, no_val_bb);

    combined_mask = b.CreateAnd(combined_mask, final_expr_mask);
  }

  // If no filter expressions, all rows pass
  if (filter_exprs.empty()) {
    combined_mask = ConstantVector::getSplat(ElementCount::getFixed(VW),
                                             ConstantInt::getTrue(llctx));
  }

  // Scatter: extract matching indices and copy projected columns
  b.SetInsertPoint(vec_scatter_bb);
  auto *mask_int_ty = IntegerType::get(llctx, VW);
  Value *sc_mask = b.CreateBitCast(combined_mask, mask_int_ty, "sc_mask");
  Value *sc_mask_wide = b.CreateZExt(sc_mask, i64);

  Function *cttz_fn = Intrinsic::getDeclaration(&mod, Intrinsic::cttz, {i64});

  // For each set bit: copy all projected columns
  BasicBlock *scatter_loop_bb = BasicBlock::Create(llctx, "scatter_loop", fn);
  BasicBlock *scatter_body_bb = BasicBlock::Create(llctx, "scatter_body", fn);
  BasicBlock *scatter_done_bb = BasicBlock::Create(llctx, "scatter_done", fn);

  b.CreateBr(scatter_loop_bb);

  b.SetInsertPoint(scatter_loop_bb);
  PHINode *s_mask_phi = b.CreatePHI(i64, 2, "s_mask");
  PHINode *s_oc = b.CreatePHI(i64, 2, "s_oc");
  s_mask_phi->addIncoming(sc_mask_wide, vec_scatter_bb);
  s_oc->addIncoming(voc, vec_scatter_bb);
  b.CreateCondBr(b.CreateICmpEQ(s_mask_phi, ConstantInt::get(i64, 0)),
                 scatter_done_bb, scatter_body_bb);

  b.SetInsertPoint(scatter_body_bb);
  Value *tz =
      b.CreateCall(cttz_fn, {s_mask_phi, ConstantInt::getTrue(llctx)}, "tz");
  Value *row_idx_in_vec = b.CreateZExt(tz, i64);
  Value *src_row = b.CreateAdd(vi, row_idx_in_vec, "src_row");

  // Copy projected columns for this matching row
  for (size_t oi = 0; oi < col_mapping.size(); oi++) {
    int in_i = col_mapping[oi];
    if (in_i < 0)
      continue;
    unsigned elem_size = DtypeElemSize(col_dtypes[oi]);
    if (elem_size == 0)
      continue;

    Value *src =
        b.CreateGEP(Type::getInt8Ty(llctx), col_data[in_i],
                    b.CreateMul(src_row, ConstantInt::get(i64, elem_size)));
    Value *dst =
        b.CreateGEP(Type::getInt8Ty(llctx), out_data_ptrs[oi],
                    b.CreateMul(s_oc, ConstantInt::get(i64, elem_size)));
    b.CreateMemCpy(dst, MaybeAlign(1), src, MaybeAlign(1),
                   ConstantInt::get(i64, elem_size));
  }

  // Clear lowest set bit and advance
  Value *new_mask = b.CreateAnd(
      s_mask_phi, b.CreateSub(s_mask_phi, ConstantInt::get(i64, 1)));
  Value *new_oc = b.CreateAdd(s_oc, ConstantInt::get(i64, 1));
  s_mask_phi->addIncoming(new_mask, scatter_body_bb);
  s_oc->addIncoming(new_oc, scatter_body_bb);
  b.CreateBr(scatter_loop_bb);

  b.SetInsertPoint(scatter_done_bb);
  PHINode *vec_oc_out = b.CreatePHI(i64, 2, "vec_oc_out");
  vec_oc_out->addIncoming(voc, vec_scatter_bb);
  vec_oc_out->addIncoming(new_oc, scatter_body_bb);

  Value *vi_next = b.CreateAdd(vi, vw_const, "vi_next");
  vi->addIncoming(vi_next, scatter_done_bb);
  voc->addIncoming(vec_oc_out, scatter_done_bb);
  b.CreateBr(vec_loop_bb);

  // Wire vec_body → vec_scatter
  {
    BasicBlock *last_bb = b.GetInsertBlock();
    // The combined_mask computation may have split into multiple BBs
    // Find the BB that computed combined_mask last
    b.SetInsertPoint(vec_scatter_bb);
    // Move vec_scatter_bb after the last combined_mask BB
    // Need to branch from last combined_mask BB to vec_scatter_bb
  }
  // We need to fix this: the vec_body block needs to fall through to
  // vec_scatter Get the last BB created during mask computation
  IRBuilder<> b_fix(vec_scatter_bb);
  // The issue is that the mask computation creates branches (for validity).
  // The val_done_bb needs to branch to vec_scatter_bb instead of falling
  // through. Let's redirect val_done_bb to vec_scatter_bb. Actually, the last
  // val_done_bb was the one from the last expression. We need to branch from
  // there to vec_scatter_bb. Since the builder is now set to scatter_loop_bb
  // area, let's fix the flow: After all expressions, the last val_done_bb
  // should br to vec_scatter_bb. We'll add a terminator to val_done_bb at the
  // end.

  // ========== PHASE 2: Scalar tail ==========
  b.SetInsertPoint(tail_bb);
  PHINode *tail_i = b.CreatePHI(i64, 2, "ti");
  PHINode *tail_oc = b.CreatePHI(i64, 2, "toc");
  tail_i->addIncoming(vec_limit, tail_bb);
  tail_oc->addIncoming(vec_oc_out, tail_bb); // will be fixed below
  // Need to also add incoming from scatter_done for the loop backedge
  b.CreateCondBr(b.CreateICmpULT(tail_i, nrows), tail_body_bb, exit_bb);

  // Use CompileCtx for scalar tail expression evaluation
  CompileCtx cc(llctx, mod, schema, in_arg, out_arg);
  cc.b.SetInsertPoint(tail_body_bb);
  cc.col_data = col_data;
  cc.col_validity = col_validity;
  cc.row_idx = tail_i;

  Value *match = ConstantInt::getTrue(llctx);
  for (const AQPExpr *e : filter_exprs) {
    Value *res = EmitExpr(cc, e);
    match = cc.b.CreateAnd(match, res);
  }
  BasicBlock *tail_cond_bb = cc.b.GetInsertBlock();
  cc.b.CreateCondBr(match, tail_write_bb, tail_next_bb);

  cc.b.SetInsertPoint(tail_write_bb);
  for (size_t oi = 0; oi < col_mapping.size(); oi++) {
    int in_i = col_mapping[oi];
    if (in_i < 0)
      continue;
    unsigned elem_size = DtypeElemSize(col_dtypes[oi]);
    if (elem_size == 0)
      continue;

    Value *src = cc.b.CreateGEP(
        Type::getInt8Ty(llctx), col_data[in_i],
        cc.b.CreateMul(tail_i, ConstantInt::get(i64, elem_size)));
    Value *dst = cc.b.CreateGEP(
        Type::getInt8Ty(llctx), out_data_ptrs[oi],
        cc.b.CreateMul(tail_oc, ConstantInt::get(i64, elem_size)));
    cc.b.CreateMemCpy(dst, MaybeAlign(1), src, MaybeAlign(1),
                      ConstantInt::get(i64, elem_size));
  }
  Value *toc1 = cc.b.CreateAdd(tail_oc, ConstantInt::get(i64, 1));
  cc.b.CreateBr(tail_next_bb);

  cc.b.SetInsertPoint(tail_next_bb);
  PHINode *toc_next = cc.b.CreatePHI(i64, 2, "toc_next");
  toc_next->addIncoming(tail_oc, tail_cond_bb);
  toc_next->addIncoming(toc1, tail_write_bb);
  Value *ti_next = cc.b.CreateAdd(tail_i, ConstantInt::get(i64, 1));
  tail_i->addIncoming(ti_next, tail_next_bb);
  tail_oc->addIncoming(toc_next, tail_next_bb);
  cc.b.CreateBr(tail_bb);

  // Exit: store output nrows and return count
  b.SetInsertPoint(exit_bb);
  PHINode *final_oc = b.CreatePHI(i64, 2, "final_oc");
  final_oc->addIncoming(tail_oc, tail_bb); // from tail loop header (not taken)
  final_oc->addIncoming(vec_oc_out, scatter_done_bb); // if vec_limit == nrows
  Value *out_nrows_ptr = b.CreateStructGEP(ChunkViewTy, out_arg, 1);
  b.CreateStore(final_oc, out_nrows_ptr);
  b.CreateRet(final_oc);

  // Fix: the last val_done_bb from the expression loop needs to branch to
  // vec_scatter_bb. We need to track the last val_done_bb.
  // Unfortunately we don't have a clean reference. Walk the function's BB list
  // to find the val_done BBs and redirect the last one.
  // Alternative: we collect val_done BBs in a vector during expression eval.
  // For now, we rely on the fact that the IR builder was left in val_done_bb
  // after the last expression. We can find the last BB that doesn't have a
  // terminator and add a branch to vec_scatter_bb.
  // Since we moved the builder around, let's just find any BB without
  // terminator.
  for (auto &bb : *fn) {
    if (bb.getTerminator() == nullptr && bb.getName().starts_with("val_done")) {
      IRBuilder<> b_term(&bb);
      b_term.CreateBr(vec_scatter_bb);
    }
  }

  // Also handle case where no filter expressions: vec_body needs to br to
  // scatter
  if (filter_exprs.empty()) {
    IRBuilder<> b_vb(vec_body_bb);
    if (!vec_body_bb->getTerminator())
      b_vb.CreateBr(vec_scatter_bb);
  }

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
static Function *BuildPipelineFunction(
    LLVMContext &llctx, Module &mod, const std::string &fn_name,
    const std::vector<const AQPExpr *> &filter_exprs,
    const std::vector<int> &col_mapping, const std::vector<int32_t> &col_dtypes,
    const std::vector<ColSchema> &schema,
    ParamsBuilder *pb = nullptr) {
  Type *i8p = PointerType::getUnqual(Type::getInt8Ty(llctx));
  Type *i32 = Type::getInt32Ty(llctx);
  Type *i64 = Type::getInt64Ty(llctx);
  Type *i64p = PointerType::getUnqual(i64);

  StructType *ColViewTy = StructType::get(llctx, {i8p, i64p, i32, i32});
  StructType *ChunkViewTy =
      StructType::get(llctx, {PointerType::getUnqual(ColViewTy), i64, i64});

  // int64_t fn(AQPChunkView *in, AQPChunkView *out, i8 *state)
  FunctionType *fn_ty =
      FunctionType::get(i64,
                        {PointerType::getUnqual(ChunkViewTy),
                         PointerType::getUnqual(ChunkViewTy), i8p},
                        false);
  Function *fn =
      Function::Create(fn_ty, Function::ExternalLinkage, fn_name, &mod);

  Value *in_arg = fn->getArg(0);
  in_arg->setName("in");
  Value *out_arg = fn->getArg(1);
  out_arg->setName("out");
  Value *state_arg = fn->getArg(2);
  state_arg->setName("state");

  BasicBlock *entry_bb = BasicBlock::Create(llctx, "entry", fn);
  BasicBlock *loop_bb = BasicBlock::Create(llctx, "loop", fn);
  BasicBlock *body_bb = BasicBlock::Create(llctx, "body", fn);
  BasicBlock *write_bb = BasicBlock::Create(llctx, "write", fn);
  BasicBlock *next_bb = BasicBlock::Create(llctx, "next", fn);
  BasicBlock *exit_bb = BasicBlock::Create(llctx, "exit", fn);

  // Use CompileCtx for filter expression emission (reuses EmitExpr
  // infrastructure) We use a dummy AQPSelView since CompileCtx requires it, but
  // we don't use it
  CompileCtx cc(llctx, mod, schema, in_arg,
                out_arg /* repurposed as sel_arg placeholder */);
  cc.b.SetInsertPoint(entry_bb);

  if (pb) {
    cc.template_mode = true;
    cc.params_builder = pb;
    cc.LoadParamsFromThreadLocal();
  }

  // Load in->nrows
  Value *nrows = cc.b.CreateLoad(
      i64, cc.b.CreateStructGEP(ChunkViewTy, in_arg, 1), "nrows");

  // Load input column data + validity (for filter expression evaluation)
  cc.col_data.resize(schema.size());
  cc.col_validity.resize(schema.size());
  for (size_t i = 0; i < schema.size(); i++) {
    cc.col_data[i] = cc.LoadColData((unsigned)i);
    cc.col_validity[i] = cc.LoadColValidity((unsigned)i);
  }

  // Load output column data pointers
  Value *out_cols_pp = cc.b.CreateStructGEP(ChunkViewTy, out_arg, 0);
  Value *out_cols = cc.b.CreateLoad(PointerType::getUnqual(ColViewTy),
                                    out_cols_pp, "out_cols");

  std::vector<Value *> out_data_ptrs;
  for (size_t oi = 0; oi < col_mapping.size(); oi++) {
    Value *col_i =
        cc.b.CreateGEP(ColViewTy, out_cols, ConstantInt::get(i64, oi));
    out_data_ptrs.push_back(
        cc.b.CreateLoad(i8p, cc.b.CreateStructGEP(ColViewTy, col_i, 0),
                        "out_data_" + std::to_string(oi)));
  }

  cc.b.CreateBr(loop_bb);

  // Loop header
  cc.b.SetInsertPoint(loop_bb);
  PHINode *row_i = cc.b.CreatePHI(i64, 2, "i");
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

  // Write: copy projected columns for this row to output (typed load/store)
  cc.b.SetInsertPoint(write_bb);
  for (size_t oi = 0; oi < col_mapping.size(); oi++) {
    int in_i = col_mapping[oi];
    if (in_i < 0)
      continue;
    unsigned elem_size = DtypeElemSize(col_dtypes[oi]);
    if (elem_size == 0)
      continue;

    int32_t dt = col_dtypes[oi];
    Type *elem_ty = nullptr;
    if (dt == AQP_DTYPE_INT32 || dt == AQP_DTYPE_DATE)
      elem_ty = Type::getInt32Ty(llctx);
    else if (dt == AQP_DTYPE_INT64)
      elem_ty = Type::getInt64Ty(llctx);
    else if (dt == AQP_DTYPE_FLOAT)
      elem_ty = Type::getFloatTy(llctx);
    else if (dt == AQP_DTYPE_DOUBLE)
      elem_ty = Type::getDoubleTy(llctx);
    else if (dt == AQP_DTYPE_INT16)
      elem_ty = Type::getInt16Ty(llctx);
    else if (dt == AQP_DTYPE_BOOL || dt == AQP_DTYPE_INT8)
      elem_ty = Type::getInt8Ty(llctx);

    if (elem_ty) {
      Type *ptr_ty = PointerType::getUnqual(elem_ty);
      Value *src_typed =
          cc.b.CreateBitCast(cc.col_data[in_i], ptr_ty);
      Value *val =
          cc.b.CreateLoad(elem_ty, cc.b.CreateGEP(elem_ty, src_typed, row_i));
      Value *dst_typed = cc.b.CreateBitCast(out_data_ptrs[oi], ptr_ty);
      cc.b.CreateStore(val, cc.b.CreateGEP(elem_ty, dst_typed, out_count));
    } else if (dt == AQP_DTYPE_VARCHAR) {
      // Safe VARCHAR copy: call aqp_copy_string which deep-copies
      // non-inline strings into the output Vector's string heap.
      // Signature: void aqp_copy_string(void *dst_data, void *src_data,
      //     uint64_t dst_row, uint64_t src_row, void *state, uint32_t col_idx)
      FunctionType *copy_ft = FunctionType::get(
          Type::getVoidTy(llctx),
          {i8p, i8p, i64, i64, i8p, Type::getInt32Ty(llctx)}, false);
      FunctionCallee copy_fn =
          mod.getOrInsertFunction("aqp_copy_string", copy_ft);
      cc.b.CreateCall(copy_fn,
                      {out_data_ptrs[oi], cc.col_data[in_i],
                       out_count, row_i, state_arg,
                       ConstantInt::get(Type::getInt32Ty(llctx), (uint32_t)oi)});
    } else {
      // Unknown type: fall back to memcpy
      Value *src = cc.b.CreateGEP(
          Type::getInt8Ty(llctx), cc.col_data[in_i],
          cc.b.CreateMul(row_i, ConstantInt::get(i64, elem_size)));
      Value *dst = cc.b.CreateGEP(
          Type::getInt8Ty(llctx), out_data_ptrs[oi],
          cc.b.CreateMul(out_count, ConstantInt::get(i64, elem_size)));
      cc.b.CreateMemCpy(dst, MaybeAlign(1), src, MaybeAlign(1),
                        ConstantInt::get(i64, elem_size));
    }
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

// Aggregate JIT disabled: JOB has no aggregate-heavy queries (only MIN on
// VARCHAR columns).  Saves ~400 lines of LLVM codegen from compilation.
// Re-enable with -DDISABLE_AGG_JIT=0 for TPC-H or other agg-heavy benchmarks.
#ifndef DISABLE_AGG_JIT
#define DISABLE_AGG_JIT 1
#endif
#if !DISABLE_AGG_JIT
// AggOp is now in the public header (aqp_jit::AggOp)
using aqp_jit::AggOp;

// ---------------------------------------------------------------------------
static bool AllAggOpsSIMDFriendly(const std::vector<AggOp> &ops) {
  for (const auto &op : ops) {
    if (op.agg_type == 6)
      continue; // CountStar always works
    if (op.dtype != AQP_DTYPE_INT32 && op.dtype != AQP_DTYPE_DATE &&
        op.dtype != AQP_DTYPE_INT64 && op.dtype != AQP_DTYPE_FLOAT &&
        op.dtype != AQP_DTYPE_DOUBLE)
      return false;
    // Only SUM(3), COUNT(5), MIN(1), MAX(2) — not AVG(4) yet
    if (op.agg_type == 4)
      return false;
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
static Function *BuildAggUpdateFunctionSIMD(
    LLVMContext &llctx, Module &mod, const std::string &fn_name,
    const std::vector<AggOp> &agg_ops, unsigned total_state_size,
    const std::vector<ColSchema> &schema, unsigned VW) {
  Type *i8 = Type::getInt8Ty(llctx);
  Type *i8p = PointerType::getUnqual(i8);
  Type *i32 = Type::getInt32Ty(llctx);
  Type *i64 = Type::getInt64Ty(llctx);
  Type *f32 = Type::getFloatTy(llctx);
  Type *f64 = Type::getDoubleTy(llctx);
  Type *i64p = PointerType::getUnqual(i64);
  Type *voidTy = Type::getVoidTy(llctx);

  StructType *ColViewTy = StructType::get(llctx, {i8p, i64p, i32, i32});
  StructType *ChunkViewTy =
      StructType::get(llctx, {PointerType::getUnqual(ColViewTy), i64, i64});

  FunctionType *fn_ty = FunctionType::get(
      voidTy, {PointerType::getUnqual(ChunkViewTy), i8p}, false);
  Function *fn =
      Function::Create(fn_ty, Function::ExternalLinkage, fn_name, &mod);

  Value *in_arg = fn->getArg(0);
  in_arg->setName("in");
  Value *state_arg = fn->getArg(1);
  state_arg->setName("state");

  BasicBlock *entry_bb = BasicBlock::Create(llctx, "entry", fn);
  BasicBlock *vec_loop_bb = BasicBlock::Create(llctx, "vec_loop", fn);
  BasicBlock *vec_body_bb = BasicBlock::Create(llctx, "vec_body", fn);
  BasicBlock *vec_next_bb = BasicBlock::Create(llctx, "vec_next", fn);
  BasicBlock *tail_bb = BasicBlock::Create(llctx, "tail", fn);
  BasicBlock *tail_body_bb = BasicBlock::Create(llctx, "tail_body", fn);
  BasicBlock *tail_next_bb = BasicBlock::Create(llctx, "tail_next", fn);
  BasicBlock *exit_bb = BasicBlock::Create(llctx, "exit", fn);

  IRBuilder<> b(entry_bb);

  Value *nrows =
      b.CreateLoad(i64, b.CreateStructGEP(ChunkViewTy, in_arg, 1), "nrows");
  Value *cols = b.CreateLoad(PointerType::getUnqual(ColViewTy),
                             b.CreateStructGEP(ChunkViewTy, in_arg, 0), "cols");

  std::map<int, Value *> col_data;
  for (const auto &op : agg_ops) {
    if (op.col_idx >= 0 && col_data.find(op.col_idx) == col_data.end()) {
      Value *col_i = b.CreateGEP(ColViewTy, cols,
                                 ConstantInt::get(i64, (uint64_t)op.col_idx));
      col_data[op.col_idx] =
          b.CreateLoad(i8p, b.CreateStructGEP(ColViewTy, col_i, 0));
    }
  }

  Value *vec_limit = b.CreateAnd(
      nrows, ConstantInt::get(i64, ~(uint64_t)(VW - 1)), "vec_limit");

  // Helper: get scalar and vector types per dtype
  auto getElemType = [&](int32_t dtype) -> Type * {
    switch (dtype) {
    case AQP_DTYPE_INT32:
    case AQP_DTYPE_DATE:
      return i32;
    case AQP_DTYPE_INT64:
      return i64;
    case AQP_DTYPE_FLOAT:
      return f32;
    case AQP_DTYPE_DOUBLE:
      return f64;
    default:
      return i32;
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
      va.init_vec =
          va.fp ? ConstantVector::getSplat(ElementCount::getFixed(VW),
                                           ConstantFP::get(va.elem_ty, 0.0))
                : ConstantVector::getSplat(ElementCount::getFixed(VW),
                                           ConstantInt::get(va.elem_ty, 0));
    } else if (op.agg_type == 1) { // MIN
      va.init_vec =
          va.fp ? ConstantVector::getSplat(
                      ElementCount::getFixed(VW),
                      ConstantFP::getInfinity(va.elem_ty, false))
                : ConstantVector::getSplat(
                      ElementCount::getFixed(VW),
                      ConstantInt::get(va.elem_ty,
                                       va.elem_ty == i64
                                           ? APInt(64, (uint64_t)INT64_MAX)
                                           : APInt(32, (uint64_t)INT32_MAX)));
    } else if (op.agg_type == 2) { // MAX
      va.init_vec =
          va.fp ? ConstantVector::getSplat(
                      ElementCount::getFixed(VW),
                      ConstantFP::getInfinity(va.elem_ty, true))
                : ConstantVector::getSplat(
                      ElementCount::getFixed(VW),
                      ConstantInt::get(
                          va.elem_ty,
                          va.elem_ty == i64
                              ? APInt(64, (uint64_t)INT64_MIN)
                              : APInt(32, (uint64_t)(uint32_t)INT32_MIN)));
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

  std::vector<PHINode *> vec_acc_phis;
  std::vector<Value *> vec_acc_vals; // current accumulator values
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
      Value *ones =
          va.fp ? ConstantVector::getSplat(ElementCount::getFixed(VW),
                                           ConstantFP::get(va.elem_ty, 1.0))
                : ConstantVector::getSplat(ElementCount::getFixed(VW),
                                           ConstantInt::get(va.elem_ty, 1));
      vec_acc_vals[ai] = va.fp ? b.CreateFAdd(vec_acc_vals[ai], ones)
                               : b.CreateAdd(vec_acc_vals[ai], ones);
      continue;
    }
    if (op.col_idx < 0)
      continue;

    // Load <VW x elem_ty> from column
    Value *typed_ptr = b.CreateBitCast(col_data[op.col_idx],
                                       PointerType::getUnqual(va.elem_ty));
    Value *vec_ptr = b.CreateBitCast(b.CreateGEP(va.elem_ty, typed_ptr, vi),
                                     PointerType::getUnqual(va.vty));
    unsigned elem_bytes = va.elem_ty->getPrimitiveSizeInBits() / 8;
    Value *data_vec =
        b.CreateAlignedLoad(va.vty, vec_ptr, Align(elem_bytes), "dvec");

    switch (op.agg_type) {
    case 3: // SUM
      vec_acc_vals[ai] = va.fp ? b.CreateFAdd(vec_acc_vals[ai], data_vec)
                               : b.CreateAdd(vec_acc_vals[ai], data_vec);
      break;
    case 5: { // COUNT non-null (count all for now)
      Value *ones =
          va.fp ? ConstantVector::getSplat(ElementCount::getFixed(VW),
                                           ConstantFP::get(va.elem_ty, 1.0))
                : ConstantVector::getSplat(ElementCount::getFixed(VW),
                                           ConstantInt::get(va.elem_ty, 1));
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
  std::vector<Value *> scalar_accs;
  for (size_t ai = 0; ai < vaccs.size(); ai++) {
    Value *vacc = vec_acc_vals[ai];
    Value *result = b.CreateExtractElement(vacc, (uint64_t)0);
    for (unsigned k = 1; k < VW; k++) {
      Value *elem = b.CreateExtractElement(vacc, (uint64_t)k);
      bool fp = vaccs[ai].fp;
      switch (agg_ops[ai].agg_type) {
      case 3:
      case 5:
      case 6:
        result = fp ? b.CreateFAdd(result, elem) : b.CreateAdd(result, elem);
        break;
      case 1: {
        Value *cmp =
            fp ? b.CreateFCmpOLT(elem, result) : b.CreateICmpSLT(elem, result);
        result = b.CreateSelect(cmp, elem, result);
        break;
      }
      case 2: {
        Value *cmp =
            fp ? b.CreateFCmpOGT(elem, result) : b.CreateICmpSGT(elem, result);
        result = b.CreateSelect(cmp, elem, result);
        break;
      }
      default:
        break;
      }
    }
    scalar_accs.push_back(result);
  }

  // Scalar tail loop
  PHINode *ti = b.CreatePHI(i64, 2, "ti");
  ti->addIncoming(vec_limit, vec_loop_bb);
  std::vector<PHINode *> tail_acc_phis;
  for (size_t ai = 0; ai < scalar_accs.size(); ai++) {
    PHINode *phi = b.CreatePHI(vaccs[ai].elem_ty, 2, "tacc");
    phi->addIncoming(scalar_accs[ai], vec_loop_bb);
    tail_acc_phis.push_back(phi);
  }
  b.CreateCondBr(b.CreateICmpEQ(ti, nrows), exit_bb, tail_body_bb);

  b.SetInsertPoint(tail_body_bb);
  std::vector<Value *> tail_updated;
  for (size_t ai = 0; ai < agg_ops.size(); ai++) {
    auto &op = agg_ops[ai];
    auto &va = vaccs[ai];
    Value *acc = tail_acc_phis[ai];

    if (op.agg_type == 6) {
      tail_updated.push_back(
          va.fp ? b.CreateFAdd(acc, ConstantFP::get(va.elem_ty, 1.0))
                : b.CreateAdd(acc, ConstantInt::get(va.elem_ty, 1)));
      continue;
    }
    if (op.col_idx < 0) {
      tail_updated.push_back(acc);
      continue;
    }

    Value *typed_ptr = b.CreateBitCast(col_data[op.col_idx],
                                       PointerType::getUnqual(va.elem_ty));
    Value *elem =
        b.CreateLoad(va.elem_ty, b.CreateGEP(va.elem_ty, typed_ptr, ti));

    switch (op.agg_type) {
    case 3:
      tail_updated.push_back(va.fp ? b.CreateFAdd(acc, elem)
                                   : b.CreateAdd(acc, elem));
      break;
    case 5:
      tail_updated.push_back(
          va.fp ? b.CreateFAdd(acc, ConstantFP::get(va.elem_ty, 1.0))
                : b.CreateAdd(acc, ConstantInt::get(va.elem_ty, 1)));
      break;
    case 1: {
      Value *cmp =
          va.fp ? b.CreateFCmpOLT(elem, acc) : b.CreateICmpSLT(elem, acc);
      tail_updated.push_back(b.CreateSelect(cmp, elem, acc));
      break;
    }
    case 2: {
      Value *cmp =
          va.fp ? b.CreateFCmpOGT(elem, acc) : b.CreateICmpSGT(elem, acc);
      tail_updated.push_back(b.CreateSelect(cmp, elem, acc));
      break;
    }
    default:
      tail_updated.push_back(acc);
      break;
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
        b.CreateGEP(i8, state_arg,
                    ConstantInt::get(i64, agg_ops[ai].state_offset)),
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
      case 3:
      case 5:
      case 6:
        combined_f = b.CreateFAdd(existing_f, as_f64);
        break;
      case 1: {
        Value *cmp = b.CreateFCmpOLT(as_f64, existing_f);
        combined_f = b.CreateSelect(cmp, as_f64, existing_f);
        break;
      }
      case 2: {
        Value *cmp = b.CreateFCmpOGT(as_f64, existing_f);
        combined_f = b.CreateSelect(cmp, as_f64, existing_f);
        break;
      }
      default:
        combined_f = existing_f;
        break;
      }
      b.CreateStore(b.CreateBitCast(combined_f, i64), state_ptr);
    } else {
      partial = (va.elem_ty == i32) ? b.CreateSExt(tail_acc_phis[ai], i64)
                                    : tail_acc_phis[ai]; // already i64
      Value *combined;
      switch (agg_ops[ai].agg_type) {
      case 3:
      case 5:
      case 6:
        combined = b.CreateAdd(existing, partial);
        break;
      case 1: {
        Value *cmp = b.CreateICmpSLT(partial, existing);
        combined = b.CreateSelect(cmp, partial, existing);
        break;
      }
      case 2: {
        Value *cmp = b.CreateICmpSGT(partial, existing);
        combined = b.CreateSelect(cmp, partial, existing);
        break;
      }
      default:
        combined = existing;
        break;
      }
      b.CreateStore(combined, state_ptr);
    }
  }
  b.CreateRetVoid();

  return fn;
}
#endif // !DISABLE_AGG_JIT (AllAggOpsSIMDFriendly + BuildAggUpdateFunctionSIMD)



// Build a hash join build function:
//   void aqp_hbuild_<id>(AQPChunkView* in, i8* hash_table)
//
// For each row: extracts key columns into a stack buffer, computes FNV-1a
// hash inline, calls aqp_ht_insert_prehash, then copies payload columns.
//
// key_col_indices[i] = input chunk column index for key column i
// key_elem_sizes[i]  = byte size of key column i
// payload_col_indices/sizes = same for payload columns
// ---------------------------------------------------------------------------
struct HashColDesc {
  int col_idx;        // chunk column index
  unsigned elem_size; // bytes per element
  int32_t dtype;
};


#if !DISABLE_AGG_JIT
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
  Type *i8p = PointerType::getUnqual(Type::getInt8Ty(llctx));
  Type *i32 = Type::getInt32Ty(llctx);
  Type *i64 = Type::getInt64Ty(llctx);
  Type *i64p = PointerType::getUnqual(i64);
  Type *f64 = Type::getDoubleTy(llctx);
  Type *voidTy = Type::getVoidTy(llctx);

  StructType *ColViewTy = StructType::get(llctx, {i8p, i64p, i32, i32});
  StructType *ChunkViewTy =
      StructType::get(llctx, {PointerType::getUnqual(ColViewTy), i64, i64});

  // void fn(AQPChunkView *in, i8 *agg_state)
  FunctionType *fn_ty = FunctionType::get(
      voidTy, {PointerType::getUnqual(ChunkViewTy), i8p}, false);
  Function *fn =
      Function::Create(fn_ty, Function::ExternalLinkage, fn_name, &mod);

  Value *in_arg = fn->getArg(0);
  in_arg->setName("in");
  Value *state_arg = fn->getArg(1);
  state_arg->setName("state");

  BasicBlock *entry_bb = BasicBlock::Create(llctx, "entry", fn);
  BasicBlock *loop_bb = BasicBlock::Create(llctx, "loop", fn);
  BasicBlock *body_bb = BasicBlock::Create(llctx, "body", fn);
  BasicBlock *next_bb = BasicBlock::Create(llctx, "next", fn);
  BasicBlock *exit_bb = BasicBlock::Create(llctx, "exit", fn);

  IRBuilder<> b(entry_bb);

  // Load nrows
  Value *nrows_ptr = b.CreateStructGEP(ChunkViewTy, in_arg, 1);
  Value *nrows = b.CreateLoad(i64, nrows_ptr, "nrows");

  // Load column data pointers
  Value *cols_pp = b.CreateStructGEP(ChunkViewTy, in_arg, 0);
  Value *cols =
      b.CreateLoad(PointerType::getUnqual(ColViewTy), cols_pp, "cols");

  // Pre-load data pointers for columns used by agg functions
  std::map<int, Value *> col_data_ptrs;
  std::map<int, Value *> col_validity_ptrs;
  for (const auto &op : agg_ops) {
    if (op.col_idx >= 0 &&
        col_data_ptrs.find(op.col_idx) == col_data_ptrs.end()) {
      Value *col_i = b.CreateGEP(ColViewTy, cols,
                                 ConstantInt::get(i64, (uint64_t)op.col_idx));
      col_data_ptrs[op.col_idx] =
          b.CreateLoad(i8p, b.CreateStructGEP(ColViewTy, col_i, 0),
                       "data_" + std::to_string(op.col_idx));
      col_validity_ptrs[op.col_idx] =
          b.CreateLoad(i64p, b.CreateStructGEP(ColViewTy, col_i, 1),
                       "val_" + std::to_string(op.col_idx));
    }
  }

  // CountStar optimization: state.count += nrows (one instruction, no loop)
  for (const auto &op : agg_ops) {
    if (op.agg_type == 6 /* CountStar */) {
      Value *acc_ptr =
          b.CreateBitCast(b.CreateGEP(Type::getInt8Ty(llctx), state_arg,
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

    if (op.col_idx < 0)
      continue;

    // Check validity (skip NULL rows)
    Value *validity = col_validity_ptrs[op.col_idx];

    // Capture current block — may be body_bb (first op) or previous op's
    // cont_bb
    BasicBlock *pre_check_bb = b.GetInsertBlock();

    BasicBlock *check_bb = BasicBlock::Create(llctx, "check_val", fn);
    BasicBlock *valid_bb = BasicBlock::Create(llctx, "valid", fn);
    BasicBlock *cont_bb = BasicBlock::Create(llctx, "cont", fn);

    // If validity pointer is non-null, check the bit; else all valid
    Value *val_nonnull = b.CreateICmpNE(b.CreatePtrToInt(validity, i64),
                                        ConstantInt::get(i64, 0));
    b.CreateCondBr(val_nonnull, check_bb, valid_bb);

    b.SetInsertPoint(check_bb);
    Value *word_idx = b.CreateLShr(row_i, ConstantInt::get(i64, 6));
    Value *bit_idx = b.CreateAnd(row_i, ConstantInt::get(i64, 63));
    Value *word = b.CreateLoad(i64, b.CreateGEP(i64, validity, word_idx));
    Value *bit =
        b.CreateAnd(b.CreateLShr(word, bit_idx), ConstantInt::get(i64, 1));
    Value *bit_valid = b.CreateICmpNE(bit, ConstantInt::get(i64, 0));
    b.CreateCondBr(bit_valid, valid_bb, cont_bb);

    // Valid path: update accumulator
    b.SetInsertPoint(valid_bb);
    PHINode *came_from = b.CreatePHI(Type::getInt1Ty(llctx), 2, "from_valid");
    came_from->addIncoming(ConstantInt::getTrue(llctx),
                           pre_check_bb); // all-valid path
    came_from->addIncoming(ConstantInt::getTrue(llctx),
                           check_bb); // bit-valid path
    (void)came_from;

    Value *data_ptr = col_data_ptrs[op.col_idx];
    Value *acc_ptr =
        b.CreateBitCast(b.CreateGEP(Type::getInt8Ty(llctx), state_arg,
                                    ConstantInt::get(i64, op.state_offset)),
                        PointerType::getUnqual(i64));

    bool is_float =
        (op.dtype == AQP_DTYPE_FLOAT || op.dtype == AQP_DTYPE_DOUBLE);

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
      Value *elem =
          b.CreateLoad(Type::getInt8Ty(llctx),
                       b.CreateGEP(Type::getInt8Ty(llctx), data_ptr, row_i));
      val = b.CreateSExt(elem, i64);
    } else if (op.dtype == AQP_DTYPE_INT16) {
      Value *typed_ptr = b.CreateBitCast(
          data_ptr, PointerType::getUnqual(Type::getInt16Ty(llctx)));
      Value *elem =
          b.CreateLoad(Type::getInt16Ty(llctx),
                       b.CreateGEP(Type::getInt16Ty(llctx), typed_ptr, row_i));
      val = b.CreateSExt(elem, i64);
    } else if (op.dtype == AQP_DTYPE_DOUBLE) {
      acc_ptr =
          b.CreateBitCast(b.CreateGEP(Type::getInt8Ty(llctx), state_arg,
                                      ConstantInt::get(i64, op.state_offset)),
                          PointerType::getUnqual(f64));
      Value *typed_ptr = b.CreateBitCast(data_ptr, PointerType::getUnqual(f64));
      val = b.CreateLoad(f64, b.CreateGEP(f64, typed_ptr, row_i));
    } else if (op.dtype == AQP_DTYPE_FLOAT) {
      acc_ptr =
          b.CreateBitCast(b.CreateGEP(Type::getInt8Ty(llctx), state_arg,
                                      ConstantInt::get(i64, op.state_offset)),
                          PointerType::getUnqual(f64));
      Value *typed_ptr = b.CreateBitCast(
          data_ptr, PointerType::getUnqual(Type::getFloatTy(llctx)));
      Value *fval =
          b.CreateLoad(Type::getFloatTy(llctx),
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
      Value *new_acc =
          is_float ? b.CreateFAdd(acc, val) : b.CreateAdd(acc, val);
      b.CreateStore(new_acc, acc_ptr);
      break;
    }
    case 5: /* Count */ {
      // Count non-null: just increment i64 accumulator
      Value *cnt_ptr =
          b.CreateBitCast(b.CreateGEP(Type::getInt8Ty(llctx), state_arg,
                                      ConstantInt::get(i64, op.state_offset)),
                          PointerType::getUnqual(i64));
      Value *cnt = b.CreateLoad(i64, cnt_ptr);
      b.CreateStore(b.CreateAdd(cnt, ConstantInt::get(i64, 1)), cnt_ptr);
      break;
    }
    case 1: /* Min */ {
      Value *cmp =
          is_float ? b.CreateFCmpOLT(val, acc) : b.CreateICmpSLT(val, acc);
      Value *new_acc = b.CreateSelect(cmp, val, acc);
      b.CreateStore(new_acc, acc_ptr);
      break;
    }
    case 2: /* Max */ {
      Value *cmp =
          is_float ? b.CreateFCmpOGT(val, acc) : b.CreateICmpSGT(val, acc);
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
        Value *sum_i = b.CreateLoad(
            i64, b.CreateBitCast(sum_ptr, PointerType::getUnqual(i64)));
        b.CreateStore(b.CreateAdd(sum_i, val),
                      b.CreateBitCast(sum_ptr, PointerType::getUnqual(i64)));
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
#endif // !DISABLE_AGG_JIT (BuildAggUpdateFunction)

// ---------------------------------------------------------------------------
// Optimise the module (skipped in debug mode)
// ---------------------------------------------------------------------------
static void OptimiseModule(Module &mod, bool skip) {
  if (skip) return;

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

  FunctionPassManager fpm;
  fpm.addPass(InstCombinePass());
  fpm.addPass(ReassociatePass());
  fpm.addPass(GVNPass());
  fpm.addPass(SimplifyCFGPass());

  ModulePassManager mpm;
  mpm.addPass(createModuleToFunctionPassAdaptor(std::move(fpm)));
  mpm.run(mod, mam);
}

// ---------------------------------------------------------------------------
// IrToLlvmCompiler public API
// ---------------------------------------------------------------------------
IrToLlvmCompiler::IrToLlvmCompiler(bool debug, SimdISA simd,
                                   FastCompileBackend fast)
    // Fast tier skips the mid-end pipeline entirely (mirrors lingo-db's
    // cheap backend, LLVMBackends.cpp:823-827).
    : skip_opt_(debug || fast != FastCompileBackend::OFF), fast_mode_(fast),
      simd_isa_(simd), use_simd_(simd != SimdISA::OFF),
      impl_(std::make_unique<Impl>(simd, fast)) {}

IrToLlvmCompiler::~IrToLlvmCompiler() = default;

void IrToLlvmCompiler::SetCache(int mode) {
  cache_mode_ = mode;
  cache_enabled_ = mode > 0;
  if (cache_enabled_ && impl_)
    impl_->EnableCache();
}

void IrToLlvmCompiler::SetDiskCacheDir(const std::string &dir) {
  if (impl_) {
    impl_->disk_cache_dir = dir;
    if (!dir.empty())
      std::filesystem::create_directories(dir);
  }
}

void IrToLlvmCompiler::ClearObjCache() {
  std::lock_guard<std::mutex> lk(Impl::ObjCacheMu());
  Impl::ObjCache().clear();
}

void *IrToLlvmCompiler::LookupCachedFn(const std::string &cache_key,
                                         const std::string &fn_name) {
  return impl_->TryCacheLoad(cache_key, fn_name);
}

const std::string &IrToLlvmCompiler::LastCacheKey() const {
  return impl_->last_cache_key;
}

const std::string &IrToLlvmCompiler::LastEntryName() const {
  return impl_->last_entry_name;
}

// --- JITTrackerHandle ---

JITTrackerHandle::~JITTrackerHandle() { Reset(); }

void JITTrackerHandle::Reset() {
  if (!ptr) return;
  auto *sp = static_cast<orc::ResourceTrackerSP *>(ptr);
  if (*sp) {
    if (auto e = (*sp)->remove())
      logAllUnhandledErrors(std::move(e), errs());
  }
  delete sp;
  ptr = nullptr;
}

JITTrackerHandle IrToLlvmCompiler::CreateIsolatedTracker() {
  JITTrackerHandle h;
  if (impl_ && impl_->jit) {
    auto sp = impl_->jit->getMainJITDylib().createResourceTracker();
    h.ptr = new orc::ResourceTrackerSP(std::move(sp));
  }
  return h;
}

void IrToLlvmCompiler::ResetModules() {
  if (impl_)
    impl_->ResetModules();
}

// --- Isolated-tracker overloads (delegate to main methods via TrackerGuard) ---

AQPExprFn IrToLlvmCompiler::CompileExpr(const AQPExpr &expr,
                                        const std::vector<ColSchema> &schema,
                                        JITTrackerHandle &tracker) {
  TrackerGuard g(impl_->current_tracker, tracker);
  return CompileExpr(expr, schema);
}

AQPExprFn IrToLlvmCompiler::CompileFilter(const AQPStmt &filter_node,
                                          const std::vector<ColSchema> &schema,
                                          JITTrackerHandle &tracker,
                                          std::vector<uint8_t> *params_out) {
  TrackerGuard g(impl_->current_tracker, tracker);
  return CompileFilter(filter_node, schema, params_out);
}

AQPPipelineFn IrToLlvmCompiler::CompilePipeline(
    const AQPStmt *filter_node, const AQPStmt *proj_node,
    const std::vector<ColSchema> &in_schema, JITTrackerHandle &tracker,
    std::vector<uint8_t> *params_out) {
  TrackerGuard g(impl_->current_tracker, tracker);
  return CompilePipeline(filter_node, proj_node, in_schema, params_out);
}

static std::string BuildCacheContent(const std::string &tag,
                                     const std::vector<ColSchema> &schema,
                                     const std::string &extra = "") {
  std::string s = tag;
  for (const auto &cs : schema) {
    s += "|";
    s += std::to_string(cs.table_idx);
    s += ".";
    s += std::to_string(cs.col_idx);
    s += ".";
    s += std::to_string(cs.dtype);
  }
  if (!extra.empty()) {
    s += "||";
    s += extra;
  }
  return s;
}

// ---------------------------------------------------------------------------
// §5.1 cache-key serializers. Every codegen-relevant input must be
// represented; runtime identities (hj pointers, eids, row counts — the
// modules load nrows/offsets from views/ctx at run time) are excluded.
// ---------------------------------------------------------------------------

template <typename V>
static void AppendIntVec(std::string &s, const char *tag, const V &v) {
  s += tag;
  for (auto x : v) {
    s += std::to_string((long long)x);
    s += ",";
  }
  s += ";";
}

static std::string SerializeProbeStages(
    const std::vector<ProbeStageInfo> &stages) {
  std::string s;
  for (const auto &st : stages) {
    s += "STAGE{";
    if (st.filter_ir)
      s += const_cast<ir_sql_converter::AQPStmt *>(st.filter_ir)
               ->Print(false, 0);
    s += "||";
    if (st.join_ir)
      s += const_cast<ir_sql_converter::AQPStmt *>(st.join_ir)->Print(false, 0);
    s += "||";
    s += BuildCacheContent("p", st.probe_schema);
    s += BuildCacheContent("y", st.payload_schema);
    AppendIntVec(s, "pri", st.payload_row_indices);
    AppendIntVec(s, "lo", st.lhs_output_idxs);
    AppendIntVec(s, "ro", st.rhs_output_layout_idxs);
    AppendIntVec(s, "ld", st.lhs_output_dtypes);
    AppendIntVec(s, "rd", st.rhs_output_dtypes);
    AppendIntVec(s, "lk", st.lhs_key_chunk_idxs);
    AppendIntVec(s, "kd", st.lhs_key_dtypes);
    s += st.skip_hash_cmp_eligible ? "E}" : "e}";
  }
  return s;
}

static void AppendQjitLoc(std::string &s, const qjit::QjitValueLoc &loc) {
  s += "(" + std::to_string(loc.src_col) + "," + std::to_string(loc.ht_id) +
       "," + std::to_string(loc.layout_col) + "," + std::to_string(loc.dtype) +
       ")";
}

static std::string SerializeQjitPlan(const qjit::QjitQueryPlan &plan) {
  std::string s;
  for (const auto &st : plan.steps) {
    s += "S{" + st.source_table + "#" + std::to_string(st.source_table_index) +
         (st.source_is_temp ? "T" : "B");
    for (const auto &c : st.cols)
      s += "|" + std::to_string(c.table_index) + "." +
           std::to_string(c.column_index) + "." +
           std::to_string(c.expected_dtype);
    for (const auto &op : st.ops) {
      if (op.kind == qjit::QjitStepOp::Filter) {
        s += ";F:";
        s += const_cast<ir_sql_converter::AQPExpr *>(op.filter)->Print(false);
      } else {
        s += ";P:" + std::to_string(op.ht_id);
        for (const auto &k : op.keys)
          AppendQjitLoc(s, k);
      }
    }
    s += ";g" + std::to_string(st.guard_pos);
    for (const auto &g : st.guards)
      s += "(" + std::to_string(g.op_index) + (g.membership ? "m" : "r") + ")";
    s += ";b" + std::to_string(st.block_skip_col);
    s += ";K" + std::to_string((int)st.sink) + ":" + std::to_string(st.sink_ht);
    for (const auto &o : st.outputs)
      AppendQjitLoc(s, o);
    for (const auto &a : st.agg_cells) {
      s += ";A" + std::to_string((int)a.fn) + (a.has_arg ? "v" : "x");
      if (a.has_arg)
        AppendQjitLoc(s, a.arg);
    }
    s += "}";
  }
  for (const auto &ht : plan.hts) {
    s += "H{" + std::to_string(ht.num_keys) + "," +
         std::to_string(ht.prefix_bytes) + "," + std::to_string(ht.tuple_size);
    for (const auto &c : ht.cols)
      s += "|" + std::to_string(c.table_index) + "." +
           std::to_string(c.column_index) + "." + std::to_string(c.dtype) +
           "@" + std::to_string(c.offset);
    s += "}";
  }
  s += "G";
  s += plan.has_agg ? "1" : "0";
  for (int c : plan.agg_output_cells)
    s += "," + std::to_string(c);
  return s;
}

static void SerializeExprTemplate(std::string &s, const AQPExpr *expr) {
  if (!expr) return;
  auto nt = expr->GetNodeType();
  if (nt == VarConstComparisonNode) {
    auto *cmp = static_cast<const SimplestVarConstComparison *>(expr);
    s += cmp->attr->Print(false);
    s += ":" + std::to_string((int)cmp->GetSimplestExprType());
    auto vt = cmp->const_var->GetType();
    if (vt == SimplestVarType::IntVar)
      s += "$I";
    else if (vt == SimplestVarType::FloatVar)
      s += "$F";
    else if (vt == SimplestVarType::StringVar) {
      auto op = cmp->GetSimplestExprType();
      if (op == SimplestExprType::TextLike ||
          op == SimplestExprType::Text_Not_Like) {
        std::string literal;
        LikeSegments seg_info;
        LikePatternKind kind = ClassifyLikePatternEx(
            cmp->const_var->GetStringValue(), literal, seg_info);
        s += "$L" + std::to_string(kind);
        if (kind == LIKE_MULTI_SEGMENT)
          s += "x" + std::to_string(seg_info.segs.size());
      } else {
        s += "$S";
      }
    } else if (vt == SimplestVarType::Date)
      s += "$D";
    else
      s += "$?";
  } else if (nt == InExprNode) {
    auto *in = static_cast<const SimplestInExpr *>(expr);
    s += in->attr->Print(false);
    s += (in->negated ? ":NI" : ":IN");
    int n = (int)in->values.size();
    if (n <= 8) {
      auto vt = in->values.empty() ? SimplestVarType::IntVar
                                   : in->values[0]->GetType();
      if (vt == SimplestVarType::IntVar)
        s += "$I*" + std::to_string(n);
      else if (vt == SimplestVarType::StringVar)
        s += "$S*" + std::to_string(n);
      else
        s += "$?*" + std::to_string(n);
    } else {
      s += const_cast<AQPExpr *>(expr)->Print(false);
    }
  } else if (nt == LogicalExprNode) {
    auto *le = static_cast<const SimplestLogicalExpr *>(expr);
    s += "(";
    SerializeExprTemplate(s, le->left_expr.get());
    s += "L" + std::to_string((int)le->GetLogicalOp());
    SerializeExprTemplate(s, le->right_expr.get());
    s += ")";
  } else if (nt == IsNullExprNode) {
    s += const_cast<AQPExpr *>(expr)->Print(false);
  } else {
    s += const_cast<AQPExpr *>(expr)->Print(false);
  }
}

static std::string SerializeQjitPlanTemplate(const qjit::QjitQueryPlan &plan) {
  std::string s;
  for (const auto &st : plan.steps) {
    s += "S{" + st.source_table + "#" + std::to_string(st.source_table_index) +
         (st.source_is_temp ? "T" : "B");
    for (const auto &c : st.cols)
      s += "|" + std::to_string(c.table_index) + "." +
           std::to_string(c.column_index) + "." +
           std::to_string(c.expected_dtype);
    for (const auto &op : st.ops) {
      if (op.kind == qjit::QjitStepOp::Filter) {
        s += ";FT:";
        SerializeExprTemplate(s, op.filter);
      } else {
        s += ";P:" + std::to_string(op.ht_id);
        for (const auto &k : op.keys)
          AppendQjitLoc(s, k);
      }
    }
    s += ";g" + std::to_string(st.guard_pos);
    for (const auto &g : st.guards)
      s += "(" + std::to_string(g.op_index) + (g.membership ? "m" : "r") + ")";
    s += ";b" + std::to_string(st.block_skip_col);
    s += ";K" + std::to_string((int)st.sink) + ":" + std::to_string(st.sink_ht);
    for (const auto &o : st.outputs)
      AppendQjitLoc(s, o);
    for (const auto &a : st.agg_cells) {
      s += ";A" + std::to_string((int)a.fn) + (a.has_arg ? "v" : "x");
      if (a.has_arg)
        AppendQjitLoc(s, a.arg);
    }
    s += "}";
  }
  for (const auto &ht : plan.hts) {
    s += "H{" + std::to_string(ht.num_keys) + "," +
         std::to_string(ht.prefix_bytes) + "," + std::to_string(ht.tuple_size);
    for (const auto &c : ht.cols)
      s += "|" + std::to_string(c.table_index) + "." +
           std::to_string(c.column_index) + "." + std::to_string(c.dtype) +
           "@" + std::to_string(c.offset);
    s += "}";
  }
  s += "G";
  s += plan.has_agg ? "1" : "0";
  for (int c : plan.agg_output_cells)
    s += "," + std::to_string(c);
  return s;
}

static void BuildParamsFromExpr(ParamsBuilder &pb, const AQPExpr *expr) {
  if (!expr) return;
  auto nt = expr->GetNodeType();
  if (nt == VarConstComparisonNode) {
    auto *cmp = static_cast<const SimplestVarConstComparison *>(expr);
    auto vt = cmp->const_var->GetType();
    if (vt == SimplestVarType::IntVar || vt == SimplestVarType::Date) {
      int64_t v = (int64_t)cmp->const_var->GetIntValue();
      pb.AllocI64(v);
    } else if (vt == SimplestVarType::FloatVar) {
      double v = (double)cmp->const_var->GetFloatValue();
      pb.AllocF64(v);
    } else if (vt == SimplestVarType::StringVar) {
      auto op = cmp->GetSimplestExprType();
      if (op == SimplestExprType::TextLike ||
          op == SimplestExprType::Text_Not_Like) {
        std::string literal;
        LikeSegments seg_info;
        LikePatternKind kind = ClassifyLikePatternEx(
            cmp->const_var->GetStringValue(), literal, seg_info);
        if (kind == LIKE_MULTI_SEGMENT) {
          for (const auto &seg : seg_info.segs)
            pb.AllocString(seg);
        } else if (kind == LIKE_CONTAINS) {
          pb.AllocString(literal);
          pb.AllocI32((int32_t)(unsigned char)literal[0]);
        } else if (kind == LIKE_PREFIX || kind == LIKE_SUFFIX) {
          pb.AllocString(literal);
        } else {
          pb.AllocString(cmp->const_var->GetStringValue());
        }
      } else {
        pb.AllocString(cmp->const_var->GetStringValue());
      }
    }
  } else if (nt == InExprNode) {
    auto *in = static_cast<const SimplestInExpr *>(expr);
    if ((int)in->values.size() <= 8) {
      for (const auto &v : in->values) {
        if (v->GetType() == SimplestVarType::IntVar ||
            v->GetType() == SimplestVarType::Date)
          pb.AllocI64((int64_t)v->GetIntValue());
        else if (v->GetType() == SimplestVarType::StringVar)
          pb.AllocString(v->GetStringValue());
        else if (v->GetType() == SimplestVarType::FloatVar)
          pb.AllocF64((double)v->GetFloatValue());
      }
    }
  } else if (nt == LogicalExprNode) {
    auto *le = static_cast<const SimplestLogicalExpr *>(expr);
    BuildParamsFromExpr(pb, le->left_expr.get());
    BuildParamsFromExpr(pb, le->right_expr.get());
  }
}

static std::vector<uint8_t> BuildParamsBuffer(
    const qjit::QjitQueryPlan &plan) {
  ParamsBuilder pb;
  for (const auto &st : plan.steps)
    for (const auto &op : st.ops)
      if (op.kind == qjit::QjitStepOp::Filter)
        BuildParamsFromExpr(pb, op.filter);
  pb.Finalize();
  return std::move(pb.buf);
}

// §7.3 template mode helpers for non-query-jit: serialize expression tree
// with constants replaced by type tags, and build params buffer from
// the expression's constant values. Walk order must match codegen order
// (qual_vec iteration order in BuildFilterFunction / EmitExpr).

static std::string SerializeStmtTemplate(const AQPStmt *node) {
  if (!node) return "";
  std::string s;
  for (const auto &qe : node->qual_vec) {
    SerializeExprTemplate(s, qe.get());
    s += "&";
  }
  return s;
}

static std::vector<uint8_t> BuildParamsFromStmt(const AQPStmt *node) {
  ParamsBuilder pb;
  if (node)
    for (const auto &qe : node->qual_vec)
      BuildParamsFromExpr(pb, qe.get());
  pb.Finalize();
  return std::move(pb.buf);
}

static std::string SerializeProbeStagesTemplate(
    const std::vector<ProbeStageInfo> &stages) {
  std::string s;
  for (const auto &st : stages) {
    s += "STAGE{";
    if (st.filter_ir)
      s += SerializeStmtTemplate(st.filter_ir);
    s += "||";
    if (st.join_ir)
      s += const_cast<ir_sql_converter::AQPStmt *>(st.join_ir)->Print(false, 0);
    s += "||";
    s += BuildCacheContent("p", st.probe_schema);
    s += BuildCacheContent("y", st.payload_schema);
    AppendIntVec(s, "pri", st.payload_row_indices);
    AppendIntVec(s, "lo", st.lhs_output_idxs);
    AppendIntVec(s, "ro", st.rhs_output_layout_idxs);
    AppendIntVec(s, "ld", st.lhs_output_dtypes);
    AppendIntVec(s, "rd", st.rhs_output_dtypes);
    AppendIntVec(s, "lk", st.lhs_key_chunk_idxs);
    AppendIntVec(s, "kd", st.lhs_key_dtypes);
    s += st.skip_hash_cmp_eligible ? "E}" : "e}";
  }
  return s;
}

static std::vector<uint8_t> BuildParamsFromProbeStages(
    const std::vector<ProbeStageInfo> &stages) {
  return BuildParamsFromStmt(stages.empty() ? nullptr : stages[0].filter_ir);
}

unsigned IrToLlvmCompiler::GetVecWidth() const {
  return (use_simd_ && impl_) ? impl_->vec_width : 1;
}

bool IrToLlvmCompiler::HasSIMD() const {
  return use_simd_ && impl_ && impl_->vec_width > 1;
}

AQPExprFn IrToLlvmCompiler::CompileExpr(const AQPExpr &expr,
                                        const std::vector<ColSchema> &schema) {

  auto ctx = std::make_unique<LLVMContext>();
  auto mod = std::make_unique<Module>("aqp_expr_mod", *ctx);

  // Serialise expression to a string for hashing
  std::ostringstream oss;
  const_cast<AQPExpr &>(expr).Print(
      false); // Print to string (side-effect: oss via cout redirect)
  uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
  std::string fn_name = "aqp_expr_" + std::to_string(fn_id);

  std::vector<const AQPExpr *> exprs = {&expr};

  // Try SIMD if enabled and expression is SIMD-friendly
  Function *fn = nullptr;
  if (use_simd_ && impl_->vec_width > 1 &&
      AllExprsSIMDFriendly(exprs, schema)) {
    fn = BuildFilterFunctionSIMD(*ctx, *mod, fn_name, exprs, schema,
                                 impl_->vec_width);
  }
  if (!fn)
    fn = BuildFilterFunction(*ctx, *mod, fn_name, exprs, schema);
  if (!fn)
    return nullptr;
  SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

  // Verify
  std::string err;
  raw_string_ostream es(err);
  if (verifyFunction(*fn, &es)) {
    // Verification failed — fall back to interpreter
    return nullptr;
  }

  OptimiseModule(*mod, skip_opt_);

  // Add module to ORC JIT
  auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
  if (auto err2 = impl_->jit->addIRModule(impl_->current_tracker, std::move(tsm))) {
    logAllUnhandledErrors(std::move(err2), errs());
    return nullptr;
  }

  // Look up the compiled function
  auto sym = impl_->jit->lookup(fn_name);
  if (!sym) {
    logAllUnhandledErrors(sym.takeError(), errs());
    return nullptr;
  }

  return AQP_JIT_GET_FN(AQPExprFn, sym);
}

AQPExprFn
IrToLlvmCompiler::CompileRangeFilter(unsigned chunk_col_idx, int32_t dtype,
                                     int64_t min_val, int64_t max_val) {
  auto ctx = std::make_unique<LLVMContext>();
  auto mod = std::make_unique<Module>("aqp_range_mod", *ctx);

  uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
  std::string fn_name = "aqp_range_" + std::to_string(fn_id);

  auto &C = *ctx;
  Type *i8p = PointerType::getUnqual(C);
  Type *i32 = Type::getInt32Ty(C);
  Type *i64 = Type::getInt64Ty(C);

  // AQPExprFn: idx_t fn(AQPChunkView*, AQPSelView*)
  FunctionType *ft = FunctionType::get(i64, {i8p, i8p}, false);
  Function *fn = Function::Create(ft, Function::ExternalLinkage, fn_name, *mod);

  auto *entry = BasicBlock::Create(C, "entry", fn);
  auto *loop_bb = BasicBlock::Create(C, "loop", fn);
  auto *match_bb = BasicBlock::Create(C, "match", fn);
  auto *next_bb = BasicBlock::Create(C, "next", fn);
  auto *exit_bb = BasicBlock::Create(C, "exit", fn);

  IRBuilder<> B(entry);
  Value *cv_ptr = fn->getArg(0);   // AQPChunkView*
  Value *sv_ptr = fn->getArg(1);   // AQPSelView*

  // nrows = cv->nrows (offset 16 in AQPChunkView: cols=8, nrows=8)
  Value *nrows_ptr = B.CreateStructGEP(
      StructType::get(C, {i8p, i64, i64}), cv_ptr, 1);
  Value *nrows = B.CreateLoad(i64, nrows_ptr);

  // cols = cv->cols (offset 0)
  Value *cols_ptr = B.CreateStructGEP(
      StructType::get(C, {i8p, i64, i64}), cv_ptr, 0);
  Value *cols = B.CreateLoad(i8p, cols_ptr);

  // AQPColView layout: {void* data, uint64_t* validity, int32_t vtype, int32_t dtype}
  // Size = 24 bytes
  Type *col_view_ty = StructType::get(C, {i8p, i8p, i32, i32});
  unsigned col_view_size = 24;

  // col_ptr = &cols[chunk_col_idx]
  Value *col_offset = ConstantInt::get(i64, chunk_col_idx * col_view_size);
  Value *col_ptr = B.CreateGEP(Type::getInt8Ty(C), cols, col_offset);

  // data = col->data
  Value *data_ptr_ptr = B.CreateStructGEP(col_view_ty, col_ptr, 0);
  Value *data_ptr = B.CreateLoad(i8p, data_ptr_ptr);

  // sel->indices
  // AQPSelView: {sel_t* indices, uint32_t count}
  Type *sel_view_ty = StructType::get(C, {i8p, i32});
  Value *sel_indices_ptr = B.CreateStructGEP(sel_view_ty, sv_ptr, 0);
  Value *sel_indices = B.CreateLoad(i8p, sel_indices_ptr);

  // out_count alloca
  Value *out_count = B.CreateAlloca(i64);
  B.CreateStore(ConstantInt::get(i64, 0), out_count);

  // i = 0
  Value *i_alloca = B.CreateAlloca(i64);
  B.CreateStore(ConstantInt::get(i64, 0), i_alloca);

  B.CreateBr(loop_bb);

  // Loop header: while (i < nrows)
  B.SetInsertPoint(loop_bb);
  Value *i_val = B.CreateLoad(i64, i_alloca);
  Value *cond = B.CreateICmpULT(i_val, nrows);
  B.CreateCondBr(cond, match_bb, exit_bb);

  // Match: check range
  B.SetInsertPoint(match_bb);
  Value *val;
  if (dtype == 3) { // INT32
    Value *elem_ptr = B.CreateGEP(i32, data_ptr, i_val);
    Value *elem = B.CreateLoad(i32, elem_ptr);
    val = B.CreateSExt(elem, i64);
  } else { // INT64
    Value *elem_ptr = B.CreateGEP(i64, data_ptr, i_val);
    val = B.CreateLoad(i64, elem_ptr);
  }
  Value *ge_min = B.CreateICmpSGE(val, ConstantInt::get(i64, min_val));
  Value *le_max = B.CreateICmpSLE(val, ConstantInt::get(i64, max_val));
  Value *in_range = B.CreateAnd(ge_min, le_max);

  // If in range, write to sel_indices[out_count++]
  auto *write_bb = BasicBlock::Create(C, "write", fn);
  B.CreateCondBr(in_range, write_bb, next_bb);

  B.SetInsertPoint(write_bb);
  Value *cnt = B.CreateLoad(i64, out_count);
  Value *idx_ptr = B.CreateGEP(i32, sel_indices, cnt);
  B.CreateStore(B.CreateTrunc(i_val, i32), idx_ptr);
  B.CreateStore(B.CreateAdd(cnt, ConstantInt::get(i64, 1)), out_count);
  B.CreateBr(next_bb);

  // Next: i++
  B.SetInsertPoint(next_bb);
  Value *i_next = B.CreateAdd(B.CreateLoad(i64, i_alloca), ConstantInt::get(i64, 1));
  B.CreateStore(i_next, i_alloca);
  B.CreateBr(loop_bb);

  // Exit: return out_count
  B.SetInsertPoint(exit_bb);
  B.CreateRet(B.CreateLoad(i64, out_count));

  SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

  std::string err;
  raw_string_ostream es(err);
  if (verifyFunction(*fn, &es)) {
    errs() << "[AQP-JIT] CompileRangeFilter verify failed: " << err << "\n";
    return nullptr;
  }

  OptimiseModule(*mod, skip_opt_);

  auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
  if (auto err2 = impl_->jit->addIRModule(impl_->current_tracker, std::move(tsm))) {
    logAllUnhandledErrors(std::move(err2), errs());
    return nullptr;
  }

  auto sym = impl_->jit->lookup(fn_name);
  if (!sym) {
    logAllUnhandledErrors(sym.takeError(), errs());
    return nullptr;
  }

  return AQP_JIT_GET_FN(AQPExprFn, sym);
}

AQPExprFn
IrToLlvmCompiler::CompileFilter(const AQPStmt &filter_node,
                                const std::vector<ColSchema> &schema,
                                std::vector<uint8_t> *params_out) {

  // Collect the qual_vec expressions from the filter node
  std::vector<const AQPExpr *> exprs;
  for (const auto &qe : filter_node.qual_vec) {
    exprs.push_back(qe.get());
  }
  if (exprs.empty())
    return nullptr;

  const bool template_mode = cache_mode_ == 2;

  // Use a monotonic counter to guarantee a unique function name within
  // this LLJIT instance (hash collisions caused "Duplicate definition").
  uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
  std::string fn_name = "aqp_expr_" + std::to_string(fn_id);

  // In-memory cache lookup
  std::string cache_key;
  if (cache_enabled_ && impl_->cache_enabled) {
    std::string expr_text = template_mode
        ? SerializeStmtTemplate(&filter_node)
        : const_cast<AQPStmt &>(filter_node).Print(false, 0);
    std::string opt_tag = std::to_string((int)simd_isa_) + "F" +
                          std::to_string((int)fast_mode_) +
                          "C" + std::to_string(cache_mode_);
    cache_key = Impl::ComputeCacheKey(
        BuildCacheContent("filter:" + opt_tag, schema, expr_text));
    fn_name = "aqp_expr_c" + cache_key.substr(0, 12);
    void *cached = impl_->TryCacheLoad(cache_key, fn_name);
    if (cached) {
      if (template_mode && params_out)
        *params_out = BuildParamsFromStmt(&filter_node);
      return reinterpret_cast<AQPExprFn>(cached);
    }
  }

  auto ctx = std::make_unique<LLVMContext>();
  auto mod = std::make_unique<Module>("aqp_filter_mod", *ctx);

  // §7.3 template mode: use plain scalar path to ensure walk order matches
  // BuildParamsFromStmt (which iterates qual_vec in order). SIMD and TwoPass
  // reorder expressions, breaking the deterministic params layout.
  ParamsBuilder tmpl_pb;
  ParamsBuilder *pb_ptr = template_mode ? &tmpl_pb : nullptr;

  // Try SIMD version first if enabled and expressions are SIMD-friendly.
  // Split: separate numeric (SIMD-able) from VARCHAR (scalar-only) expressions.
  Function *fn = nullptr;
  bool used_simd = false;
  if (!template_mode && use_simd_ && impl_->vec_width > 1) {
    // Check which top-level expressions are SIMD-friendly
    std::vector<const AQPExpr *> simd_exprs, scalar_exprs;
    for (const AQPExpr *e : exprs) {
      std::vector<const AQPExpr *> single = {e};
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
#ifndef NDEBUG
        std::cerr << "[AQP-JIT] using SIMD filter (VW=" << impl_->vec_width
                  << " all_simd=" << simd_exprs.size() << ")\n";
#endif
      }
    } else if (!simd_exprs.empty()) {
      // Mixed: use two-pass scalar (cheap-then-expensive) instead of SIMD
      // hybrid.  The SIMD hybrid adds vectorization overhead and selection
      // vector indirection that hurts when numeric predicates aren't very
      // selective.  Two-pass scalar avoids that while still pre-filtering.
      // (falls through to two-pass logic below)
    }
  }
  // Fall back to scalar — try two-pass (cheap-then-expensive) when mixed
  if (!template_mode && !fn && exprs.size() >= 2) {
    std::vector<const AQPExpr *> cheap, expensive;
    for (const AQPExpr *e : exprs) {
      if (IsExpensiveExpr(e))
        expensive.push_back(e);
      else
        cheap.push_back(e);
    }
    if (!cheap.empty() && !expensive.empty()) {
      fn = BuildFilterFunctionTwoPass(*ctx, *mod, fn_name, cheap, expensive,
                                      schema);
      if (fn) {
#ifndef NDEBUG
        std::cerr << "[AQP-JIT] using TWO-PASS filter (cheap="
                  << cheap.size() << " expensive=" << expensive.size() << ")\n";
#endif
      }
    }
  }
  if (!fn) {
    fn = BuildFilterFunction(*ctx, *mod, fn_name, exprs, schema, pb_ptr);
  }
  if (!fn)
    return nullptr;
  SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

  std::string err;
  raw_string_ostream es(err);
  if (verifyFunction(*fn, &es)) {
#ifndef NDEBUG
    std::cerr << "[AQP-JIT] verifyFunction failed"
              << (used_simd ? " (SIMD)" : "") << ": " << es.str() << "\n";
#endif
    // If SIMD failed verification, retry with scalar
    if (used_simd) {
      fn->eraseFromParent();
      fn = BuildFilterFunction(*ctx, *mod, fn_name, exprs, schema);
      if (!fn)
        return nullptr;
      SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);
      err.clear();
      if (verifyFunction(*fn, &es)) {
#ifndef NDEBUG
        std::cerr << "[AQP-JIT] scalar fallback also failed: " << es.str()
                  << "\n";
#endif
        return nullptr;
      }
      used_simd = false;
    } else {
      return nullptr;
    }
  }

  OptimiseModule(*mod, skip_opt_);

  impl_->pending_cache_key = cache_key;
  auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
  if (auto e = impl_->jit->addIRModule(impl_->current_tracker, std::move(tsm))) {
    impl_->pending_cache_key.clear();
#ifndef NDEBUG
    std::cerr << "[AQP-JIT] addIRModule failed\n";
#endif
    logAllUnhandledErrors(std::move(e), errs());
    return nullptr;
  }

  auto sym = impl_->jit->lookup(fn_name);
  impl_->pending_cache_key.clear();
  if (!sym) {
#ifndef NDEBUG
    std::cerr << "[AQP-JIT] lookup failed for " << fn_name << "\n";
#endif
    logAllUnhandledErrors(sym.takeError(), errs());
    return nullptr;
  }

  if (template_mode && params_out) {
    tmpl_pb.Finalize();
    *params_out = std::move(tmpl_pb.buf);
  }

  return AQP_JIT_GET_FN(AQPExprFn, sym);
}

AQPOperatorFn
IrToLlvmCompiler::CompileProjection(const AQPStmt &proj_node,
                                    const std::vector<ColSchema> &in_schema) {

  // Build column mapping and dtype list:
  // col_mapping[i] = input column index for output column i
  // col_dtypes[i]  = AQP_DTYPE_* for output column i (determines element size
  // for memcpy)
  std::vector<int> col_mapping;
  std::vector<int32_t> col_dtypes;
  for (const auto &attr : proj_node.target_list) {
    int found = -1;
    int32_t dtype = AQP_DTYPE_OTHER;
    for (int i = 0; i < (int)in_schema.size(); i++) {
      if (in_schema[i].table_idx == attr->GetTableIndex() &&
          in_schema[i].col_idx == attr->GetColumnIndex()) {
        found = i;
        dtype = in_schema[i].dtype;
        break;
      }
    }
    col_mapping.push_back(found);
    col_dtypes.push_back(dtype);
  }
  if (col_mapping.empty())
    return nullptr;

  uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
  std::string fn_name = "aqp_proj_" + std::to_string(fn_id);

  // In-memory cache lookup
  std::string cache_key;
  if (cache_enabled_ && impl_->cache_enabled) {
    std::string proj_text =
        const_cast<AQPStmt &>(proj_node).Print(false, 0);
    std::string opt_tag = std::to_string((int)simd_isa_) + "F" +
                          std::to_string((int)fast_mode_);
    cache_key = Impl::ComputeCacheKey(
        BuildCacheContent("proj:" + opt_tag, in_schema, proj_text));
    fn_name = "aqp_proj_c" + cache_key.substr(0, 12);
    void *cached = impl_->TryCacheLoad(cache_key, fn_name);
    if (cached)
      return reinterpret_cast<AQPOperatorFn>(cached);
  }

  auto ctx = std::make_unique<LLVMContext>();
  auto mod = std::make_unique<Module>("aqp_proj_mod", *ctx);

  Function *fn =
      BuildProjectionFunction(*ctx, *mod, fn_name, col_mapping, col_dtypes);
  if (!fn)
    return nullptr;
  SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

  std::string err;
  raw_string_ostream es(err);
  if (verifyFunction(*fn, &es)) {
#ifndef NDEBUG
    std::cerr << "[AQP-JIT] verifyFunction failed (proj): " << es.str() << "\n";
#endif
    return nullptr;
  }

  OptimiseModule(*mod, skip_opt_);

  impl_->pending_cache_key = cache_key;
  auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
  if (auto e = impl_->jit->addIRModule(impl_->current_tracker, std::move(tsm))) {
    impl_->pending_cache_key.clear();
#ifndef NDEBUG
    std::cerr << "[AQP-JIT] addIRModule failed (proj)\n";
#endif
    logAllUnhandledErrors(std::move(e), errs());
    return nullptr;
  }

  auto sym = impl_->jit->lookup(fn_name);
  impl_->pending_cache_key.clear();
  if (!sym) {
#ifndef NDEBUG
    std::cerr << "[AQP-JIT] lookup failed for " << fn_name << "\n";
#endif
    logAllUnhandledErrors(sym.takeError(), errs());
    return nullptr;
  }

  return AQP_JIT_GET_FN(AQPOperatorFn, sym);
}

#if !DISABLE_AGG_JIT
void *
IrToLlvmCompiler::CompileAggUpdate(const AQPStmt &agg_node,
                                   const std::vector<ColSchema> &in_schema) {

  auto *agg =
      dynamic_cast<const ir_sql_converter::SimplestAggregate *>(&agg_node);
  if (!agg || agg->agg_fns.empty())
    return nullptr;

  // For now: only ungrouped aggregates (no GROUP BY).
  // Grouped aggregates use the hash table and will be added in Phase 2C.
  if (!agg->groups.empty()) {
#ifndef NDEBUG
    std::cerr
        << "[AQP-JIT] grouped aggregate not yet supported → interpreter\n";
#endif
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
      op.dtype = AQP_DTYPE_INT64;
      state_offset += 8;
    } else {
      // Find column in input schema
      op.col_idx = -1;
      op.dtype = AQP_DTYPE_OTHER;
      for (int i = 0; i < (int)in_schema.size(); i++) {
        if (in_schema[i].table_idx == fn_pair.first->GetTableIndex() &&
            in_schema[i].col_idx == fn_pair.first->GetColumnIndex()) {
          op.col_idx = i;
          op.dtype = in_schema[i].dtype;
          break;
        }
      }
      if (op.col_idx < 0) {
#ifndef NDEBUG
        std::cerr << "[AQP-JIT] agg col not in schema: table="
                  << fn_pair.first->GetTableIndex()
                  << " col=" << fn_pair.first->GetColumnIndex() << "\n";
#endif
        continue; // skip this agg function
      }
      if (fn_pair.second == ir_sql_converter::SimplestAggFnType::Average)
        state_offset += 16; // sum + count
      else
        state_offset += 8;
    }
    agg_ops.push_back(op);
  }
  if (agg_ops.empty())
    return nullptr;

  uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
  std::string fn_name = "aqp_agg_" + std::to_string(fn_id);

  // In-memory cache lookup
  std::string cache_key;
  if (cache_enabled_ && impl_->cache_enabled) {
    std::string agg_text =
        const_cast<AQPStmt &>(agg_node).Print(false, 0);
    std::string opt_tag = std::to_string((int)simd_isa_) + "F" +
                          std::to_string((int)fast_mode_);
    cache_key = Impl::ComputeCacheKey(
        BuildCacheContent("agg:" + opt_tag, in_schema, agg_text));
    fn_name = "aqp_agg_c" + cache_key.substr(0, 12);
    void *cached = impl_->TryCacheLoad(cache_key, fn_name);
    if (cached)
      return cached;
  }

  auto ctx = std::make_unique<LLVMContext>();
  auto mod = std::make_unique<Module>("aqp_agg_mod", *ctx);

  // Try SIMD aggregate if enabled and all ops are SIMD-friendly
  Function *fn = nullptr;
  bool used_simd = false;
  if (use_simd_ && impl_->vec_width > 1 && AllAggOpsSIMDFriendly(agg_ops)) {
    fn = BuildAggUpdateFunctionSIMD(*ctx, *mod, fn_name, agg_ops, state_offset,
                                    in_schema, impl_->vec_width);
    if (fn) {
      used_simd = true;
#ifndef NDEBUG
      std::cerr << "[AQP-JIT] using SIMD aggregate (VW=" << impl_->vec_width
                << ")\n";
#endif
    }
  }
  if (!fn) {
    fn = BuildAggUpdateFunction(*ctx, *mod, fn_name, agg_ops, state_offset,
                                in_schema);
  }
  if (!fn)
    return nullptr;
  SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

  std::string err;
  raw_string_ostream es(err);
  if (verifyFunction(*fn, &es)) {
#ifndef NDEBUG
    std::cerr << "[AQP-JIT] verifyFunction failed (agg): " << es.str() << "\n";
#endif
    return nullptr;
  }

  OptimiseModule(*mod, skip_opt_);

  impl_->pending_cache_key = cache_key;
  auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
  if (auto e = impl_->jit->addIRModule(impl_->current_tracker, std::move(tsm))) {
    impl_->pending_cache_key.clear();
#ifndef NDEBUG
    std::cerr << "[AQP-JIT] addIRModule failed (agg)\n";
#endif
    logAllUnhandledErrors(std::move(e), errs());
    return nullptr;
  }

  auto sym = impl_->jit->lookup(fn_name);
  impl_->pending_cache_key.clear();
  if (!sym) {
#ifndef NDEBUG
    std::cerr << "[AQP-JIT] lookup failed for " << fn_name << "\n";
#endif
    logAllUnhandledErrors(sym.takeError(), errs());
    return nullptr;
  }

#ifndef NDEBUG
  std::cerr << "[AQP-JIT] compiled agg fn=" << fn_name
            << "  ops=" << agg_ops.size() << "  state_bytes=" << state_offset
            << "\n";
#endif

  return AQP_JIT_GET_ADDR(sym);
}

void *IrToLlvmCompiler::CompileAggUpdateDirect(const std::vector<AggOp> &agg_ops,
                                                unsigned total_state_size) {
  if (agg_ops.empty())
    return nullptr;

  uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
  std::string fn_name = "aqp_agg_d" + std::to_string(fn_id);

  // Build a minimal schema for BuildAggUpdateFunction (it only uses
  // col_idx via AggOp, the schema entries are unused in the loop body).
  std::vector<ColSchema> dummy_schema;
  for (const auto &op : agg_ops) {
    if (op.col_idx >= 0 && (size_t)op.col_idx >= dummy_schema.size())
      dummy_schema.resize(op.col_idx + 1);
  }

  auto ctx = std::make_unique<LLVMContext>();
  auto mod = std::make_unique<Module>("aqp_agg_d_mod", *ctx);

  Function *fn = nullptr;
  bool used_simd = false;
  if (use_simd_ && impl_->vec_width > 1 && AllAggOpsSIMDFriendly(agg_ops)) {
    fn = BuildAggUpdateFunctionSIMD(*ctx, *mod, fn_name, agg_ops, total_state_size,
                                    dummy_schema, impl_->vec_width);
    if (fn) {
      used_simd = true;
#ifndef NDEBUG
      std::cerr << "[AQP-JIT] using SIMD aggregate (VW=" << impl_->vec_width
                << ")\n";
#endif
    }
  }
  if (!fn) {
    fn = BuildAggUpdateFunction(*ctx, *mod, fn_name, agg_ops, total_state_size,
                                dummy_schema);
  }
  if (!fn)
    return nullptr;
  SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

  std::string err;
  raw_string_ostream es(err);
  if (verifyFunction(*fn, &es)) {
#ifndef NDEBUG
    std::cerr << "[AQP-JIT] verifyFunction failed (agg direct): " << es.str()
              << "\n";
#endif
    if (used_simd) {
      fn->eraseFromParent();
      fn = BuildAggUpdateFunction(*ctx, *mod, fn_name, agg_ops, total_state_size,
                                  dummy_schema);
      if (!fn)
        return nullptr;
      SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);
      err.clear();
      if (verifyFunction(*fn, &es)) {
#ifndef NDEBUG
        std::cerr << "[AQP-JIT] scalar agg fallback also failed: " << es.str()
                  << "\n";
#endif
        return nullptr;
      }
    } else {
      return nullptr;
    }
  }

  OptimiseModule(*mod, skip_opt_);

  auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
  if (auto e = impl_->jit->addIRModule(impl_->current_tracker, std::move(tsm))) {
    logAllUnhandledErrors(std::move(e), errs());
    return nullptr;
  }

  auto sym = impl_->jit->lookup(fn_name);
  if (!sym) {
    logAllUnhandledErrors(sym.takeError(), errs());
    return nullptr;
  }

#ifndef NDEBUG
  std::cerr << "[AQP-JIT] compiled agg fn=" << fn_name
            << "  ops=" << agg_ops.size() << "  state_bytes=" << total_state_size
            << "\n";
#endif

  return AQP_JIT_GET_ADDR(sym);
}

#endif // !DISABLE_AGG_JIT


AQPPipelineFn
IrToLlvmCompiler::CompilePipeline(const AQPStmt *filter_node,
                                  const AQPStmt *proj_node,
                                  const std::vector<ColSchema> &in_schema,
                                  std::vector<uint8_t> *params_out) {

  // Collect filter expressions
  std::vector<const AQPExpr *> filter_exprs;
  if (filter_node) {
    for (const auto &qe : filter_node->qual_vec)
      filter_exprs.push_back(qe.get());
  }

  const bool template_mode = cache_mode_ == 2;

  // Build projection column mapping
  std::vector<int> col_mapping;
  std::vector<int32_t> col_dtypes;
  if (proj_node && !proj_node->target_list.empty()) {
    for (const auto &attr : proj_node->target_list) {
      int found = -1;
      int32_t dtype = AQP_DTYPE_OTHER;
      for (int i = 0; i < (int)in_schema.size(); i++) {
        if (in_schema[i].table_idx == attr->GetTableIndex() &&
            in_schema[i].col_idx == attr->GetColumnIndex()) {
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

  if (filter_exprs.empty() && col_mapping.empty())
    return nullptr; // nothing to compile

  uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
  std::string fn_name = "aqp_pipe_" + std::to_string(fn_id);

  // In-memory cache lookup
  std::string cache_key;
  if (cache_enabled_ && impl_->cache_enabled) {
    std::string ft = template_mode
        ? SerializeStmtTemplate(filter_node)
        : (filter_node ? const_cast<AQPStmt *>(filter_node)->Print(false, 0) : "");
    std::string pt = proj_node
        ? const_cast<AQPStmt *>(proj_node)->Print(false, 0) : "";
    std::string opt_tag = std::to_string((int)simd_isa_) + "F" +
                          std::to_string((int)fast_mode_) +
                          "C" + std::to_string(cache_mode_);
    cache_key = Impl::ComputeCacheKey(
        BuildCacheContent("pipe:" + opt_tag, in_schema, ft + "||" + pt));
    fn_name = "aqp_pipe_c" + cache_key.substr(0, 12);
    void *cached = impl_->TryCacheLoad(cache_key, fn_name);
    if (cached) {
      if (template_mode && params_out)
        *params_out = BuildParamsFromStmt(filter_node);
      return reinterpret_cast<AQPPipelineFn>(cached);
    }
  }

  auto ctx = std::make_unique<LLVMContext>();
  auto mod = std::make_unique<Module>("aqp_pipe_mod", *ctx);

  ParamsBuilder tmpl_pb;
  ParamsBuilder *pb_ptr = template_mode ? &tmpl_pb : nullptr;

  // Try SIMD pipeline if enabled and all filter expressions are SIMD-friendly
  // TODO: SIMD pipeline has control flow issues — disabled until fixed
  Function *fn = nullptr;
  bool used_simd = false;
  if (false && !template_mode && use_simd_ && impl_->vec_width > 1 &&
      (filter_exprs.empty() || AllExprsSIMDFriendly(filter_exprs, in_schema))) {
    fn = BuildPipelineFunctionSIMD(*ctx, *mod, fn_name, filter_exprs,
                                   col_mapping, col_dtypes, in_schema,
                                   impl_->vec_width);
    if (fn) {
      used_simd = true;
#ifndef NDEBUG
      std::cerr << "[AQP-JIT] using SIMD pipeline (VW=" << impl_->vec_width
                << ")\n";
#endif
    } else {
#ifndef NDEBUG
      std::cerr << "[AQP-JIT] SIMD pipeline failed → scalar fallback\n";
#endif
    }
  }
  if (!fn)
    fn = BuildPipelineFunction(*ctx, *mod, fn_name, filter_exprs, col_mapping,
                               col_dtypes, in_schema, pb_ptr);
  if (!fn)
    return nullptr;
  SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

  std::string err;
  raw_string_ostream es(err);
  if (verifyFunction(*fn, &es)) {
#ifndef NDEBUG
    std::cerr << "[AQP-JIT] verifyFunction failed (pipeline): " << es.str()
              << "\n";
#endif
    return nullptr;
  }

  OptimiseModule(*mod, skip_opt_);

  impl_->pending_cache_key = cache_key;
  auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
  if (auto e = impl_->jit->addIRModule(impl_->current_tracker, std::move(tsm))) {
    impl_->pending_cache_key.clear();
#ifndef NDEBUG
    std::cerr << "[AQP-JIT] addIRModule failed (pipeline)\n";
#endif
    logAllUnhandledErrors(std::move(e), errs());
    return nullptr;
  }

  auto sym = impl_->jit->lookup(fn_name);
  impl_->pending_cache_key.clear();
  if (!sym) {
#ifndef NDEBUG
    std::cerr << "[AQP-JIT] lookup failed for " << fn_name << "\n";
#endif
    logAllUnhandledErrors(sym.takeError(), errs());
    return nullptr;
  }

#ifndef NDEBUG
  std::cerr << "[AQP-JIT] compiled pipeline fn=" << fn_name
            << "  filter_exprs=" << filter_exprs.size()
            << "  out_cols=" << col_mapping.size()
            << (cache_key.empty() ? "" : " [cached]") << "\n";
#endif

  if (template_mode && params_out) {
    tmpl_pb.Finalize();
    *params_out = std::move(tmpl_pb.buf);
  }

  return AQP_JIT_GET_FN(AQPPipelineFn, sym);
}

// ---------------------------------------------------------------------------
// Filter + Aggregate fusion: one loop, no intermediate DataChunk.
//   void fn(AQPChunkView *in, void *agg_state)
// For each row: evaluate filter; if match, update accumulators.
// ---------------------------------------------------------------------------
#if !DISABLE_AGG_JIT
void *IrToLlvmCompiler::CompileFilterAggFusion(
    const AQPStmt *filter_node, const AQPStmt *agg_node,
    const std::vector<ColSchema> &in_schema) {

  if (!agg_node)
    return nullptr;
  auto *agg = dynamic_cast<const SimplestAggregate *>(agg_node);
  if (!agg || agg->agg_fns.empty())
    return nullptr;
  if (!agg->groups.empty())
    return nullptr; // grouped agg not supported yet

  // Build filter expressions
  std::vector<const AQPExpr *> filter_exprs;
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
            in_schema[i].col_idx == fn_pair.first->GetColumnIndex()) {
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
  Type *i8 = Type::getInt8Ty(*ctx);
  Type *i8p = PointerType::getUnqual(i8);
  Type *i32 = Type::getInt32Ty(*ctx);
  Type *i64 = Type::getInt64Ty(*ctx);
  Type *i64p = PointerType::getUnqual(i64);
  Type *voidTy = Type::getVoidTy(*ctx);

  StructType *ColViewTy = StructType::get(*ctx, {i8p, i64p, i32, i32});
  StructType *ChunkViewTy =
      StructType::get(*ctx, {PointerType::getUnqual(ColViewTy), i64, i64});
  StructType *SelViewTy =
      StructType::get(*ctx, {PointerType::getUnqual(i32), i32});

  FunctionType *fn_ty = FunctionType::get(
      voidTy, {PointerType::getUnqual(ChunkViewTy), i8p}, false);
  Function *fn =
      Function::Create(fn_ty, Function::ExternalLinkage, fn_name, mod.get());

  Value *in_arg = fn->getArg(0);
  in_arg->setName("in");
  Value *state_arg = fn->getArg(1);
  state_arg->setName("state");

  BasicBlock *entry_bb = BasicBlock::Create(*ctx, "entry", fn);
  BasicBlock *loop_bb = BasicBlock::Create(*ctx, "loop", fn);
  BasicBlock *body_bb = BasicBlock::Create(*ctx, "body", fn);
  BasicBlock *agg_bb = BasicBlock::Create(*ctx, "agg_update", fn);
  BasicBlock *next_bb = BasicBlock::Create(*ctx, "next", fn);
  BasicBlock *exit_bb = BasicBlock::Create(*ctx, "exit", fn);

  // Use a dummy sel_arg for CompileCtx (not used for aggregation)
  Value *dummy_sel =
      ConstantPointerNull::get(PointerType::getUnqual(SelViewTy));
  CompileCtx cc(*ctx, *mod, in_schema, in_arg, dummy_sel);
  cc.b.SetInsertPoint(entry_bb);

  // Load nrows
  Value *nrows = cc.b.CreateLoad(
      i64, cc.b.CreateStructGEP(ChunkViewTy, in_arg, 1), "nrows");

  // Load column data + validity
  cc.col_data.resize(in_schema.size());
  cc.col_validity.resize(in_schema.size());
  for (size_t i = 0; i < in_schema.size(); i++) {
    cc.col_data[i] = cc.LoadColData((unsigned)i);
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
    if (op.col_idx < 0)
      continue;

    Value *acc_ptr = cc.b.CreateBitCast(
        cc.b.CreateGEP(i8, state_arg, ConstantInt::get(i64, op.state_offset)),
        PointerType::getUnqual(i64));

    bool is_float =
        (op.dtype == AQP_DTYPE_FLOAT || op.dtype == AQP_DTYPE_DOUBLE);

    // Load value based on dtype
    Value *val = nullptr;
    if (op.dtype == AQP_DTYPE_INT32 || op.dtype == AQP_DTYPE_DATE) {
      Value *p = cc.b.CreateBitCast(cc.col_data[op.col_idx],
                                    PointerType::getUnqual(i32));
      val = cc.b.CreateSExt(cc.b.CreateLoad(i32, cc.b.CreateGEP(i32, p, row_i)),
                            i64);
    } else if (op.dtype == AQP_DTYPE_INT64) {
      Value *p = cc.b.CreateBitCast(cc.col_data[op.col_idx],
                                    PointerType::getUnqual(i64));
      val = cc.b.CreateLoad(i64, cc.b.CreateGEP(i64, p, row_i));
    } else {
      continue; // unsupported dtype
    }

    Value *acc = cc.b.CreateLoad(i64, acc_ptr);
    switch (op.agg_type) {
    case 3:
      cc.b.CreateStore(cc.b.CreateAdd(acc, val), acc_ptr);
      break; // SUM
    case 5:
      cc.b.CreateStore(cc.b.CreateAdd(acc, ConstantInt::get(i64, 1)), acc_ptr);
      break;  // COUNT
    case 1: { // MIN
      Value *cmp = cc.b.CreateICmpSLT(val, acc);
      cc.b.CreateStore(cc.b.CreateSelect(cmp, val, acc), acc_ptr);
      break;
    }
    case 2: { // MAX
      Value *cmp = cc.b.CreateICmpSGT(val, acc);
      cc.b.CreateStore(cc.b.CreateSelect(cmp, val, acc), acc_ptr);
      break;
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
#ifndef NDEBUG
    std::cerr << "[AQP-JIT] verifyFunction failed (filt_agg): " << es.str()
              << "\n";
#endif
    return nullptr;
  }

  OptimiseModule(*mod, skip_opt_);

  auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
  if (auto e = impl_->jit->addIRModule(impl_->current_tracker, std::move(tsm))) {
    logAllUnhandledErrors(std::move(e), errs());
    return nullptr;
  }

  auto sym = impl_->jit->lookup(fn_name);
  if (!sym) {
    logAllUnhandledErrors(sym.takeError(), errs());
    return nullptr;
  }

#ifndef NDEBUG
  std::cerr << "[AQP-JIT] compiled filter+agg fusion fn=" << fn_name
            << "  filter_exprs=" << filter_exprs.size()
            << "  agg_ops=" << agg_ops.size() << "\n";
#endif

  return AQP_JIT_GET_ADDR(sym);
}
#endif // !DISABLE_AGG_JIT


// ---------------------------------------------------------------------------
// Level 3: Filter + HashProbe + Projection fusion (probe pipeline).
// Direct-HT path: probes DuckDB's JoinHashTable via AQPJoinHTView.
// Emits inline MurmurHash64 + salt-aware linear probe + chain walk.
// ---------------------------------------------------------------------------
void *IrToLlvmCompiler::CompileFilterProbeProjectFusion(
    const AQPStmt *filter_node, const AQPStmt *join_node,
    const AQPStmt *proj_node, const std::vector<ColSchema> &probe_schema,
    const std::vector<ColSchema> &payload_schema,
    const std::vector<int> &payload_row_indices,
    const std::vector<int> &lhs_output_idxs,
    const std::vector<int> &rhs_output_layout_idxs,
    const std::vector<int32_t> &lhs_output_dtypes,
    const std::vector<int32_t> &rhs_output_dtypes,
    const std::vector<int> &lhs_key_chunk_idxs,
    const std::vector<int32_t> &lhs_key_dtypes,
    std::vector<uint8_t> *params_out) {

  const SimplestJoin *join =
      join_node ? dynamic_cast<const SimplestJoin *>(join_node) : nullptr;
  if (!join && lhs_key_chunk_idxs.empty()) {
    return nullptr;
  }

  const bool template_mode = cache_mode_ == 2;

  // Build filter expressions
  std::vector<const AQPExpr *> filter_exprs;
  if (filter_node) {
    for (const auto &qe : filter_node->qual_vec)
      filter_exprs.push_back(qe.get());
  }
  if (!template_mode) SortFiltersByCost(filter_exprs);

  // Extract probe key columns. Prefer DuckDB-authoritative chunk positions
  // when supplied (one per join condition, same order). Otherwise fall back
  // to AQP IR (table_idx, col_idx) lookup against probe_schema — which is
  // unsafe when AQP IR ordering diverges from DuckDB's physical chunk.
  std::vector<HashColDesc> probe_key_cols;
  unsigned key_width = 0;
  const size_t num_join_conds =
      join ? join->join_conditions.size() : lhs_key_chunk_idxs.size();
  const bool use_duckdb_keys =
      !lhs_key_chunk_idxs.empty() &&
      lhs_key_chunk_idxs.size() == num_join_conds &&
      lhs_key_dtypes.size() == lhs_key_chunk_idxs.size();
  if (use_duckdb_keys) {
    for (size_t i = 0; i < lhs_key_chunk_idxs.size(); ++i) {
      HashColDesc kc;
      kc.col_idx = lhs_key_chunk_idxs[i];
      kc.dtype = lhs_key_dtypes[i];
      if (kc.col_idx < 0 || kc.col_idx >= (int)probe_schema.size()) {
#ifndef NDEBUG
        std::cerr << "[AQP-JIT] fused probe: duckdb key idx " << kc.col_idx
                  << " out of range (probe_schema size="
                  << probe_schema.size() << ")\n";
#endif
        return nullptr;
      }
      kc.elem_size = DtypeElemSize(kc.dtype);
      if (kc.elem_size == 0) {
        return nullptr;
      }
      key_width += kc.elem_size;
      probe_key_cols.push_back(kc);
    }
  } else if (join) {
    for (const auto &cond : join->join_conditions) {
      HashColDesc kc;
      kc.col_idx = -1;
      for (int i = 0; i < (int)probe_schema.size(); i++) {
        if (probe_schema[i].table_idx == cond->left_attr->GetTableIndex() &&
            probe_schema[i].col_idx == cond->left_attr->GetColumnIndex()) {
          kc.col_idx = i;
          kc.dtype = probe_schema[i].dtype;
          break;
        }
      }
      if (kc.col_idx < 0) {
        for (int i = 0; i < (int)probe_schema.size(); i++) {
          if (probe_schema[i].table_idx == cond->right_attr->GetTableIndex() &&
              probe_schema[i].col_idx == cond->right_attr->GetColumnIndex()) {
            kc.col_idx = i;
            kc.dtype = probe_schema[i].dtype;
            break;
          }
        }
      }
      if (kc.col_idx < 0) {
#ifndef NDEBUG
        std::cerr << "[AQP-JIT] fused probe: key not in probe schema\n";
#endif
        return nullptr;
      }
      kc.elem_size = DtypeElemSize(kc.dtype);
      if (kc.elem_size == 0)
        return nullptr;
      key_width += kc.elem_size;
      probe_key_cols.push_back(kc);
    }
  } else {
    return nullptr;
  }

  // skip_hash_cmp: for integer keys, skip salt comparison in probe loop
  bool all_keys_integer = true;
  for (const auto &kc : probe_key_cols) {
    if (kc.dtype != AQP_DTYPE_INT8 && kc.dtype != AQP_DTYPE_INT16 &&
        kc.dtype != AQP_DTYPE_INT32 && kc.dtype != AQP_DTYPE_INT64) {
      all_keys_integer = false;
      break;
    }
  }
  const bool do_skip_salt = (skip_hash_cmp_ == 2 && all_keys_integer);
#ifndef NDEBUG
  std::cerr << "[AQP-JIT] skip_hash_cmp: mode=" << skip_hash_cmp_
            << " all_keys_integer=" << all_keys_integer
            << " nkeys=" << probe_key_cols.size()
            << " do_skip_salt=" << do_skip_salt << "\n";
#endif

  // Compute payload column byte offsets
  struct PayloadColInfo {
    unsigned offset;
    unsigned elem_size;
    int32_t dtype;
  };
  std::vector<PayloadColInfo> payload_infos;
  unsigned payload_width = 0;
  for (const auto &ps : payload_schema) {
    PayloadColInfo pi;
    pi.offset = payload_width;
    pi.dtype = ps.dtype;
    pi.elem_size = DtypeElemSize(ps.dtype);
    if (pi.elem_size == 0) {
      return nullptr;
    }
    payload_width += pi.elem_size;
    payload_infos.push_back(pi);
  }

  // Build output column mapping: each output col comes from PROBE or PAYLOAD
  struct OutColDesc {
    enum { PROBE, PAYLOAD } source;
    int probe_col_idx;
    int payload_col_idx;
    unsigned payload_offset;
    int32_t dtype;
    unsigned elem_size;
  };
  std::vector<OutColDesc> out_cols;

  if (proj_node && !proj_node->target_list.empty()) {
    for (const auto &attr : proj_node->target_list) {
      OutColDesc oc;
      oc.probe_col_idx = -1;
      oc.payload_col_idx = -1;
      oc.payload_offset = 0;
      oc.source = OutColDesc::PROBE;

      for (int i = 0; i < (int)probe_schema.size(); i++) {
        if (probe_schema[i].table_idx == attr->GetTableIndex() &&
            probe_schema[i].col_idx == attr->GetColumnIndex()) {
          oc.source = OutColDesc::PROBE;
          oc.probe_col_idx = i;
          oc.dtype = probe_schema[i].dtype;
          oc.elem_size = DtypeElemSize(oc.dtype);
          break;
        }
      }
      if (oc.probe_col_idx < 0) {
        for (int i = 0; i < (int)payload_schema.size(); i++) {
          if (payload_schema[i].table_idx == attr->GetTableIndex() &&
              payload_schema[i].col_idx == attr->GetColumnIndex()) {
            oc.source = OutColDesc::PAYLOAD;
            oc.payload_col_idx = i;
            oc.payload_offset = payload_infos[i].offset;
            oc.dtype = payload_schema[i].dtype;
            oc.elem_size = DtypeElemSize(oc.dtype);
            break;
          }
        }
      }
      if (oc.probe_col_idx < 0 && oc.payload_col_idx < 0) {
#ifndef NDEBUG
        std::cerr << "[AQP-JIT] fused probe: projected col (table="
                  << attr->GetTableIndex() << " col=" << attr->GetColumnIndex()
                  << ") not in probe or payload schema\n";
#endif
        return nullptr;
      }
      if (oc.elem_size == 0)
        return nullptr;
      out_cols.push_back(oc);
    }
  } else if (!lhs_output_idxs.empty() || !rhs_output_layout_idxs.empty()) {
    // Explicit subset matching the operator's chunk shape: [lhs cols, rhs cols].
    // lhs_output_idxs index probe_schema; rhs_output_layout_idxs index the HT
    // layout = [keys, payload]. dtype/elem_size for each output column MUST
    // come from DuckDB's actual chunk schema (lhs/rhs_output_dtypes); the AQP
    // IR's probe_schema/payload_schema may have a different column ordering
    // and would produce wrong elem_sizes (e.g. VARCHAR=16 vs INT=4) — that
    // causes the JIT to write the wrong number of bytes per output column.
    const unsigned num_keys_ = static_cast<unsigned>(probe_key_cols.size());
    const bool have_lhs_dtypes =
        lhs_output_dtypes.size() == lhs_output_idxs.size();
    const bool have_rhs_dtypes =
        rhs_output_dtypes.size() == rhs_output_layout_idxs.size();
    for (size_t i = 0; i < lhs_output_idxs.size(); ++i) {
      int idx = lhs_output_idxs[i];
      if (idx < 0 || idx >= (int)probe_schema.size()) {
#ifndef NDEBUG
        std::cerr << "[AQP-JIT] direct-probe: lhs idx " << idx
                  << " out of range\n";
#endif
        return nullptr;
      }
      OutColDesc oc;
      oc.source = OutColDesc::PROBE;
      oc.probe_col_idx = idx;
      oc.payload_col_idx = -1;
      oc.payload_offset = 0;
      oc.dtype =
          have_lhs_dtypes ? lhs_output_dtypes[i] : probe_schema[idx].dtype;
      oc.elem_size = DtypeElemSize(oc.dtype);
      if (oc.elem_size == 0)
        return nullptr;
      out_cols.push_back(oc);
    }
    for (size_t i = 0; i < rhs_output_layout_idxs.size(); ++i) {
      int layout_idx = rhs_output_layout_idxs[i];
      OutColDesc oc;
      oc.source = OutColDesc::PAYLOAD;
      oc.probe_col_idx = -1;
      oc.payload_offset = 0;
      if (layout_idx < (int)num_keys_) {
        // Key column stored at data_offsets[layout_idx].
        // payload_col_idx encodes layout_idx - num_keys (negative -> key path)
        oc.payload_col_idx = layout_idx - (int)num_keys_;
        oc.dtype = have_rhs_dtypes ? rhs_output_dtypes[i]
                                   : probe_key_cols[layout_idx].dtype;
      } else {
        int pi = layout_idx - (int)num_keys_;
        oc.payload_col_idx = pi;
        if (have_rhs_dtypes) {
          oc.dtype = rhs_output_dtypes[i];
        } else {
          if (pi < 0 || pi >= (int)payload_schema.size()) {
#ifndef NDEBUG
            std::cerr << "[AQP-JIT] direct-probe: rhs layout_idx "
                      << layout_idx << " out of range\n";
#endif
            return nullptr;
          }
          oc.dtype = payload_schema[pi].dtype;
        }
      }
      oc.elem_size = DtypeElemSize(oc.dtype);
      if (oc.elem_size == 0)
        return nullptr;
      out_cols.push_back(oc);
    }
  } else {
    // No projection: output all probe cols then all payload cols
    for (int i = 0; i < (int)probe_schema.size(); i++) {
      OutColDesc oc;
      oc.source = OutColDesc::PROBE;
      oc.probe_col_idx = i;
      oc.payload_col_idx = -1;
      oc.payload_offset = 0;
      oc.dtype = probe_schema[i].dtype;
      oc.elem_size = DtypeElemSize(oc.dtype);
      if (oc.elem_size == 0)
        return nullptr;
      out_cols.push_back(oc);
    }
    for (int i = 0; i < (int)payload_schema.size(); i++) {
      OutColDesc oc;
      oc.source = OutColDesc::PAYLOAD;
      oc.probe_col_idx = -1;
      oc.payload_col_idx = i;
      oc.payload_offset = payload_infos[i].offset;
      oc.dtype = payload_schema[i].dtype;
      oc.elem_size = DtypeElemSize(oc.dtype);
      if (oc.elem_size == 0)
        return nullptr;
      out_cols.push_back(oc);
    }
  }

  if (out_cols.empty())
    return nullptr;

  // ---- LLVM IR generation: direct-HT probe ----
  // Function probes DuckDB's JoinHashTable directly via AQPJoinHTView:
  //   - inline MurmurHash64 per key, CombineHash for multi-key
  //   - salt-aware linear probe over ht_entry_t[]
  //   - typed key compare against row bytes (offsets from view->data_offsets)
  //   - chain walk via *(row_ptr + view->pointer_offset)
  // Pre-condition (validated by caller): payload_row_indices.size() == payload_schema.size();
  // for each output PAYLOAD column oc, the row offset is
  //   view->data_offsets[num_keys + payload_row_indices[oc.payload_col_idx]].
  uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
  std::string fn_name = "aqp_filt_hprobe_proj_direct_" + std::to_string(fn_id);

  std::string cache_key;
  if (cache_enabled_ && impl_->cache_enabled) {
    std::string extra;
    if (template_mode)
      extra += SerializeStmtTemplate(filter_node);
    else if (filter_node)
      extra += const_cast<AQPStmt *>(filter_node)->Print(false, 0);
    extra += "||";
    if (join_node)
      extra += const_cast<AQPStmt *>(join_node)->Print(false, 0);
    extra += "||";
    if (proj_node)
      extra += const_cast<AQPStmt *>(proj_node)->Print(false, 0);
    extra += "||";
    extra += BuildCacheContent("pay", payload_schema);
    AppendIntVec(extra, "pri", payload_row_indices);
    AppendIntVec(extra, "lo", lhs_output_idxs);
    AppendIntVec(extra, "ro", rhs_output_layout_idxs);
    AppendIntVec(extra, "ld", lhs_output_dtypes);
    AppendIntVec(extra, "rd", rhs_output_dtypes);
    AppendIntVec(extra, "lk", lhs_key_chunk_idxs);
    AppendIntVec(extra, "kd", lhs_key_dtypes);
    std::string opt_tag =
        std::to_string((int)simd_isa_) + "F" +
        std::to_string((int)fast_mode_) + (skip_opt_ ? "n" : "o") +
        "k" + std::to_string(skip_hash_cmp_) + (batch_probe_ ? "b" : "_") +
        (prefetch_ ? "p" : "_") + std::to_string(prefetch_distance_) + "." +
        std::to_string(prefetch_entry_distance_) + "." +
        std::to_string(prefetch_row_distance_) +
        "C" + std::to_string(cache_mode_);
    cache_key = Impl::ComputeCacheKey(
        BuildCacheContent("fprobe:" + opt_tag, probe_schema, extra));
    fn_name = "aqp_fpp_c" + cache_key.substr(0, 12);
    void *cached = impl_->TryCacheLoad(cache_key, fn_name);
    if (cached) {
      if (template_mode && params_out)
        *params_out = BuildParamsFromStmt(filter_node);
      return cached;
    }
  }

  auto ctx = std::make_unique<LLVMContext>();
  auto mod = std::make_unique<Module>("aqp_filt_hprobe_proj_direct_mod", *ctx);

  Type *i8 = Type::getInt8Ty(*ctx);
  Type *i8p = PointerType::getUnqual(i8);
  Type *i32 = Type::getInt32Ty(*ctx);
  Type *i64 = Type::getInt64Ty(*ctx);
  Type *i64p = PointerType::getUnqual(i64);
  Type *i16 = Type::getInt16Ty(*ctx);

  StructType *ColViewTy = StructType::get(*ctx, {i8p, i64p, i32, i32});
  StructType *ChunkViewTy =
      StructType::get(*ctx, {PointerType::getUnqual(ColViewTy), i64, i64});
  StructType *SelViewTy =
      StructType::get(*ctx, {PointerType::getUnqual(i32), i32});
  // AQPJoinHTView layout (must match aqp_jit_abi.h / aqp_jit.hpp):
  //   { void *entries; u64 bitmask; u64 use_salt; void *layout_ptr;
  //     u32 tuple_size; u32 pointer_offset; const u64 *data_offsets;
  //     u64 no_chains; const u64 *bf_data; u64 bf_bitmask; u64 has_row_validity; }
  StructType *ViewTy = StructType::get(
      *ctx, {i8p, i64, i64, i8p, i32, i32, i64p, i64, i64p, i64, i64});

  // int64_t fn(AQPChunkView *in, AQPChunkView *out, void *view)
  FunctionType *fn_ty = FunctionType::get(
      i64,
      {PointerType::getUnqual(ChunkViewTy),
       PointerType::getUnqual(ChunkViewTy), i8p},
      false);
  Function *fn =
      Function::Create(fn_ty, Function::ExternalLinkage, fn_name, mod.get());

  Value *in_arg = fn->getArg(0);
  in_arg->setName("in");
  Value *out_arg = fn->getArg(1);
  out_arg->setName("out");
  Value *ht_arg = fn->getArg(2);
  ht_arg->setName("view");

  BasicBlock *entry_bb = BasicBlock::Create(*ctx, "entry", fn);
  BasicBlock *outer_bb = BasicBlock::Create(*ctx, "outer", fn);
  BasicBlock *body_bb = BasicBlock::Create(*ctx, "body", fn);
  BasicBlock *hash_bb = BasicBlock::Create(*ctx, "hash", fn);
  BasicBlock *probe_bb = BasicBlock::Create(*ctx, "probe", fn);
  BasicBlock *salt_ok_bb =
      do_skip_salt ? nullptr : BasicBlock::Create(*ctx, "salt_ok", fn);
  BasicBlock *key_eq_bb = BasicBlock::Create(*ctx, "key_eq", fn);
  BasicBlock *miss_bb = BasicBlock::Create(*ctx, "miss", fn);
  BasicBlock *chain_bb = BasicBlock::Create(*ctx, "chain", fn);
  BasicBlock *emit_bb = BasicBlock::Create(*ctx, "emit", fn);
  BasicBlock *advance_bb = BasicBlock::Create(*ctx, "advance", fn);
  BasicBlock *chain_done_bb = BasicBlock::Create(*ctx, "chain_done", fn);
  BasicBlock *next_bb = BasicBlock::Create(*ctx, "next", fn);
  BasicBlock *bail_bb = BasicBlock::Create(*ctx, "bail", fn);
  BasicBlock *exit_bb = BasicBlock::Create(*ctx, "exit", fn);

  ParamsBuilder tmpl_pb;

  Value *dummy_sel =
      ConstantPointerNull::get(PointerType::getUnqual(SelViewTy));
  CompileCtx cc(*ctx, *mod, probe_schema, in_arg, dummy_sel);
  cc.b.SetInsertPoint(entry_bb);

  if (template_mode) {
    cc.template_mode = true;
    cc.params_builder = &tmpl_pb;
    cc.LoadParamsFromThreadLocal();
  }

  // ---- Entry: load view fields once ----
  Value *view_ptr =
      cc.b.CreateBitCast(ht_arg, PointerType::getUnqual(ViewTy), "view_ptr");
  Value *v_entries = cc.b.CreateLoad(
      i8p, cc.b.CreateStructGEP(ViewTy, view_ptr, 0), "v_entries");
  Value *v_bitmask = cc.b.CreateLoad(
      i64, cc.b.CreateStructGEP(ViewTy, view_ptr, 1), "v_bitmask");
  // field 2 use_salt: skipped when do_skip_salt, otherwise always checked
  // field 3 layout_ptr: only for debug
  // field 4 tuple_size: not needed (data_offsets has every column)
  Value *v_ptr_off32 = cc.b.CreateLoad(
      i32, cc.b.CreateStructGEP(ViewTy, view_ptr, 5), "v_ptr_off");
  Value *v_ptr_off = cc.b.CreateZExt(v_ptr_off32, i64);
  Value *v_offsets = cc.b.CreateLoad(
      i64p, cc.b.CreateStructGEP(ViewTy, view_ptr, 6), "v_offsets");
  Value *v_no_chains = cc.b.CreateLoad(
      i64, cc.b.CreateStructGEP(ViewTy, view_ptr, 7), "v_no_chains");
  Value *skip_chain_walk = cc.b.CreateICmpNE(
      v_no_chains, ConstantInt::get(i64, 0), "skip_chain");
  Value *v_bf_data = cc.b.CreateLoad(
      i64p, cc.b.CreateStructGEP(ViewTy, view_ptr, 8), "v_bf_data");
  Value *v_bf_bitmask = cc.b.CreateLoad(
      i64, cc.b.CreateStructGEP(ViewTy, view_ptr, 9), "v_bf_bitmask");
  Value *has_bf = cc.b.CreateICmpNE(
      cc.b.CreatePtrToInt(v_bf_data, i64), ConstantInt::get(i64, 0), "has_bf");
  Value *v_has_rv64 = cc.b.CreateLoad(
      i64, cc.b.CreateStructGEP(ViewTy, view_ptr, 10), "v_has_rv");
  Value *has_rv = cc.b.CreateICmpNE(
      v_has_rv64, ConstantInt::get(i64, 0), "has_rv");

  // Phase 7.1 — hoist data_offsets[*] used in the row loop into entry-block
  // SSA values. Inside the body LLVM treats `load i64, GEP v_offsets, idx`
  // as having unknown aliasing, so it cannot hoist on its own. Loading each
  // needed offset once here lets the register allocator keep them in GPRs.
  //
  // Key is the layout index into v_offsets, signed `int` so the build-key
  // projection case (oc.payload_col_idx < 0 → row_col_idx < 0) sums
  // naturally back to a positive layout_idx in `num_keys + row_col_idx`.
  // Earlier versions used size_t and relied on unsigned wraparound — that
  // worked but masked drift bugs because operator[] returns a default
  // nullptr Value* on a missing key, which would crash inside CreateGEP.
  std::unordered_map<int, Value *> hoisted_offsets;
  {
    auto load_off = [&](int idx) {
      if (hoisted_offsets.count(idx)) return;
      Value *off = cc.b.CreateLoad(
          i64,
          cc.b.CreateGEP(i64, v_offsets, ConstantInt::get(i64, (int64_t)idx)),
          "off_" + std::to_string(idx));
      hoisted_offsets[idx] = off;
    };
    for (int j = 0; j < (int)probe_key_cols.size(); j++) load_off(j);
    const int nk = (int)probe_key_cols.size();
    for (auto &oc : out_cols) {
      if (oc.source == OutColDesc::PROBE) continue;
      int k = oc.payload_col_idx;
      int row_col_idx =
          (k >= 0 && k < (int)payload_row_indices.size())
              ? payload_row_indices[k]
              : k;
      load_off(nk + row_col_idx);
    }
  }
  auto get_hoisted = [&](int idx) -> Value * {
    auto it = hoisted_offsets.find(idx);
    assert(it != hoisted_offsets.end() &&
           "hoisted_offsets miss — hoist/emit index math drift");
    return it->second;
  };

  Constant *POINTER_MASK = ConstantInt::get(i64, 0x0000FFFFFFFFFFFFULL);

  Value *nrows = cc.b.CreateLoad(
      i64, cc.b.CreateStructGEP(ChunkViewTy, in_arg, 1), "nrows");

  // Load input column data + validity
  cc.col_data.resize(probe_schema.size());
  cc.col_validity.resize(probe_schema.size());
  for (size_t i = 0; i < probe_schema.size(); i++) {
    cc.col_data[i] = cc.LoadColData((unsigned)i);
    cc.col_validity[i] = cc.LoadColValidity((unsigned)i);
  }

  // Load output column data + validity pointers
  Value *out_cols_pp = cc.b.CreateStructGEP(ChunkViewTy, out_arg, 0);
  Value *out_cols_v = cc.b.CreateLoad(PointerType::getUnqual(ColViewTy),
                                      out_cols_pp, "out_cols");
  std::vector<Value *> out_data_ptrs;
  std::vector<Value *> out_valid_ptrs;
  for (size_t oi = 0; oi < out_cols.size(); oi++) {
    Value *col_i =
        cc.b.CreateGEP(ColViewTy, out_cols_v, ConstantInt::get(i64, oi));
    out_data_ptrs.push_back(
        cc.b.CreateLoad(i8p, cc.b.CreateStructGEP(ColViewTy, col_i, 0),
                        "out_data_" + std::to_string(oi)));
    out_valid_ptrs.push_back(
        cc.b.CreateLoad(i64p, cc.b.CreateStructGEP(ColViewTy, col_i, 1),
                        "out_valid_" + std::to_string(oi)));
  }

  // ---- NULL-validity helpers ----
  // The dispatcher (MakeChunkViewAt with writable_validity) guarantees every
  // output column has a non-null, all-valid mask; the emit path clears bits
  // for NULL outputs. Input masks may be null (= all valid) — the select-on-
  // address trick below reads a constant all-ones word in that case, keeping
  // the hot path branchless.
  GlobalVariable *all_valid_gv = new GlobalVariable(
      *mod, i64, /*isConstant=*/true, GlobalValue::PrivateLinkage,
      ConstantInt::get(i64, -1), "aqp_all_valid");
  ArrayType *zero_pad_ty = ArrayType::get(i8, 64);
  GlobalVariable *zero_pad_gv = new GlobalVariable(
      *mod, zero_pad_ty, /*isConstant=*/true, GlobalValue::PrivateLinkage,
      ConstantAggregateZero::get(zero_pad_ty), "aqp_zero_pad");

  // i1: validity of input column col_idx at row r (nullptr mask = valid).
  auto emit_input_valid = [&](int col_idx, Value *r) -> Value * {
    Value *vp = cc.col_validity[col_idx];
    Value *has_mask = cc.b.CreateICmpNE(
        cc.b.CreatePtrToInt(vp, i64), ConstantInt::get(i64, 0));
    Value *word_idx = cc.b.CreateLShr(r, ConstantInt::get(i64, 6));
    Value *word_addr = cc.b.CreateGEP(i64, vp, word_idx);
    Value *addr = cc.b.CreateSelect(has_mask, word_addr, all_valid_gv);
    Value *word = cc.b.CreateLoad(i64, addr);
    Value *bitpos = cc.b.CreateAnd(r, ConstantInt::get(i64, 63));
    Value *bit = cc.b.CreateAnd(cc.b.CreateLShr(word, bitpos),
                                ConstantInt::get(i64, 1));
    return cc.b.CreateICmpNE(bit, ConstantInt::get(i64, 0), "in_valid");
  };
  // i1: validity of HT row column (bit col_bit_idx of the row's validity
  // byte prefix); true when the layout has no validity prefix.
  auto emit_row_valid = [&](Value *row_ptr, int col_bit_idx) -> Value * {
    Value *byte_addr = cc.b.CreateGEP(
        i8, row_ptr, ConstantInt::get(i64, col_bit_idx / 8));
    Value *byte = cc.b.CreateLoad(i8, byte_addr);
    Value *bit = cc.b.CreateAnd(
        cc.b.CreateLShr(byte, ConstantInt::get(i8, col_bit_idx % 8)),
        ConstantInt::get(i8, 1));
    Value *raw = cc.b.CreateICmpNE(bit, ConstantInt::get(i8, 0));
    return cc.b.CreateSelect(has_rv, raw, ConstantInt::getTrue(*ctx),
                             "row_valid");
  };
  // Clear output validity bit out_idx of column oi when !is_valid.
  auto emit_out_validity = [&](size_t oi, Value *out_idx, Value *is_valid) {
    Value *word_idx = cc.b.CreateLShr(out_idx, ConstantInt::get(i64, 6));
    Value *wp = cc.b.CreateGEP(i64, out_valid_ptrs[oi], word_idx);
    Value *word = cc.b.CreateLoad(i64, wp);
    Value *bitpos = cc.b.CreateAnd(out_idx, ConstantInt::get(i64, 63));
    Value *inval = cc.b.CreateZExt(cc.b.CreateNot(is_valid), i64);
    Value *clear_mask = cc.b.CreateShl(inval, bitpos);
    cc.b.CreateStore(cc.b.CreateAnd(word, cc.b.CreateNot(clear_mask)), wp);
  };

  // ---- Hash helpers (shared between stage-1 prelude and stage-2 key cmp) ----
  auto key_llvm_ty = [&](int32_t dt) -> Type * {
    if (dt == AQP_DTYPE_INT32 || dt == AQP_DTYPE_DATE) return i32;
    if (dt == AQP_DTYPE_INT64) return i64;
    if (dt == AQP_DTYPE_INT16) return i16;
    if (dt == AQP_DTYPE_BOOL || dt == AQP_DTYPE_INT8) return i8;
    return nullptr;
  };
  for (auto &kc : probe_key_cols) {
    if (!key_llvm_ty(kc.dtype)) {
      return nullptr;
    }
  }
  Constant *MURMUR_MUL = ConstantInt::get(i64, 0xd6e8feb86659fd93ULL);
  Constant *SHIFT32 = ConstantInt::get(i64, 32);
  auto emitMurmur = [&](Value *x) -> Value * {
    Value *t = cc.b.CreateXor(x, cc.b.CreateLShr(x, SHIFT32));
    t = cc.b.CreateMul(t, MURMUR_MUL);
    t = cc.b.CreateXor(t, cc.b.CreateLShr(t, SHIFT32));
    t = cc.b.CreateMul(t, MURMUR_MUL);
    t = cc.b.CreateXor(t, cc.b.CreateLShr(t, SHIFT32));
    return t;
  };
  auto emitCombine = [&](Value *a, Value *b) -> Value * {
    Value *t = cc.b.CreateXor(a, cc.b.CreateLShr(a, SHIFT32));
    t = cc.b.CreateMul(t, MURMUR_MUL);
    return cc.b.CreateXor(t, b);
  };
  // Per-row: load typed keys + compute MurmurHash64 (CombineHash for multi-key)
  auto compute_keys_hash =
      [&](Value *row_idx_val,
          std::vector<Value *> *out_keys) -> Value * {
    Value *h = nullptr;
    if (out_keys) out_keys->clear();
    for (size_t j = 0; j < probe_key_cols.size(); j++) {
      auto &kc = probe_key_cols[j];
      Type *kty = key_llvm_ty(kc.dtype);
      Value *src = cc.col_data[kc.col_idx];
      Value *elem_ptr = cc.b.CreateGEP(
          i8, src,
          cc.b.CreateMul(row_idx_val, ConstantInt::get(i64, kc.elem_size)));
      Value *typed_ptr =
          cc.b.CreateBitCast(elem_ptr, PointerType::getUnqual(kty));
      Value *kval = cc.b.CreateLoad(kty, typed_ptr);
      if (out_keys) out_keys->push_back(kval);
      // Match DuckDB's Hash<T>: int8/int16 are first cast to uint32_t
      // (sign-extending), then zero-extended to 64 bits. int32 zero-extends.
      Value *u64_val;
      if (kty == i64) {
        u64_val = kval;
      } else if (kty == cc.i32()) {
        u64_val = cc.b.CreateZExt(kval, i64);
      } else {
        u64_val = cc.b.CreateZExt(cc.b.CreateSExt(kval, cc.i32()), i64);
      }
      Value *hj = emitMurmur(u64_val);
      h = (j == 0) ? hj : emitCombine(h, hj);
    }
    return h;
  };

  // ---- Two-stage (ROF) prelude: filter + hash + prefetch ----
  // When batch_probe_ is enabled, walk all rows once first to compute hash and
  // ht_offset, store them in stack arrays, and software-prefetch the target
  // entries[ht_off] cache lines. Stage 2 (outer_bb onwards) then re-uses the
  // precomputed values, hiding L3 latency behind already-issued prefetches.
  // STANDARD_VECTOR_SIZE == 2048; that's the chunk row cap.
  Value *hash_arr_buf = nullptr;
  Value *htoff_arr_buf = nullptr;
  Value *filt_mask_buf = nullptr;
  BasicBlock *s1_outer = nullptr;
  BasicBlock *outer_pred_bb = entry_bb;
  if (batch_probe_) {
    Constant *BUF_SIZE = ConstantInt::get(i64, 2048);
    hash_arr_buf = cc.b.CreateAlloca(i64, BUF_SIZE, "hash_arr");
    htoff_arr_buf = cc.b.CreateAlloca(i64, BUF_SIZE, "htoff_arr");
    filt_mask_buf = cc.b.CreateAlloca(i8, BUF_SIZE, "filt_mask");

    s1_outer = BasicBlock::Create(*ctx, "s1_outer", fn);
    BasicBlock *s1_body = BasicBlock::Create(*ctx, "s1_body", fn);
    BasicBlock *s1_fail = BasicBlock::Create(*ctx, "s1_fail", fn);
    BasicBlock *s1_pass = BasicBlock::Create(*ctx, "s1_pass", fn);
    BasicBlock *s1_next = BasicBlock::Create(*ctx, "s1_next", fn);

    cc.b.CreateBr(s1_outer);

    cc.b.SetInsertPoint(s1_outer);
    PHINode *s1_i = cc.b.CreatePHI(i64, 2, "s1_i");
    s1_i->addIncoming(ConstantInt::get(i64, 0), entry_bb);
    cc.b.CreateCondBr(cc.b.CreateICmpEQ(s1_i, nrows), outer_bb, s1_body);

    cc.b.SetInsertPoint(s1_body);
    cc.row_idx = s1_i;
    EmitShortCircuitFilter(cc, fn, filter_exprs, s1_pass, s1_fail);

    cc.b.SetInsertPoint(s1_fail);
    cc.b.CreateStore(ConstantInt::get(i8, 0),
                     cc.b.CreateGEP(i8, filt_mask_buf, s1_i));
    cc.b.CreateBr(s1_next);

    cc.b.SetInsertPoint(s1_pass);
    Value *s1_hash = compute_keys_hash(s1_i, nullptr);
    Value *s1_htoff = cc.b.CreateAnd(s1_hash, v_bitmask, "s1_htoff");
    cc.b.CreateStore(s1_hash, cc.b.CreateGEP(i64, hash_arr_buf, s1_i));
    cc.b.CreateStore(s1_htoff, cc.b.CreateGEP(i64, htoff_arr_buf, s1_i));
    cc.b.CreateStore(ConstantInt::get(i8, 1),
                     cc.b.CreateGEP(i8, filt_mask_buf, s1_i));
    // Phase 6: stage-1 bulk prefetch removed. Prefetch is issued in stage 2
    // with look-ahead so that MSHRs (~10–12 in-flight on modern x86) are not
    // saturated up-front and the row-store cache line is also covered.
    cc.b.CreateBr(s1_next);

    cc.b.SetInsertPoint(s1_next);
    Value *s1_i_next = cc.b.CreateAdd(s1_i, ConstantInt::get(i64, 1));
    s1_i->addIncoming(s1_i_next, s1_next);
    cc.b.CreateBr(s1_outer);

    outer_pred_bb = s1_outer;
  } else {
    cc.b.CreateBr(outer_bb);
  }

  // ---- Outer loop header ----
  cc.b.SetInsertPoint(outer_bb);
  PHINode *row_i = cc.b.CreatePHI(i64, 2, "i");
  PHINode *out_count = cc.b.CreatePHI(i64, 2, "out_count");
  row_i->addIncoming(ConstantInt::get(i64, 0), outer_pred_bb);
  out_count->addIncoming(ConstantInt::get(i64, 0), outer_pred_bb);
  cc.b.CreateCondBr(cc.b.CreateICmpEQ(row_i, nrows), exit_bb, body_bb);

  // ---- Body: evaluate filter (or in batch_probe mode: read mask) ----
  cc.b.SetInsertPoint(body_bb);
  cc.row_idx = row_i;

  // Phase 6: stage-2 consumer-side look-ahead prefetch. Issued from inside
  // the body loop so MSHRs cover ~D rows of pipeline depth at any time
  // instead of being saturated up-front (see plan §10.1). Active only when
  // batch_probe_ exposes the stage-1 stack arrays for look-up.
  //
  // Safety on the mask byte load: `pf_idx = select(in_bounds, idx_raw,
  // nrows-1)` clamps to a stage-1-written slot when in-bounds, otherwise
  // falls back to nrows-1 (also written, since this code only runs once
  // we've entered body_bb, which requires nrows > 0). So the i8 load is
  // never uninitialized. When in_bounds is false the `in_bounds &&
  // mask_set` AND short-circuits before any further dereference, so even
  // if the chosen mask byte happens to be set we skip pf_bb.
  if (batch_probe_ && prefetch_ &&
      (prefetch_entry_distance_ > 0 || prefetch_row_distance_ > 0)) {
    Function *pf_intrinsic =
        Intrinsic::getDeclaration(mod.get(), Intrinsic::prefetch, {i8p});
    Value *nrows_m1 = cc.b.CreateSub(nrows, ConstantInt::get(i64, 1));

    auto emit_pf = [&](int distance, bool deref_row, int locality) {
      Value *idx_raw = cc.b.CreateAdd(
          row_i, ConstantInt::get(i64, (uint64_t)distance));
      Value *in_bounds = cc.b.CreateICmpULT(idx_raw, nrows);
      Value *idx = cc.b.CreateSelect(in_bounds, idx_raw, nrows_m1, "pf_idx");
      Value *mask_byte = cc.b.CreateLoad(
          i8, cc.b.CreateGEP(i8, filt_mask_buf, idx), "pf_mask");
      Value *mask_set = cc.b.CreateICmpNE(mask_byte, ConstantInt::get(i8, 0));
      Value *do_pf = cc.b.CreateAnd(in_bounds, mask_set);

      BasicBlock *pf_bb = BasicBlock::Create(*ctx, "pf_do", fn);
      BasicBlock *after_pf = BasicBlock::Create(*ctx, "pf_done", fn);
      cc.b.CreateCondBr(do_pf, pf_bb, after_pf);

      cc.b.SetInsertPoint(pf_bb);
      Value *htoff = cc.b.CreateLoad(
          i64, cc.b.CreateGEP(i64, htoff_arr_buf, idx), "pf_htoff");
      Value *entries_typed_pf = cc.b.CreateBitCast(v_entries, i64p);
      Value *entry_addr_pf = cc.b.CreateGEP(i64, entries_typed_pf, htoff);
      if (!deref_row) {
        cc.b.CreateCall(
            pf_intrinsic,
            {cc.b.CreateBitCast(entry_addr_pf, i8p),
             ConstantInt::get(i32, 0),         // rw: read
             ConstantInt::get(i32, locality),  // locality
             ConstantInt::get(i32, 1)});       // cache: data
        cc.b.CreateBr(after_pf);
      } else {
        // Speculatively dereference the (look-ahead) entry to obtain the
        // build-side row pointer. Entry was prefetched at a larger
        // distance, so it should be in cache by now.
        Value *entry_la = cc.b.CreateLoad(i64, entry_addr_pf, "pf_entry");
        Value *entry_nonzero =
            cc.b.CreateICmpNE(entry_la, ConstantInt::get(i64, 0));
        BasicBlock *pf_row_bb = BasicBlock::Create(*ctx, "pf_row", fn);
        cc.b.CreateCondBr(entry_nonzero, pf_row_bb, after_pf);
        cc.b.SetInsertPoint(pf_row_bb);
        Value *row_la_i64 =
            cc.b.CreateAnd(entry_la, POINTER_MASK, "pf_row_i64");
        Value *row_la = cc.b.CreateIntToPtr(row_la_i64, i8p, "pf_row_ptr");
        cc.b.CreateCall(
            pf_intrinsic,
            {row_la,
             ConstantInt::get(i32, 0),         // rw: read
             ConstantInt::get(i32, locality),  // locality
             ConstantInt::get(i32, 1)});       // cache: data
        cc.b.CreateBr(after_pf);
      }

      cc.b.SetInsertPoint(after_pf);
    };

    // Entry-table look-ahead — random access, NTA so we don't pollute L2.
    if (prefetch_entry_distance_ > 0)
      emit_pf(prefetch_entry_distance_, /*deref_row=*/false, /*loc=*/0);
    // Row-store look-ahead — may be reused during chain walk, locality=1.
    if (prefetch_row_distance_ > 0)
      emit_pf(prefetch_row_distance_, /*deref_row=*/true, /*loc=*/1);
  }

  BasicBlock *after_filter_bb;
  if (batch_probe_) {
    Value *mask_byte = cc.b.CreateLoad(
        i8, cc.b.CreateGEP(i8, filt_mask_buf, row_i), "filt_m");
    Value *mask_set = cc.b.CreateICmpNE(mask_byte, ConstantInt::get(i8, 0));
    after_filter_bb = cc.b.GetInsertBlock();
    cc.b.CreateCondBr(mask_set, hash_bb, next_bb);
  } else {
    BasicBlock *filt_fail_stub = BasicBlock::Create(*ctx, "filt_fail", fn);
    EmitShortCircuitFilter(cc, fn, filter_exprs, hash_bb, filt_fail_stub);
    cc.b.SetInsertPoint(filt_fail_stub);
    cc.b.CreateBr(next_bb);
    after_filter_bb = filt_fail_stub;
  }

  // ---- Hash: load probe keys + obtain hash (from arrays in batch mode) ----
  cc.b.SetInsertPoint(hash_bb);

  // Probe key values are always re-loaded at row_i — needed for the key
  // compare in key_eq_bb. In batch mode the *hash* comes from the array.
  std::vector<Value *> probe_key_vals;
  Value *hash_val = nullptr;
  if (batch_probe_) {
    (void)compute_keys_hash(row_i, &probe_key_vals);
    hash_val = cc.b.CreateLoad(
        i64, cc.b.CreateGEP(i64, hash_arr_buf, row_i), "hash_cached");
  } else {
    hash_val = compute_keys_hash(row_i, &probe_key_vals);
  }

  Value *ht_off_init;
  if (batch_probe_) {
    ht_off_init = cc.b.CreateLoad(
        i64, cc.b.CreateGEP(i64, htoff_arr_buf, row_i), "ht_off_cached");
  } else {
    ht_off_init = cc.b.CreateAnd(hash_val, v_bitmask, "ht_off_init");
  }
  Value *probe_salt = do_skip_salt
      ? nullptr
      : cc.b.CreateOr(hash_val, POINTER_MASK, "probe_salt");

  // ---- Bloom filter pre-check: skip HT probe for definite non-matches ----
  BasicBlock *bf_check_bb = BasicBlock::Create(*ctx, "bf_check", fn);
  BasicBlock *bf_miss_bb = BasicBlock::Create(*ctx, "bf_miss", fn);
  cc.b.CreateCondBr(has_bf, bf_check_bb, probe_bb);

  cc.b.SetInsertPoint(bf_check_bb);
  {
    Value *bf_offset = cc.b.CreateAnd(hash_val, v_bf_bitmask, "bf_off");
    Value *bf_word = cc.b.CreateLoad(
        i64, cc.b.CreateGEP(i64, v_bf_data, bf_offset), "bf_word");
    Constant *bf_shift_mask = ConstantInt::get(i64, 0x3F3F3F3F3F3F3F3FULL);
    Value *shifts = cc.b.CreateAnd(hash_val, bf_shift_mask, "bf_shifts");
    Value *bf_mask = ConstantInt::get(i64, 0);
    for (int bit_i = 4; bit_i < 8; bit_i++) {
      Value *shift_byte = cc.b.CreateAnd(
          cc.b.CreateLShr(shifts, ConstantInt::get(i64, bit_i * 8)),
          ConstantInt::get(i64, 0x3F), "bf_sh");
      bf_mask = cc.b.CreateOr(bf_mask,
          cc.b.CreateShl(ConstantInt::get(i64, 1), shift_byte), "bf_m");
    }
    Value *bf_hit = cc.b.CreateICmpEQ(
        cc.b.CreateAnd(bf_word, bf_mask), bf_mask, "bf_hit");
    cc.b.CreateCondBr(bf_hit, probe_bb, bf_miss_bb);
  }

  cc.b.SetInsertPoint(bf_miss_bb);
  cc.b.CreateBr(next_bb);

  // ---- Probe loop: load entry, check empty, check salt ----
  cc.b.SetInsertPoint(probe_bb);
  PHINode *ht_off = cc.b.CreatePHI(i64, 3, "ht_off");
  ht_off->addIncoming(ht_off_init, hash_bb);
  ht_off->addIncoming(ht_off_init, bf_check_bb);

  Value *entries_typed = cc.b.CreateBitCast(v_entries, i64p);
  Value *entry_addr = cc.b.CreateGEP(i64, entries_typed, ht_off);
  Value *entry = cc.b.CreateLoad(i64, entry_addr, "entry");
  Value *is_empty = cc.b.CreateICmpEQ(entry, ConstantInt::get(i64, 0));
  if (do_skip_salt) {
    cc.b.CreateCondBr(is_empty, next_bb, key_eq_bb);
  } else {
    cc.b.CreateCondBr(is_empty, next_bb, salt_ok_bb);

    // ---- Salt compare ----
    cc.b.SetInsertPoint(salt_ok_bb);
    Value *entry_salt = cc.b.CreateOr(entry, POINTER_MASK);
    Value *salt_match = cc.b.CreateICmpEQ(entry_salt, probe_salt);
    cc.b.CreateCondBr(salt_match, key_eq_bb, miss_bb);
  }

  // ---- Key compare ----
  cc.b.SetInsertPoint(key_eq_bb);
  Value *row_ptr_init = cc.b.CreateIntToPtr(
      cc.b.CreateAnd(entry, POINTER_MASK), i8p, "row_ptr");
  Value *all_eq = ConstantInt::getTrue(*ctx);
  for (size_t j = 0; j < probe_key_cols.size(); j++) {
    auto &kc = probe_key_cols[j];
    Type *kty = key_llvm_ty(kc.dtype);
    Value *koff = get_hoisted((int)j);
    Value *row_key_ptr = cc.b.CreateGEP(i8, row_ptr_init, koff);
    Value *typed_ptr =
        cc.b.CreateBitCast(row_key_ptr, PointerType::getUnqual(kty));
    Value *rkval = cc.b.CreateLoad(kty, typed_ptr);
    Value *eq = cc.b.CreateICmpEQ(rkval, probe_key_vals[j]);
    all_eq = cc.b.CreateAnd(all_eq, eq);
    // NULL probe key never matches (inner join semantics). The garbage
    // bytes under a NULL slot could otherwise equal a real build key.
    all_eq = cc.b.CreateAnd(all_eq, emit_input_valid(kc.col_idx, row_i));
  }
  cc.b.CreateCondBr(all_eq, chain_bb, miss_bb);

  // ---- Miss: ht_off = (ht_off + 1) & bitmask; goto probe ----
  cc.b.SetInsertPoint(miss_bb);
  Value *ht_off_next = cc.b.CreateAnd(
      cc.b.CreateAdd(ht_off, ConstantInt::get(i64, 1)), v_bitmask);
  ht_off->addIncoming(ht_off_next, miss_bb);
  cc.b.CreateBr(probe_bb);

  // ---- Chain walk header ----
  cc.b.SetInsertPoint(chain_bb);
  PHINode *chain_ptr = cc.b.CreatePHI(i8p, 2, "chain_ptr");
  PHINode *chain_oc = cc.b.CreatePHI(i64, 2, "chain_oc");
  chain_ptr->addIncoming(row_ptr_init, key_eq_bb);
  chain_oc->addIncoming(out_count, key_eq_bb);
  // Bail to DuckDB interpreter if the next emit would overflow the output
  // vector (STANDARD_VECTOR_SIZE = 2048). Output columns are sized for 2048
  // rows, so writing at index 2048 or higher corrupts the heap.
  Value *overflow =
      cc.b.CreateICmpUGE(chain_oc, ConstantInt::get(i64, 2048));
  cc.b.CreateCondBr(overflow, bail_bb, emit_bb);

  // ---- Emit: write all output columns for this match ----
  cc.b.SetInsertPoint(emit_bb);
  unsigned num_keys = (unsigned)probe_key_cols.size();
  for (size_t oi = 0; oi < out_cols.size(); oi++) {
    auto &oc = out_cols[oi];
    Type *elem_ty = nullptr;
    if (oc.dtype == AQP_DTYPE_INT32 || oc.dtype == AQP_DTYPE_DATE)
      elem_ty = i32;
    else if (oc.dtype == AQP_DTYPE_INT64)
      elem_ty = i64;
    else if (oc.dtype == AQP_DTYPE_FLOAT)
      elem_ty = Type::getFloatTy(*ctx);
    else if (oc.dtype == AQP_DTYPE_DOUBLE)
      elem_ty = Type::getDoubleTy(*ctx);
    else if (oc.dtype == AQP_DTYPE_INT16)
      elem_ty = i16;
    else if (oc.dtype == AQP_DTYPE_BOOL || oc.dtype == AQP_DTYPE_INT8)
      elem_ty = i8;

    Value *is_valid = nullptr;
    Value *src_byte_ptr = nullptr; // i8* to the source slot (memcpy path)
    if (oc.source == OutColDesc::PROBE) {
      is_valid = emit_input_valid(oc.probe_col_idx, row_i);
      if (!elem_ty) {
        src_byte_ptr = cc.b.CreateGEP(
            i8, cc.col_data[oc.probe_col_idx],
            cc.b.CreateMul(row_i, ConstantInt::get(i64, oc.elem_size)));
      }
    } else {
      // PAYLOAD: row offset = data_offsets[num_keys + payload_row_indices[k]]
      int k = oc.payload_col_idx;
      int row_col_idx =
          (k >= 0 && k < (int)payload_row_indices.size())
              ? payload_row_indices[k]
              : k;
      is_valid = emit_row_valid(chain_ptr, (int)num_keys + row_col_idx);
      src_byte_ptr =
          cc.b.CreateGEP(i8, chain_ptr, get_hoisted((int)num_keys + row_col_idx));
    }

    if (elem_ty) {
      Value *val;
      if (oc.source == OutColDesc::PROBE) {
        Type *ptr_ty = PointerType::getUnqual(elem_ty);
        Value *src_typed =
            cc.b.CreateBitCast(cc.col_data[oc.probe_col_idx], ptr_ty);
        val =
            cc.b.CreateLoad(elem_ty, cc.b.CreateGEP(elem_ty, src_typed, row_i));
      } else {
        val = cc.b.CreateLoad(
            elem_ty,
            cc.b.CreateBitCast(src_byte_ptr, PointerType::getUnqual(elem_ty)));
      }
      // NULL source: store zero instead of the garbage slot bytes
      val = cc.b.CreateSelect(is_valid, val, Constant::getNullValue(elem_ty));
      Type *ptr_ty = PointerType::getUnqual(elem_ty);
      Value *dst_typed = cc.b.CreateBitCast(out_data_ptrs[oi], ptr_ty);
      cc.b.CreateStore(val, cc.b.CreateGEP(elem_ty, dst_typed, chain_oc));
    } else {
      // NULL source: copy zero bytes (a garbage string_t must never be
      // visible — later Flatten/heap ops may dereference it)
      Value *zero_src = cc.b.CreateBitCast(zero_pad_gv, i8p);
      Value *src = cc.b.CreateSelect(is_valid, src_byte_ptr, zero_src);
      Value *dst = cc.b.CreateGEP(
          i8, out_data_ptrs[oi],
          cc.b.CreateMul(chain_oc, ConstantInt::get(i64, oc.elem_size)));
      cc.b.CreateMemCpy(dst, MaybeAlign(1), src, MaybeAlign(1),
                        ConstantInt::get(i64, oc.elem_size));
    }
    emit_out_validity(oi, chain_oc, is_valid);
  }
  Value *chain_oc_next = cc.b.CreateAdd(chain_oc, ConstantInt::get(i64, 1));
  cc.b.CreateBr(advance_bb);

  // ---- Advance: skip chain walk when no duplicate keys in HT ----
  cc.b.SetInsertPoint(advance_bb);
  BasicBlock *chain_walk_bb = BasicBlock::Create(*ctx, "chain_walk", fn);
  cc.b.CreateCondBr(skip_chain_walk, chain_done_bb, chain_walk_bb);

  cc.b.SetInsertPoint(chain_walk_bb);
  Value *next_ptr_addr = cc.b.CreateGEP(i8, chain_ptr, v_ptr_off);
  Value *next_ptr_pp =
      cc.b.CreateBitCast(next_ptr_addr, PointerType::getUnqual(i8p));
  Value *next_ptr = cc.b.CreateLoad(i8p, next_ptr_pp, "next_ptr");
  Value *next_is_null = cc.b.CreateICmpEQ(
      cc.b.CreatePtrToInt(next_ptr, i64), ConstantInt::get(i64, 0));
  cc.b.CreateCondBr(next_is_null, chain_done_bb, chain_bb);
  chain_ptr->addIncoming(next_ptr, chain_walk_bb);
  chain_oc->addIncoming(chain_oc_next, chain_walk_bb);

  cc.b.SetInsertPoint(chain_done_bb);
  cc.b.CreateBr(next_bb);

  // ---- Bail: output would overflow 2048 rows; tell caller to fall back ----
  // Return -1; the caller (PhysicalHashJoin::ExecuteInternal) treats this as
  // "fall through to the interpreter." Partial writes to chunk.data[*] from
  // this run remain in the output buffers, but DuckDB's ScanStructure::Next
  // overwrites them and calls SetCardinality with the *new* row count. The
  // interpreter never reads past cardinality, so the stale tail bytes are
  // inert. Avoiding a chunk.Reset() here keeps the common (non-bail) path
  // cheap; bail itself is rare (only when a single probe chunk yields >2048
  // matches via multi-match chain walk).
  cc.b.SetInsertPoint(bail_bb);
  cc.b.CreateRet(ConstantInt::get(i64, -1));

  // ---- Next: advance row_i, loop ----
  cc.b.SetInsertPoint(next_bb);
  PHINode *oc_next = cc.b.CreatePHI(i64, 4, "oc_next");
  oc_next->addIncoming(out_count, after_filter_bb);  // filter failed
  oc_next->addIncoming(out_count, probe_bb);         // empty entry
  oc_next->addIncoming(chain_oc_next, chain_done_bb); // chain exhausted
  oc_next->addIncoming(out_count, bf_miss_bb);        // bloom filter miss
  Value *i_next = cc.b.CreateAdd(row_i, ConstantInt::get(i64, 1));
  row_i->addIncoming(i_next, next_bb);
  out_count->addIncoming(oc_next, next_bb);
  cc.b.CreateBr(outer_bb);

  // ---- Exit ----
  cc.b.SetInsertPoint(exit_bb);
  cc.b.CreateStore(out_count,
                   cc.b.CreateStructGEP(ChunkViewTy, out_arg, 1));
  cc.b.CreateRet(out_count);

  SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

  std::string err;
  raw_string_ostream es(err);
  if (verifyFunction(*fn, &es)) {
    return nullptr;
  }

  OptimiseModule(*mod, skip_opt_);

  impl_->pending_cache_key = cache_key;
  auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
  if (auto e = impl_->jit->addIRModule(impl_->current_tracker, std::move(tsm))) {
    impl_->pending_cache_key.clear();
    logAllUnhandledErrors(std::move(e), errs());
    return nullptr;
  }

  auto sym = impl_->jit->lookup(fn_name);
  impl_->pending_cache_key.clear();
  if (!sym) {
    logAllUnhandledErrors(sym.takeError(), errs());
    return nullptr;
  }

#ifndef NDEBUG
  std::cerr << "[AQP-JIT] compiled filter+probe+proj fusion (direct-HT) fn="
            << fn_name << "  filter_exprs=" << filter_exprs.size()
            << "  num_keys=" << probe_key_cols.size()
            << "  out_cols=" << out_cols.size()
            << "  payload_width=" << payload_width << "\n";
#endif

  if (template_mode && params_out) {
    tmpl_pb.Finalize();
    *params_out = std::move(tmpl_pb.buf);
  }

  return AQP_JIT_GET_ADDR(sym);
}

// ---------------------------------------------------------------------------
// CompileMultiProbeChain — fuse N consecutive hash-join probes into one
// LLVM function. stages[0] = innermost (receives scan input), stages[N-1]
// = outermost. Output columns match the outermost HJ's output layout.
// ---------------------------------------------------------------------------
void *IrToLlvmCompiler::CompileMultiProbeChain(
    const std::vector<ProbeStageInfo> &stages,
    std::vector<uint8_t> *params_out) {

  const size_t N = stages.size();
  if (N < 2) return nullptr;

  const bool template_mode = cache_mode_ == 2;

  // --- Per-stage key descriptors ---
  struct StageKeyCol {
    int col_idx;       // stage 0: index into probe_schema. stage k>0: index into stage k-1 output
    int32_t dtype;
    unsigned elem_size;
  };
  struct StageDesc {
    std::vector<StageKeyCol> keys;
    bool do_skip_salt;
    unsigned num_keys;
  };
  std::vector<StageDesc> stage_descs(N);

  for (size_t s = 0; s < N; s++) {
    auto &sd = stage_descs[s];
    auto &st = stages[s];
    bool all_int = true;
    for (size_t j = 0; j < st.lhs_key_chunk_idxs.size(); j++) {
      StageKeyCol kc;
      kc.col_idx = st.lhs_key_chunk_idxs[j];
      kc.dtype = st.lhs_key_dtypes[j];
      kc.elem_size = DtypeElemSize(kc.dtype);
      if (kc.elem_size == 0) return nullptr;
      if (kc.dtype != AQP_DTYPE_INT8 && kc.dtype != AQP_DTYPE_INT16 &&
          kc.dtype != AQP_DTYPE_INT32 && kc.dtype != AQP_DTYPE_INT64)
        all_int = false;
      sd.keys.push_back(kc);
    }
    sd.do_skip_salt = (skip_hash_cmp_ == 2 && all_int);
    sd.num_keys = (unsigned)sd.keys.size();
  }

  // --- Resolve final output columns ---
  // FinalOutCol tells the emit block where to load each output value from.
  struct FinalOutCol {
    enum Source { PROBE_INPUT, HT_STAGE };
    Source source;
    int stage_idx;    // HT stage index (0-based), unused for PROBE_INPUT
    int col_idx;      // probe_schema idx (PROBE_INPUT) or HT layout idx (HT_STAGE)
    int32_t dtype;
    unsigned elem_size;
  };
  std::vector<FinalOutCol> final_out;

  // For a 2-probe chain: stages[0]=inner, stages[1]=outer
  // stages[1].lhs_output_idxs index into stages[0]'s output = [lhs0..., rhs0...]
  // stages[1].rhs_output_layout_idxs index into HT₂ layout = [keys1, payload1]
  //
  // For N-probe: trace lhs outputs recursively through the chain.

  // Helper: resolve an index in stage k's input to a FinalOutCol
  // stage k's input = stage k-1's output = [lhs_{k-1} cols..., rhs_{k-1} cols...]
  struct ColOrigin {
    FinalOutCol::Source source;
    int stage_idx;
    int col_idx;
    int32_t dtype;
    unsigned elem_size;
  };
  // Build stage output layouts: each stage's output is [lhs_output, rhs_output]
  // stage_output_origins[s] maps output column index → ColOrigin
  std::vector<std::vector<ColOrigin>> stage_output_origins(N);

  // Stage 0: lhs comes from probe input, rhs from HT₀
  {
    auto &st = stages[0];
    for (size_t i = 0; i < st.lhs_output_idxs.size(); i++) {
      ColOrigin co;
      co.source = FinalOutCol::PROBE_INPUT;
      co.stage_idx = -1;
      co.col_idx = st.lhs_output_idxs[i];
      co.dtype = (i < st.lhs_output_dtypes.size()) ? st.lhs_output_dtypes[i]
                     : st.probe_schema[st.lhs_output_idxs[i]].dtype;
      co.elem_size = DtypeElemSize(co.dtype);
      stage_output_origins[0].push_back(co);
    }
    for (size_t i = 0; i < st.rhs_output_layout_idxs.size(); i++) {
      ColOrigin co;
      co.source = FinalOutCol::HT_STAGE;
      co.stage_idx = 0;
      co.col_idx = st.rhs_output_layout_idxs[i];
      co.dtype = (i < st.rhs_output_dtypes.size()) ? st.rhs_output_dtypes[i]
                     : AQP_DTYPE_OTHER;
      co.elem_size = DtypeElemSize(co.dtype);
      stage_output_origins[0].push_back(co);
    }
  }

  // Stages 1..N-1: lhs indices refer to previous stage's output
  for (size_t s = 1; s < N; s++) {
    auto &st = stages[s];
    auto &prev_out = stage_output_origins[s - 1];
    for (size_t i = 0; i < st.lhs_output_idxs.size(); i++) {
      int idx = st.lhs_output_idxs[i];
      if (idx < 0 || idx >= (int)prev_out.size()) return nullptr;
      ColOrigin co = prev_out[idx];
      if (i < st.lhs_output_dtypes.size())
        co.dtype = st.lhs_output_dtypes[i];
      co.elem_size = DtypeElemSize(co.dtype);
      stage_output_origins[s].push_back(co);
    }
    for (size_t i = 0; i < st.rhs_output_layout_idxs.size(); i++) {
      ColOrigin co;
      co.source = FinalOutCol::HT_STAGE;
      co.stage_idx = (int)s;
      co.col_idx = st.rhs_output_layout_idxs[i];
      co.dtype = (i < st.rhs_output_dtypes.size()) ? st.rhs_output_dtypes[i]
                     : AQP_DTYPE_OTHER;
      co.elem_size = DtypeElemSize(co.dtype);
      stage_output_origins[s].push_back(co);
    }
  }

  // Final output = outermost stage's output
  auto &final_origins = stage_output_origins[N - 1];
  for (auto &co : final_origins) {
    FinalOutCol fc;
    fc.source = co.source;
    fc.stage_idx = co.stage_idx;
    fc.col_idx = co.col_idx;
    fc.dtype = co.dtype;
    fc.elem_size = co.elem_size;
    if (fc.elem_size == 0) return nullptr;
    final_out.push_back(fc);
  }
  if (final_out.empty()) return nullptr;

  // --- Resolve stage k>0 key sources ---
  // For stage k, each key's BoundRef index points into stage k-1's output.
  // We resolve to ColOrigin to know where to load from in LLVM.
  struct StageKeySource {
    ColOrigin origin;
  };
  std::vector<std::vector<StageKeySource>> stage_key_sources(N);
  // Stage 0 keys: from probe input directly
  for (auto &kc : stage_descs[0].keys) {
    StageKeySource sks;
    sks.origin.source = FinalOutCol::PROBE_INPUT;
    sks.origin.stage_idx = -1;
    sks.origin.col_idx = kc.col_idx;
    sks.origin.dtype = kc.dtype;
    sks.origin.elem_size = kc.elem_size;
    stage_key_sources[0].push_back(sks);
  }
  for (size_t s = 1; s < N; s++) {
    auto &prev_out = stage_output_origins[s - 1];
    for (auto &kc : stage_descs[s].keys) {
      int idx = kc.col_idx; // index into prev stage output
      if (idx < 0 || idx >= (int)prev_out.size()) return nullptr;
      StageKeySource sks;
      sks.origin = prev_out[idx];
      stage_key_sources[s].push_back(sks);
    }
  }

  // --- LLVM IR generation ---
  uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
  std::string fn_name = "aqp_multi_probe_" + std::to_string(N) + "x_" + std::to_string(fn_id);

  std::string cache_key;
  if (cache_enabled_ && impl_->cache_enabled) {
    std::string opt_tag =
        std::to_string((int)simd_isa_) + "F" +
        std::to_string((int)fast_mode_) + (skip_opt_ ? "n" : "o") +
        "k" + std::to_string(skip_hash_cmp_) + (batch_probe_ ? "b" : "_") +
        (prefetch_ ? "p" : "_") + std::to_string(prefetch_distance_) + "." +
        std::to_string(prefetch_entry_distance_) + "." +
        std::to_string(prefetch_row_distance_) +
        "C" + std::to_string(cache_mode_);
    std::string stages_serial = template_mode
        ? SerializeProbeStagesTemplate(stages)
        : SerializeProbeStages(stages);
    cache_key = Impl::ComputeCacheKey("mprobe:" + opt_tag + "||" +
                                      stages_serial);
    fn_name = "aqp_mpc_c" + cache_key.substr(0, 12);
    void *cached = impl_->TryCacheLoad(cache_key, fn_name);
    if (cached) {
      if (template_mode && params_out)
        *params_out = BuildParamsFromProbeStages(stages);
      return cached;
    }
  }

  auto ctx = std::make_unique<LLVMContext>();
  auto mod = std::make_unique<Module>("aqp_multi_probe_mod", *ctx);

  Type *i8 = Type::getInt8Ty(*ctx);
  Type *i8p = PointerType::getUnqual(i8);
  Type *i32 = Type::getInt32Ty(*ctx);
  Type *i64 = Type::getInt64Ty(*ctx);
  Type *i64p = PointerType::getUnqual(i64);
  Type *i16 = Type::getInt16Ty(*ctx);

  StructType *ColViewTy = StructType::get(*ctx, {i8p, i64p, i32, i32});
  StructType *ChunkViewTy =
      StructType::get(*ctx, {PointerType::getUnqual(ColViewTy), i64, i64});
  // AQPJoinHTView: { entries, bitmask, use_salt, layout_ptr, tuple_size,
  //   pointer_offset, data_offsets, no_chains, bf_data, bf_bitmask,
  //   has_row_validity }
  StructType *ViewTy = StructType::get(
      *ctx, {i8p, i64, i64, i8p, i32, i32, i64p, i64, i64p, i64, i64});

  // AQPMultiProbeState: { AQPJoinHTView*[4], uint32_t num_stages }
  StructType *MPSTy = StructType::get(
      *ctx, {ArrayType::get(PointerType::getUnqual(ViewTy), 4), i32});

  // int64_t fn(AQPChunkView *in, AQPChunkView *out, void *state)
  FunctionType *fn_ty = FunctionType::get(
      i64,
      {PointerType::getUnqual(ChunkViewTy),
       PointerType::getUnqual(ChunkViewTy), i8p},
      false);
  Function *fn =
      Function::Create(fn_ty, Function::ExternalLinkage, fn_name, mod.get());

  Value *in_arg = fn->getArg(0);  in_arg->setName("in");
  Value *out_arg = fn->getArg(1); out_arg->setName("out");
  Value *state_arg = fn->getArg(2); state_arg->setName("state");

  BasicBlock *entry_bb = BasicBlock::Create(*ctx, "entry", fn);
  BasicBlock *outer_bb = BasicBlock::Create(*ctx, "outer", fn);
  BasicBlock *next_bb = BasicBlock::Create(*ctx, "next", fn);
  BasicBlock *bail_bb = BasicBlock::Create(*ctx, "bail", fn);
  BasicBlock *exit_bb = BasicBlock::Create(*ctx, "exit", fn);

  ParamsBuilder tmpl_pb;

  auto &probe_schema = stages[0].probe_schema;
  Value *dummy_sel = ConstantPointerNull::get(
      PointerType::getUnqual(
          StructType::get(*ctx, {PointerType::getUnqual(i32), i32})));
  CompileCtx cc(*ctx, *mod, probe_schema, in_arg, dummy_sel);
  cc.b.SetInsertPoint(entry_bb);

  if (template_mode) {
    cc.template_mode = true;
    cc.params_builder = &tmpl_pb;
    cc.LoadParamsFromThreadLocal();
  }

  // --- Load AQPMultiProbeState, extract per-stage HT views ---
  Value *mps_ptr = cc.b.CreateBitCast(state_arg, PointerType::getUnqual(MPSTy), "mps");
  Value *views_arr = cc.b.CreateStructGEP(MPSTy, mps_ptr, 0, "views_arr");

  struct HTViewVals {
    Value *entries;
    Value *bitmask;
    Value *ptr_off;
    Value *offsets;
    Value *no_chains;
    Value *skip_chain;
    Value *has_rv; // i1: rows carry a validity byte prefix
  };
  std::vector<HTViewVals> ht_vals(N);

  for (size_t s = 0; s < N; s++) {
    Value *view_pp = cc.b.CreateGEP(
        ArrayType::get(PointerType::getUnqual(ViewTy), 4), views_arr,
        {ConstantInt::get(i64, 0), ConstantInt::get(i64, s)});
    Value *view_ptr = cc.b.CreateLoad(
        PointerType::getUnqual(ViewTy), view_pp,
        "view_" + std::to_string(s));

    auto &hv = ht_vals[s];
    std::string sfx = "_" + std::to_string(s);
    hv.entries = cc.b.CreateLoad(i8p,
        cc.b.CreateStructGEP(ViewTy, view_ptr, 0), "entries" + sfx);
    hv.bitmask = cc.b.CreateLoad(i64,
        cc.b.CreateStructGEP(ViewTy, view_ptr, 1), "bitmask" + sfx);
    Value *ptr_off32 = cc.b.CreateLoad(i32,
        cc.b.CreateStructGEP(ViewTy, view_ptr, 5), "ptr_off32" + sfx);
    hv.ptr_off = cc.b.CreateZExt(ptr_off32, i64, "ptr_off" + sfx);
    hv.offsets = cc.b.CreateLoad(i64p,
        cc.b.CreateStructGEP(ViewTy, view_ptr, 6), "offsets" + sfx);
    hv.no_chains = cc.b.CreateLoad(i64,
        cc.b.CreateStructGEP(ViewTy, view_ptr, 7), "no_chains" + sfx);
    hv.skip_chain = cc.b.CreateICmpNE(
        hv.no_chains, ConstantInt::get(i64, 0), "skip_chain" + sfx);
    Value *has_rv64 = cc.b.CreateLoad(i64,
        cc.b.CreateStructGEP(ViewTy, view_ptr, 10), "has_rv64" + sfx);
    hv.has_rv = cc.b.CreateICmpNE(
        has_rv64, ConstantInt::get(i64, 0), "has_rv" + sfx);
  }

  // --- Hoist data_offsets for all stages ---
  // Per-stage maps from layout_idx → hoisted Value*
  std::vector<std::unordered_map<int, Value *>> hoisted_per_stage(N);
  {
    for (size_t s = 0; s < N; s++) {
      auto load_off = [&](int idx) {
        if (hoisted_per_stage[s].count(idx)) return;
        Value *off = cc.b.CreateLoad(
            i64,
            cc.b.CreateGEP(i64, ht_vals[s].offsets, ConstantInt::get(i64, (int64_t)idx)),
            "off_s" + std::to_string(s) + "_" + std::to_string(idx));
        hoisted_per_stage[s][idx] = off;
      };
      // Key offsets
      for (unsigned j = 0; j < stage_descs[s].num_keys; j++) load_off(j);
      // Output column offsets (only for this stage's HT)
      int nk = (int)stage_descs[s].num_keys;
      for (auto &fc : final_out) {
        if (fc.source == FinalOutCol::HT_STAGE && fc.stage_idx == (int)s) {
          int layout_idx = fc.col_idx;
          if (layout_idx < nk) {
            load_off(layout_idx);
          } else {
            int pi = layout_idx - nk;
            int row_idx = (pi >= 0 && pi < (int)stages[s].payload_row_indices.size())
                              ? stages[s].payload_row_indices[pi] : pi;
            load_off(nk + row_idx);
          }
        }
      }
      // Key source offsets for stages that load from this HT
      for (size_t s2 = 0; s2 < N; s2++) {
        for (auto &sks : stage_key_sources[s2]) {
          if (sks.origin.source == FinalOutCol::HT_STAGE && sks.origin.stage_idx == (int)s) {
            int layout_idx = sks.origin.col_idx;
            if (layout_idx < nk) {
              load_off(layout_idx);
            } else {
              int pi = layout_idx - nk;
              int row_idx = (pi >= 0 && pi < (int)stages[s].payload_row_indices.size())
                                ? stages[s].payload_row_indices[pi] : pi;
              load_off(nk + row_idx);
            }
          }
        }
      }
    }
  }
  auto get_stage_offset = [&](int stage, int idx) -> Value * {
    auto it = hoisted_per_stage[stage].find(idx);
    assert(it != hoisted_per_stage[stage].end() && "hoisted offset miss");
    return it->second;
  };

  Constant *POINTER_MASK = ConstantInt::get(i64, 0x0000FFFFFFFFFFFFULL);

  Value *nrows = cc.b.CreateLoad(
      i64, cc.b.CreateStructGEP(ChunkViewTy, in_arg, 1), "nrows");

  // Load probe input column data
  cc.col_data.resize(probe_schema.size());
  cc.col_validity.resize(probe_schema.size());
  for (size_t i = 0; i < probe_schema.size(); i++) {
    cc.col_data[i] = cc.LoadColData((unsigned)i);
    cc.col_validity[i] = cc.LoadColValidity((unsigned)i);
  }

  // Load output column data + validity pointers
  Value *out_cols_pp = cc.b.CreateStructGEP(ChunkViewTy, out_arg, 0);
  Value *out_cols_v = cc.b.CreateLoad(
      PointerType::getUnqual(ColViewTy), out_cols_pp, "out_cols");
  std::vector<Value *> out_data_ptrs;
  std::vector<Value *> out_valid_ptrs;
  for (size_t oi = 0; oi < final_out.size(); oi++) {
    Value *col_i = cc.b.CreateGEP(ColViewTy, out_cols_v, ConstantInt::get(i64, oi));
    out_data_ptrs.push_back(
        cc.b.CreateLoad(i8p, cc.b.CreateStructGEP(ColViewTy, col_i, 0),
                        "out_data_" + std::to_string(oi)));
    out_valid_ptrs.push_back(
        cc.b.CreateLoad(i64p, cc.b.CreateStructGEP(ColViewTy, col_i, 1),
                        "out_valid_" + std::to_string(oi)));
  }

  // ---- NULL-validity helpers (see CompileFilterProbeProjectFusion) ----
  GlobalVariable *all_valid_gv = new GlobalVariable(
      *mod, i64, /*isConstant=*/true, GlobalValue::PrivateLinkage,
      ConstantInt::get(i64, -1), "aqp_all_valid");
  ArrayType *zero_pad_ty = ArrayType::get(i8, 64);
  GlobalVariable *zero_pad_gv = new GlobalVariable(
      *mod, zero_pad_ty, /*isConstant=*/true, GlobalValue::PrivateLinkage,
      ConstantAggregateZero::get(zero_pad_ty), "aqp_zero_pad");

  auto emit_input_valid = [&](int col_idx, Value *r) -> Value * {
    Value *vp = cc.col_validity[col_idx];
    Value *has_mask = cc.b.CreateICmpNE(
        cc.b.CreatePtrToInt(vp, i64), ConstantInt::get(i64, 0));
    Value *word_idx = cc.b.CreateLShr(r, ConstantInt::get(i64, 6));
    Value *word_addr = cc.b.CreateGEP(i64, vp, word_idx);
    Value *addr = cc.b.CreateSelect(has_mask, word_addr, all_valid_gv);
    Value *word = cc.b.CreateLoad(i64, addr);
    Value *bitpos = cc.b.CreateAnd(r, ConstantInt::get(i64, 63));
    Value *bit = cc.b.CreateAnd(cc.b.CreateLShr(word, bitpos),
                                ConstantInt::get(i64, 1));
    return cc.b.CreateICmpNE(bit, ConstantInt::get(i64, 0), "in_valid");
  };
  auto emit_row_valid = [&](Value *row_ptr, int col_bit_idx,
                            Value *stage_has_rv) -> Value * {
    Value *byte_addr = cc.b.CreateGEP(
        i8, row_ptr, ConstantInt::get(i64, col_bit_idx / 8));
    Value *byte = cc.b.CreateLoad(i8, byte_addr);
    Value *bit = cc.b.CreateAnd(
        cc.b.CreateLShr(byte, ConstantInt::get(i8, col_bit_idx % 8)),
        ConstantInt::get(i8, 1));
    Value *raw = cc.b.CreateICmpNE(bit, ConstantInt::get(i8, 0));
    return cc.b.CreateSelect(stage_has_rv, raw, ConstantInt::getTrue(*ctx),
                             "row_valid");
  };
  auto emit_out_validity = [&](size_t oi, Value *out_idx, Value *is_valid) {
    Value *word_idx = cc.b.CreateLShr(out_idx, ConstantInt::get(i64, 6));
    Value *wp = cc.b.CreateGEP(i64, out_valid_ptrs[oi], word_idx);
    Value *word = cc.b.CreateLoad(i64, wp);
    Value *bitpos = cc.b.CreateAnd(out_idx, ConstantInt::get(i64, 63));
    Value *inval = cc.b.CreateZExt(cc.b.CreateNot(is_valid), i64);
    Value *clear_mask = cc.b.CreateShl(inval, bitpos);
    cc.b.CreateStore(cc.b.CreateAnd(word, cc.b.CreateNot(clear_mask)), wp);
  };

  // Map an HT-stage layout reference (as used by key sources and final
  // output columns) to the index into data_offsets — also the validity
  // bit index within the row's validity prefix.
  auto layout_off_idx = [&](int src_stage, int layout_idx) -> int {
    int nk_src = (int)stage_descs[src_stage].num_keys;
    if (layout_idx < nk_src) return layout_idx;
    int pi = layout_idx - nk_src;
    int row_idx = (pi >= 0 && pi < (int)stages[src_stage].payload_row_indices.size())
                      ? stages[src_stage].payload_row_indices[pi] : pi;
    return nk_src + row_idx;
  };

  // --- Hash helpers ---
  auto key_llvm_ty = [&](int32_t dt) -> Type * {
    if (dt == AQP_DTYPE_INT32 || dt == AQP_DTYPE_DATE) return i32;
    if (dt == AQP_DTYPE_INT64) return i64;
    if (dt == AQP_DTYPE_INT16) return i16;
    if (dt == AQP_DTYPE_BOOL || dt == AQP_DTYPE_INT8) return i8;
    return nullptr;
  };
  // Validate all key types are supported
  for (size_t s = 0; s < N; s++)
    for (auto &kc : stage_descs[s].keys)
      if (!key_llvm_ty(kc.dtype)) return nullptr;

  Constant *MURMUR_MUL = ConstantInt::get(i64, 0xd6e8feb86659fd93ULL);
  Constant *SHIFT32 = ConstantInt::get(i64, 32);
  auto emitMurmur = [&](Value *x) -> Value * {
    Value *t = cc.b.CreateXor(x, cc.b.CreateLShr(x, SHIFT32));
    t = cc.b.CreateMul(t, MURMUR_MUL);
    t = cc.b.CreateXor(t, cc.b.CreateLShr(t, SHIFT32));
    t = cc.b.CreateMul(t, MURMUR_MUL);
    t = cc.b.CreateXor(t, cc.b.CreateLShr(t, SHIFT32));
    return t;
  };
  auto emitCombine = [&](Value *a, Value *b) -> Value * {
    Value *t = cc.b.CreateXor(a, cc.b.CreateLShr(a, SHIFT32));
    t = cc.b.CreateMul(t, MURMUR_MUL);
    return cc.b.CreateXor(t, b);
  };

  cc.b.CreateBr(outer_bb);

  // ---- Outer loop header ----
  cc.b.SetInsertPoint(outer_bb);
  PHINode *row_i = cc.b.CreatePHI(i64, 2, "i");
  PHINode *out_count = cc.b.CreatePHI(i64, 2, "out_count");
  row_i->addIncoming(ConstantInt::get(i64, 0), entry_bb);
  out_count->addIncoming(ConstantInt::get(i64, 0), entry_bb);

  BasicBlock *body_bb = BasicBlock::Create(*ctx, "body", fn);
  cc.b.CreateCondBr(cc.b.CreateICmpEQ(row_i, nrows), exit_bb, body_bb);

  // ---- Body: evaluate filter for stage 0 (if any) ----
  cc.b.SetInsertPoint(body_bb);
  cc.row_idx = row_i;

  // Stage 0 filter
  std::vector<const AQPExpr *> filter_exprs;
  if (stages[0].filter_ir) {
    for (const auto &qe : stages[0].filter_ir->qual_vec)
      filter_exprs.push_back(qe.get());
  }
  if (!template_mode) SortFiltersByCost(filter_exprs);

  BasicBlock *hash_0_bb = BasicBlock::Create(*ctx, "hash_0", fn);
  BasicBlock *after_filter_bb;
  if (!filter_exprs.empty()) {
    BasicBlock *filt_fail = BasicBlock::Create(*ctx, "filt_fail", fn);
    EmitShortCircuitFilter(cc, fn, filter_exprs, hash_0_bb, filt_fail);
    cc.b.SetInsertPoint(filt_fail);
    cc.b.CreateBr(next_bb);
    after_filter_bb = filt_fail;
  } else {
    cc.b.CreateBr(hash_0_bb);
    after_filter_bb = body_bb;
  }

  // ================================================================
  // Generate nested probe stages.
  //
  // For each stage s=0..N-1 we generate:
  //   hash_s  → probe_s → [salt_s] → key_eq_s → chain_s → emit/next_stage
  //   miss_s  (linear probe)
  //   advance_s (chain walk)
  //   chain_done_s
  //
  // Stage s+1 is nested inside chain_s of stage s. The innermost stage
  // (s=N-1) emits output columns. After the innermost chain exhausts,
  // control returns to the outer stage's advance block.
  //
  // SSA values for chain_ptr_s are available as PHI nodes in chain_s_bb.
  // These are used to load HT keys and output columns.
  // ================================================================

  // Per-stage SSA values we need to thread through
  struct StageBBs {
    BasicBlock *hash_bb;
    BasicBlock *probe_bb;
    BasicBlock *salt_ok_bb; // nullptr if skip_salt
    BasicBlock *key_eq_bb;
    BasicBlock *miss_bb;
    BasicBlock *chain_bb;
    BasicBlock *emit_or_next_bb; // emit for last stage, next_stage hash for non-last
    BasicBlock *advance_bb;
    BasicBlock *chain_walk_bb;
    BasicBlock *chain_done_bb;
    PHINode *ht_off;
    PHINode *chain_ptr;
    PHINode *chain_oc;   // running output count in this stage's chain walk
    Value *row_ptr_init; // initial row pointer from entry probe
    std::vector<Value *> probe_key_vals; // loaded key values for this stage
    Value *hash_val;
    Value *probe_salt;
  };
  std::vector<StageBBs> sbb(N);

  // Create all basic blocks upfront so forward references work
  for (size_t s = 0; s < N; s++) {
    std::string sfx = "_" + std::to_string(s);
    auto &sb = sbb[s];
    if (s == 0) {
      sb.hash_bb = hash_0_bb;
    } else {
      sb.hash_bb = BasicBlock::Create(*ctx, "hash" + sfx, fn);
    }
    sb.probe_bb = BasicBlock::Create(*ctx, "probe" + sfx, fn);
    sb.salt_ok_bb = stage_descs[s].do_skip_salt ? nullptr
                        : BasicBlock::Create(*ctx, "salt_ok" + sfx, fn);
    sb.key_eq_bb = BasicBlock::Create(*ctx, "key_eq" + sfx, fn);
    sb.miss_bb = BasicBlock::Create(*ctx, "miss" + sfx, fn);
    sb.chain_bb = BasicBlock::Create(*ctx, "chain" + sfx, fn);
    sb.advance_bb = BasicBlock::Create(*ctx, "advance" + sfx, fn);
    sb.chain_walk_bb = BasicBlock::Create(*ctx, "chain_walk" + sfx, fn);
    sb.chain_done_bb = BasicBlock::Create(*ctx, "chain_done" + sfx, fn);
  }

  for (size_t s = 0; s < N; s++) {
    std::string sfx = "_" + std::to_string(s);
    auto &sd = stage_descs[s];
    auto &sb = sbb[s];
    auto &hv = ht_vals[s];

    // ---- Hash: load keys + compute hash ----
    cc.b.SetInsertPoint(sb.hash_bb);
    sb.probe_key_vals.clear();
    sb.hash_val = nullptr;

    for (size_t j = 0; j < sd.keys.size(); j++) {
      auto &kc = sd.keys[j];
      auto &ks = stage_key_sources[s][j];
      Type *kty = key_llvm_ty(kc.dtype);
      Value *kval = nullptr;

      if (ks.origin.source == FinalOutCol::PROBE_INPUT) {
        // Load from input chunk
        Value *src = cc.col_data[ks.origin.col_idx];
        Value *elem_ptr = cc.b.CreateGEP(
            i8, src,
            cc.b.CreateMul(row_i, ConstantInt::get(i64, kc.elem_size)));
        Value *typed_ptr = cc.b.CreateBitCast(elem_ptr, PointerType::getUnqual(kty));
        kval = cc.b.CreateLoad(kty, typed_ptr, "key" + sfx + "_" + std::to_string(j));
      } else {
        // Load from a previous stage's chain_ptr
        int src_stage = ks.origin.stage_idx;
        int layout_idx = ks.origin.col_idx;
        int nk_src = (int)stage_descs[src_stage].num_keys;
        int off_idx;
        if (layout_idx < nk_src) {
          off_idx = layout_idx;
        } else {
          int pi = layout_idx - nk_src;
          int row_idx = (pi >= 0 && pi < (int)stages[src_stage].payload_row_indices.size())
                            ? stages[src_stage].payload_row_indices[pi] : pi;
          off_idx = nk_src + row_idx;
        }
        Value *offset_val = get_stage_offset(src_stage, off_idx);
        Value *ptr = cc.b.CreateGEP(i8, sbb[src_stage].chain_ptr, offset_val);
        Value *typed_ptr = cc.b.CreateBitCast(ptr, PointerType::getUnqual(kty));
        kval = cc.b.CreateLoad(kty, typed_ptr, "key" + sfx + "_" + std::to_string(j));
      }

      sb.probe_key_vals.push_back(kval);
      // Match DuckDB's Hash<T>: int8/int16 are first cast to uint32_t
      // (sign-extending), then zero-extended to 64 bits. int32 zero-extends.
      Value *u64_val;
      if (kty == i64) {
        u64_val = kval;
      } else if (kty == cc.i32()) {
        u64_val = cc.b.CreateZExt(kval, i64);
      } else {
        u64_val = cc.b.CreateZExt(cc.b.CreateSExt(kval, cc.i32()), i64);
      }
      Value *hj = emitMurmur(u64_val);
      sb.hash_val = (j == 0) ? hj : emitCombine(sb.hash_val, hj);
    }

    Value *ht_off_init = cc.b.CreateAnd(sb.hash_val, hv.bitmask, "ht_off_init" + sfx);
    sb.probe_salt = sd.do_skip_salt
        ? nullptr
        : cc.b.CreateOr(sb.hash_val, POINTER_MASK, "probe_salt" + sfx);
    cc.b.CreateBr(sb.probe_bb);

    // ---- Probe: load entry, check empty ----
    cc.b.SetInsertPoint(sb.probe_bb);
    sb.ht_off = cc.b.CreatePHI(i64, 2, "ht_off" + sfx);
    sb.ht_off->addIncoming(ht_off_init, sb.hash_bb);

    Value *entries_typed = cc.b.CreateBitCast(hv.entries, i64p);
    Value *entry_addr = cc.b.CreateGEP(i64, entries_typed, sb.ht_off);
    Value *entry = cc.b.CreateLoad(i64, entry_addr, "entry" + sfx);
    Value *is_empty = cc.b.CreateICmpEQ(entry, ConstantInt::get(i64, 0));

    // On empty: for stage 0 → next_bb (next input row);
    //           for stage s>0 → advance of stage s-1 (try next chain_ptr in outer HT)
    BasicBlock *empty_target = (s == 0) ? next_bb : sbb[s - 1].advance_bb;

    if (sd.do_skip_salt) {
      cc.b.CreateCondBr(is_empty, empty_target, sb.key_eq_bb);
    } else {
      cc.b.CreateCondBr(is_empty, empty_target, sb.salt_ok_bb);

      cc.b.SetInsertPoint(sb.salt_ok_bb);
      Value *entry_salt = cc.b.CreateOr(entry, POINTER_MASK);
      Value *salt_match = cc.b.CreateICmpEQ(entry_salt, sb.probe_salt);
      cc.b.CreateCondBr(salt_match, sb.key_eq_bb, sb.miss_bb);
    }

    // ---- Key compare ----
    cc.b.SetInsertPoint(sb.key_eq_bb);
    sb.row_ptr_init = cc.b.CreateIntToPtr(
        cc.b.CreateAnd(entry, POINTER_MASK), i8p, "row_ptr" + sfx);
    Value *all_eq = ConstantInt::getTrue(*ctx);
    for (size_t j = 0; j < sd.keys.size(); j++) {
      Type *kty = key_llvm_ty(sd.keys[j].dtype);
      Value *koff = get_stage_offset((int)s, (int)j);
      Value *row_key_ptr = cc.b.CreateGEP(i8, sb.row_ptr_init, koff);
      Value *typed_ptr = cc.b.CreateBitCast(row_key_ptr, PointerType::getUnqual(kty));
      Value *rkval = cc.b.CreateLoad(kty, typed_ptr);
      Value *eq = cc.b.CreateICmpEQ(rkval, sb.probe_key_vals[j]);
      all_eq = cc.b.CreateAnd(all_eq, eq);
      // NULL probe-side key never matches (inner join semantics)
      auto &ks = stage_key_sources[s][j];
      Value *key_valid;
      if (ks.origin.source == FinalOutCol::PROBE_INPUT) {
        key_valid = emit_input_valid(ks.origin.col_idx, row_i);
      } else {
        int src_stage = ks.origin.stage_idx;
        key_valid = emit_row_valid(
            sbb[src_stage].chain_ptr,
            layout_off_idx(src_stage, ks.origin.col_idx),
            ht_vals[src_stage].has_rv);
      }
      all_eq = cc.b.CreateAnd(all_eq, key_valid);
    }
    cc.b.CreateCondBr(all_eq, sb.chain_bb, sb.miss_bb);

    // ---- Miss: linear probe ----
    cc.b.SetInsertPoint(sb.miss_bb);
    Value *ht_off_next = cc.b.CreateAnd(
        cc.b.CreateAdd(sb.ht_off, ConstantInt::get(i64, 1)), hv.bitmask);
    sb.ht_off->addIncoming(ht_off_next, sb.miss_bb);
    cc.b.CreateBr(sb.probe_bb);

    // ---- Chain walk header ----
    cc.b.SetInsertPoint(sb.chain_bb);
    sb.chain_ptr = cc.b.CreatePHI(i8p, 2, "chain_ptr" + sfx);
    sb.chain_oc = cc.b.CreatePHI(i64, 2, "chain_oc" + sfx);
    sb.chain_ptr->addIncoming(sb.row_ptr_init, sb.key_eq_bb);

    if (s == 0) {
      sb.chain_oc->addIncoming(out_count, sb.key_eq_bb);
    } else {
      // Inner stages inherit out_count from outer stage's chain_oc
      sb.chain_oc->addIncoming(sbb[s - 1].chain_oc, sb.key_eq_bb);
    }

    if (s == N - 1) {
      // Innermost (last) stage: bail check + emit
      Value *overflow = cc.b.CreateICmpUGE(sb.chain_oc, ConstantInt::get(i64, 2048));
      BasicBlock *emit_bb = BasicBlock::Create(*ctx, "emit", fn);
      cc.b.CreateCondBr(overflow, bail_bb, emit_bb);

      // ---- Emit: write all final output columns ----
      cc.b.SetInsertPoint(emit_bb);
      for (size_t oi = 0; oi < final_out.size(); oi++) {
        auto &fc = final_out[oi];
        Type *elem_ty = nullptr;
        if (fc.dtype == AQP_DTYPE_INT32 || fc.dtype == AQP_DTYPE_DATE)
          elem_ty = i32;
        else if (fc.dtype == AQP_DTYPE_INT64)
          elem_ty = i64;
        else if (fc.dtype == AQP_DTYPE_FLOAT)
          elem_ty = Type::getFloatTy(*ctx);
        else if (fc.dtype == AQP_DTYPE_DOUBLE)
          elem_ty = Type::getDoubleTy(*ctx);
        else if (fc.dtype == AQP_DTYPE_INT16)
          elem_ty = i16;
        else if (fc.dtype == AQP_DTYPE_BOOL || fc.dtype == AQP_DTYPE_INT8)
          elem_ty = i8;

        Value *is_valid = nullptr;
        Value *src_byte_ptr = nullptr;
        if (fc.source == FinalOutCol::PROBE_INPUT) {
          is_valid = emit_input_valid(fc.col_idx, row_i);
          src_byte_ptr = cc.b.CreateGEP(
              i8, cc.col_data[fc.col_idx],
              cc.b.CreateMul(row_i, ConstantInt::get(i64, fc.elem_size)));
        } else {
          // Load from HT stage's chain_ptr at layout offset
          int src_stage = fc.stage_idx;
          int off_idx = layout_off_idx(src_stage, fc.col_idx);
          is_valid = emit_row_valid(sbb[src_stage].chain_ptr, off_idx,
                                    ht_vals[src_stage].has_rv);
          src_byte_ptr = cc.b.CreateGEP(i8, sbb[src_stage].chain_ptr,
                                        get_stage_offset(src_stage, off_idx));
        }

        if (elem_ty) {
          Type *ptr_ty = PointerType::getUnqual(elem_ty);
          Value *val = cc.b.CreateLoad(
              elem_ty, cc.b.CreateBitCast(src_byte_ptr, ptr_ty));
          // NULL source: store zero instead of the garbage slot bytes
          val = cc.b.CreateSelect(is_valid, val,
                                  Constant::getNullValue(elem_ty));
          Value *dst_typed = cc.b.CreateBitCast(out_data_ptrs[oi], ptr_ty);
          cc.b.CreateStore(val, cc.b.CreateGEP(elem_ty, dst_typed, sb.chain_oc));
        } else {
          // NULL source: copy zero bytes (garbage string_t must not escape)
          Value *zero_src = cc.b.CreateBitCast(zero_pad_gv, i8p);
          Value *src = cc.b.CreateSelect(is_valid, src_byte_ptr, zero_src);
          Value *dst = cc.b.CreateGEP(
              i8, out_data_ptrs[oi],
              cc.b.CreateMul(sb.chain_oc, ConstantInt::get(i64, fc.elem_size)));
          cc.b.CreateMemCpy(dst, MaybeAlign(1), src, MaybeAlign(1),
                            ConstantInt::get(i64, fc.elem_size));
        }
        emit_out_validity(oi, sb.chain_oc, is_valid);
      }
      Value *chain_oc_next = cc.b.CreateAdd(sb.chain_oc, ConstantInt::get(i64, 1));
      cc.b.CreateBr(sb.advance_bb);

      // ---- Advance (innermost): chain walk ----
      cc.b.SetInsertPoint(sb.advance_bb);
      cc.b.CreateCondBr(hv.skip_chain, sb.chain_done_bb, sb.chain_walk_bb);

      cc.b.SetInsertPoint(sb.chain_walk_bb);
      Value *next_ptr_addr = cc.b.CreateGEP(i8, sb.chain_ptr, hv.ptr_off);
      Value *next_ptr_pp = cc.b.CreateBitCast(next_ptr_addr, PointerType::getUnqual(i8p));
      Value *next_ptr = cc.b.CreateLoad(i8p, next_ptr_pp, "next_ptr" + sfx);
      Value *next_is_null = cc.b.CreateICmpEQ(
          cc.b.CreatePtrToInt(next_ptr, i64), ConstantInt::get(i64, 0));
      cc.b.CreateCondBr(next_is_null, sb.chain_done_bb, sb.chain_bb);
      sb.chain_ptr->addIncoming(next_ptr, sb.chain_walk_bb);
      sb.chain_oc->addIncoming(chain_oc_next, sb.chain_walk_bb);

      // chain_done: return to outer stage's advance
      cc.b.SetInsertPoint(sb.chain_done_bb);
      // Collect the final oc count at chain_done (from chain_walk or advance skip)
      PHINode *done_oc = cc.b.CreatePHI(i64, 2, "done_oc" + sfx);
      done_oc->addIncoming(chain_oc_next, sb.advance_bb);  // from skip_chain path
      done_oc->addIncoming(chain_oc_next, sb.chain_walk_bb); // chain exhausted (null)

      if (s > 0) {
        // Return to outer stage's advance block with updated out_count
        cc.b.CreateBr(sbb[s - 1].advance_bb);
        // Update outer stage's chain_oc phi from this chain_done
        // (deferred — set below after all stages created)
      } else {
        // Stage 0 chain done → next input row
        cc.b.CreateBr(next_bb);
      }

      // Store done_oc for use by outer stages
      // We'll wire PHI incoming values after the loop

    } else {
      // Non-innermost stage: after chain match, enter next stage
      // No bail check here — bail is checked at innermost emit
      cc.b.CreateBr(sbb[s + 1].hash_bb);

      // ---- Advance (non-innermost): after inner stage returns, chain walk ----
      cc.b.SetInsertPoint(sb.advance_bb);
      cc.b.CreateCondBr(hv.skip_chain, sb.chain_done_bb, sb.chain_walk_bb);

      cc.b.SetInsertPoint(sb.chain_walk_bb);
      Value *next_ptr_addr = cc.b.CreateGEP(i8, sb.chain_ptr, hv.ptr_off);
      Value *next_ptr_pp = cc.b.CreateBitCast(next_ptr_addr, PointerType::getUnqual(i8p));
      Value *next_ptr = cc.b.CreateLoad(i8p, next_ptr_pp, "next_ptr" + sfx);
      Value *next_is_null = cc.b.CreateICmpEQ(
          cc.b.CreatePtrToInt(next_ptr, i64), ConstantInt::get(i64, 0));
      cc.b.CreateCondBr(next_is_null, sb.chain_done_bb, sb.chain_bb);
      sb.chain_ptr->addIncoming(next_ptr, sb.chain_walk_bb);
      // chain_oc for non-innermost: updated by inner stage's chain_done
      // (wired below)

      cc.b.SetInsertPoint(sb.chain_done_bb);
      // done_oc phi merges from: advance_bb (skip_chain) and chain_walk_bb (null next)
      PHINode *done_oc = cc.b.CreatePHI(i64, 2, "done_oc" + sfx);
      // Incoming values set below after all stages exist

      if (s > 0) {
        cc.b.CreateBr(sbb[s - 1].advance_bb);
      } else {
        cc.b.CreateBr(next_bb);
      }
    }
  }

  // --- Wire up cross-stage PHI incoming values ---
  // The tricky part: chain_oc for stage s is updated by stage s+1's chain_done.
  // When s+1's chain exhausts (chain_done_s+1), the accumulated count flows
  // back to stage s's chain walk via chain_oc.
  //
  // For each non-innermost stage s:
  //   chain_oc_s gets incoming from: key_eq_s (initial = outer_oc), chain_walk_s (= done_oc from s+1)
  //   done_oc_s gets incoming from: advance_s (skip_chain), chain_walk_s (chain null)
  //
  // For innermost stage (N-1):
  //   chain_oc gets incoming from key_eq and chain_walk (= chain_oc_next)
  //   done_oc merges chain_oc_next from both paths

  // Retrieve done_oc PHIs for each stage
  std::vector<PHINode *> done_ocs(N);
  for (size_t s = 0; s < N; s++) {
    // done_oc is the first PHI in chain_done_bb
    done_ocs[s] = cast<PHINode>(&sbb[s].chain_done_bb->front());
  }

  // Wire done_oc incoming values for non-innermost stages
  for (size_t s = 0; s < N - 1; s++) {
    // done_oc_s gets the accumulated count from inner stage s+1
    PHINode *done_oc = done_ocs[s];
    Value *inner_done = done_ocs[s + 1]; // result from inner stage

    // From advance_bb (skip_chain path): the inner stage's done_oc was already
    // produced when we returned to advance_bb. But since advance_bb branches to
    // chain_done_bb (skip) or chain_walk_bb, the done_oc should capture the
    // oc from the inner return.
    //
    // Actually for non-innermost stages, advance_bb is entered from the inner
    // stage's chain_done. At that point inner_done_oc has the accumulated count.
    // We need to pass it through.
    //
    // The problem: sbb[s].advance_bb can be reached multiple times (once per
    // inner chain_done). Each time, the accumulated oc is different.
    // We need a PHI at advance_bb to capture the incoming oc.

    // advance_bb is entered from: (1) inner stage's chain_done (normal path),
    // (2) inner stage's probe_bb (empty entry → no match, oc unchanged).
    // The phi captures the accumulated oc from whichever path.
    IRBuilder<> tmp_builder(sbb[s].advance_bb, sbb[s].advance_bb->begin());
    PHINode *adv_oc = tmp_builder.CreatePHI(i64, 2, "adv_oc_" + std::to_string(s));
    adv_oc->addIncoming(inner_done, sbb[s + 1].chain_done_bb);
    adv_oc->addIncoming(sbb[s].chain_oc, sbb[s + 1].probe_bb);

    // done_oc incoming: from advance_bb (skip) = adv_oc, from chain_walk = adv_oc (same value threaded)
    done_oc->addIncoming(adv_oc, sbb[s].advance_bb);
    done_oc->addIncoming(adv_oc, sbb[s].chain_walk_bb);

    // chain_oc for this stage gets updated from chain_walk
    sbb[s].chain_oc->addIncoming(adv_oc, sbb[s].chain_walk_bb);
  }

  // --- Bail block ---
  cc.b.SetInsertPoint(bail_bb);
  cc.b.CreateRet(ConstantInt::get(i64, -1));

  // --- Next: advance row_i ---
  cc.b.SetInsertPoint(next_bb);
  // out_count update: from filter fail, stage 0 empty, or stage 0 chain_done
  PHINode *oc_next = cc.b.CreatePHI(i64, 3, "oc_next");
  if (!filter_exprs.empty()) {
    oc_next->addIncoming(out_count, after_filter_bb);  // filter failed
  }
  // stage 0 probe empty → next_bb
  oc_next->addIncoming(out_count, sbb[0].probe_bb);
  // stage 0 chain done → next_bb
  oc_next->addIncoming(done_ocs[0], sbb[0].chain_done_bb);

  Value *i_next = cc.b.CreateAdd(row_i, ConstantInt::get(i64, 1));
  row_i->addIncoming(i_next, next_bb);
  out_count->addIncoming(oc_next, next_bb);
  cc.b.CreateBr(outer_bb);

  // --- Exit ---
  cc.b.SetInsertPoint(exit_bb);
  cc.b.CreateStore(out_count,
                   cc.b.CreateStructGEP(ChunkViewTy, out_arg, 1));
  cc.b.CreateRet(out_count);

  SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);

  std::string err;
  raw_string_ostream es(err);
  if (verifyFunction(*fn, &es)) {
#ifndef NDEBUG
    std::cerr << "[AQP-JIT-MULTI] LLVM verify failed: " << err << "\n";
    fn->print(errs());
#endif
    return nullptr;
  }

  OptimiseModule(*mod, skip_opt_);

  impl_->pending_cache_key = cache_key;
  auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
  if (auto e = impl_->jit->addIRModule(impl_->current_tracker, std::move(tsm))) {
    impl_->pending_cache_key.clear();
    logAllUnhandledErrors(std::move(e), errs());
    return nullptr;
  }

  auto sym = impl_->jit->lookup(fn_name);
  impl_->pending_cache_key.clear();
  if (!sym) {
    logAllUnhandledErrors(sym.takeError(), errs());
    return nullptr;
  }

#ifndef NDEBUG
  std::cerr << "[AQP-JIT-MULTI] compiled " << N << "-probe chain fn="
            << fn_name << " out_cols=" << final_out.size() << "\n";
  for (size_t oi = 0; oi < final_out.size(); oi++) {
    auto &fc = final_out[oi];
    std::cerr << "  final_out[" << oi << "] source="
              << (fc.source == FinalOutCol::PROBE_INPUT ? "PROBE" : "HT")
              << " stage=" << fc.stage_idx << " col=" << fc.col_idx
              << " dtype=" << fc.dtype << " elem=" << fc.elem_size << "\n";
  }
  for (size_t s = 0; s < N; s++) {
    std::cerr << "  stage_output_origins[" << s << "]: ";
    for (size_t j = 0; j < stage_output_origins[s].size(); j++) {
      auto &co = stage_output_origins[s][j];
      std::cerr << (co.source == FinalOutCol::PROBE_INPUT ? "P" : "H")
                << co.stage_idx << ":" << co.col_idx << "(" << co.dtype << ") ";
    }
    std::cerr << "\n";
  }
#endif

  if (template_mode && params_out) {
    tmpl_pb.Finalize();
    *params_out = std::move(tmpl_pb.buf);
  }

  return AQP_JIT_GET_ADDR(sym);
}

// ---------------------------------------------------------------------------
// Query-jit (--jit-level=query): runtime symbol registration + Phase 2
// single-step compilation (join-free scan -> filter -> project -> sink).
// ---------------------------------------------------------------------------

bool IrToLlvmCompiler::RegisterRuntimeSymbol(const char *name, void *addr) {
  auto &es = impl_->jit->getExecutionSession();
  auto &jd = impl_->jit->getMainJITDylib();
  if (auto err = jd.define(absoluteSymbols({{es.intern(name),
                                             AQP_JIT_SYM(addr)}}))) {
    // Duplicate definition — the symbol is already registered; that is the
    // idempotent success case (defines without a tracker are permanent).
    consumeError(std::move(err));
  }
  return true;
}

void *IrToLlvmCompiler::CompileQuerySteps(const qjit::QjitQueryPlan &plan,
                                          std::vector<uint8_t> *params_out) {
  if (plan.steps.empty())
    return nullptr;

  const bool template_mode = cache_mode_ == 2;

  uint64_t fn_id = s_filter_counter.fetch_add(1, std::memory_order_relaxed);
  std::string entry_name = "qjit_query_" + std::to_string(fn_id);

  // §5.1 object cache: the module is row-count independent (nrows /
  // morsel_size are runtime loads from ctx), so a canonical plan
  // serialization + codegen flags fully determines the object code.
  std::string cache_key;
  if (cache_enabled_ && impl_->cache_enabled) {
    std::string serialized_plan = template_mode
        ? SerializeQjitPlanTemplate(plan)
        : SerializeQjitPlan(plan);
    std::string opt_tag = std::to_string((int)simd_isa_) + "F" +
                          std::to_string((int)fast_mode_) +
                          (skip_opt_ ? "n" : "o") +
                          "H" + std::to_string(skip_hash_cmp_) +
                          (bloom_tag_ ? "B" : "_") +
                          "C" + std::to_string(cache_mode_);
    cache_key = Impl::ComputeCacheKey("qjit:" + opt_tag + "||" +
                                      serialized_plan);
    entry_name = "qjit_query_c" + cache_key.substr(0, 12);
    impl_->last_cache_key = cache_key;
    impl_->last_entry_name = entry_name;
    void *cached = impl_->TryCacheLoad(cache_key, entry_name);

#ifndef NDEBUG
    // §7.3a: dump cache keys for near-miss clustering analysis.
    {
      const char *dp = std::getenv("AQP_DUMP_CACHE_KEYS");
      if (dp && dp[0]) {
        static std::mutex dmu;
        static uint64_t dseq = 0;
        std::lock_guard<std::mutex> lk(dmu);
        std::string line = "SEQ=" + std::to_string(dseq++) +
                           "\tHIT=" + std::to_string(cached ? 1 : 0) +
                           "\tKEY=" + cache_key +
                           "\tOPT=" + opt_tag +
                           "\tPLAN=" + serialized_plan + "\n";
        int fd = ::open(dp, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd >= 0) {
          (void)::write(fd, line.data(), line.size());
          ::close(fd);
        }
      }
    }
#endif

    if (cached) {
      if (template_mode && params_out)
        *params_out = BuildParamsBuffer(plan);
      return cached;
    }
  }

  auto ctx = std::make_unique<LLVMContext>();
  auto mod = std::make_unique<Module>("aqp_qjit_mod", *ctx);
  auto &C = *ctx;

  // Per-step schemas for the expression codegen (must outlive the per-step
  // CompileCtx references below).
  std::vector<std::vector<ColSchema>> schemas(plan.steps.size());
  for (size_t k = 0; k < plan.steps.size(); k++)
    for (const auto &c : plan.steps[k].cols)
      schemas[k].push_back({c.table_index, c.column_index, c.expected_dtype});

  // Types/constants. Literal struct types are uniqued per LLVMContext, so
  // every per-step CompileCtx sees the same AQPColView/AQPChunkView types.
  std::vector<ColSchema> entry_schema;
  CompileCtx ecc(C, *mod, entry_schema, /*chunk=*/nullptr, /*sel=*/nullptr);

  Type *void_ty = Type::getVoidTy(C);
  Type *i8 = Type::getInt8Ty(C);
  Type *i8p = ecc.i8p();
  Type *i8pp = PointerType::getUnqual(i8p);
  Type *i32 = ecc.i32();
  Type *i64 = ecc.i64();
  Type *i64p = PointerType::getUnqual(i64);
  Type *i16 = Type::getInt16Ty(C);

  // §6.1 bloom-tag global: declare the extern bloomMasks[2048] table for
  // codegen to GEP into (the actual data lives in query_jit_runtime.cpp).
  GlobalVariable *bloom_gv = nullptr;
  ArrayType *bloom_arr_ty = ArrayType::get(i16, 2048);
  if (bloom_tag_) {
    bloom_gv = cast<GlobalVariable>(
        mod->getOrInsertGlobal("qjit_bloom_masks", bloom_arr_ty));
    bloom_gv->setConstant(true);
    bloom_gv->setLinkage(GlobalValue::ExternalLinkage);
  }

  // QjitQueryContext (query_jit_abi.h — FIELD ORDER IS ABI):
  // 0 pool, 1 sources, 2 num_sources, 3 hash_tables, 4 num_hash_tables,
  // 5 worker_states, 6 num_workers, 7 morsel_size, 8 result, 9 user.
  // QjitTableView/QjitColView are bit-identical to AQPChunkViewTy/
  // AQPColViewTy, so the CompileCtx column helpers work unchanged with
  // chunk_arg = &ctx->sources[k].
  StructType *QjitCtxTy = StructType::get(
      C, {i8p, PointerType::getUnqual(ecc.AQPChunkViewTy), i64, i8p, i64, i8p,
          i64, i64, i8p, i8p});
  Type *ctxp = PointerType::getUnqual(QjitCtxTy);

  // QjitUserData (query_jit_abi.h): { block_stats, params, params_size }
  StructType *QjitUserDataTy = StructType::get(C, {i8p, i8p, i64});
  Type *udp = PointerType::getUnqual(QjitUserDataTy);

  FunctionCallee append_i32 = mod->getOrInsertFunction(
      "qjit_table_append_i32",
      FunctionType::get(void_ty, {i8p, i32, i64, i32}, false));
  FunctionCallee append_str = mod->getOrInsertFunction(
      "qjit_table_append_str",
      FunctionType::get(void_ty, {i8p, i32, i64, i8p}, false));
  FunctionCallee append_null = mod->getOrInsertFunction(
      "qjit_table_append_null",
      FunctionType::get(void_ty, {i8p, i32, i64}, false));
  FunctionCallee finish_row = mod->getOrInsertFunction(
      "qjit_table_finish_row", FunctionType::get(void_ty, {i8p, i32}, false));
  FunctionCallee ht_append = mod->getOrInsertFunction(
      "qjit_ht_append", FunctionType::get(i8p, {i8p, i32, i64}, false));
  // §6.12 fast-path HT append handle: {cursor, limit, stride, count}
  StructType *HtHandleTy = StructType::get(C, {i8p, i8p, i64, i64p});
  Type *ht_handle_p = PointerType::getUnqual(HtHandleTy);
  FunctionCallee ht_begin_fn = mod->getOrInsertFunction(
      "qjit_ht_begin", FunctionType::get(void_ty, {i8p, i32, ht_handle_p}, false));
  FunctionCallee ht_append_slow_fn = mod->getOrInsertFunction(
      "qjit_ht_append_slow",
      FunctionType::get(i8p, {i8p, i32, i64, ht_handle_p}, false));
  FunctionCallee ht_end_fn = mod->getOrInsertFunction(
      "qjit_ht_end", FunctionType::get(void_ty, {i8p, i32, ht_handle_p}, false));
  // §6.13 fast-path table append handle: {val_cursor, val_limit, null_cursor, null_limit}
  StructType *TblColHandleTy = StructType::get(C, {i8p, i8p, i8p, i8p});
  Type *tbl_handle_p = PointerType::getUnqual(TblColHandleTy);
  FunctionCallee tbl_begin_fn = mod->getOrInsertFunction(
      "qjit_table_begin",
      FunctionType::get(void_ty, {i8p, i32, tbl_handle_p, i64}, false));
  FunctionCallee tbl_col_slow_fn = mod->getOrInsertFunction(
      "qjit_table_col_slow",
      FunctionType::get(void_ty, {i8p, i32, i64, tbl_handle_p, i64}, false));
  FunctionCallee tbl_null_slow_fn = mod->getOrInsertFunction(
      "qjit_table_null_slow",
      FunctionType::get(void_ty, {i8p, i32, i64, tbl_handle_p}, false));
  FunctionCallee tbl_end_fn = mod->getOrInsertFunction(
      "qjit_table_end",
      FunctionType::get(void_ty, {i8p, i32, tbl_handle_p, i64, i64}, false));
  FunctionCallee tbl_str_copy_fn = mod->getOrInsertFunction(
      "qjit_table_str_copy",
      FunctionType::get(void_ty, {i8p, i32, i8p, i8p}, false));

  FunctionCallee ht_dir_fn = mod->getOrInsertFunction(
      "qjit_ht_dir", FunctionType::get(i8p, {i8p}, false));
  FunctionCallee ht_mask_fn = mod->getOrInsertFunction(
      "qjit_ht_mask", FunctionType::get(i64, {i8p}, false));
  FunctionCallee ht_finalize_fn = mod->getOrInsertFunction(
      "qjit_ht_finalize", FunctionType::get(void_ty, {i8p, i8p}, false));
  FunctionCallee ht_key0_min_fn = mod->getOrInsertFunction(
      "qjit_ht_key0_min", FunctionType::get(i64, {i8p}, false));
  FunctionCallee ht_key0_max_fn = mod->getOrInsertFunction(
      "qjit_ht_key0_max", FunctionType::get(i64, {i8p}, false));
  FunctionCallee ht_entries_fn = mod->getOrInsertFunction(
      "qjit_ht_entries", FunctionType::get(i64, {i8p}, false));
  FunctionCallee agg_upd_i64 = mod->getOrInsertFunction(
      "qjit_agg_update_i64",
      FunctionType::get(void_ty, {i8p, i64, i64}, false));
  FunctionCallee agg_upd_str = mod->getOrInsertFunction(
      "qjit_agg_update_str",
      FunctionType::get(void_ty, {i8p, i64, i8p}, false));
  FunctionCallee agg_upd_cnt = mod->getOrInsertFunction(
      "qjit_agg_update_count", FunctionType::get(void_ty, {i8p, i64}, false));

  FunctionType *morsel_ft =
      FunctionType::get(void_ty, {ctxp, i64, i64, i32}, false);

  Constant *MURMUR_MUL = ConstantInt::get(i64, 0xd6e8feb86659fd93ULL);
  Constant *SHIFT32 = ConstantInt::get(i64, 32);

  std::vector<Function *> morsel_fns;
  morsel_fns.reserve(plan.steps.size());

  auto can_skip_ehash = [&](int ht_id) -> bool {
    if (skip_hash_cmp_ == 0) return false;
    const qjit::QjitHtDesc &ht = plan.hts[ht_id];
    bool all_int = true;
    for (uint32_t j = 0; j < ht.num_keys; j++) {
      int d = ht.cols[j].dtype;
      if (d != AQP_DTYPE_INT8 && d != AQP_DTYPE_INT16 &&
          d != AQP_DTYPE_INT32 && d != AQP_DTYPE_INT64) {
        all_int = false;
        break;
      }
    }
    bool skip = (skip_hash_cmp_ == 2 && all_int);
#ifndef NDEBUG
    std::cerr << "[AQP-QJIT] skip_hash_cmp: mode=" << skip_hash_cmp_
              << " ht=" << ht_id << " nkeys=" << ht.num_keys
              << " all_int=" << all_int << " skip=" << skip << "\n";
#endif
    return skip;
  };

  // §7.3 template mode: build params layout in lock-step with codegen.
  ParamsBuilder tmpl_params;
  ParamsBuilder *tmpl_params_ptr = template_mode ? &tmpl_params : nullptr;

  // ---- one morsel body per step: void(ctx, begin, end, worker_id) ----
  for (size_t k = 0; k < plan.steps.size(); k++) {
    const qjit::QjitStep &st = plan.steps[k];
    CompileCtx cc(C, *mod, schemas[k], /*chunk=*/nullptr, /*sel=*/nullptr);
    cc.strict_null_guard = true;
    cc.template_mode = template_mode;
    cc.params_builder = tmpl_params_ptr;

    auto emitMurmur = [&](Value *x) -> Value * {
      Value *t = cc.b.CreateXor(x, cc.b.CreateLShr(x, SHIFT32));
      t = cc.b.CreateMul(t, MURMUR_MUL);
      t = cc.b.CreateXor(t, cc.b.CreateLShr(t, SHIFT32));
      t = cc.b.CreateMul(t, MURMUR_MUL);
      t = cc.b.CreateXor(t, cc.b.CreateLShr(t, SHIFT32));
      return t;
    };
    auto emitCombine = [&](Value *a, Value *bv) -> Value * {
      Value *t = cc.b.CreateXor(a, cc.b.CreateLShr(a, SHIFT32));
      t = cc.b.CreateMul(t, MURMUR_MUL);
      return cc.b.CreateXor(t, bv);
    };

    std::string morsel_name = "qjit_morsel_" + std::to_string(fn_id) + "_" +
                              std::to_string(k);
    Function *fn = Function::Create(morsel_ft, Function::InternalLinkage,
                                    morsel_name, *mod);
    Value *m_ctx = fn->getArg(0);
    Value *m_begin = fn->getArg(1);
    Value *m_end = fn->getArg(2);
    Value *m_worker = fn->getArg(3);

    BasicBlock *bb_entry = BasicBlock::Create(C, "entry", fn);
    BasicBlock *bb_head = BasicBlock::Create(C, "loop_head", fn);
    BasicBlock *bb_body = BasicBlock::Create(C, "body", fn);
    BasicBlock *bb_next = BasicBlock::Create(C, "next", fn);
    BasicBlock *bb_exit = BasicBlock::Create(C, "exit", fn);

    // -- entry: source views, runtime handles (once per morsel call) --
    cc.b.SetInsertPoint(bb_entry);
    Value *sources = cc.b.CreateLoad(
        PointerType::getUnqual(cc.AQPChunkViewTy),
        cc.b.CreateStructGEP(QjitCtxTy, m_ctx, 1), "sources");
    cc.chunk_arg = cc.b.CreateGEP(cc.AQPChunkViewTy, sources,
                                  cc.c64((int64_t)k), "src_k");
    cc.col_data.clear();
    cc.col_validity.clear();
    for (unsigned i = 0; i < (unsigned)st.cols.size(); ++i) {
      cc.col_data.push_back(cc.LoadColData(i));
      cc.col_validity.push_back(cc.LoadColValidity(i));
    }

    // §7.3 template mode: load params_base from QjitUserData once per morsel.
    if (template_mode) {
      Value *user_raw = cc.b.CreateLoad(
          i8p, cc.b.CreateStructGEP(QjitCtxTy, m_ctx, 9), "user_raw");
      Value *ud = cc.b.CreateBitCast(user_raw, udp);
      cc.params_base = cc.b.CreateLoad(
          i8p, cc.b.CreateStructGEP(QjitUserDataTy, ud, 1), "params_base");
    }

    std::vector<Value *> ht_ptr(plan.hts.size(), nullptr);
    std::vector<Value *> ht_dirv(plan.hts.size(), nullptr);
    std::vector<Value *> ht_maskv(plan.hts.size(), nullptr);
    Value *ht_base = nullptr;
    auto load_ht_ptr = [&](int id) {
      if (ht_ptr[id])
        return;
      if (!ht_base) {
        Value *raw = cc.b.CreateLoad(
            i8p, cc.b.CreateStructGEP(QjitCtxTy, m_ctx, 3), "ht_base_raw");
        ht_base = cc.b.CreateBitCast(raw, i8pp);
      }
      ht_ptr[id] = cc.b.CreateLoad(
          i8p, cc.b.CreateGEP(i8p, ht_base, cc.c64(id)), "ht");
    };
    for (const auto &op : st.ops) {
      if (op.kind != qjit::QjitStepOp::Probe)
        continue;
      load_ht_ptr(op.ht_id);
      // Probed HTs were finalized by the entry fn before this step ran.
      ht_dirv[op.ht_id] = cc.b.CreateBitCast(
          cc.b.CreateCall(ht_dir_fn, {ht_ptr[op.ht_id]}), i64p);
      ht_maskv[op.ht_id] = cc.b.CreateCall(ht_mask_fn, {ht_ptr[op.ht_id]});
    }

    // §5.5 A+ guards: loop-invariant build-key range + membership gate.
    // The membership pre-probe only pays off against small builds (the
    // walk it adds is wasted when most probe rows match); larger builds
    // skip it at runtime and rely on the range check alone.
    constexpr uint64_t kMembershipMaxEntries = 1u << 16;
    struct GuardVals {
      Value *lo = nullptr, *hi = nullptr, *member_ok = nullptr;
    };
    std::vector<GuardVals> guard_vals(st.guards.size());
    for (size_t gi = 0; gi < st.guards.size(); ++gi) {
      const qjit::QjitStep::Guard &g = st.guards[gi];
      Value *ghp = ht_ptr[st.ops[g.op_index].ht_id];
      guard_vals[gi].lo = cc.b.CreateCall(ht_key0_min_fn, {ghp}, "g_lo");
      guard_vals[gi].hi = cc.b.CreateCall(ht_key0_max_fn, {ghp}, "g_hi");
      if (g.membership)
        guard_vals[gi].member_ok = cc.b.CreateICmpULE(
            cc.b.CreateCall(ht_entries_fn, {ghp}),
            ConstantInt::get(i64, kMembershipMaxEntries), "g_mem_ok");
    }

    Value *result = nullptr, *agg_state = nullptr, *sink_ht_ptr = nullptr;
    Value *ht_handle = nullptr;
    Value *tbl_handles = nullptr, *tbl_row_count = nullptr;
    uint64_t tbl_ncols = st.outputs.size();
    switch (st.sink) {
    case qjit::QjitStep::Result:
      result = cc.b.CreateLoad(
          i8p, cc.b.CreateStructGEP(QjitCtxTy, m_ctx, 8), "result");
      tbl_handles = cc.b.CreateAlloca(
          TblColHandleTy, cc.c64((int64_t)tbl_ncols), "tbl_handles");
      tbl_row_count = cc.b.CreateAlloca(i64, nullptr, "tbl_rows");
      cc.b.CreateStore(cc.c64(0), tbl_row_count);
      cc.b.CreateCall(tbl_begin_fn, {result, m_worker, tbl_handles,
                                     cc.c64((int64_t)tbl_ncols)});
      break;
    case qjit::QjitStep::HtBuild:
      load_ht_ptr(st.sink_ht);
      sink_ht_ptr = ht_ptr[st.sink_ht];
      ht_handle = cc.b.CreateAlloca(HtHandleTy, nullptr, "ht_handle");
      cc.b.CreateCall(ht_begin_fn, {sink_ht_ptr, m_worker, ht_handle});
      break;
    case qjit::QjitStep::Agg: {
      Value *ws_raw = cc.b.CreateLoad(
          i8p, cc.b.CreateStructGEP(QjitCtxTy, m_ctx, 5), "ws_raw");
      Value *ws = cc.b.CreateBitCast(ws_raw, i8pp);
      Value *widx = cc.b.CreateZExt(m_worker, i64, "widx");
      agg_state = cc.b.CreateLoad(i8p, cc.b.CreateGEP(i8p, ws, widx),
                                  "agg_state");
      break;
    }
    }
    // §5.5 A+ block skip: per-step block-stats pointer from
    // QjitUserData::block_stats (const int32_t** indexed by step).
    bool use_block_skip = st.block_skip_col >= 0 && !st.guards.empty();
    Value *stats_ptr = nullptr;
    if (use_block_skip) {
      Value *user_raw = cc.b.CreateLoad(
          i8p, cc.b.CreateStructGEP(QjitCtxTy, m_ctx, 9), "user_raw");
      BasicBlock *bb_ust = BasicBlock::Create(C, "user_stats", fn);
      BasicBlock *bb_ujoin = BasicBlock::Create(C, "user_join", fn);
      BasicBlock *upre = cc.b.GetInsertBlock();
      cc.b.CreateCondBr(cc.b.CreateIsNotNull(user_raw, "has_user"), bb_ust,
                        bb_ujoin);
      cc.b.SetInsertPoint(bb_ust);
      // Load block_stats from QjitUserData field 0
      Value *ud_bs = cc.b.CreateBitCast(user_raw, udp);
      Value *bstats = cc.b.CreateLoad(
          i8p, cc.b.CreateStructGEP(QjitUserDataTy, ud_bs, 0), "bstats");
      Value *bstats_ok = cc.b.CreateIsNotNull(bstats, "has_bstats");
      BasicBlock *bb_bst = BasicBlock::Create(C, "bstats_load", fn);
      BasicBlock *bb_bst_join = BasicBlock::Create(C, "bstats_join", fn);
      cc.b.CreateCondBr(bstats_ok, bb_bst, bb_bst_join);

      cc.b.SetInsertPoint(bb_bst);
      Value *stats0 = cc.b.CreateLoad(
          i8p,
          cc.b.CreateGEP(i8p, cc.b.CreateBitCast(bstats, i8pp),
                         cc.c64((int64_t)k)),
          "stats_k");
      cc.b.CreateBr(bb_bst_join);
      cc.b.SetInsertPoint(bb_bst_join);
      PHINode *bsp = cc.b.CreatePHI(i8p, 2, "bstats_k");
      bsp->addIncoming(ConstantPointerNull::get(cast<PointerType>(i8p)), bb_ust);
      bsp->addIncoming(stats0, bb_bst);
      cc.b.CreateBr(bb_ujoin);

      cc.b.SetInsertPoint(bb_ujoin);
      PHINode *sp = cc.b.CreatePHI(i8p, 2, "stats");
      sp->addIncoming(ConstantPointerNull::get(cast<PointerType>(i8p)), upre);
      sp->addIncoming(bsp, bb_bst_join);
      stats_ptr = sp;
    }

    // -- §6.11 ROF decision: two-phase scan when SIMD is on and a probe
    // exists whose keys are all source columns (so Phase 1 can evaluate
    // guards + compute hash without HT lookups). TPDE can't optimise the
    // tight Phase 1 loop well enough to pay for the extra pass overhead.
    int rof_first_probe = -1; // index in st.ops of the probe used for ROF
    bool use_rof = false;
    if (use_simd_ && impl_->vec_width > 1 &&
        fast_mode_ != FastCompileBackend::TPDE) {
      for (size_t oi = 0; oi < st.ops.size(); ++oi) {
        const qjit::QjitStepOp &op = st.ops[oi];
        if (op.kind != qjit::QjitStepOp::Probe)
          continue;
        bool all_src = !op.keys.empty();
        for (const auto &kl : op.keys)
          all_src = all_src && kl.src_col >= 0;
        if (all_src) {
          rof_first_probe = (int)oi;
          use_rof = true;
        }
        break;
      }
    }

    // ROF Phase 1: scan [begin,end) evaluating guards + filters before the
    // first probe, compute hash for the ROF probe, write (row_idx, hash)
    // to a stack selection buffer.  Phase 2 iterates the selection buffer.
    // The buffer is sized to the morsel (max rows = end - begin).
    Value *rof_sel_buf = nullptr;  // i64[morsel]: selected row indices
    Value *rof_hash_buf = nullptr; // i64[morsel]: precomputed hashes
    Value *rof_count = nullptr;    // i64*: number of selected rows
    PHINode *rof_r2_idx = nullptr; // Phase 2 loop index (into sel/hash bufs)
    if (use_rof) {
      Value *morsel_n = cc.b.CreateSub(m_end, m_begin, "morsel_n");
      rof_sel_buf = cc.b.CreateAlloca(i64, morsel_n, "rof_sel");
      rof_hash_buf = cc.b.CreateAlloca(i64, morsel_n, "rof_hash");
      rof_count = cc.b.CreateAlloca(i64, nullptr, "rof_cnt");
      cc.b.CreateStore(cc.c64(0), rof_count);

      // Phase 1 loop: iterate [begin, end) with optional block-skip.
      BasicBlock *r1_head = BasicBlock::Create(C, "rof1_head", fn);
      BasicBlock *r1_body = BasicBlock::Create(C, "rof1_body", fn);
      BasicBlock *r1_next = BasicBlock::Create(C, "rof1_next", fn);
      BasicBlock *r1_done = BasicBlock::Create(C, "rof1_done", fn);

      if (!use_block_skip) {
        BasicBlock *r1_pre = cc.b.GetInsertBlock();
        cc.b.CreateBr(r1_head);
        cc.b.SetInsertPoint(r1_head);
        PHINode *r1_row = cc.b.CreatePHI(i64, 2, "r1_row");
        r1_row->addIncoming(m_begin, r1_pre);
        cc.b.CreateCondBr(cc.b.CreateICmpULT(r1_row, m_end), r1_body, r1_done);

        cc.b.SetInsertPoint(r1_body);
        cc.row_idx = r1_row;

        // Evaluate guard range checks (same logic as the in-loop guards).
        BasicBlock *r1_pass = BasicBlock::Create(C, "rof1_pass", fn);
        BasicBlock *r1_fail = r1_next;
        BasicBlock *r1_cur = r1_body;
        for (size_t gi = 0; gi < st.guards.size(); ++gi) {
          const qjit::QjitStep::Guard &g = st.guards[gi];
          const qjit::QjitStepOp &gop = st.ops[g.op_index];
          Value *k0_raw = cc.b.CreateSExt(
              LoadI32(cc, cc.col_data[gop.keys[0].src_col]), i64);
          Value *in_range = cc.b.CreateAnd(
              cc.b.CreateICmpSGE(k0_raw, guard_vals[gi].lo),
              cc.b.CreateICmpSLE(k0_raw, guard_vals[gi].hi), "r1_in_range");
          BasicBlock *bb_gnext = BasicBlock::Create(C, "r1_gnext", fn);
          cc.b.CreateCondBr(in_range, bb_gnext, r1_fail);
          cc.b.SetInsertPoint(bb_gnext);
        }

        // Evaluate filters that appear before the first probe.
        for (size_t oi = 0; oi < (size_t)rof_first_probe; ++oi) {
          const qjit::QjitStepOp &op = st.ops[oi];
          if (op.kind != qjit::QjitStepOp::Filter)
            continue;
          BasicBlock *bb_fp = BasicBlock::Create(C, "r1_fpass", fn);
          std::vector<const ir_sql_converter::AQPExpr *> one{op.filter};
          EmitShortCircuitFilter(cc, fn, one, bb_fp, r1_fail);
          cc.b.SetInsertPoint(bb_fp);
        }

        // Compute hash for the ROF probe's keys.
        {
          const qjit::QjitStepOp &rop = st.ops[rof_first_probe];
          Value *k0 = cc.b.CreateSExt(
              LoadI32(cc, cc.col_data[rop.keys[0].src_col]), i64);
          Value *h = emitMurmur(k0);
          for (size_t j = 1; j < rop.keys.size(); ++j) {
            Value *kj = cc.b.CreateSExt(
                LoadI32(cc, cc.col_data[rop.keys[j].src_col]), i64);
            h = emitCombine(h, emitMurmur(kj));
          }
          Value *cnt = cc.b.CreateLoad(i64, rof_count, "r1_cnt");
          cc.b.CreateStore(r1_row, cc.b.CreateGEP(i64, rof_sel_buf, cnt));
          cc.b.CreateStore(h, cc.b.CreateGEP(i64, rof_hash_buf, cnt));
          cc.b.CreateStore(cc.b.CreateAdd(cnt, cc.c64(1)), rof_count);
        }
        cc.b.CreateBr(r1_next);

        cc.b.SetInsertPoint(r1_next);
        Value *r1_inc = cc.b.CreateAdd(r1_row, cc.c64(1), "r1_inc");
        r1_row->addIncoming(r1_inc, r1_next);
        cc.b.CreateBr(r1_head);

        cc.b.SetInsertPoint(r1_done);
      } else {
        // Phase 1 with block-skip: two-level loop identical to the scalar
        // path but writing to the selection buffer instead of processing.
        BasicBlock *r1_oh = BasicBlock::Create(C, "rof1_blk_head", fn);
        BasicBlock *r1_obody = BasicBlock::Create(C, "rof1_blk_body", fn);
        BasicBlock *r1_bchk = BasicBlock::Create(C, "rof1_blk_chk", fn);
        BasicBlock *r1_ipre = BasicBlock::Create(C, "rof1_blk_enter", fn);
        BasicBlock *r1_onext = BasicBlock::Create(C, "rof1_blk_next", fn);
        BasicBlock *r1_pre = cc.b.GetInsertBlock();
        cc.b.CreateBr(r1_oh);

        cc.b.SetInsertPoint(r1_oh);
        PHINode *r1_cur = cc.b.CreatePHI(i64, 2, "r1_blk_cur");
        r1_cur->addIncoming(m_begin, r1_pre);
        cc.b.CreateCondBr(cc.b.CreateICmpULT(r1_cur, m_end), r1_obody, r1_done);

        cc.b.SetInsertPoint(r1_obody);
        Value *r1_blk_raw = cc.b.CreateAdd(
            cc.b.CreateAnd(r1_cur, cc.c64(~(int64_t)(QJIT_BLOCK_ROWS - 1))),
            cc.c64((int64_t)QJIT_BLOCK_ROWS), "r1_blk_raw");
        Value *r1_blk_end = cc.b.CreateSelect(
            cc.b.CreateICmpULT(r1_blk_raw, m_end), r1_blk_raw, m_end,
            "r1_blk_end");
        cc.b.CreateCondBr(cc.b.CreateIsNotNull(stats_ptr, "r1_has_stats"),
                          r1_bchk, r1_ipre);

        cc.b.SetInsertPoint(r1_bchk);
        Value *r1_stats_i32 =
            cc.b.CreateBitCast(stats_ptr, PointerType::getUnqual(i32));
        Value *r1_blk = cc.b.CreateLShr(r1_cur, cc.c64(QJIT_BLOCK_SHIFT),
                                         "r1_blk");
        Value *r1_sidx = cc.b.CreateShl(r1_blk, cc.c64(1), "r1_sidx");
        Value *r1_bmin = cc.b.CreateSExt(
            cc.b.CreateLoad(i32, cc.b.CreateGEP(i32, r1_stats_i32, r1_sidx)),
            i64, "r1_blk_min");
        Value *r1_bmax = cc.b.CreateSExt(
            cc.b.CreateLoad(
                i32, cc.b.CreateGEP(i32, r1_stats_i32,
                                    cc.b.CreateAdd(r1_sidx, cc.c64(1)))),
            i64, "r1_blk_max");
        Value *r1_skip = cc.b.CreateOr(
            cc.b.CreateICmpSLT(r1_bmax, guard_vals[0].lo),
            cc.b.CreateICmpSGT(r1_bmin, guard_vals[0].hi), "r1_blk_skip");
        cc.b.CreateCondBr(r1_skip, r1_onext, r1_ipre);

        cc.b.SetInsertPoint(r1_ipre);
        cc.b.CreateBr(r1_head);

        cc.b.SetInsertPoint(r1_head);
        PHINode *r1_row = cc.b.CreatePHI(i64, 2, "r1_row");
        r1_row->addIncoming(r1_cur, r1_ipre);
        cc.b.CreateCondBr(cc.b.CreateICmpULT(r1_row, r1_blk_end),
                          r1_body, r1_onext);

        cc.b.SetInsertPoint(r1_body);
        cc.row_idx = r1_row;

        // Guard range checks.
        for (size_t gi = 0; gi < st.guards.size(); ++gi) {
          const qjit::QjitStep::Guard &g = st.guards[gi];
          const qjit::QjitStepOp &gop = st.ops[g.op_index];
          Value *k0_raw = cc.b.CreateSExt(
              LoadI32(cc, cc.col_data[gop.keys[0].src_col]), i64);
          Value *in_range = cc.b.CreateAnd(
              cc.b.CreateICmpSGE(k0_raw, guard_vals[gi].lo),
              cc.b.CreateICmpSLE(k0_raw, guard_vals[gi].hi), "r1_in_range");
          BasicBlock *bb_gnext = BasicBlock::Create(C, "r1_gnext", fn);
          cc.b.CreateCondBr(in_range, bb_gnext, r1_next);
          cc.b.SetInsertPoint(bb_gnext);
        }

        // Filters before first probe.
        for (size_t oi = 0; oi < (size_t)rof_first_probe; ++oi) {
          const qjit::QjitStepOp &op = st.ops[oi];
          if (op.kind != qjit::QjitStepOp::Filter)
            continue;
          BasicBlock *bb_fp = BasicBlock::Create(C, "r1_fpass", fn);
          std::vector<const ir_sql_converter::AQPExpr *> one{op.filter};
          EmitShortCircuitFilter(cc, fn, one, bb_fp, r1_next);
          cc.b.SetInsertPoint(bb_fp);
        }

        // Hash for ROF probe.
        {
          const qjit::QjitStepOp &rop = st.ops[rof_first_probe];
          Value *k0 = cc.b.CreateSExt(
              LoadI32(cc, cc.col_data[rop.keys[0].src_col]), i64);
          Value *h = emitMurmur(k0);
          for (size_t j = 1; j < rop.keys.size(); ++j) {
            Value *kj = cc.b.CreateSExt(
                LoadI32(cc, cc.col_data[rop.keys[j].src_col]), i64);
            h = emitCombine(h, emitMurmur(kj));
          }
          Value *cnt = cc.b.CreateLoad(i64, rof_count, "r1_cnt");
          cc.b.CreateStore(r1_row, cc.b.CreateGEP(i64, rof_sel_buf, cnt));
          cc.b.CreateStore(h, cc.b.CreateGEP(i64, rof_hash_buf, cnt));
          cc.b.CreateStore(cc.b.CreateAdd(cnt, cc.c64(1)), rof_count);
        }
        cc.b.CreateBr(r1_next);

        cc.b.SetInsertPoint(r1_next);
        Value *r1_inc = cc.b.CreateAdd(r1_row, cc.c64(1), "r1_inc");
        r1_row->addIncoming(r1_inc, r1_next);
        cc.b.CreateBr(r1_head);

        cc.b.SetInsertPoint(r1_onext);
        r1_cur->addIncoming(r1_blk_end, r1_onext);
        cc.b.CreateBr(r1_oh);

        cc.b.SetInsertPoint(r1_done);
      }
      // Phase 2: iterate selection buffer [0, rof_count).
      Value *rof_total = cc.b.CreateLoad(i64, rof_count, "rof_total");
      BasicBlock *r2_pre = cc.b.GetInsertBlock();
      cc.b.CreateBr(bb_head);

      cc.b.SetInsertPoint(bb_head);
      rof_r2_idx = cc.b.CreatePHI(i64, 2, "r2_idx");
      rof_r2_idx->addIncoming(cc.c64(0), r2_pre);
      cc.b.CreateCondBr(cc.b.CreateICmpULT(rof_r2_idx, rof_total), bb_body,
                        bb_exit);

      cc.b.SetInsertPoint(bb_next);
      Value *r2_inc = cc.b.CreateAdd(rof_r2_idx, cc.c64(1), "r2_inc");
      rof_r2_idx->addIncoming(r2_inc, bb_next);
      cc.b.CreateBr(bb_head);
    }

    // -- row loop (scalar, non-ROF) --
    PHINode *row = nullptr;
    if (!use_rof) {
      if (!use_block_skip) {
        BasicBlock *entry_tail = cc.b.GetInsertBlock();
        cc.b.CreateBr(bb_head);
        cc.b.SetInsertPoint(bb_head);
        row = cc.b.CreatePHI(i64, 2, "row");
        row->addIncoming(m_begin, entry_tail);
        cc.b.CreateCondBr(cc.b.CreateICmpULT(row, m_end), bb_body, bb_exit);

        cc.b.SetInsertPoint(bb_next);
        Value *row_next = cc.b.CreateAdd(row, cc.c64(1), "row_next");
        row->addIncoming(row_next, bb_next);
        cc.b.CreateBr(bb_head);
      } else {
        // Two-level loop: outer iterates QJIT_BLOCK_ROWS-aligned blocks and
        // skips a whole block when its min/max range is disjoint from the
        // build-key range of guards[0]; inner is the per-row loop.
        BasicBlock *bb_oh = BasicBlock::Create(C, "blk_head", fn);
        BasicBlock *bb_obody = BasicBlock::Create(C, "blk_body", fn);
        BasicBlock *bb_bchk = BasicBlock::Create(C, "blk_chk", fn);
        BasicBlock *bb_ipre = BasicBlock::Create(C, "blk_enter", fn);
        BasicBlock *bb_onext = BasicBlock::Create(C, "blk_next", fn);
        BasicBlock *entry_tail = cc.b.GetInsertBlock();
        cc.b.CreateBr(bb_oh);

        cc.b.SetInsertPoint(bb_oh);
        PHINode *cur = cc.b.CreatePHI(i64, 2, "blk_cur");
        cur->addIncoming(m_begin, entry_tail);
        cc.b.CreateCondBr(cc.b.CreateICmpULT(cur, m_end), bb_obody, bb_exit);

        cc.b.SetInsertPoint(bb_obody);
        Value *blk_raw = cc.b.CreateAdd(
            cc.b.CreateAnd(cur, cc.c64(~(int64_t)(QJIT_BLOCK_ROWS - 1))),
            cc.c64((int64_t)QJIT_BLOCK_ROWS), "blk_raw");
        Value *blk_end = cc.b.CreateSelect(
            cc.b.CreateICmpULT(blk_raw, m_end), blk_raw, m_end, "blk_end");
        cc.b.CreateCondBr(cc.b.CreateIsNotNull(stats_ptr, "has_stats"),
                          bb_bchk, bb_ipre);

        cc.b.SetInsertPoint(bb_bchk);
        Value *stats_i32 =
            cc.b.CreateBitCast(stats_ptr, PointerType::getUnqual(i32));
        Value *blk = cc.b.CreateLShr(cur, cc.c64(QJIT_BLOCK_SHIFT), "blk");
        Value *sidx = cc.b.CreateShl(blk, cc.c64(1), "sidx");
        Value *bmin = cc.b.CreateSExt(
            cc.b.CreateLoad(i32, cc.b.CreateGEP(i32, stats_i32, sidx)), i64,
            "blk_min");
        Value *bmax = cc.b.CreateSExt(
            cc.b.CreateLoad(
                i32,
                cc.b.CreateGEP(i32, stats_i32,
                               cc.b.CreateAdd(sidx, cc.c64(1)))),
            i64, "blk_max");
        Value *skip = cc.b.CreateOr(
            cc.b.CreateICmpSLT(bmax, guard_vals[0].lo),
            cc.b.CreateICmpSGT(bmin, guard_vals[0].hi), "blk_skip");
        cc.b.CreateCondBr(skip, bb_onext, bb_ipre);

        cc.b.SetInsertPoint(bb_ipre);
        cc.b.CreateBr(bb_head);

        cc.b.SetInsertPoint(bb_head);
        row = cc.b.CreatePHI(i64, 2, "row");
        row->addIncoming(cur, bb_ipre);
        cc.b.CreateCondBr(cc.b.CreateICmpULT(row, blk_end), bb_body, bb_onext);

        cc.b.SetInsertPoint(bb_next);
        Value *row_next = cc.b.CreateAdd(row, cc.c64(1), "row_next");
        row->addIncoming(row_next, bb_next);
        cc.b.CreateBr(bb_head);

        cc.b.SetInsertPoint(bb_onext);
        cur->addIncoming(blk_end, bb_onext);
        cc.b.CreateBr(bb_oh);
      }
    }

    cc.b.SetInsertPoint(bb_exit);
    if (ht_handle)
      cc.b.CreateCall(ht_end_fn, {sink_ht_ptr, m_worker, ht_handle});
    if (tbl_handles) {
      Value *nrows_final = cc.b.CreateLoad(i64, tbl_row_count, "nrows_final");
      cc.b.CreateCall(tbl_end_fn, {result, m_worker, tbl_handles,
                                   cc.c64((int64_t)tbl_ncols), nrows_final});
    }
    cc.b.CreateRetVoid();

    cc.b.SetInsertPoint(bb_body);
    if (use_rof) {
      Value *sel_row = cc.b.CreateLoad(
          i64, cc.b.CreateGEP(i64, rof_sel_buf, rof_r2_idx), "sel_row");
      cc.row_idx = sel_row;
    } else {
      cc.row_idx = row;
    }

    // cont = where a finished/failed row iteration goes. After a probe it
    // becomes that probe's advance block, so the chain walk resumes and
    // every matching entry produces its own sink row (multi-match).
    BasicBlock *cont = bb_next;
    std::vector<Value *> live_row(plan.hts.size(), nullptr);

    // ---- value-location helpers ----
    auto src_valid_i1 = [&](int src_col) -> Value * {
      Value *validity = cc.col_validity[src_col];
      BasicBlock *pre = cc.b.GetInsertBlock();
      BasicBlock *bb_chk = BasicBlock::Create(C, "sv_chk", fn);
      BasicBlock *bb_after = BasicBlock::Create(C, "sv_after", fn);
      cc.b.CreateCondBr(cc.b.CreateIsNotNull(validity, "sv_has"), bb_chk,
                        bb_after);
      cc.b.SetInsertPoint(bb_chk);
      Value *bit = EmitValidityCheck(cc, validity);
      BasicBlock *chk_end = cc.b.GetInsertBlock();
      cc.b.CreateBr(bb_after);
      cc.b.SetInsertPoint(bb_after);
      PHINode *phi = cc.b.CreatePHI(Type::getInt1Ty(C), 2, "sv");
      phi->addIncoming(ConstantInt::getTrue(C), pre);
      phi->addIncoming(bit, chk_end);
      return phi;
    };
    auto ht_valid_i1 = [&](const qjit::QjitValueLoc &loc) -> Value * {
      const qjit::QjitHtDesc &ht = plan.hts[loc.ht_id];
      if ((uint32_t)loc.layout_col < ht.num_keys)
        return (Value *)ConstantInt::getTrue(C); // keys are never NULL
      Value *bytep = cc.b.CreateGEP(i8, live_row[loc.ht_id],
                                    cc.c64(loc.layout_col / 8));
      Value *byte = cc.b.CreateLoad(i8, bytep, "vprefix");
      Value *bit = cc.b.CreateAnd(
          byte, ConstantInt::get(i8, 1u << (loc.layout_col % 8)));
      return cc.b.CreateICmpNE(bit, ConstantInt::get(i8, 0), "ht_valid");
    };
    auto loc_valid_i1 = [&](const qjit::QjitValueLoc &loc) -> Value * {
      return loc.src_col >= 0 ? src_valid_i1(loc.src_col) : ht_valid_i1(loc);
    };
    auto loc_value_i32 = [&](const qjit::QjitValueLoc &loc) -> Value * {
      if (loc.src_col >= 0)
        return LoadI32(cc, cc.col_data[loc.src_col]);
      const qjit::QjitHtDesc &ht = plan.hts[loc.ht_id];
      const qjit::QjitHtCol &hc = ht.cols[loc.layout_col];
      Value *p = cc.b.CreateGEP(i8, live_row[loc.ht_id], cc.c64(hc.offset));
      if ((uint32_t)loc.layout_col < ht.num_keys) // key slot reuse: i64
        return cc.b.CreateTrunc(
            cc.b.CreateLoad(i64, cc.b.CreateBitCast(p, i64p)), i32);
      return cc.b.CreateLoad(
          i32, cc.b.CreateBitCast(p, PointerType::getUnqual(i32)));
    };
    auto loc_value_str = [&](const qjit::QjitValueLoc &loc) -> Value * {
      if (loc.src_col >= 0) {
        Value *off = cc.b.CreateMul(cc.row_idx, cc.c64(16));
        return cc.b.CreateGEP(i8, cc.col_data[loc.src_col], off, "str_ptr");
      }
      const qjit::QjitHtCol &hc = plan.hts[loc.ht_id].cols[loc.layout_col];
      return cc.b.CreateGEP(i8, live_row[loc.ht_id], cc.c64(hc.offset),
                            "ht_str_ptr");
    };
    // Join-key gather: NULL key => branch to `cont` (inner-join: NULL never
    // matches / is never built). Returns the key sign-extended to i64.
    auto load_key_i64_or_skip = [&](const qjit::QjitValueLoc &loc) -> Value * {
      Value *valid = loc_valid_i1(loc);
      if (!isa<ConstantInt>(valid) || !cast<ConstantInt>(valid)->isOne()) {
        BasicBlock *bb_ok = BasicBlock::Create(C, "key_ok", fn);
        cc.b.CreateCondBr(valid, bb_ok, cont);
        cc.b.SetInsertPoint(bb_ok);
      }
      if (loc.src_col >= 0)
        return cc.b.CreateSExt(LoadI32(cc, cc.col_data[loc.src_col]), i64);
      const qjit::QjitHtDesc &ht = plan.hts[loc.ht_id];
      const qjit::QjitHtCol &hc = ht.cols[loc.layout_col];
      Value *p = cc.b.CreateGEP(i8, live_row[loc.ht_id], cc.c64(hc.offset));
      if ((uint32_t)loc.layout_col < ht.num_keys)
        return cc.b.CreateLoad(i64, cc.b.CreateBitCast(p, i64p));
      return cc.b.CreateSExt(
          cc.b.CreateLoad(i32,
                          cc.b.CreateBitCast(p, PointerType::getUnqual(i32))),
          i64);
    };

    // ---- in-loop ops ----
    Value *rof_precomp_hash = nullptr;
    if (use_rof) {
      rof_precomp_hash = cc.b.CreateLoad(
          i64, cc.b.CreateGEP(i64, rof_hash_buf, rof_r2_idx), "rof_hash");
    }
    for (size_t oi = 0; oi < st.ops.size(); ++oi) {
      // In ROF mode, guards and filters before the first probe were already
      // evaluated in Phase 1 — skip them in Phase 2.
      if (use_rof && (int)oi < rof_first_probe) {
        if (st.ops[oi].kind == qjit::QjitStepOp::Filter)
          continue;
      }
      if (use_rof && !st.guards.empty() && (int)oi == st.guard_pos)
        goto rof_skip_guards;
      // §5.5 A+ guards, emitted at guard_pos (after the cheap filters,
      // before the expensive ones). cont == bb_next here — guards precede
      // every Probe — so a guard drop skips the whole row, which is
      // exactly what the guarded probe would do for that row later.
      if (!st.guards.empty() && (int)oi == st.guard_pos) {
        for (size_t gi = 0; gi < st.guards.size(); ++gi) {
          const qjit::QjitStep::Guard &g = st.guards[gi];
          const qjit::QjitStepOp &gop = st.ops[g.op_index];
          const qjit::QjitHtDesc &ght = plan.hts[gop.ht_id];
          // Key 0 range check against the build-side min/max.
          Value *k0 = load_key_i64_or_skip(gop.keys[0]);
          Value *in_range = cc.b.CreateAnd(
              cc.b.CreateICmpSGE(k0, guard_vals[gi].lo),
              cc.b.CreateICmpSLE(k0, guard_vals[gi].hi), "g_in_range");
          BasicBlock *bb_gnext = BasicBlock::Create(C, "g_next", fn);
          if (!g.membership) {
            cc.b.CreateCondBr(in_range, bb_gnext, cont);
            cc.b.SetInsertPoint(bb_gnext);
            continue;
          }
          BasicBlock *bb_grange = BasicBlock::Create(C, "g_range_ok", fn);
          cc.b.CreateCondBr(in_range, bb_grange, cont);
          cc.b.SetInsertPoint(bb_grange);
          // Existence pre-probe (runtime-gated on build size). Must mirror
          // the real probe's match predicate exactly: hash eq + per-key
          // i64 eq. First match suffices — no multi-match backtracking.
          std::vector<Value *> kv{k0};
          for (size_t j = 1; j < gop.keys.size(); ++j)
            kv.push_back(load_key_i64_or_skip(gop.keys[j]));
          Value *h = emitMurmur(kv[0]);
          for (size_t j = 1; j < kv.size(); ++j)
            h = emitCombine(h, emitMurmur(kv[j]));
          BasicBlock *bb_mem = BasicBlock::Create(C, "g_mem", fn);
          cc.b.CreateCondBr(guard_vals[gi].member_ok, bb_mem, bb_gnext);
          cc.b.SetInsertPoint(bb_mem);
          Value *slot = cc.b.CreateAnd(h, ht_maskv[gop.ht_id], "g_slot");
          Value *head_word = cc.b.CreateLoad(
              i64, cc.b.CreateGEP(i64, ht_dirv[gop.ht_id], slot), "g_head_w");
          if (bloom_tag_) {
            Value *tag = cc.b.CreateTrunc(head_word, i16, "g_tag");
            Value *idx = cc.b.CreateLShr(h, cc.c64(53), "g_bidx");
            Value *mask = cc.b.CreateLoad(
                i16,
                cc.b.CreateGEP(bloom_arr_ty, bloom_gv,
                               {cc.c64(0), idx}),
                "g_bmask");
            Value *reject = cc.b.CreateICmpNE(
                cc.b.CreateAnd(mask, cc.b.CreateNot(tag)), ConstantInt::get(i16, 0),
                "g_bloom_reject");
            BasicBlock *bb_bloom_pass = BasicBlock::Create(C, "g_bloom_ok", fn);
            cc.b.CreateCondBr(reject, cont, bb_bloom_pass);
            cc.b.SetInsertPoint(bb_bloom_pass);
          }
          Value *head = cc.b.CreateIntToPtr(
              cc.b.CreateLShr(head_word, cc.c64(16)), i8p, "g_head");
          BasicBlock *bb_from = cc.b.GetInsertBlock();
          BasicBlock *bb_ghead = BasicBlock::Create(C, "g_phead", fn);
          BasicBlock *bb_gchk = BasicBlock::Create(C, "g_check", fn);
          BasicBlock *bb_gadv = BasicBlock::Create(C, "g_adv", fn);
          cc.b.CreateBr(bb_ghead);
          cc.b.SetInsertPoint(bb_ghead);
          PHINode *e = cc.b.CreatePHI(i8p, 2, "g_entry");
          e->addIncoming(head, bb_from);
          cc.b.CreateCondBr(cc.b.CreateIsNull(e, "g_chain_end"), cont,
                            bb_gchk);
          cc.b.SetInsertPoint(bb_gchk);
          Value *row_ptr = cc.b.CreateGEP(i8, e, cc.c64(16), "g_row");
          Value *match;
          if (can_skip_ehash(gop.ht_id)) {
            match = ConstantInt::getTrue(C);
          } else {
            Value *ehash = cc.b.CreateLoad(
                i64,
                cc.b.CreateBitCast(cc.b.CreateGEP(i8, e, cc.c64(8)), i64p),
                "g_ehash");
            match = cc.b.CreateICmpEQ(ehash, h, "g_hash_eq");
          }
          for (size_t j = 0; j < kv.size(); ++j) {
            Value *kp =
                cc.b.CreateGEP(i8, row_ptr, cc.c64(ght.cols[j].offset));
            Value *bk = cc.b.CreateLoad(i64, cc.b.CreateBitCast(kp, i64p),
                                        "g_bkey");
            match = cc.b.CreateAnd(match, cc.b.CreateICmpEQ(bk, kv[j]));
          }
          cc.b.CreateCondBr(match, bb_gnext, bb_gadv);
          cc.b.SetInsertPoint(bb_gadv);
          Value *next_e =
              cc.b.CreateLoad(i8p, cc.b.CreateBitCast(e, i8pp), "g_next_e");
          e->addIncoming(next_e, bb_gadv);
          cc.b.CreateBr(bb_ghead);
          cc.b.SetInsertPoint(bb_gnext);
        }
      }
      rof_skip_guards:
      const qjit::QjitStepOp &op = st.ops[oi];
      if (op.kind == qjit::QjitStepOp::Filter) {
        BasicBlock *bb_pass = BasicBlock::Create(C, "f_pass", fn);
        std::vector<const ir_sql_converter::AQPExpr *> one{op.filter};
        EmitShortCircuitFilter(cc, fn, one, bb_pass, cont);
        cc.b.SetInsertPoint(bb_pass);
        continue;
      }

      // Probe: chained hash-table walk with multi-match backtracking.
      const qjit::QjitHtDesc &ht = plan.hts[op.ht_id];
      if (op.keys.size() != ht.num_keys)
        return nullptr;
      std::vector<Value *> kv;
      for (const auto &kl : op.keys)
        kv.push_back(load_key_i64_or_skip(kl));
      // §6.11 ROF: use precomputed hash for the first probe.
      Value *h;
      if (use_rof && (int)oi == rof_first_probe) {
        h = rof_precomp_hash;
      } else {
        h = emitMurmur(kv[0]);
        for (size_t j = 1; j < kv.size(); ++j)
          h = emitCombine(h, emitMurmur(kv[j]));
      }
      Value *slot = cc.b.CreateAnd(h, ht_maskv[op.ht_id], "slot");
      Value *head_word = cc.b.CreateLoad(
          i64, cc.b.CreateGEP(i64, ht_dirv[op.ht_id], slot), "head_w");
      if (bloom_tag_) {
        Value *tag = cc.b.CreateTrunc(head_word, i16, "tag");
        Value *idx = cc.b.CreateLShr(h, cc.c64(53), "bidx");
        Value *mask = cc.b.CreateLoad(
            i16,
            cc.b.CreateGEP(bloom_arr_ty, bloom_gv,
                           {cc.c64(0), idx}),
            "bmask");
        Value *reject = cc.b.CreateICmpNE(
            cc.b.CreateAnd(mask, cc.b.CreateNot(tag)), ConstantInt::get(i16, 0),
            "bloom_reject");
        BasicBlock *bb_bloom_pass = BasicBlock::Create(C, "bloom_ok", fn);
        cc.b.CreateCondBr(reject, cont, bb_bloom_pass);
        cc.b.SetInsertPoint(bb_bloom_pass);
      }
      Value *head = cc.b.CreateIntToPtr(
          cc.b.CreateLShr(head_word, cc.c64(16)), i8p, "head");
      BasicBlock *bb_from = cc.b.GetInsertBlock();
      BasicBlock *bb_phead = BasicBlock::Create(C, "p_head", fn);
      BasicBlock *bb_check = BasicBlock::Create(C, "p_check", fn);
      BasicBlock *bb_adv = BasicBlock::Create(C, "p_adv", fn);
      BasicBlock *bb_match = BasicBlock::Create(C, "p_match", fn);
      cc.b.CreateBr(bb_phead);

      cc.b.SetInsertPoint(bb_phead);
      PHINode *e = cc.b.CreatePHI(i8p, 2, "entry");
      e->addIncoming(head, bb_from);
      cc.b.CreateCondBr(cc.b.CreateIsNull(e, "chain_end"), cont, bb_check);

      cc.b.SetInsertPoint(bb_check);
      // Entry layout: { next @0, hash @8, row @16 } (QjitHashTable::Entry).
      Value *row_ptr = cc.b.CreateGEP(i8, e, cc.c64(16), "ht_row");
      Value *match;
      if (can_skip_ehash(op.ht_id)) {
        match = ConstantInt::getTrue(C);
      } else {
        Value *ehash = cc.b.CreateLoad(
            i64,
            cc.b.CreateBitCast(cc.b.CreateGEP(i8, e, cc.c64(8)), i64p),
            "ehash");
        match = cc.b.CreateICmpEQ(ehash, h, "hash_eq");
      }
      for (size_t j = 0; j < kv.size(); ++j) {
        Value *kp = cc.b.CreateGEP(i8, row_ptr, cc.c64(ht.cols[j].offset));
        Value *bk = cc.b.CreateLoad(i64, cc.b.CreateBitCast(kp, i64p),
                                    "build_key");
        match = cc.b.CreateAnd(match, cc.b.CreateICmpEQ(bk, kv[j]));
      }
      cc.b.CreateCondBr(match, bb_match, bb_adv);

      cc.b.SetInsertPoint(bb_adv);
      Value *next_e =
          cc.b.CreateLoad(i8p, cc.b.CreateBitCast(e, i8pp), "next_e");
      e->addIncoming(next_e, bb_adv);
      cc.b.CreateBr(bb_phead);

      cc.b.SetInsertPoint(bb_match);
      live_row[op.ht_id] = row_ptr;
      cont = bb_adv;
    }

    // ---- sink ----
    switch (st.sink) {
    case qjit::QjitStep::Result: {
      // §6.13 inline cursor-bump table append (no per-column function calls).
      for (size_t j = 0; j < st.outputs.size(); ++j) {
        const qjit::QjitValueLoc &loc = st.outputs[j];
        Value *col_j = cc.c64((int64_t)j);
        uint64_t elem_size = (loc.dtype == AQP_DTYPE_VARCHAR) ? 16 : 4;
        Value *esz = cc.c64((int64_t)elem_size);

        Value *handle_j = cc.b.CreateGEP(TblColHandleTy, tbl_handles,
                                         cc.c64((int64_t)j), "handle_j");
        Value *vc_ptr = cc.b.CreateStructGEP(TblColHandleTy, handle_j, 0);
        Value *vl_ptr = cc.b.CreateStructGEP(TblColHandleTy, handle_j, 1);
        Value *nc_ptr = cc.b.CreateStructGEP(TblColHandleTy, handle_j, 2);
        Value *nl_ptr = cc.b.CreateStructGEP(TblColHandleTy, handle_j, 3);

        // Value cursor check: val_cursor + elem_size <= val_limit
        Value *vc = cc.b.CreateLoad(i8p, vc_ptr, "vc");
        Value *vc_next = cc.b.CreateGEP(i8, vc, esz, "vc_next");
        Value *vl = cc.b.CreateLoad(i8p, vl_ptr, "vl");
        Value *v_fits = cc.b.CreateICmpULE(
            cc.b.CreatePtrToInt(vc_next, i64),
            cc.b.CreatePtrToInt(vl, i64), "v_fits");
        BasicBlock *bb_v_ok = BasicBlock::Create(C, "v_ok", fn);
        BasicBlock *bb_v_slow = BasicBlock::Create(C, "v_slow", fn);
        BasicBlock *bb_v_ready = BasicBlock::Create(C, "v_ready", fn);
        cc.b.CreateCondBr(v_fits, bb_v_ok, bb_v_slow);

        cc.b.SetInsertPoint(bb_v_slow);
        cc.b.CreateCall(tbl_col_slow_fn,
                        {result, m_worker, col_j, handle_j, esz});
        Value *vc_after_slow = cc.b.CreateLoad(i8p, vc_ptr, "vc_s");
        cc.b.CreateBr(bb_v_ready);

        cc.b.SetInsertPoint(bb_v_ok);
        cc.b.CreateBr(bb_v_ready);

        cc.b.SetInsertPoint(bb_v_ready);
        PHINode *vc_use = cc.b.CreatePHI(i8p, 2, "vc_use");
        vc_use->addIncoming(vc, bb_v_ok);
        vc_use->addIncoming(vc_after_slow, bb_v_slow);

        // Null cursor check: null_cursor + 1 <= null_limit
        Value *nc = cc.b.CreateLoad(i8p, nc_ptr, "nc");
        Value *nc_next = cc.b.CreateGEP(i8, nc, cc.c64(1), "nc_next");
        Value *nl = cc.b.CreateLoad(i8p, nl_ptr, "nl");
        Value *n_fits = cc.b.CreateICmpULE(
            cc.b.CreatePtrToInt(nc_next, i64),
            cc.b.CreatePtrToInt(nl, i64), "n_fits");
        BasicBlock *bb_n_ok = BasicBlock::Create(C, "n_ok", fn);
        BasicBlock *bb_n_slow = BasicBlock::Create(C, "n_slow", fn);
        BasicBlock *bb_n_ready = BasicBlock::Create(C, "n_ready", fn);
        cc.b.CreateCondBr(n_fits, bb_n_ok, bb_n_slow);

        cc.b.SetInsertPoint(bb_n_slow);
        cc.b.CreateCall(tbl_null_slow_fn, {result, m_worker, col_j, handle_j});
        Value *nc_after_slow = cc.b.CreateLoad(i8p, nc_ptr, "nc_s");
        cc.b.CreateBr(bb_n_ready);

        cc.b.SetInsertPoint(bb_n_ok);
        cc.b.CreateBr(bb_n_ready);

        cc.b.SetInsertPoint(bb_n_ready);
        PHINode *nc_use = cc.b.CreatePHI(i8p, 2, "nc_use");
        nc_use->addIncoming(nc, bb_n_ok);
        nc_use->addIncoming(nc_after_slow, bb_n_slow);

        Value *valid = loc_valid_i1(loc);
        BasicBlock *bb_val = BasicBlock::Create(C, "out_val", fn);
        BasicBlock *bb_null = BasicBlock::Create(C, "out_null", fn);
        BasicBlock *bb_done = BasicBlock::Create(C, "out_done", fn);
        cc.b.CreateCondBr(valid, bb_val, bb_null);

        cc.b.SetInsertPoint(bb_null);
        // Zero-fill the value slot
        if (elem_size == 4) {
          cc.b.CreateStore(ConstantInt::get(i32, 0),
                           cc.b.CreateBitCast(vc_use, PointerType::getUnqual(i32)));
        } else {
          auto *i128 = Type::getIntNTy(C, 128);
          cc.b.CreateStore(ConstantInt::get(i128, 0),
                           cc.b.CreateBitCast(vc_use, PointerType::getUnqual(i128)));
        }
        cc.b.CreateStore(ConstantInt::get(i8, 1), nc_use);
        cc.b.CreateBr(bb_done);

        cc.b.SetInsertPoint(bb_val);
        if (loc.dtype == AQP_DTYPE_INT32) {
          cc.b.CreateStore(loc_value_i32(loc),
                           cc.b.CreateBitCast(vc_use, PointerType::getUnqual(i32)));
        } else {
          // VARCHAR: copy the 16-byte QjitString slot, then deep-copy via str_copy
          Value *src_str = loc_value_str(loc);
          cc.b.CreateCall(tbl_str_copy_fn, {result, m_worker, vc_use, src_str});
        }
        cc.b.CreateStore(ConstantInt::get(i8, 0), nc_use);
        cc.b.CreateBr(bb_done);

        cc.b.SetInsertPoint(bb_done);
        // Advance cursors
        Value *vc_adv = cc.b.CreateGEP(i8, vc_use, esz);
        cc.b.CreateStore(vc_adv, vc_ptr);
        Value *nc_adv = cc.b.CreateGEP(i8, nc_use, cc.c64(1));
        cc.b.CreateStore(nc_adv, nc_ptr);
      }
      // Increment row counter (replaces finish_row call)
      Value *rc = cc.b.CreateLoad(i64, tbl_row_count, "rc");
      cc.b.CreateStore(cc.b.CreateAdd(rc, cc.c64(1)), tbl_row_count);
      cc.b.CreateBr(cont);
      break;
    }

    case qjit::QjitStep::HtBuild: {
      const qjit::QjitHtDesc &ht = plan.hts[st.sink_ht];
      if (st.outputs.size() != ht.cols.size() || ht.num_keys == 0)
        return nullptr;
      // Keys first: any NULL key skips the row (inner-join build).
      std::vector<Value *> kv;
      for (uint32_t j = 0; j < ht.num_keys; ++j)
        kv.push_back(load_key_i64_or_skip(st.outputs[j]));
      Value *h = emitMurmur(kv[0]);
      for (size_t j = 1; j < kv.size(); ++j)
        h = emitCombine(h, emitMurmur(kv[j]));
      // §6.12 inline fast-path: bump cursor within current chunk, avoiding
      // the qjit_ht_append function call on the common path.
      Value *cursor_ptr = cc.b.CreateStructGEP(HtHandleTy, ht_handle, 0);
      Value *limit_ptr = cc.b.CreateStructGEP(HtHandleTy, ht_handle, 1);
      Value *stride_val = cc.b.CreateLoad(
          i64, cc.b.CreateStructGEP(HtHandleTy, ht_handle, 2), "stride");
      Value *cursor_v = cc.b.CreateLoad(i8p, cursor_ptr, "cursor");
      Value *next_v = cc.b.CreateGEP(i8, cursor_v, stride_val, "next_cursor");
      Value *limit_v = cc.b.CreateLoad(i8p, limit_ptr, "limit");
      Value *fits = cc.b.CreateICmpULE(
          cc.b.CreatePtrToInt(next_v, i64),
          cc.b.CreatePtrToInt(limit_v, i64), "fits");
      BasicBlock *bb_fast = BasicBlock::Create(C, "ht_fast", fn);
      BasicBlock *bb_slow = BasicBlock::Create(C, "ht_slow", fn);
      BasicBlock *bb_sink = BasicBlock::Create(C, "ht_sink", fn);
      cc.b.CreateCondBr(fits, bb_fast, bb_slow);
      // Fast path: bump cursor, fill Entry header, return row pointer
      cc.b.SetInsertPoint(bb_fast);
      cc.b.CreateStore(next_v, cursor_ptr);
      Value *cnt_ptr = cc.b.CreateLoad(
          i64p, cc.b.CreateStructGEP(HtHandleTy, ht_handle, 3), "cnt_ptr");
      Value *cnt = cc.b.CreateLoad(i64, cnt_ptr, "cnt");
      cc.b.CreateStore(cc.b.CreateAdd(cnt, cc.c64(1)), cnt_ptr);
      // Entry header: next @0, hash @8, row @16
      cc.b.CreateStore(ConstantPointerNull::get(cast<PointerType>(i8p)),
                       cc.b.CreateBitCast(cursor_v, i8pp));
      cc.b.CreateStore(
          h, cc.b.CreateBitCast(cc.b.CreateGEP(i8, cursor_v, cc.c64(8)), i64p));
      Value *rowp_fast = cc.b.CreateGEP(i8, cursor_v, cc.c64(16), "rowp_fast");
      cc.b.CreateBr(bb_sink);
      // Slow path: call runtime, which allocates a new chunk + refreshes handle
      cc.b.SetInsertPoint(bb_slow);
      Value *rowp_slow = cc.b.CreateCall(
          ht_append_slow_fn, {sink_ht_ptr, m_worker, h, ht_handle}, "rowp_slow");
      cc.b.CreateBr(bb_sink);
      // Merge
      cc.b.SetInsertPoint(bb_sink);
      PHINode *rowp = cc.b.CreatePHI(i8p, 2, "rowp");
      rowp->addIncoming(rowp_fast, bb_fast);
      rowp->addIncoming(rowp_slow, bb_slow);
      // Payload validity (branchless: i1 per payload column).
      std::vector<Value *> vbits(ht.cols.size(), nullptr);
      for (size_t cx = ht.num_keys; cx < ht.cols.size(); ++cx)
        vbits[cx] = loc_valid_i1(st.outputs[cx]);
      // Validity byte prefix (bit c%8 of byte c/8; keys constant-valid).
      for (uint32_t byte = 0; byte * 8 < (uint32_t)ht.cols.size(); ++byte) {
        Value *acc = ConstantInt::get(i8, 0);
        for (uint32_t bi = 0; bi < 8; ++bi) {
          uint32_t cx = byte * 8 + bi;
          if (cx >= (uint32_t)ht.cols.size())
            break;
          Value *vb = cx < ht.num_keys ? (Value *)ConstantInt::getTrue(C)
                                       : vbits[cx];
          acc = cc.b.CreateOr(
              acc, cc.b.CreateShl(cc.b.CreateZExt(vb, i8),
                                  ConstantInt::get(i8, bi)));
        }
        cc.b.CreateStore(acc, cc.b.CreateGEP(i8, rowp, cc.c64(byte)));
      }
      // Keys (i64 slots).
      for (uint32_t j = 0; j < ht.num_keys; ++j) {
        Value *kp = cc.b.CreateGEP(i8, rowp, cc.c64(ht.cols[j].offset));
        cc.b.CreateStore(kv[j], cc.b.CreateBitCast(kp, i64p));
      }
      // Payloads. Branchless: loads are always safe (flat arrays / zeroed
      // slots); invalid INT32 slots are zeroed via select, invalid VARCHAR
      // slots are zero by construction (executor zeroes NULL source slots;
      // the invariant propagates through HT-to-HT copies).
      for (size_t cx = ht.num_keys; cx < ht.cols.size(); ++cx) {
        const qjit::QjitValueLoc &loc = st.outputs[cx];
        const qjit::QjitHtCol &hc = ht.cols[cx];
        Value *dst = cc.b.CreateGEP(i8, rowp, cc.c64(hc.offset));
        if (hc.dtype == AQP_DTYPE_VARCHAR) {
          Value *sp = loc_value_str(loc);
          for (int64_t half = 0; half < 16; half += 8) {
            Value *w = cc.b.CreateLoad(
                i64,
                cc.b.CreateBitCast(cc.b.CreateGEP(i8, sp, cc.c64(half)),
                                   i64p));
            cc.b.CreateStore(
                w, cc.b.CreateBitCast(cc.b.CreateGEP(i8, dst, cc.c64(half)),
                                      i64p));
          }
        } else {
          Value *v = loc_value_i32(loc);
          Value *vv = cc.b.CreateSelect(vbits[cx], v, cc.c32(0));
          cc.b.CreateStore(vv,
                           cc.b.CreateBitCast(dst, PointerType::getUnqual(i32)));
        }
      }
      cc.b.CreateBr(cont);
      break;
    }

    case qjit::QjitStep::Agg: {
      for (size_t i = 0; i < st.agg_cells.size(); ++i) {
        const qjit::QjitAggCellPlan &cell = st.agg_cells[i];
        Value *cell_i = cc.c64((int64_t)i);
        if (!cell.has_arg) { // CountStar
          cc.b.CreateCall(agg_upd_cnt, {agg_state, cell_i});
          continue;
        }
        Value *valid = loc_valid_i1(cell.arg);
        BasicBlock *bb_do = BasicBlock::Create(C, "agg_do", fn);
        BasicBlock *bb_skip = BasicBlock::Create(C, "agg_skip", fn);
        cc.b.CreateCondBr(valid, bb_do, bb_skip);
        cc.b.SetInsertPoint(bb_do);
        if (cell.fn == qjit::QjitAggFn::Count) {
          cc.b.CreateCall(agg_upd_cnt, {agg_state, cell_i});
        } else if (cell.arg.dtype == AQP_DTYPE_VARCHAR) {
          cc.b.CreateCall(agg_upd_str,
                          {agg_state, cell_i, loc_value_str(cell.arg)});
        } else {
          cc.b.CreateCall(
              agg_upd_i64,
              {agg_state, cell_i,
               cc.b.CreateSExt(loc_value_i32(cell.arg), i64)});
        }
        cc.b.CreateBr(bb_skip);
        cc.b.SetInsertPoint(bb_skip);
      }
      cc.b.CreateBr(cont);
      break;
    }
    }

    SetTargetAttrs(fn, impl_->host_cpu, impl_->feature_str);
    morsel_fns.push_back(fn);
  }

  // ---- entry: i64(ctx) — parallel_for per step + ht_finalize after builds --
  FunctionType *entry_ft = FunctionType::get(i64, {ctxp}, false);
  Function *entry_fn = Function::Create(entry_ft, Function::ExternalLinkage,
                                        entry_name, *mod);
  Value *e_ctx = entry_fn->getArg(0);
  BasicBlock *e_bb = BasicBlock::Create(C, "entry", entry_fn);
  ecc.b.SetInsertPoint(e_bb);
  Value *e_srcs = ecc.b.CreateLoad(PointerType::getUnqual(ecc.AQPChunkViewTy),
                                   ecc.b.CreateStructGEP(QjitCtxTy, e_ctx, 1),
                                   "sources");
  Value *morsel = ecc.b.CreateLoad(
      i64, ecc.b.CreateStructGEP(QjitCtxTy, e_ctx, 7), "morsel");
  FunctionCallee pfor = mod->getOrInsertFunction(
      "qjit_parallel_for",
      FunctionType::get(void_ty,
                        {ctxp, i64, i64, PointerType::getUnqual(morsel_ft)},
                        false));
  Value *e_ht_base = nullptr;
  for (size_t k = 0; k < plan.steps.size(); k++) {
    Value *src_k = ecc.b.CreateGEP(ecc.AQPChunkViewTy, e_srcs,
                                   ecc.c64((int64_t)k));
    Value *nrows = ecc.b.CreateLoad(
        i64, ecc.b.CreateStructGEP(ecc.AQPChunkViewTy, src_k, 1), "nrows");
    ecc.b.CreateCall(pfor, {e_ctx, nrows, morsel, morsel_fns[k]});
    if (plan.steps[k].sink == qjit::QjitStep::HtBuild) {
      if (!e_ht_base) {
        Value *raw = ecc.b.CreateLoad(
            i8p, ecc.b.CreateStructGEP(QjitCtxTy, e_ctx, 3), "ht_base_raw");
        e_ht_base = ecc.b.CreateBitCast(raw, i8pp);
      }
      Value *htv = ecc.b.CreateLoad(
          i8p,
          ecc.b.CreateGEP(i8p, e_ht_base, ecc.c64(plan.steps[k].sink_ht)),
          "ht");
      ecc.b.CreateCall(ht_finalize_fn,
                       {ecc.b.CreateBitCast(e_ctx, i8p), htv});
    }
  }
  ecc.b.CreateRet(ecc.c64(0));

  SetTargetAttrs(entry_fn, impl_->host_cpu, impl_->feature_str);

  std::string err;
  raw_string_ostream es(err);
  bool bad = verifyFunction(*entry_fn, &es);
  for (Function *fn : morsel_fns)
    bad |= verifyFunction(*fn, &es);
  if (bad) {
#ifndef NDEBUG
    std::cerr << "[AQP-QJIT] LLVM verify failed: " << err << "\n";
    for (Function *fn : morsel_fns)
      fn->print(errs());
    entry_fn->print(errs());
#endif
    return nullptr;
  }

  auto cg_t0 = std::chrono::steady_clock::now();

  OptimiseModule(*mod, skip_opt_);

  auto cg_t1 = std::chrono::steady_clock::now();

  impl_->pending_cache_key = cache_key;
  auto tsm = ThreadSafeModule(std::move(mod), std::move(ctx));
  if (auto e = impl_->jit->addIRModule(impl_->current_tracker,
                                       std::move(tsm))) {
    impl_->pending_cache_key.clear();
    logAllUnhandledErrors(std::move(e), errs());
    return nullptr;
  }

  auto cg_t2 = std::chrono::steady_clock::now();

  auto sym = impl_->jit->lookup(entry_name);
  impl_->pending_cache_key.clear();
  if (!sym) {
    logAllUnhandledErrors(sym.takeError(), errs());
    return nullptr;
  }

  auto cg_t3 = std::chrono::steady_clock::now();
  last_cg_timing_.opt_us = std::chrono::duration_cast<std::chrono::microseconds>(cg_t1 - cg_t0).count();
  last_cg_timing_.add_us = std::chrono::duration_cast<std::chrono::microseconds>(cg_t2 - cg_t1).count();
  last_cg_timing_.lookup_us = std::chrono::duration_cast<std::chrono::microseconds>(cg_t3 - cg_t2).count();

  if (template_mode && params_out) {
    tmpl_params.Finalize();
    *params_out = std::move(tmpl_params.buf);
  }

  return AQP_JIT_GET_ADDR(sym);
}

// ---------------------------------------------------------------------------
} // namespace aqp_jit
