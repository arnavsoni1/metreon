#include "metreon/GraphIR/Lowering.h"
#include "metreon/GraphIR/Passes.h"
#include "metreon/MLIR/DeviceExecution.h"
#include "metreon/MLIR/Dialect.h"
#include "metreon/MLIR/GPULowering.h"
#include "metreon/MLIR/GPUVerification.h"
#include "metreon/MLIR/HostExecution.h"
#include "metreon/Parser/Parser.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

using namespace metreon;

void check(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream os(text);
  module.print(os);
  return os.str();
}

void registerTestDialects(mlir::MLIRContext &context) {
  context.getOrLoadDialect<MetreonDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::gpu::GPUDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::affine::AffineDialect>();
  context.getOrLoadDialect<mlir::ub::UBDialect>();
}

mlir::OwningOpRef<mlir::ModuleOp> parseMLIR(const std::string &irText,
                                            mlir::MLIRContext &context) {
  registerTestDialects(context);
  auto parsed = mlir::parseSourceString<mlir::ModuleOp>(irText, &context);
  check(static_cast<bool>(parsed), "failed to parse test MLIR snippet");
  check(mlir::succeeded(mlir::verify(*parsed)), "parsed test MLIR is invalid");
  return parsed;
}

void expectMLIRAccepted(const std::string &irText, mlir::MLIRContext &context) {
  auto module = parseMLIR(irText, context);
  std::string diagText;
  mlir::ScopedDiagnosticHandler handler(
      &context, [&](mlir::Diagnostic &diag) {
        llvm::raw_string_ostream os(diagText);
        diag.print(os);
        return mlir::success();
      });
  check(mlir::succeeded(verifyGPUContextHierarchyAndLoans(*module)),
        "expected MLIR module to pass GPU-2 verification, got: " + diagText);
}

void expectMLIRRejected(const std::string &irText, mlir::MLIRContext &context,
                        const std::string &expectedSubstring) {
  auto module = parseMLIR(irText, context);
  std::string diagText;
  mlir::ScopedDiagnosticHandler handler(
      &context, [&](mlir::Diagnostic &diag) {
        llvm::raw_string_ostream os(diagText);
        diag.print(os);
        return mlir::success();
      });
  check(mlir::failed(verifyGPUContextHierarchyAndLoans(*module)),
        "expected MLIR module to fail GPU-2 verification (" +
            expectedSubstring + "), but it succeeded");
  check(diagText.find(expectedSubstring) != std::string::npos,
        "expected diagnostic containing `" + expectedSubstring + "`, got: " +
            diagText);
}

void testUnifiedStructuredLowering(const char *vectorAddPath) {
  std::ifstream input(vectorAddPath);
  check(static_cast<bool>(input), "cannot read vector_add.mtr");
  const std::string source((std::istreambuf_iterator<char>(input)), {});

  mlir::MLIRContext context;
  registerTestDialects(context);

  for (const bool optimize : {false, true}) {
    parser::Parser parser(source);
    auto graph = graphir::lowerModule(parser.parseModule(), "vector_add.mtr",
                                      /*contextOnly=*/true);
    if (optimize) graphir::passes::runDefaultPipeline(graph);

    // 1. GPULowering (`--emit=gpu`) now handles device spans + intrinsics cleanly.
    auto gpuFlow = lowerGPUControlFlow(graph, context);
    check(mlir::succeeded(gpuFlow),
          "unified lowerGPUControlFlow failed on vector_add.mtr");
    const std::string gpuFlowText = printModule(**gpuFlow);
    check(gpuFlowText.find("scf.if") != std::string::npos ||
              gpuFlowText.find("affine.if") != std::string::npos,
          "lowerGPUControlFlow did not lift vector_add bounds check to scf/affine");
    check(gpuFlowText.find("cf.cond_br") == std::string::npos,
          "lowerGPUControlFlow left unstructured cf.cond_br");

    // 2. DeviceExecution (`lowerMetreonDeviceToGPU`) now emits structured scf/affine
    // inside `gpu.module` with zero `cf` ops, and lowers cleanly to NVVM/PTX and ROCDL!
    auto devImported = importDeviceExecution(graph, context);
    check(mlir::succeeded(devImported), "importDeviceExecution failed");
    check(mlir::succeeded(lowerMetreonDeviceToGPU(**devImported)),
          "lowerMetreonDeviceToGPU failed");
    const std::string devGPUText = printModule(**devImported);
    check((devGPUText.find("scf.if") != std::string::npos ||
           devGPUText.find("affine.if") != std::string::npos) &&
              devGPUText.find("cf.br") == std::string::npos &&
              devGPUText.find("cf.cond_br") == std::string::npos,
          "DeviceExecution failed to unify with structured scf/affine control flow");

    auto nvvm = lowerDeviceToNVVM(**devImported);
    check(mlir::succeeded(nvvm), "structured GPU -> NVVM lowering failed");
    auto ptx = emitDevicePTX(**nvvm, "sm_89");
    check(mlir::succeeded(ptx) && ptx->find(".entry vector_add") != std::string::npos,
          "structured GPU -> PTX emission failed");

    auto rocdl = lowerDeviceToROCDL(**devImported);
    check(mlir::succeeded(rocdl) &&
              printModule(**rocdl).find("llvm.zext") != std::string::npos,
          "structured GPU -> ROCDL lowering failed");
  }

  // 3. Counted device `for` loop lifts to `affine.for` in `DeviceExecution` while
  // preserving `Gpu::Block` workgroup barrier support!
  const std::string blockKernelSrc = R"(
context Gpu::Block grants {gpu_load, gpu_store, gpu_barrier} block_ctx;
context Gpu::Loop gpu_loop;

kernel tiled_sum(@cx: block_ctx; a: gpu::span<f32>, out: gpu::span<f32>, n: index) {
  index i = gpu::global_id(@cx);
  f32 acc = 0.0;
  for<@cx: gpu_loop> (index step) in 0..4 {
    if (i < n) {
      acc = acc + gpu::load(@cx; a, i);
    }
  }
  if (i < n) {
    gpu::store(@cx; out, i, acc);
  }
}
)";
  parser::Parser blockParser(blockKernelSrc);
  auto blockGraph =
      graphir::lowerModule(blockParser.parseModule(), "block.mtr", true);
  graphir::passes::runDefaultPipeline(blockGraph);
  auto blockDev = importDeviceExecution(blockGraph, context);
  check(mlir::succeeded(blockDev) &&
            mlir::succeeded(lowerMetreonDeviceToGPU(**blockDev)),
        "Gpu::Block counted-loop kernel failed structured DeviceExecution lowering");
  const std::string blockText = printModule(**blockDev);
  check(blockText.find("affine.for") != std::string::npos,
        "DeviceExecution did not promote synthetic IV counted loop to affine.for");
}

void testMLIRStructuredOwnershipAndAsyncLoans() {
  mlir::MLIRContext context;

  const std::string resourceDecls = R"(
    "metreon.context"() <{
      generic_parameters = [],
      grants = [
        {arguments = [], name = "gpu_async_copy"},
        {arguments = [], name = "gpu_await"},
        {arguments = [], name = "gpu_barrier"},
        {arguments = [], name = "gpu_load"},
        {arguments = [], name = "gpu_store"}
      ],
      identifier = "block_ctx",
      nominal_type = {arguments = [], name = "Gpu::Block"},
      resolution = "declared",
      sym_name = "block_ctx"
    }> : () -> ()

    "metreon.resource"() <{
      entry_states = [@Vacant],
      generic_parameters = [],
      sym_name = "Slot"
    }> ({
      "metreon.state"() <{
        fields = [],
        lifecycle = "unspecified",
        sym_name = "Vacant"
      }> : () -> ()
      "metreon.state"() <{
        fields = [],
        lifecycle = "must_complete",
        sym_name = "InFlight"
      }> : () -> ()
      "metreon.state"() <{
        fields = [],
        lifecycle = "terminal",
        sym_name = "Done"
      }> : () -> ()
      "metreon.transition_decl"() <{
        context_requirements = [],
        effects = [],
        function_type = (!metreon.resource<@Slot::@Vacant, []>) -> !metreon.resource<@Slot::@InFlight, []>,
        generic_parameters = [],
        is_await = false,
        parameter_attrs = [{name = "self", ownership = "own", source_type = {arguments = [], name = "Vacant"}, type = !metreon.resource<@Slot::@Vacant, []>}],
        result_attrs = [{ownership = "own", source_type = {arguments = [], name = "InFlight"}, type = !metreon.resource<@Slot::@InFlight, []>}],
        self_index = 0 : i64,
        source_state = @Vacant,
        sym_name = "issue",
        target_state = @InFlight
      }> ({
      }) : () -> ()
      "metreon.transition_decl"() <{
        context_requirements = [],
        effects = [],
        function_type = (!metreon.resource<@Slot::@InFlight, []>) -> !metreon.resource<@Slot::@Done, []>,
        generic_parameters = [],
        is_await = true,
        parameter_attrs = [{name = "self", ownership = "own", source_type = {arguments = [], name = "InFlight"}, type = !metreon.resource<@Slot::@InFlight, []>}],
        result_attrs = [{ownership = "own", source_type = {arguments = [], name = "Done"}, type = !metreon.resource<@Slot::@Done, []>}],
        self_index = 0 : i64,
        source_state = @InFlight,
        sym_name = "wait_done",
        target_state = @Done
      }> ({
      }) : () -> ()
    }) : () -> ()
  )";

  // 1. Positive: structured `scf.if` mutually exclusive resource completion +
  // dominating `async_wait` across both branches + workgroup `gpu.barrier` +
  // `affine.for` invariant resource `iter_arg`!
  expectMLIRAccepted(
      "module {\n" + resourceDecls + R"(
      func.func @valid_pipeline(%cond: i1, %idx: index,
                                %gsrc: memref<64xf32, 1>,
                                %sdst: memref<64xf32, 3>,
                                %slot0: !metreon.resource<@Slot::@Vacant, []>)
          -> f32 attributes {metreon.context = @block_ctx} {
        %inflight = "metreon.transition"(%slot0) <{
          callee = @Slot::@issue,
          context = @block_ctx,
          type_arguments = []
        }> : (!metreon.resource<@Slot::@Vacant, []>) -> !metreon.resource<@Slot::@InFlight, []>

        "metreon.async_copy"(%gsrc, %sdst) <{
          context = @block_ctx,
          generation = "0",
          slot = "smem_0"
        }> : (memref<64xf32, 1>, memref<64xf32, 3>) -> ()

        %done = scf.if %cond -> (!metreon.resource<@Slot::@Done, []>) {
          "metreon.async_wait"() <{
            context = @block_ctx,
            generation = "0",
            slot = "smem_0"
          }> : () -> ()
          %d1 = "metreon.transition"(%inflight) <{
            callee = @Slot::@wait_done,
            context = @block_ctx,
            type_arguments = []
          }> : (!metreon.resource<@Slot::@InFlight, []>) -> !metreon.resource<@Slot::@Done, []>
          scf.yield %d1 : !metreon.resource<@Slot::@Done, []>
        } else {
          "metreon.async_wait"() <{
            context = @block_ctx,
            generation = "0",
            slot = "smem_0"
          }> : () -> ()
          %d2 = "metreon.transition"(%inflight) <{
            callee = @Slot::@wait_done,
            context = @block_ctx,
            type_arguments = []
          }> : (!metreon.resource<@Slot::@InFlight, []>) -> !metreon.resource<@Slot::@Done, []>
          scf.yield %d2 : !metreon.resource<@Slot::@Done, []>
        }

        gpu.barrier

        %carried_done = affine.for %k = 0 to 4 iter_args(%cur = %done) -> (!metreon.resource<@Slot::@Done, []>) {
          "metreon.async_copy"(%gsrc, %sdst) <{
            context = @block_ctx,
            generation = "iter",
            slot = "smem_loop"
          }> : (memref<64xf32, 1>, memref<64xf32, 3>) -> ()
          "metreon.async_wait"() <{
            context = @block_ctx,
            generation = "iter",
            slot = "smem_loop"
          }> : () -> ()
          affine.yield %cur : !metreon.resource<@Slot::@Done, []>
        }

        %val = memref.load %sdst[%idx] : memref<64xf32, 3>
        return %val : f32
      }
      })",
      context);

  // 2. Negative: Double consumption on MLIR `!metreon.resource`.
  expectMLIRRejected(
      "module {\n" + resourceDecls + R"(
      func.func @bad_double(%s: !metreon.resource<@Slot::@Vacant, []>)
          attributes {metreon.context = @block_ctx} {
        %f1 = "metreon.transition"(%s) <{callee = @Slot::@issue, context = @block_ctx, type_arguments = []}>
            : (!metreon.resource<@Slot::@Vacant, []>) -> !metreon.resource<@Slot::@InFlight, []>
        %f2 = "metreon.transition"(%s) <{callee = @Slot::@issue, context = @block_ctx, type_arguments = []}>
            : (!metreon.resource<@Slot::@Vacant, []>) -> !metreon.resource<@Slot::@InFlight, []>
        %d1 = "metreon.transition"(%f1) <{callee = @Slot::@wait_done, context = @block_ctx, type_arguments = []}>
            : (!metreon.resource<@Slot::@InFlight, []>) -> !metreon.resource<@Slot::@Done, []>
        %d2 = "metreon.transition"(%f2) <{callee = @Slot::@wait_done, context = @block_ctx, type_arguments = []}>
            : (!metreon.resource<@Slot::@InFlight, []>) -> !metreon.resource<@Slot::@Done, []>
        return
      }
      })",
      context, "double consumption");

  // 3. Negative: Branch lifetime leak (`scf.if` completes `InFlight` on `then` only).
  expectMLIRRejected(
      "module {\n" + resourceDecls + R"(
      func.func @bad_branch(%cond: i1, %f: !metreon.resource<@Slot::@InFlight, []>)
          attributes {metreon.context = @block_ctx} {
        scf.if %cond {
          %d = "metreon.transition"(%f) <{callee = @Slot::@wait_done, context = @block_ctx, type_arguments = []}>
              : (!metreon.resource<@Slot::@InFlight, []>) -> !metreon.resource<@Slot::@Done, []>
        }
        return
      }
      })",
      context, "branch lifetime leak");

  // 4. Negative: Pre-wait buffer access (`affine.load` before `metreon.async_wait`).
  expectMLIRRejected(
      "module {\n" + resourceDecls + R"(
      func.func @bad_pre_wait(%gsrc: memref<64xf32, 1>, %sdst: memref<64xf32, 3>) -> f32
          attributes {metreon.context = @block_ctx} {
        "metreon.async_copy"(%gsrc, %sdst) <{context = @block_ctx, generation = "0", slot = "s0"}>
            : (memref<64xf32, 1>, memref<64xf32, 3>) -> ()
        %c0 = arith.constant 0 : index
        %v = affine.load %sdst[%c0] : memref<64xf32, 3>
        "metreon.async_wait"() <{context = @block_ctx, generation = "0", slot = "s0"}> : () -> ()
        return %v : f32
      }
      })",
      context, "pre-wait buffer access");

  // 5. Negative: Stale-generation completion reuse (`async_wait` twice on gen "0").
  expectMLIRRejected(
      "module {\n" + resourceDecls + R"(
      func.func @bad_stale_gen(%gsrc: memref<64xf32, 1>, %sdst: memref<64xf32, 3>)
          attributes {metreon.context = @block_ctx} {
        "metreon.async_copy"(%gsrc, %sdst) <{context = @block_ctx, generation = "0", slot = "s0"}>
            : (memref<64xf32, 1>, memref<64xf32, 3>) -> ()
        "metreon.async_wait"() <{context = @block_ctx, generation = "0", slot = "s0"}> : () -> ()
        "metreon.async_wait"() <{context = @block_ctx, generation = "0", slot = "s0"}> : () -> ()
        return
      }
      })",
      context, "stale async completion reuse");

  // 6. Negative: Unawaited `must_complete` resource at callable exit.
  expectMLIRRejected(
      "module {\n" + resourceDecls + R"(
      func.func @bad_unawaited(%s: !metreon.resource<@Slot::@Vacant, []>)
          attributes {metreon.context = @block_ctx} {
        %f = "metreon.transition"(%s) <{callee = @Slot::@issue, context = @block_ctx, type_arguments = []}>
            : (!metreon.resource<@Slot::@Vacant, []>) -> !metreon.resource<@Slot::@InFlight, []>
        return
      }
      })",
      context, "unawaited must_complete");

  // 7. Negative: Loop backedge async loan leak inside `scf.for`.
  expectMLIRRejected(
      "module {\n" + resourceDecls + R"(
      func.func @bad_loop_loan(%gsrc: memref<64xf32, 1>, %sdst: memref<64xf32, 3>)
          attributes {metreon.context = @block_ctx} {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %c2 step %c1 {
          "metreon.async_copy"(%gsrc, %sdst) <{context = @block_ctx, generation = "0", slot = "s0"}>
              : (memref<64xf32, 1>, memref<64xf32, 3>) -> ()
        }
        return
      }
      })",
      context, "loop backedge lifetime leak");

  // 8. Negative: Cross-space dereference (device reading rank-1 host default memref).
  expectMLIRRejected(
      "module {\n" + resourceDecls + R"(
      func.func @bad_cross_space(%hbuf: memref<64xf32>, %idx: index) -> f32
          attributes {metreon.context = @block_ctx} {
        %v = memref.load %hbuf[%idx] : memref<64xf32>
        return %v : f32
      }
      })",
      context, "cross-space dereference");

  // 9. Negative: `gpu.barrier` invoked under `Gpu::Thread` scope!
  expectMLIRRejected(
      R"(
      module {
        "metreon.context"() <{
          generic_parameters = [],
          grants = [{arguments = [], name = "gpu_load"}],
          identifier = "thr_ctx",
          nominal_type = {arguments = [], name = "Gpu::Thread"},
          resolution = "declared",
          sym_name = "thr_ctx"
        }> : () -> ()
        func.func @bad_thread_barrier() attributes {metreon.context = @thr_ctx} {
          gpu.barrier
          return
        }
      })",
      context, "Workgroup");
}

} // namespace

int main(int argc, char **argv) {
  try {
    check(argc == 2, "expected vector_add.mtr fixture path");
    testUnifiedStructuredLowering(argv[1]);
    testMLIRStructuredOwnershipAndAsyncLoans();
    std::cout << "GPU Phase 2 MLIR structured lowering, ownership, and async loan tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "GPUMLIRVerificationTests failed: " << error.what() << '\n';
    return 1;
  }
}
