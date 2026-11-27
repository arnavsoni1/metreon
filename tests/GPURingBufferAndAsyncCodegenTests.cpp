#include "metreon/CodeGen/AOT.h"
#include "metreon/CodeGen/Toolchain.h"
#include "metreon/GraphIR/Behavioral.h"
#include "metreon/GraphIR/Lowering.h"
#include "metreon/GraphIR/Passes.h"
#include "metreon/MLIR/DeviceExecution.h"
#include "metreon/MLIR/Dialect.h"
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
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
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

void registerDialects(mlir::MLIRContext &context) {
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

graphir::StageGraph *findStage(graphir::Module &mod, const std::string &name) {
  for (auto &s : mod.stageGraphs()) {
    if (s.name() == name) return &s;
  }
  return nullptr;
}

bool canAssembleCUDA() {
  const std::string ptxas(toolchain::ptxas);
  return !ptxas.empty() && !ptxas.ends_with("-NOTFOUND") &&
         std::filesystem::exists(ptxas);
}

bool canEmitAndLinkAMDGPU() {
  const std::string ockl(toolchain::ockl);
  const std::string abi500(toolchain::abi500);
  const std::string lld(toolchain::amdLinker);
  return !ockl.empty() && std::filesystem::exists(ockl) && !abi500.empty() &&
         std::filesystem::exists(abi500) && !lld.empty() &&
         !lld.ends_with("-NOTFOUND") && std::filesystem::exists(lld);
}

struct CompiledTargets {
  std::string gpuMLIR;
  std::string nvvmMLIR;
  std::string ptx;
  std::string cubin;
  std::string rocdlMLIR;
  std::string amdAsm;
  std::string hsaco;
};

CompiledTargets compileBothGPUBackends(mlir::ModuleOp semantic,
                                       mlir::MLIRContext &context) {
  CompiledTargets out;

  // Round-trip semantic Metreon MLIR before lowering.
  auto parsedSem =
      mlir::parseSourceString<mlir::ModuleOp>(printModule(semantic), &context);
  check(static_cast<bool>(parsedSem) && mlir::succeeded(mlir::verify(*parsedSem)),
        "semantic Metreon MLIR failed to round-trip");

  check(mlir::succeeded(lowerMetreonDeviceToGPU(*parsedSem)),
        "lowerMetreonDeviceToGPU failed");
  check(mlir::succeeded(mlir::verify(*parsedSem)),
        "lowered GPU MLIR failed mlir::verify");

  // Acceptance gate: zero residual MetreonDialect ops, metreon.* attributes, or cf ops!
  parsedSem->walk([](mlir::Operation *op) {
    const auto ns = op->getName().getDialectNamespace();
    check(ns != "metreon", "unresolved MetreonDialect op remained after lowering");
    check(ns != "cf", "unstructured cf op remained after GPU lowering");
    for (auto attr : op->getAttrs()) {
      check(!attr.getName().getValue().starts_with("metreon."),
            "residual metreon.* attribute remained after lowering");
    }
  });
  out.gpuMLIR = printModule(*parsedSem);

  // Round-trip lowered GPU MLIR!
  auto gpuRoundTrip =
      mlir::parseSourceString<mlir::ModuleOp>(out.gpuMLIR, &context);
  check(static_cast<bool>(gpuRoundTrip),
        "lowered GPU MLIR failed textual round-trip");

  // 1. NVIDIA NVVM -> LLVM IR (`llvm::verifyModule`) -> PTX -> offline `ptxas` `.cubin`!
  auto nvvm = lowerDeviceToNVVM(*gpuRoundTrip);
  check(mlir::succeeded(nvvm), "lowerDeviceToNVVM failed");
  (**nvvm).walk([](mlir::Operation *op) {
    const auto ns = op->getName().getDialectNamespace();
    check(ns == "builtin" || ns == "llvm" || ns == "nvvm",
          "unexpected dialect in NVVM output: " + ns.str());
  });
  out.nvvmMLIR = printModule(**nvvm);
  check(out.nvvmMLIR.find("__gpu_ring_") == std::string::npos &&
            out.nvvmMLIR.find("__gpu_async_") == std::string::npos,
        "unresolved GPU runtime adapter symbol left in NVVM output");

  auto ptx = emitDevicePTX(**nvvm, "sm_89");
  check(mlir::succeeded(ptx) && ptx->find(".target sm_89") != std::string::npos &&
            ptx->find(".entry") != std::string::npos,
        "emitDevicePTX failed or missing .entry");
  out.ptx = *ptx;

  if (canAssembleCUDA()) {
    out.cubin = assembleCUDACodeObject(out.ptx, "sm_89");
    check(out.cubin.size() > 64 &&
              out.cubin.substr(0, 4) == std::string("\x7f" "ELF", 4),
          "offline ptxas .cubin assembly produced invalid ELF");
  }

  // 2. AMD ROCDL -> LLVM IR (`llvm::verifyModule` + OCKL) -> AMDGPU ISA & `.hsaco`!
  auto rocdl = lowerDeviceToROCDL(*gpuRoundTrip);
  check(mlir::succeeded(rocdl), "lowerDeviceToROCDL failed");
  (**rocdl).walk([](mlir::Operation *op) {
    const auto ns = op->getName().getDialectNamespace();
    check(ns == "builtin" || ns == "llvm" || ns == "rocdl",
          "unexpected dialect in ROCDL output: " + ns.str());
  });
  out.rocdlMLIR = printModule(**rocdl);
  check(out.rocdlMLIR.find("__gpu_ring_") == std::string::npos &&
            out.rocdlMLIR.find("__gpu_async_") == std::string::npos,
        "unresolved GPU runtime adapter symbol left in ROCDL output");

  if (canEmitAndLinkAMDGPU()) {
    auto amdAsm = emitDeviceAMDGPU(**rocdl, "gfx1030", /*object=*/false);
    check(mlir::succeeded(amdAsm) &&
              amdAsm->find("amdgcn-amd-amdhsa--gfx1030") != std::string::npos &&
              amdAsm->find(".amdhsa_code_object_version 5") != std::string::npos,
          "emitDeviceAMDGPU assembly emission failed");
    out.amdAsm = *amdAsm;

    auto amdObj = emitDeviceAMDGPU(**rocdl, "gfx1030", /*object=*/true);
    check(mlir::succeeded(amdObj) && amdObj->size() > 64,
          "emitDeviceAMDGPU ELF object emission failed");
    out.hsaco = linkAMDGPUCodeObject(*amdObj);
    check(out.hsaco.size() > 64 &&
              out.hsaco.substr(0, 4) == std::string("\x7f" "ELF", 4) &&
              llvm::support::endian::read16le(out.hsaco.data() + 18) ==
                  llvm::ELF::EM_AMDGPU &&
              static_cast<unsigned char>(out.hsaco[7]) ==
                  llvm::ELF::ELFOSABI_AMDGPU_HSA &&
              static_cast<unsigned char>(out.hsaco[8]) ==
                  llvm::ELF::ELFABIVERSION_AMDGPU_HSA_V5,
          "offline ld.lld .hsaco linking produced invalid HSA v5 ELF");
  }

  return out;
}

void testWorkgroupRingBufferPipelineCodegen(const char *examplePath) {
  std::ifstream input(examplePath);
  check(static_cast<bool>(input), "cannot read protocols_and_pipelines.mtr");
  const std::string exampleSrc((std::istreambuf_iterator<char>(input)), {});

  mlir::MLIRContext context;
  registerDialects(context);

  // 1. Compile `protocols_and_pipelines.mtr` with explicit 2-tile producer/consumer
  //    stage bodies AND mutable stage local counter (verifying AMDGPU AS-5 private
  //    allocas + `llvm.zext` unsigned hardware IDs + acquire/release ring atomics!)
  //    under both unoptimized and optimized GraphIR flows!
  for (const bool opt : {false, true}) {
    parser::Parser p(exampleSrc);
    auto graph = graphir::lowerModule(p.parseModule(), "tiled_gemm.mtr");
    auto *loadA = findStage(graph, "TiledGemm::load_a");
    auto *loadB = findStage(graph, "TiledGemm::load_b");
    auto *mma = findStage(graph, "TiledGemm::mma");
    check(loadA && loadB && mma, "missing TiledGemm stages");

    auto aVal = loadA->makeOperation(graphir::CallableOperationKind::Literal, "",
                                     {48, 5});
    aVal.attributes = {{"type", "f32"}, {"kind", "floating"}, {"value", "2.5"}};
    loadA->setBody({
        aVal,
        graphir::makeChannelAction(
            *loadA, {graphir::ChannelDirection::Send, "out", "tile",
                     graphir::ChannelDeliveryMode::BlockingQueue, false, "void",
                     {aVal.id}, {}, {49, 5}}),
        graphir::makeChannelAction(
            *loadA, {graphir::ChannelDirection::Send, "out", "stop",
                     graphir::ChannelDeliveryMode::BlockingQueue, false, "void",
                     {}, {}, {50, 5}}),
    });

    auto bVal = loadB->makeOperation(graphir::CallableOperationKind::Literal, "",
                                     {58, 5});
    bVal.attributes = {{"type", "f32"}, {"kind", "floating"}, {"value", "3.5"}};
    loadB->setBody({
        bVal,
        graphir::makeChannelAction(
            *loadB, {graphir::ChannelDirection::Send, "out", "tile",
                     graphir::ChannelDeliveryMode::BlockingQueue, false, "void",
                     {bVal.id}, {}, {59, 5}}),
        graphir::makeChannelAction(
            *loadB, {graphir::ChannelDirection::Send, "out", "stop",
                     graphir::ChannelDeliveryMode::BlockingQueue, false, "void",
                     {}, {}, {60, 5}}),
    });

    // Give `mma` a mutable local accumulator slot so AMDGPU private AS-5 slot
    // lowering is exercised inside a warp-specialized stage!
    auto accSlot = mma->makeOperation(graphir::CallableOperationKind::Variable,
                                      "acc", {66, 5});
    accSlot.attributes = {{"initialized", "true"},
                          {"mutability", "mutable"},
                          {"mutable_local", "true"},
                          {"storage", "automatic"},
                          {"type", "f32"},
                          {"initializer.kind", "floating"},
                          {"initializer.value", "0.0"}};
    auto recvA = graphir::makeChannelAction(
        *mma, {graphir::ChannelDirection::Receive, "a_in", "tile",
               graphir::ChannelDeliveryMode::BlockingQueue, false, "f32", {}, {},
               {67, 5}});
    auto recvB = graphir::makeChannelAction(
        *mma, {graphir::ChannelDirection::Receive, "b_in", "tile",
               graphir::ChannelDeliveryMode::BlockingQueue, false, "f32", {}, {},
               {68, 5}});
    auto sumAB = mma->makeOperation(graphir::CallableOperationKind::Binary, "",
                                    {69, 5});
    sumAB.attributes = {{"type", "f32"}, {"operator", "+"}};
    sumAB.operands = {recvA.id, recvB.id};
    auto storeAcc = mma->makeOperation(graphir::CallableOperationKind::Assign,
                                       "", {70, 5});
    storeAcc.attributes = {{"type", "void"}};
    storeAcc.operands = {accSlot.id, sumAB.id};

    mma->setBody({
        accSlot,
        recvA,
        recvB,
        sumAB,
        storeAcc,
        graphir::makeChannelAction(
            *mma, {graphir::ChannelDirection::Receive, "a_in", "stop",
                   graphir::ChannelDeliveryMode::BlockingQueue, false, "void",
                   {}, {}, {71, 5}}),
        graphir::makeChannelAction(
            *mma, {graphir::ChannelDirection::Receive, "b_in", "stop",
                   graphir::ChannelDeliveryMode::BlockingQueue, false, "void",
                   {}, {}, {72, 5}}),
        graphir::makeProgressMarkerOperation(
            *mma, {"mma_done", "finished", {}, {73, 5}}),
    });

    if (opt) graphir::passes::runDefaultPipeline(graph);

    auto imported = importGPUPipelineExecution(graph, context);
    check(mlir::succeeded(imported),
          "importGPUPipelineExecution failed on TiledGemm");
    const auto manifest = deviceLaunchManifest(**imported);
    check(manifest.getInteger("index_bits") == 64 &&
              manifest.getArray("kernels") &&
              manifest.getArray("kernels")->size() == 1,
          "pipeline launch manifest missing fused kernel entry");

    const CompiledTargets res = compileBothGPUBackends(**imported, context);
    check(res.gpuMLIR.find("memref.global \"private\" @__ring_") !=
                  std::string::npos &&
              res.gpuMLIR.find("gpu.barrier") != std::string::npos &&
              res.gpuMLIR.find("gpu.shuffle") != std::string::npos,
          "lowered GPU MLIR missing shared ring globals, barrier, or warp shuffle");

    check(res.nvvmMLIR.find("llvm.atomicrmw") != std::string::npos &&
              res.nvvmMLIR.find("fence.acq_rel.cta") != std::string::npos,
          "NVVM output missing fence.acq_rel.cta or atomicrmw on ring queue");
    check(res.ptx.find(".shared") != std::string::npos &&
              res.ptx.find("atom.shared") != std::string::npos &&
              res.ptx.find("fence.acq_rel.cta") != std::string::npos &&
              res.ptx.find("bar.sync") != std::string::npos,
          "PTX output missing .shared ring storage, atom.shared, fence.acq_rel.cta, or bar.sync");

    check(res.rocdlMLIR.find("syncscope(\"workgroup\")") != std::string::npos &&
              res.rocdlMLIR.find("llvm.zext") != std::string::npos &&
              res.rocdlMLIR.find("ptr<5>") != std::string::npos,
          "ROCDL output missing workgroup syncscope, unsigned ID zext, or private AS-5 slot");
    if (!res.amdAsm.empty()) {
      check(res.amdAsm.find("s_barrier") != std::string::npos,
            "AMDGPU assembly missing s_barrier");
    }
  }

  // 2. Credit-seeded cyclic GPU pipeline (`seeded_items 1`) compiles cleanly to
  //    both PTX/.cubin and ROCDL/.hsaco!
  const std::string seededRingSrc = R"(
protocol DataFlow roles prod, cons {
  state Streaming { prod -> cons: item(v: f32) => Streaming; }
}
protocol CreditFlow roles issuer, holder {
  state Ready { issuer -> holder: grant => Ready; }
}
pipeline SeededRing(input in_buf: gpu::span<f32>, output out_buf: gpu::span<f32>) {
  edge data_q: channel<DataFlow> {
    initial Streaming;
    capacity 2;
    storage gpu::shared;
    scope block;
    backpressure cooperative;
    publication acq_rel_workgroup;
  }
  edge credit_q: channel<CreditFlow> {
    initial Ready;
    capacity 2;
    seeded_items 1;
    storage gpu::shared;
    scope block;
    backpressure cooperative;
    publication acq_rel_workgroup;
  }
  stage prod(src: gpu::span<f32>, c_in: CreditFlow::holder<Ready>, d_out: DataFlow::prod<Streaming>) -> void { return; }
  stage cons(d_in: DataFlow::cons<Streaming>, c_out: CreditFlow::issuer<Ready>, dst: gpu::span<f32>) -> void { return; }
  topology {
    in_buf     => prod.src;
    prod.d_out =>[data_q] cons.d_in;
    cons.c_out =>[credit_q] prod.c_in;
    cons.dst   => out_buf;
  }
  schedule sm89 {
    kernel worker = fuse(prod, cons) { blocks 1; }
    place prod on worker.warp[0];
    place cons on worker.warp[1];
  }
}
)";
  {
    parser::Parser p(seededRingSrc);
    auto graph = graphir::lowerModule(p.parseModule(), "seeded_ring_codegen.mtr");
    auto *prod = findStage(graph, "SeededRing::prod");
    auto *cons = findStage(graph, "SeededRing::cons");
    prod->setBody({
        graphir::makeChannelAction(
            *prod, {graphir::ChannelDirection::Receive, "c_in", "grant",
                    graphir::ChannelDeliveryMode::BlockingQueue, false, "void",
                    {}, {}, {10, 1}}),
        graphir::makeChannelAction(
            *prod, {graphir::ChannelDirection::Send, "d_out", "item",
                    graphir::ChannelDeliveryMode::BlockingQueue, false, "void",
                    {}, {}, {11, 1}}),
    });
    cons->setBody({
        graphir::makeChannelAction(
            *cons, {graphir::ChannelDirection::Receive, "d_in", "item",
                    graphir::ChannelDeliveryMode::BlockingQueue, false, "f32",
                    {}, {}, {20, 1}}),
        graphir::makeChannelAction(
            *cons, {graphir::ChannelDirection::Send, "c_out", "grant",
                    graphir::ChannelDeliveryMode::BlockingQueue, false, "void",
                    {}, {}, {21, 1}}),
    });
    graphir::passes::runDefaultPipeline(graph);
    auto imported = importGPUPipelineExecution(graph, context);
    check(mlir::succeeded(imported), "seeded ring import failed");
    (void)compileBothGPUBackends(**imported, context);
  }
}

void testAsyncCopyAndWaitCodegenAndNegativeGates() {
  mlir::MLIRContext context;
  registerDialects(context);

  // 1. Positive: `Gpu::Block` generation-tracked `metreon.async_copy` (global ->
  //    shared `memref<16xf32, 3>`) + `metreon.async_wait` followed by disjoint
  //    post-wait shared load -> global store!
  //    Verifies `nvvm.cp.async.shared.global`, `nvvm.cp.async.commit.group`,
  //    and `nvvm.cp.async.wait.group 0` (`cp.async.ca.shared.global` in PTX +
  //    `.cubin`) AND AMDGPU cooperative LDS copy + workgroup fences + `.hsaco`!
  const std::string validAsyncMLIR = R"(
module {
  module @__metreon_bodies attributes {metreon.ordinary_lowering = "primary_with_context_intrinsics"} {
    memref.global "private" @tile_smem : memref<16xf32, 3> = uninitialized
    "metreon.context"() <{
      generic_parameters = [],
      grants = [
        {arguments = [], name = "gpu_async_copy"},
        {arguments = [], name = "gpu_await"},
        {arguments = [], name = "gpu_barrier"},
        {arguments = [], name = "gpu_load"},
        {arguments = [], name = "gpu_store"}
      ],
      identifier = "blk_ctx",
      nominal_type = {arguments = [], name = "Gpu::Block"},
      resolution = "declared",
      sym_name = "blk_ctx"
    }> : () -> ()
    func.func @async_tile_copy(%src: memref<?xf32, 1>, %dst: memref<?xf32, 1>) {
      %smem = memref.get_global @tile_smem : memref<16xf32, 3>
      "metreon.async_copy"(%src, %smem) <{
        context = @blk_ctx,
        generation = "gen0",
        slot = "slot0"
      }> : (memref<?xf32, 1>, memref<16xf32, 3>) -> ()
      "metreon.async_wait"() <{
        context = @blk_ctx,
        generation = "gen0",
        slot = "slot0"
      }> : () -> ()
      %tid = "metreon.device_intrinsic"() <{context = @blk_ctx, kind = "thread_id"}> : () -> index
      %gid = "metreon.device_intrinsic"() <{context = @blk_ctx, kind = "global_id"}> : () -> index
      %c16 = arith.constant 16 : index
      %in_tile = arith.cmpi ult, %tid, %c16 : index
      scf.if %in_tile {
        %val = memref.load %smem[%tid] : memref<16xf32, 3>
        "metreon.device_intrinsic"(%dst, %gid, %val) <{context = @blk_ctx, kind = "store"}> : (memref<?xf32, 1>, index, f32) -> ()
      }
      return
    }
  }
  "metreon.kernel"() <{
    body_ref = @__metreon_bodies::@async_tile_copy,
    context = @__metreon_bodies::@blk_ctx,
    context_requirements = [],
    deferred_body = [],
    deferred_cfg = [],
    effects = [],
    execution_domain = "Gpu",
    function_type = (memref<?xf32, 1>, memref<?xf32, 1>) -> (),
    parameter_attrs = [
      {name = "src", ownership = "none", source_type = {arguments = [{arguments = [], name = "f32"}], name = "gpu::span"}, type = memref<?xf32, 1>},
      {name = "dst", ownership = "none", source_type = {arguments = [{arguments = [], name = "f32"}], name = "gpu::span"}, type = memref<?xf32, 1>}
    ],
    result_attrs = [],
    sym_name = "async_tile_copy"
  }> : () -> ()
}
)";
  {
    auto mod = mlir::parseSourceString<mlir::ModuleOp>(validAsyncMLIR, &context);
    check(static_cast<bool>(mod) && mlir::succeeded(mlir::verify(*mod)),
          "failed to parse validAsyncMLIR");
    const CompiledTargets res = compileBothGPUBackends(*mod, context);
    check(res.nvvmMLIR.find("nvvm.cp.async.shared.global") != std::string::npos &&
              res.nvvmMLIR.find("nvvm.cp.async.commit.group") !=
                  std::string::npos &&
              res.nvvmMLIR.find("nvvm.cp.async.wait.group") !=
                  std::string::npos,
          "NVVM output missing cp.async intrinsics");
    check(res.ptx.find("cp.async.ca.shared.global") != std::string::npos &&
              res.ptx.find("cp.async.commit_group") != std::string::npos &&
              res.ptx.find("cp.async.wait_group") != std::string::npos,
          "PTX output missing cp.async.ca.shared.global / commit_group / wait_group");
    check(res.rocdlMLIR.find("syncscope(\"workgroup\")") != std::string::npos &&
              res.rocdlMLIR.find("rocdl.barrier") != std::string::npos &&
              res.rocdlMLIR.find("llvm.zext") != std::string::npos,
          "ROCDL async_copy output missing workgroup fences, barrier, or zext");
  }

  // 2. Positive: GraphIR `makeAsyncTokenOperation` (`AsyncIssue` + `AsyncWait`)
  //    through `importDeviceExecution` -> `lowerMetreonDeviceToGPU` -> NVVM & ROCDL!
  {
    const std::string asyncSrc = R"(
context Gpu::Block grants {gpu_load, gpu_store, gpu_barrier, gpu_async_copy, gpu_await} blk_ctx;
kernel stage_copy(@cx: blk_ctx; src: gpu::span<f32>, smem: gpu::shared_span<f32>) {
  index tid = gpu::thread_id(@cx);
}
)";
    parser::Parser p(asyncSrc);
    auto graph = graphir::lowerModule(p.parseModule(), "graphir_async.mtr", true);
    auto &k = graph.kernelGraphs()[0];
    const auto srcParamId = *k.parameters()[1].id;
    const auto dstParamId = *k.parameters()[2].id;
    auto body = k.body();
    body.push_back(graphir::makeAsyncTokenOperation(
        k, {graphir::AsyncActionKind::Issue,
            "slot0",
            "g0",
            "gpu::global",
            "gpu::shared",
            "void",
            {srcParamId, dstParamId},
            graph.contexts()[0],
            {3, 3}}));
    body.push_back(graphir::makeAsyncTokenOperation(
        k, {graphir::AsyncActionKind::Wait,
            "slot0",
            "g0",
            "gpu::global",
            "gpu::shared",
            "void",
            {},
            graph.contexts()[0],
            {4, 3}}));
    k.setBody(std::move(body));

    auto imported = importDeviceExecution(graph, context);
    check(mlir::succeeded(imported),
          "GraphIR AsyncIssue + AsyncWait importDeviceExecution failed");
    (void)compileBothGPUBackends(**imported, context);
  }

  // 3. Negative Gate A: Pre-wait read on in-flight `metreon.async_copy` destination
  //    is rejected BEFORE GPU codegen!
  {
    std::string badPreWait = validAsyncMLIR;
    const std::string needle = R"("metreon.async_wait"() <{)";
    badPreWait.replace(
        badPreWait.find(needle), 0,
        "%c0_pre = arith.constant 0 : index\n"
        "%racy = memref.load %smem[%c0_pre] : memref<16xf32, 3>\n");
    auto mod = mlir::parseSourceString<mlir::ModuleOp>(badPreWait, &context);
    check(static_cast<bool>(mod), "failed to parse badPreWait");
    std::string diag;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &d) {
      llvm::raw_string_ostream os(diag);
      d.print(os);
      os.flush();
      return mlir::success();
    });
    const auto res = lowerMetreonDeviceToGPU(*mod);
    check(mlir::failed(res) && diag.find("pre-wait") != std::string::npos,
          "pre-wait read of async_copy destination must fail before GPU codegen, got: " +
              diag);
  }

  // 4. Negative Gate B: Unseeded cyclic pipeline deadlock (`gpu.queue.deadlock`)
  //    and `relaxed` publication (`gpu.memory.insufficient_publication_scope`)
  //    are rejected BEFORE ring-buffer GPU codegen!
  {
    const std::string badRelaxedPipe = R"(
protocol Flow roles p, c { state S { p -> c: m(v: f32) => S; } }
pipeline BadPub(input x: f32, output y: f32) {
  edge q: channel<Flow> {
    initial S;
    capacity 1;
    storage gpu::shared;
    scope block;
    backpressure cooperative;
    publication relaxed;
  }
  stage s1(i: f32, o: Flow::p<S>) -> void { return; }
  stage s2(i: Flow::c<S>, o: f32) -> void { return; }
  topology { x => s1.i; s1.o =>[q] s2.i; s2.o => y; }
  schedule sm89 {
    kernel w = fuse(s1, s2) { blocks 1; }
    place s1 on w.warp[0];
    place s2 on w.warp[1];
  }
}
)";
    parser::Parser p(badRelaxedPipe);
    auto g = graphir::lowerModule(p.parseModule(), "bad_pub_codegen.mtr");
    std::string diag;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &d) {
      llvm::raw_string_ostream os(diag);
      d.print(os);
      os.flush();
      return mlir::success();
    });
    auto imp = importGPUPipelineExecution(g, context);
    check(mlir::failed(imp) &&
              diag.find("gpu.memory.insufficient_publication_scope") !=
                  std::string::npos,
          "relaxed publication pipeline must fail before GPU codegen, got: " +
              diag);

    const std::string badDeadlockPipe = R"(
protocol Flow2 roles p, c { state S { p -> c: m(v: f32) => S; } }
pipeline TwoSendCap1Deadlock(input x: f32, output y: f32) {
  edge q: channel<Flow2> {
    initial S;
    capacity 1;
    storage gpu::shared;
    scope block;
    backpressure cooperative;
  }
  stage s1(i: f32, o: Flow2::p<S>) -> void { return; }
  stage s2(i: Flow2::c<S>, o: f32) -> void { return; }
  topology { x => s1.i; s1.o =>[q] s2.i; s2.o => y; }
  schedule sm89 {
    kernel w = fuse(s1, s2) { blocks 1; }
    place s1 on w.warp[0];
    place s2 on w.warp[1];
  }
}
)";
    parser::Parser pd(badDeadlockPipe);
    auto gd = graphir::lowerModule(pd.parseModule(), "bad_deadlock_codegen.mtr");
    auto *s1 = findStage(gd, "TwoSendCap1Deadlock::s1");
    check(s1 != nullptr, "missing TwoSendCap1Deadlock::s1");
    auto lit = s1->makeOperation(graphir::CallableOperationKind::Literal, "", {11, 5});
    lit.attributes = {{"type", "f32"}, {"kind", "floating"}, {"value", "1.0"}};
    s1->setBody({
        lit,
        graphir::makeChannelAction(
            *s1, {graphir::ChannelDirection::Send, "o", "m",
                  graphir::ChannelDeliveryMode::BlockingQueue, false, "void",
                  {lit.id}, {}, {11, 10}}),
        graphir::makeChannelAction(
            *s1, {graphir::ChannelDirection::Send, "o", "m",
                  graphir::ChannelDeliveryMode::BlockingQueue, false, "void",
                  {lit.id}, {}, {11, 20}}),
    });

    diag.clear();
    auto impDead = importGPUPipelineExecution(gd, context);
    check(mlir::failed(impDead) &&
              diag.find("gpu.queue.deadlock") != std::string::npos,
          "two-send capacity-1 deadlock must fail before GPU codegen, got: " +
              diag);
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    check(argc == 2, "expected protocols_and_pipelines.mtr fixture path");
    testWorkgroupRingBufferPipelineCodegen(argv[1]);
    testAsyncCopyAndWaitCodegenAndNegativeGates();
    std::cout << "GPU Phase 5 workgroup ring-buffer and async-copy NVVM/ROCDL codegen tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "GPURingBufferAndAsyncCodegenTests failed: " << error.what()
              << '\n';
    return 1;
  }
}
