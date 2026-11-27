#include "metreon/GraphIR/Behavioral.h"
#include "metreon/GraphIR/Lowering.h"
#include "metreon/GraphIR/Passes.h"
#include "metreon/MLIR/Dialect.h"
#include "metreon/MLIR/GPUVerification.h"
#include "metreon/MLIR/Import.h"
#include "metreon/Parser/Parser.h"

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

graphir::StageGraph *findStage(graphir::Module &mod, const std::string &name) {
  for (auto &s : mod.stageGraphs()) {
    if (s.name() == name) return &s;
  }
  return nullptr;
}

mlir::OwningOpRef<mlir::ModuleOp>
importAndRoundTrip(const graphir::Module &graph, mlir::MLIRContext &context) {
  context.getOrLoadDialect<MetreonDialect>();
  auto imported = importGraphIR(graph, context);
  check(mlir::succeeded(imported), "importGraphIR failed on GPU-4 pipeline test");
  auto parsed =
      mlir::parseSourceString<mlir::ModuleOp>(printModule(**imported), &context);
  check(static_cast<bool>(parsed) && mlir::succeeded(mlir::verify(*parsed)),
        "round-tripped Metreon MLIR pipeline module is invalid");
  return parsed;
}

void expectMLIRPipelineAccepted(
    mlir::ModuleOp module, mlir::MLIRContext &context,
    const verification::GPUBoundedQueueOptions &opts = {}) {
  std::string diagText;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &d) {
    llvm::raw_string_ostream os(diagText);
    d.print(os);
    os.flush();
    return mlir::success();
  });
  const mlir::LogicalResult res = verifyGPUPipelineProtocols(module, opts);
  check(mlir::succeeded(res),
        "expected MLIR pipeline to pass GPU-4 verification, got: " + diagText);
}

void expectMLIRPipelineRejected(
    mlir::ModuleOp module, mlir::MLIRContext &context,
    const std::string &expectedSubstr,
    const verification::GPUBoundedQueueOptions &opts = {}) {
  std::string diagText;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &d) {
    llvm::raw_string_ostream os(diagText);
    d.print(os);
    os.flush();
    return mlir::success();
  });
  const mlir::LogicalResult res = verifyGPUPipelineProtocols(module, opts);
  check(mlir::failed(res),
        "expected MLIR pipeline to fail GPU-4 verification (" + expectedSubstr +
            "), but it succeeded");
  check(diagText.find(expectedSubstr) != std::string::npos,
        "expected MLIR diagnostic containing `" + expectedSubstr +
            "`, got: " + diagText);
}

void testMLIRPipelineProtocolVerification(const char *examplePath) {
  std::ifstream input(examplePath);
  check(static_cast<bool>(input), "cannot read protocols_and_pipelines.mtr");
  const std::string exampleSrc((std::istreambuf_iterator<char>(input)), {});

  mlir::MLIRContext context;
  context.getOrLoadDialect<MetreonDialect>();

  // 1. Positive: `protocols_and_pipelines.mtr` imported & round-tripped through
  //    textual Metreon MLIR passes `verifyGPUPipelineProtocols`!
  for (const bool opt : {false, true}) {
    parser::Parser p(exampleSrc);
    auto graph = graphir::lowerModule(p.parseModule(), "example.mtr");
    if (opt) graphir::passes::runDefaultPipeline(graph);
    auto mlirMod = importAndRoundTrip(graph, context);
    expectMLIRPipelineAccepted(*mlirMod, context);
  }

  // 2. Positive: Credit-seeded cyclic GPU pipeline imported to Metreon MLIR!
  const std::string seededRingSrc = R"(
protocol DataFlow roles prod, cons {
  state Streaming { prod -> cons: item(v: f32) => Streaming; }
}
protocol CreditFlow roles issuer, holder {
  state Ready { issuer -> holder: grant => Ready; }
}
pipeline SeededRing(input x: f32, output y: f32) {
  edge data_q: channel<DataFlow> {
    initial Streaming;
    capacity 1;
    storage gpu::shared;
    scope block;
    backpressure cooperative;
    publication acq_rel_workgroup;
  }
  edge credit_q: channel<CreditFlow> {
    initial Ready;
    capacity 1;
    seeded_items 1;
    storage gpu::shared;
    scope block;
    backpressure cooperative;
    publication acq_rel_workgroup;
  }
  stage prod(src: f32, c_in: CreditFlow::holder<Ready>, d_out: DataFlow::prod<Streaming>) -> void { return; }
  stage cons(d_in: DataFlow::cons<Streaming>, c_out: CreditFlow::issuer<Ready>, dst: f32) -> void { return; }
  topology {
    x          => prod.src;
    prod.d_out =>[data_q] cons.d_in;
    cons.c_out =>[credit_q] prod.c_in;
    cons.dst   => y;
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
    auto graph = graphir::lowerModule(p.parseModule(), "seeded_ring.mtr");
    auto *prod = findStage(graph, "SeededRing::prod");
    auto *cons = findStage(graph, "SeededRing::cons");
    prod->setBody({
        graphir::makeChannelAction(*prod, {graphir::ChannelDirection::Receive,
                                           "c_in", "grant",
                                           graphir::ChannelDeliveryMode::BlockingQueue,
                                           false, "void", {}, {}, {10, 1}}),
        graphir::makeChannelAction(*prod, {graphir::ChannelDirection::Send,
                                           "d_out", "item",
                                           graphir::ChannelDeliveryMode::BlockingQueue,
                                           false, "void", {}, {}, {11, 1}}),
    });
    cons->setBody({
        graphir::makeChannelAction(*cons, {graphir::ChannelDirection::Receive,
                                           "d_in", "item",
                                           graphir::ChannelDeliveryMode::BlockingQueue,
                                           false, "void", {}, {}, {20, 1}}),
        graphir::makeChannelAction(*cons, {graphir::ChannelDirection::Send,
                                           "c_out", "grant",
                                           graphir::ChannelDeliveryMode::BlockingQueue,
                                           false, "void", {}, {}, {21, 1}}),
    });
    graphir::passes::runDefaultPipeline(graph);
    auto mlirMod = importAndRoundTrip(graph, context);
    expectMLIRPipelineAccepted(*mlirMod, context);

    // Tight solver limit on the exact same MLIR module reports `state_limit_exceeded`!
    verification::GPUBoundedQueueOptions tightOpts;
    tightOpts.maxExploredStates = 2;
    expectMLIRPipelineRejected(*mlirMod, context,
                               "gpu.solver.state_limit_exceeded", tightOpts);
  }

  // 3. Negative: Two-send capacity-1 deadlock caught directly on parsed Metreon MLIR!
  {
    std::string unseeded = seededRingSrc;
    unseeded.replace(unseeded.find("seeded_items 1;"), 15, "seeded_items 0;");
    parser::Parser p(unseeded);
    auto graph = graphir::lowerModule(p.parseModule(), "deadlock_ring.mtr");
    auto *prod = findStage(graph, "SeededRing::prod");
    auto *cons = findStage(graph, "SeededRing::cons");
    prod->setBody({
        graphir::makeChannelAction(*prod, {graphir::ChannelDirection::Send,
                                           "d_out", "item",
                                           graphir::ChannelDeliveryMode::BlockingQueue,
                                           false, "void", {}, {}, {10, 1}}),
        graphir::makeChannelAction(*prod, {graphir::ChannelDirection::Send,
                                           "d_out", "item",
                                           graphir::ChannelDeliveryMode::BlockingQueue,
                                           false, "void", {}, {}, {11, 1}}),
        graphir::makeChannelAction(*prod, {graphir::ChannelDirection::Receive,
                                           "c_in", "grant",
                                           graphir::ChannelDeliveryMode::BlockingQueue,
                                           false, "void", {}, {}, {12, 1}}),
    });
    cons->setBody({
        graphir::makeChannelAction(*cons, {graphir::ChannelDirection::Send,
                                           "c_out", "grant",
                                           graphir::ChannelDeliveryMode::BlockingQueue,
                                           false, "void", {}, {}, {20, 1}}),
        graphir::makeChannelAction(*cons, {graphir::ChannelDirection::Send,
                                           "c_out", "grant",
                                           graphir::ChannelDeliveryMode::BlockingQueue,
                                           false, "void", {}, {}, {21, 1}}),
        graphir::makeChannelAction(*cons, {graphir::ChannelDirection::Receive,
                                           "d_in", "item",
                                           graphir::ChannelDeliveryMode::BlockingQueue,
                                           false, "void", {}, {}, {22, 1}}),
    });
    auto mlirMod = importAndRoundTrip(graph, context);
    expectMLIRPipelineRejected(*mlirMod, context, "gpu.queue.deadlock");
  }

  // 4. Negative: Unscheduled multi-block (`blocks 2;`) and `relaxed` publication in MLIR!
  {
    std::string badBlocks = exampleSrc;
    badBlocks.replace(badBlocks.find("blocks 1;"), 9, "blocks 2;");
    parser::Parser p1(badBlocks);
    auto g1 = graphir::lowerModule(p1.parseModule(), "bad_blocks.mtr");
    auto m1 = importAndRoundTrip(g1, context);
    expectMLIRPipelineRejected(
        *m1, context, "gpu.residency.unscheduled_cross_block_blocking");

    std::string badPub = exampleSrc;
    badPub.replace(badPub.find("backpressure cooperative;"), 25,
                   "backpressure cooperative;\n        publication relaxed;");
    parser::Parser p2(badPub);
    auto g2 = graphir::lowerModule(p2.parseModule(), "bad_pub.mtr");
    auto m2 = importAndRoundTrip(g2, context);
    expectMLIRPipelineRejected(
        *m2, context, "gpu.memory.insufficient_publication_scope");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    check(argc == 2, "expected protocols_and_pipelines.mtr fixture path");
    testMLIRPipelineProtocolVerification(argv[1]);
    std::cout << "GPU Phase 4 MLIR pipeline protocol, residency, and memory-scope tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "GPUPipelineProtocolMLIRTests failed: " << error.what()
              << '\n';
    return 1;
  }
}
