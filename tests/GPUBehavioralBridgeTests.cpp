#include "metreon/Basic/Diagnostic.h"
#include "metreon/GraphIR/Behavioral.h"
#include "metreon/GraphIR/ControlFlow.h"
#include "metreon/GraphIR/Lowering.h"
#include "metreon/GraphIR/Passes.h"
#include "metreon/Parser/Parser.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace metreon;
using namespace metreon::graphir;

void check(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

Module parseAndLower(const std::string &source,
                     const char *name = "behavioral-bridge.mtr",
                     bool contextOnly = false) {
  parser::Parser parser(source);
  return lowerModule(parser.parseModule(), name, contextOnly);
}

template <typename Fn>
void expectDiagnostic(Fn &&fn, const std::string &expectedCode) {
  try {
    fn();
  } catch (const DiagnosticError &error) {
    check(error.diagnostic().code == expectedCode,
          "expected diagnostic `" + expectedCode + "`, got `" +
              error.diagnostic().code + "`: " + error.diagnostic().message);
    return;
  }
  throw std::runtime_error("expected diagnostic `" + expectedCode +
                           "` was not thrown");
}

template <typename Fn> void expectFailure(Fn &&fn, const std::string &label) {
  try {
    fn();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("expected failure did not occur: " + label);
}

void testBehavioralAndAsyncConstructs() {
  Module module("gpu_stage_behavior.mtr");
  const auto stageCtx = module.addContextMetadata(
      "Gpu::WarpStage<producer>", "warp_prod", 1, ContextResolution::Declared,
      {0, 1, 1});
  const auto loopCtx = module.addContextMetadata(
      "Gpu::Loop", "gpu_loop", 0, ContextResolution::Declared, {0, 2, 1});

  auto &stage = module.addKernelGraph("tile_producer", {0, 3, 1});
  stage.setSignature({{"return_type", "void"},
                      {"execution_domain", "Gpu"},
                      {"effects", ""},
                      {"where", ""}},
                     stageCtx);
  const auto tileParam =
      stage.addParameter("tile_val", "f32", {}, {0, 3, 10}).value();
  const auto readyA =
      stage.addParameter("ready_a", "bool", {}, {0, 3, 20}).value();
  const auto readyB =
      stage.addParameter("ready_b", "bool", {}, {0, 3, 30}).value();
  const auto coin = stage.addParameter("coin", "bool", {}, {0, 3, 40}).value();

  std::vector<CallableOperation> body;

  // Adjacent async issue + wait + channel send + progress marker share
  // stageCtx so ContextHoistingPass groups them into a ContextRegion.
  body.push_back(makeAsyncTokenOperation(
      stage, AsyncTokenSpec{AsyncActionKind::Issue,
                            "smem_slot_0",
                            "epoch_0",
                            "gpu::global",
                            "gpu::shared",
                            "void",
                            {tileParam},
                            stageCtx,
                            {0, 4, 1}}));
  body.push_back(makeAsyncTokenOperation(
      stage, AsyncTokenSpec{AsyncActionKind::Wait,
                            "smem_slot_0",
                            "epoch_0",
                            "gpu::global",
                            "gpu::shared",
                            "void",
                            {},
                            stageCtx,
                            {0, 5, 1}}));
  body.push_back(makeChannelAction(
      stage, ChannelActionSpec{ChannelDirection::Send,
                               "a_tiles",
                               "tile",
                               ChannelDeliveryMode::BlockingQueue,
                               true,
                               "void",
                               {tileParam},
                               stageCtx,
                               {0, 6, 1}}));
  body.push_back(makeProgressMarkerOperation(
      stage, ProgressMarkerSpec{"tile_published", "completed_tile_transfer",
                                stageCtx, {0, 7, 1}}));

  // Exclusive offer with two channel receive arms and a default continuation.
  auto recvArmA = makeChannelAction(
      stage, ChannelActionSpec{ChannelDirection::Receive,
                               "ctrl_a",
                               "ack",
                               ChannelDeliveryMode::SameStep,
                               false,
                               "void",
                               {},
                               stageCtx,
                               {0, 8, 1}});
  auto recvArmB = makeChannelAction(
      stage, ChannelActionSpec{ChannelDirection::Receive,
                               "ctrl_b",
                               "ack",
                               ChannelDeliveryMode::SameStep,
                               false,
                               "void",
                               {},
                               stageCtx,
                               {0, 9, 1}});
  auto defaultProgress = makeProgressMarkerOperation(
      stage, ProgressMarkerSpec{"idle_step", "polled_empty", stageCtx,
                                {0, 10, 1}});
  body.push_back(makeOfferOperation(
      stage,
      OfferSpec{OfferMode::Exclusive,
                {},
                {OfferBranchSpec{ChannelDirection::Receive,
                                 "ctrl_a",
                                 "ack",
                                 readyA,
                                 {std::move(recvArmA)},
                                 {0, 8, 1}},
                 OfferBranchSpec{ChannelDirection::Receive,
                                 "ctrl_b",
                                 "ack",
                                 readyB,
                                 {std::move(recvArmB)},
                                 {0, 9, 1}}},
                {std::move(defaultProgress)},
                stageCtx,
                {0, 8, 1}}));

  // Fair-tagged choice inside a loop with Gpu::Loop evidence kept disjoint from
  // execution authority.
  auto fairTrue = makeProgressMarkerOperation(
      stage,
      ProgressMarkerSpec{"arb_left", "selected_left_lane", stageCtx, {0, 11, 1}});
  auto fairFalse = makeProgressMarkerOperation(
      stage, ProgressMarkerSpec{"arb_right", "selected_right_lane", stageCtx,
                                {0, 12, 1}});
  auto fairOp = makeFairChoiceOperation(
      stage, FairChoiceSpec{"arbiter_eta",
                            false,
                            {},
                            coin,
                            {std::move(fairTrue)},
                            {std::move(fairFalse)},
                            {0, 11, 1}});

  auto loopOp = stage.makeOperation(CallableOperationKind::While, "", {0, 13, 1});
  loopOp.loopContext = loopCtx;
  loopOp.operands = {readyA};
  loopOp.body = {std::move(fairOp)};
  body.push_back(std::move(loopOp));

  stage.setBody(std::move(body));

  validateBehavioralGraphIR(module);
  const auto snapshot = capturePreservationSnapshot(module);

  passes::ContextHoistingPass{}.run(module);
  check(stage.body().front().kind == CallableOperationKind::ContextRegion &&
            stage.body().front().body.size() == 4 &&
            !stage.body().front().body[0].context,
        "ContextHoistingPass did not group adjacent async/channel/progress "
        "actions into a ContextRegion");

  verifyPassPreservation(snapshot, module);

  const auto cfg = buildControlFlow(stage);
  bool sawOfferBranch = false;
  bool sawOfferDefault = false;
  bool sawFairThen = false;
  bool sawFairElse = false;
  for (const auto &block : cfg.blocks) {
    const auto roleIt = block.attributes.find("role");
    if (roleIt == block.attributes.end()) continue;
    if (roleIt->second == "offer.branch") sawOfferBranch = true;
    if (roleIt->second == "offer.default") sawOfferDefault = true;
    if (roleIt->second == "fair.then") {
      sawFairThen = block.attributes.at("fair_tag") == "arbiter_eta" &&
                    block.attributes.at("assumption_justified") == "false";
    }
    if (roleIt->second == "fair.else") {
      sawFairElse = block.attributes.at("fair_tag") == "arbiter_eta";
    }
  }
  check(sawOfferBranch && sawOfferDefault && sawFairThen && sawFairElse,
        "CFG projection lost offer branches, default block, or fairness tags");

  const std::string text = module.print();
  check(text.find("async_issue") != std::string::npos &&
            text.find("async_wait") != std::string::npos &&
            text.find("channel_send") != std::string::npos &&
            text.find("progress_marker") != std::string::npos &&
            text.find("fair:arbiter_eta") != std::string::npos,
        "serialized GraphIR CFG omitted behavioral or async actions");

  // Negative behavioral validation checks.
  {
    Module bad("bad_channel.mtr");
    auto &fn = bad.addProcedureGraph("p", {});
    fn.setSignature({{"return_type", "void"}}, {});
    auto op = fn.makeOperation(CallableOperationKind::ChannelSend, "x", {});
    op.attributes = {{"type", "void"}, {"endpoint", ""}, {"message", "m"},
                     {"direction", "send"}, {"delivery", "blocking_queue"}};
    fn.setBody({std::move(op)});
    expectDiagnostic([&] { validateBehavioralGraphIR(bad); },
                     "graphir.invalid_channel_action");
  }
  {
    Module bad("bad_async.mtr");
    auto &fn = bad.addProcedureGraph("p", {});
    fn.setSignature({{"return_type", "void"}}, {});
    auto op = fn.makeOperation(CallableOperationKind::AsyncWait, "slot", {});
    op.attributes = {{"type", "void"}, {"slot", "s0"}, {"generation", ""},
                     {"async_kind", "wait"}};
    fn.setBody({std::move(op)});
    expectDiagnostic([&] { validateBehavioralGraphIR(bad); },
                     "graphir.invalid_async_token");
  }
  {
    Module bad("bad_fair.mtr");
    auto &fn = bad.addProcedureGraph("p", {});
    fn.setSignature({{"return_type", "void"}}, {});
    const auto p = fn.addParameter("c", "bool", {}, {}).value();
    auto op = fn.makeOperation(CallableOperationKind::FairChoice, "", {});
    op.operands = {p};
    op.attributes = {{"type", "void"}, {"assumption_justified", "false"}};
    fn.setBody({std::move(op)});
    expectDiagnostic([&] { validateBehavioralGraphIR(bad); },
                     "graphir.invalid_fair_choice");
  }
}

void testCollapsedResourceExpansionAndWitnesses() {
  auto module = parseAndLower(R"(
context Host::Process grants {poll, gpu_await} host;
resource TransferSlot {
  state Idle(ptr: i64);
  state InFlight(ticket: own event<Copy>);
  state Ready(ptr: i64);
  transition issue<C: Context>(@cx: C; self: own Idle) -> InFlight
      !{poll} where C allows {poll};
  await transition wait_ready<C: Context>(@cx: C; self: own InFlight) -> Ready
      !{gpu_await} where C allows {gpu_await};
}
procedure ping(@cx: host; x: i32) -> i32 !{poll} { return x; }
procedure drive(@cx: host; x: i32) -> i32 !{poll} {
  i32 a = ping(@cx; x);
  i32 b = ping(@cx; a);
  return b;
}
)");

  const auto prePassExpanded = expandResourceGraph(module.resourceGraphs()[0]);
  check(prePassExpanded.states.size() == 3 &&
            prePassExpanded.transitions.size() == 2 &&
            prePassExpanded.entryStates == std::vector<ResourceNodeId>{0},
        "unexpected pre-collapse expanded resource structure");

  const auto snapshot = capturePreservationSnapshot(module);
  passes::runDefaultPipeline(module);

  check(module.resourceGraphs()[0].states().size() == 1 &&
            !module.resourceGraphs()[0].states()[0].members.empty(),
        "expected linear resource chain to collapse");
  check(module.procedureGraphs()[1].body().front().kind ==
            CallableOperationKind::ContextRegion,
        "expected adjacent context calls to hoist into ContextRegion");

  const auto postPassExpanded = expandResourceGraph(module.resourceGraphs()[0]);
  check(postPassExpanded.states.size() == prePassExpanded.states.size() &&
            postPassExpanded.transitions.size() ==
                prePassExpanded.transitions.size(),
        "expandResourceGraph lost states or internal transitions after collapse");
  for (std::size_t i = 0; i < prePassExpanded.transitions.size(); ++i) {
    check(postPassExpanded.transitions[i].source ==
              prePassExpanded.transitions[i].source &&
          postPassExpanded.transitions[i].target ==
              prePassExpanded.transitions[i].target &&
          postPassExpanded.transitions[i].attributes ==
              prePassExpanded.transitions[i].attributes,
          "expandResourceGraph changed transition endpoints or attributes");
  }

  verifyPassPreservation(snapshot, module);

  // Witness must reject a corrupted collapsed chain that drops its await flag.
  {
    auto corrupted = parseAndLower(R"(
context Host::Process grants {poll, gpu_await} host;
resource TransferSlot {
  state Idle(ptr: i64);
  state InFlight(ticket: own event<Copy>);
  state Ready(ptr: i64);
  transition issue<C: Context>(@cx: C; self: own Idle) -> InFlight
      !{poll} where C allows {poll};
  await transition wait_ready<C: Context>(@cx: C; self: own InFlight) -> Ready
      !{gpu_await} where C allows {gpu_await};
}
)");
    const auto snap = capturePreservationSnapshot(corrupted);
    passes::runDefaultPipeline(corrupted);
    corrupted.resourceGraphs()[0]
        .states()[0]
        .internalTransitions[1]
        .attributes["await"] = "false";
    expectFailure([&] { verifyPassPreservation(snap, corrupted); },
                  "corrupted await attribute inside collapsed chain");
  }

  // Witness and FusionLegalityPass must reject a StageContextRegion whose
  // context disagrees with its member stage's nominal context.
  {
    auto pipelineMod = parseAndLower(R"(
context Gpu::Thread grants {} ctx_a;
context Gpu::Thread grants {} ctx_b;
pipeline P() {
  stage s1(@cx: ctx_a; x: f32) -> f32 { return x; }
  stage s2(@cx: ctx_a; x: f32) -> f32 { return x; }
  topology { s1.x => s2.x; }
  schedule sm90 { kernel worker = fuse(s1, s2) { blocks 1; } }
}
)");
    const auto snap = capturePreservationSnapshot(pipelineMod);
    passes::runDefaultPipeline(pipelineMod);
    verifyPassPreservation(snap, pipelineMod);

    for (auto &def : pipelineMod.compositionDefinitions()) {
      if (def.kind == DefinitionKind::Schedule) {
        for (auto &rec : def.records) {
          if (rec.kind == "fused_kernel" && !rec.contextRegions.empty()) {
            rec.contextRegions[0].context = pipelineMod.contexts()[1];
          }
        }
      }
    }
    expectDiagnostic([&] { passes::FusionLegalityPass{}.run(pipelineMod); },
                     "graphir.stage_region_context_mismatch");
    expectDiagnostic([&] { validateBehavioralGraphIR(pipelineMod); },
                     "graphir.stage_region_context_mismatch");
  }
}

void testGPUContextRegionPreservation(const char *vectorAddPath) {
  std::ifstream input(vectorAddPath);
  check(static_cast<bool>(input), "cannot open vector_add.mtr");
  const std::string source((std::istreambuf_iterator<char>(input)), {});

  auto module = parseAndLower(source, "vector_add.mtr", true);
  const auto before = capturePreservationSnapshot(module);
  passes::runDefaultPipeline(module);
  verifyPassPreservation(before, module);

  bool sawKernelRegion = false;
  walkCallableOperations(module.kernelGraphs()[0].body(), [&](const auto &op) {
    if (op.kind == CallableOperationKind::ContextRegion) sawKernelRegion = true;
  });
  bool sawHostRegion = false;
  walkCallableOperations(module.procedureGraphs()[0].body(), [&](const auto &op) {
    if (op.kind == CallableOperationKind::ContextRegion) sawHostRegion = true;
  });
  check(sawKernelRegion && sawHostRegion,
        "expected ContextHoistingPass to create ContextRegions in both GPU "
        "kernel and host launcher");

  // Mutating the hoisted ContextRegion to a distinct nominal context must fail
  // preservation verification.
  const auto extraCtx = module.addContextMetadata(
      "Gpu::Thread", "rogue_gpu", 0, ContextResolution::Declared, {});
  auto mutatedBody = module.kernelGraphs()[0].body();
  for (auto &op : mutatedBody) {
    if (op.kind == CallableOperationKind::If && !op.body.empty() &&
        op.body[0].kind == CallableOperationKind::Block &&
        !op.body[0].body.empty() &&
        op.body[0].body[0].kind == CallableOperationKind::ContextRegion) {
      op.body[0].body[0].context = extraCtx;
    }
  }
  module.kernelGraphs()[0].setBody(std::move(mutatedBody));
  expectFailure([&] { verifyPassPreservation(before, module); },
                "mutated ContextRegion nominal context");
}

} // namespace

int main(int argc, char **argv) {
  try {
    check(argc == 2, "expected vector_add.mtr fixture path");
    testBehavioralAndAsyncConstructs();
    testCollapsedResourceExpansionAndWitnesses();
    testGPUContextRegionPreservation(argv[1]);
    std::cout << "GPU-1 behavioral GraphIR and ContextRegion bridge checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
