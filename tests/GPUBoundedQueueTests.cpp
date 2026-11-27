#include "metreon/GraphIR/Behavioral.h"
#include "metreon/GraphIR/Lowering.h"
#include "metreon/GraphIR/Passes.h"
#include "metreon/Parser/Parser.h"
#include "metreon/Verification/Behavioral.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

using namespace metreon;
using namespace metreon::graphir;
using namespace metreon::verification;

void check(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

std::string readFile(const char *path) {
  std::ifstream input(path);
  check(static_cast<bool>(input), std::string("cannot open fixture: ") + path);
  return std::string((std::istreambuf_iterator<char>(input)), {});
}

Module parseModuleText(const std::string &source,
                       const char *name = "gpu-bounded-queue-test.mtr",
                       bool optimize = false) {
  parser::Parser parser(source);
  auto mod = lowerModule(parser.parseModule(), name);
  if (optimize) passes::runDefaultPipeline(mod);
  return mod;
}

bool hasCode(const GPUBoundedQueueReport &report, std::string_view code) {
  return std::any_of(report.violations.begin(), report.violations.end(),
                     [&](const GPUBoundedQueueViolation &v) {
                       return v.code == code;
                     });
}

std::string describe(const GPUBoundedQueueReport &report) {
  std::string out;
  for (const auto &v : report.violations) {
    if (!out.empty()) out += "; ";
    out += "[" + v.code + " on " + v.subject + "] " + v.message;
  }
  return out;
}

StageGraph *findStage(Module &module, const std::string &qualifiedName) {
  for (auto &s : module.stageGraphs()) {
    if (s.name() == qualifiedName) return &s;
  }
  return nullptr;
}

void testValidWorkgroupSPSCAndMultiStagePipelines(const char *examplePath) {
  const std::string exampleSrc = readFile(examplePath);

  // 1. Structural `examples/protocols_and_pipelines.mtr` passes across default
  //    GraphIR optimization passes and unoptimized mode!
  for (const bool opt : {false, true}) {
    auto mod = parseModuleText(exampleSrc, "protocols_and_pipelines.mtr", opt);
    const auto rep = verifyGPUBoundedQueueCompatibility(mod);
    check(rep.accepted() && rep.status == VerificationStatus::UnboundedProved &&
              rep.checkedPipelines == 1 && rep.checkedQueues == 2 &&
              rep.checkedSchedules == 1 && rep.validatedCertificates == 1,
          "protocols_and_pipelines.mtr must pass GPU-4 verification: " +
              describe(rep));
  }

  // 2. Explicit behavioral producer/consumer `TiledGemm` pipeline with capacity
  //    C = 1 and C = 2, multi-tile transfers, `stop` final transition, and
  //    progress marker `$mma_completed`!
  for (const int cap : {1, 2}) {
    std::string cappedSrc = exampleSrc;
    if (cap == 2) {
      std::size_t pos = 0;
      while ((pos = cappedSrc.find("capacity 1;", pos)) != std::string::npos) {
        cappedSrc.replace(pos, 11, "capacity 2;");
        pos += 11;
      }
    }
    auto mod = parseModuleText(cappedSrc, "tiled-explicit.mtr", false);
    auto *loadA = findStage(mod, "TiledGemm::load_a");
    auto *loadB = findStage(mod, "TiledGemm::load_b");
    auto *mma = findStage(mod, "TiledGemm::mma");
    check(loadA && loadB && mma, "missing TiledGemm stages");

    // load_a sends 2 tiles then `stop` on port `out` (`TiledGemm::a_tiles`).
    loadA->setBody({
        makeChannelAction(*loadA, {ChannelDirection::Send, "out", "tile",
                                   ChannelDeliveryMode::BlockingQueue, true,
                                   "void", {}, {}, {48, 5}}),
        makeChannelAction(*loadA, {ChannelDirection::Send, "out", "tile",
                                   ChannelDeliveryMode::BlockingQueue, true,
                                   "void", {}, {}, {49, 5}}),
        makeChannelAction(*loadA, {ChannelDirection::Send, "out", "stop",
                                   ChannelDeliveryMode::BlockingQueue, false,
                                   "void", {}, {}, {50, 5}}),
    });

    // load_b sends 2 tiles then `stop` on port `out` (`TiledGemm::b_tiles`).
    loadB->setBody({
        makeChannelAction(*loadB, {ChannelDirection::Send, "out", "tile",
                                   ChannelDeliveryMode::BlockingQueue, true,
                                   "void", {}, {}, {58, 5}}),
        makeChannelAction(*loadB, {ChannelDirection::Send, "out", "tile",
                                   ChannelDeliveryMode::BlockingQueue, true,
                                   "void", {}, {}, {59, 5}}),
        makeChannelAction(*loadB, {ChannelDirection::Send, "out", "stop",
                                   ChannelDeliveryMode::BlockingQueue, false,
                                   "void", {}, {}, {60, 5}}),
    });

    // mma drains 2 tiles + `stop` from both `a_in` and `b_in` and hits `$mma_done`.
    mma->setBody({
        makeChannelAction(*mma, {ChannelDirection::Receive, "a_in", "tile",
                                 ChannelDeliveryMode::BlockingQueue, true,
                                 "void", {}, {}, {66, 5}}),
        makeChannelAction(*mma, {ChannelDirection::Receive, "b_in", "tile",
                                 ChannelDeliveryMode::BlockingQueue, true,
                                 "void", {}, {}, {67, 5}}),
        makeChannelAction(*mma, {ChannelDirection::Receive, "a_in", "tile",
                                 ChannelDeliveryMode::BlockingQueue, true,
                                 "void", {}, {}, {68, 5}}),
        makeChannelAction(*mma, {ChannelDirection::Receive, "b_in", "tile",
                                 ChannelDeliveryMode::BlockingQueue, true,
                                 "void", {}, {}, {69, 5}}),
        makeChannelAction(*mma, {ChannelDirection::Receive, "a_in", "stop",
                                 ChannelDeliveryMode::BlockingQueue, false,
                                 "void", {}, {}, {70, 5}}),
        makeChannelAction(*mma, {ChannelDirection::Receive, "b_in", "stop",
                                 ChannelDeliveryMode::BlockingQueue, false,
                                 "void", {}, {}, {71, 5}}),
        makeProgressMarkerOperation(*mma, {"mma_done", "completed_gemm", {}, {72, 5}}),
    });

    passes::runDefaultPipeline(mod);
    const auto rep = verifyGPUBoundedQueueCompatibility(mod);
    check(rep.accepted() && rep.status == VerificationStatus::UnboundedProved &&
              rep.validatedCertificates == 1 && !rep.certificates.empty(),
          "explicit 2-tile TiledGemm must pass with capacity=" +
              std::to_string(cap) + ": " + describe(rep));

    // Corrupting slot conservation in the emitted certificate must be rejected
    // by the independent certificate validator!
    CompatibilityCertificate tampered = rep.certificates.front();
    tampered.states[0].queueStates[0].freeCredits += 1;
    std::string reason;
    check(!validateCompatibilityCertificate(tampered, true, &reason) &&
              reason.find("slot conservation") != std::string::npos,
          "tampered queue slot conservation must fail certificate validation");
  }
}

void testCreditSeededCyclicPipelineAndUnseededDeadlock() {
  const std::string cyclicTemplate = R"(
protocol DataFlow roles prod, cons {
  state Streaming {
    prod -> cons: item(v: f32) => Streaming;
  }
}

protocol CreditFlow roles issuer, holder {
  state Ready {
    issuer -> holder: grant => Ready;
  }
}

pipeline RingPipeline(input x: f32, output y: f32) {
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
    seeded_items SEED_COUNT;
    storage gpu::shared;
    scope block;
    backpressure cooperative;
    publication acq_rel_workgroup;
  }

  stage producer(src: f32, cred_in: CreditFlow::holder<Ready>, data_out: DataFlow::prod<Streaming>) -> void {
    return;
  }

  stage consumer(data_in: DataFlow::cons<Streaming>, cred_out: CreditFlow::issuer<Ready>, dst: f32) -> void {
    return;
  }

  topology {
    x                 => producer.src;
    producer.data_out =>[data_q] consumer.data_in;
    consumer.cred_out =>[credit_q] producer.cred_in;
    consumer.dst      => y;
  }

  schedule sm89 {
    kernel worker = fuse(producer, consumer) {
      blocks 1;
    }
    place producer on worker.warp[0];
    place consumer on worker.warp[1];
  }
}
)";

  auto populateCyclicBodies = [](Module &mod, bool infiniteLoop) {
    const auto loopCtx = mod.addContextMetadata(
        "Gpu::Loop", "ring_loop", 0, ContextResolution::Declared, {1, 1});
    auto *prod = findStage(mod, "RingPipeline::producer");
    auto *cons = findStage(mod, "RingPipeline::consumer");
    check(prod && cons, "missing RingPipeline stages");

    std::vector<CallableOperation> prodStep = {
        makeChannelAction(*prod, {ChannelDirection::Receive, "cred_in", "grant",
                                  ChannelDeliveryMode::BlockingQueue, false,
                                  "void", {}, {}, {10, 1}}),
        makeChannelAction(*prod, {ChannelDirection::Send, "data_out", "item",
                                  ChannelDeliveryMode::BlockingQueue, true,
                                  "void", {}, {}, {11, 1}}),
        makeProgressMarkerOperation(*prod, {"prod_step", "sent_item", {}, {12, 1}}),
    };
    std::vector<CallableOperation> consStep = {
        makeChannelAction(*cons, {ChannelDirection::Receive, "data_in", "item",
                                  ChannelDeliveryMode::BlockingQueue, true,
                                  "void", {}, {}, {20, 1}}),
        makeChannelAction(*cons, {ChannelDirection::Send, "cred_out", "grant",
                                  ChannelDeliveryMode::BlockingQueue, false,
                                  "void", {}, {}, {21, 1}}),
        makeProgressMarkerOperation(*cons, {"cons_step", "returned_credit", {}, {22, 1}}),
    };

    if (infiniteLoop) {
      auto pLoop = prod->makeOperation(CallableOperationKind::Loop, "", {9, 1});
      pLoop.loopContext = loopCtx;
      pLoop.body = std::move(prodStep);
      prod->setBody({std::move(pLoop)});

      auto cLoop = cons->makeOperation(CallableOperationKind::Loop, "", {19, 1});
      cLoop.loopContext = loopCtx;
      cLoop.body = std::move(consStep);
      cons->setBody({std::move(cLoop)});
    } else {
      prod->setBody(std::move(prodStep));
      cons->setBody(std::move(consStep));
    }
  };

  // 1. Positive: `seeded_items 1;` credit-seeded cyclic pipeline passes BOTH as
  //    a finite round-trip AND as an infinite recurring streaming pipeline, and
  //    survives `passes::runDefaultPipeline` (`FusionLegalityPass`)!
  for (const bool infinite : {false, true}) {
    std::string seededSrc = cyclicTemplate;
    seededSrc.replace(seededSrc.find("SEED_COUNT"), 10, "1");
    auto mod = parseModuleText(seededSrc, "seeded-ring.mtr", false);
    populateCyclicBodies(mod, infinite);
    passes::runDefaultPipeline(mod);

    const auto rep = verifyGPUBoundedQueueCompatibility(mod);
    check(rep.accepted() && rep.status == VerificationStatus::UnboundedProved &&
              rep.validatedCertificates == 1,
          std::string("credit-seeded cyclic pipeline (infinite=") +
              (infinite ? "true" : "false") + ") must pass: " + describe(rep));
  }

  // 2. Negative: `seeded_items 0;` unseeded cyclic pipeline deadlocks immediately
  //    (both `producer` and `consumer` block waiting on empty queues)!
  {
    std::string unseededSrc = cyclicTemplate;
    unseededSrc.replace(unseededSrc.find("SEED_COUNT"), 10, "0");
    auto mod = parseModuleText(unseededSrc, "unseeded-ring.mtr", false);
    populateCyclicBodies(mod, /*infiniteLoop=*/false);

    const auto rep = verifyGPUBoundedQueueCompatibility(mod);
    check(!rep.accepted() && hasCode(rep, "gpu.queue.deadlock"),
          "unseeded cyclic pipeline must fail with gpu.queue.deadlock");
  }

  // 3. Negative: `seeded_items 2;` on a `capacity 1` queue violates slot conservation!
  {
    std::string overSeeded = cyclicTemplate;
    overSeeded.replace(overSeeded.find("SEED_COUNT"), 10, "2");
    auto mod = parseModuleText(overSeeded, "overseeded-ring.mtr", false);
    const auto rep = verifyGPUBoundedQueueCompatibility(mod);
    check(!rep.accepted() &&
              hasCode(rep, "gpu.queue.slot_conservation_violation"),
          "seeded_items > capacity must fail with slot_conservation_violation");
  }
}

void testTwoSendCapacityOneDeadlockVsCapacityTwo() {
  const std::string exchangeTemplate = R"(
protocol MsgFlow roles sender, receiver {
  state Active {
    sender -> receiver: pkt(v: f32) => Active;
  }
}

pipeline CrossExchange(input a: f32, output b: f32) {
  edge ch_x: channel<MsgFlow> {
    initial Active;
    capacity CAP_VAL;
    storage gpu::shared;
    scope block;
    backpressure blocking;
  }
  edge ch_y: channel<MsgFlow> {
    initial Active;
    capacity CAP_VAL;
    storage gpu::shared;
    scope block;
    backpressure blocking;
  }

  stage actor_a(in_a: f32, x_out: MsgFlow::sender<Active>, y_in: MsgFlow::receiver<Active>) -> void {
    return;
  }
  stage actor_b(y_out: MsgFlow::sender<Active>, x_in: MsgFlow::receiver<Active>, out_b: f32) -> void {
    return;
  }

  topology {
    a             => actor_a.in_a;
    actor_a.x_out =>[ch_x] actor_b.x_in;
    actor_b.y_out =>[ch_y] actor_a.y_in;
    actor_b.out_b => b;
  }

  schedule sm89 {
    kernel worker = fuse(actor_a, actor_b) {
      blocks 1;
    }
    place actor_a on worker.warp[0];
    place actor_b on worker.warp[1];
  }
}
)";

  auto attachTwoSendBodies = [](Module &mod) {
    auto *a = findStage(mod, "CrossExchange::actor_a");
    auto *b = findStage(mod, "CrossExchange::actor_b");
    check(a && b, "missing CrossExchange stages");
    // Actor A: send x; send x; recv y; recv y
    a->setBody({
        makeChannelAction(*a, {ChannelDirection::Send, "x_out", "pkt",
                               ChannelDeliveryMode::BlockingQueue, false, "void",
                               {}, {}, {10, 1}}),
        makeChannelAction(*a, {ChannelDirection::Send, "x_out", "pkt",
                               ChannelDeliveryMode::BlockingQueue, false, "void",
                               {}, {}, {11, 1}}),
        makeChannelAction(*a, {ChannelDirection::Receive, "y_in", "pkt",
                               ChannelDeliveryMode::BlockingQueue, false, "void",
                               {}, {}, {12, 1}}),
        makeChannelAction(*a, {ChannelDirection::Receive, "y_in", "pkt",
                               ChannelDeliveryMode::BlockingQueue, false, "void",
                               {}, {}, {13, 1}}),
    });
    // Actor B: send y; send y; recv x; recv x
    b->setBody({
        makeChannelAction(*b, {ChannelDirection::Send, "y_out", "pkt",
                               ChannelDeliveryMode::BlockingQueue, false, "void",
                               {}, {}, {20, 1}}),
        makeChannelAction(*b, {ChannelDirection::Send, "y_out", "pkt",
                               ChannelDeliveryMode::BlockingQueue, false, "void",
                               {}, {}, {21, 1}}),
        makeChannelAction(*b, {ChannelDirection::Receive, "x_in", "pkt",
                               ChannelDeliveryMode::BlockingQueue, false, "void",
                               {}, {}, {22, 1}}),
        makeChannelAction(*b, {ChannelDirection::Receive, "x_in", "pkt",
                               ChannelDeliveryMode::BlockingQueue, false, "void",
                               {}, {}, {23, 1}}),
    });
  };

  // 1. Capacity 1: both `ch_x` and `ch_y` fill on the first `send` and both
  //    stages block forever on their second `send` -> MUST REJECT (`gpu.queue.deadlock`)!
  {
    std::string cap1 = exchangeTemplate;
    while (cap1.find("CAP_VAL") != std::string::npos)
      cap1.replace(cap1.find("CAP_VAL"), 7, "1");
    auto mod = parseModuleText(cap1, "two-send-cap1.mtr", false);
    attachTwoSendBodies(mod);
    const auto rep = verifyGPUBoundedQueueCompatibility(mod);
    check(!rep.accepted() && hasCode(rep, "gpu.queue.deadlock") &&
              !rep.violations.front().counterexampleTrace.empty(),
          "two-send capacity-1 cross exchange must fail with deadlock trace");
  }

  // 2. Capacity 2: both `ch_x` and `ch_y` absorb 2 items before draining -> PASSES!
  {
    std::string cap2 = exchangeTemplate;
    while (cap2.find("CAP_VAL") != std::string::npos)
      cap2.replace(cap2.find("CAP_VAL"), 7, "2");
    auto mod = parseModuleText(cap2, "two-send-cap2.mtr", false);
    attachTwoSendBodies(mod);
    const auto rep = verifyGPUBoundedQueueCompatibility(mod);
    check(rep.accepted() && rep.status == VerificationStatus::UnboundedProved,
          "two-send capacity-2 cross exchange must pass without deadlock: " +
              describe(rep));
  }
}

void testResidencyMemoryScopesFairnessAndSolverBounds() {
  const std::string basePipe = R"(
protocol OneShot roles prod, cons {
  state Open {
    prod -> cons: msg(x: f32) => Done;
  }
  state Done final;
}

pipeline P(input a: f32, output b: f32) {
  edge q: channel<OneShot> {
    initial Open;
    capacity 1;
    storage STORAGE_KIND;
    scope SCOPE_KIND;
    backpressure blocking;
    publication PUB_KIND;
  }
  stage s1(in_a: f32, out_q: OneShot::prod<Open>) -> void { return; }
  stage s2(in_q: OneShot::cons<Open>, out_b: f32) -> void { return; }
  topology {
    a        => s1.in_a;
    s1.out_q =>[q] s2.in_q;
    s2.out_b => b;
  }
  SCHEDULE_BLOCK
}
)";

  auto instantiate = [&](std::string storage, std::string scope, std::string pub,
                         std::string sched) {
    std::string s = basePipe;
    s.replace(s.find("STORAGE_KIND"), 12, storage);
    s.replace(s.find("SCOPE_KIND"), 10, scope);
    s.replace(s.find("PUB_KIND"), 8, pub);
    s.replace(s.find("SCHEDULE_BLOCK"), 14, sched);
    return parseModuleText(s);
  };

  // 1. Negative: Unscheduled cross-block blocking (`blocks 4` in fused worker).
  {
    auto mod = instantiate("gpu::shared", "block", "acq_rel_workgroup", R"(
      schedule sm89 {
        kernel worker = fuse(s1, s2) { blocks 4; }
        place s1 on worker.warp[0];
        place s2 on worker.warp[1];
      }
    )");
    const auto rep = verifyGPUBoundedQueueCompatibility(mod);
    check(!rep.accepted() &&
              hasCode(rep, "gpu.residency.unscheduled_cross_block_blocking"),
          "multi-block workgroup channel must fail co-residency check");
  }

  // 2. Negative: Cross-kernel blocking (s1 and s2 split across separate workers).
  {
    auto mod = instantiate("gpu::global", "device", "acq_rel_device", R"(
      schedule sm89 {
        kernel w1 = fuse(s1) { blocks 1; }
        kernel w2 = fuse(s2) { blocks 1; }
        place s1 on w1.warp[0];
        place s2 on w2.warp[0];
      }
    )");
    const auto rep = verifyGPUBoundedQueueCompatibility(mod);
    check(!rep.accepted() &&
              hasCode(rep, "gpu.residency.unscheduled_cross_block_blocking"),
          "cross-kernel blocking channel must fail co-residency check");
  }

  // 3. Negative: Insufficient atomic memory scope (`publication relaxed;` on `block` queue).
  {
    auto mod = instantiate("gpu::shared", "block", "relaxed", R"(
      schedule sm89 {
        kernel worker = fuse(s1, s2) { blocks 1; }
        place s1 on worker.warp[0];
        place s2 on worker.warp[1];
      }
    )");
    const auto rep = verifyGPUBoundedQueueCompatibility(mod);
    check(!rep.accepted() &&
              hasCode(rep, "gpu.memory.insufficient_publication_scope"),
          "relaxed publication order must fail memory-scope sufficiency check");
  }

  // 4. Negative: Workgroup-only `acq_rel_workgroup` on a `CooperativeGrid` (`device`) queue!
  {
    auto mod = instantiate("gpu::global", "device", "acq_rel_workgroup", R"(
      schedule sm89 {
        kernel worker = fuse(s1, s2) {
          blocks 2;
          residency cooperative;
        }
        place s1 on worker.warp[0];
        place s2 on worker.warp[1];
      }
    )");
    GPUBoundedQueueOptions coopOpts;
    coopOpts.profile.allowCooperativeGridChannels = true;
    const auto rep = verifyGPUBoundedQueueCompatibility(mod, coopOpts);
    check(!rep.accepted() &&
              hasCode(rep, "gpu.memory.insufficient_publication_scope"),
          "workgroup-scoped atomics on cooperative grid queue must be rejected");
  }

  // 5. Positive: Cooperative grid channel with `acq_rel_device` when explicitly enabled,
  //    plus boundary summary extraction!
  {
    auto mod = instantiate("gpu::global", "device", "acq_rel_device", R"(
      schedule sm89 {
        kernel worker = fuse(s1, s2) {
          blocks 2;
          residency cooperative;
        }
        place s1 on worker.warp[0];
        place s2 on worker.warp[1];
      }
    )");
    GPUBoundedQueueOptions coopOpts;
    coopOpts.profile.allowCooperativeGridChannels = true;
    const auto rep = verifyGPUBoundedQueueCompatibility(mod, coopOpts);
    check(rep.accepted() && rep.status == VerificationStatus::UnboundedProved,
          "cooperative grid channel with acq_rel_device must pass when enabled: " +
              describe(rep));
  }

  // 6. Negative: Protocol conformance violation (sending `msg` twice on `OneShot`,
  //    where first `msg` transitions `Open -> Done (final)`!).
  {
    auto mod = instantiate("gpu::shared", "block", "acq_rel_workgroup", R"(
      schedule sm89 {
        kernel worker = fuse(s1, s2) { blocks 1; }
        place s1 on worker.warp[0];
        place s2 on worker.warp[1];
      }
    )");
    auto *s1 = findStage(mod, "P::s1");
    auto *s2 = findStage(mod, "P::s2");
    s1->setBody({
        makeChannelAction(*s1, {ChannelDirection::Send, "out_q", "msg",
                                ChannelDeliveryMode::BlockingQueue, false,
                                "void", {}, {}, {10, 1}}),
        makeChannelAction(*s1, {ChannelDirection::Send, "out_q", "msg",
                                ChannelDeliveryMode::BlockingQueue, false,
                                "void", {}, {}, {11, 1}}),
    });
    s2->setBody({
        makeChannelAction(*s2, {ChannelDirection::Receive, "in_q", "msg",
                                ChannelDeliveryMode::BlockingQueue, false,
                                "void", {}, {}, {20, 1}}),
        makeChannelAction(*s2, {ChannelDirection::Receive, "in_q", "msg",
                                ChannelDeliveryMode::BlockingQueue, false,
                                "void", {}, {}, {21, 1}}),
    });
    const auto rep = verifyGPUBoundedQueueCompatibility(mod);
    check(!rep.accepted() &&
              hasCode(rep, "gpu.protocol.conformance_violation"),
          "sending message from final protocol state must fail conformance");
  }

  // 7. Offer polling livelock vs Fair-choice conditional progress & `unknown` solver limit!
  {
    auto mod = instantiate("gpu::shared", "block", "acq_rel_workgroup", R"(
      schedule sm89 {
        kernel worker = fuse(s1, s2) { blocks 1; }
        place s1 on worker.warp[0];
        place s2 on worker.warp[1];
      }
    )");
    const auto loopCtx = mod.addContextMetadata(
        "Gpu::Loop", "poll_loop", 0, ContextResolution::Declared, {1, 1});
    auto *s1 = findStage(mod, "P::s1");
    auto *s2 = findStage(mod, "P::s2");

    // s1 exits immediately without sending; s2 polls in an Offer loop expecting
    // `$got_msg` -> default branch loops forever without visiting `$got_msg`!
    s1->setBody({});
    auto rdyLit = s2->makeOperation(CallableOperationKind::Literal, "true", {1, 1});
    rdyLit.attributes = {{"type", "bool"}, {"kind", "boolean"}, {"value", "true"}};
    auto recvAct = makeChannelAction(
        *s2, {ChannelDirection::Receive, "in_q", "msg",
              ChannelDeliveryMode::BlockingQueue, false, "void", {}, {}, {2, 1}});
    auto doneMark =
        makeProgressMarkerOperation(*s2, {"got_msg", "received", {}, {3, 1}});
    auto brk = s2->makeOperation(CallableOperationKind::Break, "", {4, 1});
    auto offerOp = makeOfferOperation(
        *s2, OfferSpec{OfferMode::Exclusive,
                       {rdyLit},
                       {OfferBranchSpec{ChannelDirection::Receive,
                                        "in_q",
                                        "msg",
                                        rdyLit.id,
                                        {recvAct, doneMark, brk},
                                        {2, 1}}},
                       {},
                       {},
                       {2, 1}});
    auto pollLoop = s2->makeOperation(CallableOperationKind::Loop, "", {1, 1});
    pollLoop.loopContext = loopCtx;
    pollLoop.body = {std::move(offerOp)};
    s2->setBody({std::move(pollLoop)});

    const auto badPoll = verifyGPUBoundedQueueCompatibility(mod);
    check(!badPoll.accepted() &&
              hasCode(badPoll, "gpu.queue.livelock_or_starvation"),
          "non-terminating offer poll without sender must fail livelock/starvation");

    // Now make s1 send `msg`! Because s1 eventually enqueues `msg`, once `msg`
    // is in `q` the exclusive Offer's `default` branch is disabled (`anyOfferArmEnabled == true`),
    // forcing s2 to take the `Receive` arm, hit `$got_msg`, break, and terminate!
    s1->setBody({makeChannelAction(
        *s1, {ChannelDirection::Send, "out_q", "msg",
              ChannelDeliveryMode::BlockingQueue, false, "void", {}, {}, {1, 1}})});
    const auto okOffer = verifyGPUBoundedQueueCompatibility(mod);
    check(okOffer.accepted() &&
              okOffer.status == VerificationStatus::UnboundedProved,
          "polling offer paired with live producer must terminate cleanly: " +
              describe(okOffer));

    // Tight exploration limit (`maxExploredStates = 2`) must report `unknown`,
    // never silently passing!
    GPUBoundedQueueOptions tightOpts;
    tightOpts.maxExploredStates = 2;
    const auto unkRep = verifyGPUBoundedQueueCompatibility(mod, tightOpts);
    check(!unkRep.accepted() &&
              unkRep.status == VerificationStatus::Unknown &&
              isIndeterminateOrUnsupported(unkRep.status) &&
              hasCode(unkRep, "gpu.solver.state_limit_exceeded"),
          "exceeding maxExploredStates must report VerificationStatus::Unknown");
  }

  // 8. Fair-choice conditional progress + Boundary Summary Automaton!
  {
    GPUPipelineSystemSpec sys;
    sys.name = "FairArbiterSubsys";

    BehavioralAutomaton stage;
    stage.name = "arbiter";
    stage.initialState = 0;
    stage.states = {
        {0, "loop_head", false, std::nullopt, {}},
        {1, "left_arm", false, std::nullopt, {}},
        {2, "right_arm", false, std::nullopt, {}},
    };
    stage.requiredProgressMarkers = {"served_left", "served_right"};
    stage.transitions = {
        {0, 1, BehavioralActionKind::Internal, "", "", ChannelDeliveryMode::BlockingQueue,
         false, false, "eta_rr", true, /*assumptionJustified=*/false, {}, {}},
        {0, 2, BehavioralActionKind::Internal, "", "", ChannelDeliveryMode::BlockingQueue,
         false, false, "eta_rr", false, /*assumptionJustified=*/false, {}, {}},
        {1, 0, BehavioralActionKind::Send, "ext_out", "left",
         ChannelDeliveryMode::BlockingQueue, false, false, std::nullopt, true,
         false, {"served_left"}, {}},
        {2, 0, BehavioralActionKind::Send, "ext_out", "right",
         ChannelDeliveryMode::BlockingQueue, false, false, std::nullopt, true,
         false, {"served_right"}, {}},
    };
    sys.stages.push_back(std::move(stage));

    GPUBoundedQueueReport rep;
    verifyGPUPipelineSystem(sys, {}, rep);
    check(rep.accepted() && rep.conditionalFairnessTags.contains("eta_rr"),
          "fair-choice arbiter must pass and record conditional fairness tag `eta_rr`");

    const auto summary = computeBoundarySummary(sys);
    check(summary.has_value() &&
              summary->boundaryEndpoints.contains("ext_out") &&
              summary->conditionalFairnessTags.contains("eta_rr"),
          "boundary summary automaton must preserve external endpoints and fairness tags");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    check(argc == 2, "expected protocols_and_pipelines.mtr fixture path");
    testValidWorkgroupSPSCAndMultiStagePipelines(argv[1]);
    testCreditSeededCyclicPipelineAndUnseededDeadlock();
    testTwoSendCapacityOneDeadlockVsCapacityTwo();
    testResidencyMemoryScopesFairnessAndSolverBounds();
    std::cout << "GPU Phase 4 bounded-queue compatibility, co-residency, and memory-scope tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "GPUBoundedQueueTests failed: " << error.what() << '\n';
    return 1;
  }
}
