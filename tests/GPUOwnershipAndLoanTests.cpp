#include "metreon/GraphIR/Behavioral.h"
#include "metreon/GraphIR/Lowering.h"
#include "metreon/GraphIR/Passes.h"
#include "metreon/Parser/Parser.h"
#include "metreon/Verification/GPUOwnershipAndLoans.h"

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

Module parseAndLower(const std::string &source,
                     const char *name = "gpu-ownership-test.mtr",
                     bool optimize = false, bool contextOnly = false) {
  parser::Parser parser(source);
  auto module = lowerModule(parser.parseModule(), name, contextOnly);
  if (optimize) passes::runDefaultPipeline(module);
  return module;
}

bool hasCode(const GPUOwnershipReport &report, std::string_view code) {
  return std::any_of(report.violations.begin(), report.violations.end(),
                     [&](const GPUOwnershipViolation &v) {
                       return v.code == code;
                     });
}

std::string describeViolations(const GPUOwnershipReport &report) {
  std::string out;
  for (const auto &v : report.violations) {
    if (!out.empty()) out += "; ";
    out += "[" + v.code + " in " + v.callableName + "] " + v.message;
  }
  return out;
}

void testScopeHierarchyMatrix() {
  check(classifyContextExecutionScope("Host::Process") == GPUExecutionScope::Host,
        "Host::Process must map to Host scope");
  check(classifyContextExecutionScope("Gpu::Grid") == GPUExecutionScope::DeviceGrid,
        "Gpu::Grid must map to DeviceGrid scope");
  check(classifyContextExecutionScope("Gpu::Block") == GPUExecutionScope::Workgroup,
        "Gpu::Block must map to Workgroup scope");
  check(classifyContextExecutionScope("Gpu::WarpStage<producer>") ==
            GPUExecutionScope::Subgroup,
        "Gpu::WarpStage must map to Subgroup scope");
  check(classifyContextExecutionScope("Gpu::Thread<D, U>") ==
            GPUExecutionScope::Thread,
        "Gpu::Thread must map to Thread scope");
  check(!classifyContextExecutionScope("Gpu::Loop").has_value() &&
            !classifyContextExecutionScope("Host::Loop").has_value(),
        "loop contexts must remain disjoint from execution scopes");
  check(classifyLoopContextDomain("Gpu::Loop") == GPULoopDomain::DeviceLoop &&
            classifyLoopContextDomain("Host::Loop") == GPULoopDomain::HostLoop,
        "loop domain classification failed");

  // Scope capability enforcement without implicit widening/narrowing.
  check(!isCapabilityAllowedInScope(GPUExecutionScope::Thread, "gpu_barrier") &&
            !isCapabilityAllowedInScope(GPUExecutionScope::Thread,
                                        "gpu_warp_shuffle") &&
            !isCapabilityAllowedInScope(GPUExecutionScope::Thread,
                                        "gpu_grid_sync"),
        "Gpu::Thread must reject collective block/warp/grid capabilities");
  check(isCapabilityAllowedInScope(GPUExecutionScope::Subgroup,
                                   "gpu_warp_shuffle") &&
            !isCapabilityAllowedInScope(GPUExecutionScope::Subgroup,
                                        "gpu_barrier"),
        "Gpu::Warp must allow subgroup capabilities and reject block barriers");
  check(isCapabilityAllowedInScope(GPUExecutionScope::Workgroup, "gpu_barrier") &&
            !isCapabilityAllowedInScope(GPUExecutionScope::Workgroup,
                                        "gpu_grid_sync"),
        "Gpu::Block must allow workgroup barriers and reject grid sync");
  check(isCapabilityAllowedInScope(GPUExecutionScope::DeviceGrid,
                                   "gpu_grid_sync"),
        "Gpu::Grid must allow grid sync");
  check(!isCapabilityAllowedInScope(GPUExecutionScope::Host, "gpu_load") &&
            !isCapabilityAllowedInScope(GPUExecutionScope::Host, "gpu_barrier"),
        "Host::* must reject device-resident memory/barrier grants");

  // Scope violation in GraphIR context declaration.
  {
    auto badThreadBarrier = parseAndLower(R"(
context Gpu::Thread grants {gpu_load, gpu_barrier} bad_thread;
procedure noop(@cx: bad_thread) -> void { return; }
)");
    const auto report = verifyGPUOwnershipAndLoans(badThreadBarrier);
    check(!report.accepted() && hasCode(report, "gpu.context.scope_violation"),
          "Gpu::Thread granting gpu_barrier must be rejected");
  }
  {
    auto badHostLoad = parseAndLower(R"(
context Host::Process grants {gpu_load} bad_host;
procedure noop(@cx: bad_host) -> void { return; }
)");
    const auto report = verifyGPUOwnershipAndLoans(badHostLoad);
    check(!report.accepted() && hasCode(report, "gpu.context.scope_violation"),
          "Host::Process granting gpu_load must be rejected");
  }
}

void testValidExamples(const char *vectorAddPath, const char *proceduresPath,
                       const char *contextsPath) {
  for (const bool optimize : {false, true}) {
    const auto vecMod =
        parseAndLower(readFile(vectorAddPath), "vector_add.mtr", optimize, true);
    const auto vecReport = verifyGPUOwnershipAndLoans(vecMod);
    check(vecReport.accepted() && vecReport.checkedCallables == 2,
          "vector_add.mtr must pass GPU-2 ownership and loan checks: " +
              describeViolations(vecReport));

    const auto procMod =
        parseAndLower(readFile(proceduresPath), "procedures.mtr", optimize, false);
    const auto procReport = verifyGPUOwnershipAndLoans(procMod);
    check(procReport.accepted(),
          "procedures.mtr CopySlot lifecycle must pass GPU-2 checks: " +
              describeViolations(procReport));

    const auto ctxMod =
        parseAndLower(readFile(contextsPath), "contexts.mtr", optimize, false);
    const auto ctxReport = verifyGPUOwnershipAndLoans(ctxMod);
    check(ctxReport.accepted(),
          "contexts.mtr scopes must pass GPU-2 checks: " +
              describeViolations(ctxReport));
  }
}

void testResourceOwnershipAndCollapsedMustComplete() {
  const std::string resourceHeader = R"(
context Gpu::Warp grants {gpu_async_copy, gpu_await} warp_ctx;
context Gpu::Loop gpu_loop;

resource CopyTicket {
  state Vacant(ptr: i64);
  state Armed(ptr: i64);
  state InFlight(ptr: i64, ev: own event<Copy>);
  state Done(ptr: i64);

  transition arm<C: Context>(@cx: C; self: own Vacant) -> Armed
      !{gpu_async_copy} where C allows {gpu_async_copy};
  transition launch_copy<C: Context>(@cx: C; self: own Armed) -> InFlight
      !{gpu_async_copy} where C allows {gpu_async_copy};
  await transition finish_copy<C: Context>(@cx: C; self: own InFlight) -> Done
      !{gpu_await} where C allows {gpu_await};
}
)";

  // 1. Valid collapsed chain consumed through `await` and returned cleanly.
  for (const bool optimize : {false, true}) {
    auto okMod = parseAndLower(resourceHeader + R"(
procedure run_valid(@cx: warp_ctx; t: own CopyTicket::Vacant)
    -> own CopyTicket::Done !{gpu_async_copy, gpu_await}
    where warp_ctx allows {gpu_async_copy, gpu_await} {
  own CopyTicket::Armed a = CopyTicket::arm(@cx; t);
  own CopyTicket::InFlight f = CopyTicket::launch_copy(@cx; a);
  own CopyTicket::Done d = CopyTicket::finish_copy(@cx; f);
  return d;
}
)",
                               "valid-collapse.mtr", optimize);
    const auto report = verifyGPUOwnershipAndLoans(okMod);
    check(report.accepted() && report.checkedTransitions == 3,
          "valid collapsed CopyTicket chain must be accepted");
  }

  // 2. Unawaited `must_complete` InFlight state at procedure exit (even after
  // StateChainCollapsePass!).
  for (const bool optimize : {false, true}) {
    auto unawaitedMod = parseAndLower(resourceHeader + R"(
procedure leak_inflight(@cx: warp_ctx; t: own CopyTicket::Vacant)
    -> void !{gpu_async_copy}
    where warp_ctx allows {gpu_async_copy} {
  own CopyTicket::Armed a = CopyTicket::arm(@cx; t);
  own CopyTicket::InFlight f = CopyTicket::launch_copy(@cx; a);
  return;
}
)",
                                      "leak-inflight.mtr", optimize);
    const auto report = verifyGPUOwnershipAndLoans(unawaitedMod);
    check(!report.accepted() &&
              hasCode(report, "gpu.ownership.unawaited_must_complete"),
          "unawaited InFlight resource must fail with unawaited_must_complete");
  }

  // 3. Double consumption of an owned handle injected in GraphIR.
  {
    auto dblMod = parseAndLower(resourceHeader + R"(
procedure valid_seed(@cx: warp_ctx; t: own CopyTicket::Vacant)
    -> own CopyTicket::Armed !{gpu_async_copy}
    where warp_ctx allows {gpu_async_copy} {
  own CopyTicket::Armed a = CopyTicket::arm(@cx; t);
  return a;
}
)");
    auto &proc = dblMod.procedureGraphs()[0];
    auto body = proc.body();
    const auto paramId = proc.parameters()[1].id.value();
    auto secondConsume = proc.makeOperation(CallableOperationKind::Call,
                                            "CopyTicket::arm", {0, 10, 1});
    secondConsume.context = proc.context();
    secondConsume.operands = {paramId};
    secondConsume.attributes = {{"callee_kind", "transition"},
                                {"type", "own CopyTicket::Armed"}};
    body.insert(body.begin() + 1, std::move(secondConsume));
    proc.setBody(std::move(body));

    const auto report = verifyGPUOwnershipAndLoans(dblMod);
    check(!report.accepted() &&
              hasCode(report, "gpu.ownership.double_consumption"),
          "double consumption of CopyTicket::Vacant must be rejected");
  }

  // 4. Branch lifetime leak: `InFlight` completed on `then` only, leaked on `else`.
  {
    auto branchMod = parseAndLower(resourceHeader + R"(
procedure branch_seed(@cx: warp_ctx; flag: bool, f: own CopyTicket::InFlight)
    -> void !{gpu_await} where warp_ctx allows {gpu_await} {
  return;
}
)");
    auto &proc = branchMod.procedureGraphs()[0];
    const auto flagId = proc.parameters()[1].id.value();
    const auto flightId = proc.parameters()[2].id.value();

    // Mark Done as droppable so completing on `then` is clean while `else` leaks `f`.
    for (auto &st : branchMod.resourceGraphs()[0].states()) {
      if (st.name == "Done") st.attributes["lifecycle"] = "droppable";
    }

    auto finishCall = proc.makeOperation(CallableOperationKind::Call,
                                         "CopyTicket::finish_copy", {0, 5, 1});
    finishCall.context = proc.context();
    finishCall.operands = {flightId};
    finishCall.attributes = {{"callee_kind", "transition"},
                             {"type", "own CopyTicket::Done"}};

    auto ifOp = proc.makeOperation(CallableOperationKind::If, "", {0, 4, 1});
    ifOp.operands = {flagId};
    ifOp.body = {std::move(finishCall)};
    auto retOp = proc.makeOperation(CallableOperationKind::Return, "", {0, 6, 1});
    proc.setBody({std::move(ifOp), std::move(retOp)});

    const auto report = verifyGPUOwnershipAndLoans(branchMod);
    check(!report.accepted() &&
              hasCode(report, "gpu.ownership.branch_lifetime_leak"),
          "completing must_complete resource on only one branch must fail");
  }

  // 5. Loop backedge consumption of outer owned handle.
  {
    auto loopMod = parseAndLower(resourceHeader + R"(
procedure loop_seed(@cx: warp_ctx; t: own CopyTicket::Vacant) -> void {
  return;
}
)");
    for (auto &st : loopMod.resourceGraphs()[0].states()) {
      if (st.name == "Armed") st.attributes["lifecycle"] = "droppable";
    }
    auto &proc = loopMod.procedureGraphs()[0];
    const auto tId = proc.parameters()[1].id.value();
    const auto loopCtx = loopMod.contexts()[1];

    auto armCall = proc.makeOperation(CallableOperationKind::Call,
                                      "CopyTicket::arm", {0, 5, 1});
    armCall.context = proc.context();
    armCall.operands = {tId};
    armCall.attributes = {{"callee_kind", "transition"},
                          {"type", "own CopyTicket::Armed"}};

    auto forOp = proc.makeOperation(CallableOperationKind::For, "k", {0, 4, 1});
    forOp.loopContext = loopCtx;
    forOp.body = {std::move(armCall)};
    auto retOp = proc.makeOperation(CallableOperationKind::Return, "", {0, 7, 1});
    proc.setBody({std::move(forOp), std::move(retOp)});

    const auto report = verifyGPUOwnershipAndLoans(loopMod);
    check(!report.accepted() &&
              hasCode(report, "gpu.ownership.loop_backedge_consumption"),
          "consuming outer owned handle inside loop must fail");
  }
}

void testAsyncGenerationLoansAndAddressSpaces() {
  // Base module with Warp context + global/shared spans.
  auto makeWarpModule = [] {
    Module m("async-loans.mtr");
    const auto warpCtx = m.addContextMetadata(
        "Gpu::Warp", "warp", 0, ContextResolution::Declared, {0, 1, 1});
    const auto loopCtx = m.addContextMetadata(
        "Gpu::Loop", "gpu_loop", 0, ContextResolution::Declared, {0, 2, 1});
    m.addNode(NodeKind::Grant, "gpu_async_copy", {{"capability", "gpu_async_copy"}},
              warpCtx, {0, 1, 10});
    m.addNode(NodeKind::Grant, "gpu_await", {{"capability", "gpu_await"}}, warpCtx,
              {0, 1, 20});
    m.addNode(NodeKind::Grant, "gpu_load", {{"capability", "gpu_load"}}, warpCtx,
              {0, 1, 30});
    m.addNode(NodeKind::Grant, "gpu_store", {{"capability", "gpu_store"}}, warpCtx,
              {0, 1, 40});
    auto &kern = m.addKernelGraph("stage_copy", {0, 3, 1});
    kern.setSignature({{"return_type", "void"},
                       {"execution_domain", "Gpu"},
                       {"effects", ""},
                       {"where", ""}},
                      warpCtx);
    (void)kern.addParameter("cx", "warp", warpCtx, {0, 3, 10});
    (void)kern.addParameter("src", "gpu::span<f32>", {}, {0, 3, 20});
    (void)kern.addParameter("dst", "gpu::span<f32>", {}, {0, 3, 30});
    (void)kern.addParameter("idx", "index", {}, {0, 3, 40});
    (void)kern.addParameter("cond", "bool", {}, {0, 3, 50});
    (void)loopCtx;
    return m;
  };

  // 1. Positive: issue(slot0, 0) -> immutable read of src -> wait(slot0, 0) ->
  // read/write dst -> issue(slot0, 1) -> wait(slot0, 1), including ContextHoisting!
  {
    auto m = makeWarpModule();
    auto &k = m.kernelGraphs()[0];
    const auto ctx = k.context();
    const auto src = k.parameters()[1].id.value();
    const auto dst = k.parameters()[2].id.value();
    const auto idx = k.parameters()[3].id.value();

    std::vector<CallableOperation> body;
    body.push_back(makeAsyncTokenOperation(
        k, {AsyncActionKind::Issue, "slot0", "0", "gpu::global", "gpu::shared",
            "void", {src, dst}, ctx, {0, 4, 1}}));

    auto srcRead = k.makeOperation(CallableOperationKind::Call, "gpu::load", {0, 5, 1});
    srcRead.context = ctx;
    srcRead.operands = {src, idx};
    srcRead.attributes = {{"callee_kind", "gpu_builtin"}, {"type", "f32"}};
    const auto valId = srcRead.id;
    body.push_back(std::move(srcRead));

    body.push_back(makeAsyncTokenOperation(
        k, {AsyncActionKind::Wait, "slot0", "0", "gpu::global", "gpu::shared",
            "void", {}, ctx, {0, 6, 1}}));

    auto dstWrite = k.makeOperation(CallableOperationKind::Call, "gpu::store", {0, 7, 1});
    dstWrite.context = ctx;
    dstWrite.operands = {dst, idx, valId};
    dstWrite.attributes = {{"callee_kind", "gpu_builtin"}, {"type", "void"}};
    body.push_back(std::move(dstWrite));

    body.push_back(makeAsyncTokenOperation(
        k, {AsyncActionKind::Issue, "slot0", "1", "gpu::global", "gpu::shared",
            "void", {src, dst}, ctx, {0, 8, 1}}));
    body.push_back(makeAsyncTokenOperation(
        k, {AsyncActionKind::Wait, "slot0", "1", "gpu::global", "gpu::shared",
            "void", {}, ctx, {0, 9, 1}}));

    k.setBody(std::move(body));
    passes::ContextHoistingPass{}.run(m);

    const auto report = verifyGPUOwnershipAndLoans(m);
    check(report.accepted() && report.checkedAsyncIssues == 2 &&
              report.checkedAsyncWaits == 2,
          "valid multi-epoch async issue/wait sequence must pass");
  }

  // 2. Negative: Pre-wait buffer read on `dst` (even after waiting on unrelated `slot1`!).
  {
    auto m = makeWarpModule();
    auto &k = m.kernelGraphs()[0];
    const auto ctx = k.context();
    const auto src = k.parameters()[1].id.value();
    const auto dst = k.parameters()[2].id.value();
    const auto idx = k.parameters()[3].id.value();

    std::vector<CallableOperation> body;
    body.push_back(makeAsyncTokenOperation(
        k, {AsyncActionKind::Issue, "slot0", "0", "gpu::global", "gpu::shared",
            "void", {src, dst}, ctx, {0, 4, 1}}));
    body.push_back(makeAsyncTokenOperation(
        k, {AsyncActionKind::Issue, "slot1", "0", "gpu::global", "gpu::shared",
            "void", {src}, ctx, {0, 4, 2}}));
    // Wait ONLY on slot1; slot0 still holds exclusive write loan on `dst`.
    body.push_back(makeAsyncTokenOperation(
        k, {AsyncActionKind::Wait, "slot1", "0", "gpu::global", "gpu::shared",
            "void", {}, ctx, {0, 5, 1}}));

    auto prematureRead =
        k.makeOperation(CallableOperationKind::Call, "gpu::load", {0, 6, 1});
    prematureRead.context = ctx;
    prematureRead.operands = {dst, idx};
    prematureRead.attributes = {{"callee_kind", "gpu_builtin"}, {"type", "f32"}};
    body.push_back(std::move(prematureRead));

    body.push_back(makeAsyncTokenOperation(
        k, {AsyncActionKind::Wait, "slot0", "0", "gpu::global", "gpu::shared",
            "void", {}, ctx, {0, 7, 1}}));
    k.setBody(std::move(body));

    const auto report = verifyGPUOwnershipAndLoans(m);
    check(!report.accepted() && hasCode(report, "gpu.loan.pre_wait_buffer_access"),
          "reading dst after unrelated slot wait must fail with pre_wait_buffer_access");
  }

  // 3. Negative: Stale-generation completion reuse (`wait(slot0, 0)` twice).
  {
    auto m = makeWarpModule();
    auto &k = m.kernelGraphs()[0];
    const auto ctx = k.context();
    const auto src = k.parameters()[1].id.value();
    const auto dst = k.parameters()[2].id.value();

    std::vector<CallableOperation> body;
    body.push_back(makeAsyncTokenOperation(
        k, {AsyncActionKind::Issue, "slot0", "0", "gpu::global", "gpu::shared",
            "void", {src, dst}, ctx, {0, 4, 1}}));
    body.push_back(makeAsyncTokenOperation(
        k, {AsyncActionKind::Wait, "slot0", "0", "gpu::global", "gpu::shared",
            "void", {}, ctx, {0, 5, 1}}));
    body.push_back(makeAsyncTokenOperation(
        k, {AsyncActionKind::Wait, "slot0", "0", "gpu::global", "gpu::shared",
            "void", {}, ctx, {0, 6, 1}}));
    k.setBody(std::move(body));

    const auto report = verifyGPUOwnershipAndLoans(m);
    check(!report.accepted() && hasCode(report, "gpu.loan.stale_generation_reuse"),
          "re-awaiting completed generation must fail with stale_generation_reuse");
  }

  // 4. Negative: Asymmetric branch wait + loop backedge loan leak.
  {
    auto m = makeWarpModule();
    auto &k = m.kernelGraphs()[0];
    const auto ctx = k.context();
    const auto loopCtx = m.contexts()[1];
    const auto src = k.parameters()[1].id.value();
    const auto dst = k.parameters()[2].id.value();
    const auto cond = k.parameters()[4].id.value();

    std::vector<CallableOperation> body;
    body.push_back(makeAsyncTokenOperation(
        k, {AsyncActionKind::Issue, "slot_branch", "0", "gpu::global",
            "gpu::shared", "void", {src, dst}, ctx, {0, 4, 1}}));
    auto ifOp = k.makeOperation(CallableOperationKind::If, "", {0, 5, 1});
    ifOp.operands = {cond};
    ifOp.body = {makeAsyncTokenOperation(
        k, {AsyncActionKind::Wait, "slot_branch", "0", "gpu::global",
            "gpu::shared", "void", {}, ctx, {0, 5, 5}})};
    body.push_back(std::move(ifOp));

    auto forOp = k.makeOperation(CallableOperationKind::For, "step", {0, 7, 1});
    forOp.loopContext = loopCtx;
    forOp.body = {makeAsyncTokenOperation(
        k, {AsyncActionKind::Issue, "slot_loop", "iter", "gpu::global",
            "gpu::shared", "void", {src, dst}, ctx, {0, 7, 5}})};
    body.push_back(std::move(forOp));
    k.setBody(std::move(body));

    const auto report = verifyGPUOwnershipAndLoans(m);
    check(!report.accepted() &&
              hasCode(report, "gpu.loan.asymmetric_branch_wait") &&
              hasCode(report, "gpu.loan.loop_backedge_leak"),
          "asymmetric branch wait and unawaited loop issue must both fail");
  }

  // 5. Negative: Cross-space dereference (device reading host::span, and device
  // async_issue touching host::default).
  {
    auto m = makeWarpModule();
    auto &k = m.kernelGraphs()[0];
    const auto ctx = k.context();
    const auto idx = k.parameters()[3].id.value();
    const auto hostParam =
        k.addParameter("host_buf", "host::span<f32>", {}, {0, 3, 60}).value();

    auto badLoad = k.makeOperation(CallableOperationKind::Call, "gpu::load", {0, 4, 1});
    badLoad.context = ctx;
    badLoad.operands = {hostParam, idx};
    badLoad.attributes = {{"callee_kind", "gpu_builtin"}, {"type", "f32"}};

    auto badAsync = makeAsyncTokenOperation(
        k, {AsyncActionKind::Issue, "slot_host", "0", "host::default",
            "gpu::shared", "void", {hostParam}, ctx, {0, 5, 1}});
    auto waitAsync = makeAsyncTokenOperation(
        k, {AsyncActionKind::Wait, "slot_host", "0", "host::default",
            "gpu::shared", "void", {}, ctx, {0, 6, 1}});
    k.setBody({std::move(badLoad), std::move(badAsync), std::move(waitAsync)});

    const auto report = verifyGPUOwnershipAndLoans(m);
    check(!report.accepted() &&
              hasCode(report, "gpu.address_space.cross_space_dereference"),
          "device dereferencing host::span or host::default async space must fail");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    check(argc == 4, "expected vector_add.mtr, procedures.mtr, and contexts.mtr");
    testScopeHierarchyMatrix();
    testValidExamples(argv[1], argv[2], argv[3]);
    testResourceOwnershipAndCollapsedMustComplete();
    testAsyncGenerationLoansAndAddressSpaces();
    std::cout << "GPU Phase 2 GraphIR scope, ownership, and async loan tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "GPUOwnershipAndLoanTests failed: " << error.what() << '\n';
    return 1;
  }
}
