#include "metreon/GraphIR/Lowering.h"
#include "metreon/GraphIR/Passes.h"
#include "metreon/Parser/Parser.h"
#include "metreon/Verification/GPUMemoryAccessProtocol.h"

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
                     const char *name = "gpu-access-protocol-test.mtr",
                     bool optimize = false, bool contextOnly = true) {
  parser::Parser parser(source);
  auto module = lowerModule(parser.parseModule(), name, contextOnly);
  if (optimize) passes::runDefaultPipeline(module);
  return module;
}

bool hasCode(const GPUProtocolReport &report, std::string_view code) {
  return std::any_of(report.violations.begin(), report.violations.end(),
                     [&](const GPUProtocolViolation &v) {
                       return v.code == code;
                     });
}

std::string describe(const GPUProtocolReport &report) {
  std::string out;
  for (const auto &v : report.violations) {
    if (!out.empty()) out += "; ";
    out += "[" + v.code + " in " + v.kernelName + "] " + v.message;
  }
  return out;
}

void testSIMTUniformityAndParticipationLattice() {
  check(joinUniformity(SIMTUniformity::WorkgroupUniform,
                       SIMTUniformity::SubgroupUniform) ==
            SIMTUniformity::SubgroupUniform,
        "WorkgroupUniform join SubgroupUniform must be SubgroupUniform");
  check(joinUniformity(SIMTUniformity::SubgroupUniform,
                       SIMTUniformity::LaneDivergent) ==
            SIMTUniformity::LaneDivergent,
        "SubgroupUniform join LaneDivergent must be LaneDivergent");

  const std::string header = R"(
context Gpu::Block grants {gpu_load, gpu_store, gpu_barrier, gpu_warp_barrier} blk;
context Gpu::Loop gpu_loop;
)";

  // 1. Positive: Uniform workgroup barrier inside uniform conditional and counted loop.
  for (const bool opt : {false, true}) {
    auto okUniform = parseAndLower(header + R"(
kernel uniform_sync(@cx: blk; a: gpu::span<f32>, out: gpu::span<f32>, n: index) {
  index i = gpu::global_id(@cx);
  if (n > 0) {
    for<@cx: gpu_loop> (index step) in 0..4 {
      gpu::barrier(@cx);
    }
  }
  if (i < n) {
    f32 v = gpu::load(@cx; a, i);
    gpu::store(@cx; out, i, v);
  }
}
)",
                                   "uniform-sync.mtr", opt);
    const auto rep = verifyGPUMemoryAccessProtocol(okUniform);
    check(rep.accepted() && rep.checkedWorkgroupBarriers == 1,
          "uniform workgroup barrier must be accepted: " + describe(rep));
  }

  // 2. Positive: Subgroup-uniform conditional (`tid / 32 == 0`) permits `gpu::warp_barrier`!
  {
    auto okWarpSpec = parseAndLower(header + R"(
kernel warp_specialized(@cx: blk; out: gpu::span<f32>) {
  index tid = gpu::thread_id(@cx);
  index bid = gpu::block_id(@cx);
  if (tid / 32 == 0) {
    gpu::warp_barrier(@cx);
  }
  if (tid == 0) {
    gpu::store(@cx; out, bid, 1.0);
  }
}
)");
    const auto rep = verifyGPUMemoryAccessProtocol(okWarpSpec);
    check(rep.accepted() && rep.checkedSubgroupBarriers == 1,
          "subgroup-uniform warp_barrier must be accepted: " + describe(rep));
  }

  // 3. Negative: Workgroup `gpu::barrier` inside subgroup-uniform branch (`tid / 32 == 0`)
  // must be rejected (convergent within a warp, but divergent across the workgroup!).
  {
    auto badWgInWarpBranch = parseAndLower(header + R"(
kernel bad_wg_in_warp(@cx: blk) {
  index tid = gpu::thread_id(@cx);
  if (tid / 32 == 0) {
    gpu::barrier(@cx);
  }
}
)");
    const auto rep = verifyGPUMemoryAccessProtocol(badWgInWarpBranch);
    check(!rep.accepted() &&
              hasCode(rep, "gpu.simt.divergent_workgroup_barrier"),
          "workgroup barrier inside warp-only branch must be rejected");
  }

  // 4. Negative: Workgroup `gpu::barrier` inside lane-divergent `if (i < n)`.
  {
    auto badIfBarrier = parseAndLower(header + R"(
kernel bad_if_barrier(@cx: blk; n: index) {
  index i = gpu::global_id(@cx);
  if (i < n) {
    gpu::barrier(@cx);
  }
}
)");
    const auto rep = verifyGPUMemoryAccessProtocol(badIfBarrier);
    check(!rep.accepted() &&
              hasCode(rep, "gpu.simt.divergent_workgroup_barrier"),
          "workgroup barrier inside lane-divergent if must be rejected");
  }

  // 5. Negative: Indirect control-dependence taint (`if (tid == 0) flag = 1; if (flag == 1) barrier`).
  for (const bool opt : {false, true}) {
    auto badIndirect = parseAndLower(header + R"(
kernel bad_indirect_taint(@cx: blk) {
  index tid = gpu::thread_id(@cx);
  i32 flag = 0;
  if (tid == 0) {
    flag = 1;
  }
  if (flag == 1) {
    gpu::barrier(@cx);
  }
}
)",
                                     "indirect-taint.mtr", opt);
    const auto rep = verifyGPUMemoryAccessProtocol(badIndirect);
    check(!rep.accepted() &&
              hasCode(rep, "gpu.simt.divergent_workgroup_barrier"),
          "indirectly tainted workgroup barrier must be rejected");
  }

  // 6. Negative: Subgroup `gpu::warp_barrier` under lane-divergent `if (tid % 2 == 0)`
  // (ambiguous subgroup participation!).
  {
    auto badSubgroup = parseAndLower(header + R"(
kernel bad_warp_even_lanes(@cx: blk) {
  index tid = gpu::thread_id(@cx);
  if (tid % 2 == 0) {
    gpu::warp_barrier(@cx);
  }
}
)");
    const auto rep = verifyGPUMemoryAccessProtocol(badSubgroup);
    check(!rep.accepted() &&
              hasCode(rep, "gpu.simt.ambiguous_subgroup_participation"),
          "lane-divergent warp_barrier must be rejected");
  }

  // 7. Negative: Divergent `while` loop containing a workgroup barrier.
  {
    auto badWhileBarrier = parseAndLower(header + R"(
kernel bad_while_barrier(@cx: blk) {
  index tid = gpu::thread_id(@cx);
  index step = 0;
  while<@cx: gpu_loop> (step < tid) {
    gpu::barrier(@cx);
    step = step + 1;
  }
}
)");
    const auto rep = verifyGPUMemoryAccessProtocol(badWhileBarrier);
    check(!rep.accepted() &&
              hasCode(rep, "gpu.simt.divergent_workgroup_barrier"),
          "workgroup barrier inside thread-divergent while loop must be rejected");
  }
}

void testGlobalSpanDisjointnessAndRaces(const char *vectorAddPath) {
  const std::string header = R"(
context Gpu::Block grants {gpu_load, gpu_store, gpu_barrier, gpu_warp_barrier} blk;
context Gpu::Loop gpu_loop;
)";

  // 1. Positive: `examples/vector_add.mtr` passes both DRF and proven bounds (`i < n`)!
  for (const bool opt : {false, true}) {
    auto vecMod =
        parseAndLower(readFile(vectorAddPath), "vector_add.mtr", opt, true);
    GPUMemoryAccessProtocolOptions strictOpts;
    strictOpts.requireProvenBounds = true;
    const auto rep = verifyGPUMemoryAccessProtocol(vecMod, strictOpts);
    check(rep.accepted() && rep.checkedKernels == 1,
          "vector_add.mtr must pass DRF + proven bounds: " + describe(rep));
  }

  // 2. Positive: Disjoint even/odd strided writes (`2 * i` and `2 * i + 1`) proved by GCD parity.
  {
    auto okParity = parseAndLower(header + R"(
kernel strided_pairs(@cx: blk; out: gpu::span<f32>, n: index) {
  index i = gpu::global_id(@cx);
  if (i < n) {
    gpu::store(@cx; out, 2 * i, 1.0);
    gpu::store(@cx; out, 2 * i + 1, 2.0);
  }
}
)");
    const auto rep = verifyGPUMemoryAccessProtocol(okParity);
    check(rep.accepted(),
          "disjoint 2*i and 2*i+1 writes must be accepted: " + describe(rep));
  }

  // 3. Positive: Per-block leader write (`if (tid == 0) { out[bid] = ... }`).
  {
    auto okLeader = parseAndLower(header + R"(
kernel block_leader_write(@cx: blk; out: gpu::span<f32>) {
  index tid = gpu::thread_id(@cx);
  index bid = gpu::block_id(@cx);
  if (tid == 0) {
    gpu::store(@cx; out, bid, 42.0);
  }
}
)");
    const auto rep = verifyGPUMemoryAccessProtocol(okLeader);
    check(rep.accepted(),
          "per-block leader write out[bid] under tid == 0 must be accepted: " +
              describe(rep));
  }

  // 4. Negative: Constant index write-write race (`out[0] = 1.0` by all threads).
  {
    auto badConstWW = parseAndLower(header + R"(
kernel racy_const(@cx: blk; out: gpu::span<f32>) {
  gpu::store(@cx; out, 0, 1.0);
}
)");
    const auto rep = verifyGPUMemoryAccessProtocol(badConstWW);
    check(!rep.accepted() && hasCode(rep, "gpu.map.write_write_data_race"),
          "unguarded out[0] write must fail with write_write_data_race");
  }

  // 5. Negative: Cross-block global write-write race (`out[tid] = 1.0` without `block_id`!).
  {
    auto badCrossBlock = parseAndLower(header + R"(
kernel racy_cross_block(@cx: blk; out: gpu::span<f32>) {
  index tid = gpu::thread_id(@cx);
  gpu::store(@cx; out, tid, 1.0);
}
)");
    const auto rep = verifyGPUMemoryAccessProtocol(badCrossBlock);
    check(!rep.accepted() && hasCode(rep, "gpu.map.write_write_data_race"),
          "out[tid] in multi-block grid must fail with cross-block W-W race");
  }

  // 6. Negative: Neighbor read-write race (`v = buf[i + 1]; buf[i] = v` without barrier).
  {
    auto badRW = parseAndLower(header + R"(
kernel racy_shift(@cx: blk; buf: gpu::span<f32>) {
  index i = gpu::global_id(@cx);
  f32 neighbor = gpu::load(@cx; buf, i + 1);
  gpu::store(@cx; buf, i, neighbor);
}
)");
    const auto rep = verifyGPUMemoryAccessProtocol(badRW);
    check(!rep.accepted() && hasCode(rep, "gpu.map.read_write_data_race"),
          "unsynchronized neighbor shift must fail with read_write_data_race");
  }

  // 7. Negative: Unguarded span access rejected when `requireProvenBounds = true`.
  {
    auto unguarded = parseAndLower(header + R"(
kernel missing_guard(@cx: blk; out: gpu::span<f32>, n: index) {
  index i = gpu::global_id(@cx);
  gpu::store(@cx; out, i, 1.0);
}
)");
    GPUMemoryAccessProtocolOptions strictOpts;
    strictOpts.requireProvenBounds = true;
    const auto rep = verifyGPUMemoryAccessProtocol(unguarded, strictOpts);
    check(!rep.accepted() && hasCode(rep, "gpu.map.unproved_index_bounds"),
          "unguarded span write must fail requireProvenBounds check");
  }
}

void testLoopBarrierAligningAndSubgroupEpochs() {
  const auto tid = AffineSymbolicExpr::makeVar(SymbolicVarKind::ThreadIdX);
  const auto tidPlus1Mod64 =
      remExpr(addExpr(tid, AffineSymbolicExpr::makeConstant(1)),
              AffineSymbolicExpr::makeConstant(64), 32);

  // 1. Positive: Properly two-barrier aligned shared-memory stencil loop:
  //    for step in 0..4 {
  //      read smem[(tid + 1) % 64];
  //      workgroup_barrier;
  //      write smem[tid];
  //      workgroup_barrier;
  //    }
  {
    std::vector<ProtocolStmt> loopBody;
    loopBody.push_back(ProtocolStmt::makeAccess(
        MemoryAccessMode::Read, "smem", GPUAddressSpace::WorkgroupShared,
        tidPlus1Mod64, {10, 1}, 64));
    loopBody.push_back(ProtocolStmt::makeWorkgroupBarrier(
        SIMTUniformity::WorkgroupUniform, {11, 1}));
    loopBody.push_back(ProtocolStmt::makeAccess(
        MemoryAccessMode::Write, "smem", GPUAddressSpace::WorkgroupShared, tid,
        {12, 1}, 64));
    loopBody.push_back(ProtocolStmt::makeWorkgroupBarrier(
        SIMTUniformity::WorkgroupUniform, {13, 1}));

    const auto loop = ProtocolStmt::makeLoop(
        "step", AffineSymbolicExpr::makeConstant(0),
        AffineSymbolicExpr::makeConstant(4), 1,
        SIMTUniformity::WorkgroupUniform, /*boundedFor=*/true,
        std::move(loopBody), {9, 1});

    GPUProtocolReport rep;
    verifyKernelAccessProtocol("aligned_stencil", loop, {}, rep);
    check(rep.accepted(),
          "two-barrier aligned stencil loop must be accepted: " + describe(rep));
  }

  // 2. Negative: Missing trailing barrier in shared-memory stencil loop!
  //    Iteration k writes `smem[tid]` in trailing sub-phase and wraps across
  //    loop backedge into iteration k+1's leading read `smem[(tid + 1) % 64]`!
  {
    std::vector<ProtocolStmt> badLoopBody;
    badLoopBody.push_back(ProtocolStmt::makeAccess(
        MemoryAccessMode::Read, "smem", GPUAddressSpace::WorkgroupShared,
        tidPlus1Mod64, {20, 1}, 64));
    badLoopBody.push_back(ProtocolStmt::makeWorkgroupBarrier(
        SIMTUniformity::WorkgroupUniform, {21, 1}));
    badLoopBody.push_back(ProtocolStmt::makeAccess(
        MemoryAccessMode::Write, "smem", GPUAddressSpace::WorkgroupShared, tid,
        {22, 1}, 64));

    const auto badLoop = ProtocolStmt::makeLoop(
        "step", AffineSymbolicExpr::makeConstant(0),
        AffineSymbolicExpr::makeConstant(4), 1,
        SIMTUniformity::WorkgroupUniform, /*boundedFor=*/true,
        std::move(badLoopBody), {19, 1});

    GPUProtocolReport rep;
    verifyKernelAccessProtocol("unaligned_stencil", badLoop, {}, rep);
    check(!rep.accepted() && hasCode(rep, "gpu.map.read_write_data_race"),
          "missing trailing barrier in stencil loop must be caught by barrier aligning");
  }

  // 3. Positive: Warp-0 intra-warp shared exchange separated by `SubgroupBarrier`!
  //    if (warp_id == 0) {
  //      write smem[lane_id];
  //      subgroup_barrier;
  //      read smem[(lane_id + 1) % 32];
  //    }
  {
    const auto warpId = AffineSymbolicExpr::makeVar(SymbolicVarKind::WarpId);
    const auto laneId = AffineSymbolicExpr::makeVar(SymbolicVarKind::LaneId);
    const auto nextLane =
        remExpr(addExpr(tid, AffineSymbolicExpr::makeConstant(1)),
                AffineSymbolicExpr::makeConstant(32), 32);

    SymbolicGuard thenG, elseG;
    thenG.addComparison("==", warpId, AffineSymbolicExpr::makeConstant(0));
    elseG.addComparison("!=", warpId, AffineSymbolicExpr::makeConstant(0));

    std::vector<ProtocolStmt> warpBody;
    warpBody.push_back(ProtocolStmt::makeAccess(
        MemoryAccessMode::Write, "warp_smem", GPUAddressSpace::WorkgroupShared,
        laneId, {30, 1}, 32));
    warpBody.push_back(ProtocolStmt::makeSubgroupBarrier(
        SIMTUniformity::SubgroupUniform, "gpu::warp_barrier", {31, 1}));
    warpBody.push_back(ProtocolStmt::makeAccess(
        MemoryAccessMode::Read, "warp_smem", GPUAddressSpace::WorkgroupShared,
        nextLane, {32, 1}, 32));

    const auto cond = ProtocolStmt::makeConditional(
        SIMTUniformity::SubgroupUniform, thenG, elseG, std::move(warpBody), {},
        {29, 1});

    GPUProtocolReport rep;
    verifyKernelAccessProtocol("warp0_exchange", cond, {}, rep);
    check(rep.accepted(),
          "intra-warp shared exchange synchronized by warp_barrier must pass: " +
              describe(rep));
  }

  // 4. Negative: Inter-warp shared race across `SubgroupBarrier`!
  //    Warp 0 writes `smem[0]`, `subgroup_barrier` executes across all warps,
  //    then all warps read `smem[0]` -> warp 1 is NOT ordered with warp 0 by a
  //    subgroup-only barrier!
  {
    const auto tidZeroGuard =
        SymbolicGuard{}.withComparison("==", tid, AffineSymbolicExpr::makeConstant(0));
    const auto tidNonZeroGuard =
        SymbolicGuard{}.withComparison("!=", tid, AffineSymbolicExpr::makeConstant(0));

    std::vector<ProtocolStmt> seq;
    seq.push_back(ProtocolStmt::makeConditional(
        SIMTUniformity::LaneDivergent, tidZeroGuard, tidNonZeroGuard,
        {ProtocolStmt::makeAccess(MemoryAccessMode::Write, "smem",
                                  GPUAddressSpace::WorkgroupShared,
                                  AffineSymbolicExpr::makeConstant(0), {40, 1},
                                  64)},
        {}, {40, 1}));
    seq.push_back(ProtocolStmt::makeSubgroupBarrier(
        SIMTUniformity::WorkgroupUniform, "gpu::warp_barrier", {41, 1}));
    seq.push_back(ProtocolStmt::makeAccess(
        MemoryAccessMode::Read, "smem", GPUAddressSpace::WorkgroupShared,
        AffineSymbolicExpr::makeConstant(0), {42, 1}, 64));

    GPUProtocolReport rep;
    verifyKernelAccessProtocol("inter_warp_race",
                               ProtocolStmt::makeSequence(std::move(seq)), {},
                               rep);
    check(!rep.accepted() && hasCode(rep, "gpu.map.read_write_data_race"),
          "subgroup barrier must NOT mask inter-warp shared R-W race across warp 0 and warp 1");
  }

  // 5. Positive: Tiled single-block loop `smem[tile * 64 + tid]` and static OOB rejection.
  {
    const auto tileVar =
        AffineSymbolicExpr::makeVar(SymbolicVarKind::LoopVar, "tile");
    const auto tiledIdx = addExpr(
        mulExpr(tileVar, AffineSymbolicExpr::makeConstant(64)), tid);
    const auto tiledLoop = ProtocolStmt::makeLoop(
        "tile", AffineSymbolicExpr::makeConstant(0),
        AffineSymbolicExpr::makeConstant(4), 1,
        SIMTUniformity::WorkgroupUniform, /*boundedFor=*/true,
        {ProtocolStmt::makeAccess(MemoryAccessMode::Write, "smem",
                                  GPUAddressSpace::WorkgroupShared, tiledIdx,
                                  {50, 1}, 256)},
        {49, 1});
    GPUMemoryAccessProtocolOptions strictOpts;
    strictOpts.requireProvenBounds = true;
    GPUProtocolReport okRep;
    verifyKernelAccessProtocol("tiled_smem", tiledLoop, strictOpts, okRep);
    check(okRep.accepted(),
          "disjoint tiled shared-memory loop must pass DRF + bounds: " +
              describe(okRep));

    const auto oobStmt = ProtocolStmt::makeAccess(
        MemoryAccessMode::Write, "smem", GPUAddressSpace::WorkgroupShared,
        AffineSymbolicExpr::makeConstant(64), {55, 1}, 64);
    GPUProtocolReport badOob;
    verifyKernelAccessProtocol("oob_smem", oobStmt, {}, badOob);
    check(!badOob.accepted() && hasCode(badOob, "gpu.map.out_of_bounds_access"),
          "static index 64 on extent-64 shared array must fail bounds check");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    check(argc == 2, "expected vector_add.mtr fixture path");
    testSIMTUniformityAndParticipationLattice();
    testGlobalSpanDisjointnessAndRaces(argv[1]);
    testLoopBarrierAligningAndSubgroupEpochs();
    std::cout << "GPU Phase 3 SIMT uniformity, subgroup participation, and memory access protocol tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "GPUMemoryAccessProtocolTests failed: " << error.what()
              << '\n';
    return 1;
  }
}
