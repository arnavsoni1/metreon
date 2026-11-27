#include "metreon/Verification/GPUMemoryAccessProtocol.h"

#include "metreon/Verification/GPUOwnershipAndLoans.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>

namespace metreon::verification {
namespace {

std::optional<std::int64_t> parseSignedInt(std::string_view text) noexcept {
  if (text.empty()) return std::nullopt;
  std::int64_t val = 0;
  const char *begin = text.data();
  const char *end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, val);
  if (ec != std::errc{} || ptr != end) return std::nullopt;
  return val;
}

std::string invertComparison(std::string_view op) {
  if (op == "==") return "!=";
  if (op == "!=") return "==";
  if (op == "<") return ">=";
  if (op == "<=") return ">";
  if (op == ">") return "<=";
  if (op == ">=") return "<";
  return "";
}

void recordProtocolViolation(GPUProtocolReport &report, std::string code,
                             std::string kernelName, std::string subject,
                             std::string message, SourceLocation location) {
  report.status = VerificationStatus::Rejected;
  report.violations.push_back(GPUProtocolViolation{
      std::move(code), std::move(kernelName), std::move(subject),
      std::move(message), location});
}

void pruneZeroTerms(AffineSymbolicExpr &expr) {
  std::erase_if(expr.terms,
                [](const auto &kv) { return kv.second == 0; });
}

std::int64_t positiveMod(std::int64_t value, std::int64_t mod) noexcept {
  if (mod <= 0) return value;
  const std::int64_t r = value % mod;
  return r < 0 ? r + mod : r;
}

struct SyncSummary {
  bool hasWorkgroupBarrier = false;
  bool hasSubgroupBarrier = false;
};

SyncSummary summarizeSync(const ProtocolStmt &stmt) {
  SyncSummary s;
  switch (stmt.kind) {
  case ProtocolStmtKind::Access:
    return s;
  case ProtocolStmtKind::WorkgroupBarrier:
    s.hasWorkgroupBarrier = true;
    return s;
  case ProtocolStmtKind::SubgroupBarrier:
    s.hasSubgroupBarrier = true;
    return s;
  case ProtocolStmtKind::Sequence:
  case ProtocolStmtKind::Loop:
    for (const auto &c : stmt.children) {
      const auto cs = summarizeSync(c);
      s.hasWorkgroupBarrier |= cs.hasWorkgroupBarrier;
      s.hasSubgroupBarrier |= cs.hasSubgroupBarrier;
    }
    return s;
  case ProtocolStmtKind::Conditional:
    for (const auto &c : stmt.thenBody) {
      const auto cs = summarizeSync(c);
      s.hasWorkgroupBarrier |= cs.hasWorkgroupBarrier;
      s.hasSubgroupBarrier |= cs.hasSubgroupBarrier;
    }
    for (const auto &c : stmt.elseBody) {
      const auto cs = summarizeSync(c);
      s.hasWorkgroupBarrier |= cs.hasWorkgroupBarrier;
      s.hasSubgroupBarrier |= cs.hasSubgroupBarrier;
    }
    return s;
  }
  return s;
}

void verifyUniformityTree(const std::string &kernelName,
                          const ProtocolStmt &stmt,
                          SIMTUniformity ambientPC,
                          GPUProtocolReport &report) {
  switch (stmt.kind) {
  case ProtocolStmtKind::Access:
    return;

  case ProtocolStmtKind::WorkgroupBarrier: {
    ++report.checkedWorkgroupBarriers;
    const SIMTUniformity eff = joinUniformity(ambientPC, stmt.uniformity);
    if (eff != SIMTUniformity::WorkgroupUniform) {
      recordProtocolViolation(
          report, "gpu.simt.divergent_workgroup_barrier", kernelName,
          "gpu.barrier",
          std::string("workgroup barrier executed under ") +
              simtUniformityName(eff) +
              " control flow (requires workgroup-uniform participation)",
          stmt.location);
    }
    return;
  }

  case ProtocolStmtKind::SubgroupBarrier: {
    ++report.checkedSubgroupBarriers;
    const SIMTUniformity eff = joinUniformity(ambientPC, stmt.uniformity);
    if (eff == SIMTUniformity::LaneDivergent) {
      recordProtocolViolation(
          report, "gpu.simt.ambiguous_subgroup_participation", kernelName,
          stmt.barrierKindName.empty() ? "warp_barrier" : stmt.barrierKindName,
          "subgroup collective/barrier executed under lane-divergent control "
          "flow (ambiguous intra-warp participation)",
          stmt.location);
    }
    return;
  }

  case ProtocolStmtKind::Sequence:
    for (const auto &child : stmt.children) {
      verifyUniformityTree(kernelName, child, ambientPC, report);
    }
    return;

  case ProtocolStmtKind::Conditional: {
    const SIMTUniformity nextPC = joinUniformity(ambientPC, stmt.uniformity);
    for (const auto &child : stmt.thenBody) {
      verifyUniformityTree(kernelName, child, nextPC, report);
    }
    for (const auto &child : stmt.elseBody) {
      verifyUniformityTree(kernelName, child, nextPC, report);
    }
    return;
  }

  case ProtocolStmtKind::Loop: {
    const SIMTUniformity nextPC = joinUniformity(ambientPC, stmt.uniformity);
    for (const auto &child : stmt.children) {
      verifyUniformityTree(kernelName, child, nextPC, report);
    }
    return;
  }
  }
}

struct ProjectedAccess {
  MemoryAccessMode mode = MemoryAccessMode::Read;
  std::string bufferName;
  GPUAddressSpace addressSpace = GPUAddressSpace::GlobalDevice;
  std::optional<std::int64_t> staticExtent;
  std::optional<std::string> dynamicExtentParam;
  AffineSymbolicExpr indexExpr;
  SymbolicGuard guard;
  std::size_t wgEpoch = 0;
  std::size_t sgEpoch = 0;
  SourceLocation location;
};

SymbolicGuard conjoinGuards(const SymbolicGuard &a, const SymbolicGuard &b) {
  SymbolicGuard out = a;
  if (b.contradiction) out.contradiction = true;
  out.constraints.insert(out.constraints.end(), b.constraints.begin(),
                         b.constraints.end());
  return out;
}

SymbolicGuard substituteInGuard(const SymbolicGuard &guard,
                                const SymbolicVar &var,
                                const AffineSymbolicExpr &replacement) {
  SymbolicGuard out;
  out.contradiction = guard.contradiction;
  for (const auto &c : guard.constraints) {
    AffineConstraint subC;
    subC.rel = c.rel;
    subC.expr = c.expr.substitute(var, replacement);
    if (subC.expr.isAffine && subC.expr.modulus == 0) {
      if (subC.expr.terms.empty()) {
        const bool ok =
            (subC.rel == ConstraintRel::EqZero && subC.expr.constant == 0) ||
            (subC.rel == ConstraintRel::NeZero && subC.expr.constant != 0) ||
            (subC.rel == ConstraintRel::GeZero && subC.expr.constant >= 0);
        if (!ok) out.contradiction = true;
      } else {
        out.constraints.push_back(std::move(subC));
      }
    }
  }
  return out;
}

ProtocolStmt substituteInStmt(const ProtocolStmt &stmt,
                              const SymbolicVar &var,
                              const AffineSymbolicExpr &replacement) {
  ProtocolStmt copy = stmt;
  copy.indexExpr = stmt.indexExpr.substitute(var, replacement);
  copy.thenGuard = substituteInGuard(stmt.thenGuard, var, replacement);
  copy.elseGuard = substituteInGuard(stmt.elseGuard, var, replacement);
  copy.loopLower = stmt.loopLower.substitute(var, replacement);
  copy.loopUpper = stmt.loopUpper.substitute(var, replacement);
  for (auto &c : copy.thenBody) c = substituteInStmt(c, var, replacement);
  for (auto &c : copy.elseBody) c = substituteInStmt(c, var, replacement);
  for (auto &c : copy.children) c = substituteInStmt(c, var, replacement);
  return copy;
}

class PhaseProjector {
public:
  std::vector<ProjectedAccess> accesses;
  std::size_t nextEpochId = 1;

  struct EpochPair {
    std::size_t wg = 0;
    std::size_t sg = 0;
  };

  EpochPair project(const ProtocolStmt &stmt, EpochPair cur,
                    const SymbolicGuard &guard) {
    if (guard.contradiction) return cur;

    switch (stmt.kind) {
    case ProtocolStmtKind::Access:
      accesses.push_back(ProjectedAccess{
          stmt.accessMode, stmt.bufferName, stmt.addressSpace,
          stmt.staticExtent, stmt.dynamicExtentParam, stmt.indexExpr, guard,
          cur.wg, cur.sg, stmt.location});
      return cur;

    case ProtocolStmtKind::WorkgroupBarrier: {
      const std::size_t fresh = nextEpochId++;
      return EpochPair{fresh, fresh};
    }

    case ProtocolStmtKind::SubgroupBarrier: {
      const std::size_t freshSg = nextEpochId++;
      return EpochPair{cur.wg, freshSg};
    }

    case ProtocolStmtKind::Sequence: {
      EpochPair e = cur;
      for (const auto &child : stmt.children) {
        e = project(child, e, guard);
      }
      return e;
    }

    case ProtocolStmtKind::Conditional: {
      const SyncSummary sync = summarizeSync(stmt);
      const SymbolicGuard gThen = conjoinGuards(guard, stmt.thenGuard);
      const SymbolicGuard gElse = conjoinGuards(guard, stmt.elseGuard);

      if (!sync.hasWorkgroupBarrier && !sync.hasSubgroupBarrier) {
        for (const auto &c : stmt.thenBody) (void)project(c, cur, gThen);
        for (const auto &c : stmt.elseBody) (void)project(c, cur, gElse);
        return cur;
      }

      EpochPair eThen = cur;
      for (const auto &c : stmt.thenBody) eThen = project(c, eThen, gThen);
      EpochPair eElse = cur;
      for (const auto &c : stmt.elseBody) eElse = project(c, eElse, gElse);

      if (!sync.hasWorkgroupBarrier) {
        // Subgroup-only synchronization: workgroup epoch remains `cur.wg`, while
        // post-conditional subgroup epoch advances so intra-warp ordering holds
        // when both branches synchronize, or stays conservative if one branch had
        // no subgroup barrier.
        const SyncSummary thenSync = summarizeSync(
            ProtocolStmt::makeSequence(stmt.thenBody));
        const SyncSummary elseSync = summarizeSync(
            ProtocolStmt::makeSequence(stmt.elseBody));
        if (thenSync.hasSubgroupBarrier && elseSync.hasSubgroupBarrier) {
          return EpochPair{cur.wg, nextEpochId++};
        }
        return EpochPair{cur.wg,
                         !thenSync.hasSubgroupBarrier ? eThen.sg : eElse.sg};
      }

      // Workgroup-synchronized uniform conditional: `thenGuard` and `elseGuard`
      // are mutually exclusive uniform predicates, so only one branch executes
      // dynamically. If both branches executed at least one workgroup barrier,
      // post-branch code is separated from pre-branch code; if one branch had
      // zero barriers, post-branch code under that branch's guard shares `cur.wg`.
      const SyncSummary thenSync = summarizeSync(
          ProtocolStmt::makeSequence(stmt.thenBody));
      const SyncSummary elseSync = summarizeSync(
          ProtocolStmt::makeSequence(stmt.elseBody));
      if (thenSync.hasWorkgroupBarrier && elseSync.hasWorkgroupBarrier) {
        const std::size_t fresh = nextEpochId++;
        return EpochPair{fresh, fresh};
      }
      return !thenSync.hasWorkgroupBarrier ? eThen : eElse;
    }

    case ProtocolStmtKind::Loop: {
      const SyncSummary sync = summarizeSync(stmt);
      const SymbolicVar lv{SymbolicVarKind::LoopVar,
                           stmt.loopVarName.empty() ? "iv" : stmt.loopVarName};
      SymbolicGuard iterGuard = guard;
      if (stmt.isBoundedFor && stmt.loopLower.isAffine &&
          stmt.loopUpper.isAffine) {
        const auto lvExpr =
            AffineSymbolicExpr::makeVar(SymbolicVarKind::LoopVar, lv.name);
        iterGuard.addComparison(">=", lvExpr, stmt.loopLower);
        iterGuard.addComparison("<", lvExpr, stmt.loopUpper);
      }

      if (!sync.hasWorkgroupBarrier && !sync.hasSubgroupBarrier) {
        for (const auto &c : stmt.children) (void)project(c, cur, iterGuard);
        return cur;
      }

      // Synchronized loop — apply Barrier Aligning:
      // 1. Symbolic iteration `k` starts at `cur` (merging iteration 0's leading
      //    sub-phase with pre-loop accesses in `cur`).
      const std::string k1Name = lv.name + "#a";
      const SymbolicVar k1{SymbolicVarKind::LoopVar, k1Name};
      const auto k1Expr =
          AffineSymbolicExpr::makeVar(SymbolicVarKind::LoopVar, k1Name);

      SymbolicGuard g1 = guard;
      if (stmt.isBoundedFor && stmt.loopLower.isAffine &&
          stmt.loopUpper.isAffine) {
        g1.addComparison(">=", k1Expr, stmt.loopLower);
        g1.addComparison("<", k1Expr, stmt.loopUpper);
      }
      EpochPair afterIter1 = cur;
      for (const auto &c : stmt.children) {
        afterIter1 = project(substituteInStmt(c, lv, k1Expr), afterIter1, g1);
      }

      // 2. Check whether the loop may execute >= 2 iterations; if so, evaluate
      //    iteration `k + step` starting directly at `afterIter1` so iteration k's
      //    trailing sub-phase and iteration k+1's leading sub-phase share
      //    `afterIter1` (catching loop-carried cross-iteration hazards!).
      bool canRepeat = true;
      if (stmt.isBoundedFor && stmt.loopLower.isConstant() &&
          stmt.loopUpper.isConstant() && stmt.loopStep > 0) {
        const std::int64_t span =
            stmt.loopUpper.constant - stmt.loopLower.constant;
        if (span <= stmt.loopStep) canRepeat = false;
      }
      if (!canRepeat) return afterIter1;

      const auto stepExpr = AffineSymbolicExpr::makeConstant(
          stmt.loopStep > 0 ? stmt.loopStep : 1);
      const auto k2Expr = addExpr(k1Expr, stepExpr);
      SymbolicGuard g2 = g1;
      if (stmt.isBoundedFor && stmt.loopUpper.isAffine) {
        g2.addComparison("<", k2Expr, stmt.loopUpper);
      }
      EpochPair afterIter2 = afterIter1;
      for (const auto &c : stmt.children) {
        afterIter2 = project(substituteInStmt(c, lv, k2Expr), afterIter2, g2);
      }
      return afterIter2;
    }
    }
    return cur;
  }
};

// === Symbolic Interval & Pairwise Race / Bounds Decision Procedure ===

struct Interval {
  std::int64_t lo = -1000000;
  std::int64_t hi = 1000000;
  bool empty() const noexcept { return lo > hi; }
};

bool extractGuardIntervals(const SymbolicGuard &guard,
                           std::size_t subgroupSize,
                           std::size_t blockDimX,
                           std::map<SymbolicVar, Interval> &bounds) {
  if (guard.contradiction) return false;

  const auto setDefault = [&](SymbolicVar v, std::int64_t lo, std::int64_t hi) {
    auto [it, inserted] = bounds.emplace(v, Interval{lo, hi});
    if (!inserted) {
      it->second.lo = std::max(it->second.lo, lo);
      it->second.hi = std::min(it->second.hi, hi);
    }
  };

  setDefault({SymbolicVarKind::ThreadIdX, ""}, 0,
             static_cast<std::int64_t>(blockDimX) - 1);
  setDefault({SymbolicVarKind::ThreadIdY, ""}, 0, 0);
  setDefault({SymbolicVarKind::ThreadIdZ, ""}, 0, 0);
  setDefault({SymbolicVarKind::LaneId, ""}, 0,
             static_cast<std::int64_t>(subgroupSize) - 1);
  setDefault({SymbolicVarKind::WarpId, ""}, 0,
             static_cast<std::int64_t>((blockDimX + subgroupSize - 1) /
                                       subgroupSize) -
                 1);
  setDefault({SymbolicVarKind::BlockIdX, ""}, 0, 1024);
  setDefault({SymbolicVarKind::BlockDimX, ""},
             static_cast<std::int64_t>(blockDimX),
             static_cast<std::int64_t>(blockDimX));

  // Propagate single-variable and two-variable bounds iteratively.
  for (int pass = 0; pass < 3; ++pass) {
    for (const auto &c : guard.constraints) {
      if (!c.expr.isAffine || c.expr.modulus != 0) continue;
      if (c.expr.terms.size() == 1) {
        const auto &[var, coeff] = *c.expr.terms.begin();
        if (coeff == 0) continue;
        // coeff * v + constant {==, >=} 0
        if (c.rel == ConstraintRel::EqZero) {
          if ((-c.expr.constant) % coeff != 0) return false;
          const std::int64_t exact = (-c.expr.constant) / coeff;
          setDefault(var, exact, exact);
        } else if (c.rel == ConstraintRel::GeZero) {
          if (coeff > 0) {
            // v >= ceil(-constant / coeff)
            const std::int64_t num = -c.expr.constant;
            const std::int64_t lb =
                num >= 0 ? (num + coeff - 1) / coeff : num / coeff;
            setDefault(var, lb, 1000000);
          } else {
            // (-coeff) * v <= constant => v <= floor(constant / (-coeff))
            const std::int64_t posCoeff = -coeff;
            const std::int64_t num = c.expr.constant;
            const std::int64_t ub =
                num >= 0 ? num / posCoeff : (num - posCoeff + 1) / posCoeff;
            setDefault(var, -1000000, ub);
          }
        }
      }
    }
    // Synchronize ThreadIdX <-> WarpId * subgroupSize + LaneId bounds.
    const SymbolicVar tidVar{SymbolicVarKind::ThreadIdX, ""};
    const SymbolicVar warpVar{SymbolicVarKind::WarpId, ""};
    const SymbolicVar laneVar{SymbolicVarKind::LaneId, ""};
    const auto W = static_cast<std::int64_t>(subgroupSize);
    if (bounds.contains(warpVar) && bounds.contains(laneVar)) {
      const auto wB = bounds[warpVar];
      const auto lB = bounds[laneVar];
      if (!wB.empty() && !lB.empty()) {
        setDefault(tidVar, wB.lo * W + lB.lo, wB.hi * W + lB.hi);
      }
    }
    if (bounds.contains(tidVar) && !bounds[tidVar].empty() && W > 0) {
      const auto tB = bounds[tidVar];
      if (tB.lo >= 0) {
        setDefault(warpVar, tB.lo / W, tB.hi / W);
        if (tB.lo / W == tB.hi / W) {
          setDefault(laneVar, tB.lo % W, tB.hi % W);
        }
      }
    }
    for (const auto &[_, iv] : bounds) {
      if (iv.empty()) return false;
    }
  }

  // Check single-variable `!=` against exact pinned interval `[x, x]`.
  for (const auto &c : guard.constraints) {
    if (c.rel == ConstraintRel::NeZero && c.expr.terms.size() == 1) {
      const auto &[var, coeff] = *c.expr.terms.begin();
      if (bounds.contains(var) && bounds[var].lo == bounds[var].hi) {
        if (coeff * bounds[var].lo + c.expr.constant == 0) return false;
      }
    }
  }
  return true;
}

bool areUniformGuardsMutuallyExclusive(const SymbolicGuard &g1,
                                       const SymbolicGuard &g2) {
  for (const auto &c1 : g1.constraints) {
    for (const auto &c2 : g2.constraints) {
      // Only compare constraints over thread-independent (uniform) variables.
      auto isUniformExpr = [](const AffineSymbolicExpr &e) {
        for (const auto &[v, _] : e.terms) {
          if (v.kind == SymbolicVarKind::ThreadIdX ||
              v.kind == SymbolicVarKind::ThreadIdY ||
              v.kind == SymbolicVarKind::ThreadIdZ ||
              v.kind == SymbolicVarKind::WarpId ||
              v.kind == SymbolicVarKind::LaneId ||
              v.kind == SymbolicVarKind::LoopVar) {
            return false;
          }
        }
        return true;
      };
      if (!isUniformExpr(c1.expr) || !isUniformExpr(c2.expr)) continue;
      if (c1.expr.terms == c2.expr.terms &&
          c1.expr.constant == c2.expr.constant) {
        if ((c1.rel == ConstraintRel::EqZero &&
             c2.rel == ConstraintRel::NeZero) ||
            (c1.rel == ConstraintRel::NeZero &&
             c2.rel == ConstraintRel::EqZero)) {
          return true;
        }
      }
      // Check `E >= 0` vs `-E - 1 >= 0` (`E < 0`).
      const auto sum = addExpr(c1.expr, c2.expr);
      if (c1.rel == ConstraintRel::GeZero && c2.rel == ConstraintRel::GeZero &&
          sum.isConstant() && sum.constant < 0) {
        return true;
      }
    }
  }
  return false;
}

std::optional<std::int64_t>
evalExprConcrete(const AffineSymbolicExpr &expr, std::int64_t tidX,
                 std::int64_t bidX, std::int64_t bdimX, std::int64_t subgroupW,
                 const std::map<std::string, std::int64_t> &loopVals,
                 const std::map<std::string, std::int64_t> &paramVals) {
  if (!expr.isAffine) return std::nullopt;
  std::int64_t acc = expr.constant;
  for (const auto &[var, coeff] : expr.terms) {
    std::int64_t v = 0;
    switch (var.kind) {
    case SymbolicVarKind::ThreadIdX: v = tidX; break;
    case SymbolicVarKind::ThreadIdY:
    case SymbolicVarKind::ThreadIdZ: v = 0; break;
    case SymbolicVarKind::BlockBaseX: v = bidX * bdimX; break;
    case SymbolicVarKind::BlockBaseY:
    case SymbolicVarKind::BlockBaseZ: v = 0; break;
    case SymbolicVarKind::BlockIdX: v = bidX; break;
    case SymbolicVarKind::BlockIdY:
    case SymbolicVarKind::BlockIdZ: v = 0; break;
    case SymbolicVarKind::BlockDimX: v = bdimX; break;
    case SymbolicVarKind::BlockDimY:
    case SymbolicVarKind::BlockDimZ: v = 1; break;
    case SymbolicVarKind::WarpId: v = tidX / subgroupW; break;
    case SymbolicVarKind::LaneId: v = tidX % subgroupW; break;
    case SymbolicVarKind::ScalarParam:
      v = paramVals.contains(var.name) ? paramVals.at(var.name) : 128;
      break;
    case SymbolicVarKind::LoopVar:
      v = loopVals.contains(var.name) ? loopVals.at(var.name) : 0;
      break;
    }
    acc += coeff * v;
  }
  if (expr.modulus > 0) acc = positiveMod(acc, expr.modulus);
  return acc;
}

bool evalGuardConcrete(const SymbolicGuard &guard, std::int64_t tidX,
                       std::int64_t bidX, std::int64_t bdimX,
                       std::int64_t subgroupW,
                       const std::map<std::string, std::int64_t> &loopVals,
                       const std::map<std::string, std::int64_t> &paramVals) {
  if (guard.contradiction) return false;
  for (const auto &c : guard.constraints) {
    const auto val = evalExprConcrete(c.expr, tidX, bidX, bdimX, subgroupW,
                                      loopVals, paramVals);
    if (!val) return false;
    switch (c.rel) {
    case ConstraintRel::EqZero:
      if (*val != 0) return false;
      break;
    case ConstraintRel::NeZero:
      if (*val == 0) return false;
      break;
    case ConstraintRel::GeZero:
      if (*val < 0) return false;
      break;
    }
  }
  return true;
}

AffineSymbolicExpr canonicalizeThreadTerms(const AffineSymbolicExpr &in,
                                           std::size_t subgroupSize) {
  if (!in.isAffine) return in;
  AffineSymbolicExpr out = in;
  const SymbolicVar tidVar{SymbolicVarKind::ThreadIdX, ""};
  const SymbolicVar warpVar{SymbolicVarKind::WarpId, ""};
  const SymbolicVar laneVar{SymbolicVarKind::LaneId, ""};
  const auto W = static_cast<std::int64_t>(subgroupSize);
  // Collapse `W * WarpId + LaneId` into `ThreadIdX`.
  if (out.terms.contains(warpVar) && out.terms.contains(laneVar)) {
    const std::int64_t cLane = out.terms[laneVar];
    if (out.terms[warpVar] == cLane * W) {
      out.terms[tidVar] += cLane;
      out.terms.erase(warpVar);
      out.terms.erase(laneVar);
      pruneZeroTerms(out);
    }
  }
  return out;
}

struct RaceWitness {
  std::int64_t bid1 = 0;
  std::int64_t tid1 = 0;
  std::int64_t bid2 = 0;
  std::int64_t tid2 = 1;
  std::int64_t collidingIndex = 0;
};

// Returns `std::nullopt` when pairwise disjointness is proved; returns a
// `RaceWitness` (or conservative witness) when a race can occur.
std::optional<RaceWitness>
checkPairConflict(const ProjectedAccess &a1, const ProjectedAccess &a2,
                  const GPUMemoryAccessProtocolOptions &options) {
  if (a1.mode == MemoryAccessMode::Read && a2.mode == MemoryAccessMode::Read)
    return std::nullopt;
  if (a1.wgEpoch != a2.wgEpoch) return std::nullopt;

  const bool sameSubgroupOrdered = (a1.sgEpoch != a2.sgEpoch);
  const bool sharedMem = (a1.addressSpace == GPUAddressSpace::WorkgroupShared);
  const bool allowDiffBlocks = !sharedMem && options.multiBlockGrid;

  if (areUniformGuardsMutuallyExclusive(a1.guard, a2.guard))
    return std::nullopt;

  std::map<SymbolicVar, Interval> b1;
  std::map<SymbolicVar, Interval> b2;
  if (!extractGuardIntervals(a1.guard, options.subgroupSize,
                             options.defaultBlockDimX, b1) ||
      !extractGuardIntervals(a2.guard, options.subgroupSize,
                             options.defaultBlockDimX, b2)) {
    return std::nullopt;
  }

  const SymbolicVar tidVar{SymbolicVarKind::ThreadIdX, ""};
  const SymbolicVar warpVar{SymbolicVarKind::WarpId, ""};
  const SymbolicVar laneVar{SymbolicVarKind::LaneId, ""};
  const SymbolicVar bidVar{SymbolicVarKind::BlockIdX, ""};

  // Single-thread or single-warp guard exclusions.
  const bool singleTid =
      (b1[tidVar].lo == b1[tidVar].hi) && (b2[tidVar].lo == b2[tidVar].hi) &&
      (b1[tidVar].lo == b2[tidVar].lo);
  const bool singleBid =
      !allowDiffBlocks ||
      ((b1[bidVar].lo == b1[bidVar].hi) && (b2[bidVar].lo == b2[bidVar].hi) &&
       (b1[bidVar].lo == b2[bidVar].lo));
  const bool singleWarp =
      (b1[warpVar].lo == b1[warpVar].hi) && (b2[warpVar].lo == b2[warpVar].hi) &&
      (b1[warpVar].lo == b2[warpVar].lo);
  if (singleTid && singleBid) return std::nullopt;

  if (sameSubgroupOrdered && singleBid && singleWarp) {
    // Both accesses are restricted to the exact same single warp and separated
    // by a subgroup barrier!
    return std::nullopt;
  }

  const AffineSymbolicExpr i1 =
      canonicalizeThreadTerms(a1.indexExpr, options.subgroupSize);
  const AffineSymbolicExpr i2 =
      canonicalizeThreadTerms(a2.indexExpr, options.subgroupSize);

  if (!i1.isAffine || !i2.isAffine) {
    return RaceWitness{0, 0, 0, 1, 0};
  }

  // Tier 1: Exact algebraic & interval disjointness proofs when modulus == 0.
  if (i1.modulus == 0 && i2.modulus == 0) {
    // Cancel identical uniform scalar parameters (`ScalarParam`, `BlockDim*`,
    // and `BlockId*`/`BlockBase*` when `singleBid` holds).
    std::int64_t constDiff = i1.constant - i2.constant;
    std::vector<std::int64_t> allCoeffs;

    auto isSameAcrossBothThreads = [&](const SymbolicVar &v) {
      if (v.kind == SymbolicVarKind::ScalarParam ||
          v.kind == SymbolicVarKind::BlockDimX ||
          v.kind == SymbolicVarKind::BlockDimY ||
          v.kind == SymbolicVarKind::BlockDimZ) {
        return true;
      }
      if (singleBid &&
          (v.kind == SymbolicVarKind::BlockIdX ||
           v.kind == SymbolicVarKind::BlockBaseX)) {
        return true;
      }
      return false;
    };

    std::map<SymbolicVar, std::int64_t> t1Terms;
    std::map<SymbolicVar, std::int64_t> t2Terms;
    std::set<SymbolicVar> vars;
    for (const auto &[v, _] : i1.terms) vars.insert(v);
    for (const auto &[v, _] : i2.terms) vars.insert(v);

    for (const auto &v : vars) {
      const std::int64_t c1 = i1.terms.contains(v) ? i1.terms.at(v) : 0;
      const std::int64_t c2 = i2.terms.contains(v) ? i2.terms.at(v) : 0;
      if (isSameAcrossBothThreads(v)) {
        const std::int64_t net = c1 - c2;
        if (net != 0) {
          t1Terms[v] = net;
          allCoeffs.push_back(std::abs(net));
        }
      } else {
        if (c1 != 0) {
          t1Terms[v] = c1;
          allCoeffs.push_back(std::abs(c1));
        }
        if (c2 != 0) {
          t2Terms[v] = -c2;
          allCoeffs.push_back(std::abs(c2));
        }
      }
    }

    // 1a. Constant non-zero difference (`smem[0]` vs `smem[1]`).
    if (t1Terms.empty() && t2Terms.empty() && constDiff != 0) {
      return std::nullopt;
    }

    // 1b. Diophantine GCD parity (`2 * i` vs `2 * i + 1`).
    if (!allCoeffs.empty()) {
      std::int64_t g = allCoeffs.front();
      for (std::int64_t c : allCoeffs) g = std::gcd(g, c);
      if (g > 1 && (constDiff % g) != 0) return std::nullopt;
    }

    // 1c. Canonical injective indexing patterns:
    // Pattern A (Shared / single-block): `c * tid.x + uniform` with `constDiff == 0`.
    if (singleBid && constDiff == 0 && t1Terms.size() == 1 &&
        t2Terms.size() == 1 && t1Terms.contains(tidVar) &&
        t2Terms.contains(tidVar) && t1Terms[tidVar] == -t2Terms[tidVar]) {
      return std::nullopt;
    }

    // Pattern A2 (Single-warp intra-subgroup): `c * lane_id + uniform` with `constDiff == 0`.
    if (singleBid && singleWarp && constDiff == 0 && t1Terms.size() == 1 &&
        t2Terms.size() == 1 && t1Terms.contains(laneVar) &&
        t2Terms.contains(laneVar) && t1Terms[laneVar] == -t2Terms[laneVar]) {
      return std::nullopt;
    }

    // Pattern B (Grid-wide global_id): `c * (BlockBaseX + ThreadIdX) + uniform`.
    const SymbolicVar baseVar{SymbolicVarKind::BlockBaseX, ""};
    if (constDiff == 0 && t1Terms.size() == 2 && t2Terms.size() == 2 &&
        t1Terms.contains(baseVar) && t1Terms.contains(tidVar) &&
        t2Terms.contains(baseVar) && t2Terms.contains(tidVar) &&
        t1Terms[baseVar] == t1Terms[tidVar] &&
        t2Terms[baseVar] == t2Terms[tidVar] &&
        t1Terms[tidVar] == -t2Terms[tidVar]) {
      return std::nullopt;
    }

    // Pattern C (Per-block leader write): `tid.x == c` on both and `k * bid.x`.
    if (singleTid && !singleBid && constDiff == 0 && t1Terms.size() == 1 &&
        t2Terms.size() == 1 && t1Terms.contains(bidVar) &&
        t2Terms.contains(bidVar) && t1Terms[bidVar] == -t2Terms[bidVar]) {
      return std::nullopt;
    }

    // Pattern D (Tiled loop in single block): `S * loopVar + tid.x` where `0 <= tid.x < S`.
    if (singleBid && constDiff == 0 && t1Terms.size() == 2 &&
        t2Terms.size() == 2 && t1Terms.contains(tidVar) &&
        t2Terms.contains(tidVar) && t1Terms[tidVar] == 1 &&
        t2Terms[tidVar] == -1) {
      std::optional<SymbolicVar> lv1, lv2;
      for (const auto &[v, _] : t1Terms)
        if (v.kind == SymbolicVarKind::LoopVar) lv1 = v;
      for (const auto &[v, _] : t2Terms)
        if (v.kind == SymbolicVarKind::LoopVar) lv2 = v;
      if (lv1 && lv2 && std::abs(t1Terms[*lv1]) == std::abs(t2Terms[*lv2])) {
        const std::int64_t stride = std::abs(t1Terms[*lv1]);
        if (b1[tidVar].lo >= 0 && b1[tidVar].hi < stride &&
            b2[tidVar].lo >= 0 && b2[tidVar].hi < stride) {
          return std::nullopt;
        }
      }
    }

    // Pattern E (Interval non-overlap, e.g. warp 0 -> [0..31], warp 1 -> [32..63]).
    auto evalInterval = [&](const AffineSymbolicExpr &e,
                            const std::map<SymbolicVar, Interval> &b)
        -> std::optional<Interval> {
      Interval total{e.constant, e.constant};
      for (const auto &[v, coeff] : e.terms) {
        if (!b.contains(v)) return std::nullopt;
        const Interval iv = b.at(v);
        if (coeff >= 0) {
          total.lo += coeff * iv.lo;
          total.hi += coeff * iv.hi;
        } else {
          total.lo += coeff * iv.hi;
          total.hi += coeff * iv.lo;
        }
      }
      return total;
    };
    const auto iv1 = evalInterval(i1, b1);
    const auto iv2 = evalInterval(i2, b2);
    if (iv1 && iv2 && (iv1->hi < iv2->lo || iv2->hi < iv1->lo)) {
      return std::nullopt;
    }
  } else if (singleBid && i1.modulus > 0 && i1.modulus == i2.modulus &&
             i1.terms == i2.terms && i1.constant == i2.constant &&
             i1.terms.size() == 1 && i1.terms.contains(tidVar) &&
             std::gcd(std::abs(i1.terms.at(tidVar)), i1.modulus) == 1) {
    // Bijection modulo M (`(tid.x + c) % M`) when block size <= M.
    if (b1[tidVar].hi - b1[tidVar].lo < i1.modulus &&
        b2[tidVar].hi - b2[tidVar].lo < i2.modulus &&
        b1[tidVar].lo >= 0 && b2[tidVar].lo >= 0 &&
        b1[tidVar].hi < i1.modulus && b2[tidVar].hi < i2.modulus) {
      return std::nullopt;
    }
  }

  // Tier 2: Concrete counterexample witness search over bounded workgroup/grid
  // threads and loop iterations.
  const auto bdim = static_cast<std::int64_t>(options.defaultBlockDimX);
  const auto subW = static_cast<std::int64_t>(options.subgroupSize);
  const std::int64_t maxBid = allowDiffBlocks ? 1 : 0;
  std::map<std::string, std::int64_t> paramVals;
  std::set<std::string> loopNames;
  for (const auto &acc : {a1, a2}) {
    for (const auto &[v, _] : acc.indexExpr.terms) {
      if (v.kind == SymbolicVarKind::LoopVar) loopNames.insert(v.name);
      if (v.kind == SymbolicVarKind::ScalarParam) paramVals[v.name] = 128;
    }
    for (const auto &c : acc.guard.constraints) {
      for (const auto &[v, _] : c.expr.terms) {
        if (v.kind == SymbolicVarKind::LoopVar) loopNames.insert(v.name);
        if (v.kind == SymbolicVarKind::ScalarParam) paramVals[v.name] = 128;
      }
    }
  }

  std::vector<std::map<std::string, std::int64_t>> loopAssignments{{}};
  for (const auto &ln : loopNames) {
    std::vector<std::map<std::string, std::int64_t>> expanded;
    for (const auto &base : loopAssignments) {
      for (std::int64_t lv = 0; lv <= 2; ++lv) {
        auto next = base;
        next[ln] = lv;
        expanded.push_back(std::move(next));
      }
    }
    loopAssignments = std::move(expanded);
    if (loopAssignments.size() > 27) break;
  }

  const std::int64_t maxTidScan = std::min<std::int64_t>(bdim, 64);
  for (const auto &l1 : loopAssignments) {
    for (const auto &l2 : loopAssignments) {
      for (std::int64_t bid1 = 0; bid1 <= maxBid; ++bid1) {
        for (std::int64_t tid1 = 0; tid1 < maxTidScan; ++tid1) {
          if (!evalGuardConcrete(a1.guard, tid1, bid1, bdim, subW, l1,
                                 paramVals)) {
            continue;
          }
          const auto idx1 = evalExprConcrete(a1.indexExpr, tid1, bid1, bdim,
                                             subW, l1, paramVals);
          if (!idx1) continue;

          for (std::int64_t bid2 = 0; bid2 <= maxBid; ++bid2) {
            for (std::int64_t tid2 = 0; tid2 < maxTidScan; ++tid2) {
              if (bid1 == bid2 && tid1 == tid2) continue;
              if (sameSubgroupOrdered && bid1 == bid2 &&
                  (tid1 / subW) == (tid2 / subW)) {
                continue;
              }
              if (!evalGuardConcrete(a2.guard, tid2, bid2, bdim, subW, l2,
                                     paramVals)) {
                continue;
              }
              const auto idx2 = evalExprConcrete(a2.indexExpr, tid2, bid2, bdim,
                                                 subW, l2, paramVals);
              if (idx2 && *idx1 == *idx2) {
                return RaceWitness{bid1, tid1, bid2, tid2, *idx1};
              }
            }
          }
        }
      }
    }
  }

  // Conservative fallback if Tier 1 could not prove unbounded disjointness even
  // though small grid search found no counterexample.
  return RaceWitness{0, 0, 0, 1, -1};
}

void checkSingleAccessBounds(const std::string &kernelName,
                             const ProjectedAccess &acc,
                             const GPUMemoryAccessProtocolOptions &options,
                             GPUProtocolReport &report) {
  if (!acc.staticExtent && !options.requireProvenBounds) return;
  ++report.checkedBoundsObligations;

  if (!acc.indexExpr.isAffine) {
    if (options.requireProvenBounds) {
      recordProtocolViolation(
          report, "gpu.map.unproved_index_bounds", kernelName, acc.bufferName,
          "cannot prove bounds for non-affine index on `" + acc.bufferName + "`",
          acc.location);
    }
    return;
  }

  // 1. Definite constant out-of-bounds on statically-sized buffers.
  if (acc.indexExpr.isConstant()) {
    if (acc.indexExpr.constant < 0 ||
        (acc.staticExtent && acc.indexExpr.constant >= *acc.staticExtent)) {
      recordProtocolViolation(
          report, "gpu.map.out_of_bounds_access", kernelName, acc.bufferName,
          "static index " + std::to_string(acc.indexExpr.constant) +
              " is out of bounds for `" + acc.bufferName + "`",
          acc.location);
      return;
    }
    if (!options.requireProvenBounds) return;
  }

  std::map<SymbolicVar, Interval> bounds;
  if (!extractGuardIntervals(acc.guard, options.subgroupSize,
                             options.defaultBlockDimX, bounds)) {
    return;
  }

  if (acc.staticExtent && acc.indexExpr.modulus == 0) {
    Interval total{acc.indexExpr.constant, acc.indexExpr.constant};
    bool knownAll = true;
    for (const auto &[v, coeff] : acc.indexExpr.terms) {
      if (!bounds.contains(v)) {
        knownAll = false;
        break;
      }
      const Interval iv = bounds.at(v);
      total.lo += (coeff >= 0 ? coeff * iv.lo : coeff * iv.hi);
      total.hi += (coeff >= 0 ? coeff * iv.hi : coeff * iv.lo);
    }
    if (knownAll) {
      if (total.lo >= *acc.staticExtent || total.hi < 0) {
        recordProtocolViolation(
            report, "gpu.map.out_of_bounds_access", kernelName, acc.bufferName,
            "index interval [" + std::to_string(total.lo) + ", " +
                std::to_string(total.hi) + "] is out of bounds [0, " +
                std::to_string(*acc.staticExtent) + ") on `" + acc.bufferName +
                "`",
            acc.location);
        return;
      }
      if (options.requireProvenBounds &&
          (total.lo < 0 || total.hi >= *acc.staticExtent)) {
        recordProtocolViolation(
            report, "gpu.map.unproved_index_bounds", kernelName, acc.bufferName,
            "index interval [" + std::to_string(total.lo) + ", " +
                std::to_string(total.hi) + "] may exceed [0, " +
                std::to_string(*acc.staticExtent) + ") on `" + acc.bufferName +
                "`",
            acc.location);
        return;
      }
    } else if (options.requireProvenBounds) {
      recordProtocolViolation(
          report, "gpu.map.unproved_index_bounds", kernelName, acc.bufferName,
          "index `" + acc.indexExpr.str() +
              "` lacks proven upper/lower bounds on `" + acc.bufferName + "`",
          acc.location);
      return;
    }
  } else if (options.requireProvenBounds && acc.dynamicExtentParam) {
    // Check whether guard contains `acc.dynamicExtentParam - indexExpr - 1 >= 0`
    // (i.e. `indexExpr < extentParam`) and `indexExpr >= 0`.
    const auto extExpr = AffineSymbolicExpr::makeVar(
        SymbolicVarKind::ScalarParam, *acc.dynamicExtentParam);
    const auto neededUpper =
        subExpr(subExpr(extExpr, acc.indexExpr),
                AffineSymbolicExpr::makeConstant(1));
    bool hasUpper = false;
    for (const auto &c : acc.guard.constraints) {
      if (c.rel == ConstraintRel::GeZero &&
          c.expr.terms == neededUpper.terms &&
          c.expr.constant <= neededUpper.constant) {
        // e.g. guard `ext - idx - 1 >= 0` or stronger `ext - idx - 2 >= 0`
        hasUpper = true;
        break;
      }
    }
    if (!hasUpper) {
      recordProtocolViolation(
          report, "gpu.map.unproved_index_bounds", kernelName, acc.bufferName,
          "cannot prove `" + acc.indexExpr.str() + " < " +
              *acc.dynamicExtentParam + "` on `" + acc.bufferName + "`",
          acc.location);
    }
  }
}

// === GraphIR -> ProtocolStmt Extractor ===

class GraphIRProtocolBuilder {
public:
  GraphIRProtocolBuilder(
      const graphir::Module &module,
      const std::map<std::string, const graphir::CallableGraph *> &procedures,
      const GPUMemoryAccessProtocolOptions &options)
      : module_(module), procedures_(procedures), options_(options) {}

  ProtocolStmt buildForKernel(const graphir::CallableGraph &kernel) {
    taints_.clear();
    exprs_.clear();
    bufferNames_.clear();
    bufferSpaces_.clear();
    bufferDynExtents_.clear();
    slotTaints_.clear();
    slotExprs_.clear();

    std::vector<std::string> indexParams;
    for (const auto &param : kernel.parameters()) {
      if (!param.context && param.id && param.type == "index")
        indexParams.push_back(param.name);
    }
    const std::optional<std::string> inferredExtent =
        indexParams.size() == 1 ? std::optional<std::string>(indexParams.front())
                                : std::nullopt;

    for (const auto &param : kernel.parameters()) {
      if (param.context || !param.id) continue;
      if (const auto space = classifyTypeAddressSpace(param.type)) {
        bufferNames_[*param.id] = param.name;
        bufferSpaces_[*param.id] = *space;
        if (*space == GPUAddressSpace::GlobalDevice && inferredExtent)
          bufferDynExtents_[*param.id] = *inferredExtent;
      } else {
        taints_[*param.id] = SIMTUniformity::WorkgroupUniform;
        exprs_[*param.id] =
            AffineSymbolicExpr::makeVar(SymbolicVarKind::ScalarParam, param.name);
      }
    }

    auto stmts = extractSeq(kernel.body(), SIMTUniformity::WorkgroupUniform);
    return ProtocolStmt::makeSequence(std::move(stmts), kernel.location());
  }

private:
  const graphir::Module &module_;
  const std::map<std::string, const graphir::CallableGraph *> &procedures_;
  const GPUMemoryAccessProtocolOptions &options_;
  std::map<graphir::CallableValueId, SIMTUniformity> taints_;
  std::map<graphir::CallableValueId, AffineSymbolicExpr> exprs_;
  std::map<graphir::CallableValueId, std::string> bufferNames_;
  std::map<graphir::CallableValueId, GPUAddressSpace> bufferSpaces_;
  std::map<graphir::CallableValueId, std::string> bufferDynExtents_;
  std::map<graphir::CallableValueId, SIMTUniformity> slotTaints_;
  std::map<graphir::CallableValueId, AffineSymbolicExpr> slotExprs_;
  std::set<std::string> activeCallStack_;

  SIMTUniformity taintOf(graphir::CallableValueId id) const {
    auto it = taints_.find(id);
    return it != taints_.end() ? it->second : SIMTUniformity::WorkgroupUniform;
  }

  AffineSymbolicExpr exprOf(graphir::CallableValueId id) const {
    auto it = exprs_.find(id);
    return it != exprs_.end() ? it->second : AffineSymbolicExpr::nonAffine();
  }

  std::vector<ProtocolStmt>
  extractSeq(const std::vector<graphir::CallableOperation> &ops,
             SIMTUniformity pcTaint) {
    std::vector<ProtocolStmt> out;
    for (const auto &op : ops) {
      extractOp(op, pcTaint, out);
    }
    return out;
  }

  void extractOp(const graphir::CallableOperation &op, SIMTUniformity pcTaint,
                 std::vector<ProtocolStmt> &out) {
    using Kind = graphir::CallableOperationKind;
    switch (op.kind) {
    case Kind::ContextRegion:
    case Kind::Block: {
      auto nested = extractSeq(op.body, pcTaint);
      out.insert(out.end(), std::make_move_iterator(nested.begin()),
                 std::make_move_iterator(nested.end()));
      return;
    }

    case Kind::Literal: {
      taints_[op.id] = SIMTUniformity::WorkgroupUniform;
      if (op.attributes.contains("value")) {
        if (const auto iv = parseSignedInt(op.attributes.at("value"))) {
          exprs_[op.id] = AffineSymbolicExpr::makeConstant(*iv);
          return;
        }
      }
      exprs_[op.id] = AffineSymbolicExpr::nonAffine();
      return;
    }

    case Kind::Variable: {
      const bool isSlot = op.attributes.contains("mutable_local") &&
                          op.attributes.at("mutable_local") == "true";
      SIMTUniformity initTaint = pcTaint;
      AffineSymbolicExpr initExpr = AffineSymbolicExpr::nonAffine();
      if (!op.operands.empty()) {
        const auto srcId = op.operands[0];
        if (bufferNames_.contains(srcId)) {
          bufferNames_[op.id] = bufferNames_[srcId];
          bufferSpaces_[op.id] = bufferSpaces_[srcId];
          if (bufferDynExtents_.contains(srcId))
            bufferDynExtents_[op.id] = bufferDynExtents_[srcId];
          return;
        }
        initTaint = joinUniformity(pcTaint, taintOf(srcId));
        initExpr = exprOf(srcId);
      } else if (op.attributes.contains("initializer.value")) {
        if (const auto iv =
                parseSignedInt(op.attributes.at("initializer.value"))) {
          initExpr = AffineSymbolicExpr::makeConstant(*iv);
        }
      }
      if (isSlot) {
        slotTaints_[op.id] = initTaint;
        slotExprs_[op.id] = initExpr;
      } else {
        taints_[op.id] = initTaint;
        exprs_[op.id] = initExpr;
      }
      return;
    }

    case Kind::Load: {
      if (!op.operands.empty()) {
        const auto slotId = op.operands[0];
        taints_[op.id] = slotTaints_.contains(slotId)
                             ? slotTaints_[slotId]
                             : SIMTUniformity::WorkgroupUniform;
        exprs_[op.id] = slotExprs_.contains(slotId)
                            ? slotExprs_[slotId]
                            : AffineSymbolicExpr::nonAffine();
      }
      return;
    }

    case Kind::Assign: {
      if (op.operands.size() >= 2) {
        const auto slotId = op.operands[0];
        const auto valId = op.operands[1];
        slotTaints_[slotId] = joinUniformity(
            slotTaints_[slotId], joinUniformity(pcTaint, taintOf(valId)));
        slotExprs_[slotId] = AffineSymbolicExpr::nonAffine();
      }
      return;
    }

    case Kind::Unary: {
      if (op.operands.empty()) return;
      taints_[op.id] = taintOf(op.operands[0]);
      const std::string uop =
          op.attributes.contains("operator") ? op.attributes.at("operator") : "";
      if (uop == "-") exprs_[op.id] = negExpr(exprOf(op.operands[0]));
      else if (uop == "+") exprs_[op.id] = exprOf(op.operands[0]);
      else exprs_[op.id] = AffineSymbolicExpr::nonAffine();
      return;
    }

    case Kind::Binary: {
      if (op.operands.size() < 2) return;
      const auto lhsId = op.operands[0];
      const auto rhsId = op.operands[1];
      const auto lhsE = exprOf(lhsId);
      const auto rhsE = exprOf(rhsId);
      const std::string bop =
          op.attributes.contains("operator") ? op.attributes.at("operator") : "";

      SIMTUniformity resTaint = joinUniformity(taintOf(lhsId), taintOf(rhsId));
      AffineSymbolicExpr resExpr = AffineSymbolicExpr::nonAffine();
      if (bop == "+") resExpr = addExpr(lhsE, rhsE);
      else if (bop == "-") resExpr = subExpr(lhsE, rhsE);
      else if (bop == "*") resExpr = mulExpr(lhsE, rhsE, options_.subgroupSize);
      else if (bop == "/") {
        resExpr = divExpr(lhsE, rhsE, options_.subgroupSize);
        if (resExpr.isAffine && resExpr.terms.size() == 1 &&
            resExpr.terms.begin()->first.kind == SymbolicVarKind::WarpId) {
          resTaint = SIMTUniformity::SubgroupUniform;
        }
      } else if (bop == "%") {
        resExpr = remExpr(lhsE, rhsE, options_.subgroupSize);
      }
      taints_[op.id] = resTaint;
      exprs_[op.id] = resExpr;
      cmpOps_[op.id] = {bop, lhsE, rhsE};
      return;
    }

    case Kind::Call: {
      const std::string ck =
          op.attributes.contains("callee_kind") ? op.attributes.at("callee_kind")
                                                : "procedure";
      if (ck == "gpu_builtin") {
        if (op.name == "gpu::global_id") {
          taints_[op.id] = SIMTUniformity::LaneDivergent;
          exprs_[op.id] = AffineSymbolicExpr::makeGlobalLinearIdX();
        } else if (op.name == "gpu::thread_id") {
          taints_[op.id] = SIMTUniformity::LaneDivergent;
          exprs_[op.id] =
              AffineSymbolicExpr::makeVar(SymbolicVarKind::ThreadIdX);
        } else if (op.name == "gpu::block_id") {
          taints_[op.id] = SIMTUniformity::WorkgroupUniform;
          exprs_[op.id] =
              AffineSymbolicExpr::makeVar(SymbolicVarKind::BlockIdX);
        } else if (op.name == "gpu::block_dim") {
          taints_[op.id] = SIMTUniformity::WorkgroupUniform;
          exprs_[op.id] =
              AffineSymbolicExpr::makeVar(SymbolicVarKind::BlockDimX);
        } else if (op.name == "gpu::barrier") {
          out.push_back(ProtocolStmt::makeWorkgroupBarrier(pcTaint, op.location));
        } else if (op.name == "gpu::warp_barrier" ||
                   op.name == "gpu::warp_shuffle" ||
                   op.name == "gpu::warp_vote") {
          out.push_back(ProtocolStmt::makeSubgroupBarrier(pcTaint, op.name,
                                                          op.location));
        } else if (op.name == "gpu::load" && op.operands.size() >= 2) {
          const auto bufId = op.operands[0];
          const auto idxId = op.operands[1];
          taints_[op.id] = joinUniformity(pcTaint, taintOf(idxId));
          exprs_[op.id] = AffineSymbolicExpr::nonAffine();
          const std::string buf =
              bufferNames_.contains(bufId) ? bufferNames_[bufId] : "span";
          const GPUAddressSpace sp = bufferSpaces_.contains(bufId)
                                         ? bufferSpaces_[bufId]
                                         : GPUAddressSpace::GlobalDevice;
          const std::optional<std::string> dynExt =
              bufferDynExtents_.contains(bufId)
                  ? std::optional<std::string>(bufferDynExtents_[bufId])
                  : std::nullopt;
          out.push_back(ProtocolStmt::makeAccess(
              MemoryAccessMode::Read, buf, sp, exprOf(idxId), op.location,
              std::nullopt, dynExt));
        } else if (op.name == "gpu::store" && op.operands.size() >= 3) {
          const auto bufId = op.operands[0];
          const auto idxId = op.operands[1];
          const std::string buf =
              bufferNames_.contains(bufId) ? bufferNames_[bufId] : "span";
          const GPUAddressSpace sp = bufferSpaces_.contains(bufId)
                                         ? bufferSpaces_[bufId]
                                         : GPUAddressSpace::GlobalDevice;
          const std::optional<std::string> dynExt =
              bufferDynExtents_.contains(bufId)
                  ? std::optional<std::string>(bufferDynExtents_[bufId])
                  : std::nullopt;
          out.push_back(ProtocolStmt::makeAccess(
              MemoryAccessMode::Write, buf, sp, exprOf(idxId), op.location,
              std::nullopt, dynExt));
        }
        return;
      }

      if (ck == "procedure" && procedures_.contains(op.name) &&
          activeCallStack_.insert(op.name).second) {
        const auto *callee = procedures_.at(op.name);
        auto savedTaints = taints_;
        auto savedExprs = exprs_;
        auto savedBufs = bufferNames_;
        auto savedSpaces = bufferSpaces_;
        auto savedExtents = bufferDynExtents_;

        std::size_t runtimeIdx = 0;
        for (const auto &param : callee->parameters()) {
          if (param.context || !param.id) continue;
          if (runtimeIdx >= op.operands.size()) break;
          const auto argId = op.operands[runtimeIdx++];
          if (savedBufs.contains(argId)) {
            bufferNames_[*param.id] = savedBufs[argId];
            bufferSpaces_[*param.id] = savedSpaces[argId];
            if (savedExtents.contains(argId))
              bufferDynExtents_[*param.id] = savedExtents[argId];
            else
              bufferDynExtents_.erase(*param.id);
          } else {
            taints_[*param.id] =
                savedTaints.contains(argId)
                    ? savedTaints[argId]
                    : SIMTUniformity::WorkgroupUniform;
            exprs_[*param.id] = savedExprs.contains(argId)
                                    ? savedExprs[argId]
                                    : AffineSymbolicExpr::nonAffine();
          }
        }
        auto calleeStmts = extractSeq(callee->body(), pcTaint);
        out.insert(out.end(), std::make_move_iterator(calleeStmts.begin()),
                   std::make_move_iterator(calleeStmts.end()));

        taints_ = std::move(savedTaints);
        exprs_ = std::move(savedExprs);
        bufferNames_ = std::move(savedBufs);
        bufferSpaces_ = std::move(savedSpaces);
        bufferDynExtents_ = std::move(savedExtents);
        activeCallStack_.erase(op.name);

        SIMTUniformity callTaint = pcTaint;
        for (const auto argId : op.operands) {
          callTaint = joinUniformity(callTaint, taintOf(argId));
        }
        taints_[op.id] = callTaint;
        exprs_[op.id] = AffineSymbolicExpr::nonAffine();
      }
      return;
    }

    case Kind::If:
    case Kind::FairChoice: {
      auto condSideEffects = extractSeq(op.condition, pcTaint);
      out.insert(out.end(), std::make_move_iterator(condSideEffects.begin()),
                 std::make_move_iterator(condSideEffects.end()));
      SIMTUniformity condTaint = pcTaint;
      SymbolicGuard thenDelta;
      SymbolicGuard elseDelta;
      if (!op.operands.empty()) {
        const auto condId = op.operands[0];
        condTaint = joinUniformity(pcTaint, taintOf(condId));
        if (cmpOps_.contains(condId)) {
          const auto &[cmpOp, lhsE, rhsE] = cmpOps_[condId];
          thenDelta.addComparison(cmpOp, lhsE, rhsE);
          const auto inv = invertComparison(cmpOp);
          if (!inv.empty()) elseDelta.addComparison(inv, lhsE, rhsE);
        }
      }
      auto thenBody = extractSeq(op.body, condTaint);
      auto elseBody = extractSeq(op.alternative, condTaint);
      out.push_back(ProtocolStmt::makeConditional(
          condTaint, std::move(thenDelta), std::move(elseDelta),
          std::move(thenBody), std::move(elseBody), op.location));
      return;
    }

    case Kind::For: {
      // Lowering structure for range `for`: initialization has `[lo, hi, $range_index]`.
      auto initStmts = extractSeq(op.initialization, pcTaint);
      out.insert(out.end(), std::make_move_iterator(initStmts.begin()),
                 std::make_move_iterator(initStmts.end()));
      auto condStmts = extractSeq(op.condition, pcTaint);
      out.insert(out.end(), std::make_move_iterator(condStmts.begin()),
                 std::make_move_iterator(condStmts.end()));

      SIMTUniformity headerTaint = pcTaint;
      AffineSymbolicExpr loExpr = AffineSymbolicExpr::makeConstant(0);
      AffineSymbolicExpr hiExpr = AffineSymbolicExpr::nonAffine();
      bool bounded = false;
      const std::string ivName = "for_" + std::to_string(op.id);

      if (!op.operands.empty() && cmpOps_.contains(op.operands[0])) {
        headerTaint = joinUniformity(pcTaint, taintOf(op.operands[0]));
        const auto &[_, lhsE, rhsE] = cmpOps_[op.operands[0]];
        (void)lhsE;
        hiExpr = rhsE;
      }
      for (const auto &initOp : op.initialization) {
        if (initOp.kind == Kind::Variable && initOp.name == "$range_index" &&
            !initOp.operands.empty()) {
          loExpr = exprOf(initOp.operands[0]);
          headerTaint = joinUniformity(headerTaint, taintOf(initOp.operands[0]));
          slotTaints_[initOp.id] = headerTaint;
          slotExprs_[initOp.id] =
              AffineSymbolicExpr::makeVar(SymbolicVarKind::LoopVar, ivName);
          bounded = loExpr.isAffine && hiExpr.isAffine;
        }
      }

      auto bodyStmts = extractSeq(op.body, headerTaint);
      out.push_back(ProtocolStmt::makeLoop(ivName, loExpr, hiExpr, 1,
                                           headerTaint, bounded,
                                           std::move(bodyStmts), op.location));
      return;
    }

    case Kind::While:
    case Kind::Until:
    case Kind::Loop: {
      auto condStmts = extractSeq(op.condition, pcTaint);
      out.insert(out.end(), std::make_move_iterator(condStmts.begin()),
                 std::make_move_iterator(condStmts.end()));
      SIMTUniformity headerTaint = pcTaint;
      if (!op.operands.empty()) {
        headerTaint = joinUniformity(pcTaint, taintOf(op.operands[0]));
      }
      // Also capture break conditions inside `Loop`!
      std::function<void(const std::vector<graphir::CallableOperation> &,
                         SIMTUniformity)>
          scanBreaks = [&](const auto &seq, SIMTUniformity cur) {
            for (const auto &item : seq) {
              if (item.kind == Kind::Break) {
                headerTaint = joinUniformity(headerTaint, cur);
              } else if (item.kind == Kind::If) {
                SIMTUniformity sub =
                    item.operands.empty()
                        ? cur
                        : joinUniformity(cur, taintOf(item.operands[0]));
                scanBreaks(item.body, sub);
                scanBreaks(item.alternative, sub);
              } else if (item.kind == Kind::Block ||
                         item.kind == Kind::ContextRegion) {
                scanBreaks(item.body, cur);
              }
            }
          };
      // Pre-evaluate scalar expressions once so `scanBreaks` sees inner condition taints.
      (void)extractSeq(op.body, headerTaint);
      scanBreaks(op.body, headerTaint);
      auto bodyStmts = extractSeq(op.body, headerTaint);
      out.push_back(ProtocolStmt::makeLoop(
          "while_" + std::to_string(op.id), AffineSymbolicExpr::nonAffine(),
          AffineSymbolicExpr::nonAffine(), 1, headerTaint, /*boundedFor=*/false,
          std::move(bodyStmts), op.location));
      return;
    }

    default:
      return;
    }
  }

  struct CmpRecord {
    std::string op;
    AffineSymbolicExpr lhs;
    AffineSymbolicExpr rhs;
  };
  std::map<graphir::CallableValueId, CmpRecord> cmpOps_;
};

} // namespace

const char *simtUniformityName(SIMTUniformity level) noexcept {
  switch (level) {
  case SIMTUniformity::WorkgroupUniform: return "workgroup-uniform";
  case SIMTUniformity::SubgroupUniform: return "subgroup-uniform";
  case SIMTUniformity::LaneDivergent: return "lane-divergent";
  }
  return "lane-divergent";
}

AffineSymbolicExpr AffineSymbolicExpr::makeConstant(std::int64_t value) {
  AffineSymbolicExpr e;
  e.constant = value;
  return e;
}

AffineSymbolicExpr AffineSymbolicExpr::makeVar(SymbolicVarKind kind,
                                               std::string name) {
  AffineSymbolicExpr e;
  e.terms[SymbolicVar{kind, std::move(name)}] = 1;
  return e;
}

AffineSymbolicExpr AffineSymbolicExpr::makeGlobalLinearIdX() {
  AffineSymbolicExpr e;
  e.terms[SymbolicVar{SymbolicVarKind::BlockBaseX, ""}] = 1;
  e.terms[SymbolicVar{SymbolicVarKind::ThreadIdX, ""}] = 1;
  return e;
}

AffineSymbolicExpr AffineSymbolicExpr::nonAffine() {
  AffineSymbolicExpr e;
  e.isAffine = false;
  return e;
}

AffineSymbolicExpr
AffineSymbolicExpr::substitute(const SymbolicVar &var,
                               const AffineSymbolicExpr &replacement) const {
  if (!isAffine) return *this;
  auto it = terms.find(var);
  if (it == terms.end()) return *this;
  if (!replacement.isAffine || replacement.modulus != 0) return nonAffine();

  const std::int64_t coeff = it->second;
  AffineSymbolicExpr out = *this;
  out.terms.erase(var);
  out.constant += coeff * replacement.constant;
  for (const auto &[rv, rc] : replacement.terms) {
    out.terms[rv] += coeff * rc;
  }
  pruneZeroTerms(out);
  if (out.modulus > 0 && out.terms.empty()) {
    out.constant = positiveMod(out.constant, out.modulus);
    out.modulus = 0;
  }
  return out;
}

std::string AffineSymbolicExpr::str() const {
  if (!isAffine) return "<non-affine>";
  std::ostringstream os;
  if (modulus > 0) os << "(";
  bool first = true;
  if (constant != 0 || terms.empty()) {
    os << constant;
    first = false;
  }
  for (const auto &[var, coeff] : terms) {
    if (!first) os << (coeff >= 0 ? " + " : " - ");
    else if (coeff < 0) os << "-";
    first = false;
    const std::int64_t mag = std::abs(coeff);
    if (mag != 1) os << mag << "*";
    switch (var.kind) {
    case SymbolicVarKind::ThreadIdX: os << "tid.x"; break;
    case SymbolicVarKind::ThreadIdY: os << "tid.y"; break;
    case SymbolicVarKind::ThreadIdZ: os << "tid.z"; break;
    case SymbolicVarKind::BlockBaseX: os << "(bid.x*bdim.x)"; break;
    case SymbolicVarKind::BlockBaseY: os << "(bid.y*bdim.y)"; break;
    case SymbolicVarKind::BlockBaseZ: os << "(bid.z*bdim.z)"; break;
    case SymbolicVarKind::BlockIdX: os << "bid.x"; break;
    case SymbolicVarKind::BlockIdY: os << "bid.y"; break;
    case SymbolicVarKind::BlockIdZ: os << "bid.z"; break;
    case SymbolicVarKind::BlockDimX: os << "bdim.x"; break;
    case SymbolicVarKind::BlockDimY: os << "bdim.y"; break;
    case SymbolicVarKind::BlockDimZ: os << "bdim.z"; break;
    case SymbolicVarKind::WarpId: os << "warp_id"; break;
    case SymbolicVarKind::LaneId: os << "lane_id"; break;
    case SymbolicVarKind::ScalarParam:
    case SymbolicVarKind::LoopVar: os << var.name; break;
    }
  }
  if (modulus > 0) os << ") % " << modulus;
  return os.str();
}

AffineSymbolicExpr addExpr(const AffineSymbolicExpr &lhs,
                           const AffineSymbolicExpr &rhs) {
  if (!lhs.isAffine || !rhs.isAffine || lhs.modulus != 0 || rhs.modulus != 0)
    return AffineSymbolicExpr::nonAffine();
  AffineSymbolicExpr out = lhs;
  out.constant += rhs.constant;
  for (const auto &[v, c] : rhs.terms) out.terms[v] += c;
  pruneZeroTerms(out);
  return out;
}

AffineSymbolicExpr subExpr(const AffineSymbolicExpr &lhs,
                           const AffineSymbolicExpr &rhs) {
  if (!lhs.isAffine || !rhs.isAffine || lhs.modulus != 0 || rhs.modulus != 0)
    return AffineSymbolicExpr::nonAffine();
  AffineSymbolicExpr out = lhs;
  out.constant -= rhs.constant;
  for (const auto &[v, c] : rhs.terms) out.terms[v] -= c;
  pruneZeroTerms(out);
  return out;
}

AffineSymbolicExpr negExpr(const AffineSymbolicExpr &expr) {
  return subExpr(AffineSymbolicExpr::makeConstant(0), expr);
}

AffineSymbolicExpr mulExpr(const AffineSymbolicExpr &lhs,
                           const AffineSymbolicExpr &rhs,
                           std::size_t subgroupSize) {
  (void)subgroupSize;
  if (!lhs.isAffine || !rhs.isAffine || lhs.modulus != 0 || rhs.modulus != 0)
    return AffineSymbolicExpr::nonAffine();
  if (lhs.isConstant()) {
    if (lhs.constant == 0) return AffineSymbolicExpr::makeConstant(0);
    AffineSymbolicExpr out = rhs;
    out.constant *= lhs.constant;
    for (auto &[_, c] : out.terms) c *= lhs.constant;
    pruneZeroTerms(out);
    return out;
  }
  if (rhs.isConstant()) return mulExpr(rhs, lhs, subgroupSize);

  // Canonicalize `block_id.{x,y,z} * block_dim.{x,y,z}` -> `BlockBase{X,Y,Z}`.
  if (lhs.constant == 0 && rhs.constant == 0 && lhs.terms.size() == 1 &&
      rhs.terms.size() == 1) {
    const auto &[v1, c1] = *lhs.terms.begin();
    const auto &[v2, c2] = *rhs.terms.begin();
    auto matchAxis = [&](SymbolicVarKind idK, SymbolicVarKind dimK,
                         SymbolicVarKind baseK) -> std::optional<AffineSymbolicExpr> {
      if ((v1.kind == idK && v2.kind == dimK) ||
          (v1.kind == dimK && v2.kind == idK)) {
        AffineSymbolicExpr out;
        out.terms[SymbolicVar{baseK, ""}] = c1 * c2;
        return out;
      }
      return std::nullopt;
    };
    if (auto x = matchAxis(SymbolicVarKind::BlockIdX, SymbolicVarKind::BlockDimX,
                           SymbolicVarKind::BlockBaseX))
      return *x;
    if (auto y = matchAxis(SymbolicVarKind::BlockIdY, SymbolicVarKind::BlockDimY,
                           SymbolicVarKind::BlockBaseY))
      return *y;
    if (auto z = matchAxis(SymbolicVarKind::BlockIdZ, SymbolicVarKind::BlockDimZ,
                           SymbolicVarKind::BlockBaseZ))
      return *z;
  }
  return AffineSymbolicExpr::nonAffine();
}

AffineSymbolicExpr divExpr(const AffineSymbolicExpr &lhs,
                           const AffineSymbolicExpr &rhs,
                           std::size_t subgroupSize) {
  if (!lhs.isAffine || !rhs.isAffine || !rhs.isConstant() || rhs.constant <= 0 ||
      lhs.modulus != 0) {
    return AffineSymbolicExpr::nonAffine();
  }
  if (lhs.isConstant()) {
    return AffineSymbolicExpr::makeConstant(lhs.constant / rhs.constant);
  }
  if (lhs.constant == 0 && lhs.terms.size() == 1) {
    const auto &[v, c] = *lhs.terms.begin();
    if (v.kind == SymbolicVarKind::ThreadIdX && c == 1 &&
        rhs.constant == static_cast<std::int64_t>(subgroupSize)) {
      return AffineSymbolicExpr::makeVar(SymbolicVarKind::WarpId);
    }
    if (c % rhs.constant == 0) {
      AffineSymbolicExpr out;
      out.terms[v] = c / rhs.constant;
      return out;
    }
  }
  return AffineSymbolicExpr::nonAffine();
}

AffineSymbolicExpr remExpr(const AffineSymbolicExpr &lhs,
                           const AffineSymbolicExpr &rhs,
                           std::size_t subgroupSize) {
  if (!lhs.isAffine || !rhs.isAffine || !rhs.isConstant() || rhs.constant <= 0 ||
      lhs.modulus != 0) {
    return AffineSymbolicExpr::nonAffine();
  }
  if (lhs.isConstant()) {
    return AffineSymbolicExpr::makeConstant(lhs.constant % rhs.constant);
  }
  if (lhs.constant == 0 && lhs.terms.size() == 1) {
    const auto &[v, c] = *lhs.terms.begin();
    if (v.kind == SymbolicVarKind::ThreadIdX && c == 1 &&
        rhs.constant == static_cast<std::int64_t>(subgroupSize)) {
      return AffineSymbolicExpr::makeVar(SymbolicVarKind::LaneId);
    }
  }
  AffineSymbolicExpr modExpr = lhs;
  modExpr.modulus = rhs.constant;
  return modExpr;
}

void SymbolicGuard::addComparison(std::string_view op,
                                  const AffineSymbolicExpr &lhs,
                                  const AffineSymbolicExpr &rhs) {
  if (!lhs.isAffine || !rhs.isAffine || lhs.modulus != 0 || rhs.modulus != 0)
    return;
  AffineConstraint c;
  if (op == "==") {
    c.rel = ConstraintRel::EqZero;
    c.expr = subExpr(lhs, rhs);
  } else if (op == "!=") {
    c.rel = ConstraintRel::NeZero;
    c.expr = subExpr(lhs, rhs);
  } else if (op == "<") {
    c.rel = ConstraintRel::GeZero;
    c.expr = subExpr(subExpr(rhs, lhs), AffineSymbolicExpr::makeConstant(1));
  } else if (op == "<=") {
    c.rel = ConstraintRel::GeZero;
    c.expr = subExpr(rhs, lhs);
  } else if (op == ">") {
    c.rel = ConstraintRel::GeZero;
    c.expr = subExpr(subExpr(lhs, rhs), AffineSymbolicExpr::makeConstant(1));
  } else if (op == ">=") {
    c.rel = ConstraintRel::GeZero;
    c.expr = subExpr(lhs, rhs);
  } else {
    return;
  }

  if (c.expr.terms.empty()) {
    const bool ok =
        (c.rel == ConstraintRel::EqZero && c.expr.constant == 0) ||
        (c.rel == ConstraintRel::NeZero && c.expr.constant != 0) ||
        (c.rel == ConstraintRel::GeZero && c.expr.constant >= 0);
    if (!ok) contradiction = true;
    return;
  }
  constraints.push_back(std::move(c));
}

SymbolicGuard SymbolicGuard::withComparison(
    std::string_view op, const AffineSymbolicExpr &lhs,
    const AffineSymbolicExpr &rhs) const {
  SymbolicGuard out = *this;
  out.addComparison(op, lhs, rhs);
  return out;
}

ProtocolStmt ProtocolStmt::makeAccess(MemoryAccessMode mode, std::string buffer,
                                      GPUAddressSpace space,
                                      AffineSymbolicExpr index,
                                      SourceLocation loc,
                                      std::optional<std::int64_t> extent,
                                      std::optional<std::string> dynExtentParam) {
  ProtocolStmt s;
  s.kind = ProtocolStmtKind::Access;
  s.accessMode = mode;
  s.bufferName = std::move(buffer);
  s.addressSpace = space;
  s.staticExtent = extent;
  s.dynamicExtentParam = std::move(dynExtentParam);
  s.indexExpr = std::move(index);
  s.location = loc;
  return s;
}

ProtocolStmt ProtocolStmt::makeWorkgroupBarrier(SIMTUniformity pcUniformity,
                                                SourceLocation loc) {
  ProtocolStmt s;
  s.kind = ProtocolStmtKind::WorkgroupBarrier;
  s.uniformity = pcUniformity;
  s.location = loc;
  return s;
}

ProtocolStmt ProtocolStmt::makeSubgroupBarrier(SIMTUniformity pcUniformity,
                                               std::string kindName,
                                               SourceLocation loc) {
  ProtocolStmt s;
  s.kind = ProtocolStmtKind::SubgroupBarrier;
  s.uniformity = pcUniformity;
  s.barrierKindName = std::move(kindName);
  s.location = loc;
  return s;
}

ProtocolStmt ProtocolStmt::makeSequence(std::vector<ProtocolStmt> items,
                                        SourceLocation loc) {
  ProtocolStmt s;
  s.kind = ProtocolStmtKind::Sequence;
  s.children = std::move(items);
  s.location = loc;
  return s;
}

ProtocolStmt ProtocolStmt::makeConditional(SIMTUniformity condUniformity,
                                           SymbolicGuard thenDelta,
                                           SymbolicGuard elseDelta,
                                           std::vector<ProtocolStmt> thenStmts,
                                           std::vector<ProtocolStmt> elseStmts,
                                           SourceLocation loc) {
  ProtocolStmt s;
  s.kind = ProtocolStmtKind::Conditional;
  s.uniformity = condUniformity;
  s.thenGuard = std::move(thenDelta);
  s.elseGuard = std::move(elseDelta);
  s.thenBody = std::move(thenStmts);
  s.elseBody = std::move(elseStmts);
  s.location = loc;
  return s;
}

ProtocolStmt ProtocolStmt::makeLoop(std::string ivName,
                                    AffineSymbolicExpr lower,
                                    AffineSymbolicExpr upper, std::int64_t step,
                                    SIMTUniformity headerUniformity,
                                    bool boundedFor,
                                    std::vector<ProtocolStmt> bodyStmts,
                                    SourceLocation loc) {
  ProtocolStmt s;
  s.kind = ProtocolStmtKind::Loop;
  s.loopVarName = std::move(ivName);
  s.loopLower = std::move(lower);
  s.loopUpper = std::move(upper);
  s.loopStep = step;
  s.uniformity = headerUniformity;
  s.isBoundedFor = boundedFor;
  s.children = std::move(bodyStmts);
  s.location = loc;
  return s;
}

void verifyKernelAccessProtocol(const std::string &kernelName,
                                const ProtocolStmt &root,
                                const GPUMemoryAccessProtocolOptions &options,
                                GPUProtocolReport &report) {
  ++report.checkedKernels;
  const std::size_t violationsBefore = report.violations.size();

  // 1. Hierarchical SIMT barrier-uniformity & unambiguous participation check.
  verifyUniformityTree(kernelName, root, SIMTUniformity::WorkgroupUniform,
                       report);

  // Do not run aligned phase projection on kernels whose barriers are already
  // divergent (phase boundaries are ill-defined under divergent barriers).
  if (report.violations.size() != violationsBefore) return;

  // 2. Two-level (workgroup + subgroup) phase projection with loop barrier
  //    aligning.
  PhaseProjector projector;
  (void)projector.project(root, {0, 0}, SymbolicGuard{});

  std::set<std::pair<std::size_t, std::size_t>> distinctPhases;
  for (const auto &acc : projector.accesses) {
    distinctPhases.emplace(acc.wgEpoch, acc.sgEpoch);
    checkSingleAccessBounds(kernelName, acc, options, report);
  }
  report.checkedPhases += std::max<std::size_t>(1, distinctPhases.size());

  // 3. Pairwise symbolic data-race freedom check across projected accesses.
  std::set<std::string> reportedBuffersInEpoch;
  for (std::size_t i = 0; i < projector.accesses.size(); ++i) {
    for (std::size_t j = i; j < projector.accesses.size(); ++j) {
      const auto &a1 = projector.accesses[i];
      const auto &a2 = projector.accesses[j];
      if (a1.bufferName != a2.bufferName) continue;
      if (a1.wgEpoch != a2.wgEpoch) continue;
      if (a1.mode == MemoryAccessMode::Read &&
          a2.mode == MemoryAccessMode::Read) {
        continue;
      }

      ++report.checkedAccessPairs;
      const auto witness = checkPairConflict(a1, a2, options);
      if (!witness) continue;

      const std::string dedupKey =
          a1.bufferName + "#" + std::to_string(a1.wgEpoch);
      if (!reportedBuffersInEpoch.insert(dedupKey).second) continue;

      const bool writeWrite = (a1.mode == MemoryAccessMode::Write &&
                               a2.mode == MemoryAccessMode::Write);
      const char *code =
          (!a1.indexExpr.isAffine || !a2.indexExpr.isAffine)
              ? "gpu.map.non_affine_write_index"
              : (writeWrite ? "gpu.map.write_write_data_race"
                            : "gpu.map.read_write_data_race");

      std::ostringstream msg;
      msg << (writeWrite ? "unsynchronized write-write data race"
                         : "unsynchronized read-write data race")
          << " on `" << a1.bufferName << "` between `" << a1.indexExpr.str()
          << "` and `" << a2.indexExpr.str() << "`";
      if (witness->collidingIndex >= 0) {
        msg << " (witness threads (" << witness->bid1 << ", " << witness->tid1
            << ") and (" << witness->bid2 << ", " << witness->tid2
            << ") collide at index " << witness->collidingIndex << ")";
      }
      recordProtocolViolation(report, code, kernelName, a1.bufferName,
                              msg.str(), a2.location);
    }
  }
}

GPUProtocolReport verifyGPUMemoryAccessProtocol(
    const graphir::Module &module,
    const GPUMemoryAccessProtocolOptions &options) {
  GPUProtocolReport report;
  std::map<std::string, const graphir::CallableGraph *> procedures;
  for (const auto &proc : module.procedureGraphs()) {
    procedures.emplace(proc.name(), &proc);
  }

  GraphIRProtocolBuilder builder(module, procedures, options);
  for (const auto &kernel : module.kernelGraphs()) {
    const auto protocol = builder.buildForKernel(kernel);
    verifyKernelAccessProtocol(kernel.name(), protocol, options, report);
  }
  return report;
}

} // namespace metreon::verification
