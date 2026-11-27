#pragma once

#include "metreon/Basic/SourceLocation.h"
#include "metreon/GraphIR/Graph.h"
#include "metreon/Verification/GPUProfile.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace metreon::verification {

// Three-point SIMT participation / divergence taint lattice:
//   WorkgroupUniform < SubgroupUniform < LaneDivergent
enum class SIMTUniformity : std::uint8_t {
  WorkgroupUniform = 0,
  SubgroupUniform = 1,
  LaneDivergent = 2,
};

constexpr SIMTUniformity joinUniformity(SIMTUniformity a,
                                        SIMTUniformity b) noexcept {
  return static_cast<SIMTUniformity>(
      static_cast<std::uint8_t>(a) > static_cast<std::uint8_t>(b)
          ? static_cast<std::uint8_t>(a)
          : static_cast<std::uint8_t>(b));
}

const char *simtUniformityName(SIMTUniformity level) noexcept;

enum class SymbolicVarKind : std::uint8_t {
  ThreadIdX,
  ThreadIdY,
  ThreadIdZ,
  BlockBaseX, // Canonical block_id.x * block_dim.x
  BlockBaseY,
  BlockBaseZ,
  BlockIdX,
  BlockIdY,
  BlockIdZ,
  BlockDimX,
  BlockDimY,
  BlockDimZ,
  WarpId, // thread_id.x / subgroupSize
  LaneId, // thread_id.x % subgroupSize
  ScalarParam,
  LoopVar,
};

struct SymbolicVar {
  SymbolicVarKind kind = SymbolicVarKind::ScalarParam;
  std::string name;

  bool operator==(const SymbolicVar &) const noexcept = default;
  bool operator<(const SymbolicVar &other) const noexcept {
    if (kind != other.kind) return kind < other.kind;
    return name < other.name;
  }
};

struct AffineSymbolicExpr {
  bool isAffine = true;
  std::int64_t constant = 0;
  std::map<SymbolicVar, std::int64_t> terms;
  std::int64_t modulus = 0; // > 0 represents `(constant + sum terms) mod modulus`

  static AffineSymbolicExpr makeConstant(std::int64_t value);
  static AffineSymbolicExpr makeVar(SymbolicVarKind kind,
                                    std::string name = {});
  static AffineSymbolicExpr makeGlobalLinearIdX();
  static AffineSymbolicExpr nonAffine();

  bool isConstant() const noexcept {
    return isAffine && terms.empty() && modulus == 0;
  }

  AffineSymbolicExpr substitute(const SymbolicVar &var,
                                const AffineSymbolicExpr &replacement) const;
  std::string str() const;
};

AffineSymbolicExpr addExpr(const AffineSymbolicExpr &lhs,
                           const AffineSymbolicExpr &rhs);
AffineSymbolicExpr subExpr(const AffineSymbolicExpr &lhs,
                           const AffineSymbolicExpr &rhs);
AffineSymbolicExpr negExpr(const AffineSymbolicExpr &expr);
AffineSymbolicExpr mulExpr(const AffineSymbolicExpr &lhs,
                           const AffineSymbolicExpr &rhs,
                           std::size_t subgroupSize = 32);
AffineSymbolicExpr divExpr(const AffineSymbolicExpr &lhs,
                           const AffineSymbolicExpr &rhs,
                           std::size_t subgroupSize = 32);
AffineSymbolicExpr remExpr(const AffineSymbolicExpr &lhs,
                           const AffineSymbolicExpr &rhs,
                           std::size_t subgroupSize = 32);

enum class ConstraintRel : std::uint8_t {
  EqZero, // expr == 0
  NeZero, // expr != 0
  GeZero, // expr >= 0
};

struct AffineConstraint {
  ConstraintRel rel = ConstraintRel::GeZero;
  AffineSymbolicExpr expr; // Requires `expr.isAffine && expr.modulus == 0`
};

struct SymbolicGuard {
  bool contradiction = false;
  std::vector<AffineConstraint> constraints;

  void addComparison(std::string_view op, const AffineSymbolicExpr &lhs,
                     const AffineSymbolicExpr &rhs);
  SymbolicGuard withComparison(std::string_view op,
                               const AffineSymbolicExpr &lhs,
                               const AffineSymbolicExpr &rhs) const;
};

enum class MemoryAccessMode : std::uint8_t {
  Read,
  Write,
};

enum class ProtocolStmtKind : std::uint8_t {
  Access,
  WorkgroupBarrier,
  SubgroupBarrier,
  Sequence,
  Conditional,
  Loop,
};

struct ProtocolStmt {
  ProtocolStmtKind kind = ProtocolStmtKind::Sequence;

  // Access fields
  MemoryAccessMode accessMode = MemoryAccessMode::Read;
  std::string bufferName;
  GPUAddressSpace addressSpace = GPUAddressSpace::GlobalDevice;
  std::optional<std::int64_t> staticExtent;
  std::optional<std::string> dynamicExtentParam;
  AffineSymbolicExpr indexExpr;

  // Barrier / Control uniformity fields
  SIMTUniformity uniformity = SIMTUniformity::WorkgroupUniform;
  std::string barrierKindName;

  // Conditional fields
  SymbolicGuard thenGuard;
  SymbolicGuard elseGuard;
  std::vector<ProtocolStmt> thenBody;
  std::vector<ProtocolStmt> elseBody;

  // Sequence / Loop fields
  std::string loopVarName;
  AffineSymbolicExpr loopLower;
  AffineSymbolicExpr loopUpper;
  std::int64_t loopStep = 1;
  bool isBoundedFor = false;
  std::vector<ProtocolStmt> children;

  SourceLocation location;

  static ProtocolStmt makeAccess(
      MemoryAccessMode mode, std::string buffer, GPUAddressSpace space,
      AffineSymbolicExpr index, SourceLocation loc,
      std::optional<std::int64_t> extent = std::nullopt,
      std::optional<std::string> dynExtentParam = std::nullopt);
  static ProtocolStmt makeWorkgroupBarrier(SIMTUniformity pcUniformity,
                                           SourceLocation loc);
  static ProtocolStmt makeSubgroupBarrier(SIMTUniformity pcUniformity,
                                          std::string kindName,
                                          SourceLocation loc);
  static ProtocolStmt makeSequence(std::vector<ProtocolStmt> items,
                                   SourceLocation loc = {});
  static ProtocolStmt makeConditional(SIMTUniformity condUniformity,
                                      SymbolicGuard thenDelta,
                                      SymbolicGuard elseDelta,
                                      std::vector<ProtocolStmt> thenStmts,
                                      std::vector<ProtocolStmt> elseStmts,
                                      SourceLocation loc);
  static ProtocolStmt makeLoop(std::string ivName, AffineSymbolicExpr lower,
                               AffineSymbolicExpr upper, std::int64_t step,
                               SIMTUniformity headerUniformity,
                               bool boundedFor,
                               std::vector<ProtocolStmt> bodyStmts,
                               SourceLocation loc);
};

struct GPUMemoryAccessProtocolOptions {
  std::size_t subgroupSize = 32;
  std::size_t defaultBlockDimX = 64;
  bool multiBlockGrid = true;
  bool requireProvenBounds = false;
};

struct GPUProtocolViolation {
  std::string code;
  std::string kernelName;
  std::string subject;
  std::string message;
  SourceLocation location;
};

struct GPUProtocolReport {
  VerificationStatus status = VerificationStatus::UnboundedProved;
  std::size_t checkedKernels = 0;
  std::size_t checkedWorkgroupBarriers = 0;
  std::size_t checkedSubgroupBarriers = 0;
  std::size_t checkedPhases = 0;
  std::size_t checkedAccessPairs = 0;
  std::size_t checkedBoundsObligations = 0;
  std::vector<GPUProtocolViolation> violations;

  bool accepted() const noexcept {
    return violations.empty() &&
           (status == VerificationStatus::StaticChecked ||
            status == VerificationStatus::UnboundedProved ||
            status == VerificationStatus::Certified);
  }
};

// Verifies a single kernel's protocol tree:
// 1. Hierarchical SIMT barrier-uniformity & unambiguous subgroup participation.
// 2. Loop barrier aligning across loop backedges + two-level (workgroup and
//    subgroup) phase projection.
// 3. Pairwise symbolic thread data-race freedom (W-W and R-W) and optional
//    array/span index bounds checking.
void verifyKernelAccessProtocol(const std::string &kernelName,
                                const ProtocolStmt &root,
                                const GPUMemoryAccessProtocolOptions &options,
                                GPUProtocolReport &report);

// Extracts and verifies SIMT uniformity and memory-access protocols for all
// GPU kernels in a GraphIR module.
GPUProtocolReport verifyGPUMemoryAccessProtocol(
    const graphir::Module &module,
    const GPUMemoryAccessProtocolOptions &options = {});

} // namespace metreon::verification
