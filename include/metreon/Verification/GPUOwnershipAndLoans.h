#pragma once

#include "metreon/Basic/SourceLocation.h"
#include "metreon/GraphIR/Graph.h"
#include "metreon/Verification/GPUProfile.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace metreon::verification {

enum class ResourceLifecycleKind {
  Droppable,
  LinearOwned,
  MustComplete,
};

enum class AsyncLoanMode {
  ImmutableRead,
  ExclusiveWrite,
};

struct AsyncLoanKey {
  std::string allocationId;
  std::string slotId;
  std::string generation;

  bool operator==(const AsyncLoanKey &) const noexcept = default;
  bool operator<(const AsyncLoanKey &other) const noexcept {
    if (allocationId != other.allocationId)
      return allocationId < other.allocationId;
    if (slotId != other.slotId) return slotId < other.slotId;
    return generation < other.generation;
  }
};

struct ActiveAsyncLoan {
  AsyncLoanKey key;
  AsyncLoanMode mode = AsyncLoanMode::ExclusiveWrite;
  SourceLocation location;
};

struct GPUOwnershipViolation {
  std::string code;
  std::string callableName;
  std::string subject;
  std::string message;
  SourceLocation location;
};

struct GPUOwnershipReport {
  VerificationStatus status = VerificationStatus::StaticChecked;
  std::size_t checkedContexts = 0;
  std::size_t checkedCallables = 0;
  std::size_t checkedTransitions = 0;
  std::size_t checkedAsyncIssues = 0;
  std::size_t checkedAsyncWaits = 0;
  std::vector<GPUOwnershipViolation> violations;

  bool accepted() const noexcept {
    return violations.empty() &&
           status == VerificationStatus::StaticChecked;
  }
};

// Maps nominal Host::* and Gpu::{Grid, Block, Warp, Thread} context type names
// onto their execution scope without implicit casting. Returns std::nullopt for
// loop contexts (Host::Loop, Gpu::Loop) and non-GPU/Host domains.
std::optional<GPUExecutionScope>
classifyContextExecutionScope(std::string_view contextTypeName) noexcept;

// Maps grant-free loop contexts onto their loop domain.
std::optional<GPULoopDomain>
classifyLoopContextDomain(std::string_view contextTypeName) noexcept;

// Maps canonical storage/span type annotations onto GPU address spaces.
std::optional<GPUAddressSpace>
classifyTypeAddressSpace(std::string_view typeText) noexcept;

// Enforces capability-to-scope discipline: e.g., workgroup barriers require
// Workgroup scope (never Thread or Subgroup), subgroup shuffles/barriers require
// Subgroup or Workgroup scope (never Thread), grid sync requires DeviceGrid,
// and Host contexts cannot hold device-resident memory/barrier grants.
bool isCapabilityAllowedInScope(GPUExecutionScope scope,
                                std::string_view capabilityName) noexcept;

// Enforces minimum/exact scope requirements for built-in intrinsics.
bool isIntrinsicAllowedInScope(GPUExecutionScope scope,
                               std::string_view intrinsicName) noexcept;

// Runs static context-scope hierarchy, cross-space dereference isolation,
// path-sensitive ownership typestate (expanding collapsed ResourceGraphs), and
// (AllocationId, SlotId, Generation) async-copy loan verification across all
// host and GPU callables in a GraphIR module.
GPUOwnershipReport verifyGPUOwnershipAndLoans(const graphir::Module &module);

} // namespace metreon::verification
