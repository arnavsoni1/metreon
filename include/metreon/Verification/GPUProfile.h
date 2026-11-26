#pragma once

#include "metreon/Basic/SourceLocation.h"
#include "metreon/GraphIR/Graph.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace metreon::verification {

enum class GPUExecutionScope {
  Host,
  DeviceGrid,
  Workgroup,
  Subgroup,
  Thread,
};

enum class GPULoopDomain {
  HostLoop,
  DeviceLoop,
};

enum class GPUAddressSpace : std::uint32_t {
  HostDefault = 0,
  GlobalDevice = 1,
  WorkgroupShared = 3,
  Constant = 4,
  ThreadPrivate = 5,
};

enum class GPUResidencyContract {
  WorkgroupLocal,
  CooperativeGrid,
  UnscheduledGrid,
};

enum class GPUPublicationOrder {
  Relaxed,
  AcquireWorkgroup,
  ReleaseWorkgroup,
  AcqRelWorkgroup,
  AcquireDevice,
  ReleaseDevice,
  AcqRelDevice,
};

enum class VerificationStatus {
  StaticChecked,
  ContractOnly,
  BoundedVerified,
  UnboundedProved,
  Certified,
  Unknown,
  Timeout,
  Unsupported,
  Rejected,
};

struct GPUWorkgroupPipelineProfile {
  std::string_view profileId = "gpu.workgroup_pipeline.v1";
  int pinnedLLVMMajor = 18;
  unsigned indexBits = 64;
  std::size_t minBoundedQueueCapacity = 1;
  std::size_t maxBoundedQueueCapacity = 2;
  std::size_t maxWorkgroupThreads = 1024;
  std::size_t nvidiaSubgroupSize = 32;
  std::size_t amdSubgroupSize = 64;
  std::size_t maxWorkgroupWarps = 32;
  bool allowCooperativeGridChannels = false;
};

struct GPUProfileViolation {
  std::string code;
  std::string subject;
  std::string message;
  SourceLocation location;
};

struct GPUProfileReport {
  std::string profileId;
  VerificationStatus status = VerificationStatus::StaticChecked;
  std::size_t checkedContexts = 0;
  std::size_t checkedPipelines = 0;
  std::size_t checkedEdges = 0;
  std::size_t checkedSchedules = 0;
  std::vector<GPUProfileViolation> violations;

  bool accepted() const noexcept {
    return violations.empty() &&
           (status == VerificationStatus::StaticChecked ||
            status == VerificationStatus::BoundedVerified ||
            status == VerificationStatus::UnboundedProved ||
            status == VerificationStatus::Certified);
  }
};

const char *verificationStatusName(VerificationStatus status) noexcept;
const char *gpuExecutionScopeName(GPUExecutionScope scope) noexcept;
const char *gpuAddressSpaceName(GPUAddressSpace space) noexcept;
const char *gpuResidencyContractName(GPUResidencyContract residency) noexcept;

// Enforces proof-strength discipline: timeouts, unknown solver exits,
// unsupported constructs, and unverified external contracts never count as an
// unbounded implementation proof.
bool satisfiesUnboundedProof(VerificationStatus status) noexcept;
bool isIndeterminateOrUnsupported(VerificationStatus status) noexcept;

// Enforces static host/device address-space dereference isolation.
bool canDereferenceAddressSpace(GPUExecutionScope scope,
                                GPUAddressSpace space) noexcept;

// Checks bounded-channel slot conservation: free + reserved + queued = capacity.
bool holdsQueueSlotConservation(std::size_t capacity, std::size_t freeCredits,
                                std::size_t reservedSlots,
                                std::size_t queuedItems) noexcept;

// Validates that a publication order provides sufficient acquire/release
// visibility for a channel operating at the given residency scope.
bool hasSufficientPublicationScope(GPUResidencyContract residency,
                                   GPUPublicationOrder order) noexcept;

// Validates GraphIR contexts, GPU pipeline edges, storage/scope pairings,
// bounded capacities, and warp/workgroup schedules against the pinned
// workgroup-local GPU verification profile.
GPUProfileReport validateGPUVerificationProfile(
    const graphir::Module &module,
    const GPUWorkgroupPipelineProfile &profile = {});

} // namespace metreon::verification
