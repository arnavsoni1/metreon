#include "metreon/Verification/GPUProfile.h"
#include "metreon/Verification/Toolchain.h"

#include <algorithm>
#include <charconv>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>

namespace metreon::verification {
namespace {

std::optional<std::size_t> parseUnsigned(std::string_view text) noexcept {
  if (text.empty()) return std::nullopt;
  std::size_t value = 0;
  const auto *begin = text.data();
  const auto *end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc{} || ptr != end) return std::nullopt;
  return value;
}

bool startsWith(std::string_view text, std::string_view prefix) noexcept {
  return text.size() >= prefix.size() &&
         text.substr(0, prefix.size()) == prefix;
}

std::string_view stripGenericArguments(std::string_view typeName) noexcept {
  const auto angle = typeName.find('<');
  return angle == std::string_view::npos ? typeName : typeName.substr(0, angle);
}

void recordViolation(GPUProfileReport &report, VerificationStatus failureStatus,
                     std::string code, std::string subject, std::string message,
                     SourceLocation location) {
  if (report.status == VerificationStatus::StaticChecked ||
      (report.status == VerificationStatus::Unsupported &&
       failureStatus == VerificationStatus::Rejected)) {
    report.status = failureStatus;
  }
  report.violations.push_back(GPUProfileViolation{
      std::move(code), std::move(subject), std::move(message), location});
}

void validateContexts(const graphir::Module &module, GPUProfileReport &report) {
  std::map<std::size_t, std::size_t> grantCounts;
  for (const auto &node : module.nodes()) {
    if (node.kind != graphir::NodeKind::Grant || !node.context) continue;
    try {
      const auto ordinal = module.contextOrdinal(node.context);
      ++grantCounts[ordinal];
    } catch (const std::logic_error &error) {
      recordViolation(report, VerificationStatus::Rejected,
                      "gpu.profile.foreign_grant_context", node.name,
                      error.what(), node.location);
    }
  }

  for (const auto &context : module.contexts()) {
    if (!context) {
      recordViolation(report, VerificationStatus::Rejected,
                      "gpu.profile.null_context", "<null>",
                      "GraphIR module contains a null context metadata entry",
                      {});
      continue;
    }
    std::size_t ordinal = 0;
    try {
      module.requireContextUse(context, graphir::ContextUse::MetadataAttachment);
      ordinal = module.contextOrdinal(context);
    } catch (const std::logic_error &error) {
      recordViolation(report, VerificationStatus::Rejected,
                      "gpu.profile.foreign_context", context->name(),
                      error.what(), context->location());
      continue;
    }

    if (context->isRuntimeMaterializable() ||
        context->permits(graphir::ContextUse::RuntimeStore) ||
        context->permits(graphir::ContextUse::Cast) ||
        context->permits(graphir::ContextUse::Transmute) ||
        context->permits(graphir::ContextUse::CrossThreadSend) ||
        context->permits(graphir::ContextUse::RuntimeCapture)) {
      recordViolation(report, VerificationStatus::Rejected,
                      "gpu.profile.materializable_context", context->name(),
                      "execution context evidence must remain compile-time metadata",
                      context->location());
    }

    const auto baseName = stripGenericArguments(context->name());
    const bool isGpu = startsWith(baseName, "Gpu::");
    const bool isHost = startsWith(baseName, "Host::");
    if (!isGpu && !isHost) continue;

    ++report.checkedContexts;
    const bool isLoop = (baseName == "Gpu::Loop" || baseName == "Host::Loop");
    if (isLoop) {
      if (context->genericArity() != 0 || context->name() != baseName) {
        recordViolation(report, VerificationStatus::Rejected,
                        "gpu.profile.generic_loop_context", context->name(),
                        "loop evidence contexts cannot declare generic parameters",
                        context->location());
      }
      if (grantCounts[ordinal] != 0) {
        recordViolation(report, VerificationStatus::Rejected,
                        "gpu.profile.grant_bearing_loop_context", context->name(),
                        "loop evidence contexts cannot grant execution capabilities",
                        context->location());
      }
    }
  }
}

void validatePipelineEdge(const graphir::CompositionDefinition &edge,
                          const GPUWorkgroupPipelineProfile &profile,
                          GPUProfileReport &report) {
  const auto storageIt = edge.attributes.find("storage");
  const auto scopeIt = edge.attributes.find("scope");
  const bool gpuEdge =
      (storageIt != edge.attributes.end() &&
       startsWith(storageIt->second, "gpu::")) ||
      (scopeIt != edge.attributes.end() &&
       (scopeIt->second == "block" || scopeIt->second == "warp" ||
        scopeIt->second == "grid" || scopeIt->second == "device"));
  if (!gpuEdge) return;

  ++report.checkedEdges;

  const auto capacityIt = edge.attributes.find("capacity");
  if (capacityIt == edge.attributes.end()) {
    recordViolation(report, VerificationStatus::Rejected,
                    "gpu.profile.missing_capacity", edge.name,
                    "GPU pipeline channel edge requires an explicit capacity",
                    edge.location);
  } else {
    const auto capacity = parseUnsigned(capacityIt->second);
    if (!capacity || *capacity < profile.minBoundedQueueCapacity) {
      recordViolation(report, VerificationStatus::Rejected,
                      "gpu.profile.invalid_capacity", edge.name,
                      "GPU pipeline channel capacity must be at least " +
                          std::to_string(profile.minBoundedQueueCapacity),
                      edge.location);
    } else if (*capacity > profile.maxBoundedQueueCapacity) {
      recordViolation(report, VerificationStatus::Unsupported,
                      "gpu.profile.unsupported_capacity", edge.name,
                      "GPU pipeline channel capacity exceeds pinned profile bound (" +
                          std::to_string(profile.maxBoundedQueueCapacity) + ")",
                      edge.location);
    }
  }

  const std::string storage =
      storageIt != edge.attributes.end() ? storageIt->second : "";
  const std::string scope =
      scopeIt != edge.attributes.end() ? scopeIt->second : "";
  const auto backpressureIt = edge.attributes.find("backpressure");
  const std::string backpressure =
      backpressureIt != edge.attributes.end() ? backpressureIt->second : "";

  if (storage == "gpu::shared" && scope != "block" && scope != "warp") {
    recordViolation(
        report, VerificationStatus::Rejected,
        "gpu.profile.shared_storage_scope_mismatch", edge.name,
        "gpu::shared channel storage is valid only for block or warp scope",
        edge.location);
  }

  if (scope == "grid" || scope == "device") {
    const bool blocking =
        backpressure.empty() || backpressure == "cooperative" ||
        backpressure == "blocking";
    if (blocking && !profile.allowCooperativeGridChannels) {
      recordViolation(
          report, VerificationStatus::Unsupported,
          "gpu.profile.unscheduled_grid_blocking", edge.name,
          "blocking inter-block GPU channel requires a verified cooperative "
          "grid residency profile; workgroup-local profile rejects it",
          edge.location);
    }
  } else if (scope != "block" && scope != "warp") {
    recordViolation(report, VerificationStatus::Unsupported,
                    "gpu.profile.unsupported_channel_scope", edge.name,
                    "GPU channel scope `" + scope +
                        "` is outside the workgroup-local profile",
                    edge.location);
  }

  if (!storage.empty() && storage != "gpu::shared" &&
      storage != "gpu::global") {
    recordViolation(report, VerificationStatus::Unsupported,
                    "gpu.profile.unsupported_channel_storage", edge.name,
                    "GPU channel storage `" + storage +
                        "` is outside the workgroup-local profile",
                    edge.location);
  }
}

void validatePipelineSchedule(const graphir::CompositionDefinition &schedule,
                              const GPUWorkgroupPipelineProfile &profile,
                              bool hasWorkgroupEdges,
                              GPUProfileReport &report) {
  const auto targetIt = schedule.attributes.find("target");
  const std::string target =
      targetIt != schedule.attributes.end() ? targetIt->second : "";
  const bool gpuTarget = startsWith(target, "sm") || startsWith(target, "gfx") ||
                         hasWorkgroupEdges;
  if (!gpuTarget) return;

  ++report.checkedSchedules;
  std::map<std::string, std::size_t> workerBlocks;
  std::map<std::string, std::set<std::size_t>> occupiedWarps;

  for (const auto &record : schedule.records) {
    if (record.kind == "fused_kernel") {
      if (record.references.empty()) {
        recordViolation(report, VerificationStatus::Rejected,
                        "gpu.profile.empty_fused_worker", record.name,
                        "fused GPU worker must contain at least one stage",
                        record.location);
      }
      const auto blocksIt = record.attributes.find("blocks");
      if (blocksIt != record.attributes.end()) {
        const auto blocks = parseUnsigned(blocksIt->second);
        if (!blocks || *blocks == 0) {
          recordViolation(report, VerificationStatus::Rejected,
                          "gpu.profile.invalid_worker_blocks", record.name,
                          "fused GPU worker block count must be positive",
                          record.location);
        } else {
          workerBlocks[record.name] = *blocks;
          if (hasWorkgroupEdges && *blocks != 1 &&
              !profile.allowCooperativeGridChannels) {
            recordViolation(
                report, VerificationStatus::Unsupported,
                "gpu.profile.multi_block_workgroup_pipeline", record.name,
                "workgroup-local pipeline profile requires single-workgroup "
                "co-residency (blocks = 1) for fused stage communication",
                record.location);
          }
        }
      }
      for (const auto &region : record.contextRegions) {
        if (!region.context || region.stages.empty()) {
          recordViolation(report, VerificationStatus::Rejected,
                          "gpu.profile.invalid_stage_context_region",
                          record.name,
                          "fused stage context region requires context metadata "
                          "and at least one stage",
                          region.location);
        }
      }
    } else if (record.kind == "placement") {
      const auto unitIt = record.attributes.find("unit");
      const auto workerIt = record.attributes.find("worker");
      const auto firstIt = record.attributes.find("index");
      if (unitIt == record.attributes.end() ||
          workerIt == record.attributes.end() ||
          firstIt == record.attributes.end()) {
        recordViolation(report, VerificationStatus::Rejected,
                        "gpu.profile.incomplete_placement", record.name,
                        "GPU stage placement is missing worker, unit, or index",
                        record.location);
        continue;
      }
      if (unitIt->second != "warp") {
        recordViolation(report, VerificationStatus::Unsupported,
                        "gpu.profile.unsupported_placement_unit", record.name,
                        "workgroup-local GPU profile supports `warp` stage "
                        "placement units only",
                        record.location);
        continue;
      }
      const auto first = parseUnsigned(firstIt->second);
      const auto endIt = record.attributes.find("end");
      const auto last =
          endIt != record.attributes.end() ? parseUnsigned(endIt->second) : first;
      if (!first || !last || *last < *first) {
        recordViolation(report, VerificationStatus::Rejected,
                        "gpu.profile.invalid_warp_range", record.name,
                        "invalid warp index range in GPU stage placement",
                        record.location);
        continue;
      }
      if (*last >= profile.maxWorkgroupWarps) {
        recordViolation(report, VerificationStatus::Rejected,
                        "gpu.profile.warp_out_of_bounds", record.name,
                        "warp placement exceeds workgroup warp limit (" +
                            std::to_string(profile.maxWorkgroupWarps) + ")",
                        record.location);
        continue;
      }
      auto &used = occupiedWarps[workerIt->second];
      for (std::size_t warp = *first; warp <= *last; ++warp) {
        if (!used.insert(warp).second) {
          recordViolation(
              report, VerificationStatus::Rejected,
              "gpu.profile.overlapping_warp_placement", record.name,
              "multiple pipeline stages share warp " + std::to_string(warp) +
                  " in worker `" + workerIt->second + "`",
              record.location);
          break;
        }
      }
    }
  }
}

} // namespace

const char *verificationStatusName(VerificationStatus status) noexcept {
  switch (status) {
  case VerificationStatus::StaticChecked: return "static-checked";
  case VerificationStatus::ContractOnly: return "contract-only";
  case VerificationStatus::BoundedVerified: return "bounded-verified";
  case VerificationStatus::UnboundedProved: return "unbounded-proved";
  case VerificationStatus::Certified: return "certified";
  case VerificationStatus::Unknown: return "unknown";
  case VerificationStatus::Timeout: return "timeout";
  case VerificationStatus::Unsupported: return "unsupported";
  case VerificationStatus::Rejected: return "rejected";
  }
  return "unknown";
}

const char *gpuExecutionScopeName(GPUExecutionScope scope) noexcept {
  switch (scope) {
  case GPUExecutionScope::Host: return "host";
  case GPUExecutionScope::DeviceGrid: return "device_grid";
  case GPUExecutionScope::Workgroup: return "workgroup";
  case GPUExecutionScope::Subgroup: return "subgroup";
  case GPUExecutionScope::Thread: return "thread";
  }
  return "unknown";
}

const char *gpuAddressSpaceName(GPUAddressSpace space) noexcept {
  switch (space) {
  case GPUAddressSpace::HostDefault: return "host_default";
  case GPUAddressSpace::GlobalDevice: return "global_device";
  case GPUAddressSpace::WorkgroupShared: return "workgroup_shared";
  case GPUAddressSpace::Constant: return "constant";
  case GPUAddressSpace::ThreadPrivate: return "thread_private";
  }
  return "unknown";
}

const char *gpuResidencyContractName(GPUResidencyContract residency) noexcept {
  switch (residency) {
  case GPUResidencyContract::WorkgroupLocal: return "workgroup_local";
  case GPUResidencyContract::CooperativeGrid: return "cooperative_grid";
  case GPUResidencyContract::UnscheduledGrid: return "unscheduled_grid";
  }
  return "unknown";
}

bool satisfiesUnboundedProof(VerificationStatus status) noexcept {
  return status == VerificationStatus::UnboundedProved ||
         status == VerificationStatus::Certified;
}

bool isIndeterminateOrUnsupported(VerificationStatus status) noexcept {
  return status == VerificationStatus::Unknown ||
         status == VerificationStatus::Timeout ||
         status == VerificationStatus::Unsupported;
}

bool canDereferenceAddressSpace(GPUExecutionScope scope,
                                GPUAddressSpace space) noexcept {
  if (scope == GPUExecutionScope::Host) {
    return space == GPUAddressSpace::HostDefault;
  }
  return space == GPUAddressSpace::GlobalDevice ||
         space == GPUAddressSpace::WorkgroupShared ||
         space == GPUAddressSpace::Constant ||
         space == GPUAddressSpace::ThreadPrivate;
}

bool holdsQueueSlotConservation(std::size_t capacity, std::size_t freeCredits,
                                std::size_t reservedSlots,
                                std::size_t queuedItems) noexcept {
  if (capacity == 0) return false;
  if (freeCredits > capacity || reservedSlots > capacity ||
      queuedItems > capacity) {
    return false;
  }
  return freeCredits + reservedSlots + queuedItems == capacity;
}

bool hasSufficientPublicationScope(GPUResidencyContract residency,
                                   GPUPublicationOrder order) noexcept {
  switch (residency) {
  case GPUResidencyContract::WorkgroupLocal:
    return order == GPUPublicationOrder::AcqRelWorkgroup ||
           order == GPUPublicationOrder::AcqRelDevice ||
           order == GPUPublicationOrder::ReleaseWorkgroup ||
           order == GPUPublicationOrder::AcquireWorkgroup ||
           order == GPUPublicationOrder::ReleaseDevice ||
           order == GPUPublicationOrder::AcquireDevice;
  case GPUResidencyContract::CooperativeGrid:
    return order == GPUPublicationOrder::AcqRelDevice ||
           order == GPUPublicationOrder::ReleaseDevice ||
           order == GPUPublicationOrder::AcquireDevice;
  case GPUResidencyContract::UnscheduledGrid:
    return false;
  }
  return false;
}

GPUProfileReport validateGPUVerificationProfile(
    const graphir::Module &module,
    const GPUWorkgroupPipelineProfile &profile) {
  GPUProfileReport report;
  report.profileId = std::string(profile.profileId);

  if (profile.pinnedLLVMMajor != toolchain::pinnedLLVMMajor ||
      profile.indexBits != 64 || profile.minBoundedQueueCapacity == 0 ||
      profile.maxBoundedQueueCapacity < profile.minBoundedQueueCapacity ||
      profile.maxWorkgroupThreads == 0 || profile.maxWorkgroupWarps == 0) {
    recordViolation(report, VerificationStatus::Rejected,
                    "gpu.profile.invalid_configuration", report.profileId,
                    "GPU verification profile parameters violate baseline bounds",
                    {});
    return report;
  }

  validateContexts(module, report);

  for (const auto &pipeline : module.pipelineGraphs()) {
    ++report.checkedPipelines;
    bool hasWorkgroupEdges = false;
    std::vector<const graphir::CompositionDefinition *> schedules;

    for (const auto &member : pipeline.members) {
      if (member.graph >= module.resourceGraphs().size() ||
          member.member >=
              module.resourceGraphs()[member.graph].references().size()) {
        recordViolation(report, VerificationStatus::Rejected,
                        "gpu.profile.dangling_pipeline_member", pipeline.name,
                        "pipeline member refers to an invalid GraphIR resource slot",
                        pipeline.location);
        continue;
      }
      const auto ref =
          module.resourceGraphs()[member.graph].references()[member.member].target;
      if (ref.kind == graphir::DefinitionKind::PipelineEdge) {
        const auto &edge = module.compositionDefinitions().at(ref.index);
        const auto before = report.checkedEdges;
        validatePipelineEdge(edge, profile, report);
        if (report.checkedEdges != before) hasWorkgroupEdges = true;
      } else if (ref.kind == graphir::DefinitionKind::Schedule) {
        schedules.push_back(&module.compositionDefinitions().at(ref.index));
      }
    }

    for (const auto *schedule : schedules) {
      validatePipelineSchedule(*schedule, profile, hasWorkgroupEdges, report);
    }
  }

  return report;
}

} // namespace metreon::verification
