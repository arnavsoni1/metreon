#include "metreon/GraphIR/Lowering.h"
#include "metreon/GraphIR/Passes.h"
#include "metreon/Parser/Parser.h"
#include "metreon/Verification/GPUProfile.h"
#include "metreon/Verification/Toolchain.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

using namespace metreon;
using namespace metreon::verification;

void check(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

std::string readFile(const char *path) {
  std::ifstream input(path);
  check(static_cast<bool>(input), std::string("cannot open fixture: ") + path);
  return std::string((std::istreambuf_iterator<char>(input)), {});
}

graphir::Module lowerSource(const std::string &source,
                            const char *name = "gpu-profile-test.mtr",
                            bool optimize = true,
                            bool contextOnly = false) {
  parser::Parser parser(source);
  auto module = graphir::lowerModule(parser.parseModule(), name, contextOnly);
  if (optimize && !contextOnly) graphir::passes::runDefaultPipeline(module);
  return module;
}

bool hasViolationCode(const GPUProfileReport &report, std::string_view code) {
  return std::any_of(report.violations.begin(), report.violations.end(),
                     [&](const GPUProfileViolation &item) {
                       return item.code == code;
                     });
}

void testProofStrengthAndInvariants() {
  check(toolchain::pinnedLLVMMajor == 18, "unexpected pinned LLVM major version");

  // Proof-strength discipline: timeout, unknown, unsupported, bounded, and
  // contract-only outcomes never masquerade as unbounded proofs.
  check(satisfiesUnboundedProof(VerificationStatus::UnboundedProved),
        "unbounded-proved must satisfy unbounded proof check");
  check(satisfiesUnboundedProof(VerificationStatus::Certified),
        "certified must satisfy unbounded proof check");
  for (const auto status :
       {VerificationStatus::StaticChecked, VerificationStatus::ContractOnly,
        VerificationStatus::BoundedVerified, VerificationStatus::Unknown,
        VerificationStatus::Timeout, VerificationStatus::Unsupported,
        VerificationStatus::Rejected}) {
    check(!satisfiesUnboundedProof(status),
          std::string("non-unbounded status satisfied proof check: ") +
              verificationStatusName(status));
  }
  check(isIndeterminateOrUnsupported(VerificationStatus::Unknown) &&
            isIndeterminateOrUnsupported(VerificationStatus::Timeout) &&
            isIndeterminateOrUnsupported(VerificationStatus::Unsupported),
        "indeterminate statuses must be flagged explicitly");
  check(!isIndeterminateOrUnsupported(VerificationStatus::StaticChecked),
        "static-checked is not indeterminate");

  // Host vs. device address-space dereference isolation.
  check(canDereferenceAddressSpace(GPUExecutionScope::Host,
                                   GPUAddressSpace::HostDefault),
        "host must dereference default host space");
  for (const auto deviceSpace :
       {GPUAddressSpace::GlobalDevice, GPUAddressSpace::WorkgroupShared,
        GPUAddressSpace::Constant, GPUAddressSpace::ThreadPrivate}) {
    check(!canDereferenceAddressSpace(GPUExecutionScope::Host, deviceSpace),
          "host must not dereference device address space");
    check(canDereferenceAddressSpace(GPUExecutionScope::Thread, deviceSpace) &&
              canDereferenceAddressSpace(GPUExecutionScope::Subgroup, deviceSpace) &&
              canDereferenceAddressSpace(GPUExecutionScope::Workgroup, deviceSpace),
          "device scopes must dereference device address spaces");
  }
  check(!canDereferenceAddressSpace(GPUExecutionScope::Thread,
                                    GPUAddressSpace::HostDefault),
        "device thread must not dereference host default space");

  // Slot conservation: free + reserved + queued == capacity (capacity > 0).
  check(!holdsQueueSlotConservation(0, 0, 0, 0),
        "zero-capacity queue must be rejected");
  check(holdsQueueSlotConservation(2, 2, 0, 0), "initial empty conservation failed");
  check(holdsQueueSlotConservation(2, 0, 1, 1), "mixed conservation failed");
  check(!holdsQueueSlotConservation(2, 1, 1, 1), "overflow conservation accepted");
  check(!holdsQueueSlotConservation(2, 1, 0, 0), "lost-credit conservation accepted");

  // Publication order sufficiency per residency scope.
  check(!hasSufficientPublicationScope(GPUResidencyContract::WorkgroupLocal,
                                       GPUPublicationOrder::Relaxed),
        "relaxed publication cannot synchronize workgroup channels");
  check(hasSufficientPublicationScope(GPUResidencyContract::WorkgroupLocal,
                                      GPUPublicationOrder::AcqRelWorkgroup),
        "workgroup acq_rel must synchronize workgroup channels");
  check(!hasSufficientPublicationScope(GPUResidencyContract::CooperativeGrid,
                                       GPUPublicationOrder::AcqRelWorkgroup),
        "workgroup scope is insufficient for grid-wide channels");
  check(hasSufficientPublicationScope(GPUResidencyContract::CooperativeGrid,
                                      GPUPublicationOrder::AcqRelDevice),
        "device acq_rel must synchronize cooperative grid channels");
  check(!hasSufficientPublicationScope(GPUResidencyContract::UnscheduledGrid,
                                       GPUPublicationOrder::AcqRelDevice),
        "unscheduled grid has no valid blocking publication contract");
}

void testValidExamples(const char *pipelinePath, const char *vectorAddPath,
                       const char *contextsPath) {
  for (const bool optimize : {false, true}) {
    const auto pipelineModule =
        lowerSource(readFile(pipelinePath), "protocols_and_pipelines.mtr", optimize);
    const auto pipelineReport = validateGPUVerificationProfile(pipelineModule);
    check(pipelineReport.accepted(),
          "TiledGemm workgroup pipeline must pass Phase 0 profile check");
    check(pipelineReport.checkedPipelines == 1 &&
              pipelineReport.checkedEdges == 2 &&
              pipelineReport.checkedSchedules == 1,
          "unexpected TiledGemm checked entity counts");

    const auto vectorModule =
        lowerSource(readFile(vectorAddPath), "vector_add.mtr", optimize, true);
    const auto vectorReport = validateGPUVerificationProfile(vectorModule);
    check(vectorReport.accepted() && vectorReport.checkedContexts == 3,
          "vector_add contexts must pass Phase 0 profile check");

    const auto contextsModule =
        lowerSource(readFile(contextsPath), "contexts.mtr", optimize);
    const auto contextsReport = validateGPUVerificationProfile(contextsModule);
    check(contextsReport.accepted() && contextsReport.checkedContexts == 6,
          "contexts.mtr host/GPU contexts must pass Phase 0 profile check");
  }
}

void testProfileRejections(const char *pipelinePath) {
  const std::string basePipeline = readFile(pipelinePath);

  // 1. Capacity above the pinned bounded-queue limit (2) reports Unsupported.
  {
    auto module = lowerSource(basePipeline);
    for (auto &def : module.compositionDefinitions()) {
      if (def.kind == graphir::DefinitionKind::PipelineEdge) {
        def.attributes["capacity"] = "4";
      }
    }
    const auto report = validateGPUVerificationProfile(module);
    check(!report.accepted() &&
              report.status == VerificationStatus::Unsupported &&
              hasViolationCode(report, "gpu.profile.unsupported_capacity"),
          "oversized queue capacity must report unsupported");
  }

  // 2. Zero capacity is rejected outright.
  {
    auto module = lowerSource(basePipeline);
    for (auto &def : module.compositionDefinitions()) {
      if (def.kind == graphir::DefinitionKind::PipelineEdge) {
        def.attributes["capacity"] = "0";
      }
    }
    const auto report = validateGPUVerificationProfile(module);
    check(!report.accepted() &&
              report.status == VerificationStatus::Rejected &&
              hasViolationCode(report, "gpu.profile.invalid_capacity"),
          "zero queue capacity must be rejected");
  }

  // 3. gpu::shared storage paired with grid scope is rejected.
  {
    auto module = lowerSource(basePipeline);
    for (auto &def : module.compositionDefinitions()) {
      if (def.kind == graphir::DefinitionKind::PipelineEdge) {
        def.attributes["scope"] = "grid";
      }
    }
    const auto report = validateGPUVerificationProfile(module);
    check(!report.accepted() &&
              report.status == VerificationStatus::Rejected &&
              hasViolationCode(report, "gpu.profile.shared_storage_scope_mismatch"),
          "gpu::shared with grid scope must be rejected");
  }

  // 4. Global storage with blocking grid scope is unsupported without a
  // verified cooperative grid residency contract.
  {
    auto module = lowerSource(basePipeline);
    for (auto &def : module.compositionDefinitions()) {
      if (def.kind == graphir::DefinitionKind::PipelineEdge) {
        def.attributes["storage"] = "gpu::global";
        def.attributes["scope"] = "grid";
        def.attributes["backpressure"] = "cooperative";
      }
    }
    const auto report = validateGPUVerificationProfile(module);
    check(!report.accepted() &&
              report.status == VerificationStatus::Unsupported &&
              hasViolationCode(report, "gpu.profile.unscheduled_grid_blocking"),
          "unscheduled blocking grid channel must report unsupported");
  }

  // 5. Multi-block schedule on workgroup-local pipeline is unsupported.
  {
    auto module = lowerSource(basePipeline);
    for (auto &def : module.compositionDefinitions()) {
      if (def.kind == graphir::DefinitionKind::Schedule) {
        for (auto &rec : def.records) {
          if (rec.kind == "fused_kernel") rec.attributes["blocks"] = "4";
        }
      }
    }
    const auto report = validateGPUVerificationProfile(module);
    check(!report.accepted() &&
              report.status == VerificationStatus::Unsupported &&
              hasViolationCode(report, "gpu.profile.multi_block_workgroup_pipeline"),
          "multi-block workgroup pipeline must report unsupported");
  }

  // 6. Overlapping warp placements and out-of-bounds warps are rejected.
  {
    auto module = lowerSource(basePipeline);
    for (auto &def : module.compositionDefinitions()) {
      if (def.kind == graphir::DefinitionKind::Schedule) {
        for (auto &rec : def.records) {
          if (rec.kind == "placement" && rec.name == "load_b") {
            rec.attributes["index"] = "0";
          }
        }
      }
    }
    const auto report = validateGPUVerificationProfile(module);
    check(!report.accepted() &&
              report.status == VerificationStatus::Rejected &&
              hasViolationCode(report, "gpu.profile.overlapping_warp_placement"),
          "overlapping warp placement must be rejected");
  }
  {
    auto module = lowerSource(basePipeline);
    for (auto &def : module.compositionDefinitions()) {
      if (def.kind == graphir::DefinitionKind::Schedule) {
        for (auto &rec : def.records) {
          if (rec.kind == "placement" && rec.name == "mma") {
            rec.attributes["index"] = "32";
          }
        }
      }
    }
    const auto report = validateGPUVerificationProfile(module);
    check(!report.accepted() &&
              report.status == VerificationStatus::Rejected &&
              hasViolationCode(report, "gpu.profile.warp_out_of_bounds"),
          "out-of-bounds warp index must be rejected");
  }

  // 7. Grant-bearing or generic Gpu::Loop contexts are rejected.
  {
    graphir::Module module("invalid-loop.mtr");
    auto loopCtx = module.addContextMetadata(
        "Gpu::Loop", "bad_loop", 0, graphir::ContextResolution::Declared, {});
    module.addNode(graphir::NodeKind::Grant, "gpu_load", {}, loopCtx, {});
    const auto report = validateGPUVerificationProfile(module);
    check(!report.accepted() &&
              report.status == VerificationStatus::Rejected &&
              hasViolationCode(report, "gpu.profile.grant_bearing_loop_context"),
          "grant-bearing Gpu::Loop context must be rejected");
  }
  {
    graphir::Module module("generic-loop.mtr");
    module.addContextMetadata("Gpu::Loop<T>", "generic_loop", 1,
                              graphir::ContextResolution::Declared, {});
    const auto report = validateGPUVerificationProfile(module);
    check(!report.accepted() &&
              report.status == VerificationStatus::Rejected &&
              hasViolationCode(report, "gpu.profile.generic_loop_context"),
          "generic Gpu::Loop context must be rejected");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    check(argc == 4,
          "expected protocols_and_pipelines.mtr, vector_add.mtr, and contexts.mtr paths");
    testProofStrengthAndInvariants();
    testValidExamples(argv[1], argv[2], argv[3]);
    testProfileRejections(argv[1]);
    std::cout << "GPU verification Phase 0 profile and toolchain checks passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
