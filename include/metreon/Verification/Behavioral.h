#pragma once

#include "metreon/Basic/SourceLocation.h"
#include "metreon/GraphIR/Behavioral.h"
#include "metreon/GraphIR/Graph.h"
#include "metreon/Verification/GPUProfile.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace metreon::verification {

using AutomatonStateId = std::size_t;

// Target-independent behavioral process action kinds shared by the GPU
// bounded-queue profile (`GPU-4`) and clocked hardware rendezvous profile
// (`FPGA-3`). GPU execution never uses `ClockTick` as a surrogate for async or
// queue waits.
enum class BehavioralActionKind : std::uint8_t {
  Send,
  Receive,
  Internal,
  ClockTick,
};

const char *behavioralActionKindName(BehavioralActionKind kind) noexcept;

struct OfferArmSpec {
  BehavioralActionKind kind = BehavioralActionKind::Receive;
  std::string endpoint;
  std::string message;
  AutomatonStateId targetState = 0;
  SourceLocation location;
};

struct AutomatonOfferGroup {
  graphir::OfferMode mode = graphir::OfferMode::Exclusive;
  std::vector<OfferArmSpec> arms;
  std::optional<AutomatonStateId> defaultTarget;
  SourceLocation location;
};

struct AutomatonState {
  AutomatonStateId id = 0;
  std::string label;
  bool isAcceptingTerminal = false;
  std::optional<AutomatonOfferGroup> offer;
  SourceLocation location;
};

struct ProcessTransition {
  AutomatonStateId source = 0;
  AutomatonStateId target = 0;
  BehavioralActionKind kind = BehavioralActionKind::Internal;
  std::string endpoint;
  std::string message;
  graphir::ChannelDeliveryMode delivery =
      graphir::ChannelDeliveryMode::BlockingQueue;
  bool transfersOwnership = false;
  bool isOfferDefault = false;
  std::optional<std::string> fairTag;
  bool fairBranchOutcome = true;
  bool assumptionJustified = false;
  std::vector<std::string> progressMarkers;
  SourceLocation location;
};

// Target-independent finite-state behavioral process automaton extracted from
// a checked callable CFG (`buildControlFlow()`) or constructed as a boundary
// summary automaton.
struct BehavioralAutomaton {
  std::string name;
  AutomatonStateId initialState = 0;
  std::vector<AutomatonState> states;
  std::vector<ProcessTransition> transitions;
  std::set<std::string> requiredProgressMarkers;
  std::set<std::string> unprovedFairnessAssumptions;
  SourceLocation location;

  bool hasCommunicationActions() const noexcept;
};

struct ProtocolMessageSpec {
  std::string name;
  std::string senderRole;
  std::string receiverRole;
  std::string successorState;
  SourceLocation location;
};

struct ProtocolStateSpec {
  std::string name;
  bool isFinal = false;
  std::vector<ProtocolMessageSpec> messages;
  SourceLocation location;
};

struct ProtocolAutomatonSpec {
  std::string name;
  std::vector<std::string> roles;
  std::map<std::string, ProtocolStateSpec> states;
  SourceLocation location;
};

// Explicit capacity-C bounded queue configuration for GPU pipeline channels.
struct GPUBoundedQueueSpec {
  std::string name;
  std::string protocolName;
  std::string initialProtocolState;
  std::size_t capacity = 1;
  std::size_t seededItems = 0;
  std::string seededMessage;
  std::string storage = "gpu::shared";
  std::string scope = "block";
  std::string backpressure = "cooperative";
  GPUResidencyContract residency = GPUResidencyContract::WorkgroupLocal;
  GPUPublicationOrder publication = GPUPublicationOrder::AcqRelWorkgroup;
  bool explicitPublication = false;
  std::string producerStage;
  std::string producerRole;
  std::string consumerStage;
  std::string consumerRole;
  SourceLocation location;
};

struct GPUStagePlacementInfo {
  std::string stageName;
  std::string workerName;
  std::size_t workerBlocks = 1;
  std::string workerResidency = "persistent";
  bool hasWarpPlacement = false;
  std::size_t firstWarp = 0;
  std::size_t lastWarp = 0;
  SourceLocation location;
};

struct GPUPipelineSystemSpec {
  std::string name;
  std::map<std::string, ProtocolAutomatonSpec> protocols;
  std::vector<BehavioralAutomaton> stages;
  std::vector<GPUBoundedQueueSpec> queues;
  std::map<std::string, GPUStagePlacementInfo> placements;
  std::size_t scheduleCount = 0;
  bool hasUnconnectedDuplicateConsumer = false;
  std::string duplicateConsumerEdge;
  SourceLocation duplicateConsumerLoc;
  SourceLocation location;
};

// Explicit queue state maintained during product-game exploration:
// enforces `freeCredits + reservedSlots + items.size() == capacity`.
struct BoundedQueueRuntimeState {
  std::size_t capacity = 1;
  std::size_t freeCredits = 1;
  std::size_t reservedSlots = 0;
  std::vector<std::string> items;

  bool operator==(const BoundedQueueRuntimeState &) const noexcept = default;
  bool operator<(const BoundedQueueRuntimeState &other) const noexcept {
    if (capacity != other.capacity) return capacity < other.capacity;
    if (freeCredits != other.freeCredits)
      return freeCredits < other.freeCredits;
    if (reservedSlots != other.reservedSlots)
      return reservedSlots < other.reservedSlots;
    return items < other.items;
  }
};

struct CompositeSystemState {
  std::vector<AutomatonStateId> stageStates;
  std::vector<BoundedQueueRuntimeState> queueStates;
  std::vector<std::string> protocolStates;

  bool operator==(const CompositeSystemState &) const noexcept = default;
  bool operator<(const CompositeSystemState &other) const noexcept {
    if (stageStates != other.stageStates)
      return stageStates < other.stageStates;
    if (queueStates != other.queueStates)
      return queueStates < other.queueStates;
    return protocolStates < other.protocolStates;
  }
};

struct CompositeStepEdge {
  std::size_t fromStateIndex = 0;
  std::size_t toStateIndex = 0;
  std::size_t actorStageIndex = 0;
  BehavioralActionKind actionKind = BehavioralActionKind::Internal;
  std::string endpoint;
  std::string message;
  std::optional<std::string> fairTag;
  bool fairBranchOutcome = true;
  bool assumptionJustified = false;
  std::vector<std::string> progressMarkers;
};

// Self-contained invariant and strategy witness emitted by the solver and
// independently validated by `validateCompatibilityCertificate`.
struct CompatibilityCertificate {
  std::string pipelineName;
  std::vector<std::size_t> queueCapacities;
  std::vector<std::size_t> queueSeededItems;
  std::set<std::string> requiredProgressMarkers;
  std::vector<CompositeSystemState> states;
  std::vector<bool> isAcceptingTerminalState;
  std::vector<CompositeStepEdge> edges;
  std::size_t initialStateIndex = 0;
};

// Reusable boundary summary automaton produced by composing stages + internal
// bounded queues and hiding internal channel exchanges.
struct BehavioralSummaryAutomaton {
  BehavioralAutomaton automaton;
  std::set<std::string> boundaryEndpoints;
  std::set<std::string> hiddenInternalEndpoints;
  std::set<std::string> conditionalFairnessTags;
};

struct GPUBoundedQueueOptions {
  GPUWorkgroupPipelineProfile profile{};
  std::size_t maxExploredStates = 4096;
  std::size_t maxExploredTransitions = 16384;
  std::size_t maxWorkgroupSharedBytes = 49152;
  bool requireCleanQueueDrain = true;
  bool validateGeneratedCertificates = true;
};

struct GPUBoundedQueueViolation {
  std::string code;
  std::string pipelineName;
  std::string subject;
  std::string message;
  std::vector<std::string> counterexampleTrace;
  SourceLocation location;
};

struct GPUBoundedQueueReport {
  VerificationStatus status = VerificationStatus::UnboundedProved;
  std::size_t checkedPipelines = 0;
  std::size_t checkedStages = 0;
  std::size_t checkedQueues = 0;
  std::size_t checkedSchedules = 0;
  std::size_t exploredCompositeStates = 0;
  std::size_t exploredCompositeTransitions = 0;
  std::size_t validatedCertificates = 0;
  std::set<std::string> conditionalFairnessTags;
  std::vector<CompatibilityCertificate> certificates;
  std::vector<GPUBoundedQueueViolation> violations;

  bool accepted() const noexcept {
    return violations.empty() &&
           (status == VerificationStatus::StaticChecked ||
            status == VerificationStatus::BoundedVerified ||
            status == VerificationStatus::UnboundedProved ||
            status == VerificationStatus::Certified);
  }
};

// Extracts a finite-state behavioral process automaton from a single walk of
// `buildControlFlow(callable)`, mapping local stage port names through
// `portToChannelMap` when provided.
BehavioralAutomaton extractStageAutomaton(
    const graphir::CallableGraph &callable,
    const std::map<std::string, std::string> &portToChannelMap = {});

// Independently validates a `CompatibilityCertificate` (slot conservation in
// every reachable state, transition closure, absence of stuck non-terminal
// states, clean queue drain at terminal states, and fair-cycle progress).
bool validateCompatibilityCertificate(const CompatibilityCertificate &cert,
                                      bool requireCleanQueueDrain = true,
                                      std::string *failureReason = nullptr);

// Composes a GPU pipeline's stages and internal bounded queues while hiding
// internal channel actions to form a boundary summary automaton.
std::optional<BehavioralSummaryAutomaton>
computeBoundarySummary(const GPUPipelineSystemSpec &system,
                       const GPUBoundedQueueOptions &options = {});

// Verifies a single GPU pipeline specification across Layers G1, G2, G4, G5:
// 1. Protocol role/state conformance and SPSC/MPSC queue cardinality (G1).
// 2. Explicit capacity-C (`1..2`) queue actors, slot conservation
//    (`free + reserved + queued = C`), deadlock freedom, credit-seeded cycle
//    liveness, and fair-choice progress markers (G2).
// 3. Workgroup-local co-residency, disjoint warp placement, and workgroup
//    resource budgets, rejecting unscheduled cross-block blocking (G4).
// 4. Acquire/release atomic publication ordering and scope sufficiency (G5).
void verifyGPUPipelineSystem(const GPUPipelineSystemSpec &system,
                             const GPUBoundedQueueOptions &options,
                             GPUBoundedQueueReport &report);

// Extracts and verifies all GPU pipelines in a GraphIR module.
GPUBoundedQueueReport verifyGPUBoundedQueueCompatibility(
    const graphir::Module &module,
    const GPUBoundedQueueOptions &options = {});

} // namespace metreon::verification
