#pragma once

#include "metreon/GraphIR/Graph.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace metreon::graphir {

enum class ChannelDirection {
  Send,
  Receive,
};

enum class ChannelDeliveryMode {
  SameStep,
  BlockingQueue,
};

enum class OfferMode {
  Exclusive,
  Conjunctive,
};

enum class AsyncActionKind {
  Issue,
  Wait,
};

struct ChannelActionSpec {
  ChannelDirection direction = ChannelDirection::Send;
  std::string endpoint;
  std::string message;
  ChannelDeliveryMode delivery = ChannelDeliveryMode::BlockingQueue;
  bool transfersOwnership = false;
  std::string resultType = "void";
  std::vector<CallableValueId> operands;
  ContextMetadataRef context;
  SourceLocation location;
};

struct OfferBranchSpec {
  ChannelDirection direction = ChannelDirection::Receive;
  std::string endpoint;
  std::string message;
  CallableValueId availabilityOperand = 0;
  std::vector<CallableOperation> continuation;
  SourceLocation location;
};

struct OfferSpec {
  OfferMode mode = OfferMode::Exclusive;
  std::vector<CallableOperation> readinessPrelude;
  std::vector<OfferBranchSpec> branches;
  std::vector<CallableOperation> defaultContinuation;
  ContextMetadataRef context;
  SourceLocation location;
};

struct FairChoiceSpec {
  std::string tag;
  bool assumptionJustified = false;
  std::vector<CallableOperation> conditionPrelude;
  CallableValueId predicateOperand = 0;
  std::vector<CallableOperation> trueBranch;
  std::vector<CallableOperation> falseBranch;
  SourceLocation location;
};

struct ProgressMarkerSpec {
  std::string marker;
  std::string obligation;
  ContextMetadataRef context;
  SourceLocation location;
};

struct AsyncTokenSpec {
  AsyncActionKind kind = AsyncActionKind::Issue;
  std::string slot;
  std::string generation;
  std::string sourceAddressSpace;
  std::string destinationAddressSpace;
  std::string resultType = "void";
  std::vector<CallableValueId> operands;
  ContextMetadataRef context;
  SourceLocation location;
};

CallableOperation makeChannelAction(CallableGraph &callable,
                                    const ChannelActionSpec &spec);
CallableOperation makeOfferOperation(CallableGraph &callable,
                                     const OfferSpec &spec);
CallableOperation makeFairChoiceOperation(CallableGraph &callable,
                                          const FairChoiceSpec &spec);
CallableOperation makeProgressMarkerOperation(CallableGraph &callable,
                                              const ProgressMarkerSpec &spec);
CallableOperation makeAsyncTokenOperation(CallableGraph &callable,
                                          const AsyncTokenSpec &spec);

// Validates behavioral/async callable operations and schedule StageContextRegions
// across the module. Throws DiagnosticError or std::logic_error on invalid IR.
void validateBehavioralGraphIR(const Module &module);

struct ExpandedResourceGraph {
  std::string name;
  std::vector<ResourceStateNode> states;
  std::vector<ResourceTransitionEdge> transitions;
  std::vector<ResourceNodeId> entryStates;
  SourceLocation location;
};

// Reconstructs the uncollapsed logical state and transition sequence of a
// ResourceGraph, visiting each collapsed member and internalTransition once
// and mapping external container edges back to their exact leaf endpoints.
ExpandedResourceGraph expandResourceGraph(const ResourceGraph &graph);

struct SemanticOperationWitness {
  CallableValueId id = 0;
  CallableOperationKind kind = CallableOperationKind::Variable;
  std::string name;
  std::vector<CallableValueId> operands;
  std::optional<std::size_t> effectiveContextOrdinal;
  std::optional<std::size_t> loopContextOrdinal;
  std::string signatureTag;
};

struct CallablePreservationWitness {
  std::string kind;
  std::string name;
  std::optional<std::size_t> ambientContextOrdinal;
  std::vector<SemanticOperationWitness> operations;
};

struct ResourcePreservationWitness {
  std::string name;
  std::vector<std::pair<ResourceNodeId, std::string>> reachableStates;
  std::vector<ResourceTransitionEdge> reachableTransitions;
};

struct ScheduleWorkerWitness {
  std::string scheduleName;
  std::string workerName;
  std::vector<std::string> stageOrder;
  std::vector<std::optional<std::size_t>> stageContextOrdinals;
};

struct PreservationSnapshot {
  std::vector<CallablePreservationWitness> callables;
  std::vector<ResourcePreservationWitness> resources;
  std::vector<ScheduleWorkerWitness> workers;
};

PreservationSnapshot capturePreservationSnapshot(const Module &module);
void verifyPassPreservation(const PreservationSnapshot &before,
                            const Module &after);

} // namespace metreon::graphir
