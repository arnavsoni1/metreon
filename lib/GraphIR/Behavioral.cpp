#include "metreon/GraphIR/Behavioral.h"
#include "metreon/Basic/Diagnostic.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace metreon::graphir {
namespace {

const char *directionName(ChannelDirection direction) noexcept {
  return direction == ChannelDirection::Send ? "send" : "receive";
}

const char *deliveryModeName(ChannelDeliveryMode mode) noexcept {
  return mode == ChannelDeliveryMode::SameStep ? "same_step" : "blocking_queue";
}

const char *offerModeName(OfferMode mode) noexcept {
  return mode == OfferMode::Exclusive ? "exclusive" : "conjunctive";
}

const char *asyncKindName(AsyncActionKind kind) noexcept {
  return kind == AsyncActionKind::Issue ? "issue" : "wait";
}

[[noreturn]] void failBehavioral(const std::string &code,
                                 const std::string &message,
                                 SourceLocation location) {
  throw DiagnosticError({"graphir." + code, message, location});
}

std::string requiredAttr(const std::map<std::string, std::string> &attrs,
                         const char *key, const char *code,
                         SourceLocation location) {
  const auto it = attrs.find(key);
  if (it == attrs.end() || it->second.empty()) {
    failBehavioral(code, std::string("missing or empty behavioral attribute `") +
                             key + "`",
                   location);
  }
  return it->second;
}

std::string trimStateQualifier(const std::string &typeText,
                               const std::string &resourceName) {
  const auto space = typeText.rfind(' ');
  std::string core =
      space == std::string::npos ? typeText : typeText.substr(space + 1);
  const auto angle = core.find('<');
  if (angle != std::string::npos) core = core.substr(0, angle);
  const std::string prefix = resourceName + "::";
  if (core.size() > prefix.size() && core.substr(0, prefix.size()) == prefix) {
    core = core.substr(prefix.size());
  }
  return core;
}

void collectLeafStates(const ResourceStateNode &node,
                       std::vector<ResourceStateNode> &leaves,
                       std::map<ResourceNodeId, std::size_t> &byId) {
  if (node.members.empty()) {
    ResourceStateNode copy = node;
    copy.internalTransitions.clear();
    if (!byId.emplace(copy.id, leaves.size()).second) {
      throw std::logic_error("duplicate logical state ID in ResourceGraph");
    }
    leaves.push_back(std::move(copy));
    return;
  }
  for (const auto &member : node.members) {
    collectLeafStates(member, leaves, byId);
  }
}

void collectInternalTransitions(
    const ResourceStateNode &node,
    std::vector<ResourceTransitionEdge> &transitions) {
  for (const auto &edge : node.internalTransitions) {
    transitions.push_back(edge);
  }
  for (const auto &member : node.members) {
    collectInternalTransitions(member, transitions);
  }
}

const ResourceStateNode *findContainerOrState(const ResourceStateNode &root,
                                              ResourceNodeId id) {
  if (root.id == id) return &root;
  for (const auto &member : root.members) {
    if (const auto *found = findContainerOrState(member, id)) return found;
  }
  return nullptr;
}

const ResourceStateNode &firstLeaf(const ResourceStateNode &node) {
  const ResourceStateNode *cursor = &node;
  while (!cursor->members.empty()) cursor = &cursor->members.front();
  return *cursor;
}

const ResourceStateNode &lastLeaf(const ResourceStateNode &node) {
  const ResourceStateNode *cursor = &node;
  while (!cursor->members.empty()) cursor = &cursor->members.back();
  return *cursor;
}

const ResourceStateNode *findLeafByName(const ResourceStateNode &node,
                                        const std::string &name) {
  if (node.members.empty()) {
    return node.name == name ? &node : nullptr;
  }
  for (const auto &member : node.members) {
    if (const auto *found = findLeafByName(member, name)) return found;
  }
  return nullptr;
}

std::string buildOperationSignatureTag(const CallableOperation &op) {
  std::ostringstream out;
  for (const char *key :
       {"type", "await", "effects", "callee_kind", "endpoint", "message",
        "direction", "delivery", "ownership_transfer", "offer_mode",
        "assumption_justified", "obligation", "async_kind", "slot",
        "generation", "source_space", "destination_space"}) {
    const auto it = op.attributes.find(key);
    if (it != op.attributes.end()) {
      out << key << '=' << it->second << ';';
    }
  }
  return out.str();
}

void appendCallableWitnesses(const Module &module,
                             const std::vector<CallableGraph> &graphs,
                             const char *kind,
                             std::vector<CallablePreservationWitness> &out) {
  for (const auto &callable : graphs) {
    CallablePreservationWitness witness;
    witness.kind = kind;
    witness.name = callable.name();
    if (callable.context()) {
      witness.ambientContextOrdinal = module.contextOrdinal(callable.context());
    }
    walkCallableOperationsWithContext(
        callable.body(), {},
        [&](const CallableOperation &op, const ContextMetadataRef &effective) {
          if (op.kind == CallableOperationKind::ContextRegion) {
            if (!op.context) {
              throw std::logic_error("ContextRegion has no context metadata");
            }
            (void)module.contextOrdinal(op.context);
            return;
          }
          SemanticOperationWitness item;
          item.id = op.id;
          item.kind = op.kind;
          item.name = op.name;
          item.operands = op.operands;
          // Only record inherited region context for context-sensitive actions
          // or operations that explicitly carried context metadata.
          const bool inheritsRegion =
              op.kind == CallableOperationKind::Call ||
              isBehavioralOrAsyncOperation(op.kind);
          const ContextMetadataRef resolved =
              op.context ? op.context
                         : (inheritsRegion ? effective : ContextMetadataRef{});
          if (resolved) {
            // Context-free procedure/stage calls inside a ContextRegion are
            // never produced by ContextHoistingPass; if an explicit or hoisted
            // call carries context, record its exact module ordinal.
            item.effectiveContextOrdinal = module.contextOrdinal(resolved);
          }
          if (op.loopContext) {
            item.loopContextOrdinal = module.contextOrdinal(op.loopContext);
          }
          item.signatureTag = buildOperationSignatureTag(op);
          witness.operations.push_back(std::move(item));
        });
    out.push_back(std::move(witness));
  }
}

ResourcePreservationWitness captureResourceWitness(const ResourceGraph &graph) {
  const auto expanded = expandResourceGraph(graph);
  ResourcePreservationWitness witness;
  witness.name = expanded.name;

  std::set<ResourceNodeId> reachableIds;
  if (expanded.entryStates.empty()) {
    for (const auto &state : expanded.states) reachableIds.insert(state.id);
  } else {
    std::map<ResourceNodeId, std::vector<ResourceNodeId>> adj;
    for (const auto &state : expanded.states) adj[state.id];
    for (const auto &edge : expanded.transitions) {
      adj[edge.source].push_back(edge.target);
    }
    std::vector<ResourceNodeId> worklist(expanded.entryStates.begin(),
                                         expanded.entryStates.end());
    while (!worklist.empty()) {
      const auto current = worklist.back();
      worklist.pop_back();
      if (!reachableIds.insert(current).second) continue;
      if (const auto it = adj.find(current); it != adj.end()) {
        for (const auto next : it->second) worklist.push_back(next);
      }
    }
  }

  for (const auto &state : expanded.states) {
    if (reachableIds.contains(state.id)) {
      witness.reachableStates.emplace_back(state.id, state.name);
    }
  }
  for (const auto &edge : expanded.transitions) {
    if (reachableIds.contains(edge.source) &&
        reachableIds.contains(edge.target)) {
      witness.reachableTransitions.push_back(edge);
    }
  }
  std::sort(witness.reachableStates.begin(), witness.reachableStates.end());
  std::sort(witness.reachableTransitions.begin(),
            witness.reachableTransitions.end(),
            [](const ResourceTransitionEdge &a, const ResourceTransitionEdge &b) {
              if (a.source != b.source) return a.source < b.source;
              if (a.target != b.target) return a.target < b.target;
              return a.attributes < b.attributes;
            });
  return witness;
}

} // namespace

CallableOperation makeChannelAction(CallableGraph &callable,
                                    const ChannelActionSpec &spec) {
  const auto kind = spec.direction == ChannelDirection::Send
                        ? CallableOperationKind::ChannelSend
                        : CallableOperationKind::ChannelReceive;
  auto op = callable.makeOperation(
      kind, spec.endpoint + "::" + spec.message, spec.location);
  op.context = spec.context;
  op.operands = spec.operands;
  op.attributes = {
      {"type", spec.resultType},
      {"endpoint", spec.endpoint},
      {"message", spec.message},
      {"direction", directionName(spec.direction)},
      {"delivery", deliveryModeName(spec.delivery)},
      {"ownership_transfer", spec.transfersOwnership ? "move" : "none"},
  };
  return op;
}

CallableOperation makeOfferOperation(CallableGraph &callable,
                                     const OfferSpec &spec) {
  auto op = callable.makeOperation(CallableOperationKind::Offer, "", spec.location);
  op.context = spec.context;
  op.condition = spec.readinessPrelude;
  op.alternative = spec.defaultContinuation;
  op.attributes = {
      {"type", "void"},
      {"offer_mode", offerModeName(spec.mode)},
      {"branch_count", std::to_string(spec.branches.size())},
  };
  for (const auto &branch : spec.branches) {
    op.operands.push_back(branch.availabilityOperand);
    auto arm = callable.makeOperation(
        CallableOperationKind::Block,
        branch.endpoint + "::" + branch.message, branch.location);
    arm.attributes = {
        {"type", "void"},
        {"endpoint", branch.endpoint},
        {"message", branch.message},
        {"direction", directionName(branch.direction)},
    };
    arm.body = branch.continuation;
    op.offerBranches.push_back(std::move(arm));
  }
  return op;
}

CallableOperation makeFairChoiceOperation(CallableGraph &callable,
                                          const FairChoiceSpec &spec) {
  auto op = callable.makeOperation(CallableOperationKind::FairChoice, spec.tag,
                                   spec.location);
  op.condition = spec.conditionPrelude;
  op.operands = {spec.predicateOperand};
  op.body = spec.trueBranch;
  op.alternative = spec.falseBranch;
  op.attributes = {
      {"type", "void"},
      {"fair_tag", spec.tag},
      {"assumption_justified", spec.assumptionJustified ? "true" : "false"},
  };
  return op;
}

CallableOperation makeProgressMarkerOperation(CallableGraph &callable,
                                              const ProgressMarkerSpec &spec) {
  auto op = callable.makeOperation(CallableOperationKind::ProgressMarker,
                                   spec.marker, spec.location);
  op.context = spec.context;
  op.attributes = {
      {"type", "void"},
      {"obligation", spec.obligation},
  };
  return op;
}

CallableOperation makeAsyncTokenOperation(CallableGraph &callable,
                                          const AsyncTokenSpec &spec) {
  const auto kind = spec.kind == AsyncActionKind::Issue
                        ? CallableOperationKind::AsyncIssue
                        : CallableOperationKind::AsyncWait;
  auto op = callable.makeOperation(
      kind, spec.slot + "@" + spec.generation, spec.location);
  op.context = spec.context;
  op.operands = spec.operands;
  op.attributes = {
      {"type", spec.resultType},
      {"async_kind", asyncKindName(spec.kind)},
      {"slot", spec.slot},
      {"generation", spec.generation},
      {"loan_policy", "retain_until_matching_wait"},
  };
  if (!spec.sourceAddressSpace.empty()) {
    op.attributes.emplace("source_space", spec.sourceAddressSpace);
  }
  if (!spec.destinationAddressSpace.empty()) {
    op.attributes.emplace("destination_space", spec.destinationAddressSpace);
  }
  return op;
}

void validateBehavioralGraphIR(const Module &module) {
  const auto checkCallables = [&](const std::vector<CallableGraph> &graphs) {
    for (const auto &callable : graphs) {
      if (callable.context()) {
        module.requireContextUse(callable.context(),
                                 ContextUse::MetadataAttachment);
      }
      walkCallableOperationsWithContext(
          callable.body(), {},
          [&](const CallableOperation &op, const ContextMetadataRef &effective) {
            if (op.context) {
              module.requireContextUse(op.context,
                                       ContextUse::MetadataAttachment);
            }
            if (op.loopContext) {
              module.requireContextUse(op.loopContext,
                                       ContextUse::MetadataAttachment);
            }
            if (effective) {
              module.requireContextUse(effective,
                                       ContextUse::MetadataAttachment);
            }
            switch (op.kind) {
            case CallableOperationKind::ContextRegion:
              if (!op.context) {
                failBehavioral("empty_context_region",
                               "context region requires explicit context metadata",
                               op.location);
              }
              break;
            case CallableOperationKind::ChannelSend:
            case CallableOperationKind::ChannelReceive: {
              (void)requiredAttr(op.attributes, "endpoint",
                                 "invalid_channel_action", op.location);
              (void)requiredAttr(op.attributes, "message",
                                 "invalid_channel_action", op.location);
              const auto dir = requiredAttr(op.attributes, "direction",
                                            "invalid_channel_action", op.location);
              const auto expectedDir =
                  op.kind == CallableOperationKind::ChannelSend ? "send"
                                                                : "receive";
              if (dir != expectedDir) {
                failBehavioral("invalid_channel_action",
                               "channel operation direction attribute disagrees "
                               "with operation kind",
                               op.location);
              }
              const auto delivery =
                  requiredAttr(op.attributes, "delivery",
                               "invalid_channel_action", op.location);
              if (delivery != "same_step" && delivery != "blocking_queue") {
                failBehavioral("invalid_channel_action",
                               "unsupported channel delivery mode `" + delivery +
                                   "`",
                               op.location);
              }
              break;
            }
            case CallableOperationKind::Offer: {
              const auto mode = requiredAttr(op.attributes, "offer_mode",
                                             "invalid_offer", op.location);
              if (mode != "exclusive" && mode != "conjunctive") {
                failBehavioral("invalid_offer",
                               "offer_mode must be `exclusive` or `conjunctive`",
                               op.location);
              }
              if (op.offerBranches.empty() ||
                  op.operands.size() != op.offerBranches.size()) {
                failBehavioral("invalid_offer",
                               "offer requires at least one branch and matching "
                               "availability operands",
                               op.location);
              }
              for (const auto &branch : op.offerBranches) {
                (void)requiredAttr(branch.attributes, "endpoint",
                                   "invalid_offer", branch.location);
                (void)requiredAttr(branch.attributes, "message",
                                   "invalid_offer", branch.location);
              }
              break;
            }
            case CallableOperationKind::FairChoice: {
              if (op.name.empty() || op.operands.size() != 1) {
                failBehavioral("invalid_fair_choice",
                               "fair choice requires a tag name and one Boolean "
                               "predicate operand",
                               op.location);
              }
              const auto justified =
                  requiredAttr(op.attributes, "assumption_justified",
                               "invalid_fair_choice", op.location);
              if (justified != "true" && justified != "false") {
                failBehavioral("invalid_fair_choice",
                               "fair choice assumption_justified must be boolean",
                               op.location);
              }
              break;
            }
            case CallableOperationKind::ProgressMarker:
              if (op.name.empty()) {
                failBehavioral("invalid_progress_marker",
                               "progress marker requires a non-empty identifier",
                               op.location);
              }
              (void)requiredAttr(op.attributes, "obligation",
                                 "invalid_progress_marker", op.location);
              break;
            case CallableOperationKind::AsyncIssue:
            case CallableOperationKind::AsyncWait: {
              (void)requiredAttr(op.attributes, "slot", "invalid_async_token",
                                 op.location);
              (void)requiredAttr(op.attributes, "generation",
                                 "invalid_async_token", op.location);
              const auto asyncKind = requiredAttr(
                  op.attributes, "async_kind", "invalid_async_token", op.location);
              const auto expected =
                  op.kind == CallableOperationKind::AsyncIssue ? "issue" : "wait";
              if (asyncKind != expected) {
                failBehavioral("invalid_async_token",
                               "async token operation disagrees with async_kind",
                               op.location);
              }
              break;
            }
            default:
              break;
            }
          });
    }
  };

  checkCallables(module.kernelGraphs());
  checkCallables(module.procedureGraphs());
  checkCallables(module.stageGraphs());

  for (const auto &definition : module.compositionDefinitions()) {
    if (definition.kind != DefinitionKind::Schedule) continue;
    for (const auto &record : definition.records) {
      if (record.kind != "fused_kernel") continue;
      for (const auto &region : record.contextRegions) {
        if (!region.context) {
          failBehavioral("invalid_stage_context_region",
                         "fused stage context region has no context metadata",
                         region.location);
        }
        module.requireContextUse(region.context, ContextUse::MetadataAttachment);
        if (region.stages.empty()) {
          failBehavioral("empty_stage_context_region",
                         "fused stage context region must cover at least one stage",
                         region.location);
        }
        for (const auto stageRef : region.stages) {
          if (stageRef.kind != DefinitionKind::Stage ||
              stageRef.index >= module.stageGraphs().size()) {
            failBehavioral("invalid_stage_context_region",
                           "stage context region references a non-stage definition",
                           region.location);
          }
          const bool inWorker = std::any_of(
              record.references.begin(), record.references.end(),
              [&](const DefinitionReference &member) {
                return member.kind == stageRef.kind &&
                       member.index == stageRef.index;
              });
          if (!inWorker) {
            failBehavioral("stage_region_outside_worker",
                           "stage context region references a stage outside its "
                           "fused worker",
                           region.location);
          }
          const auto &stageCtx = module.stageGraphs()[stageRef.index].context();
          if (!stageCtx || !stageCtx->hasSameIdentity(*region.context)) {
            failBehavioral("stage_region_context_mismatch",
                           "stage context region disagrees with stage's nominal "
                           "execution context",
                           region.location);
          }
        }
      }
    }
  }
}

ExpandedResourceGraph expandResourceGraph(const ResourceGraph &graph) {
  ExpandedResourceGraph expanded;
  expanded.name = graph.name();
  expanded.location = graph.location();

  std::map<ResourceNodeId, std::size_t> leafIndexById;
  for (const auto &node : graph.states()) {
    collectLeafStates(node, expanded.states, leafIndexById);
    collectInternalTransitions(node, expanded.transitions);
  }

  const auto resolveContainerEndpoint =
      [&](ResourceNodeId id, const std::map<std::string, std::string> &attrs,
          const char *typeKey, bool isSource) -> ResourceNodeId {
    const ResourceStateNode *matched = nullptr;
    for (const auto &root : graph.states()) {
      if ((matched = findContainerOrState(root, id)) != nullptr) break;
    }
    if (!matched) {
      throw std::logic_error(
          "ResourceGraph transition refers to an unknown state or container ID");
    }
    if (matched->members.empty()) return matched->id;
    if (const auto it = attrs.find(typeKey); it != attrs.end()) {
      const auto logicalName = trimStateQualifier(it->second, graph.name());
      if (const auto *namedLeaf = findLeafByName(*matched, logicalName)) {
        return namedLeaf->id;
      }
      throw std::logic_error(
          "collapsed container does not contain transition's logical state `" +
          logicalName + "`");
    }
    return isSource ? lastLeaf(*matched).id : firstLeaf(*matched).id;
  };

  for (const auto &edge : graph.transitions()) {
    ResourceTransitionEdge resolved = edge;
    resolved.source =
        resolveContainerEndpoint(edge.source, edge.attributes, "source_type", true);
    resolved.target = resolveContainerEndpoint(edge.target, edge.attributes,
                                               "target_type", false);
    expanded.transitions.push_back(std::move(resolved));
  }

  for (const auto entryId : graph.entryStates()) {
    const ResourceStateNode *matched = nullptr;
    for (const auto &root : graph.states()) {
      if ((matched = findContainerOrState(root, entryId)) != nullptr) break;
    }
    if (!matched) {
      throw std::logic_error("ResourceGraph entry refers to an unknown state ID");
    }
    const auto resolvedId = firstLeaf(*matched).id;
    if (std::find(expanded.entryStates.begin(), expanded.entryStates.end(),
                  resolvedId) == expanded.entryStates.end()) {
      expanded.entryStates.push_back(resolvedId);
    }
  }

  for (const auto &edge : expanded.transitions) {
    if (!leafIndexById.contains(edge.source) ||
        !leafIndexById.contains(edge.target)) {
      throw std::logic_error(
          "expanded ResourceGraph transition refers to a non-leaf state ID");
    }
  }

  return expanded;
}

PreservationSnapshot capturePreservationSnapshot(const Module &module) {
  PreservationSnapshot snapshot;
  appendCallableWitnesses(module, module.kernelGraphs(), "kernel",
                          snapshot.callables);
  appendCallableWitnesses(module, module.procedureGraphs(), "procedure",
                          snapshot.callables);
  appendCallableWitnesses(module, module.stageGraphs(), "stage",
                          snapshot.callables);

  for (const auto &resource : module.resourceGraphs()) {
    if (resource.attributes().contains("kind")) continue;
    snapshot.resources.push_back(captureResourceWitness(resource));
  }

  for (const auto &definition : module.compositionDefinitions()) {
    if (definition.kind != DefinitionKind::Schedule) continue;
    for (const auto &record : definition.records) {
      if (record.kind != "fused_kernel") continue;
      ScheduleWorkerWitness worker;
      worker.scheduleName = definition.name;
      worker.workerName = record.name;
      for (const auto ref : record.references) {
        worker.stageOrder.push_back(module.definitionName(ref));
        if (ref.kind == DefinitionKind::Stage &&
            ref.index < module.stageGraphs().size() &&
            module.stageGraphs()[ref.index].context()) {
          worker.stageContextOrdinals.push_back(
              module.contextOrdinal(module.stageGraphs()[ref.index].context()));
        } else {
          worker.stageContextOrdinals.push_back(std::nullopt);
        }
      }
      snapshot.workers.push_back(std::move(worker));
    }
  }

  return snapshot;
}

void verifyPassPreservation(const PreservationSnapshot &before,
                            const Module &after) {
  validateBehavioralGraphIR(after);
  const auto current = capturePreservationSnapshot(after);

  if (before.callables.size() != current.callables.size()) {
    throw std::runtime_error("pass changed callable count");
  }
  for (std::size_t i = 0; i < before.callables.size(); ++i) {
    const auto &lhs = before.callables[i];
    const auto &rhs = current.callables[i];
    if (lhs.kind != rhs.kind || lhs.name != rhs.name ||
        lhs.ambientContextOrdinal != rhs.ambientContextOrdinal) {
      throw std::runtime_error("pass altered callable identity or ambient context: " +
                               lhs.name);
    }
    if (lhs.operations.size() != rhs.operations.size()) {
      throw std::runtime_error(
          "pass dropped or duplicated semantic operations in callable `" +
          lhs.name + "`");
    }
    for (std::size_t j = 0; j < lhs.operations.size(); ++j) {
      const auto &opBefore = lhs.operations[j];
      const auto &opAfter = rhs.operations[j];
      if (opBefore.id != opAfter.id || opBefore.kind != opAfter.kind ||
          opBefore.name != opAfter.name ||
          opBefore.operands != opAfter.operands ||
          opBefore.effectiveContextOrdinal != opAfter.effectiveContextOrdinal ||
          opBefore.loopContextOrdinal != opAfter.loopContextOrdinal ||
          opBefore.signatureTag != opAfter.signatureTag) {
        throw std::runtime_error(
            "pass changed operation order, operands, effective context, or "
            "semantic contract in callable `" +
            lhs.name + "`");
      }
    }
  }

  if (before.resources.size() != current.resources.size()) {
    throw std::runtime_error("pass changed ordinary resource graph count");
  }
  for (std::size_t i = 0; i < before.resources.size(); ++i) {
    const auto &lhs = before.resources[i];
    const auto &rhs = current.resources[i];
    if (lhs.name != rhs.name || lhs.reachableStates != rhs.reachableStates ||
        lhs.reachableTransitions.size() != rhs.reachableTransitions.size()) {
      throw std::runtime_error(
          "pass altered reachable logical states or transitions in resource `" +
          lhs.name + "`");
    }
    for (std::size_t j = 0; j < lhs.reachableTransitions.size(); ++j) {
      const auto &edgeA = lhs.reachableTransitions[j];
      const auto &edgeB = rhs.reachableTransitions[j];
      if (edgeA.source != edgeB.source || edgeA.target != edgeB.target ||
          edgeA.attributes != edgeB.attributes) {
        throw std::runtime_error(
            "pass altered collapsed or external transition contract in `" +
            lhs.name + "`");
      }
    }
  }

  if (before.workers.size() != current.workers.size()) {
    throw std::runtime_error("pass changed schedule fused-worker count");
  }
  for (std::size_t i = 0; i < before.workers.size(); ++i) {
    const auto &lhs = before.workers[i];
    const auto &rhs = current.workers[i];
    if (lhs.scheduleName != rhs.scheduleName ||
        lhs.workerName != rhs.workerName || lhs.stageOrder != rhs.stageOrder ||
        lhs.stageContextOrdinals != rhs.stageContextOrdinals) {
      throw std::runtime_error(
          "pass altered fused worker stage order or nominal stage contexts in `" +
          lhs.scheduleName + "::" + lhs.workerName + "`");
    }
  }
}

} // namespace metreon::graphir
