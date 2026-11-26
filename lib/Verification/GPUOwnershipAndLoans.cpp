#include "metreon/Verification/GPUOwnershipAndLoans.h"

#include "metreon/GraphIR/Behavioral.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace metreon::verification {
namespace {

bool startsWith(std::string_view text, std::string_view prefix) noexcept {
  return text.size() >= prefix.size() &&
         text.substr(0, prefix.size()) == prefix;
}

std::string_view stripGenerics(std::string_view text) noexcept {
  const auto pos = text.find('<');
  return pos == std::string_view::npos ? text : text.substr(0, pos);
}

std::string_view trimSpaces(std::string_view text) noexcept {
  while (!text.empty() &&
         (text.front() == ' ' || text.front() == '\t' || text.front() == '\n' ||
          text.front() == '\r')) {
    text.remove_prefix(1);
  }
  while (!text.empty() &&
         (text.back() == ' ' || text.back() == '\t' || text.back() == '\n' ||
          text.back() == '\r')) {
    text.remove_suffix(1);
  }
  return text;
}

bool isOwnedTypeText(std::string_view typeText) noexcept {
  const auto trimmed = trimSpaces(typeText);
  return startsWith(trimmed, "own ") || startsWith(trimmed, "own\t");
}

std::string_view stripOwnership(std::string_view typeText) noexcept {
  auto trimmed = trimSpaces(typeText);
  if (startsWith(trimmed, "own ") || startsWith(trimmed, "own\t")) {
    trimmed.remove_prefix(4);
    trimmed = trimSpaces(trimmed);
  }
  return trimmed;
}

std::optional<GPUAddressSpace>
parseAddressSpaceToken(std::string_view token) noexcept {
  token = trimSpaces(token);
  if (token == "host::default" || token == "host::pinned" || token == "host")
    return GPUAddressSpace::HostDefault;
  if (token == "gpu::global" || token == "global")
    return GPUAddressSpace::GlobalDevice;
  if (token == "gpu::shared" || token == "shared" || token == "workgroup")
    return GPUAddressSpace::WorkgroupShared;
  if (token == "gpu::constant" || token == "constant")
    return GPUAddressSpace::Constant;
  if (token == "gpu::private" || token == "private" || token == "local")
    return GPUAddressSpace::ThreadPrivate;
  return std::nullopt;
}

void recordViolation(GPUOwnershipReport &report, std::string code,
                     std::string callableName, std::string subject,
                     std::string message, SourceLocation location) {
  report.status = VerificationStatus::Rejected;
  report.violations.push_back(GPUOwnershipViolation{
      std::move(code), std::move(callableName), std::move(subject),
      std::move(message), location});
}

struct ResourceStateMetadata {
  std::string qualifiedName;
  ResourceLifecycleKind lifecycle = ResourceLifecycleKind::LinearOwned;
  bool hasEventField = false;
  bool hasAwaitOutgoing = false;
};

struct TransitionMetadata {
  std::string qualifiedName;
  std::string sourceState;
  std::string targetState;
  bool isAwait = false;
  std::vector<bool> ownedRuntimeParams;
};

struct ModuleResourceIndex {
  std::map<std::string, ResourceStateMetadata> states;
  std::map<std::string, TransitionMetadata> transitions;
};

std::vector<std::string> splitTopLevelComma(std::string_view list) {
  std::vector<std::string> parts;
  std::size_t depth = 0;
  std::size_t start = 0;
  for (std::size_t i = 0; i < list.size(); ++i) {
    const char c = list[i];
    if (c == '<' || c == '(' || c == '{') ++depth;
    else if ((c == '>' || c == ')' || c == '}') && depth > 0) --depth;
    else if (c == ',' && depth == 0) {
      parts.emplace_back(trimSpaces(list.substr(start, i - start)));
      start = i + 1;
    }
  }
  if (start <= list.size()) {
    const auto tail = trimSpaces(list.substr(start));
    if (!tail.empty()) parts.emplace_back(tail);
  }
  return parts;
}

ModuleResourceIndex buildResourceIndex(const graphir::Module &module) {
  ModuleResourceIndex index;
  for (const auto &rawGraph : module.resourceGraphs()) {
    if (rawGraph.attributes().contains("kind")) continue;
    const auto expanded = graphir::expandResourceGraph(rawGraph);
    std::map<graphir::ResourceNodeId, std::string> stateNames;
    for (const auto &state : expanded.states) {
      const std::string qualified = expanded.name + "::" + state.name;
      stateNames.emplace(state.id, qualified);
      ResourceStateMetadata meta;
      meta.qualifiedName = qualified;

      for (const auto &[k, v] : state.attributes) {
        if (k.ends_with(".type") && v.find("event<") != std::string::npos) {
          meta.hasEventField = true;
        }
      }
      if (const auto it = state.attributes.find("lifecycle");
          it != state.attributes.end()) {
        if (it->second == "droppable" || it->second == "terminal") {
          meta.lifecycle = ResourceLifecycleKind::Droppable;
        } else if (it->second == "must_complete") {
          meta.lifecycle = ResourceLifecycleKind::MustComplete;
        }
      } else if (meta.hasEventField) {
        meta.lifecycle = ResourceLifecycleKind::MustComplete;
      }
      index.states.emplace(qualified, std::move(meta));
    }

    for (const auto &edge : expanded.transitions) {
      const auto nameIt = edge.attributes.find("name");
      if (nameIt == edge.attributes.end()) continue;
      const std::string qualified = expanded.name + "::" + nameIt->second;
      TransitionMetadata meta;
      meta.qualifiedName = qualified;
      meta.sourceState = stateNames.at(edge.source);
      meta.targetState = stateNames.at(edge.target);
      meta.isAwait = edge.attributes.contains("await") &&
                     edge.attributes.at("await") == "true";

      if (meta.isAwait) {
        auto &srcMeta = index.states.at(meta.sourceState);
        srcMeta.hasAwaitOutgoing = true;
        if (srcMeta.lifecycle != ResourceLifecycleKind::Droppable) {
          srcMeta.lifecycle = ResourceLifecycleKind::MustComplete;
        }
      }

      if (const auto paramIt = edge.attributes.find("parameters");
          paramIt != edge.attributes.end()) {
        for (const auto &entry : splitTopLevelComma(paramIt->second)) {
          const auto colon = entry.find(':');
          if (colon == std::string::npos) continue;
          const auto paramName = trimSpaces(std::string_view(entry).substr(0, colon));
          if (paramName == "@cx" || startsWith(paramName, "@")) continue;
          const auto paramType =
              trimSpaces(std::string_view(entry).substr(colon + 1));
          meta.ownedRuntimeParams.push_back(isOwnedTypeText(paramType) ||
                                            paramName == "self");
        }
      } else {
        meta.ownedRuntimeParams.push_back(true);
      }

      index.transitions.emplace(qualified, std::move(meta));
    }
  }
  return index;
}

enum class HandleStateKind {
  Live,
  Consumed,
  MaybeConsumed,
};

struct HandleTypestate {
  std::string label;
  std::string typeText;
  std::string stateName;
  ResourceLifecycleKind lifecycle = ResourceLifecycleKind::Droppable;
  HandleStateKind state = HandleStateKind::Live;
  bool isDynamicSpan = false;
  SourceLocation declarationLocation;
  SourceLocation lastConsumeLocation;
};

enum class SlotGenerationStatus {
  InFlight,
  Completed,
};

using SlotGenerationKey = std::pair<std::string, std::string>;

struct PathState {
  bool reachable = true;
  std::map<graphir::CallableValueId, HandleTypestate> handles;
  std::vector<ActiveAsyncLoan> activeLoans;
  std::map<SlotGenerationKey, SlotGenerationStatus> generations;
  std::map<SlotGenerationKey, SourceLocation> issueLocations;
};

enum class FlowMask : unsigned {
  Continue = 1u << 0,
  Return = 1u << 1,
  Break = 1u << 2,
  LoopContinue = 1u << 3,
};

struct LoopExitStates {
  std::vector<PathState> breaks;
  std::vector<PathState> continues;
};

class CallableAnalyzer {
public:
  CallableAnalyzer(const graphir::Module &module,
                   const ModuleResourceIndex &resources,
                   const std::map<std::string, const graphir::CallableGraph *> &procedures,
                   const std::map<std::size_t, std::set<std::string>> &contextGrants,
                   GPUOwnershipReport &report)
      : module_(module), resources_(resources), procedures_(procedures),
        contextGrants_(contextGrants), report_(report) {}

  void analyze(const graphir::CallableGraph &callable) {
    ++report_.checkedCallables;
    callable_ = &callable;
    callableName_ = callable.name();
    valueTypes_.clear();
    allocationNames_.clear();
    aliasRoot_.clear();

    const auto ambient = callable.context();
    ambientScope_.reset();
    if (ambient) {
      ambientScope_ = classifyContextExecutionScope(ambient->name());
      if (!ambientScope_ && classifyLoopContextDomain(ambient->name())) {
        recordViolation(report_, "gpu.context.loop_as_execution", callableName_,
                        ambient->name(),
                        "loop context cannot supply callable execution authority",
                        callable.location());
      }
    }

    PathState state;
    for (const auto &param : callable.parameters()) {
      if (param.context) {
        if (ambient && !ambient->hasSameIdentity(*param.context)) {
          recordViolation(report_, "gpu.context.parameter_mismatch",
                          callableName_, param.name,
                          "parameter changes nominal execution context",
                          param.location);
        }
        continue;
      }
      if (!param.id) continue;
      valueTypes_[*param.id] = param.type;
      allocationNames_[*param.id] =
          param.name.empty() ? ("arg_" + std::to_string(*param.id)) : param.name;
      registerValueIfTracked(state, *param.id, param.name, param.type, false,
                             param.location);
    }

    std::vector<LoopExitStates> loopStack;
    const unsigned flow =
        analyzeSequence(callable.body(), ambient, state, loopStack);
    if ((flow & static_cast<unsigned>(FlowMask::Continue)) != 0 &&
        state.reachable) {
      checkExitObligations(state, {}, callable.location());
    }
  }

private:
  const graphir::Module &module_;
  const ModuleResourceIndex &resources_;
  const std::map<std::string, const graphir::CallableGraph *> &procedures_;
  const std::map<std::size_t, std::set<std::string>> &contextGrants_;
  GPUOwnershipReport &report_;
  const graphir::CallableGraph *callable_ = nullptr;
  std::string callableName_;
  std::optional<GPUExecutionScope> ambientScope_;
  std::map<graphir::CallableValueId, std::string> valueTypes_;
  std::map<graphir::CallableValueId, std::string> allocationNames_;
  std::map<graphir::CallableValueId, graphir::CallableValueId> aliasRoot_;

  graphir::CallableValueId resolveRoot(graphir::CallableValueId id) const {
    auto cur = id;
    std::set<graphir::CallableValueId> visited;
    while (visited.insert(cur).second) {
      auto it = aliasRoot_.find(cur);
      if (it == aliasRoot_.end()) break;
      cur = it->second;
    }
    return cur;
  }

  std::string allocationIdFor(graphir::CallableValueId id) const {
    const auto root = resolveRoot(id);
    if (const auto it = allocationNames_.find(root);
        it != allocationNames_.end()) {
      return it->second;
    }
    return "val_" + std::to_string(root);
  }

  std::optional<ResourceStateMetadata>
  lookupResourceState(std::string_view typeText) const {
    const auto base = stripGenerics(stripOwnership(typeText));
    if (const auto it = resources_.states.find(std::string(base));
        it != resources_.states.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  void registerValueIfTracked(PathState &state, graphir::CallableValueId id,
                              std::string_view name, std::string_view typeText,
                              bool dynamicSpanAllocation,
                              SourceLocation location) {
    const auto resMeta = lookupResourceState(typeText);
    const bool explicitOwn = isOwnedTypeText(typeText);
    if (!resMeta && !explicitOwn && !dynamicSpanAllocation) return;

    HandleTypestate handle;
    handle.label =
        name.empty() ? ("%" + std::to_string(id)) : std::string(name);
    handle.typeText = std::string(typeText);
    handle.declarationLocation = location;
    handle.state = HandleStateKind::Live;
    handle.isDynamicSpan = dynamicSpanAllocation;

    if (resMeta) {
      handle.stateName = resMeta->qualifiedName;
      handle.lifecycle = resMeta->lifecycle;
    } else if (dynamicSpanAllocation) {
      handle.lifecycle = ResourceLifecycleKind::Droppable;
    } else if (stripOwnership(typeText).find("event<") !=
               std::string_view::npos) {
      handle.lifecycle = ResourceLifecycleKind::MustComplete;
    } else {
      handle.lifecycle = ResourceLifecycleKind::LinearOwned;
    }
    state.handles[id] = std::move(handle);
  }

  void requireLiveHandle(const PathState &state, graphir::CallableValueId rawId,
                         std::string_view action, SourceLocation location) {
    const auto id = resolveRoot(rawId);
    const auto it = state.handles.find(id);
    if (it == state.handles.end()) return;
    if (it->second.state == HandleStateKind::Consumed) {
      recordViolation(report_, "gpu.ownership.double_consumption", callableName_,
                      it->second.label,
                      "cannot " + std::string(action) + " handle `" +
                          it->second.label + "` after it was already consumed",
                      location);
    } else if (it->second.state == HandleStateKind::MaybeConsumed) {
      recordViolation(report_, "gpu.ownership.double_consumption", callableName_,
                      it->second.label,
                      "cannot " + std::string(action) + " handle `" +
                          it->second.label +
                          "` after partial consumption on a prior branch",
                      location);
    }
  }

  void consumeHandle(PathState &state, graphir::CallableValueId rawId,
                     SourceLocation location) {
    const auto id = resolveRoot(rawId);
    const auto it = state.handles.find(id);
    if (it == state.handles.end()) return;
    requireLiveHandle(state, id, "consume", location);
    it->second.state = HandleStateKind::Consumed;
    it->second.lastConsumeLocation = location;
  }

  void checkBufferAccess(const PathState &state, graphir::CallableValueId rawId,
                         bool isWrite, bool isFree, SourceLocation location) {
    const std::string alloc = allocationIdFor(rawId);
    requireLiveHandle(state, rawId, isFree ? "free" : (isWrite ? "write" : "read"),
                      location);
    for (const auto &loan : state.activeLoans) {
      if (loan.key.allocationId != alloc) continue;
      if (isFree || isWrite || loan.mode == AsyncLoanMode::ExclusiveWrite) {
        recordViolation(
            report_, "gpu.loan.pre_wait_buffer_access", callableName_, alloc,
            std::string(isFree ? "deallocating" : (isWrite ? "writing" : "reading")) +
                " buffer `" + alloc + "` while async loan (" + loan.key.slotId +
                ", gen=" + loan.key.generation + ") is still in flight",
            location);
        return;
      }
    }
  }

  void checkAddressSpaceDereference(GPUExecutionScope scope,
                                    graphir::CallableValueId spanOperand,
                                    std::string_view opName,
                                    SourceLocation location) {
    const auto root = resolveRoot(spanOperand);
    const auto typeIt = valueTypes_.find(root);
    if (typeIt == valueTypes_.end()) return;
    const auto space = classifyTypeAddressSpace(typeIt->second);
    if (!space) return;
    if (!canDereferenceAddressSpace(scope, *space)) {
      recordViolation(
          report_, "gpu.address_space.cross_space_dereference", callableName_,
          std::string(opName),
          std::string(gpuExecutionScopeName(scope)) +
              " context cannot directly dereference `" + typeIt->second +
              "` in address space `" + gpuAddressSpaceName(*space) + "`",
          location);
    }
  }

  void verifyEffectiveContext(const graphir::ContextMetadataRef &ambient,
                              const graphir::ContextMetadataRef &effective,
                              std::string_view subject,
                              SourceLocation location) {
    if (!effective) return;
    if (!ambient || !ambient->hasSameIdentity(*effective)) {
      recordViolation(report_, "gpu.context.nominal_mismatch", callableName_,
                      std::string(subject),
                      "operation changes nominal execution context without an "
                      "authorized boundary",
                      location);
      return;
    }
    const auto scope = classifyContextExecutionScope(effective->name());
    if (!scope && classifyLoopContextDomain(effective->name())) {
      recordViolation(report_, "gpu.context.loop_as_execution", callableName_,
                      effective->name(),
                      "loop evidence context cannot authorize operations",
                      location);
    }
  }

  void verifyLoopEvidence(const graphir::ContextMetadataRef &loopCtx,
                          SourceLocation location) {
    if (!loopCtx) {
      if (ambientScope_) {
        recordViolation(report_, "gpu.context.missing_loop_evidence",
                        callableName_, callableName_,
                        "loop requires grant-free Host::Loop or Gpu::Loop evidence",
                        location);
      }
      return;
    }
    const auto domain = classifyLoopContextDomain(loopCtx->name());
    if (!domain) {
      recordViolation(report_, "gpu.context.invalid_loop_evidence", callableName_,
                      loopCtx->name(),
                      "loop context must be grant-free Host::Loop or Gpu::Loop",
                      location);
      return;
    }
    const auto ordinal = module_.contextOrdinal(loopCtx);
    if (loopCtx->genericArity() != 0 ||
        (contextGrants_.contains(ordinal) && !contextGrants_.at(ordinal).empty())) {
      recordViolation(report_, "gpu.context.grant_bearing_loop_evidence",
                      callableName_, loopCtx->name(),
                      "loop context cannot declare generic parameters or grants",
                      location);
    }
    if (ambientScope_) {
      const bool expectHost = (*ambientScope_ == GPUExecutionScope::Host);
      const bool actualHost = (*domain == GPULoopDomain::HostLoop);
      if (expectHost != actualHost) {
        recordViolation(
            report_, "gpu.context.loop_domain_mismatch", callableName_,
            loopCtx->name(),
            "loop evidence domain disagrees with enclosing execution scope",
            location);
      }
    }
  }

  bool contextHasGrant(const graphir::ContextMetadataRef &ctx,
                       std::string_view capability) const {
    if (!ctx) return false;
    const auto ordinal = module_.contextOrdinal(ctx);
    const auto it = contextGrants_.find(ordinal);
    return it != contextGrants_.end() &&
           it->second.contains(std::string(capability));
  }

  void handleAsyncIssue(const graphir::CallableOperation &op,
                        const graphir::ContextMetadataRef &effective,
                        PathState &state) {
    ++report_.checkedAsyncIssues;
    verifyEffectiveContext(callable_->context(), effective, "async_issue",
                           op.location);
    if (effective) {
      const auto scope = classifyContextExecutionScope(effective->name());
      if (scope && !isCapabilityAllowedInScope(*scope, "gpu_async_copy")) {
        recordViolation(report_, "gpu.context.scope_violation", callableName_,
                        "gpu_async_copy",
                        "scope `" + std::string(gpuExecutionScopeName(*scope)) +
                            "` cannot issue async copy",
                        op.location);
      }
    }

    const std::string slot =
        op.attributes.contains("slot") ? op.attributes.at("slot") : op.name;
    const std::string generation = op.attributes.contains("generation")
                                       ? op.attributes.at("generation")
                                       : "";
    if (slot.empty() || generation.empty()) {
      recordViolation(report_, "gpu.loan.missing_async_identity", callableName_,
                      op.name, "async issue requires non-empty slot and generation",
                      op.location);
      return;
    }

    if (const auto srcSpaceIt = op.attributes.find("source_space");
        srcSpaceIt != op.attributes.end() && !srcSpaceIt->second.empty()) {
      const auto srcSpace = parseAddressSpaceToken(srcSpaceIt->second);
      const auto dstSpace =
          op.attributes.contains("destination_space")
              ? parseAddressSpaceToken(op.attributes.at("destination_space"))
              : std::nullopt;
      if (ambientScope_ && *ambientScope_ != GPUExecutionScope::Host) {
        if ((srcSpace && *srcSpace == GPUAddressSpace::HostDefault) ||
            (dstSpace && *dstSpace == GPUAddressSpace::HostDefault)) {
          recordViolation(
              report_, "gpu.address_space.cross_space_dereference",
              callableName_, slot,
              "device async copy cannot reference host default memory space",
              op.location);
        }
      }
    }

    const SlotGenerationKey sgKey{slot, generation};
    if (const auto genIt = state.generations.find(sgKey);
        genIt != state.generations.end()) {
      recordViolation(
          report_, "gpu.loan.stale_generation_reuse", callableName_,
          slot + "@" + generation,
          "async token (" + slot + ", gen=" + generation +
              ") reuses an active or already-completed generation",
          op.location);
      return;
    }

    // Ensure the slot does not already hold an unawaited prior generation unless
    // explicitly pipelined on a distinct slot.
    state.generations[sgKey] = SlotGenerationStatus::InFlight;
    state.issueLocations[sgKey] = op.location;

    if (op.operands.size() >= 2) {
      checkBufferAccess(state, op.operands[0], /*isWrite=*/false, /*isFree=*/false,
                        op.location);
      checkBufferAccess(state, op.operands[1], /*isWrite=*/true, /*isFree=*/false,
                        op.location);
      state.activeLoans.push_back(ActiveAsyncLoan{
          AsyncLoanKey{allocationIdFor(op.operands[0]), slot, generation},
          AsyncLoanMode::ImmutableRead, op.location});
      state.activeLoans.push_back(ActiveAsyncLoan{
          AsyncLoanKey{allocationIdFor(op.operands[1]), slot, generation},
          AsyncLoanMode::ExclusiveWrite, op.location});
    } else if (op.operands.size() == 1) {
      checkBufferAccess(state, op.operands[0], /*isWrite=*/true, /*isFree=*/false,
                        op.location);
      state.activeLoans.push_back(ActiveAsyncLoan{
          AsyncLoanKey{allocationIdFor(op.operands[0]), slot, generation},
          AsyncLoanMode::ExclusiveWrite, op.location});
    } else {
      state.activeLoans.push_back(ActiveAsyncLoan{
          AsyncLoanKey{"slot:" + slot, slot, generation},
          AsyncLoanMode::ExclusiveWrite, op.location});
    }
  }

  void handleAsyncWait(const graphir::CallableOperation &op,
                       const graphir::ContextMetadataRef &effective,
                       PathState &state) {
    ++report_.checkedAsyncWaits;
    verifyEffectiveContext(callable_->context(), effective, "async_wait",
                           op.location);

    const std::string slot =
        op.attributes.contains("slot") ? op.attributes.at("slot") : op.name;
    const std::string generation = op.attributes.contains("generation")
                                       ? op.attributes.at("generation")
                                       : "";
    const SlotGenerationKey sgKey{slot, generation};
    const auto genIt = state.generations.find(sgKey);
    if (genIt == state.generations.end()) {
      recordViolation(report_, "gpu.loan.unissued_async_wait", callableName_,
                      slot + "@" + generation,
                      "async wait references unissued token (" + slot +
                          ", gen=" + generation + ")",
                      op.location);
      return;
    }
    if (genIt->second == SlotGenerationStatus::Completed) {
      recordViolation(
          report_, "gpu.loan.stale_generation_reuse", callableName_,
          slot + "@" + generation,
          "stale async completion reuse on already-completed token (" + slot +
              ", gen=" + generation + ")",
          op.location);
      return;
    }

    genIt->second = SlotGenerationStatus::Completed;
    std::erase_if(state.activeLoans, [&](const ActiveAsyncLoan &loan) {
      return loan.key.slotId == slot && loan.key.generation == generation;
    });
  }

  void handleCall(const graphir::CallableOperation &op,
                  const graphir::ContextMetadataRef &effective,
                  PathState &state) {
    verifyEffectiveContext(callable_->context(), effective, op.name,
                           op.location);
    const std::string calleeKind =
        op.attributes.contains("callee_kind") ? op.attributes.at("callee_kind")
                                              : "procedure";
    const std::string resultType =
        op.attributes.contains("type") ? op.attributes.at("type") : "void";
    valueTypes_[op.id] = resultType;

    if (calleeKind == "gpu_builtin" || calleeKind == "host_builtin") {
      if (effective) {
        if (const auto scope = classifyContextExecutionScope(effective->name())) {
          if (!isIntrinsicAllowedInScope(*scope, op.name)) {
            recordViolation(
                report_, "gpu.context.scope_violation", callableName_, op.name,
                "intrinsic `" + op.name + "` is not permitted in `" +
                    gpuExecutionScopeName(*scope) + "` execution scope",
                op.location);
          }
        }
      }

      if (op.name == "gpu::alloc" || op.name == "host::alloc") {
        allocationNames_[op.id] = op.name + "_" + std::to_string(op.id);
        registerValueIfTracked(state, op.id, allocationNames_[op.id], resultType,
                               /*dynamicSpanAllocation=*/true, op.location);
      } else if (op.name == "gpu::free" || op.name == "host::free") {
        if (!op.operands.empty()) {
          if (ambientScope_) {
            const auto expectedSpace =
                op.name == "gpu::free" ? GPUAddressSpace::GlobalDevice
                                       : GPUAddressSpace::HostDefault;
            const auto actualType = valueTypes_[resolveRoot(op.operands[0])];
            const auto actualSpace = classifyTypeAddressSpace(actualType);
            if (actualSpace && *actualSpace != expectedSpace) {
              recordViolation(report_,
                              "gpu.address_space.cross_space_dereference",
                              callableName_, op.name,
                              "`" + op.name + "` cannot deallocate `" +
                                  actualType + "`",
                              op.location);
            }
          }
          checkBufferAccess(state, op.operands[0], /*isWrite=*/true,
                            /*isFree=*/true, op.location);
          consumeHandle(state, op.operands[0], op.location);
        }
      } else if (op.name == "gpu::load" || op.name == "host::load") {
        if (!op.operands.empty()) {
          const auto opScope = op.name == "gpu::load"
                                   ? ambientScope_.value_or(GPUExecutionScope::Thread)
                                   : GPUExecutionScope::Host;
          checkAddressSpaceDereference(opScope, op.operands[0], op.name,
                                       op.location);
          checkBufferAccess(state, op.operands[0], /*isWrite=*/false,
                            /*isFree=*/false, op.location);
        }
      } else if (op.name == "gpu::store" || op.name == "host::store") {
        if (!op.operands.empty()) {
          const auto opScope = op.name == "gpu::store"
                                   ? ambientScope_.value_or(GPUExecutionScope::Thread)
                                   : GPUExecutionScope::Host;
          checkAddressSpaceDereference(opScope, op.operands[0], op.name,
                                       op.location);
          checkBufferAccess(state, op.operands[0], /*isWrite=*/true,
                            /*isFree=*/false, op.location);
        }
      } else if (op.name == "gpu::upload") {
        if (op.operands.size() >= 2) {
          checkBufferAccess(state, op.operands[0], /*isWrite=*/true,
                            /*isFree=*/false, op.location);
          checkBufferAccess(state, op.operands[1], /*isWrite=*/false,
                            /*isFree=*/false, op.location);
        }
      } else if (op.name == "gpu::download") {
        if (op.operands.size() >= 2) {
          checkBufferAccess(state, op.operands[0], /*isWrite=*/false,
                            /*isFree=*/false, op.location);
          checkBufferAccess(state, op.operands[1], /*isWrite=*/true,
                            /*isFree=*/false, op.location);
        }
      } else if (startsWith(op.name, "gpu::launch::")) {
        for (std::size_t idx = 2; idx < op.operands.size(); ++idx) {
          checkBufferAccess(state, op.operands[idx], /*isWrite=*/true,
                            /*isFree=*/false, op.location);
        }
      }
      return;
    }

    if (calleeKind == "transition") {
      ++report_.checkedTransitions;
      const auto transIt = resources_.transitions.find(op.name);
      if (transIt != resources_.transitions.end()) {
        const auto &trans = transIt->second;
        for (std::size_t idx = 0; idx < op.operands.size(); ++idx) {
          const bool consumes = idx < trans.ownedRuntimeParams.size()
                                    ? trans.ownedRuntimeParams[idx]
                                    : (idx == 0);
          if (consumes) {
            consumeHandle(state, op.operands[idx], op.location);
          } else {
            checkBufferAccess(state, op.operands[idx], /*isWrite=*/false,
                              /*isFree=*/false, op.location);
          }
        }
      } else if (!op.operands.empty()) {
        consumeHandle(state, op.operands[0], op.location);
      }
      registerValueIfTracked(state, op.id, "", resultType, false, op.location);
      return;
    }

    if (const auto procIt = procedures_.find(op.name);
        procIt != procedures_.end()) {
      const auto *callee = procIt->second;
      std::size_t runtimeIdx = 0;
      for (const auto &param : callee->parameters()) {
        if (param.context) continue;
        if (runtimeIdx >= op.operands.size()) break;
        const auto operandId = op.operands[runtimeIdx++];
        if (isOwnedTypeText(param.type) || lookupResourceState(param.type)) {
          consumeHandle(state, operandId, op.location);
        } else {
          checkBufferAccess(state, operandId, /*isWrite=*/false,
                            /*isFree=*/false, op.location);
        }
      }
    }
    registerValueIfTracked(state, op.id, "", resultType, false, op.location);
  }

  PathState mergePaths(const std::vector<PathState> &continuing,
                       const PathState &preBranch,
                       SourceLocation joinLocation) {
    if (continuing.empty()) {
      PathState dead = preBranch;
      dead.reachable = false;
      return dead;
    }
    if (continuing.size() == 1) return continuing.front();

    PathState merged = continuing.front();

    // 1. Merge handle typestates across all continuing branches.
    std::set<graphir::CallableValueId> allHandles;
    for (const auto &path : continuing) {
      for (const auto &[id, _] : path.handles) allHandles.insert(id);
    }
    for (const auto id : allHandles) {
      bool anyLive = false;
      bool anyConsumed = false;
      HandleTypestate sample;
      bool haveSample = false;
      std::set<std::string> liveStateNames;

      for (const auto &path : continuing) {
        const auto it = path.handles.find(id);
        if (it == path.handles.end()) continue;
        if (!haveSample) {
          sample = it->second;
          haveSample = true;
        }
        if (it->second.state == HandleStateKind::Live) {
          anyLive = true;
          liveStateNames.insert(it->second.stateName);
        } else {
          anyConsumed = true;
        }
      }
      if (!haveSample) continue;

      // Branch-local handles allocated in only a subset of branches.
      if (!preBranch.handles.contains(id)) {
        bool missingOnSomeBranch = false;
        for (const auto &path : continuing) {
          if (!path.handles.contains(id)) missingOnSomeBranch = true;
        }
        if (missingOnSomeBranch && anyLive &&
            sample.lifecycle != ResourceLifecycleKind::Droppable) {
          recordViolation(
              report_, "gpu.ownership.branch_lifetime_leak", callableName_,
              sample.label,
              "owned handle `" + sample.label +
                  "` escapes a branch without completion or consumption",
              joinLocation);
        }
        continue;
      }

      if (anyLive && anyConsumed) {
        if (sample.lifecycle == ResourceLifecycleKind::MustComplete) {
          recordViolation(
              report_, "gpu.ownership.branch_lifetime_leak", callableName_,
              sample.label,
              "must-complete resource `" + sample.label +
                  "` is consumed on one branch and left unawaited on another",
              joinLocation);
        } else if (sample.lifecycle == ResourceLifecycleKind::LinearOwned) {
          recordViolation(
              report_, "gpu.ownership.branch_lifetime_leak", callableName_,
              sample.label,
              "owned handle `" + sample.label +
                  "` is consumed on one branch and leaked on another",
              joinLocation);
        }
        sample.state = HandleStateKind::MaybeConsumed;
      } else if (anyLive) {
        if (liveStateNames.size() > 1) {
          recordViolation(
              report_, "gpu.ownership.branch_typestate_mismatch", callableName_,
              sample.label,
              "resource `" + sample.label +
                  "` reaches branch join with conflicting typestates",
              joinLocation);
        }
        sample.state = HandleStateKind::Live;
      } else {
        sample.state = HandleStateKind::Consumed;
      }
      merged.handles[id] = std::move(sample);
    }

    // 2. Merge async generation statuses and active loans (may-active union).
    std::set<SlotGenerationKey> allGenKeys;
    for (const auto &path : continuing) {
      for (const auto &[k, _] : path.generations) allGenKeys.insert(k);
    }
    merged.generations.clear();
    for (const auto &k : allGenKeys) {
      bool anyInFlight = false;
      bool anyCompleted = false;
      bool missingOnSome = false;
      for (const auto &path : continuing) {
        const auto it = path.generations.find(k);
        if (it == path.generations.end()) {
          missingOnSome = true;
        } else if (it->second == SlotGenerationStatus::InFlight) {
          anyInFlight = true;
        } else {
          anyCompleted = true;
        }
      }
      if (anyInFlight && (anyCompleted || missingOnSome)) {
        recordViolation(
            report_, "gpu.loan.asymmetric_branch_wait", callableName_,
            k.first + "@" + k.second,
            "async token (" + k.first + ", gen=" + k.second +
                ") is awaited on only a subset of continuing branches",
            joinLocation);
        merged.generations[k] = SlotGenerationStatus::InFlight;
      } else if (anyInFlight) {
        merged.generations[k] = SlotGenerationStatus::InFlight;
      } else if (anyCompleted) {
        merged.generations[k] = SlotGenerationStatus::Completed;
      }
    }

    merged.activeLoans.clear();
    std::map<AsyncLoanKey, ActiveAsyncLoan> loanUnion;
    for (const auto &path : continuing) {
      for (const auto &loan : path.activeLoans) {
        const auto it = loanUnion.find(loan.key);
        if (it == loanUnion.end() ||
            loan.mode == AsyncLoanMode::ExclusiveWrite) {
          loanUnion[loan.key] = loan;
        }
      }
    }
    for (auto &[_, loan] : loanUnion) {
      merged.activeLoans.push_back(std::move(loan));
    }

    return merged;
  }

  void checkLoopBackedge(const PathState &entryState,
                         const PathState &backedgeState,
                         SourceLocation loopLocation) {
    // 1. Pre-loop live handles must retain identical live state across the
    // loop backedge (cannot consume an outer owned handle inside a loop).
    for (const auto &[id, preHandle] : entryState.handles) {
      if (preHandle.state != HandleStateKind::Live) continue;
      const auto postIt = backedgeState.handles.find(id);
      if (postIt == backedgeState.handles.end() ||
          postIt->second.state != HandleStateKind::Live ||
          postIt->second.stateName != preHandle.stateName) {
        recordViolation(
            report_, "gpu.ownership.loop_backedge_consumption", callableName_,
            preHandle.label,
            "loop body consumes or alters outer owned handle `" +
                preHandle.label + "` across the loop backedge",
            loopLocation);
      }
    }

    // 2. Any non-droppable handle minted inside the loop iteration must be
    // consumed/completed before the backedge repeats.
    for (const auto &[id, postHandle] : backedgeState.handles) {
      if (entryState.handles.contains(id)) continue;
      if (postHandle.state == HandleStateKind::Live &&
          postHandle.lifecycle != ResourceLifecycleKind::Droppable) {
        const char *code =
            postHandle.lifecycle == ResourceLifecycleKind::MustComplete
                ? "gpu.ownership.unawaited_must_complete"
                : "gpu.ownership.loop_backedge_consumption";
        recordViolation(
            report_, code, callableName_, postHandle.label,
            "loop iteration leaves owned handle `" + postHandle.label +
                "` unconsumed across the loop backedge",
            loopLocation);
      }
    }

    // 3. No newly issued async loan may remain in flight across the backedge,
    // and pre-existing in-flight loans must remain invariant.
    for (const auto &[sgKey, postStatus] : backedgeState.generations) {
      const auto preIt = entryState.generations.find(sgKey);
      if (preIt == entryState.generations.end()) {
        if (postStatus == SlotGenerationStatus::InFlight) {
          recordViolation(
              report_, "gpu.loan.loop_backedge_leak", callableName_,
              sgKey.first + "@" + sgKey.second,
              "async token (" + sgKey.first + ", gen=" + sgKey.second +
                  ") remains unawaited across loop backedge",
              loopLocation);
        }
      } else if (preIt->second != postStatus) {
        recordViolation(
            report_, "gpu.loan.loop_backedge_leak", callableName_,
            sgKey.first + "@" + sgKey.second,
            "loop body alters outer async token (" + sgKey.first +
                ", gen=" + sgKey.second + ") state across loop backedge",
            loopLocation);
      }
    }
  }

  void checkExitObligations(
      const PathState &state,
      const std::vector<graphir::CallableValueId> &returnedOperands,
      SourceLocation exitLocation) {
    std::set<graphir::CallableValueId> returnedRoots;
    for (const auto opId : returnedOperands) {
      returnedRoots.insert(resolveRoot(opId));
    }

    for (const auto &[sgKey, status] : state.generations) {
      if (status == SlotGenerationStatus::InFlight) {
        recordViolation(
            report_, "gpu.ownership.unawaited_must_complete", callableName_,
            sgKey.first + "@" + sgKey.second,
            "callable exits while async token (" + sgKey.first +
                ", gen=" + sgKey.second + ") is still in flight",
            exitLocation);
      }
    }

    for (const auto &[id, handle] : state.handles) {
      if (returnedRoots.contains(id)) continue;
      if (handle.state != HandleStateKind::Live) continue;
      if (handle.lifecycle == ResourceLifecycleKind::MustComplete) {
        recordViolation(report_, "gpu.ownership.unawaited_must_complete",
                        callableName_, handle.label,
                        "must-complete resource `" + handle.label + "` (" +
                            handle.typeText +
                            ") is unawaited at callable exit",
                        exitLocation);
      } else if (handle.lifecycle == ResourceLifecycleKind::LinearOwned) {
        recordViolation(report_, "gpu.ownership.unconsumed_owned_value",
                        callableName_, handle.label,
                        "owned handle `" + handle.label + "` (" +
                            handle.typeText +
                            ") is neither consumed nor returned before exit",
                        exitLocation);
      }
    }
  }

  unsigned analyzeSequence(const std::vector<graphir::CallableOperation> &ops,
                           const graphir::ContextMetadataRef &inherited,
                           PathState &state,
                           std::vector<LoopExitStates> &loopStack) {
    unsigned combinedFlow = static_cast<unsigned>(FlowMask::Continue);
    for (const auto &op : ops) {
      if (!state.reachable) break;
      const unsigned opFlow = analyzeOperation(op, inherited, state, loopStack);
      combinedFlow = (combinedFlow & ~static_cast<unsigned>(FlowMask::Continue)) |
                     opFlow;
      if ((opFlow & static_cast<unsigned>(FlowMask::Continue)) == 0) {
        state.reachable = false;
        break;
      }
    }
    return combinedFlow;
  }

  unsigned analyzeOperation(const graphir::CallableOperation &op,
                            const graphir::ContextMetadataRef &inherited,
                            PathState &state,
                            std::vector<LoopExitStates> &loopStack) {
    using Kind = graphir::CallableOperationKind;
    const auto effective = op.context ? op.context : inherited;

    switch (op.kind) {
    case Kind::ContextRegion:
      verifyEffectiveContext(callable_->context(), op.context, "context_region",
                             op.location);
      return analyzeSequence(op.body, effective, state, loopStack);

    case Kind::Block:
      return analyzeSequence(op.body, inherited, state, loopStack);

    case Kind::Literal:
      if (const auto it = op.attributes.find("type"); it != op.attributes.end())
        valueTypes_[op.id] = it->second;
      return static_cast<unsigned>(FlowMask::Continue);

    case Kind::Variable: {
      const std::string varType =
          op.attributes.contains("type") ? op.attributes.at("type") : "";
      valueTypes_[op.id] = varType;
      if (!op.name.empty() && !allocationNames_.contains(op.id)) {
        allocationNames_[op.id] = op.name;
      }
      if (!op.operands.empty()) {
        const auto initRoot = resolveRoot(op.operands[0]);
        if (isOwnedTypeText(varType) || lookupResourceState(varType)) {
          consumeHandle(state, initRoot, op.location);
          registerValueIfTracked(state, op.id, op.name, varType, false,
                                 op.location);
        } else {
          aliasRoot_[op.id] = initRoot;
        }
      } else {
        registerValueIfTracked(state, op.id, op.name, varType, false,
                               op.location);
      }
      return static_cast<unsigned>(FlowMask::Continue);
    }

    case Kind::Load:
      if (!op.operands.empty()) {
        requireLiveHandle(state, op.operands[0], "read", op.location);
        aliasRoot_[op.id] = resolveRoot(op.operands[0]);
      }
      if (const auto it = op.attributes.find("type"); it != op.attributes.end())
        valueTypes_[op.id] = it->second;
      return static_cast<unsigned>(FlowMask::Continue);

    case Kind::Assign:
      if (op.operands.size() >= 2) {
        checkBufferAccess(state, op.operands[1], /*isWrite=*/false,
                          /*isFree=*/false, op.location);
        if (state.handles.contains(resolveRoot(op.operands[1])) &&
            !state.handles.at(resolveRoot(op.operands[1])).isDynamicSpan) {
          consumeHandle(state, op.operands[1], op.location);
        }
      }
      return static_cast<unsigned>(FlowMask::Continue);

    case Kind::Binary:
    case Kind::Unary:
      if (const auto it = op.attributes.find("type"); it != op.attributes.end())
        valueTypes_[op.id] = it->second;
      return static_cast<unsigned>(FlowMask::Continue);

    case Kind::Call:
      handleCall(op, effective, state);
      return static_cast<unsigned>(FlowMask::Continue);

    case Kind::AsyncIssue:
      handleAsyncIssue(op, effective, state);
      return static_cast<unsigned>(FlowMask::Continue);

    case Kind::AsyncWait:
      handleAsyncWait(op, effective, state);
      return static_cast<unsigned>(FlowMask::Continue);

    case Kind::ChannelSend:
      verifyEffectiveContext(callable_->context(), effective, op.name,
                             op.location);
      for (const auto operand : op.operands) {
        checkBufferAccess(state, operand, /*isWrite=*/false, /*isFree=*/false,
                          op.location);
        if (op.attributes.contains("transfers_ownership") &&
            op.attributes.at("transfers_ownership") == "true") {
          consumeHandle(state, operand, op.location);
        }
      }
      return static_cast<unsigned>(FlowMask::Continue);

    case Kind::ChannelReceive:
      verifyEffectiveContext(callable_->context(), effective, op.name,
                             op.location);
      if (const auto it = op.attributes.find("type");
          it != op.attributes.end()) {
        valueTypes_[op.id] = it->second;
        registerValueIfTracked(state, op.id, "", it->second, false,
                               op.location);
      }
      return static_cast<unsigned>(FlowMask::Continue);

    case Kind::ProgressMarker:
      verifyEffectiveContext(callable_->context(), effective, op.name,
                             op.location);
      return static_cast<unsigned>(FlowMask::Continue);

    case Kind::If:
    case Kind::FairChoice: {
      const unsigned condFlow =
          analyzeSequence(op.condition, inherited, state, loopStack);
      if ((condFlow & static_cast<unsigned>(FlowMask::Continue)) == 0)
        return condFlow;

      const PathState preBranch = state;
      PathState thenState = preBranch;
      PathState elseState = preBranch;
      const unsigned thenFlow =
          analyzeSequence(op.body, inherited, thenState, loopStack);
      const unsigned elseFlow =
          analyzeSequence(op.alternative, inherited, elseState, loopStack);

      std::vector<PathState> continuing;
      if ((thenFlow & static_cast<unsigned>(FlowMask::Continue)) != 0 &&
          thenState.reachable) {
        continuing.push_back(std::move(thenState));
      }
      if ((elseFlow & static_cast<unsigned>(FlowMask::Continue)) != 0 &&
          elseState.reachable) {
        continuing.push_back(std::move(elseState));
      }

      state = mergePaths(continuing, preBranch, op.location);
      const unsigned exitFlags =
          (thenFlow | elseFlow) & ~static_cast<unsigned>(FlowMask::Continue);
      return exitFlags |
             (continuing.empty() ? 0u
                                 : static_cast<unsigned>(FlowMask::Continue));
    }

    case Kind::Offer: {
      verifyEffectiveContext(callable_->context(), effective, "offer",
                             op.location);
      const unsigned initFlow =
          analyzeSequence(op.initialization, inherited, state, loopStack);
      if ((initFlow & static_cast<unsigned>(FlowMask::Continue)) == 0)
        return initFlow;

      const PathState preOffer = state;
      std::vector<PathState> continuing;
      unsigned exitFlags = 0;

      for (const auto &branch : op.offerBranches) {
        PathState branchState = preOffer;
        const unsigned branchFlow =
            analyzeSequence(branch.body, inherited, branchState, loopStack);
        exitFlags |=
            branchFlow & ~static_cast<unsigned>(FlowMask::Continue);
        if ((branchFlow & static_cast<unsigned>(FlowMask::Continue)) != 0 &&
            branchState.reachable) {
          continuing.push_back(std::move(branchState));
        }
      }
      if (!op.alternative.empty()) {
        PathState defState = preOffer;
        const unsigned defFlow =
            analyzeSequence(op.alternative, inherited, defState, loopStack);
        exitFlags |= defFlow & ~static_cast<unsigned>(FlowMask::Continue);
        if ((defFlow & static_cast<unsigned>(FlowMask::Continue)) != 0 &&
            defState.reachable) {
          continuing.push_back(std::move(defState));
        }
      } else if (op.offerBranches.empty()) {
        continuing.push_back(preOffer);
      }

      state = mergePaths(continuing, preOffer, op.location);
      return exitFlags |
             (continuing.empty() ? 0u
                                 : static_cast<unsigned>(FlowMask::Continue));
    }

    case Kind::For:
    case Kind::While:
    case Kind::Until:
    case Kind::Loop: {
      verifyLoopEvidence(op.loopContext, op.location);
      const unsigned initFlow =
          analyzeSequence(op.initialization, inherited, state, loopStack);
      if ((initFlow & static_cast<unsigned>(FlowMask::Continue)) == 0)
        return initFlow;

      const unsigned condFlow =
          analyzeSequence(op.condition, inherited, state, loopStack);
      if ((condFlow & static_cast<unsigned>(FlowMask::Continue)) == 0)
        return condFlow;

      const PathState headerState = state;
      PathState bodyState = headerState;
      loopStack.emplace_back();

      const unsigned bodyFlow =
          analyzeSequence(op.body, inherited, bodyState, loopStack);
      LoopExitStates loopExits = std::move(loopStack.back());
      loopStack.pop_back();

      std::vector<PathState> stepInputs;
      if ((bodyFlow & static_cast<unsigned>(FlowMask::Continue)) != 0 &&
          bodyState.reachable) {
        stepInputs.push_back(std::move(bodyState));
      }
      for (auto &contState : loopExits.continues) {
        if (contState.reachable) stepInputs.push_back(std::move(contState));
      }

      if (!stepInputs.empty()) {
        PathState latchState = mergePaths(stepInputs, headerState, op.location);
        const unsigned stepFlow =
            analyzeSequence(op.step, inherited, latchState, loopStack);
        if ((stepFlow & static_cast<unsigned>(FlowMask::Continue)) != 0 &&
            latchState.reachable) {
          checkLoopBackedge(headerState, latchState, op.location);
        }
      }

      std::vector<PathState> postLoopPaths;
      if (op.kind != Kind::Loop) {
        // Counted/conditional loops may exit from the header (0 or N trips).
        postLoopPaths.push_back(headerState);
      }
      for (auto &brkState : loopExits.breaks) {
        if (brkState.reachable) postLoopPaths.push_back(std::move(brkState));
      }

      const unsigned returnFlag =
          bodyFlow & static_cast<unsigned>(FlowMask::Return);
      if (postLoopPaths.empty()) {
        state.reachable = false;
        return returnFlag;
      }
      state = mergePaths(postLoopPaths, headerState, op.location);
      return returnFlag | static_cast<unsigned>(FlowMask::Continue);
    }

    case Kind::Return:
      for (const auto retOp : op.operands) {
        checkBufferAccess(state, retOp, /*isWrite=*/false, /*isFree=*/false,
                          op.location);
        consumeHandle(state, retOp, op.location);
      }
      checkExitObligations(state, op.operands, op.location);
      state.reachable = false;
      return static_cast<unsigned>(FlowMask::Return);

    case Kind::Break:
      if (!loopStack.empty()) loopStack.back().breaks.push_back(state);
      state.reachable = false;
      return static_cast<unsigned>(FlowMask::Break);

    case Kind::Continue:
      if (!loopStack.empty()) loopStack.back().continues.push_back(state);
      state.reachable = false;
      return static_cast<unsigned>(FlowMask::LoopContinue);

    default:
      return static_cast<unsigned>(FlowMask::Continue);
    }
  }
};

} // namespace

std::optional<GPUExecutionScope>
classifyContextExecutionScope(std::string_view contextTypeName) noexcept {
  const auto base = stripGenerics(trimSpaces(contextTypeName));
  if (startsWith(base, "Host::")) {
    if (base == "Host::Loop") return std::nullopt;
    return GPUExecutionScope::Host;
  }
  if (!startsWith(base, "Gpu::") || base == "Gpu::Loop") return std::nullopt;
  if (base == "Gpu::Grid") return GPUExecutionScope::DeviceGrid;
  if (base == "Gpu::Block" || base == "Gpu::Workgroup")
    return GPUExecutionScope::Workgroup;
  if (base == "Gpu::Warp" || base == "Gpu::Subgroup" ||
      base == "Gpu::WarpStage")
    return GPUExecutionScope::Subgroup;
  if (base == "Gpu::Thread" || base == "Gpu::PersistentWorker")
    return GPUExecutionScope::Thread;
  return std::nullopt;
}

std::optional<GPULoopDomain>
classifyLoopContextDomain(std::string_view contextTypeName) noexcept {
  const auto base = stripGenerics(trimSpaces(contextTypeName));
  if (base == "Host::Loop") return GPULoopDomain::HostLoop;
  if (base == "Gpu::Loop") return GPULoopDomain::DeviceLoop;
  return std::nullopt;
}

std::optional<GPUAddressSpace>
classifyTypeAddressSpace(std::string_view typeText) noexcept {
  const auto stripped = stripOwnership(typeText);
  if (startsWith(stripped, "gpu::span")) return GPUAddressSpace::GlobalDevice;
  if (startsWith(stripped, "host::span")) return GPUAddressSpace::HostDefault;
  const auto lt = stripped.find('<');
  const auto gt = stripped.rfind('>');
  if (lt == std::string_view::npos || gt == std::string_view::npos || gt <= lt)
    return std::nullopt;
  const auto base = trimSpaces(stripped.substr(0, lt));
  if (base != "buffer" && base != "view" && base != "span") return std::nullopt;
  const auto inner = stripped.substr(lt + 1, gt - lt - 1);
  const auto comma = inner.rfind(',');
  const auto spaceText =
      comma == std::string_view::npos ? inner : inner.substr(comma + 1);
  return parseAddressSpaceToken(spaceText);
}

bool isCapabilityAllowedInScope(GPUExecutionScope scope,
                                std::string_view capabilityName) noexcept {
  const auto base = stripGenerics(trimSpaces(capabilityName));
  if (scope == GPUExecutionScope::Host) {
    return base != "gpu_load" && base != "gpu_store" &&
           base != "gpu_barrier" && base != "gpu_block_barrier" &&
           base != "gpu_warp_barrier" && base != "gpu_warp_shuffle" &&
           base != "gpu_warp_vote" && base != "gpu_grid_sync" &&
           base != "gpu_shared_atomic" && base != "gpu_persistent_work";
  }

  // Device scopes never permit host memory/IO or host-side driver allocation/launch.
  if (startsWith(base, "host_") || base == "allocate" || base == "sleep" ||
      base == "block" || base == "mmio" || base == "irq_ack" ||
      base == "defer" || base == "gpu_alloc" || base == "gpu_free" ||
      base == "gpu_copy" || base == "gpu_launch") {
    return false;
  }

  switch (scope) {
  case GPUExecutionScope::DeviceGrid:
    return true;
  case GPUExecutionScope::Workgroup:
    return base != "gpu_grid_sync";
  case GPUExecutionScope::Subgroup:
    return base != "gpu_grid_sync" && base != "gpu_barrier" &&
           base != "gpu_block_barrier";
  case GPUExecutionScope::Thread:
    return base != "gpu_grid_sync" && base != "gpu_barrier" &&
           base != "gpu_block_barrier" && base != "gpu_warp_barrier" &&
           base != "gpu_warp_shuffle" && base != "gpu_warp_vote";
  case GPUExecutionScope::Host:
    break;
  }
  return false;
}

bool isIntrinsicAllowedInScope(GPUExecutionScope scope,
                               std::string_view intrinsicName) noexcept {
  const auto trimmed = trimSpaces(intrinsicName);
  if (startsWith(trimmed, "host::") || trimmed == "gpu::alloc" ||
      trimmed == "gpu::free" || trimmed == "gpu::upload" ||
      trimmed == "gpu::download" || trimmed == "gpu::wait" ||
      startsWith(trimmed, "gpu::launch")) {
    return scope == GPUExecutionScope::Host;
  }
  if (scope == GPUExecutionScope::Host) return false;

  if (trimmed == "gpu::grid_barrier" || trimmed == "grid_barrier") {
    return scope == GPUExecutionScope::DeviceGrid;
  }
  if (trimmed == "gpu::barrier" || trimmed == "barrier" ||
      trimmed == "gpu::block_barrier") {
    return scope == GPUExecutionScope::Workgroup;
  }
  if (trimmed == "gpu::warp_barrier" || trimmed == "warp_barrier" ||
      trimmed == "gpu::warp_shuffle" || trimmed == "warp_shuffle" ||
      trimmed == "gpu::warp_vote" || trimmed == "warp_vote") {
    return scope == GPUExecutionScope::Subgroup;
  }
  return true;
}

GPUOwnershipReport verifyGPUOwnershipAndLoans(const graphir::Module &module) {
  GPUOwnershipReport report;

  std::map<std::size_t, std::set<std::string>> contextGrants;
  for (const auto &node : module.nodes()) {
    if (node.kind != graphir::NodeKind::Grant || !node.context) continue;
    try {
      const auto ordinal = module.contextOrdinal(node.context);
      const std::string cap = node.attributes.contains("capability")
                                  ? node.attributes.at("capability")
                                  : std::string(stripGenerics(node.name));
      contextGrants[ordinal].insert(cap);

      if (const auto scope = classifyContextExecutionScope(node.context->name())) {
        if (!isCapabilityAllowedInScope(*scope, cap)) {
          recordViolation(
              report, "gpu.context.scope_violation", "<contexts>", node.name,
              "capability `" + node.name + "` is forbidden on `" +
                  node.context->name() + "` (" +
                  gpuExecutionScopeName(*scope) + " scope)",
              node.location);
        }
      } else if (classifyLoopContextDomain(node.context->name())) {
        recordViolation(report, "gpu.context.grant_bearing_loop_evidence",
                        "<contexts>", node.context->name(),
                        "loop context cannot grant `" + node.name + "`",
                        node.location);
      }
    } catch (const std::logic_error &error) {
      recordViolation(report, "gpu.context.foreign_context", "<contexts>",
                      node.name, error.what(), node.location);
    }
  }

  for (const auto &ctx : module.contexts()) {
    if (!ctx) continue;
    const auto base = stripGenerics(ctx->name());
    if (!startsWith(base, "Gpu::") && !startsWith(base, "Host::")) continue;
    ++report.checkedContexts;
    if (const auto loopDomain = classifyLoopContextDomain(ctx->name())) {
      (void)loopDomain;
      if (ctx->genericArity() != 0) {
        recordViolation(report, "gpu.context.generic_loop_evidence",
                        "<contexts>", ctx->name(),
                        "loop context cannot declare generic parameters",
                        ctx->location());
      }
    } else if (!classifyContextExecutionScope(ctx->name())) {
      recordViolation(
          report, "gpu.context.unknown_gpu_scope", "<contexts>", ctx->name(),
          "unrecognized GPU/Host execution scope `" + ctx->name() +
              "` (expected Host::* or Gpu::{Grid, Block, Warp, Thread})",
          ctx->location());
    }
  }

  const auto resources = buildResourceIndex(module);
  std::map<std::string, const graphir::CallableGraph *> procedures;
  for (const auto &proc : module.procedureGraphs()) {
    procedures.emplace(proc.name(), &proc);
  }

  CallableAnalyzer analyzer(module, resources, procedures, contextGrants,
                            report);
  for (const auto &kernel : module.kernelGraphs()) analyzer.analyze(kernel);
  for (const auto &proc : module.procedureGraphs()) {
    const auto domIt = proc.attributes().find("execution_domain");
    const std::string dom =
        domIt != proc.attributes().end() ? domIt->second : "";
    if (dom == "Gpu" || dom == "Host" || dom == "context_independent") {
      analyzer.analyze(proc);
    }
  }
  for (const auto &stage : module.stageGraphs()) {
    const auto domIt = stage.attributes().find("execution_domain");
    const std::string dom =
        domIt != stage.attributes().end() ? domIt->second : "";
    if (dom == "Gpu" || dom == "Host" || dom == "context_independent") {
      analyzer.analyze(stage);
    }
  }

  return report;
}

} // namespace metreon::verification
