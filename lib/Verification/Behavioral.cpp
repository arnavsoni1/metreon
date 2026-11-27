#include "metreon/Verification/Behavioral.h"

#include "metreon/GraphIR/ControlFlow.h"

#include <algorithm>
#include <charconv>
#include <deque>
#include <functional>
#include <sstream>
#include <utility>

namespace metreon::verification {
namespace {

std::optional<std::size_t> parseUnsignedText(std::string_view text) noexcept {
  if (text.empty()) return std::nullopt;
  std::size_t val = 0;
  const auto *begin = text.data();
  const auto *end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, val);
  if (ec != std::errc{} || ptr != end) return std::nullopt;
  return val;
}

bool startsWithPrefix(std::string_view s, std::string_view p) noexcept {
  return s.size() >= p.size() && s.substr(0, p.size()) == p;
}

std::optional<GPUPublicationOrder>
parsePublicationOrder(std::string_view text) noexcept {
  if (text == "relaxed") return GPUPublicationOrder::Relaxed;
  if (text == "acquire_workgroup") return GPUPublicationOrder::AcquireWorkgroup;
  if (text == "release_workgroup") return GPUPublicationOrder::ReleaseWorkgroup;
  if (text == "acq_rel_workgroup") return GPUPublicationOrder::AcqRelWorkgroup;
  if (text == "acquire_device") return GPUPublicationOrder::AcquireDevice;
  if (text == "release_device") return GPUPublicationOrder::ReleaseDevice;
  if (text == "acq_rel_device") return GPUPublicationOrder::AcqRelDevice;
  return std::nullopt;
}

bool isBidirectionalAcqRelOrder(GPUPublicationOrder order) noexcept {
  return order == GPUPublicationOrder::AcqRelWorkgroup ||
         order == GPUPublicationOrder::AcqRelDevice;
}

void recordQueueViolation(GPUBoundedQueueReport &report,
                          VerificationStatus failureStatus, std::string code,
                          std::string pipelineName, std::string subject,
                          std::string message, SourceLocation loc,
                          std::vector<std::string> trace = {}) {
  // Precedence: Rejected overrides UnboundedProved/StaticChecked/Unsupported/Unknown;
  // Unknown/Unsupported override UnboundedProved/StaticChecked.
  if (failureStatus == VerificationStatus::Rejected) {
    report.status = VerificationStatus::Rejected;
  } else if (report.status != VerificationStatus::Rejected) {
    report.status = failureStatus;
  }
  report.violations.push_back(GPUBoundedQueueViolation{
      std::move(code), std::move(pipelineName), std::move(subject),
      std::move(message), std::move(trace), loc});
}

std::string resolveEndpointName(
    const std::string &rawEndpoint,
    const std::map<std::string, std::string> &portToChannelMap) {
  auto it = portToChannelMap.find(rawEndpoint);
  return it != portToChannelMap.end() ? it->second : rawEndpoint;
}

void indexCallableOpsById(
    const std::vector<graphir::CallableOperation> &ops,
    std::map<graphir::CallableValueId, const graphir::CallableOperation *> &out) {
  for (const auto &op : ops) {
    out.emplace(op.id, &op);
    for (const auto *nested :
         {&op.initialization, &op.condition, &op.body, &op.alternative,
          &op.step}) {
      indexCallableOpsById(*nested, out);
    }
    for (const auto &arm : op.offerBranches) {
      out.emplace(arm.id, &arm);
      indexCallableOpsById(arm.body, out);
    }
  }
}

std::string formatCompositeState(const GPUPipelineSystemSpec &system,
                                 const CompositeSystemState &st) {
  std::ostringstream os;
  os << "{stages=[";
  for (std::size_t i = 0; i < st.stageStates.size(); ++i) {
    if (i) os << ", ";
    const auto sid = st.stageStates[i];
    os << system.stages[i].name << "@"
       << (sid < system.stages[i].states.size()
               ? system.stages[i].states[sid].label
               : std::to_string(sid));
  }
  os << "], queues=[";
  for (std::size_t q = 0; q < st.queueStates.size(); ++q) {
    if (q) os << ", ";
    const auto &qs = st.queueStates[q];
    os << system.queues[q].name << "(free=" << qs.freeCredits
       << ",res=" << qs.reservedSlots << ",queued=" << qs.items.size() << "/"
       << qs.capacity << ",proto=" << st.protocolStates[q] << ")";
  }
  os << "]}";
  return os.str();
}

// Tarjan's Strongly Connected Components over directed graph `[0..numNodes-1]`.
struct SCCResult {
  std::vector<std::vector<std::size_t>> components;
};

SCCResult computeTarjanSCCs(
    std::size_t numNodes,
    const std::vector<std::vector<std::size_t>> &adj) {
  SCCResult res;
  std::vector<int> index(numNodes, -1);
  std::vector<int> lowlink(numNodes, -1);
  std::vector<bool> onStack(numNodes, false);
  std::vector<std::size_t> stack;
  int currentIndex = 0;

  std::function<void(std::size_t)> strongConnect = [&](std::size_t v) {
    index[v] = currentIndex;
    lowlink[v] = currentIndex;
    ++currentIndex;
    stack.push_back(v);
    onStack[v] = true;

    for (std::size_t w : adj[v]) {
      if (index[w] == -1) {
        strongConnect(w);
        lowlink[v] = std::min(lowlink[v], lowlink[w]);
      } else if (onStack[w]) {
        lowlink[v] = std::min(lowlink[v], index[w]);
      }
    }

    if (lowlink[v] == index[v]) {
      std::vector<std::size_t> comp;
      while (true) {
        const std::size_t w = stack.back();
        stack.pop_back();
        onStack[w] = false;
        comp.push_back(w);
        if (w == v) break;
      }
      res.components.push_back(std::move(comp));
    }
  };

  for (std::size_t v = 0; v < numNodes; ++v) {
    if (index[v] == -1) strongConnect(v);
  }
  return res;
}

} // namespace

const char *behavioralActionKindName(BehavioralActionKind kind) noexcept {
  switch (kind) {
  case BehavioralActionKind::Send: return "send";
  case BehavioralActionKind::Receive: return "receive";
  case BehavioralActionKind::Internal: return "internal";
  case BehavioralActionKind::ClockTick: return "clock_tick";
  }
  return "internal";
}

bool BehavioralAutomaton::hasCommunicationActions() const noexcept {
  return std::any_of(transitions.begin(), transitions.end(),
                     [](const ProcessTransition &t) {
                       return t.kind == BehavioralActionKind::Send ||
                              t.kind == BehavioralActionKind::Receive;
                     });
}

BehavioralAutomaton extractStageAutomaton(
    const graphir::CallableGraph &callable,
    const std::map<std::string, std::string> &portToChannelMap) {
  BehavioralAutomaton aut;
  aut.name = callable.name();
  aut.location = callable.location();

  std::map<graphir::CallableValueId, const graphir::CallableOperation *> opById;
  indexCallableOpsById(callable.body(), opById);

  const auto cfg = graphir::buildControlFlow(callable);
  std::map<graphir::ControlFlowBlockId, AutomatonStateId> blockEntryState;

  auto addState = [&](std::string label, SourceLocation loc,
                      bool accepting = false) -> AutomatonStateId {
    const AutomatonStateId id = aut.states.size();
    aut.states.push_back(
        AutomatonState{id, std::move(label), accepting, std::nullopt, loc});
    return id;
  };

  for (const auto &block : cfg.blocks) {
    std::string lbl = "bb" + std::to_string(block.id);
    if (const auto rIt = block.attributes.find("role");
        rIt != block.attributes.end()) {
      lbl += ":" + rIt->second;
    }
    blockEntryState[block.id] = addState(std::move(lbl), block.location, false);
  }

  const AutomatonStateId terminalId =
      addState("terminal", callable.location(), /*accepting=*/true);
  aut.initialState = blockEntryState.contains(cfg.entry)
                         ? blockEntryState.at(cfg.entry)
                         : terminalId;

  std::map<graphir::ControlFlowBlockId, const graphir::ControlFlowBlock *>
      cfgBlockMap;
  for (const auto &b : cfg.blocks) cfgBlockMap[b.id] = &b;

  for (const auto &block : cfg.blocks) {
    AutomatonStateId cursor = blockEntryState.at(block.id);
    std::vector<std::string> pendingMarkers;

    for (const auto opId : block.operations) {
      auto opIt = opById.find(opId);
      if (opIt == opById.end()) continue;
      const auto &op = *opIt->second;

      if (op.kind == graphir::CallableOperationKind::ProgressMarker) {
        pendingMarkers.push_back(op.name);
        aut.requiredProgressMarkers.insert(op.name);
        continue;
      }

      if (op.kind == graphir::CallableOperationKind::ChannelSend ||
          op.kind == graphir::CallableOperationKind::ChannelReceive) {
        const AutomatonStateId afterAct =
            addState("post_" + std::to_string(op.id), op.location, false);
        ProcessTransition tr;
        tr.source = cursor;
        tr.target = afterAct;
        tr.kind = op.kind == graphir::CallableOperationKind::ChannelSend
                      ? BehavioralActionKind::Send
                      : BehavioralActionKind::Receive;
        const std::string rawEp = op.attributes.contains("endpoint")
                                      ? op.attributes.at("endpoint")
                                      : "";
        tr.endpoint = resolveEndpointName(rawEp, portToChannelMap);
        tr.message = op.attributes.contains("message")
                         ? op.attributes.at("message")
                         : "";
        tr.delivery = (op.attributes.contains("delivery") &&
                       op.attributes.at("delivery") == "same_step")
                          ? graphir::ChannelDeliveryMode::SameStep
                          : graphir::ChannelDeliveryMode::BlockingQueue;
        tr.transfersOwnership =
            op.attributes.contains("ownership_transfer") &&
            op.attributes.at("ownership_transfer") == "move";
        tr.progressMarkers = std::move(pendingMarkers);
        pendingMarkers.clear();
        tr.location = op.location;
        aut.transitions.push_back(std::move(tr));
        cursor = afterAct;
      }
    }

    const auto &term = block.terminator;
    if (term.kind == graphir::ControlFlowTerminatorKind::Return) {
      ProcessTransition tr;
      tr.source = cursor;
      tr.target = terminalId;
      tr.kind = BehavioralActionKind::Internal;
      tr.progressMarkers = std::move(pendingMarkers);
      tr.location = term.location;
      aut.transitions.push_back(std::move(tr));
      continue;
    }

    if (term.kind == graphir::ControlFlowTerminatorKind::Branch &&
        term.successor && blockEntryState.contains(*term.successor)) {
      ProcessTransition tr;
      tr.source = cursor;
      tr.target = blockEntryState.at(*term.successor);
      tr.kind = BehavioralActionKind::Internal;
      tr.progressMarkers = std::move(pendingMarkers);
      tr.location = term.location;
      aut.transitions.push_back(std::move(tr));
      continue;
    }

    if (term.kind == graphir::ControlFlowTerminatorKind::ConditionalBranch &&
        term.successor && term.falseSuccessor &&
        blockEntryState.contains(*term.successor) &&
        blockEntryState.contains(*term.falseSuccessor)) {
      // Detect `Offer` dispatch chain (`term.control == "offer:exclusive"` or
      // `"offer:conjunctive"`) when we are at the head of an Offer op!
      const bool isOfferDispatch = startsWithPrefix(term.control, "offer:");
      const bool isOfferContinuation =
          block.attributes.contains("role") &&
          block.attributes.at("role") == "offer.next";

      if (isOfferDispatch && !isOfferContinuation && term.operation &&
          opById.contains(*term.operation)) {
        const auto &offerOp = *opById.at(*term.operation);
        AutomatonOfferGroup group;
        group.mode = (offerOp.attributes.contains("offer_mode") &&
                      offerOp.attributes.at("offer_mode") == "conjunctive")
                         ? graphir::OfferMode::Conjunctive
                         : graphir::OfferMode::Exclusive;
        group.location = offerOp.location;

        // Walk the linear chain of `offer.next` dispatch blocks in `cfg` to
        // attach every arm directly to `cursor` and mark the final
        // `offer.default` block as `isOfferDefault = true`!
        graphir::ControlFlowBlockId curDispatch = block.id;
        std::size_t armIdx = 0;
        while (cfgBlockMap.contains(curDispatch)) {
          const auto &dBlk = *cfgBlockMap.at(curDispatch);
          const auto &dTerm = dBlk.terminator;
          if (dTerm.kind !=
                  graphir::ControlFlowTerminatorKind::ConditionalBranch ||
              !dTerm.successor || !dTerm.falseSuccessor) {
            break;
          }
          const AutomatonStateId armTarget =
              blockEntryState.at(*dTerm.successor);
          if (armIdx < offerOp.offerBranches.size()) {
            const auto &armAst = offerOp.offerBranches[armIdx];
            const BehavioralActionKind armKind =
                (armAst.attributes.contains("direction") &&
                 armAst.attributes.at("direction") == "send")
                    ? BehavioralActionKind::Send
                    : BehavioralActionKind::Receive;
            const std::string armEp = resolveEndpointName(
                armAst.attributes.contains("endpoint")
                    ? armAst.attributes.at("endpoint")
                    : "",
                portToChannelMap);
            const std::string armMsg = armAst.attributes.contains("message")
                                           ? armAst.attributes.at("message")
                                           : "";
            group.arms.push_back(
                OfferArmSpec{armKind, armEp, armMsg, armTarget, armAst.location});

            // Notice: if `armBlock` itself begins with the matching
            // `ChannelReceive`/`ChannelSend` op, jumping to `armTarget` executes
            // that channel action only when available; if `armBlock` contains
            // only continuation statements, emit the guarded channel transition
            // directly from `cursor` to `armTarget`!
            bool armHasExplicitChannelOp = false;
            if (cfgBlockMap.contains(*dTerm.successor)) {
              for (auto bOpId : cfgBlockMap.at(*dTerm.successor)->operations) {
                if (opById.contains(bOpId)) {
                  const auto k = opById.at(bOpId)->kind;
                  if (k == graphir::CallableOperationKind::ChannelSend ||
                      k == graphir::CallableOperationKind::ChannelReceive) {
                    armHasExplicitChannelOp = true;
                    break;
                  }
                }
              }
            }
            ProcessTransition armTr;
            armTr.source = cursor;
            armTr.target = armTarget;
            armTr.kind =
                armHasExplicitChannelOp ? BehavioralActionKind::Internal : armKind;
            armTr.endpoint = armEp;
            armTr.message = armMsg;
            armTr.progressMarkers = pendingMarkers;
            armTr.location = armAst.location;
            aut.transitions.push_back(std::move(armTr));
          }

          const auto nextFallbackId = *dTerm.falseSuccessor;
          if (!cfgBlockMap.contains(nextFallbackId)) break;
          const auto &fbBlk = *cfgBlockMap.at(nextFallbackId);
          const auto rIt = fbBlk.attributes.find("role");
          if (rIt != fbBlk.attributes.end() && rIt->second == "offer.next") {
            curDispatch = nextFallbackId;
            ++armIdx;
            continue;
          }
          // Reached `offer.default`!
          const AutomatonStateId defTarget = blockEntryState.at(nextFallbackId);
          group.defaultTarget = defTarget;
          ProcessTransition defTr;
          defTr.source = cursor;
          defTr.target = defTarget;
          defTr.kind = BehavioralActionKind::Internal;
          defTr.isOfferDefault = true;
          defTr.progressMarkers = pendingMarkers;
          defTr.location = offerOp.location;
          aut.transitions.push_back(std::move(defTr));
          break;
        }
        aut.states[cursor].offer = std::move(group);
        pendingMarkers.clear();
        continue;
      }

      if (isOfferContinuation) {
        // Already folded into the head `Offer` state's `AutomatonOfferGroup`.
        continue;
      }

      std::optional<std::string> fairTag;
      bool justified = false;
      if (startsWithPrefix(term.control, "fair:")) {
        fairTag = term.control.substr(5);
        if (cfgBlockMap.contains(*term.successor)) {
          const auto &yesAttrs = cfgBlockMap.at(*term.successor)->attributes;
          auto jIt = yesAttrs.find("assumption_justified");
          justified = (jIt != yesAttrs.end() && jIt->second == "true");
        }
        if (!justified) aut.unprovedFairnessAssumptions.insert(*fairTag);
      }

      ProcessTransition tTrue;
      tTrue.source = cursor;
      tTrue.target = blockEntryState.at(*term.successor);
      tTrue.kind = BehavioralActionKind::Internal;
      tTrue.fairTag = fairTag;
      tTrue.fairBranchOutcome = true;
      tTrue.assumptionJustified = justified;
      tTrue.progressMarkers = pendingMarkers;
      tTrue.location = term.location;
      aut.transitions.push_back(std::move(tTrue));

      ProcessTransition tFalse;
      tFalse.source = cursor;
      tFalse.target = blockEntryState.at(*term.falseSuccessor);
      tFalse.kind = BehavioralActionKind::Internal;
      tFalse.fairTag = fairTag;
      tFalse.fairBranchOutcome = false;
      tFalse.assumptionJustified = justified;
      tFalse.progressMarkers = std::move(pendingMarkers);
      tFalse.location = term.location;
      aut.transitions.push_back(std::move(tFalse));
    }
  }

  return aut;
}

bool validateCompatibilityCertificate(const CompatibilityCertificate &cert,
                                      bool requireCleanQueueDrain,
                                      std::string *failureReason) {
  auto fail = [&](std::string msg) {
    if (failureReason) *failureReason = std::move(msg);
    return false;
  };

  if (cert.states.empty() || cert.initialStateIndex >= cert.states.size() ||
      cert.isAcceptingTerminalState.size() != cert.states.size()) {
    return fail("certificate state table or initial state index is malformed");
  }
  if (cert.queueCapacities.size() != cert.queueSeededItems.size()) {
    return fail("certificate queue metadata dimensions disagree");
  }

  // 1. Check queue slot conservation (`free + reserved + queued == C`) and
  //    terminal queue drain in every reachable state.
  for (std::size_t idx = 0; idx < cert.states.size(); ++idx) {
    const auto &st = cert.states[idx];
    if (st.queueStates.size() != cert.queueCapacities.size()) {
      return fail("state queue vector size disagrees with pipeline queue count");
    }
    for (std::size_t q = 0; q < st.queueStates.size(); ++q) {
      const auto &qs = st.queueStates[q];
      if (qs.capacity != cert.queueCapacities[q] ||
          !holdsQueueSlotConservation(qs.capacity, qs.freeCredits,
                                      qs.reservedSlots, qs.items.size())) {
        return fail("queue slot conservation invariant violated at state " +
                    std::to_string(idx));
      }
      if (cert.isAcceptingTerminalState[idx] && requireCleanQueueDrain) {
        if (qs.reservedSlots != 0 ||
            qs.items.size() != cert.queueSeededItems[q]) {
          return fail("terminal state leaves undrained or unreturned queue items");
        }
      }
    }
  }

  // 2. Build outgoing edge index & check transition bounds + stuck-state freedom.
  std::vector<std::vector<const CompositeStepEdge *>> outEdges(
      cert.states.size());
  for (const auto &e : cert.edges) {
    if (e.fromStateIndex >= cert.states.size() ||
        e.toStateIndex >= cert.states.size()) {
      return fail("certificate step edge points outside invariant state set");
    }
    outEdges[e.fromStateIndex].push_back(&e);
  }

  for (std::size_t idx = 0; idx < cert.states.size(); ++idx) {
    if (!cert.isAcceptingTerminalState[idx] && outEdges[idx].empty()) {
      return fail("non-terminal state " + std::to_string(idx) +
                  " has no outgoing transitions (stuck / deadlocked)");
    }
  }

  // 3. Check fair-cycle progress across all non-trivial SCCs using iterative
  //    fair-SCC refinement (weak co-resident stage justice + strong FairChoice
  //    branch fairness).
  const std::size_t numStages =
      cert.states.empty() ? 0 : cert.states[0].stageStates.size();
  std::vector<std::vector<bool>> stageEnabledInG(
      cert.states.size(), std::vector<bool>(numStages, false));
  std::vector<std::map<std::string, bool>> dualFairTagAtU(cert.states.size());

  for (std::size_t u = 0; u < cert.states.size(); ++u) {
    std::map<std::string, std::tuple<bool, bool, bool>> tagAvail;
    for (const CompositeStepEdge *edge : outEdges[u]) {
      if (edge->actorStageIndex < numStages) {
        stageEnabledInG[u][edge->actorStageIndex] = true;
      }
      if (edge->fairTag) {
        auto &[tSeen, fSeen, just] = tagAvail[*edge->fairTag];
        if (edge->fairBranchOutcome) tSeen = true;
        else fSeen = true;
        just = edge->assumptionJustified;
      }
    }
    for (const auto &[tag, avail] : tagAvail) {
      const auto [tSeen, fSeen, just] = avail;
      if (tSeen && fSeen) dualFairTagAtU[u][tag] = just;
    }
  }

  auto findFairBadCycle =
      [&](const auto &isBadEdge) -> std::optional<std::size_t> {
    std::vector<bool> active(cert.states.size(), true);
    bool changed = true;
    while (changed) {
      changed = false;
      std::vector<std::vector<std::size_t>> subAdj(cert.states.size());
      for (std::size_t u = 0; u < cert.states.size(); ++u) {
        if (!active[u]) continue;
        for (const CompositeStepEdge *e : outEdges[u]) {
          if (active[e->toStateIndex] && isBadEdge(*e)) {
            subAdj[u].push_back(e->toStateIndex);
          }
        }
      }

      const SCCResult sccs = computeTarjanSCCs(cert.states.size(), subAdj);
      for (const auto &comp : sccs.components) {
        bool hasCycle = comp.size() > 1;
        if (!hasCycle && comp.size() == 1) {
          const std::size_t u = comp[0];
          hasCycle =
              std::find(subAdj[u].begin(), subAdj[u].end(), u) != subAdj[u].end();
        }
        if (!hasCycle) continue;

        std::set<std::size_t> compSet(comp.begin(), comp.end());
        std::vector<bool> stageStepsInComp(numStages, false);
        std::map<std::string, std::pair<bool, bool>> tagInComp;

        for (std::size_t u : comp) {
          for (const CompositeStepEdge *e : outEdges[u]) {
            if (!compSet.contains(e->toStateIndex) || !isBadEdge(*e)) continue;
            if (e->actorStageIndex < numStages) {
              stageStepsInComp[e->actorStageIndex] = true;
            }
            if (e->fairTag) {
              auto &pair = tagInComp[*e->fairTag];
              if (e->fairBranchOutcome) pair.first = true;
              else pair.second = true;
            }
          }
        }

        // Weak co-resident stage scheduling fairness (justice): if a stage `s`
        // never steps inside `comp` yet is continuously enabled at EVERY vertex
        // of `comp`, no subset of `comp` admits a weakly-fair infinite run.
        bool violatesWeakStageJustice = false;
        for (std::size_t s = 0; s < numStages; ++s) {
          if (stageStepsInComp[s]) continue;
          bool alwaysEnabled = true;
          for (std::size_t u : comp) {
            if (!stageEnabledInG[u][s]) {
              alwaysEnabled = false;
              break;
            }
          }
          if (alwaysEnabled) {
            violatesWeakStageJustice = true;
            break;
          }
        }
        if (violatesWeakStageJustice) continue;

        // Strong FairChoice branch fairness: prune vertices in `comp` that have
        // both branch outcomes enabled in `G` when `comp` omits an outcome.
        bool prunedChoiceVertex = false;
        for (std::size_t u : comp) {
          for (const auto &[tag, just] : dualFairTagAtU[u]) {
            (void)just;
            const auto outcomes = tagInComp[tag];
            if (!(outcomes.first && outcomes.second)) {
              active[u] = false;
              prunedChoiceVertex = true;
              changed = true;
              break;
            }
          }
        }
        if (prunedChoiceVertex) continue;

        return comp.front();
      }
    }
    return std::nullopt;
  };

  if (!cert.requiredProgressMarkers.empty()) {
    for (const auto &req : cert.requiredProgressMarkers) {
      if (findFairBadCycle([&](const CompositeStepEdge &e) {
            return std::find(e.progressMarkers.begin(), e.progressMarkers.end(),
                             req) == e.progressMarkers.end();
          }).has_value()) {
        return fail("fair cycle fails to revisit required progress marker `" +
                    req + "`");
      }
    }
  } else {
    if (findFairBadCycle([](const CompositeStepEdge &e) {
          return e.actionKind != BehavioralActionKind::Send &&
                 e.actionKind != BehavioralActionKind::Receive;
        }).has_value()) {
      return fail("fair cycle spins internally without communication or termination");
    }
  }

  return true;
}

namespace {

struct InternalSolverResult {
  bool hitBudgetLimit = false;
  CompatibilityCertificate cert;
  std::set<std::string> conditionalFairness;
};

// Checks static Layer G1, G4, and G5 obligations on `system` before running
// Layer G2 product-game exploration. Returns `false` if fatal structural errors
// preclude building the queue product state space.
bool verifyStaticPipelineLayers(const GPUPipelineSystemSpec &system,
                                const GPUBoundedQueueOptions &options,
                                GPUBoundedQueueReport &report) {
  bool canExplore = true;

  if (system.hasUnconnectedDuplicateConsumer) {
    recordQueueViolation(
        report, VerificationStatus::Rejected,
        "gpu.pipeline.invalid_consumer_fanout", system.name,
        system.duplicateConsumerEdge,
        "GPU bounded queue `" + system.duplicateConsumerEdge +
            "` has multiple competing consumer stages (expected SPSC/MPSC)",
        system.duplicateConsumerLoc);
    canExplore = false;
  }

  // Layer G4: Check schedule warp placements & workgroup thread budgets.
  std::map<std::string, std::map<std::size_t, std::string>> workerWarpOwners;
  for (const auto &[stageName, place] : system.placements) {
    if (place.workerBlocks == 0) {
      recordQueueViolation(report, VerificationStatus::Rejected,
                           "gpu.residency.invalid_worker_blocks", system.name,
                           place.workerName,
                           "fused GPU worker `" + place.workerName +
                               "` must declare blocks >= 1",
                           place.location);
      canExplore = false;
    }
    if (place.hasWarpPlacement) {
      const std::size_t requiredThreads =
          (place.lastWarp + 1) * options.profile.nvidiaSubgroupSize;
      if (place.lastWarp >= options.profile.maxWorkgroupWarps ||
          requiredThreads > options.profile.maxWorkgroupThreads) {
        recordQueueViolation(
            report, VerificationStatus::Rejected,
            "gpu.residency.budget_exceeded", system.name, stageName,
            "warp placement [" + std::to_string(place.firstWarp) + ".." +
                std::to_string(place.lastWarp) +
                "] exceeds workgroup warp/thread budget",
            place.location);
      }
      auto &used = workerWarpOwners[place.workerName];
      for (std::size_t w = place.firstWarp; w <= place.lastWarp; ++w) {
        auto [it, inserted] = used.emplace(w, stageName);
        if (!inserted && it->second != stageName) {
          recordQueueViolation(
              report, VerificationStatus::Rejected,
              "gpu.residency.overlapping_warp_placement", system.name,
              stageName,
              "stages `" + it->second + "` and `" + stageName +
                  "` overlap on warp " + std::to_string(w) + " in worker `" +
                  place.workerName + "`",
              place.location);
        }
      }
    }
  }

  // Layers G1 + G4 + G5 per queue edge:
  for (const auto &q : system.queues) {
    ++report.checkedQueues;

    // G1: Capacity bounds & initial slot conservation.
    if (q.capacity < options.profile.minBoundedQueueCapacity) {
      recordQueueViolation(
          report, VerificationStatus::Rejected, "gpu.queue.invalid_capacity",
          system.name, q.name,
          "GPU bounded queue capacity must be >= " +
              std::to_string(options.profile.minBoundedQueueCapacity),
          q.location);
      canExplore = false;
      continue;
    }
    if (q.capacity > options.profile.maxBoundedQueueCapacity) {
      recordQueueViolation(
          report, VerificationStatus::Unsupported,
          "gpu.queue.unsupported_capacity", system.name, q.name,
          "GPU bounded queue capacity " + std::to_string(q.capacity) +
              " exceeds pinned profile bound (" +
              std::to_string(options.profile.maxBoundedQueueCapacity) + ")",
          q.location);
      canExplore = false;
      continue;
    }
    if (q.seededItems > q.capacity ||
        !holdsQueueSlotConservation(q.capacity, q.capacity - q.seededItems, 0,
                                    q.seededItems)) {
      recordQueueViolation(
          report, VerificationStatus::Rejected,
          "gpu.queue.slot_conservation_violation", system.name, q.name,
          "seeded item count (" + std::to_string(q.seededItems) +
              ") exceeds queue capacity (" + std::to_string(q.capacity) +
              "), violating `free + reserved + queued = capacity`",
          q.location);
      canExplore = false;
      continue;
    }

    // G1: Protocol existence & initial state validity.
    if (!q.protocolName.empty()) {
      auto pIt = system.protocols.find(q.protocolName);
      if (pIt == system.protocols.end() ||
          !pIt->second.states.contains(q.initialProtocolState)) {
        recordQueueViolation(
            report, VerificationStatus::Rejected,
            "gpu.protocol.unknown_initial_state", system.name, q.name,
            "channel `" + q.name + "` references unknown protocol or initial state",
            q.location);
        canExplore = false;
        continue;
      }
    }

    // G4: Workgroup co-residency check across producer and consumer stages.
    const bool isBlockingEdge =
        q.backpressure.empty() || q.backpressure == "cooperative" ||
        q.backpressure == "blocking";
    bool crossBlockOrCrossWorker = (q.scope == "device" || q.scope == "grid" ||
                                    q.scope == "cluster");
    bool cooperativeContractSatisfied = false;

    if (!system.placements.empty() && !q.producerStage.empty() &&
        !q.consumerStage.empty()) {
      const auto prodIt = system.placements.find(q.producerStage);
      const auto consIt = system.placements.find(q.consumerStage);
      if (prodIt == system.placements.end() ||
          consIt == system.placements.end()) {
        crossBlockOrCrossWorker = true;
      } else {
        const auto &pP = prodIt->second;
        const auto &cP = consIt->second;
        if (pP.workerName != cP.workerName || pP.workerBlocks != 1) {
          crossBlockOrCrossWorker = true;
          if (pP.workerName == cP.workerName &&
              pP.workerResidency == "cooperative" &&
              options.profile.allowCooperativeGridChannels &&
              q.storage == "gpu::global" &&
              (q.scope == "device" || q.scope == "grid")) {
            cooperativeContractSatisfied = true;
          }
        }
      }
    } else if (crossBlockOrCrossWorker &&
               options.profile.allowCooperativeGridChannels &&
               q.storage == "gpu::global") {
      cooperativeContractSatisfied = true;
    }

    if (q.storage == "gpu::shared" && q.scope != "block" && q.scope != "warp") {
      recordQueueViolation(
          report, VerificationStatus::Rejected,
          "gpu.residency.shared_storage_scope_mismatch", system.name, q.name,
          "`gpu::shared` queue storage cannot span `" + q.scope + "` scope",
          q.location);
    }

    GPUResidencyContract effectiveResidency = q.residency;
    if (crossBlockOrCrossWorker) {
      effectiveResidency = cooperativeContractSatisfied
                               ? GPUResidencyContract::CooperativeGrid
                               : GPUResidencyContract::UnscheduledGrid;
    }

    if (isBlockingEdge &&
        effectiveResidency == GPUResidencyContract::UnscheduledGrid) {
      recordQueueViolation(
          report, VerificationStatus::Rejected,
          "gpu.residency.unscheduled_cross_block_blocking", system.name, q.name,
          "blocking GPU channel `" + q.name +
              "` crosses thread-block or kernel boundaries without a verified "
              "cooperative grid residency contract",
          q.location);
    }

    // G5: Acquire/release publication memory-scope sufficiency!
    if (!hasSufficientPublicationScope(effectiveResidency, q.publication) ||
        !isBidirectionalAcqRelOrder(q.publication)) {
      if (effectiveResidency != GPUResidencyContract::UnscheduledGrid ||
          q.explicitPublication) {
        recordQueueViolation(
            report, VerificationStatus::Rejected,
            "gpu.memory.insufficient_publication_scope", system.name, q.name,
            "queue `" + q.name +
                "` atomic publication order lacks required acquire/release "
                "visibility for `" +
                gpuResidencyContractName(effectiveResidency) + "` residency",
            q.location);
      }
    }
  }

  // Verify no stage uses hardware `ClockTick` inside a GPU pipeline!
  for (const auto &st : system.stages) {
    for (const auto &tr : st.transitions) {
      if (tr.kind == BehavioralActionKind::ClockTick) {
        recordQueueViolation(
            report, VerificationStatus::Rejected,
            "gpu.behavioral.clock_tick_in_gpu_stage", system.name, st.name,
            "GPU stage `" + st.name +
                "` contains a hardware clock-tick action; GPU channels are "
                "asynchronous and cannot model waits as shared RTL ticks",
            tr.location);
        canExplore = false;
      }
    }
  }

  return canExplore;
}

} // namespace

void verifyGPUPipelineSystem(const GPUPipelineSystemSpec &system,
                             const GPUBoundedQueueOptions &options,
                             GPUBoundedQueueReport &report) {
  ++report.checkedPipelines;
  report.checkedStages += system.stages.size();
  report.checkedSchedules += system.scheduleCount;

  const std::size_t preStaticViolations = report.violations.size();
  const bool canExplore = verifyStaticPipelineLayers(system, options, report);
  if (!canExplore) return;

  // Map internal queue names -> index `[0 .. system.queues.size() - 1]`.
  std::map<std::string, std::size_t> queueIndexByName;
  for (std::size_t i = 0; i < system.queues.size(); ++i) {
    queueIndexByName[system.queues[i].name] = i;
    // Also map unqualified suffix (`Pipeline::edge` -> `edge`) when unique.
    const auto pos = system.queues[i].name.rfind("::");
    if (pos != std::string::npos) {
      queueIndexByName.try_emplace(system.queues[i].name.substr(pos + 2), i);
    }
  }

  // Verify stage communication endpoints bind to known queues (or external
  // boundary ports when explicitly unbound).
  // Build initial composite state.
  CompositeSystemState initState;
  for (const auto &stage : system.stages) {
    initState.stageStates.push_back(stage.initialState);
  }
  for (const auto &q : system.queues) {
    BoundedQueueRuntimeState qs;
    qs.capacity = q.capacity;
    qs.freeCredits = q.capacity - q.seededItems;
    qs.reservedSlots = 0;
    std::string seedTag = q.seededMessage;
    if (seedTag.empty() && !q.protocolName.empty() &&
        system.protocols.contains(q.protocolName)) {
      const auto &proto = system.protocols.at(q.protocolName);
      auto stIt = proto.states.find(q.initialProtocolState);
      if (stIt != proto.states.end() && !stIt->second.messages.empty()) {
        seedTag = stIt->second.messages.front().name;
      }
    }
    if (seedTag.empty()) seedTag = "token";
    for (std::size_t k = 0; k < q.seededItems; ++k) qs.items.push_back(seedTag);
    initState.queueStates.push_back(std::move(qs));
    initState.protocolStates.push_back(q.initialProtocolState);
  }

  // Index outgoing transitions per `(stageIdx, stateId)`.
  std::vector<std::vector<std::vector<const ProcessTransition *>>> stageOut(
      system.stages.size());
  std::set<std::string> allRequiredMarkers;
  for (std::size_t s = 0; s < system.stages.size(); ++s) {
    stageOut[s].resize(system.stages[s].states.size());
    for (const auto &tr : system.stages[s].transitions) {
      if (tr.source < stageOut[s].size()) stageOut[s][tr.source].push_back(&tr);
    }
    allRequiredMarkers.insert(
        system.stages[s].requiredProgressMarkers.begin(),
        system.stages[s].requiredProgressMarkers.end());
  }

  auto isArmEnabledInState = [&](const CompositeSystemState &st,
                                 BehavioralActionKind kind,
                                 const std::string &ep,
                                 const std::string &msg) -> bool {
    auto qIt = queueIndexByName.find(ep);
    if (qIt == queueIndexByName.end()) {
      // Boundary channel: enabled under honest environment strategy.
      return true;
    }
    const auto &qs = st.queueStates[qIt->second];
    if (kind == BehavioralActionKind::Send) {
      return qs.freeCredits >= 1;
    }
    if (kind == BehavioralActionKind::Receive) {
      return !qs.items.empty() && (msg.empty() || qs.items.front() == msg);
    }
    return true;
  };

  auto isAllTerminated = [&](const CompositeSystemState &st) -> bool {
    for (std::size_t s = 0; s < system.stages.size(); ++s) {
      const auto sid = st.stageStates[s];
      if (sid >= system.stages[s].states.size() ||
          !system.stages[s].states[sid].isAcceptingTerminal) {
        return false;
      }
    }
    return true;
  };

  CompatibilityCertificate cert;
  cert.pipelineName = system.name;
  for (const auto &q : system.queues) {
    cert.queueCapacities.push_back(q.capacity);
    cert.queueSeededItems.push_back(q.seededItems);
  }
  cert.requiredProgressMarkers = allRequiredMarkers;

  std::map<CompositeSystemState, std::size_t> stateIdMap;
  std::vector<std::optional<std::pair<std::size_t, std::string>>> parentTrace;
  std::deque<std::size_t> worklist;

  stateIdMap.emplace(initState, 0);
  cert.states.push_back(initState);
  cert.isAcceptingTerminalState.push_back(isAllTerminated(initState));
  parentTrace.push_back(std::nullopt);
  worklist.push_back(0);

  auto reconstructTrace = [&](std::size_t targetIdx) {
    std::vector<std::string> trace;
    std::optional<std::size_t> cur = targetIdx;
    while (cur.has_value()) {
      const std::string snap = formatCompositeState(system, cert.states[*cur]);
      if (parentTrace[*cur].has_value()) {
        trace.push_back(parentTrace[*cur]->second + " -> " + snap);
        cur = parentTrace[*cur]->first;
      } else {
        trace.push_back("init: " + snap);
        break;
      }
    }
    std::reverse(trace.begin(), trace.end());
    return trace;
  };

  while (!worklist.empty()) {
    const std::size_t curIdx = worklist.front();
    worklist.pop_front();
    const CompositeSystemState curState = cert.states[curIdx];

    if (cert.isAcceptingTerminalState[curIdx]) {
      if (options.requireCleanQueueDrain) {
        for (std::size_t q = 0; q < system.queues.size(); ++q) {
          const auto &qs = curState.queueStates[q];
          if (qs.reservedSlots != 0 ||
              qs.items.size() != system.queues[q].seededItems) {
            recordQueueViolation(
                report, VerificationStatus::Rejected,
                "gpu.queue.undrained_at_termination", system.name,
                system.queues[q].name,
                "pipeline terminated with undrained items (" +
                    std::to_string(qs.items.size()) + " vs seeded " +
                    std::to_string(system.queues[q].seededItems) +
                    ") in queue `" + system.queues[q].name + "`",
                system.queues[q].location, reconstructTrace(curIdx));
            return;
          }
        }
      }
      continue;
    }

    std::size_t enabledFromCur = 0;

    for (std::size_t s = 0; s < system.stages.size(); ++s) {
      const AutomatonStateId sid = curState.stageStates[s];
      if (sid >= system.stages[s].states.size()) continue;
      const auto &stNode = system.stages[s].states[sid];
      if (stNode.isAcceptingTerminal) continue;

      bool anyOfferArmEnabled = false;
      if (stNode.offer.has_value()) {
        for (const auto &arm : stNode.offer->arms) {
          if (isArmEnabledInState(curState, arm.kind, arm.endpoint,
                                  arm.message)) {
            anyOfferArmEnabled = true;
            break;
          }
        }
      }

      for (const ProcessTransition *tr : stageOut[s][sid]) {
        if (tr->isOfferDefault && anyOfferArmEnabled) {
          // Offer `else`/`default` fallback commits ONLY when no offered arm is
          // currently enabled!
          continue;
        }
        if (stNode.offer.has_value() && !tr->isOfferDefault &&
            !tr->endpoint.empty()) {
          // Determine the arm's direction from `stNode.offer` when the arm
          // jumps to an explicit `ChannelReceive`/`ChannelSend` block.
          BehavioralActionKind checkKind = tr->kind;
          if (checkKind == BehavioralActionKind::Internal) {
            for (const auto &arm : stNode.offer->arms) {
              if (arm.targetState == tr->target) {
                checkKind = arm.kind;
                break;
              }
            }
          }
          if (!isArmEnabledInState(curState, checkKind, tr->endpoint,
                                   tr->message)) {
            continue;
          }
        }

        CompositeSystemState nextState = curState;
        nextState.stageStates[s] = tr->target;

        if (tr->kind == BehavioralActionKind::Send ||
            tr->kind == BehavioralActionKind::Receive) {
          if (tr->delivery == graphir::ChannelDeliveryMode::SameStep) {
            recordQueueViolation(
                report, VerificationStatus::Rejected,
                "gpu.queue.unsupported_rendezvous_on_gpu", system.name,
                tr->endpoint,
                "same-step zero-capacity rendezvous on `" + tr->endpoint +
                    "` is invalid for asynchronous GPU warps; use a bounded queue",
                tr->location);
            return;
          }

          auto qIt = queueIndexByName.find(tr->endpoint);
          if (qIt != queueIndexByName.end()) {
            const std::size_t qIdx = qIt->second;
            const auto &qSpec = system.queues[qIdx];
            auto &qs = nextState.queueStates[qIdx];

            if (tr->kind == BehavioralActionKind::Send) {
              if (qs.freeCredits == 0) continue; // Blocked on full queue!

              // Check protocol transition conformance on enqueue!
              if (!qSpec.protocolName.empty() &&
                  system.protocols.contains(qSpec.protocolName)) {
                const auto &proto = system.protocols.at(qSpec.protocolName);
                const std::string &curPState = nextState.protocolStates[qIdx];
                const auto psIt = proto.states.find(curPState);
                if (psIt == proto.states.end()) {
                  recordQueueViolation(
                      report, VerificationStatus::Rejected,
                      "gpu.protocol.conformance_violation", system.name,
                      qSpec.name,
                      "channel `" + qSpec.name + "` entered invalid protocol state",
                      tr->location);
                  return;
                }
                const auto msgIt = std::find_if(
                    psIt->second.messages.begin(), psIt->second.messages.end(),
                    [&](const ProtocolMessageSpec &m) {
                      return m.name == tr->message &&
                             (qSpec.producerRole.empty() ||
                              m.senderRole == qSpec.producerRole);
                    });
                if (msgIt == psIt->second.messages.end()) {
                  recordQueueViolation(
                      report, VerificationStatus::Rejected,
                      "gpu.protocol.conformance_violation", system.name,
                      qSpec.name,
                      "stage `" + system.stages[s].name + "` sent illegal message `" +
                          tr->message + "` on `" + qSpec.name +
                          "` in protocol state `" + curPState + "`",
                      tr->location, reconstructTrace(curIdx));
                  return;
                }
                nextState.protocolStates[qIdx] = msgIt->successorState;
              }

              // Reserve slot -> check intermediate conservation -> commit item!
              --qs.freeCredits;
              ++qs.reservedSlots;
              if (!holdsQueueSlotConservation(qs.capacity, qs.freeCredits,
                                              qs.reservedSlots,
                                              qs.items.size())) {
                recordQueueViolation(
                    report, VerificationStatus::Rejected,
                    "gpu.queue.slot_conservation_violation", system.name,
                    qSpec.name,
                    "intermediate slot reservation violated `free + reserved + queued = C`",
                    tr->location);
                return;
              }
              --qs.reservedSlots;
              qs.items.push_back(tr->message);
            } else {
              // Receive from bounded queue:
              if (qs.items.empty()) continue; // Blocked on empty queue!
              if (!tr->message.empty() && qs.items.front() != tr->message) {
                // Head-of-line message mismatch: receiver expects `tr->message`
                // while front of FIFO is `qs.items.front()`!
                continue;
              }
              qs.items.erase(qs.items.begin());
              ++qs.freeCredits;
            }

            if (!holdsQueueSlotConservation(qs.capacity, qs.freeCredits,
                                            qs.reservedSlots,
                                            qs.items.size())) {
              recordQueueViolation(
                  report, VerificationStatus::Rejected,
                  "gpu.queue.slot_conservation_violation", system.name,
                  qSpec.name,
                  "queue transition violated `free + reserved + queued = C`",
                  tr->location);
              return;
            }
          }
        }

        ++enabledFromCur;
        if (cert.edges.size() >= options.maxExploredTransitions) {
          report.exploredCompositeStates = cert.states.size();
          report.exploredCompositeTransitions = cert.edges.size();
          recordQueueViolation(
              report, VerificationStatus::Unknown,
              "gpu.solver.transition_limit_exceeded", system.name, system.name,
              "bounded-queue product exploration exceeded transition limit (" +
                  std::to_string(options.maxExploredTransitions) +
                  "); reporting `unknown`",
              system.location);
          return;
        }

        std::size_t dstIdx = 0;
        auto [it, inserted] = stateIdMap.emplace(nextState, cert.states.size());
        if (inserted) {
          if (cert.states.size() >= options.maxExploredStates) {
            report.exploredCompositeStates = cert.states.size();
            report.exploredCompositeTransitions = cert.edges.size();
            recordQueueViolation(
                report, VerificationStatus::Unknown,
                "gpu.solver.state_limit_exceeded", system.name, system.name,
                "bounded-queue product exploration exceeded state limit (" +
                    std::to_string(options.maxExploredStates) +
                    "); reporting `unknown`",
                system.location);
            return;
          }
          dstIdx = cert.states.size();
          cert.states.push_back(nextState);
          cert.isAcceptingTerminalState.push_back(isAllTerminated(nextState));
          std::string stepDesc =
              system.stages[s].name + ":" +
              std::string(behavioralActionKindName(tr->kind));
          if (!tr->endpoint.empty()) {
            stepDesc += "(" + tr->endpoint + "::" + tr->message + ")";
          }
          parentTrace.push_back(std::make_pair(curIdx, std::move(stepDesc)));
          worklist.push_back(dstIdx);
        } else {
          dstIdx = it->second;
        }

        cert.edges.push_back(CompositeStepEdge{
            curIdx, dstIdx, s, tr->kind, tr->endpoint, tr->message, tr->fairTag,
            tr->fairBranchOutcome, tr->assumptionJustified,
            tr->progressMarkers});
      }
    }

    if (enabledFromCur == 0) {
      // Reachable non-terminal stuck state -> DEADLOCK!
      report.exploredCompositeStates = cert.states.size();
      report.exploredCompositeTransitions = cert.edges.size();
      recordQueueViolation(
          report, VerificationStatus::Rejected, "gpu.queue.deadlock",
          system.name, system.name,
          "reachable composite deadlock / stuck state in GPU pipeline `" +
              system.name + "`: " + formatCompositeState(system, curState),
          system.location, reconstructTrace(curIdx));
      return;
    }
  }

  report.exploredCompositeStates += cert.states.size();
  report.exploredCompositeTransitions += cert.edges.size();

  // Check SCC liveness / progress markers / conditional fairness assumptions
  // using iterative fair-SCC refinement (weak co-resident stage justice +
  // strong FairChoice branch fairness).
  std::vector<std::vector<const CompositeStepEdge *>> outEdges(
      cert.states.size());
  for (const auto &e : cert.edges) {
    outEdges[e.fromStateIndex].push_back(&e);
  }

  const std::size_t numStages = system.stages.size();
  std::vector<std::vector<bool>> stageEnabledInG(
      cert.states.size(), std::vector<bool>(numStages, false));
  std::vector<std::map<std::string, bool>> dualFairTagAtU(cert.states.size());

  for (std::size_t u = 0; u < cert.states.size(); ++u) {
    std::map<std::string, std::tuple<bool, bool, bool>> tagAvail;
    for (const CompositeStepEdge *edge : outEdges[u]) {
      if (edge->actorStageIndex < numStages) {
        stageEnabledInG[u][edge->actorStageIndex] = true;
      }
      if (edge->fairTag) {
        auto &[tSeen, fSeen, just] = tagAvail[*edge->fairTag];
        if (edge->fairBranchOutcome) tSeen = true;
        else fSeen = true;
        just = edge->assumptionJustified;
      }
    }
    for (const auto &[tag, avail] : tagAvail) {
      const auto [tSeen, fSeen, just] = avail;
      if (tSeen && fSeen) dualFairTagAtU[u][tag] = just;
    }
  }

  auto findFairBadCycle =
      [&](const auto &isBadEdge) -> std::optional<std::size_t> {
    std::vector<bool> active(cert.states.size(), true);
    bool changed = true;
    while (changed) {
      changed = false;
      std::vector<std::vector<std::size_t>> subAdj(cert.states.size());
      for (std::size_t u = 0; u < cert.states.size(); ++u) {
        if (!active[u]) continue;
        for (const CompositeStepEdge *e : outEdges[u]) {
          if (active[e->toStateIndex] && isBadEdge(*e)) {
            subAdj[u].push_back(e->toStateIndex);
          }
        }
      }

      const SCCResult sccs = computeTarjanSCCs(cert.states.size(), subAdj);
      for (const auto &comp : sccs.components) {
        bool hasCycle = comp.size() > 1;
        if (!hasCycle && comp.size() == 1) {
          const std::size_t u = comp[0];
          hasCycle =
              std::find(subAdj[u].begin(), subAdj[u].end(), u) != subAdj[u].end();
        }
        if (!hasCycle) continue;

        std::set<std::size_t> compSet(comp.begin(), comp.end());
        std::vector<bool> stageStepsInComp(numStages, false);
        std::map<std::string, std::pair<bool, bool>> tagInComp;

        for (std::size_t u : comp) {
          for (const CompositeStepEdge *e : outEdges[u]) {
            if (!compSet.contains(e->toStateIndex) || !isBadEdge(*e)) continue;
            if (e->actorStageIndex < numStages) {
              stageStepsInComp[e->actorStageIndex] = true;
            }
            if (e->fairTag) {
              auto &pair = tagInComp[*e->fairTag];
              if (e->fairBranchOutcome) pair.first = true;
              else pair.second = true;
            }
          }
        }

        // Weak co-resident stage scheduling fairness (justice): if a stage `s`
        // never steps inside `comp` yet is continuously enabled at EVERY vertex
        // of `comp`, no subset of `comp` admits a weakly-fair infinite run.
        bool violatesWeakStageJustice = false;
        for (std::size_t s = 0; s < numStages; ++s) {
          if (stageStepsInComp[s]) continue;
          bool alwaysEnabled = true;
          for (std::size_t u : comp) {
            if (!stageEnabledInG[u][s]) {
              alwaysEnabled = false;
              break;
            }
          }
          if (alwaysEnabled) {
            violatesWeakStageJustice = true;
            break;
          }
        }
        if (violatesWeakStageJustice) continue;

        // Strong FairChoice branch fairness: prune vertices in `comp` that have
        // both branch outcomes enabled in `G` when `comp` omits an outcome,
        // recording any unjustified fairness assumption relied upon.
        bool prunedChoiceVertex = false;
        for (std::size_t u : comp) {
          for (const auto &[tag, just] : dualFairTagAtU[u]) {
            const auto outcomes = tagInComp[tag];
            if (!(outcomes.first && outcomes.second)) {
              active[u] = false;
              prunedChoiceVertex = true;
              changed = true;
              if (!just) report.conditionalFairnessTags.insert(tag);
            }
          }
        }
        if (prunedChoiceVertex) continue;

        return comp.front();
      }
    }
    return std::nullopt;
  };

  if (!allRequiredMarkers.empty()) {
    for (const auto &req : allRequiredMarkers) {
      if (const auto badNode = findFairBadCycle([&](const CompositeStepEdge &e) {
            return std::find(e.progressMarkers.begin(), e.progressMarkers.end(),
                             req) == e.progressMarkers.end();
          })) {
        recordQueueViolation(
            report, VerificationStatus::Rejected,
            "gpu.queue.livelock_or_starvation", system.name, req,
            "reachable fair execution cycle in `" + system.name +
                "` starves required progress marker `" + req + "`",
            system.location, reconstructTrace(*badNode));
        return;
      }
    }
  } else {
    if (const auto badNode = findFairBadCycle([](const CompositeStepEdge &e) {
          return e.actionKind != BehavioralActionKind::Send &&
                 e.actionKind != BehavioralActionKind::Receive;
        })) {
      recordQueueViolation(
          report, VerificationStatus::Rejected,
          "gpu.queue.livelock_or_starvation", system.name, system.name,
          "reachable fair execution cycle in `" + system.name +
              "` spins internally without communication or termination",
          system.location, reconstructTrace(*badNode));
      return;
    }
  }

  // Run independent certificate validator before recording `UnboundedProved`!
  if (options.validateGeneratedCertificates &&
      report.violations.size() == preStaticViolations) {
    std::string certFailure;
    if (!validateCompatibilityCertificate(cert, options.requireCleanQueueDrain,
                                         &certFailure)) {
      recordQueueViolation(
          report, VerificationStatus::Rejected,
          "gpu.solver.invalid_certificate", system.name, system.name,
          "independent certificate validator rejected solver witness: " +
              certFailure,
          system.location);
      return;
    }
    ++report.validatedCertificates;
    report.certificates.push_back(std::move(cert));
  }
}

std::optional<BehavioralSummaryAutomaton>
computeBoundarySummary(const GPUPipelineSystemSpec &system,
                       const GPUBoundedQueueOptions &options) {
  GPUBoundedQueueReport rep;
  verifyGPUPipelineSystem(system, options, rep);
  if (!rep.accepted() || rep.certificates.empty()) return std::nullopt;

  const auto &cert = rep.certificates.front();
  BehavioralSummaryAutomaton summary;
  summary.automaton.name = system.name + "::summary";
  summary.automaton.initialState = cert.initialStateIndex;
  summary.automaton.requiredProgressMarkers = cert.requiredProgressMarkers;
  summary.conditionalFairnessTags = rep.conditionalFairnessTags;
  summary.automaton.unprovedFairnessAssumptions = rep.conditionalFairnessTags;
  summary.automaton.location = system.location;

  for (const auto &q : system.queues) {
    summary.hiddenInternalEndpoints.insert(q.name);
    const auto pos = q.name.rfind("::");
    if (pos != std::string::npos)
      summary.hiddenInternalEndpoints.insert(q.name.substr(pos + 2));
  }

  for (std::size_t idx = 0; idx < cert.states.size(); ++idx) {
    summary.automaton.states.push_back(
        AutomatonState{idx, "s" + std::to_string(idx),
                       cert.isAcceptingTerminalState[idx], std::nullopt,
                       system.location});
  }

  for (const auto &edge : cert.edges) {
    ProcessTransition tr;
    tr.source = edge.fromStateIndex;
    tr.target = edge.toStateIndex;
    const bool isInternalQueue =
        summary.hiddenInternalEndpoints.contains(edge.endpoint);
    if (isInternalQueue || edge.actionKind == BehavioralActionKind::Internal) {
      tr.kind = BehavioralActionKind::Internal;
    } else {
      tr.kind = edge.actionKind;
      tr.endpoint = edge.endpoint;
      tr.message = edge.message;
      summary.boundaryEndpoints.insert(edge.endpoint);
    }
    tr.fairTag = edge.fairTag;
    tr.fairBranchOutcome = edge.fairBranchOutcome;
    tr.assumptionJustified = edge.assumptionJustified;
    tr.progressMarkers = edge.progressMarkers;
    summary.automaton.transitions.push_back(std::move(tr));
  }

  return summary;
}

namespace {

std::map<std::string, ProtocolAutomatonSpec>
extractModuleProtocols(const graphir::Module &module) {
  std::map<std::string, ProtocolAutomatonSpec> out;
  for (const auto &proto : module.protocolGraphs()) {
    ProtocolAutomatonSpec spec;
    spec.name = proto.name;
    spec.roles = proto.roles;
    spec.location = proto.location;

    for (const auto &mem : proto.members) {
      if (mem.graph >= module.resourceGraphs().size() ||
          mem.member >= module.resourceGraphs()[mem.graph].references().size()) {
        continue;
      }
      const auto ref =
          module.resourceGraphs()[mem.graph].references()[mem.member].target;
      if (ref.kind != graphir::DefinitionKind::ProtocolState ||
          ref.index >= module.compositionDefinitions().size()) {
        continue;
      }
      const auto &stDef = module.compositionDefinitions()[ref.index];
      ProtocolStateSpec pState;
      const auto sep = stDef.name.rfind("::");
      pState.name =
          sep == std::string::npos ? stDef.name : stDef.name.substr(sep + 2);
      pState.isFinal = stDef.attributes.contains("final") &&
                       stDef.attributes.at("final") == "true";
      pState.location = stDef.location;

      for (const auto &rec : stDef.records) {
        if (rec.kind != "message") continue;
        ProtocolMessageSpec m;
        m.name = rec.name;
        m.senderRole = rec.attributes.contains("sender")
                           ? rec.attributes.at("sender")
                           : "";
        m.receiverRole = rec.attributes.contains("receiver")
                             ? rec.attributes.at("receiver")
                             : "";
        m.successorState = rec.attributes.contains("successor")
                               ? rec.attributes.at("successor")
                               : "";
        m.location = rec.location;
        pState.messages.push_back(std::move(m));
      }
      spec.states.emplace(pState.name, std::move(pState));
    }
    out.emplace(spec.name, std::move(spec));
  }
  return out;
}

} // namespace

GPUBoundedQueueReport verifyGPUBoundedQueueCompatibility(
    const graphir::Module &module, const GPUBoundedQueueOptions &options) {
  GPUBoundedQueueReport report;
  const auto allProtocols = extractModuleProtocols(module);

  for (const auto &pipeline : module.pipelineGraphs()) {
    std::vector<const graphir::CompositionDefinition *> edgeDefs;
    std::vector<const graphir::CompositionDefinition *> topoDefs;
    std::vector<const graphir::CompositionDefinition *> schedDefs;
    std::vector<const graphir::StageGraph *> stageDefs;

    for (const auto &mem : pipeline.members) {
      if (mem.graph >= module.resourceGraphs().size() ||
          mem.member >= module.resourceGraphs()[mem.graph].references().size()) {
        continue;
      }
      const auto ref =
          module.resourceGraphs()[mem.graph].references()[mem.member].target;
      if (ref.kind == graphir::DefinitionKind::Stage &&
          ref.index < module.stageGraphs().size()) {
        stageDefs.push_back(&module.stageGraphs()[ref.index]);
      } else if (ref.kind == graphir::DefinitionKind::PipelineEdge &&
                 ref.index < module.compositionDefinitions().size()) {
        edgeDefs.push_back(&module.compositionDefinitions()[ref.index]);
      } else if (ref.kind == graphir::DefinitionKind::Topology &&
                 ref.index < module.compositionDefinitions().size()) {
        topoDefs.push_back(&module.compositionDefinitions()[ref.index]);
      } else if (ref.kind == graphir::DefinitionKind::Schedule &&
                 ref.index < module.compositionDefinitions().size()) {
        schedDefs.push_back(&module.compositionDefinitions()[ref.index]);
      }
    }

    // Determine whether this pipeline targets GPU execution (any GPU edge, GPU
    // schedule target `sm*`/`gfx*`, or GPU stage execution domain).
    bool isGPUPipeline = false;
    for (const auto *e : edgeDefs) {
      const auto stIt = e->attributes.find("storage");
      const auto scIt = e->attributes.find("scope");
      if ((stIt != e->attributes.end() &&
           startsWithPrefix(stIt->second, "gpu::")) ||
          (scIt != e->attributes.end() &&
           (scIt->second == "block" || scIt->second == "warp" ||
            scIt->second == "device" || scIt->second == "grid" ||
            scIt->second == "cluster"))) {
        isGPUPipeline = true;
      }
    }
    for (const auto *sc : schedDefs) {
      const auto tIt = sc->attributes.find("target");
      if (tIt != sc->attributes.end() &&
          (startsWithPrefix(tIt->second, "sm") ||
           startsWithPrefix(tIt->second, "gfx"))) {
        isGPUPipeline = true;
      }
    }
    for (const auto *st : stageDefs) {
      const auto dIt = st->attributes().find("execution_domain");
      if (dIt != st->attributes().end() && dIt->second == "Gpu")
        isGPUPipeline = true;
    }
    if (!isGPUPipeline) continue;

    GPUPipelineSystemSpec sys;
    sys.name = pipeline.name;
    sys.protocols = allProtocols;
    sys.location = pipeline.location;
    sys.scheduleCount = schedDefs.size();

    // Map stage parameter role (`StageName.port -> roleName`) for protocol checks.
    std::map<std::string, std::string> portRoleMap;
    for (const auto *st : stageDefs) {
      for (const auto &p : st->parameters()) {
        if (p.context) continue;
        const auto angle = p.type.find('<');
        const std::string base =
            angle == std::string::npos ? p.type : p.type.substr(0, angle);
        const auto sep = base.rfind("::");
        if (sep != std::string::npos) {
          portRoleMap[st->name() + "." + p.name] = base.substr(sep + 2);
        }
      }
    }

    // Extract topology bindings (`stage -> (localPort -> canonicalEdgeName)`
    // and edge producer/consumer stage + role).
    std::map<std::string, std::map<std::string, std::string>> stagePortMaps;
    std::map<std::string, std::string> edgeProducerStage, edgeConsumerStage;
    std::map<std::string, std::string> edgeProducerRole, edgeConsumerRole;

    auto splitEndpoint = [](const std::string &ep)
        -> std::pair<std::string, std::string> {
      const auto dot = ep.rfind('.');
      if (dot == std::string::npos) return {ep, ""};
      return {ep.substr(0, dot), ep.substr(dot + 1)};
    };

    for (const auto *topo : topoDefs) {
      for (const auto &conn : topo->records) {
        if (conn.kind != "connection" || !conn.attributes.contains("edge"))
          continue;
        const std::string edgeName = conn.attributes.at("edge");
        const std::string srcFull = conn.attributes.at("source");
        const std::string dstFull = conn.attributes.at("target");
        const auto [srcStage, srcPort] = splitEndpoint(srcFull);
        const auto [dstStage, dstPort] = splitEndpoint(dstFull);

        if (!srcPort.empty()) {
          stagePortMaps[srcStage][srcPort] = edgeName;
        }
        if (!dstPort.empty()) {
          stagePortMaps[dstStage][dstPort] = edgeName;
        }
        const auto shortSep = edgeName.rfind("::");
        if (shortSep != std::string::npos) {
          const std::string shortEdge = edgeName.substr(shortSep + 2);
          stagePortMaps[srcStage][shortEdge] = edgeName;
          stagePortMaps[dstStage][shortEdge] = edgeName;
        }

        edgeProducerStage[edgeName] = srcStage;
        if (portRoleMap.contains(srcFull))
          edgeProducerRole[edgeName] = portRoleMap[srcFull];

        if (edgeConsumerStage.contains(edgeName) &&
            edgeConsumerStage[edgeName] != dstStage) {
          sys.hasUnconnectedDuplicateConsumer = true;
          sys.duplicateConsumerEdge = edgeName;
          sys.duplicateConsumerLoc = conn.location;
        } else {
          edgeConsumerStage[edgeName] = dstStage;
          if (portRoleMap.contains(dstFull))
            edgeConsumerRole[edgeName] = portRoleMap[dstFull];
        }
      }
    }

    // Extract queue specs from `edgeDefs`.
    for (const auto *edge : edgeDefs) {
      GPUBoundedQueueSpec q;
      q.name = edge->name;
      q.location = edge->location;
      for (const auto &rec : edge->records) {
        if (rec.kind == "channel_protocol") q.protocolName = rec.name;
      }
      if (edge->attributes.contains("initial"))
        q.initialProtocolState = edge->attributes.at("initial");
      if (edge->attributes.contains("capacity")) {
        q.capacity = parseUnsignedText(edge->attributes.at("capacity"))
                         .value_or(0);
      }
      if (edge->attributes.contains("seeded_items")) {
        q.seededItems = parseUnsignedText(edge->attributes.at("seeded_items"))
                            .value_or(q.capacity + 1);
      }
      if (edge->attributes.contains("storage"))
        q.storage = edge->attributes.at("storage");
      if (edge->attributes.contains("scope"))
        q.scope = edge->attributes.at("scope");
      if (edge->attributes.contains("backpressure"))
        q.backpressure = edge->attributes.at("backpressure");

      const bool workgroupLocalScope =
          (q.scope == "block" || q.scope == "warp");
      q.residency = workgroupLocalScope
                        ? GPUResidencyContract::WorkgroupLocal
                        : (options.profile.allowCooperativeGridChannels
                               ? GPUResidencyContract::CooperativeGrid
                               : GPUResidencyContract::UnscheduledGrid);
      if (edge->attributes.contains("publication")) {
        q.explicitPublication = true;
        q.publication =
            parsePublicationOrder(edge->attributes.at("publication"))
                .value_or(GPUPublicationOrder::Relaxed);
      } else {
        q.publication = workgroupLocalScope
                            ? GPUPublicationOrder::AcqRelWorkgroup
                            : GPUPublicationOrder::AcqRelDevice;
      }

      if (edgeProducerStage.contains(edge->name))
        q.producerStage = edgeProducerStage[edge->name];
      if (edgeProducerRole.contains(edge->name))
        q.producerRole = edgeProducerRole[edge->name];
      if (edgeConsumerStage.contains(edge->name))
        q.consumerStage = edgeConsumerStage[edge->name];
      if (edgeConsumerRole.contains(edge->name))
        q.consumerRole = edgeConsumerRole[edge->name];

      sys.queues.push_back(std::move(q));
    }

    // Extract schedule placements.
    for (const auto *sched : schedDefs) {
      std::map<std::string, std::pair<std::size_t, std::string>> workerInfo;
      for (const auto &rec : sched->records) {
        if (rec.kind != "fused_kernel") continue;
        const std::size_t blocks =
            rec.attributes.contains("blocks")
                ? parseUnsignedText(rec.attributes.at("blocks")).value_or(0)
                : 1;
        const std::string resMode = rec.attributes.contains("residency")
                                        ? rec.attributes.at("residency")
                                        : "persistent";
        workerInfo[rec.name] = {blocks, resMode};
        for (const auto &stageRef : rec.references) {
          const std::string sName = module.definitionName(stageRef);
          GPUStagePlacementInfo pInfo;
          pInfo.stageName = sName;
          pInfo.workerName = rec.name;
          pInfo.workerBlocks = blocks;
          pInfo.workerResidency = resMode;
          pInfo.location = rec.location;
          sys.placements[sName] = std::move(pInfo);
        }
      }
      for (const auto &rec : sched->records) {
        if (rec.kind != "placement" || rec.references.empty()) continue;
        const std::string sName = module.definitionName(rec.references.front());
        auto &pInfo = sys.placements[sName];
        pInfo.stageName = sName;
        if (rec.attributes.contains("worker")) {
          pInfo.workerName = rec.attributes.at("worker");
          if (workerInfo.contains(pInfo.workerName)) {
            pInfo.workerBlocks = workerInfo[pInfo.workerName].first;
            pInfo.workerResidency = workerInfo[pInfo.workerName].second;
          }
        }
        pInfo.hasWarpPlacement = true;
        pInfo.firstWarp =
            rec.attributes.contains("index")
                ? parseUnsignedText(rec.attributes.at("index")).value_or(0)
                : 0;
        pInfo.lastWarp =
            rec.attributes.contains("end")
                ? parseUnsignedText(rec.attributes.at("end"))
                      .value_or(pInfo.firstWarp)
                : pInfo.firstWarp;
        pInfo.location = rec.location;
      }
    }

    // Extract stage behavioral automata (single walk via `buildControlFlow`).
    for (const auto *st : stageDefs) {
      const auto mapIt = stagePortMaps.find(st->name());
      const std::map<std::string, std::string> emptyMap;
      sys.stages.push_back(extractStageAutomaton(
          *st, mapIt != stagePortMaps.end() ? mapIt->second : emptyMap));
    }

    verifyGPUPipelineSystem(sys, options, report);
  }

  return report;
}

} // namespace metreon::verification
