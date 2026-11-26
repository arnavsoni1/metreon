#include "metreon/MLIR/GPUVerification.h"

#include "metreon/MLIR/Dialect.h"
#include "metreon/Verification/GPUOwnershipAndLoans.h"
#include "metreon/Verification/GPUProfile.h"

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefMemorySlot.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Mem2Reg.h"

#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace metreon {
namespace {

using verification::AsyncLoanMode;
using verification::GPUAddressSpace;
using verification::GPUExecutionScope;
using verification::GPULoopDomain;
using verification::ResourceLifecycleKind;

Operation *lookupSymbolInAncestors(Operation *from, SymbolRefAttr ref) {
  if (!ref) return nullptr;
  for (Operation *scope = from; scope != nullptr; scope = scope->getParentOp()) {
    if (scope->hasTrait<OpTrait::SymbolTable>()) {
      if (Operation *found = SymbolTable::lookupSymbolIn(scope, ref))
        return found;
    }
  }
  return nullptr;
}

ContextOp resolveContextOp(Operation *from, SymbolRefAttr ref) {
  return dyn_cast_or_null<ContextOp>(lookupSymbolInAncestors(from, ref));
}

bool contextHasGrant(ContextOp context, StringRef capability,
                     ContextOp target = {}) {
  if (!context) return false;
  for (Attribute item : context.getGrants()) {
    auto dict = dyn_cast<DictionaryAttr>(item);
    if (!dict) continue;
    auto nameAttr = dict.getAs<StringAttr>("name");
    auto argsAttr = dict.getAs<ArrayAttr>("arguments");
    if (!nameAttr || !argsAttr || nameAttr.getValue() != capability) continue;
    if (!target && argsAttr.empty()) return true;
    if (target && argsAttr.size() == 1) {
      auto argDict = dyn_cast<DictionaryAttr>(argsAttr[0]);
      auto argName = argDict ? argDict.getAs<StringAttr>("name") : StringAttr{};
      if (argName &&
          resolveContextOp(context, FlatSymbolRefAttr::get(
                                        context.getContext(), argName.getValue())) ==
              target) {
        return true;
      }
    }
  }
  return false;
}

std::optional<GPUAddressSpace> memrefAddressSpace(MemRefType type) {
  Attribute spaceAttr = type.getMemorySpace();
  if (!spaceAttr) return GPUAddressSpace::HostDefault;
  if (auto intAttr = dyn_cast<IntegerAttr>(spaceAttr)) {
    switch (intAttr.getInt()) {
    case 0: return GPUAddressSpace::HostDefault;
    case 1: return GPUAddressSpace::GlobalDevice;
    case 3: return GPUAddressSpace::WorkgroupShared;
    case 4: return GPUAddressSpace::Constant;
    case 5: return GPUAddressSpace::ThreadPrivate;
    default: return std::nullopt;
    }
  }
  return std::nullopt;
}

// Preserves potentially non-terminating `scf.while` loops while folding regionless
// arithmetic and applying SCF canonicalization patterns.
LogicalResult normalizeStructuredRegions(ModuleOp module) {
  MLIRContext *context = module.getContext();
  auto hoistConstants = [](Region &body) {
    if (body.empty()) return;
    SmallVector<Operation *> constants;
    body.walk([&](Operation *op) {
      if (isa<arith::ConstantOp, ub::PoisonOp>(op)) constants.push_back(op);
    });
    for (Operation *op : constants) {
      op->moveBefore(&body.front(), body.front().begin());
    }
  };
  module.walk([&](func::FuncOp fn) { hoistConstants(fn.getBody()); });
  module.walk([&](gpu::GPUFuncOp fn) { hoistConstants(fn.getBody()); });

  RewritePatternSet patterns(context);
  scf::WhileOp::getCanonicalizationPatterns(patterns, context);
  scf::IfOp::getCanonicalizationPatterns(patterns, context);
  for (RegisteredOperationName op : context->getRegisteredOperations()) {
    if (op.getDialectNamespace() == "arith")
      op.getCanonicalizationPatterns(patterns, context);
  }
  FrozenRewritePatternSet frozen(std::move(patterns));
  PatternApplicator applicator(frozen);
  applicator.applyDefaultCostModel();
  struct NormalizationRewriter : PatternRewriter {
    explicit NormalizationRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
  } rewriter(context);

  for (unsigned rewrites = 0; rewrites < 100000; ++rewrites) {
    OperationFolder folder(context);
    const WalkResult changed =
        module.walk<WalkOrder::PostOrder>([&](Operation *op) {
          if (!op->getNumRegions() && isOpTriviallyDead(op)) {
            rewriter.eraseOp(op);
            return WalkResult::interrupt();
          }
          if (!op->getNumRegions() && !op->hasTrait<OpTrait::ConstantLike>() &&
              succeeded(folder.tryToFold(op))) {
            return WalkResult::interrupt();
          }
          rewriter.setInsertionPoint(op);
          if (succeeded(applicator.matchAndRewrite(op, rewriter)))
            return WalkResult::interrupt();
          return WalkResult::advance();
        });
    if (!changed.wasInterrupted()) return success();
  }
  return module.emitError("GPU structured CFG normalization did not converge");
}

void promoteCountedLoop(scf::WhileOp loop) {
  Block &before = loop.getBefore().front();
  Block &after = loop.getAfter().front();
  if (before.getOperations().size() != 3 || after.getOperations().size() != 1)
    return;
  auto compare = dyn_cast<arith::CmpIOp>(before.front());
  auto branch = dyn_cast<scf::IfOp>(*std::next(before.begin()));
  auto condition = dyn_cast<scf::ConditionOp>(before.back());
  auto repeat = dyn_cast<scf::YieldOp>(after.back());
  if (!compare || !branch || !condition || !repeat ||
      compare.getPredicate() != arith::CmpIPredicate::slt ||
      branch.getCondition() != compare.getResult() ||
      condition.getCondition() != compare.getResult() ||
      branch.getElseRegion().empty()) {
    return;
  }
  if (branch.getElseRegion().front().getOperations().size() != 1) return;
  auto induction = dyn_cast<BlockArgument>(compare.getLhs());
  if (!induction || induction.getOwner() != &before ||
      !loop.isDefinedOutsideOfLoop(compare.getRhs())) {
    return;
  }
  const unsigned iv = induction.getArgNumber();
  auto thenYield =
      cast<scf::YieldOp>(branch.getThenRegion().front().getTerminator());
  SmallVector<Value> next;
  for (Value value : repeat.getOperands()) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &after) return;
    auto result =
        dyn_cast<OpResult>(condition.getArgs()[argument.getArgNumber()]);
    if (!result || result.getOwner() != branch.getOperation()) return;
    next.push_back(thenYield.getOperand(result.getResultNumber()));
  }
  if (next.size() != before.getNumArguments()) return;
  auto increment = next[iv].getDefiningOp<arith::AddIOp>();
  if (!increment || increment.getLhs() != induction ||
      !matchPattern(increment.getRhs(), m_One())) {
    return;
  }
  for (auto [result, value] : llvm::zip(loop.getResults(), condition.getArgs())) {
    if (result.use_empty()) continue;
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &before || argument == induction)
      return;
  }
  OpBuilder builder(loop);
  SmallVector<Value> inits;
  for (auto [index, value] : llvm::enumerate(loop.getInits())) {
    if (index != iv) inits.push_back(value);
  }
  auto counted = builder.create<scf::ForOp>(
      loop.getLoc(), loop.getInits()[iv], compare.getRhs(), increment.getRhs(),
      inits,
      [&](OpBuilder &body, Location loc, Value newIV, ValueRange carried) {
        IRMapping mapping;
        unsigned carriedIndex = 0;
        for (BlockArgument argument : before.getArguments()) {
          mapping.map(argument,
                      argument == induction ? newIV : carried[carriedIndex++]);
        }
        mapping.map(compare.getResult(),
                    body.create<arith::ConstantIntOp>(loc, 1, 1));
        for (Operation &op : branch.getThenRegion().front().without_terminator())
          body.clone(op, mapping);
        SmallVector<Value> yielded;
        for (auto [index, value] : llvm::enumerate(next)) {
          if (index != iv) yielded.push_back(mapping.lookupOrDefault(value));
        }
        body.create<scf::YieldOp>(loc, yielded);
      });
  for (auto [result, value] : llvm::zip(loop.getResults(), condition.getArgs())) {
    if (result.use_empty()) continue;
    const unsigned index = cast<BlockArgument>(value).getArgNumber();
    result.replaceAllUsesWith(
        counted.getResult(index < iv ? index : index - 1));
  }
  loop.erase();
}

bool promoteAffineLoop(scf::ForOp loop) {
  if (!matchPattern(loop.getStep(), m_One())) return false;
  Type type = loop.getInductionVar().getType();
  if (!type.isIndex() &&
      (!isa<IntegerType>(type) || cast<IntegerType>(type).getWidth() > 32)) {
    return false;
  }
  OpBuilder builder(loop);
  SmallVector<Operation *> casts;
  auto indexValue = [&](Value value) -> Value {
    if (value.getType().isIndex()) return value;
    IntegerAttr constant;
    if (matchPattern(value, m_Constant(&constant))) {
      auto op = builder.create<arith::ConstantIndexOp>(
          loop.getLoc(), constant.getValue().getSExtValue());
      casts.push_back(op);
      return op;
    }
    auto op = builder.create<arith::IndexCastOp>(
        loop.getLoc(), builder.getIndexType(), value);
    casts.push_back(op);
    return op;
  };
  Value lower = indexValue(loop.getLowerBound());
  Value upper = indexValue(loop.getUpperBound());
  Region *scope = affine::getAffineScope(loop);
  if (!affine::isValidDim(lower, scope) || !affine::isValidDim(upper, scope)) {
    for (Operation *op : llvm::reverse(casts)) op->erase();
    return false;
  }
  AffineMap identity = AffineMap::get(1, 0, builder.getAffineDimExpr(0));
  AffineMap lowerMap = identity;
  AffineMap upperMap = identity;
  SmallVector<Value> lowerOperands{lower};
  SmallVector<Value> upperOperands{upper};
  IntegerAttr constant;
  if (matchPattern(lower, m_Constant(&constant))) {
    lowerMap = AffineMap::getConstantMap(constant.getValue().getSExtValue(),
                                         loop.getContext());
    lowerOperands.clear();
  }
  if (matchPattern(upper, m_Constant(&constant))) {
    upperMap = AffineMap::getConstantMap(constant.getValue().getSExtValue(),
                                         loop.getContext());
    upperOperands.clear();
  }
  auto affineLoop = builder.create<affine::AffineForOp>(
      loop.getLoc(), lowerOperands, lowerMap, upperOperands, upperMap, 1,
      loop.getInitArgs(),
      [&](OpBuilder &body, Location loc, Value induction, ValueRange carried) {
        IRMapping mapping;
        Value sourceIV = induction;
        if (!type.isIndex())
          sourceIV = body.create<arith::IndexCastOp>(loc, type, induction);
        mapping.map(loop.getInductionVar(), sourceIV);
        mapping.map(loop.getRegionIterArgs(), carried);
        for (Operation &op : loop.getBody()->without_terminator())
          body.clone(op, mapping);
        SmallVector<Value> values;
        for (Value value :
             cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperands()) {
          values.push_back(mapping.lookupOrDefault(value));
        }
        body.create<affine::AffineYieldOp>(loc, values);
      });
  loop.replaceAllUsesWith(affineLoop.getResults());
  loop.erase();
  return true;
}

void promoteAffineConditional(scf::IfOp branch) {
  auto compare = branch.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!compare || !compare.getLhs().getType().isIndex()) return;
  Region *scope = affine::getAffineScope(branch);
  if (!affine::isValidDim(compare.getLhs(), scope) ||
      !affine::isValidDim(compare.getRhs(), scope)) {
    return;
  }
  auto nonnegative = [](Value value) {
    IntegerAttr constant;
    if (matchPattern(value, m_Constant(&constant)))
      return !constant.getValue().isNegative();
    auto owner = affine::getForInductionVarOwner(value);
    return owner && owner.hasConstantLowerBound() &&
           owner.getConstantLowerBound() >= 0;
  };
  if (!nonnegative(compare.getLhs()) || !nonnegative(compare.getRhs())) return;
  OpBuilder builder(branch);
  AffineExpr left = builder.getAffineDimExpr(0);
  AffineExpr right = builder.getAffineDimExpr(1);
  AffineExpr expression;
  bool equal = false;
  switch (compare.getPredicate()) {
  case arith::CmpIPredicate::eq:
    expression = left - right;
    equal = true;
    break;
  case arith::CmpIPredicate::slt:
    expression = right - left - 1;
    break;
  case arith::CmpIPredicate::sle:
    expression = right - left;
    break;
  case arith::CmpIPredicate::sgt:
    expression = left - right - 1;
    break;
  case arith::CmpIPredicate::sge:
    expression = left - right;
    break;
  default:
    return;
  }
  IntegerSet set = IntegerSet::get(2, 0, {expression}, {equal});
  auto affineIf = builder.create<affine::AffineIfOp>(
      branch.getLoc(), branch.getResultTypes(), set,
      ValueRange{compare.getLhs(), compare.getRhs()},
      !branch.getElseRegion().empty());
  auto transfer = [&](Region &from, Region &to) {
    if (from.empty()) return;
    to.takeBody(from);
    auto yield = cast<scf::YieldOp>(to.front().getTerminator());
    OpBuilder atYield(yield);
    atYield.create<affine::AffineYieldOp>(yield.getLoc(), yield.getOperands());
    yield.erase();
  };
  transfer(branch.getThenRegion(), affineIf.getThenRegion());
  transfer(branch.getElseRegion(), affineIf.getElseRegion());
  branch.replaceAllUsesWith(affineIf.getResults());
  branch.erase();
}

// === Phase GPU-2 MLIR Verifier State ===

enum class MLIRHandleState {
  Live,
  Consumed,
  MaybeConsumed,
};

struct MLIRHandleInfo {
  std::string stateSymbol;
  ResourceLifecycleKind lifecycle = ResourceLifecycleKind::Droppable;
  MLIRHandleState state = MLIRHandleState::Live;
  bool isDynamicSpan = false;
};

enum class MLIRGenStatus {
  InFlight,
  Completed,
};

using SlotGenPair = std::pair<std::string, std::string>;

struct MLIRLoanEntry {
  Value allocation;
  std::string slot;
  std::string generation;
  AsyncLoanMode mode = AsyncLoanMode::ExclusiveWrite;
};

struct MLIRPathState {
  bool reachable = true;
  llvm::DenseMap<Value, MLIRHandleInfo> handles;
  std::vector<MLIRLoanEntry> activeLoans;
  std::map<SlotGenPair, MLIRGenStatus> generations;
};

class MLIRGPUVerifier {
public:
  explicit MLIRGPUVerifier(ModuleOp root) : root_(root) {}

  LogicalResult run() {
    indexResourceLifecycles();
    indexCallableContracts();
    verifyContextDeclarations();

    root_.walk([&](func::FuncOp fn) {
      if (!fn.isExternal()) verifyCallableBody(fn, fn.getBody());
    });
    root_.walk([&](gpu::GPUFuncOp fn) {
      if (!fn.isDeclaration()) verifyCallableBody(fn, fn.getBody());
    });

    return success(valid_);
  }

private:
  ModuleOp root_;
  bool valid_ = true;
  llvm::DenseMap<Operation *, ResourceLifecycleKind> stateLifecycles_;
  llvm::DenseMap<Operation *, Operation *> bodyToContract_;
  std::map<std::string, Operation *> namedContracts_;

  void reject(Operation *op, const Twine &message) {
    op->emitError(message);
    valid_ = false;
  }

  void indexResourceLifecycles() {
    root_.walk([&](ResourceOp resource) {
      llvm::DenseMap<Operation *, bool> hasAwaitOut;
      for (Operation &member : resource.getBody().front()) {
        if (auto trans = dyn_cast<TransitionDeclOp>(member)) {
          if (trans.getIsAwait()) {
            if (Operation *src = SymbolTable::lookupSymbolIn(
                    resource, trans.getSourceStateAttr())) {
              hasAwaitOut[src] = true;
            }
          }
        }
      }
      for (Operation &member : resource.getBody().front()) {
        auto state = dyn_cast<StateOp>(member);
        if (!state) continue;
        bool hasEventField = false;
        for (Attribute item : state.getFields()) {
          auto dict = dyn_cast<DictionaryAttr>(item);
          auto srcType = dict ? dict.getAs<DictionaryAttr>("source_type")
                              : DictionaryAttr{};
          auto srcName =
              srcType ? srcType.getAs<StringAttr>("name") : StringAttr{};
          if (srcName && srcName.getValue() == "event") hasEventField = true;
        }
        const StringRef policy = state.getLifecycle();
        ResourceLifecycleKind kind = ResourceLifecycleKind::LinearOwned;
        if (policy == "droppable" || policy == "terminal") {
          kind = ResourceLifecycleKind::Droppable;
        } else if (policy == "must_complete" || hasEventField ||
                   hasAwaitOut.lookup(state)) {
          kind = ResourceLifecycleKind::MustComplete;
        }
        stateLifecycles_[state] = kind;
      }
    });
  }

  void indexCallableContracts() {
    root_.walk([&](Operation *op) {
      if (!isa<KernelOp, ProcedureOp, StageOp>(op)) return;
      auto symName = op->getAttrOfType<StringAttr>("sym_name");
      if (symName) namedContracts_[symName.str()] = op;
      if (auto bodyRef = op->getAttrOfType<SymbolRefAttr>("body_ref")) {
        if (Operation *body = lookupSymbolInAncestors(op, bodyRef))
          bodyToContract_[body] = op;
      }
      if (auto devBody = op->getAttrOfType<SymbolRefAttr>("metreon.device_body")) {
        if (Operation *body = lookupSymbolInAncestors(op, devBody))
          bodyToContract_[body] = op;
      }
    });
  }

  void verifyContextDeclarations() {
    root_.walk([&](ContextOp ctx) {
      auto nomNameAttr = ctx.getNominalType().getAs<StringAttr>("name");
      if (!nomNameAttr) return;
      const StringRef nomName = nomNameAttr.getValue();
      if (!nomName.starts_with("Gpu::") && !nomName.starts_with("Host::"))
        return;

      if (const auto loopDom = verification::classifyLoopContextDomain(nomName)) {
        (void)loopDom;
        if (!ctx.getGrants().empty() || !ctx.getGenericParameters().empty()) {
          reject(ctx, "loop evidence context cannot declare grants or generic parameters");
        }
        return;
      }

      const auto scope = verification::classifyContextExecutionScope(nomName);
      if (!scope) {
        if (ctx.getResolution() == "declared") {
          reject(ctx, "unsupported GPU/Host execution context scope `" +
                          nomName + "`");
        }
        return;
      }

      for (Attribute grantAttr : ctx.getGrants()) {
        auto dict = dyn_cast<DictionaryAttr>(grantAttr);
        auto capName = dict ? dict.getAs<StringAttr>("name") : StringAttr{};
        if (!capName) continue;
        if (!verification::isCapabilityAllowedInScope(*scope,
                                                      capName.getValue())) {
          reject(ctx, "capability `" + capName.getValue() +
                          "` is forbidden in `" +
                          verification::gpuExecutionScopeName(*scope) +
                          "` execution scope (`" + nomName + "`)");
        }
      }
    });
  }

  std::optional<MLIRHandleInfo> classifyValueHandle(Operation *scopeAnchor,
                                                    Type type,
                                                    bool explicitOwnAttr = false,
                                                    bool dynamicSpan = false) {
    if (auto resType = dyn_cast<ResourceType>(type)) {
      MLIRHandleInfo info;
      std::string stateStr;
      llvm::raw_string_ostream os(stateStr);
      resType.getState().print(os);
      info.stateSymbol = os.str();
      info.lifecycle = ResourceLifecycleKind::LinearOwned;
      if (Operation *stateOp =
              lookupSymbolInAncestors(scopeAnchor, resType.getState())) {
        if (stateLifecycles_.contains(stateOp))
          info.lifecycle = stateLifecycles_[stateOp];
      }
      return info;
    }
    if (auto ownedType = dyn_cast<OwnedType>(type)) {
      MLIRHandleInfo info;
      info.stateSymbol = "owned";
      info.lifecycle = ResourceLifecycleKind::LinearOwned;
      if (auto unres = dyn_cast<UnresolvedType>(ownedType.getValueType())) {
        auto name = unres.getSourceType().getAs<StringAttr>("name");
        if (name && name.getValue() == "event")
          info.lifecycle = ResourceLifecycleKind::MustComplete;
      }
      return info;
    }
    if (explicitOwnAttr) {
      MLIRHandleInfo info;
      info.stateSymbol = "owned_param";
      info.lifecycle = ResourceLifecycleKind::LinearOwned;
      return info;
    }
    if (dynamicSpan) {
      MLIRHandleInfo info;
      info.stateSymbol = "dynamic_span";
      info.lifecycle = ResourceLifecycleKind::Droppable;
      info.isDynamicSpan = true;
      return info;
    }
    return std::nullopt;
  }

  void requireLive(Operation *user, const MLIRPathState &state, Value val,
                   StringRef action) {
    auto it = state.handles.find(val);
    if (it == state.handles.end()) return;
    if (it->second.state == MLIRHandleState::Consumed) {
      reject(user, "double consumption: cannot " + action +
                       " handle after it was already consumed");
    } else if (it->second.state == MLIRHandleState::MaybeConsumed) {
      reject(user, "double consumption: cannot " + action +
                       " handle after partial consumption on a branch");
    }
  }

  void consumeValue(Operation *user, MLIRPathState &state, Value val) {
    auto it = state.handles.find(val);
    if (it == state.handles.end()) return;
    requireLive(user, state, val, "consume");
    it->second.state = MLIRHandleState::Consumed;
  }

  void checkLoanAccess(Operation *user, const MLIRPathState &state, Value buf,
                       bool isWrite, bool isFree) {
    requireLive(user, state, buf, isFree ? "free" : (isWrite ? "write" : "read"));
    for (const MLIRLoanEntry &loan : state.activeLoans) {
      if (loan.allocation != buf) continue;
      if (isFree || isWrite || loan.mode == AsyncLoanMode::ExclusiveWrite) {
        reject(user, Twine("pre-wait buffer access on in-flight async transfer (") +
                         loan.slot + ", gen=" + loan.generation + ")");
        return;
      }
    }
  }

  void checkMemRefSpace(Operation *op, MemRefType memType,
                        std::optional<GPUExecutionScope> scope,
                        ContextOp ambient, bool isStore, Value memVal) {
    const auto space = memrefAddressSpace(memType);
    if (!space) {
      reject(op, "unsupported memref address space in GPU/Host verification");
      return;
    }
    if (!scope) return;
    if (*scope == GPUExecutionScope::Host) {
      if (*space != GPUAddressSpace::HostDefault) {
        reject(op, "cross-space dereference: host code cannot dereference a GPU address");
      }
      return;
    }
    // Device scopes: allow rank-0 entry-block scalar `memref.alloca` slots
    // prior to target private-space rewriting, but reject rank >= 1 host-space
    // buffers.
    const bool localScalarSlot =
        memType.getRank() == 0 &&
        isa_and_nonnull<memref::AllocaOp>(memVal.getDefiningOp());
    if (!localScalarSlot &&
        !verification::canDereferenceAddressSpace(*scope, *space)) {
      reject(op, "cross-space dereference: device code cannot dereference host default address space");
      return;
    }
    if (*space == GPUAddressSpace::WorkgroupShared &&
        *scope == GPUExecutionScope::Thread && ambient &&
        !contextHasGrant(ambient, isStore ? "gpu_store" : "gpu_load")) {
      reject(op, "device shared memory access lacks required memory grant");
    }
  }

  MLIRPathState mergePathStates(Operation *anchor,
                                const std::vector<MLIRPathState> &paths,
                                const MLIRPathState *preBranch) {
    if (paths.empty()) {
      MLIRPathState dead;
      dead.reachable = false;
      return dead;
    }
    if (paths.size() == 1) return paths.front();

    MLIRPathState merged = paths.front();
    llvm::DenseSet<Value> allValues;
    for (const MLIRPathState &p : paths) {
      for (const auto &kv : p.handles) allValues.insert(kv.first);
    }

    for (Value val : allValues) {
      bool anyLive = false;
      bool anyConsumed = false;
      MLIRHandleInfo sample;
      bool haveSample = false;
      std::set<std::string> stateNames;

      for (const MLIRPathState &p : paths) {
        auto it = p.handles.find(val);
        if (it == p.handles.end()) continue;
        if (!haveSample) {
          sample = it->second;
          haveSample = true;
        }
        if (it->second.state == MLIRHandleState::Live) {
          anyLive = true;
          stateNames.insert(it->second.stateSymbol);
        } else {
          anyConsumed = true;
        }
      }
      if (!haveSample) continue;

      const bool presentInAll = llvm::all_of(
          paths, [&](const MLIRPathState &p) { return p.handles.contains(val); });
      const bool existedBefore =
          preBranch ? preBranch->handles.contains(val) : presentInAll;
      if (!existedBefore) {
        if (!presentInAll && anyLive &&
            sample.lifecycle != ResourceLifecycleKind::Droppable) {
          reject(anchor, "branch lifetime leak: owned handle escapes branch without completion");
        }
        merged.handles.erase(val);
        continue;
      }

      if (anyLive && anyConsumed) {
        if (sample.lifecycle == ResourceLifecycleKind::MustComplete) {
          reject(anchor, "branch lifetime leak: must_complete resource consumed on only one branch");
        } else if (sample.lifecycle == ResourceLifecycleKind::LinearOwned) {
          reject(anchor, "branch lifetime leak: owned handle consumed on only one branch");
        }
        sample.state = MLIRHandleState::MaybeConsumed;
      } else if (anyLive) {
        if (stateNames.size() > 1) {
          reject(anchor, "branch typestate mismatch on owned resource handle");
        }
        sample.state = MLIRHandleState::Live;
      } else {
        sample.state = MLIRHandleState::Consumed;
      }
      merged.handles[val] = sample;
    }

    std::set<SlotGenPair> genKeys;
    for (const MLIRPathState &p : paths) {
      for (const auto &[k, _] : p.generations) genKeys.insert(k);
    }
    merged.generations.clear();
    for (const SlotGenPair &k : genKeys) {
      bool anyInFlight = false;
      bool anyCompleted = false;
      bool missing = false;
      for (const MLIRPathState &p : paths) {
        auto it = p.generations.find(k);
        if (it == p.generations.end()) missing = true;
        else if (it->second == MLIRGenStatus::InFlight) anyInFlight = true;
        else anyCompleted = true;
      }
      if (anyInFlight && (anyCompleted || missing)) {
        reject(anchor, Twine("branch lifetime leak: async token (") + k.first +
                           ", gen=" + k.second +
                           ") awaited on only a subset of continuing branches");
        merged.generations[k] = MLIRGenStatus::InFlight;
      } else if (anyInFlight) {
        merged.generations[k] = MLIRGenStatus::InFlight;
      } else if (anyCompleted) {
        merged.generations[k] = MLIRGenStatus::Completed;
      }
    }

    merged.activeLoans.clear();
    for (const MLIRPathState &p : paths) {
      for (const MLIRLoanEntry &loan : p.activeLoans) {
        auto existing = llvm::find_if(
            merged.activeLoans, [&](const MLIRLoanEntry &item) {
              return item.allocation == loan.allocation &&
                     item.slot == loan.slot &&
                     item.generation == loan.generation;
            });
        if (existing == merged.activeLoans.end()) {
          merged.activeLoans.push_back(loan);
        } else if (loan.mode == AsyncLoanMode::ExclusiveWrite) {
          existing->mode = AsyncLoanMode::ExclusiveWrite;
        }
      }
    }

    return merged;
  }

  void checkLoopBackedgeState(Operation *loopOp, const MLIRPathState &entryState,
                              const MLIRPathState &latchState) {
    for (const auto &[val, preInfo] : entryState.handles) {
      if (preInfo.state != MLIRHandleState::Live) continue;
      auto postIt = latchState.handles.find(val);
      if (postIt == latchState.handles.end() ||
          postIt->second.state != MLIRHandleState::Live ||
          postIt->second.stateSymbol != preInfo.stateSymbol) {
        reject(loopOp, "loop backedge lifetime leak: outer owned handle consumed or altered inside loop");
      }
    }
    for (const auto &[val, postInfo] : latchState.handles) {
      if (entryState.handles.contains(val)) continue;
      if (postInfo.state == MLIRHandleState::Live &&
          postInfo.lifecycle != ResourceLifecycleKind::Droppable) {
        if (postInfo.lifecycle == ResourceLifecycleKind::MustComplete) {
          reject(loopOp, "unawaited must_complete resource leaked across loop backedge");
        } else {
          reject(loopOp, "loop backedge lifetime leak: unconsumed owned handle at iteration end");
        }
      }
    }
    for (const auto &[sgKey, postStatus] : latchState.generations) {
      auto preIt = entryState.generations.find(sgKey);
      if (preIt == entryState.generations.end()) {
        if (postStatus == MLIRGenStatus::InFlight) {
          reject(loopOp, Twine("loop backedge lifetime leak: async token (") +
                             sgKey.first + ", gen=" + sgKey.second +
                             ") remains unawaited across loop backedge");
        }
      } else if (preIt->second != postStatus) {
        reject(loopOp, Twine("loop backedge lifetime leak: outer async token (") +
                           sgKey.first + ", gen=" + sgKey.second +
                           ") state mutated inside loop");
      }
    }
  }

  void checkReturnObligations(Operation *retOp, MLIRPathState &state,
                              ValueRange returnedValues) {
    llvm::DenseSet<Value> returnedSet;
    for (Value v : returnedValues) {
      checkLoanAccess(retOp, state, v, /*isWrite=*/false, /*isFree=*/false);
      consumeValue(retOp, state, v);
      returnedSet.insert(v);
    }
    for (const auto &[sgKey, status] : state.generations) {
      if (status == MLIRGenStatus::InFlight) {
        reject(retOp, Twine("unawaited must_complete async transfer (") +
                          sgKey.first + ", gen=" + sgKey.second +
                          ") at callable exit");
      }
    }
    for (const auto &[val, info] : state.handles) {
      if (returnedSet.contains(val) || info.state != MLIRHandleState::Live)
        continue;
      if (info.lifecycle == ResourceLifecycleKind::MustComplete) {
        reject(retOp, "unawaited must_complete resource `" + info.stateSymbol +
                          "` at callable exit");
      } else if (info.lifecycle == ResourceLifecycleKind::LinearOwned) {
        reject(retOp, "lifetime leak: owned handle `" + info.stateSymbol +
                          "` is neither consumed nor returned");
      }
    }
    state.reachable = false;
  }

  void verifyCallableBody(Operation *callableOp, Region &body) {
    Operation *contract = bodyToContract_.lookup(callableOp);
    ContextOp ambient;
    std::optional<GPUExecutionScope> scope;

    if (contract) {
      ambient = resolveContextOp(
          contract, contract->getAttrOfType<SymbolRefAttr>("context"));
      auto domAttr = contract->getAttrOfType<StringAttr>("execution_domain");
      if (domAttr && domAttr.getValue() == "Gpu" && !ambient &&
          isa<gpu::GPUFuncOp>(callableOp)) {
        scope = GPUExecutionScope::Thread;
      }
    } else if (auto ctxRef =
                   callableOp->getAttrOfType<SymbolRefAttr>("metreon.context")) {
      ambient = resolveContextOp(callableOp, ctxRef);
    }

    if (ambient) {
      auto nomName = ambient.getNominalType().getAs<StringAttr>("name");
      if (nomName) {
        scope = verification::classifyContextExecutionScope(nomName.getValue());
      }
    }

    MLIRPathState entryState;
    ArrayAttr paramAttrs =
        contract ? contract->getAttrOfType<ArrayAttr>("parameter_attrs")
                 : ArrayAttr{};
    for (auto [idx, arg] : llvm::enumerate(body.front().getArguments())) {
      bool explicitOwn = false;
      if (paramAttrs && idx < paramAttrs.size()) {
        if (auto dict = dyn_cast<DictionaryAttr>(paramAttrs[idx])) {
          if (auto ownAttr = dict.getAs<StringAttr>("ownership"))
            explicitOwn = (ownAttr.getValue() == "own");
        }
      }
      if (auto info = classifyValueHandle(callableOp, arg.getType(), explicitOwn)) {
        entryState.handles[arg] = *info;
      }
    }

    analyzeRegion(callableOp, body, ambient, scope, entryState);
  }

  void analyzeRegion(Operation *callableOp, Region &region, ContextOp ambient,
                     std::optional<GPUExecutionScope> scope,
                     MLIRPathState &state) {
    if (region.empty()) return;
    if (region.hasOneBlock()) {
      analyzeBlockOps(callableOp, region.front(), ambient, scope, state);
      return;
    }

    // Multi-block CFG analysis with DFS backedge detection and reverse post-order
    // propagation.
    llvm::DenseSet<Block *> visited;
    llvm::DenseSet<Block *> activeStack;
    std::set<std::pair<Block *, Block *>> backedges;
    std::vector<Block *> postOrder;

    std::function<void(Block *)> dfs = [&](Block *block) {
      visited.insert(block);
      activeStack.insert(block);
      if (Operation *term = block->getTerminator()) {
        for (Block *succ : term->getSuccessors()) {
          if (activeStack.contains(succ)) {
            backedges.emplace(block, succ);
          } else if (!visited.contains(succ)) {
            dfs(succ);
          }
        }
      }
      activeStack.erase(block);
      postOrder.push_back(block);
    };
    dfs(&region.front());
    std::reverse(postOrder.begin(), postOrder.end());

    llvm::DenseMap<Block *, std::vector<MLIRPathState>> incoming;
    llvm::DenseMap<Block *, MLIRPathState> blockEntryStates;
    incoming[&region.front()].push_back(state);

    auto passBranchOperands = [&](Operation *term, OperandRange ops,
                                  Block *dest, const MLIRPathState &srcState) {
      MLIRPathState out = srcState;
      for (auto [val, arg] : llvm::zip(ops, dest->getArguments())) {
        if (out.handles.contains(val)) {
          MLIRHandleInfo info = out.handles[val];
          consumeValue(term, out, val);
          info.state = MLIRHandleState::Live;
          out.handles[arg] = info;
        }
      }
      return out;
    };

    for (Block *block : postOrder) {
      auto incIt = incoming.find(block);
      if (incIt == incoming.end() || incIt->second.empty()) continue;
      MLIRPathState curState = mergePathStates(
          &block->front(), incIt->second, nullptr);
      if (!curState.reachable) continue;
      blockEntryStates[block] = curState;

      analyzeBlockOps(callableOp, *block, ambient, scope, curState);
      if (!curState.reachable) continue;

      Operation *term = block->getTerminator();
      if (auto br = dyn_cast<cf::BranchOp>(term)) {
        Block *dest = br.getDest();
        MLIRPathState edgeState =
            passBranchOperands(br, br.getDestOperands(), dest, curState);
        if (backedges.contains({block, dest})) {
          if (blockEntryStates.contains(dest))
            checkLoopBackedgeState(br, blockEntryStates[dest], edgeState);
        } else {
          incoming[dest].push_back(std::move(edgeState));
        }
      } else if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
        Block *trueDest = condBr.getTrueDest();
        MLIRPathState trueState = passBranchOperands(
            condBr, condBr.getTrueDestOperands(), trueDest, curState);
        if (backedges.contains({block, trueDest})) {
          if (blockEntryStates.contains(trueDest))
            checkLoopBackedgeState(condBr, blockEntryStates[trueDest], trueState);
        } else {
          incoming[trueDest].push_back(std::move(trueState));
        }

        Block *falseDest = condBr.getFalseDest();
        MLIRPathState falseState = passBranchOperands(
            condBr, condBr.getFalseDestOperands(), falseDest, curState);
        if (backedges.contains({block, falseDest})) {
          if (blockEntryStates.contains(falseDest))
            checkLoopBackedgeState(condBr, blockEntryStates[falseDest],
                                   falseState);
        } else {
          incoming[falseDest].push_back(std::move(falseState));
        }
      }
    }
    state.reachable = false;
  }

  void analyzeBlockOps(Operation *callableOp, Block &block, ContextOp ambient,
                       std::optional<GPUExecutionScope> scope,
                       MLIRPathState &state) {
    for (Operation &op : block) {
      if (!state.reachable) break;
      analyzeOp(callableOp, &op, ambient, scope, state);
    }
  }

  void consumeYieldedResults(Operation *branchOp, Region &branchRegion,
                             MLIRPathState &branchState,
                             ValueRange parentResults) {
    if (branchRegion.empty() || !branchState.reachable) return;
    Operation *term = branchRegion.front().getTerminator();
    OperandRange yielded = term->getOperands();
    for (auto [operand, res] : llvm::zip(yielded, parentResults)) {
      if (branchState.handles.contains(operand)) {
        MLIRHandleInfo info = branchState.handles[operand];
        consumeValue(term, branchState, operand);
        info.state = MLIRHandleState::Live;
        branchState.handles[res] = info;
      } else if (auto info = classifyValueHandle(branchOp, res.getType())) {
        branchState.handles[res] = *info;
      }
    }
  }

  template <typename ForLikeOp>
  void analyzeCountedLoop(Operation *callableOp, ForLikeOp loopOp,
                          ValueRange initArgs, Block::BlockArgListType iterArgs,
                          Region &bodyRegion, ContextOp ambient,
                          std::optional<GPUExecutionScope> scope,
                          MLIRPathState &state) {
    MLIRPathState loopEntry = state;
    std::vector<MLIRHandleInfo> carriedExpected;
    carriedExpected.reserve(initArgs.size());

    for (auto [initVal, iterArg] : llvm::zip(initArgs, iterArgs)) {
      checkLoanAccess(loopOp, loopEntry, initVal, /*isWrite=*/false,
                      /*isFree=*/false);
      if (loopEntry.handles.contains(initVal)) {
        MLIRHandleInfo info = loopEntry.handles[initVal];
        consumeValue(loopOp, loopEntry, initVal);
        info.state = MLIRHandleState::Live;
        loopEntry.handles[iterArg] = info;
        carriedExpected.push_back(info);
      } else if (auto info = classifyValueHandle(loopOp, iterArg.getType())) {
        loopEntry.handles[iterArg] = *info;
        carriedExpected.push_back(*info);
      } else {
        carriedExpected.push_back(MLIRHandleInfo{});
      }
    }

    const MLIRPathState headerSnapshot = loopEntry;
    MLIRPathState latchState = headerSnapshot;
    analyzeRegion(callableOp, bodyRegion, ambient, scope, latchState);

    if (latchState.reachable && !bodyRegion.empty()) {
      Operation *yieldOp = bodyRegion.front().getTerminator();
      for (auto [idx, yieldedVal] : llvm::enumerate(yieldOp->getOperands())) {
        BlockArgument iterArg = iterArgs[idx];
        if (!headerSnapshot.handles.contains(iterArg)) continue;
        checkLoanAccess(yieldOp, latchState, yieldedVal, /*isWrite=*/false,
                        /*isFree=*/false);
        auto yIt = latchState.handles.find(yieldedVal);
        if (yIt == latchState.handles.end() ||
            yIt->second.state != MLIRHandleState::Live ||
            yIt->second.stateSymbol != carriedExpected[idx].stateSymbol) {
          reject(yieldOp, "loop backedge changes owned resource typestate or yields consumed value");
        } else {
          consumeValue(yieldOp, latchState, yieldedVal);
          MLIRHandleInfo restored = carriedExpected[idx];
          restored.state = MLIRHandleState::Live;
          latchState.handles[iterArg] = restored;
        }
      }
      checkLoopBackedgeState(loopOp, headerSnapshot, latchState);
    }

    // Post-loop state: consumed `initArgs` transfer ownership to `loopOp` results.
    state = loopEntry;
    for (BlockArgument iterArg : iterArgs) state.handles.erase(iterArg);
    for (auto [idx, res] : llvm::enumerate(loopOp.getResults())) {
      if (!carriedExpected[idx].stateSymbol.empty()) {
        MLIRHandleInfo outInfo = carriedExpected[idx];
        outInfo.state = MLIRHandleState::Live;
        state.handles[res] = outInfo;
      } else if (auto info = classifyValueHandle(loopOp, res.getType())) {
        state.handles[res] = *info;
      }
    }
  }

  void analyzeOp(Operation *callableOp, Operation *op, ContextOp ambient,
                 std::optional<GPUExecutionScope> scope, MLIRPathState &state) {
    if (auto scfIf = dyn_cast<scf::IfOp>(op)) {
      const MLIRPathState preBranch = state;
      MLIRPathState thenState = preBranch;
      MLIRPathState elseState = preBranch;
      analyzeRegion(callableOp, scfIf.getThenRegion(), ambient, scope,
                    thenState);
      consumeYieldedResults(scfIf, scfIf.getThenRegion(), thenState,
                            scfIf.getResults());
      if (!scfIf.getElseRegion().empty()) {
        analyzeRegion(callableOp, scfIf.getElseRegion(), ambient, scope,
                      elseState);
        consumeYieldedResults(scfIf, scfIf.getElseRegion(), elseState,
                              scfIf.getResults());
      }
      std::vector<MLIRPathState> continuing;
      if (thenState.reachable) continuing.push_back(std::move(thenState));
      if (elseState.reachable) continuing.push_back(std::move(elseState));
      state = mergePathStates(scfIf, continuing, &preBranch);
      return;
    }

    if (auto affIf = dyn_cast<affine::AffineIfOp>(op)) {
      const MLIRPathState preBranch = state;
      MLIRPathState thenState = preBranch;
      MLIRPathState elseState = preBranch;
      analyzeRegion(callableOp, affIf.getThenRegion(), ambient, scope,
                    thenState);
      consumeYieldedResults(affIf, affIf.getThenRegion(), thenState,
                            affIf.getResults());
      if (!affIf.getElseRegion().empty()) {
        analyzeRegion(callableOp, affIf.getElseRegion(), ambient, scope,
                      elseState);
        consumeYieldedResults(affIf, affIf.getElseRegion(), elseState,
                              affIf.getResults());
      }
      std::vector<MLIRPathState> continuing;
      if (thenState.reachable) continuing.push_back(std::move(thenState));
      if (elseState.reachable) continuing.push_back(std::move(elseState));
      state = mergePathStates(affIf, continuing, &preBranch);
      return;
    }

    if (auto scfFor = dyn_cast<scf::ForOp>(op)) {
      analyzeCountedLoop(callableOp, scfFor, scfFor.getInitArgs(),
                         scfFor.getRegionIterArgs(), scfFor.getRegion(), ambient,
                         scope, state);
      return;
    }

    if (auto affFor = dyn_cast<affine::AffineForOp>(op)) {
      analyzeCountedLoop(callableOp, affFor, affFor.getInits(),
                         affFor.getRegionIterArgs(), affFor.getRegion(), ambient,
                         scope, state);
      return;
    }

    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      MLIRPathState beforeEntry = state;
      std::vector<MLIRHandleInfo> expectedBefore;
      for (auto [initVal, bArg] :
           llvm::zip(whileOp.getInits(), whileOp.getBeforeArguments())) {
        checkLoanAccess(whileOp, beforeEntry, initVal, false, false);
        if (beforeEntry.handles.contains(initVal)) {
          MLIRHandleInfo info = beforeEntry.handles[initVal];
          consumeValue(whileOp, beforeEntry, initVal);
          info.state = MLIRHandleState::Live;
          beforeEntry.handles[bArg] = info;
          expectedBefore.push_back(info);
        } else if (auto info = classifyValueHandle(whileOp, bArg.getType())) {
          beforeEntry.handles[bArg] = *info;
          expectedBefore.push_back(*info);
        } else {
          expectedBefore.push_back(MLIRHandleInfo{});
        }
      }

      const MLIRPathState headerSnapshot = beforeEntry;
      MLIRPathState condState = headerSnapshot;
      analyzeBlockOps(callableOp, whileOp.getBefore().front(), ambient, scope,
                      condState);
      auto condOp =
          cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());

      MLIRPathState afterState = condState;
      for (auto [argVal, afterArg] :
           llvm::zip(condOp.getArgs(), whileOp.getAfterArguments())) {
        if (afterState.handles.contains(argVal)) {
          MLIRHandleInfo info = afterState.handles[argVal];
          consumeValue(condOp, afterState, argVal);
          info.state = MLIRHandleState::Live;
          afterState.handles[afterArg] = info;
        }
      }
      analyzeBlockOps(callableOp, whileOp.getAfter().front(), ambient, scope,
                      afterState);

      if (afterState.reachable) {
        auto yieldOp =
            cast<scf::YieldOp>(whileOp.getAfter().front().getTerminator());
        for (auto [idx, yVal] : llvm::enumerate(yieldOp.getOperands())) {
          BlockArgument bArg = whileOp.getBeforeArguments()[idx];
          if (!headerSnapshot.handles.contains(bArg)) continue;
          auto yIt = afterState.handles.find(yVal);
          if (yIt == afterState.handles.end() ||
              yIt->second.state != MLIRHandleState::Live ||
              yIt->second.stateSymbol != expectedBefore[idx].stateSymbol) {
            reject(yieldOp, "scf.while backedge changes owned resource typestate");
          } else {
            consumeValue(yieldOp, afterState, yVal);
            MLIRHandleInfo restored = expectedBefore[idx];
            restored.state = MLIRHandleState::Live;
            afterState.handles[bArg] = restored;
          }
        }
        for (BlockArgument aArg : whileOp.getAfterArguments())
          afterState.handles.erase(aArg);
        checkLoopBackedgeState(whileOp, headerSnapshot, afterState);
      }

      MLIRPathState exitState = condState;
      for (BlockArgument bArg : whileOp.getBeforeArguments())
        exitState.handles.erase(bArg);
      for (auto [argVal, res] :
           llvm::zip(condOp.getArgs(), whileOp.getResults())) {
        if (condState.handles.contains(argVal)) {
          MLIRHandleInfo info = condState.handles[argVal];
          info.state = MLIRHandleState::Live;
          exitState.handles[res] = info;
        } else if (auto info = classifyValueHandle(whileOp, res.getType())) {
          exitState.handles[res] = *info;
        }
      }
      state = std::move(exitState);
      return;
    }

    if (auto devInt = dyn_cast<DeviceIntrinsicOp>(op)) {
      ContextOp selected = resolveContextOp(devInt, devInt.getContextAttr());
      if (!ambient || !selected || selected != ambient) {
        reject(devInt, "device operation requires the same nominal execution context as its body");
        return;
      }
      const StringRef kind = devInt.getKind();
      StringRef requiredGrant;
      if (kind == "load") requiredGrant = "gpu_load";
      else if (kind == "store") requiredGrant = "gpu_store";
      else if (kind == "barrier") requiredGrant = "gpu_barrier";
      else if (kind == "warp_barrier") requiredGrant = "gpu_warp_barrier";
      else if (kind == "grid_barrier") requiredGrant = "gpu_grid_sync";

      if (!requiredGrant.empty() && !contextHasGrant(selected, requiredGrant)) {
        reject(devInt, "device context lacks required grant: " + requiredGrant);
      }
      if (scope && !verification::isIntrinsicAllowedInScope(*scope, kind)) {
        reject(devInt, "device intrinsic `" + kind + "` is forbidden in `" +
                           verification::gpuExecutionScopeName(*scope) +
                           "` execution scope");
      }
      if (kind == "load") {
        checkLoanAccess(devInt, state, devInt.getOperand(0), /*isWrite=*/false,
                        /*isFree=*/false);
      } else if (kind == "store") {
        checkLoanAccess(devInt, state, devInt.getOperand(0), /*isWrite=*/true,
                        /*isFree=*/false);
      }
      return;
    }

    if (isa<gpu::BarrierOp>(op)) {
      if (!scope || *scope != GPUExecutionScope::Workgroup) {
        reject(op, "workgroup gpu.barrier requires a Gpu::Block / Workgroup execution context");
      } else if (ambient && !contextHasGrant(ambient, "gpu_barrier")) {
        reject(op, "device context lacks required grant: gpu_barrier");
      }
      return;
    }

    if (auto hostInt = dyn_cast<HostIntrinsicOp>(op)) {
      const StringRef kind = hostInt.getKind();
      if (kind == "gpu::alloc" || kind == "host::alloc") {
        if (auto info = classifyValueHandle(hostInt, hostInt.getResult(0).getType(),
                                            /*explicitOwnAttr=*/false,
                                            /*dynamicSpan=*/true)) {
          state.handles[hostInt.getResult(0)] = *info;
        }
      } else if (kind == "gpu::free" || kind == "host::free") {
        checkLoanAccess(hostInt, state, hostInt.getOperand(0), /*isWrite=*/true,
                        /*isFree=*/true);
        consumeValue(hostInt, state, hostInt.getOperand(0));
      } else if (kind == "host::load") {
        checkLoanAccess(hostInt, state, hostInt.getOperand(0),
                        /*isWrite=*/false, /*isFree=*/false);
      } else if (kind == "host::store") {
        checkLoanAccess(hostInt, state, hostInt.getOperand(0), /*isWrite=*/true,
                        /*isFree=*/false);
      } else if (kind == "gpu::upload") {
        checkLoanAccess(hostInt, state, hostInt.getOperand(0), /*isWrite=*/true,
                        /*isFree=*/false);
        checkLoanAccess(hostInt, state, hostInt.getOperand(1),
                        /*isWrite=*/false, /*isFree=*/false);
      } else if (kind == "gpu::download") {
        checkLoanAccess(hostInt, state, hostInt.getOperand(0),
                        /*isWrite=*/false, /*isFree=*/false);
        checkLoanAccess(hostInt, state, hostInt.getOperand(1), /*isWrite=*/true,
                        /*isFree=*/false);
      } else if (kind == "gpu::launch") {
        for (Value arg : hostInt.getInputs().drop_front(2)) {
          if (isa<MemRefType>(arg.getType())) {
            checkLoanAccess(hostInt, state, arg, /*isWrite=*/true,
                            /*isFree=*/false);
          }
        }
      }
      return;
    }

    if (auto asyncCopy = dyn_cast<AsyncCopyOp>(op)) {
      ContextOp selected = resolveContextOp(asyncCopy, asyncCopy.getContextAttr());
      if (!ambient || !selected || selected != ambient) {
        reject(asyncCopy, "async_copy requires the same nominal execution context as its body");
        return;
      }
      if (!contextHasGrant(selected, "gpu_async_copy")) {
        reject(asyncCopy, "context lacks required grant: gpu_async_copy");
      }
      if (scope &&
          !verification::isCapabilityAllowedInScope(*scope, "gpu_async_copy")) {
        reject(asyncCopy, "async_copy is forbidden in `" +
                              Twine(verification::gpuExecutionScopeName(*scope)) +
                              "` scope");
      }

      const auto srcSpace =
          memrefAddressSpace(cast<MemRefType>(asyncCopy.getSource().getType()));
      const auto dstSpace = memrefAddressSpace(
          cast<MemRefType>(asyncCopy.getDestination().getType()));
      if (!srcSpace || !dstSpace) {
        reject(asyncCopy, "async_copy uses an unsupported memref address space");
        return;
      }
      if (scope) {
        if (*scope == GPUExecutionScope::Host) {
          const bool validHostDMA =
              (*srcSpace == GPUAddressSpace::HostDefault &&
               *dstSpace == GPUAddressSpace::GlobalDevice) ||
              (*srcSpace == GPUAddressSpace::GlobalDevice &&
               *dstSpace == GPUAddressSpace::HostDefault);
          if (!validHostDMA) {
            reject(asyncCopy, "cross-space dereference: host async_copy supports host<->global_device transfers only");
          }
        } else if (*srcSpace == GPUAddressSpace::HostDefault ||
                   *dstSpace == GPUAddressSpace::HostDefault) {
          reject(asyncCopy, "cross-space dereference: device async_copy cannot access host default address space");
        }
      }

      const SlotGenPair sgKey{asyncCopy.getSlot().str(),
                              asyncCopy.getGeneration().str()};
      if (state.generations.contains(sgKey)) {
        reject(asyncCopy, Twine("stale or duplicate async generation reuse on (") +
                              asyncCopy.getSlot() +
                              ", gen=" + asyncCopy.getGeneration() + ")");
        return;
      }
      checkLoanAccess(asyncCopy, state, asyncCopy.getSource(),
                      /*isWrite=*/false, /*isFree=*/false);
      checkLoanAccess(asyncCopy, state, asyncCopy.getDestination(),
                      /*isWrite=*/true, /*isFree=*/false);

      state.generations[sgKey] = MLIRGenStatus::InFlight;
      state.activeLoans.push_back(
          MLIRLoanEntry{asyncCopy.getSource(), sgKey.first, sgKey.second,
                        AsyncLoanMode::ImmutableRead});
      state.activeLoans.push_back(
          MLIRLoanEntry{asyncCopy.getDestination(), sgKey.first, sgKey.second,
                        AsyncLoanMode::ExclusiveWrite});
      return;
    }

    if (auto asyncWait = dyn_cast<AsyncWaitOp>(op)) {
      ContextOp selected = resolveContextOp(asyncWait, asyncWait.getContextAttr());
      if (!ambient || !selected || selected != ambient) {
        reject(asyncWait, "async_wait requires the same nominal execution context as its body");
        return;
      }
      if (!contextHasGrant(selected, "gpu_await")) {
        reject(asyncWait, "context lacks required grant: gpu_await");
      }

      const SlotGenPair sgKey{asyncWait.getSlot().str(),
                              asyncWait.getGeneration().str()};
      auto it = state.generations.find(sgKey);
      if (it == state.generations.end()) {
        reject(asyncWait, Twine("async_wait references unissued generation (") +
                              asyncWait.getSlot() +
                              ", gen=" + asyncWait.getGeneration() + ")");
        return;
      }
      if (it->second == MLIRGenStatus::Completed) {
        reject(asyncWait, Twine("stale async completion reuse on (") +
                              asyncWait.getSlot() +
                              ", gen=" + asyncWait.getGeneration() + ")");
        return;
      }
      it->second = MLIRGenStatus::Completed;
      std::erase_if(state.activeLoans, [&](const MLIRLoanEntry &entry) {
        return entry.slot == sgKey.first && entry.generation == sgKey.second;
      });
      return;
    }

    if (auto transOp = dyn_cast<TransitionOp>(op)) {
      auto decl = dyn_cast_or_null<TransitionDeclOp>(
          lookupSymbolInAncestors(transOp, transOp.getCallee()));
      if (auto ctxRef = transOp.getContextAttr()) {
        ContextOp selected = resolveContextOp(transOp, ctxRef);
        if (!ambient || !selected || selected != ambient) {
          reject(transOp, "transition requires the same nominal execution context as its body");
        }
      }
      ArrayAttr paramAttrs = decl ? decl.getParameterAttrs() : ArrayAttr{};
      for (auto [idx, operand] : llvm::enumerate(transOp.getArguments())) {
        bool consumes = (decl && idx == decl.getSelfIndex()) ||
                        isa<ResourceType, OwnedType>(operand.getType());
        if (paramAttrs && idx < paramAttrs.size()) {
          if (auto dict = dyn_cast<DictionaryAttr>(paramAttrs[idx])) {
            if (auto ownAttr = dict.getAs<StringAttr>("ownership"))
              consumes |= (ownAttr.getValue() == "own");
          }
        }
        checkLoanAccess(transOp, state, operand, /*isWrite=*/consumes,
                        /*isFree=*/false);
        if (consumes) consumeValue(transOp, state, operand);
      }
      if (auto succInfo =
              classifyValueHandle(transOp, transOp.getSuccessor().getType())) {
        state.handles[transOp.getSuccessor()] = *succInfo;
      }
      return;
    }

    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      checkMemRefSpace(loadOp, loadOp.getMemRefType(), scope, ambient,
                       /*isStore=*/false, loadOp.getMemref());
      checkLoanAccess(loadOp, state, loadOp.getMemref(), /*isWrite=*/false,
                      /*isFree=*/false);
      return;
    }
    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      checkMemRefSpace(storeOp, storeOp.getMemRefType(), scope, ambient,
                       /*isStore=*/true, storeOp.getMemref());
      checkLoanAccess(storeOp, state, storeOp.getMemref(), /*isWrite=*/true,
                      /*isFree=*/false);
      return;
    }
    if (auto affLoad = dyn_cast<affine::AffineLoadOp>(op)) {
      checkMemRefSpace(affLoad, affLoad.getMemRefType(), scope, ambient,
                       /*isStore=*/false, affLoad.getMemRef());
      checkLoanAccess(affLoad, state, affLoad.getMemRef(), /*isWrite=*/false,
                      /*isFree=*/false);
      return;
    }
    if (auto affStore = dyn_cast<affine::AffineStoreOp>(op)) {
      checkMemRefSpace(affStore, affStore.getMemRefType(), scope, ambient,
                       /*isStore=*/true, affStore.getMemRef());
      checkLoanAccess(affStore, state, affStore.getMemRef(), /*isWrite=*/true,
                      /*isFree=*/false);
      return;
    }
    if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
      checkLoanAccess(deallocOp, state, deallocOp.getMemref(), /*isWrite=*/true,
                      /*isFree=*/true);
      consumeValue(deallocOp, state, deallocOp.getMemref());
      return;
    }

    if (auto callOp = dyn_cast<func::CallOp>(op)) {
      Operation *calleeContract =
          namedContracts_.contains(callOp.getCallee().str())
              ? namedContracts_[callOp.getCallee().str()]
              : nullptr;
      ArrayAttr paramAttrs =
          calleeContract
              ? calleeContract->getAttrOfType<ArrayAttr>("parameter_attrs")
              : ArrayAttr{};
      for (auto [idx, operand] : llvm::enumerate(callOp.getOperands())) {
        bool consumes = isa<ResourceType, OwnedType>(operand.getType());
        if (paramAttrs && idx < paramAttrs.size()) {
          if (auto dict = dyn_cast<DictionaryAttr>(paramAttrs[idx])) {
            if (auto ownAttr = dict.getAs<StringAttr>("ownership"))
              consumes |= (ownAttr.getValue() == "own");
          }
        }
        checkLoanAccess(callOp, state, operand, /*isWrite=*/consumes,
                        /*isFree=*/false);
        if (consumes) consumeValue(callOp, state, operand);
      }
      for (Value res : callOp.getResults()) {
        if (auto info = classifyValueHandle(callOp, res.getType())) {
          state.handles[res] = *info;
        }
      }
      return;
    }

    if (auto funcRet = dyn_cast<func::ReturnOp>(op)) {
      checkReturnObligations(funcRet, state, funcRet.getOperands());
      return;
    }
    if (auto gpuRet = dyn_cast<gpu::ReturnOp>(op)) {
      checkReturnObligations(gpuRet, state, gpuRet.getOperands());
      return;
    }
  }
};

} // namespace

void promoteGPUStructuredLoopsAndBranches(Operation *root) {
  SmallVector<scf::WhileOp> whileLoops;
  root->walk<WalkOrder::PostOrder>(
      [&](scf::WhileOp loop) { whileLoops.push_back(loop); });
  for (scf::WhileOp loop : whileLoops) promoteCountedLoop(loop);

  while (root->walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
               return promoteAffineLoop(loop) ? WalkResult::interrupt()
                                              : WalkResult::advance();
             })
             .wasInterrupted()) {
  }

  SmallVector<scf::IfOp> branches;
  root->walk<WalkOrder::PostOrder>(
      [&](scf::IfOp branch) { branches.push_back(branch); });
  for (scf::IfOp branch : branches) promoteAffineConditional(branch);
}

LogicalResult structureGPUFunctionControlFlow(ModuleOp module,
                                              bool promoteAllScalarSlots,
                                              bool promoteAffine) {
  MLIRContext *context = module.getContext();
  DialectRegistry registry;
  memref::registerMemorySlotExternalModels(registry);
  context->appendDialectRegistry(registry);
  context->getOrLoadDialect<scf::SCFDialect>();
  context->getOrLoadDialect<affine::AffineDialect>();
  context->getOrLoadDialect<gpu::GPUDialect>();
  context->getOrLoadDialect<ub::UBDialect>();

  auto promoteRegionSlots = [&](Region &body) {
    if (body.empty()) return;
    SmallVector<PromotableAllocationOpInterface> allocators;
    body.walk([&](memref::AllocaOp alloca) {
      if (promoteAllScalarSlots || alloca->hasAttr("metreon.synthetic_iv")) {
        if (auto promotable =
                dyn_cast<PromotableAllocationOpInterface>(alloca.getOperation())) {
          allocators.push_back(promotable);
        }
      }
    });
    if (!allocators.empty()) {
      IRRewriter rewriter(context);
      (void)tryToPromoteMemorySlots(allocators, rewriter);
    }
  };
  module.walk([&](func::FuncOp fn) {
    if (!fn.isExternal()) promoteRegionSlots(fn.getBody());
  });
  module.walk([&](gpu::GPUFuncOp fn) {
    if (!fn.isDeclaration()) promoteRegionSlots(fn.getBody());
  });

  ControlFlowToSCFTransformation conversion;
  bool cfgFailed = false;
  auto liftRegion = [&](Operation *owner, Region &body) {
    if (body.empty() || body.hasOneBlock()) return;
    DominanceInfo dominance(owner);
    if (failed(transformCFGToSCF(body, conversion, dominance)))
      cfgFailed = true;
  };
  module.walk([&](func::FuncOp fn) {
    if (!fn.isExternal()) liftRegion(fn, fn.getBody());
  });
  module.walk([&](gpu::GPUFuncOp fn) {
    if (!fn.isDeclaration()) liftRegion(fn, fn.getBody());
  });
  if (cfgFailed) return failure();

  if (failed(normalizeStructuredRegions(module))) return failure();

  if (promoteAffine) promoteGPUStructuredLoopsAndBranches(module);

  return success();
}

LogicalResult verifyGPUContextHierarchyAndLoans(ModuleOp module) {
  return MLIRGPUVerifier(module).run();
}

} // namespace metreon
