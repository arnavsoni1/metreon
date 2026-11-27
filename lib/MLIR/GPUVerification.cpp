#include "metreon/MLIR/GPUVerification.h"

#include "metreon/MLIR/Dialect.h"
#include "metreon/Verification/Behavioral.h"
#include "metreon/Verification/GPUMemoryAccessProtocol.h"
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
#include "mlir/IR/AffineExpr.h"
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

namespace {

using verification::addExpr;
using verification::AffineSymbolicExpr;
using verification::divExpr;
using verification::GPUMemoryAccessProtocolOptions;
using verification::GPUProtocolReport;
using verification::joinUniformity;
using verification::MemoryAccessMode;
using verification::mulExpr;
using verification::negExpr;
using verification::ProtocolStmt;
using verification::remExpr;
using verification::SIMTUniformity;
using verification::subExpr;
using verification::SymbolicGuard;
using verification::SymbolicVar;
using verification::SymbolicVarKind;

SourceLocation mlirLocToSourceLoc(Location loc) {
  if (auto flc = dyn_cast<FileLineColLoc>(loc)) {
    return SourceLocation{flc.getLine(), flc.getColumn()};
  }
  return SourceLocation{0, 0};
}

struct GuardPair {
  SymbolicGuard whenTrue;
  SymbolicGuard whenFalse;
};

struct MLIRBufferDesc {
  std::string name;
  GPUAddressSpace space = GPUAddressSpace::GlobalDevice;
  std::optional<std::int64_t> staticExtent;
  std::optional<std::string> dynamicExtentParam;
  SmallVector<std::int64_t, 4> staticShape;
};

Value unwrapMemRefBase(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (isa<memref::CastOp, memref::ReinterpretCastOp,
            UnrealizedConversionCastOp>(def) &&
        def->getNumOperands() == 1) {
      v = def->getOperand(0);
      continue;
    }
    break;
  }
  return v;
}

class MLIRAccessProtocolVerifier {
public:
  MLIRAccessProtocolVerifier(ModuleOp root,
                             const GPUMemoryAccessProtocolOptions &options)
      : root_(root), options_(options) {}

  LogicalResult run() {
    indexModuleCallables();

    bool foundGPUModule = false;
    root_.walk([&](gpu::GPUModuleOp gpuMod) {
      foundGPUModule = true;
      verifyGPUModule(gpuMod);
    });

    if (!foundGPUModule) {
      verifyStandaloneGPUFunctions();
    }

    return success(valid_);
  }

private:
  struct Env {
    llvm::DenseMap<Value, AffineSymbolicExpr> exprs;
    llvm::DenseMap<Value, SIMTUniformity> uniformity;
    llvm::DenseSet<Value> controlTainted;
    llvm::DenseMap<Value, GuardPair> guards;
    llvm::DenseMap<Value, MLIRBufferDesc> buffers;
    llvm::DenseMap<Value, AffineSymbolicExpr> slotExprs;
    llvm::DenseMap<Value, SIMTUniformity> slotUniformity;
    llvm::DenseMap<Value, GuardPair> slotGuards;
    std::map<std::string, SIMTUniformity> loopVarUniformity;
  };

  ModuleOp root_;
  GPUMemoryAccessProtocolOptions options_;
  bool valid_ = true;
  unsigned nextLoopId_ = 0;
  unsigned nextAnonBufId_ = 0;
  llvm::DenseMap<Operation *, Operation *> bodyToContract_;
  std::map<std::string, func::FuncOp> helperFuncs_;
  std::set<std::string> activeCallStack_;
  std::map<std::pair<unsigned, unsigned>, Operation *> locToOp_;

  void recordOpLoc(Operation *op) {
    const SourceLocation sl = mlirLocToSourceLoc(op->getLoc());
    if (sl.line != 0 || sl.column != 0) {
      locToOp_.try_emplace({sl.line, sl.column}, op);
    }
  }

  void indexModuleCallables() {
    root_.walk([&](Operation *op) {
      if (!isa<KernelOp, ProcedureOp, StageOp>(op)) return;
      if (auto bodyRef = op->getAttrOfType<SymbolRefAttr>("body_ref")) {
        if (Operation *body = lookupSymbolInAncestors(op, bodyRef))
          bodyToContract_[body] = op;
      }
      if (auto devBody =
              op->getAttrOfType<SymbolRefAttr>("metreon.device_body")) {
        if (Operation *body = lookupSymbolInAncestors(op, devBody))
          bodyToContract_[body] = op;
      }
    });
  }

  bool isGPUFuncCandidate(func::FuncOp fn) {
    if (fn.isExternal()) return false;
    if (Operation *contract = bodyToContract_.lookup(fn)) {
      auto dom = contract->getAttrOfType<StringAttr>("execution_domain");
      if (dom) return dom.getValue() == "Gpu";
    }
    if (auto ctxRef = fn->getAttrOfType<SymbolRefAttr>("metreon.context")) {
      if (ContextOp ctx = resolveContextOp(fn, ctxRef)) {
        if (auto nom = ctx.getNominalType().getAs<StringAttr>("name")) {
          if (nom.getValue().starts_with("Host::")) return false;
          if (nom.getValue().starts_with("Gpu::")) return true;
        }
      }
    }
    bool hasGPUConstruct = false;
    fn.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "gpu" ||
          isa<DeviceIntrinsicOp>(op)) {
        hasGPUConstruct = true;
        return WalkResult::interrupt();
      }
      for (Type t : op->getOperandTypes()) {
        if (auto mt = dyn_cast<MemRefType>(t)) {
          const auto sp = memrefAddressSpace(mt);
          if (sp && (*sp == GPUAddressSpace::GlobalDevice ||
                     *sp == GPUAddressSpace::WorkgroupShared)) {
            hasGPUConstruct = true;
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    return hasGPUConstruct;
  }

  void verifyGPUModule(gpu::GPUModuleOp gpuMod) {
    helperFuncs_.clear();
    std::set<std::string> calledHelpers;
    for (func::FuncOp fn : gpuMod.getOps<func::FuncOp>()) {
      if (!fn.isExternal()) helperFuncs_[fn.getSymName().str()] = fn;
    }
    gpuMod.walk([&](func::CallOp call) {
      calledHelpers.insert(call.getCallee().str());
    });

    for (gpu::GPUFuncOp kernel : gpuMod.getOps<gpu::GPUFuncOp>()) {
      if (kernel.isDeclaration()) continue;
      verifyCallableRoot(kernel, kernel.getName().str(), kernel.getBody(),
                         kernel.getFunctionType().getNumInputs(),
                         kernel.getWorkgroupAttributions());
    }

    // Also verify uncalled root GPU procedures when a gpu.module has no kernels
    // (or standalone root procedures).
    for (func::FuncOp fn : gpuMod.getOps<func::FuncOp>()) {
      if (fn.isExternal()) continue;
      if (calledHelpers.contains(fn.getSymName().str())) continue;
      verifyCallableRoot(fn, fn.getSymName().str(), fn.getBody(),
                         fn.getFunctionType().getNumInputs(), {});
    }
  }

  void verifyStandaloneGPUFunctions() {
    helperFuncs_.clear();
    SmallVector<func::FuncOp> candidates;
    std::set<std::string> calledHelpers;

    root_.walk([&](func::FuncOp fn) {
      if (!fn.isExternal()) helperFuncs_[fn.getSymName().str()] = fn;
      if (isGPUFuncCandidate(fn)) candidates.push_back(fn);
    });
    for (func::FuncOp fn : candidates) {
      fn.walk([&](func::CallOp call) {
        calledHelpers.insert(call.getCallee().str());
      });
    }

    for (func::FuncOp fn : candidates) {
      Operation *contract = bodyToContract_.lookup(fn);
      const bool isKernel = isa_and_nonnull<KernelOp>(contract);
      if (!isKernel && calledHelpers.contains(fn.getSymName().str())) continue;
      verifyCallableRoot(fn, fn.getSymName().str(), fn.getBody(),
                         fn.getFunctionType().getNumInputs(), {});
    }
  }

  void verifyCallableRoot(Operation *callableOp, const std::string &name,
                          Region &body, unsigned numInputs,
                          ArrayRef<BlockArgument> workgroupAttrs) {
    if (body.empty() || !body.hasOneBlock()) return;
    locToOp_.clear();
    activeCallStack_.clear();
    activeCallStack_.insert(name);

    Env env;
    Operation *contract = bodyToContract_.lookup(callableOp);
    ArrayAttr paramAttrs =
        contract ? contract->getAttrOfType<ArrayAttr>("parameter_attrs")
                 : ArrayAttr{};

    auto paramNameFor = [&](unsigned idx) -> std::string {
      if (paramAttrs && idx < paramAttrs.size()) {
        if (auto dict = dyn_cast<DictionaryAttr>(paramAttrs[idx])) {
          if (auto nm = dict.getAs<StringAttr>("name"))
            if (!nm.getValue().empty()) return nm.getValue().str();
        }
      }
      return "arg" + std::to_string(idx);
    };

    Block &entry = body.front();
    std::vector<std::string> indexParams;
    for (unsigned i = 0; i < numInputs && i < entry.getNumArguments(); ++i) {
      BlockArgument arg = entry.getArgument(i);
      if (arg.getType().isIndex()) indexParams.push_back(paramNameFor(i));
    }
    const std::optional<std::string> inferredDynExtent =
        indexParams.size() == 1 ? std::optional<std::string>(indexParams.front())
                                : std::nullopt;

    for (unsigned i = 0; i < numInputs && i < entry.getNumArguments(); ++i) {
      BlockArgument arg = entry.getArgument(i);
      const std::string pName = paramNameFor(i);
      if (auto memType = dyn_cast<MemRefType>(arg.getType())) {
        if (auto desc = buildBufferDesc(pName, memType, inferredDynExtent)) {
          env.buffers[arg] = *desc;
        }
      } else if (isa<IndexType, IntegerType, FloatType>(arg.getType())) {
        env.exprs[arg] =
            AffineSymbolicExpr::makeVar(SymbolicVarKind::ScalarParam, pName);
        env.uniformity[arg] = SIMTUniformity::WorkgroupUniform;
      }
    }

    for (auto [idx, wgArg] : llvm::enumerate(workgroupAttrs)) {
      if (auto memType = dyn_cast<MemRefType>(wgArg.getType())) {
        if (auto desc = buildBufferDesc("wg_attr_" + std::to_string(idx),
                                        memType, std::nullopt)) {
          env.buffers[wgArg] = *desc;
        }
      }
    }

    std::vector<ProtocolStmt> stmts;
    extractBlock(entry, SIMTUniformity::WorkgroupUniform, env, stmts, nullptr);
    ProtocolStmt rootStmt = ProtocolStmt::makeSequence(
        std::move(stmts), mlirLocToSourceLoc(callableOp->getLoc()));

    GPUProtocolReport report;
    verification::verifyKernelAccessProtocol(name, rootStmt, options_, report);
    for (const auto &v : report.violations) {
      Operation *diagAnchor = callableOp;
      auto it = locToOp_.find({v.location.line, v.location.column});
      if (it != locToOp_.end() && it->second != nullptr) diagAnchor = it->second;
      diagAnchor->emitError("[" + v.code + "] " + v.message);
      valid_ = false;
    }
  }

  std::optional<MLIRBufferDesc>
  buildBufferDesc(std::string name, MemRefType memType,
                  std::optional<std::string> dynExtent) {
    const auto space = memrefAddressSpace(memType);
    if (!space || (*space != GPUAddressSpace::GlobalDevice &&
                   *space != GPUAddressSpace::WorkgroupShared)) {
      return std::nullopt;
    }
    MLIRBufferDesc desc;
    desc.name = std::move(name);
    desc.space = *space;
    if (memType.getRank() >= 1) {
      bool allStatic = true;
      std::int64_t prod = 1;
      for (std::int64_t d = 0; d < memType.getRank(); ++d) {
        const std::int64_t dim = memType.getDimSize(d);
        desc.staticShape.push_back(dim);
        if (memType.isDynamicDim(d)) allStatic = false;
        else prod *= dim;
      }
      if (allStatic) desc.staticExtent = prod;
      else if (*space == GPUAddressSpace::GlobalDevice &&
               memType.getRank() == 1 && dynExtent) {
        desc.dynamicExtentParam = std::move(dynExtent);
      }
    }
    return desc;
  }

  std::optional<MLIRBufferDesc> resolveBuffer(Value rawVal, MemRefType memType,
                                              Env &env) {
    Value base = unwrapMemRefBase(rawVal);
    auto it = env.buffers.find(base);
    if (it != env.buffers.end()) return it->second;

    if (memType.getRank() == 0 &&
        isa_and_nonnull<memref::AllocaOp>(base.getDefiningOp())) {
      return std::nullopt;
    }
    std::string bufName;
    if (auto glob = base.getDefiningOp<memref::GetGlobalOp>()) {
      bufName = "@" + glob.getName().str();
    } else {
      bufName = "shared_buf_" + std::to_string(nextAnonBufId_++);
    }
    auto desc = buildBufferDesc(std::move(bufName), memType, std::nullopt);
    if (desc) env.buffers[base] = *desc;
    return desc;
  }

  SIMTUniformity valUniformity(Value v, const Env &env) const {
    auto it = env.uniformity.find(v);
    return it != env.uniformity.end() ? it->second
                                      : SIMTUniformity::WorkgroupUniform;
  }

  AffineSymbolicExpr valExpr(Value v, const Env &env) const {
    auto it = env.exprs.find(v);
    return it != env.exprs.end() ? it->second : AffineSymbolicExpr::nonAffine();
  }

  GuardPair valGuard(Value v, const Env &env) const {
    auto it = env.guards.find(v);
    return it != env.guards.end() ? it->second : GuardPair{};
  }

  SIMTUniformity deriveExprUniformity(const AffineSymbolicExpr &e,
                                      const Env &env,
                                      SIMTUniformity fallback) const {
    if (!e.isAffine || e.modulus != 0) return fallback;
    SIMTUniformity u = SIMTUniformity::WorkgroupUniform;
    for (const auto &[var, coeff] : e.terms) {
      if (coeff == 0) continue;
      switch (var.kind) {
      case SymbolicVarKind::ThreadIdX:
      case SymbolicVarKind::ThreadIdY:
      case SymbolicVarKind::ThreadIdZ:
      case SymbolicVarKind::LaneId:
        u = joinUniformity(u, SIMTUniformity::LaneDivergent);
        break;
      case SymbolicVarKind::WarpId:
        u = joinUniformity(u, SIMTUniformity::SubgroupUniform);
        break;
      case SymbolicVarKind::LoopVar: {
        auto it = env.loopVarUniformity.find(var.name);
        if (it != env.loopVarUniformity.end())
          u = joinUniformity(u, it->second);
        break;
      }
      default:
        break;
      }
    }
    return u;
  }

  AffineSymbolicExpr linearizeIndices(ArrayRef<AffineSymbolicExpr> indices,
                                      const MLIRBufferDesc &buf) const {
    if (indices.empty()) return AffineSymbolicExpr::makeConstant(0);
    if (indices.size() == 1) return indices.front();
    if (indices.size() != buf.staticShape.size())
      return AffineSymbolicExpr::nonAffine();
    AffineSymbolicExpr acc = indices.front();
    for (std::size_t d = 1; d < indices.size(); ++d) {
      if (buf.staticShape[d] < 0) return AffineSymbolicExpr::nonAffine();
      acc = addExpr(
          mulExpr(acc, AffineSymbolicExpr::makeConstant(buf.staticShape[d]),
                  options_.subgroupSize),
          indices[d]);
    }
    return acc;
  }

  std::pair<AffineSymbolicExpr, SIMTUniformity>
  evalAffineExpr(AffineExpr expr, OperandRange dims, OperandRange syms,
                 const Env &env) const {
    if (auto cst = dyn_cast<AffineConstantExpr>(expr)) {
      return {AffineSymbolicExpr::makeConstant(cst.getValue()),
              SIMTUniformity::WorkgroupUniform};
    }
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
      if (dim.getPosition() < dims.size()) {
        Value v = dims[dim.getPosition()];
        return {valExpr(v, env), valUniformity(v, env)};
      }
      return {AffineSymbolicExpr::nonAffine(), SIMTUniformity::LaneDivergent};
    }
    if (auto sym = dyn_cast<AffineSymbolExpr>(expr)) {
      if (sym.getPosition() < syms.size()) {
        Value v = syms[sym.getPosition()];
        return {valExpr(v, env), valUniformity(v, env)};
      }
      return {AffineSymbolicExpr::nonAffine(), SIMTUniformity::LaneDivergent};
    }
    if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
      const auto [lE, lU] = evalAffineExpr(bin.getLHS(), dims, syms, env);
      const auto [rE, rU] = evalAffineExpr(bin.getRHS(), dims, syms, env);
      const SIMTUniformity joined = joinUniformity(lU, rU);
      switch (bin.getKind()) {
      case AffineExprKind::Add:
        return {addExpr(lE, rE), joined};
      case AffineExprKind::Mul:
        return {mulExpr(lE, rE, options_.subgroupSize),
                (lE.isConstant() && lE.constant == 0) ||
                        (rE.isConstant() && rE.constant == 0)
                    ? SIMTUniformity::WorkgroupUniform
                    : joined};
      case AffineExprKind::FloorDiv: {
        const auto dE = divExpr(lE, rE, options_.subgroupSize);
        return {dE, deriveExprUniformity(dE, env, joined)};
      }
      case AffineExprKind::Mod:
        return {remExpr(lE, rE, options_.subgroupSize), joined};
      case AffineExprKind::CeilDiv:
        return {AffineSymbolicExpr::nonAffine(), joined};
      default:
        break;
      }
    }
    return {AffineSymbolicExpr::nonAffine(), SIMTUniformity::LaneDivergent};
  }

  void mergeBranchSlots(SIMTUniformity branchPC, const Env &preEnv,
                        const Env &thenEnv, const Env &elseEnv, Env &outEnv) {
    llvm::DenseSet<Value> slots;
    for (const auto &kv : preEnv.slotUniformity) slots.insert(kv.first);
    for (const auto &kv : thenEnv.slotUniformity) slots.insert(kv.first);
    for (const auto &kv : elseEnv.slotUniformity) slots.insert(kv.first);

    for (Value s : slots) {
      const auto tE = thenEnv.slotExprs.lookup(s);
      const auto eE = elseEnv.slotExprs.lookup(s);
      const auto pE = preEnv.slotExprs.lookup(s);
      const SIMTUniformity tU = thenEnv.slotUniformity.lookup(s);
      const SIMTUniformity eU = elseEnv.slotUniformity.lookup(s);
      const bool sameUnchanged =
          tE.isAffine && eE.isAffine && pE.isAffine &&
          tE.constant == pE.constant && eE.constant == pE.constant &&
          tE.terms == pE.terms && eE.terms == pE.terms &&
          tE.modulus == pE.modulus && eE.modulus == pE.modulus &&
          tU == preEnv.slotUniformity.lookup(s) &&
          eU == preEnv.slotUniformity.lookup(s);
      if (!sameUnchanged) {
        outEnv.slotUniformity[s] =
            joinUniformity(branchPC, joinUniformity(tU, eU));
        outEnv.slotExprs[s] = AffineSymbolicExpr::nonAffine();
        outEnv.slotGuards.erase(s);
      }
    }
  }

  struct CallReturnSummary {
    AffineSymbolicExpr expr = AffineSymbolicExpr::nonAffine();
    SIMTUniformity uniformity = SIMTUniformity::WorkgroupUniform;
    bool seen = false;
  };

  void extractBlock(Block &block, SIMTUniformity pcUniformity, Env &env,
                    std::vector<ProtocolStmt> &out,
                    CallReturnSummary *retSummary) {
    for (Operation &opRef : block) {
      Operation *op = &opRef;
      recordOpLoc(op);
      const SourceLocation loc = mlirLocToSourceLoc(op->getLoc());

      if (isa<arith::ConstantOp>(op)) {
        Value res = op->getResult(0);
        env.uniformity[res] = SIMTUniformity::WorkgroupUniform;
        IntegerAttr intAttr;
        if (matchPattern(op, m_Constant(&intAttr))) {
          if (intAttr.getType().isInteger(1)) {
            const bool bv = intAttr.getValue().getBoolValue();
            env.exprs[res] = AffineSymbolicExpr::makeConstant(bv ? 1 : 0);
            GuardPair gp;
            if (bv) gp.whenFalse.contradiction = true;
            else gp.whenTrue.contradiction = true;
            env.guards[res] = std::move(gp);
          } else if (intAttr.getValue().getBitWidth() <= 64) {
            env.exprs[res] = AffineSymbolicExpr::makeConstant(
                intAttr.getValue().getSExtValue());
          } else {
            env.exprs[res] = AffineSymbolicExpr::nonAffine();
          }
        } else {
          env.exprs[res] = AffineSymbolicExpr::nonAffine();
        }
        continue;
      }

      if (auto tid = dyn_cast<gpu::ThreadIdOp>(op)) {
        SymbolicVarKind vk = SymbolicVarKind::ThreadIdX;
        if (tid.getDimension() == gpu::Dimension::y)
          vk = SymbolicVarKind::ThreadIdY;
        else if (tid.getDimension() == gpu::Dimension::z)
          vk = SymbolicVarKind::ThreadIdZ;
        env.exprs[tid.getResult()] = AffineSymbolicExpr::makeVar(vk);
        env.uniformity[tid.getResult()] = SIMTUniformity::LaneDivergent;
        continue;
      }

      if (auto bid = dyn_cast<gpu::BlockIdOp>(op)) {
        SymbolicVarKind vk = SymbolicVarKind::BlockIdX;
        if (bid.getDimension() == gpu::Dimension::y)
          vk = SymbolicVarKind::BlockIdY;
        else if (bid.getDimension() == gpu::Dimension::z)
          vk = SymbolicVarKind::BlockIdZ;
        env.exprs[bid.getResult()] = AffineSymbolicExpr::makeVar(vk);
        env.uniformity[bid.getResult()] = SIMTUniformity::WorkgroupUniform;
        continue;
      }

      if (auto bdim = dyn_cast<gpu::BlockDimOp>(op)) {
        SymbolicVarKind vk = SymbolicVarKind::BlockDimX;
        if (bdim.getDimension() == gpu::Dimension::y)
          vk = SymbolicVarKind::BlockDimY;
        else if (bdim.getDimension() == gpu::Dimension::z)
          vk = SymbolicVarKind::BlockDimZ;
        env.exprs[bdim.getResult()] = AffineSymbolicExpr::makeVar(vk);
        env.uniformity[bdim.getResult()] = SIMTUniformity::WorkgroupUniform;
        continue;
      }

      if (auto sgId = dyn_cast<gpu::SubgroupIdOp>(op)) {
        env.exprs[sgId.getResult()] =
            AffineSymbolicExpr::makeVar(SymbolicVarKind::WarpId);
        env.uniformity[sgId.getResult()] = SIMTUniformity::SubgroupUniform;
        continue;
      }

      if (auto laneId = dyn_cast<gpu::LaneIdOp>(op)) {
        env.exprs[laneId.getResult()] =
            AffineSymbolicExpr::makeVar(SymbolicVarKind::LaneId);
        env.uniformity[laneId.getResult()] = SIMTUniformity::LaneDivergent;
        continue;
      }

      if (isa<gpu::BarrierOp>(op)) {
        out.push_back(ProtocolStmt::makeWorkgroupBarrier(pcUniformity, loc));
        continue;
      }

      if (auto allRed = dyn_cast<gpu::AllReduceOp>(op)) {
        out.push_back(ProtocolStmt::makeWorkgroupBarrier(pcUniformity, loc));
        env.uniformity[allRed.getResult()] = SIMTUniformity::WorkgroupUniform;
        env.exprs[allRed.getResult()] = AffineSymbolicExpr::nonAffine();
        continue;
      }

      if (auto subRed = dyn_cast<gpu::SubgroupReduceOp>(op)) {
        out.push_back(ProtocolStmt::makeSubgroupBarrier(
            pcUniformity, "gpu.subgroup_reduce", loc));
        env.uniformity[subRed.getResult()] = SIMTUniformity::SubgroupUniform;
        env.exprs[subRed.getResult()] = AffineSymbolicExpr::nonAffine();
        continue;
      }

      if (auto shfl = dyn_cast<gpu::ShuffleOp>(op)) {
        out.push_back(
            ProtocolStmt::makeSubgroupBarrier(pcUniformity, "gpu.shuffle", loc));
        env.uniformity[shfl.getShuffleResult()] = SIMTUniformity::LaneDivergent;
        env.exprs[shfl.getShuffleResult()] = AffineSymbolicExpr::nonAffine();
        env.uniformity[shfl.getValid()] = SIMTUniformity::SubgroupUniform;
        env.exprs[shfl.getValid()] = AffineSymbolicExpr::nonAffine();
        continue;
      }

      if (auto devInt = dyn_cast<DeviceIntrinsicOp>(op)) {
        const StringRef kind = devInt.getKind();
        if (kind == "global_id") {
          env.exprs[devInt.getResult(0)] =
              AffineSymbolicExpr::makeGlobalLinearIdX();
          env.uniformity[devInt.getResult(0)] = SIMTUniformity::LaneDivergent;
        } else if (kind == "thread_id") {
          env.exprs[devInt.getResult(0)] =
              AffineSymbolicExpr::makeVar(SymbolicVarKind::ThreadIdX);
          env.uniformity[devInt.getResult(0)] = SIMTUniformity::LaneDivergent;
        } else if (kind == "block_id") {
          env.exprs[devInt.getResult(0)] =
              AffineSymbolicExpr::makeVar(SymbolicVarKind::BlockIdX);
          env.uniformity[devInt.getResult(0)] =
              SIMTUniformity::WorkgroupUniform;
        } else if (kind == "block_dim") {
          env.exprs[devInt.getResult(0)] =
              AffineSymbolicExpr::makeVar(SymbolicVarKind::BlockDimX);
          env.uniformity[devInt.getResult(0)] =
              SIMTUniformity::WorkgroupUniform;
        } else if (kind == "barrier") {
          out.push_back(ProtocolStmt::makeWorkgroupBarrier(pcUniformity, loc));
        } else if (kind == "warp_barrier") {
          out.push_back(ProtocolStmt::makeSubgroupBarrier(
              pcUniformity, "gpu::warp_barrier", loc));
        } else if (kind == "load" && devInt.getNumOperands() >= 2) {
          if (auto mt = dyn_cast<MemRefType>(devInt.getOperand(0).getType())) {
            if (auto buf = resolveBuffer(devInt.getOperand(0), mt, env)) {
              out.push_back(ProtocolStmt::makeAccess(
                  MemoryAccessMode::Read, buf->name, buf->space,
                  valExpr(devInt.getOperand(1), env), loc, buf->staticExtent,
                  buf->dynamicExtentParam));
            }
          }
          if (devInt.getNumResults() > 0) {
            env.uniformity[devInt.getResult(0)] =
                valUniformity(devInt.getOperand(1), env);
            env.exprs[devInt.getResult(0)] = AffineSymbolicExpr::nonAffine();
          }
        } else if (kind == "store" && devInt.getNumOperands() >= 3) {
          if (auto mt = dyn_cast<MemRefType>(devInt.getOperand(0).getType())) {
            if (auto buf = resolveBuffer(devInt.getOperand(0), mt, env)) {
              out.push_back(ProtocolStmt::makeAccess(
                  MemoryAccessMode::Write, buf->name, buf->space,
                  valExpr(devInt.getOperand(1), env), loc, buf->staticExtent,
                  buf->dynamicExtentParam));
            }
          }
        }
        continue;
      }

      if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
              arith::ExtUIOp, arith::TruncIOp>(op)) {
        Value src = op->getOperand(0);
        Value dst = op->getResult(0);
        env.exprs[dst] = valExpr(src, env);
        env.uniformity[dst] = valUniformity(src, env);
        if (env.controlTainted.contains(src)) env.controlTainted.insert(dst);
        continue;
      }

      if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
        Value l = addOp.getLhs(), r = addOp.getRhs(), res = addOp.getResult();
        env.exprs[res] = addExpr(valExpr(l, env), valExpr(r, env));
        env.uniformity[res] =
            joinUniformity(valUniformity(l, env), valUniformity(r, env));
        if (env.controlTainted.contains(l) || env.controlTainted.contains(r))
          env.controlTainted.insert(res);
        continue;
      }

      if (auto subOp = dyn_cast<arith::SubIOp>(op)) {
        Value l = subOp.getLhs(), r = subOp.getRhs(), res = subOp.getResult();
        env.exprs[res] = subExpr(valExpr(l, env), valExpr(r, env));
        env.uniformity[res] =
            joinUniformity(valUniformity(l, env), valUniformity(r, env));
        if (env.controlTainted.contains(l) || env.controlTainted.contains(r))
          env.controlTainted.insert(res);
        continue;
      }

      if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
        Value l = mulOp.getLhs(), r = mulOp.getRhs(), res = mulOp.getResult();
        const auto lE = valExpr(l, env), rE = valExpr(r, env);
        env.exprs[res] = mulExpr(lE, rE, options_.subgroupSize);
        env.uniformity[res] =
            ((lE.isConstant() && lE.constant == 0) ||
             (rE.isConstant() && rE.constant == 0))
                ? SIMTUniformity::WorkgroupUniform
                : joinUniformity(valUniformity(l, env), valUniformity(r, env));
        if (env.controlTainted.contains(l) || env.controlTainted.contains(r))
          env.controlTainted.insert(res);
        continue;
      }

      if (isa<arith::DivSIOp, arith::DivUIOp>(op)) {
        Value l = op->getOperand(0), r = op->getOperand(1),
              res = op->getResult(0);
        const auto dE =
            divExpr(valExpr(l, env), valExpr(r, env), options_.subgroupSize);
        const SIMTUniformity rawU =
            joinUniformity(valUniformity(l, env), valUniformity(r, env));
        const bool tainted =
            env.controlTainted.contains(l) || env.controlTainted.contains(r);
        env.exprs[res] = dE;
        env.uniformity[res] =
            tainted ? rawU : deriveExprUniformity(dE, env, rawU);
        if (tainted) env.controlTainted.insert(res);
        continue;
      }

      if (isa<arith::RemSIOp, arith::RemUIOp>(op)) {
        Value l = op->getOperand(0), r = op->getOperand(1),
              res = op->getResult(0);
        env.exprs[res] =
            remExpr(valExpr(l, env), valExpr(r, env), options_.subgroupSize);
        env.uniformity[res] =
            joinUniformity(valUniformity(l, env), valUniformity(r, env));
        if (env.controlTainted.contains(l) || env.controlTainted.contains(r))
          env.controlTainted.insert(res);
        continue;
      }

      if (auto shl = dyn_cast<arith::ShLIOp>(op)) {
        Value l = shl.getLhs(), r = shl.getRhs(), res = shl.getResult();
        const auto rE = valExpr(r, env);
        if (rE.isConstant() && rE.constant >= 0 && rE.constant < 62) {
          env.exprs[res] = mulExpr(
              valExpr(l, env),
              AffineSymbolicExpr::makeConstant(std::int64_t{1} << rE.constant),
              options_.subgroupSize);
        } else {
          env.exprs[res] = AffineSymbolicExpr::nonAffine();
        }
        env.uniformity[res] =
            joinUniformity(valUniformity(l, env), valUniformity(r, env));
        if (env.controlTainted.contains(l) || env.controlTainted.contains(r))
          env.controlTainted.insert(res);
        continue;
      }

      if (isa<arith::ShRSIOp, arith::ShRUIOp>(op)) {
        Value l = op->getOperand(0), r = op->getOperand(1),
              res = op->getResult(0);
        const auto rE = valExpr(r, env);
        const SIMTUniformity rawU =
            joinUniformity(valUniformity(l, env), valUniformity(r, env));
        const bool tainted =
            env.controlTainted.contains(l) || env.controlTainted.contains(r);
        if (rE.isConstant() && rE.constant >= 0 && rE.constant < 62) {
          const auto dE = divExpr(
              valExpr(l, env),
              AffineSymbolicExpr::makeConstant(std::int64_t{1} << rE.constant),
              options_.subgroupSize);
          env.exprs[res] = dE;
          env.uniformity[res] =
              tainted ? rawU : deriveExprUniformity(dE, env, rawU);
        } else {
          env.exprs[res] = AffineSymbolicExpr::nonAffine();
          env.uniformity[res] = rawU;
        }
        if (tainted) env.controlTainted.insert(res);
        continue;
      }

      if (auto andOp = dyn_cast<arith::AndIOp>(op)) {
        Value l = andOp.getLhs(), r = andOp.getRhs(), res = andOp.getResult();
        env.uniformity[res] =
            joinUniformity(valUniformity(l, env), valUniformity(r, env));
        if (env.controlTainted.contains(l) || env.controlTainted.contains(r))
          env.controlTainted.insert(res);
        if (res.getType().isInteger(1)) {
          GuardPair gp;
          const GuardPair gL = valGuard(l, env);
          const GuardPair gR = valGuard(r, env);
          gp.whenTrue.contradiction =
              gL.whenTrue.contradiction || gR.whenTrue.contradiction;
          gp.whenTrue.constraints = gL.whenTrue.constraints;
          gp.whenTrue.constraints.insert(gp.whenTrue.constraints.end(),
                                         gR.whenTrue.constraints.begin(),
                                         gR.whenTrue.constraints.end());
          env.guards[res] = std::move(gp);
          env.exprs[res] = AffineSymbolicExpr::nonAffine();
        } else {
          const auto rE = valExpr(r, env);
          if (rE.isConstant() && rE.constant > 0 &&
              (rE.constant & (rE.constant + 1)) == 0) {
            env.exprs[res] =
                remExpr(valExpr(l, env),
                        AffineSymbolicExpr::makeConstant(rE.constant + 1),
                        options_.subgroupSize);
          } else {
            env.exprs[res] = AffineSymbolicExpr::nonAffine();
          }
        }
        continue;
      }

      if (auto xorOp = dyn_cast<arith::XOrIOp>(op)) {
        Value l = xorOp.getLhs(), r = xorOp.getRhs(), res = xorOp.getResult();
        env.uniformity[res] =
            joinUniformity(valUniformity(l, env), valUniformity(r, env));
        if (env.controlTainted.contains(l) || env.controlTainted.contains(r))
          env.controlTainted.insert(res);
        env.exprs[res] = AffineSymbolicExpr::nonAffine();
        if (res.getType().isInteger(1)) {
          const auto rE = valExpr(r, env);
          if (rE.isConstant() && rE.constant == 1) {
            const GuardPair gL = valGuard(l, env);
            env.guards[res] = GuardPair{gL.whenFalse, gL.whenTrue};
          }
        }
        continue;
      }

      if (auto selOp = dyn_cast<arith::SelectOp>(op)) {
        Value c = selOp.getCondition(), t = selOp.getTrueValue(),
              f = selOp.getFalseValue(), res = selOp.getResult();
        const SIMTUniformity cU = valUniformity(c, env);
        env.uniformity[res] = joinUniformity(
            cU, joinUniformity(valUniformity(t, env), valUniformity(f, env)));
        env.exprs[res] = AffineSymbolicExpr::nonAffine();
        if (cU != SIMTUniformity::WorkgroupUniform ||
            env.controlTainted.contains(c) || env.controlTainted.contains(t) ||
            env.controlTainted.contains(f)) {
          env.controlTainted.insert(res);
        }
        continue;
      }

      if (auto cmpOp = dyn_cast<arith::CmpIOp>(op)) {
        Value l = cmpOp.getLhs(), r = cmpOp.getRhs(), res = cmpOp.getResult();
        env.uniformity[res] =
            joinUniformity(valUniformity(l, env), valUniformity(r, env));
        if (env.controlTainted.contains(l) || env.controlTainted.contains(r))
          env.controlTainted.insert(res);
        env.exprs[res] = AffineSymbolicExpr::nonAffine();

        std::string_view posOp, negOp;
        switch (cmpOp.getPredicate()) {
        case arith::CmpIPredicate::eq: posOp = "=="; negOp = "!="; break;
        case arith::CmpIPredicate::ne: posOp = "!="; negOp = "=="; break;
        case arith::CmpIPredicate::slt:
        case arith::CmpIPredicate::ult: posOp = "<"; negOp = ">="; break;
        case arith::CmpIPredicate::sle:
        case arith::CmpIPredicate::ule: posOp = "<="; negOp = ">"; break;
        case arith::CmpIPredicate::sgt:
        case arith::CmpIPredicate::ugt: posOp = ">"; negOp = "<="; break;
        case arith::CmpIPredicate::sge:
        case arith::CmpIPredicate::uge: posOp = ">="; negOp = "<"; break;
        }
        GuardPair gp;
        gp.whenTrue.addComparison(posOp, valExpr(l, env), valExpr(r, env));
        gp.whenFalse.addComparison(negOp, valExpr(l, env), valExpr(r, env));
        env.guards[res] = std::move(gp);
        continue;
      }

      if (auto affApply = dyn_cast<affine::AffineApplyOp>(op)) {
        AffineMap map = affApply.getAffineMap();
        OperandRange ops = affApply.getMapOperands();
        OperandRange dims = ops.take_front(map.getNumDims());
        OperandRange syms = ops.drop_front(map.getNumDims());
        if (map.getNumResults() == 1) {
          auto [e, u] = evalAffineExpr(map.getResult(0), dims, syms, env);
          env.exprs[affApply.getResult()] = std::move(e);
          env.uniformity[affApply.getResult()] = u;
        } else {
          env.exprs[affApply.getResult()] = AffineSymbolicExpr::nonAffine();
          env.uniformity[affApply.getResult()] = SIMTUniformity::LaneDivergent;
        }
        continue;
      }

      if (auto allocaOp = dyn_cast<memref::AllocaOp>(op)) {
        if (allocaOp.getType().getRank() == 0) {
          env.slotUniformity[allocaOp.getResult()] = pcUniformity;
          env.slotExprs[allocaOp.getResult()] = AffineSymbolicExpr::nonAffine();
        } else {
          (void)resolveBuffer(allocaOp.getResult(), allocaOp.getType(), env);
        }
        continue;
      }

      if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
        (void)resolveBuffer(allocOp.getResult(), allocOp.getType(), env);
        continue;
      }

      if (auto getGlob = dyn_cast<memref::GetGlobalOp>(op)) {
        (void)resolveBuffer(getGlob.getResult(), getGlob.getType(), env);
        continue;
      }

      if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
        Value base = unwrapMemRefBase(loadOp.getMemref());
        if (loadOp.getMemRefType().getRank() == 0 &&
            isa_and_nonnull<memref::AllocaOp>(base.getDefiningOp())) {
          Value res = loadOp.getResult();
          const SIMTUniformity sU = env.slotUniformity.lookup(base);
          env.uniformity[res] = sU;
          env.exprs[res] = env.slotExprs.contains(base)
                               ? env.slotExprs[base]
                               : AffineSymbolicExpr::nonAffine();
          if (env.slotGuards.contains(base)) env.guards[res] = env.slotGuards[base];
          if (sU != SIMTUniformity::WorkgroupUniform)
            env.controlTainted.insert(res);
          continue;
        }
        SIMTUniformity idxU = SIMTUniformity::WorkgroupUniform;
        SmallVector<AffineSymbolicExpr, 4> idxExprs;
        for (Value idx : loadOp.getIndices()) {
          idxExprs.push_back(valExpr(idx, env));
          idxU = joinUniformity(idxU, valUniformity(idx, env));
        }
        if (auto buf = resolveBuffer(loadOp.getMemref(), loadOp.getMemRefType(),
                                     env)) {
          out.push_back(ProtocolStmt::makeAccess(
              MemoryAccessMode::Read, buf->name, buf->space,
              linearizeIndices(idxExprs, *buf), loc, buf->staticExtent,
              buf->dynamicExtentParam));
        }
        env.uniformity[loadOp.getResult()] = idxU;
        env.exprs[loadOp.getResult()] = AffineSymbolicExpr::nonAffine();
        continue;
      }

      if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
        Value base = unwrapMemRefBase(storeOp.getMemref());
        if (storeOp.getMemRefType().getRank() == 0 &&
            isa_and_nonnull<memref::AllocaOp>(base.getDefiningOp())) {
          Value val = storeOp.getValueToStore();
          const SIMTUniformity newU = joinUniformity(
              env.slotUniformity.lookup(base),
              joinUniformity(pcUniformity, valUniformity(val, env)));
          env.slotUniformity[base] = newU;
          env.slotExprs[base] = (pcUniformity == SIMTUniformity::WorkgroupUniform)
                                    ? valExpr(val, env)
                                    : AffineSymbolicExpr::nonAffine();
          if (pcUniformity == SIMTUniformity::WorkgroupUniform &&
              env.guards.contains(val)) {
            env.slotGuards[base] = env.guards[val];
          } else {
            env.slotGuards.erase(base);
          }
          continue;
        }
        SmallVector<AffineSymbolicExpr, 4> idxExprs;
        for (Value idx : storeOp.getIndices()) {
          idxExprs.push_back(valExpr(idx, env));
        }
        if (auto buf = resolveBuffer(storeOp.getMemref(),
                                     storeOp.getMemRefType(), env)) {
          out.push_back(ProtocolStmt::makeAccess(
              MemoryAccessMode::Write, buf->name, buf->space,
              linearizeIndices(idxExprs, *buf), loc, buf->staticExtent,
              buf->dynamicExtentParam));
        }
        continue;
      }

      if (auto affLoad = dyn_cast<affine::AffineLoadOp>(op)) {
        AffineMap map = affLoad.getAffineMap();
        OperandRange ops = affLoad.getMapOperands();
        OperandRange dims = ops.take_front(map.getNumDims());
        OperandRange syms = ops.drop_front(map.getNumDims());
        SmallVector<AffineSymbolicExpr, 4> idxExprs;
        SIMTUniformity idxU = SIMTUniformity::WorkgroupUniform;
        for (AffineExpr r : map.getResults()) {
          auto [e, u] = evalAffineExpr(r, dims, syms, env);
          idxExprs.push_back(std::move(e));
          idxU = joinUniformity(idxU, u);
        }
        if (auto buf = resolveBuffer(affLoad.getMemRef(),
                                     affLoad.getMemRefType(), env)) {
          out.push_back(ProtocolStmt::makeAccess(
              MemoryAccessMode::Read, buf->name, buf->space,
              linearizeIndices(idxExprs, *buf), loc, buf->staticExtent,
              buf->dynamicExtentParam));
        }
        env.uniformity[affLoad.getResult()] = idxU;
        env.exprs[affLoad.getResult()] = AffineSymbolicExpr::nonAffine();
        continue;
      }

      if (auto affStore = dyn_cast<affine::AffineStoreOp>(op)) {
        AffineMap map = affStore.getAffineMap();
        OperandRange ops = affStore.getMapOperands();
        OperandRange dims = ops.take_front(map.getNumDims());
        OperandRange syms = ops.drop_front(map.getNumDims());
        SmallVector<AffineSymbolicExpr, 4> idxExprs;
        for (AffineExpr r : map.getResults()) {
          auto [e, _] = evalAffineExpr(r, dims, syms, env);
          idxExprs.push_back(std::move(e));
        }
        if (auto buf = resolveBuffer(affStore.getMemRef(),
                                     affStore.getMemRefType(), env)) {
          out.push_back(ProtocolStmt::makeAccess(
              MemoryAccessMode::Write, buf->name, buf->space,
              linearizeIndices(idxExprs, *buf), loc, buf->staticExtent,
              buf->dynamicExtentParam));
        }
        continue;
      }

      if (auto scfIf = dyn_cast<scf::IfOp>(op)) {
        const SIMTUniformity condU = valUniformity(scfIf.getCondition(), env);
        const SIMTUniformity branchPC = joinUniformity(pcUniformity, condU);
        const GuardPair gp = valGuard(scfIf.getCondition(), env);
        const Env preEnv = env;
        Env thenEnv = preEnv;
        Env elseEnv = preEnv;
        std::vector<ProtocolStmt> thenStmts;
        std::vector<ProtocolStmt> elseStmts;

        if (!scfIf.getThenRegion().empty()) {
          extractBlock(scfIf.getThenRegion().front(), branchPC, thenEnv,
                       thenStmts, retSummary);
        }
        if (!scfIf.getElseRegion().empty()) {
          extractBlock(scfIf.getElseRegion().front(), branchPC, elseEnv,
                       elseStmts, retSummary);
        }
        mergeBranchSlots(branchPC, preEnv, thenEnv, elseEnv, env);

        Operation *thenTerm = scfIf.getThenRegion().empty()
                                  ? nullptr
                                  : scfIf.getThenRegion().front().getTerminator();
        Operation *elseTerm = scfIf.getElseRegion().empty()
                                  ? nullptr
                                  : scfIf.getElseRegion().front().getTerminator();
        for (auto [idx, res] : llvm::enumerate(scfIf.getResults())) {
          SIMTUniformity rU = condU;
          if (thenTerm && idx < thenTerm->getNumOperands())
            rU = joinUniformity(rU,
                                valUniformity(thenTerm->getOperand(idx), thenEnv));
          if (elseTerm && idx < elseTerm->getNumOperands())
            rU = joinUniformity(rU,
                                valUniformity(elseTerm->getOperand(idx), elseEnv));
          env.uniformity[res] = rU;
          env.exprs[res] = AffineSymbolicExpr::nonAffine();
          if (condU != SIMTUniformity::WorkgroupUniform)
            env.controlTainted.insert(res);
        }

        out.push_back(ProtocolStmt::makeConditional(
            condU, gp.whenTrue, gp.whenFalse, std::move(thenStmts),
            std::move(elseStmts), loc));
        continue;
      }

      if (auto affIf = dyn_cast<affine::AffineIfOp>(op)) {
        IntegerSet set = affIf.getIntegerSet();
        OperandRange ops = affIf.getOperands();
        OperandRange dims = ops.take_front(set.getNumDims());
        OperandRange syms = ops.drop_front(set.getNumDims());

        SIMTUniformity condU = SIMTUniformity::WorkgroupUniform;
        SymbolicGuard thenG, elseG;
        for (unsigned c = 0; c < set.getNumConstraints(); ++c) {
          auto [cE, cU] = evalAffineExpr(set.getConstraint(c), dims, syms, env);
          condU = joinUniformity(condU, cU);
          if (set.isEq(c)) {
            thenG.addComparison("==", cE, AffineSymbolicExpr::makeConstant(0));
            if (set.getNumConstraints() == 1)
              elseG.addComparison("!=", cE, AffineSymbolicExpr::makeConstant(0));
          } else {
            thenG.addComparison(">=", cE, AffineSymbolicExpr::makeConstant(0));
            if (set.getNumConstraints() == 1)
              elseG.addComparison("<", cE, AffineSymbolicExpr::makeConstant(0));
          }
        }

        const SIMTUniformity branchPC = joinUniformity(pcUniformity, condU);
        const Env preEnv = env;
        Env thenEnv = preEnv;
        Env elseEnv = preEnv;
        std::vector<ProtocolStmt> thenStmts;
        std::vector<ProtocolStmt> elseStmts;

        if (!affIf.getThenRegion().empty()) {
          extractBlock(affIf.getThenRegion().front(), branchPC, thenEnv,
                       thenStmts, retSummary);
        }
        if (!affIf.getElseRegion().empty()) {
          extractBlock(affIf.getElseRegion().front(), branchPC, elseEnv,
                       elseStmts, retSummary);
        }
        mergeBranchSlots(branchPC, preEnv, thenEnv, elseEnv, env);

        Operation *thenTerm = affIf.getThenRegion().empty()
                                  ? nullptr
                                  : affIf.getThenRegion().front().getTerminator();
        Operation *elseTerm = affIf.getElseRegion().empty()
                                  ? nullptr
                                  : affIf.getElseRegion().front().getTerminator();
        for (auto [idx, res] : llvm::enumerate(affIf.getResults())) {
          SIMTUniformity rU = condU;
          if (thenTerm && idx < thenTerm->getNumOperands())
            rU = joinUniformity(rU,
                                valUniformity(thenTerm->getOperand(idx), thenEnv));
          if (elseTerm && idx < elseTerm->getNumOperands())
            rU = joinUniformity(rU,
                                valUniformity(elseTerm->getOperand(idx), elseEnv));
          env.uniformity[res] = rU;
          env.exprs[res] = AffineSymbolicExpr::nonAffine();
          if (condU != SIMTUniformity::WorkgroupUniform)
            env.controlTainted.insert(res);
        }

        out.push_back(ProtocolStmt::makeConditional(
            condU, std::move(thenG), std::move(elseG), std::move(thenStmts),
            std::move(elseStmts), loc));
        continue;
      }

      if (auto scfFor = dyn_cast<scf::ForOp>(op)) {
        const auto lbE = valExpr(scfFor.getLowerBound(), env);
        const auto ubE = valExpr(scfFor.getUpperBound(), env);
        const auto stE = valExpr(scfFor.getStep(), env);
        const SIMTUniformity headerU = joinUniformity(
            valUniformity(scfFor.getLowerBound(), env),
            joinUniformity(valUniformity(scfFor.getUpperBound(), env),
                           valUniformity(scfFor.getStep(), env)));
        const SIMTUniformity bodyPC = joinUniformity(pcUniformity, headerU);
        const std::string ivName = "iv_" + std::to_string(nextLoopId_++);
        const std::int64_t stepVal =
            (stE.isConstant() && stE.constant > 0) ? stE.constant : 1;
        const bool bounded = stE.isConstant() && stE.constant > 0;

        Env loopEnv = env;
        loopEnv.exprs[scfFor.getInductionVar()] =
            AffineSymbolicExpr::makeVar(SymbolicVarKind::LoopVar, ivName);
        loopEnv.uniformity[scfFor.getInductionVar()] = headerU;
        loopEnv.loopVarUniformity[ivName] = headerU;
        for (auto [initV, iterArg] :
             llvm::zip(scfFor.getInitArgs(), scfFor.getRegionIterArgs())) {
          loopEnv.uniformity[iterArg] =
              joinUniformity(headerU, valUniformity(initV, env));
          loopEnv.exprs[iterArg] = AffineSymbolicExpr::nonAffine();
        }

        std::vector<ProtocolStmt> bodyStmts;
        extractBlock(*scfFor.getBody(), bodyPC, loopEnv, bodyStmts, retSummary);

        Operation *yieldOp = scfFor.getBody()->getTerminator();
        for (auto [idx, res] : llvm::enumerate(scfFor.getResults())) {
          SIMTUniformity yU = headerU;
          if (idx < scfFor.getInitArgs().size())
            yU = joinUniformity(yU, valUniformity(scfFor.getInitArgs()[idx], env));
          if (yieldOp && idx < yieldOp->getNumOperands())
            yU = joinUniformity(yU,
                                valUniformity(yieldOp->getOperand(idx), loopEnv));
          env.uniformity[res] = yU;
          env.exprs[res] = AffineSymbolicExpr::nonAffine();
        }
        mergeBranchSlots(bodyPC, env, loopEnv, env, env);

        out.push_back(ProtocolStmt::makeLoop(ivName, lbE, ubE, stepVal, headerU,
                                             bounded, std::move(bodyStmts), loc));
        continue;
      }

      if (auto affFor = dyn_cast<affine::AffineForOp>(op)) {
        AffineMap lbMap = affFor.getLowerBoundMap();
        OperandRange lbOps = affFor.getLowerBoundOperands();
        AffineMap ubMap = affFor.getUpperBoundMap();
        OperandRange ubOps = affFor.getUpperBoundOperands();

        AffineSymbolicExpr lbE = AffineSymbolicExpr::nonAffine();
        SIMTUniformity lbU = SIMTUniformity::WorkgroupUniform;
        if (lbMap.getNumResults() == 1) {
          std::tie(lbE, lbU) = evalAffineExpr(
              lbMap.getResult(0), lbOps.take_front(lbMap.getNumDims()),
              lbOps.drop_front(lbMap.getNumDims()), env);
        } else {
          for (Value v : lbOps) lbU = joinUniformity(lbU, valUniformity(v, env));
        }

        AffineSymbolicExpr ubE = AffineSymbolicExpr::nonAffine();
        SIMTUniformity ubU = SIMTUniformity::WorkgroupUniform;
        if (ubMap.getNumResults() == 1) {
          std::tie(ubE, ubU) = evalAffineExpr(
              ubMap.getResult(0), ubOps.take_front(ubMap.getNumDims()),
              ubOps.drop_front(ubMap.getNumDims()), env);
        } else {
          for (Value v : ubOps) ubU = joinUniformity(ubU, valUniformity(v, env));
        }

        const SIMTUniformity headerU = joinUniformity(lbU, ubU);
        const SIMTUniformity bodyPC = joinUniformity(pcUniformity, headerU);
        const std::string ivName = "iv_" + std::to_string(nextLoopId_++);
        const std::int64_t stepVal = affFor.getStepAsInt();

        Env loopEnv = env;
        loopEnv.exprs[affFor.getInductionVar()] =
            AffineSymbolicExpr::makeVar(SymbolicVarKind::LoopVar, ivName);
        loopEnv.uniformity[affFor.getInductionVar()] = headerU;
        loopEnv.loopVarUniformity[ivName] = headerU;
        for (auto [initV, iterArg] :
             llvm::zip(affFor.getInits(), affFor.getRegionIterArgs())) {
          loopEnv.uniformity[iterArg] =
              joinUniformity(headerU, valUniformity(initV, env));
          loopEnv.exprs[iterArg] = AffineSymbolicExpr::nonAffine();
        }

        std::vector<ProtocolStmt> bodyStmts;
        extractBlock(*affFor.getBody(), bodyPC, loopEnv, bodyStmts, retSummary);

        Operation *yieldOp = affFor.getBody()->getTerminator();
        for (auto [idx, res] : llvm::enumerate(affFor.getResults())) {
          SIMTUniformity yU = headerU;
          if (idx < affFor.getInits().size())
            yU = joinUniformity(yU, valUniformity(affFor.getInits()[idx], env));
          if (yieldOp && idx < yieldOp->getNumOperands())
            yU = joinUniformity(yU,
                                valUniformity(yieldOp->getOperand(idx), loopEnv));
          env.uniformity[res] = yU;
          env.exprs[res] = AffineSymbolicExpr::nonAffine();
        }
        mergeBranchSlots(bodyPC, env, loopEnv, env, env);

        out.push_back(ProtocolStmt::makeLoop(ivName, lbE, ubE, stepVal, headerU,
                                             /*boundedFor=*/true,
                                             std::move(bodyStmts), loc));
        continue;
      }

      if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
        SmallVector<SIMTUniformity> argU;
        for (Value initV : whileOp.getInits())
          argU.push_back(valUniformity(initV, env));

        SIMTUniformity headerU = SIMTUniformity::WorkgroupUniform;
        for (SIMTUniformity u : argU) headerU = joinUniformity(headerU, u);

        // Two-pass fixed-point so loop-carried or exit-condition divergence
        // taints barriers anywhere inside `before` or `after`.
        std::vector<ProtocolStmt> bodyStmts;
        Env finalLoopEnv = env;
        for (int pass = 0; pass < 2; ++pass) {
          bodyStmts.clear();
          Env loopEnv = env;
          for (auto [idx, bArg] : llvm::enumerate(whileOp.getBeforeArguments())) {
            loopEnv.uniformity[bArg] = joinUniformity(headerU, argU[idx]);
            loopEnv.exprs[bArg] = AffineSymbolicExpr::nonAffine();
          }
          const SIMTUniformity curPC = joinUniformity(pcUniformity, headerU);
          extractBlock(whileOp.getBefore().front(), curPC, loopEnv, bodyStmts,
                       retSummary);
          auto condOp =
              cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
          headerU =
              joinUniformity(headerU, valUniformity(condOp.getCondition(), loopEnv));

          const SIMTUniformity afterPC = joinUniformity(pcUniformity, headerU);
          for (auto [cVal, aArg] :
               llvm::zip(condOp.getArgs(), whileOp.getAfterArguments())) {
            loopEnv.uniformity[aArg] =
                joinUniformity(headerU, valUniformity(cVal, loopEnv));
            loopEnv.exprs[aArg] = AffineSymbolicExpr::nonAffine();
          }
          extractBlock(whileOp.getAfter().front(), afterPC, loopEnv, bodyStmts,
                       retSummary);
          auto yieldOp =
              cast<scf::YieldOp>(whileOp.getAfter().front().getTerminator());
          for (auto [idx, yVal] : llvm::enumerate(yieldOp.getOperands())) {
            if (idx < argU.size()) {
              argU[idx] = joinUniformity(argU[idx], valUniformity(yVal, loopEnv));
              headerU = joinUniformity(headerU, argU[idx]);
            }
          }
          finalLoopEnv = std::move(loopEnv);
        }

        auto condOp =
            cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
        for (auto [cVal, res] :
             llvm::zip(condOp.getArgs(), whileOp.getResults())) {
          env.uniformity[res] =
              joinUniformity(headerU, valUniformity(cVal, finalLoopEnv));
          env.exprs[res] = AffineSymbolicExpr::nonAffine();
        }
        mergeBranchSlots(joinUniformity(pcUniformity, headerU), env,
                         finalLoopEnv, env, env);

        const std::string ivName = "while_" + std::to_string(nextLoopId_++);
        out.push_back(ProtocolStmt::makeLoop(
            ivName, AffineSymbolicExpr::nonAffine(),
            AffineSymbolicExpr::nonAffine(), 1, headerU, /*boundedFor=*/false,
            std::move(bodyStmts), loc));
        continue;
      }

      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        const std::string calleeName = callOp.getCallee().str();
        auto calleeIt = helperFuncs_.find(calleeName);
        if (calleeIt != helperFuncs_.end() &&
            !calleeIt->second.getBody().empty() &&
            calleeIt->second.getBody().hasOneBlock() &&
            !activeCallStack_.contains(calleeName)) {
          activeCallStack_.insert(calleeName);
          func::FuncOp callee = calleeIt->second;
          Block &calleeEntry = callee.getBody().front();
          Env calleeEnv;
          calleeEnv.loopVarUniformity = env.loopVarUniformity;
          for (auto [operand, param] :
               llvm::zip(callOp.getOperands(), calleeEntry.getArguments())) {
            if (auto mt = dyn_cast<MemRefType>(operand.getType())) {
              if (auto buf = resolveBuffer(operand, mt, env))
                calleeEnv.buffers[param] = *buf;
            } else {
              calleeEnv.exprs[param] = valExpr(operand, env);
              calleeEnv.uniformity[param] = valUniformity(operand, env);
              if (env.guards.contains(operand))
                calleeEnv.guards[param] = env.guards[operand];
              if (env.controlTainted.contains(operand))
                calleeEnv.controlTainted.insert(param);
            }
          }
          CallReturnSummary subRet;
          extractBlock(calleeEntry, pcUniformity, calleeEnv, out, &subRet);
          activeCallStack_.erase(calleeName);
          if (callOp.getNumResults() == 1 && subRet.seen) {
            env.exprs[callOp.getResult(0)] = subRet.expr;
            env.uniformity[callOp.getResult(0)] = subRet.uniformity;
          } else {
            for (Value r : callOp.getResults()) {
              env.exprs[r] = AffineSymbolicExpr::nonAffine();
              env.uniformity[r] = pcUniformity;
            }
          }
          continue;
        }
      }

      if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
        if (retSummary && retOp.getNumOperands() == 1) {
          Value rv = retOp.getOperand(0);
          const auto rE = valExpr(rv, env);
          const SIMTUniformity rU =
              joinUniformity(pcUniformity, valUniformity(rv, env));
          if (!retSummary->seen) {
            retSummary->expr = rE;
            retSummary->uniformity = rU;
            retSummary->seen = true;
          } else {
            retSummary->expr = AffineSymbolicExpr::nonAffine();
            retSummary->uniformity = joinUniformity(retSummary->uniformity, rU);
          }
        }
        continue;
      }

      // Default propagation for floating-point / miscellaneous pure ops.
      if (op->getNumResults() > 0 && op->getNumRegions() == 0) {
        SIMTUniformity u = SIMTUniformity::WorkgroupUniform;
        bool tainted = false;
        for (Value operand : op->getOperands()) {
          u = joinUniformity(u, valUniformity(operand, env));
          if (env.controlTainted.contains(operand)) tainted = true;
        }
        for (Value res : op->getResults()) {
          env.uniformity[res] = u;
          env.exprs[res] = AffineSymbolicExpr::nonAffine();
          if (tainted) env.controlTainted.insert(res);
        }
      }
    }
  }
};

} // namespace

LogicalResult verifyGPUBarrierUniformityAndAccessProtocols(
    ModuleOp module, const verification::GPUMemoryAccessProtocolOptions &options) {
  return MLIRAccessProtocolVerifier(module, options).run();
}

namespace {

std::optional<std::size_t> parseAttrUnsigned(StringRef text) {
  if (text.empty()) return std::nullopt;
  unsigned long long v = 0;
  if (text.getAsInteger(10, v)) return std::nullopt;
  return static_cast<std::size_t>(v);
}

std::optional<verification::GPUPublicationOrder>
parseMLIRPublication(StringRef text) {
  if (text == "relaxed") return verification::GPUPublicationOrder::Relaxed;
  if (text == "acquire_workgroup")
    return verification::GPUPublicationOrder::AcquireWorkgroup;
  if (text == "release_workgroup")
    return verification::GPUPublicationOrder::ReleaseWorkgroup;
  if (text == "acq_rel_workgroup")
    return verification::GPUPublicationOrder::AcqRelWorkgroup;
  if (text == "acquire_device")
    return verification::GPUPublicationOrder::AcquireDevice;
  if (text == "release_device")
    return verification::GPUPublicationOrder::ReleaseDevice;
  if (text == "acq_rel_device")
    return verification::GPUPublicationOrder::AcqRelDevice;
  return std::nullopt;
}

std::string getDictStr(DictionaryAttr dict, StringRef key) {
  if (!dict) return {};
  if (auto s = dict.getAs<StringAttr>(key)) return s.getValue().str();
  return {};
}

// Reconstructs a `BehavioralAutomaton` from a `StageOp`'s `deferred_body` +
// `deferred_cfg` attributes so parsed or in-memory MLIR modules are verified
// directly without relying on an external GraphIR object.
verification::BehavioralAutomaton
extractMLIRStageAutomaton(StageOp stageOp, const std::string &qualifiedName,
                          const std::map<std::string, std::string> &portMap) {
  verification::BehavioralAutomaton aut;
  aut.name = qualifiedName;
  aut.location = mlirLocToSourceLoc(stageOp.getLoc());

  // Index deferred_body operations by integer `id`!
  std::map<std::int64_t, DictionaryAttr> opById;
  std::function<void(ArrayAttr)> indexDeferredOps = [&](ArrayAttr arr) {
    if (!arr) return;
    for (Attribute elem : arr) {
      auto d = dyn_cast<DictionaryAttr>(elem);
      if (!d) continue;
      if (auto idAttr = d.getAs<IntegerAttr>("id")) {
        opById[idAttr.getInt()] = d;
      }
      for (StringRef subKey : {"initialization", "condition", "body",
                               "alternative", "step", "offer_branches"}) {
        if (auto subArr = d.getAs<ArrayAttr>(subKey)) indexDeferredOps(subArr);
      }
    }
  };
  indexDeferredOps(stageOp.getDeferredBody());

  auto resolveEp = [&](const std::string &ep) -> std::string {
    auto it = portMap.find(ep);
    return it != portMap.end() ? it->second : ep;
  };

  ArrayAttr cfgBlocks = stageOp.getDeferredCfg();
  if (!cfgBlocks || cfgBlocks.empty()) {
    aut.states.push_back({0, "terminal", true, std::nullopt, aut.location});
    aut.initialState = 0;
    return aut;
  }

  std::map<std::int64_t, verification::AutomatonStateId> blockEntry;
  std::map<std::int64_t, DictionaryAttr> cfgBlockById;

  auto addState = [&](std::string lbl, bool term = false) {
    const verification::AutomatonStateId id = aut.states.size();
    aut.states.push_back(
        {id, std::move(lbl), term, std::nullopt, aut.location});
    return id;
  };

  std::optional<std::int64_t> firstBlockId;
  for (Attribute bAttr : cfgBlocks) {
    auto bDict = dyn_cast<DictionaryAttr>(bAttr);
    if (!bDict) continue;
    auto idAttr = bDict.getAs<IntegerAttr>("id");
    if (!idAttr) continue;
    const std::int64_t bid = idAttr.getInt();
    if (!firstBlockId) firstBlockId = bid;
    cfgBlockById[bid] = bDict;
    std::string lbl = "bb" + std::to_string(bid);
    if (auto props = bDict.getAs<DictionaryAttr>("properties")) {
      const std::string role = getDictStr(props, "role");
      if (!role.empty()) lbl += ":" + role;
    }
    blockEntry[bid] = addState(std::move(lbl), false);
  }

  const auto terminalId = addState("terminal", true);
  aut.initialState =
      (firstBlockId && blockEntry.contains(*firstBlockId))
          ? blockEntry[*firstBlockId]
          : terminalId;

  for (Attribute bAttr : cfgBlocks) {
    auto bDict = dyn_cast<DictionaryAttr>(bAttr);
    if (!bDict) continue;
    auto idAttr = bDict.getAs<IntegerAttr>("id");
    if (!idAttr) continue;
    verification::AutomatonStateId cursor = blockEntry[idAttr.getInt()];
    std::vector<std::string> pendingMarkers;

    if (auto opsArr = bDict.getAs<ArrayAttr>("operations")) {
      for (Attribute opIdAttr : opsArr) {
        auto intId = dyn_cast<IntegerAttr>(opIdAttr);
        if (!intId || !opById.contains(intId.getInt())) continue;
        DictionaryAttr opDict = opById[intId.getInt()];
        const std::string kind = getDictStr(opDict, "kind");
        DictionaryAttr props = opDict.getAs<DictionaryAttr>("properties");

        if (kind == "progress_marker") {
          const std::string mName = getDictStr(opDict, "name");
          if (!mName.empty()) {
            pendingMarkers.push_back(mName);
            aut.requiredProgressMarkers.insert(mName);
          }
          continue;
        }
        if (kind == "channel_send" || kind == "channel_recv" ||
            kind == "channel_receive") {
          const auto afterState =
              addState("post_" + std::to_string(intId.getInt()), false);
          verification::ProcessTransition tr;
          tr.source = cursor;
          tr.target = afterState;
          tr.kind = kind == "channel_send"
                        ? verification::BehavioralActionKind::Send
                        : verification::BehavioralActionKind::Receive;
          tr.endpoint = resolveEp(getDictStr(props, "endpoint"));
          tr.message = getDictStr(props, "message");
          tr.delivery = getDictStr(props, "delivery") == "same_step"
                            ? graphir::ChannelDeliveryMode::SameStep
                            : graphir::ChannelDeliveryMode::BlockingQueue;
          tr.transfersOwnership =
              getDictStr(props, "ownership_transfer") == "move";
          tr.progressMarkers = std::move(pendingMarkers);
          pendingMarkers.clear();
          aut.transitions.push_back(std::move(tr));
          cursor = afterState;
        }
      }
    }

    const std::string termKind = getDictStr(bDict, "terminator");
    if (termKind == "return") {
      verification::ProcessTransition tr;
      tr.source = cursor;
      tr.target = terminalId;
      tr.kind = verification::BehavioralActionKind::Internal;
      tr.progressMarkers = std::move(pendingMarkers);
      aut.transitions.push_back(std::move(tr));
    } else if (termKind == "branch") {
      auto succ = bDict.getAs<IntegerAttr>("successor");
      if (succ && blockEntry.contains(succ.getInt())) {
        verification::ProcessTransition tr;
        tr.source = cursor;
        tr.target = blockEntry[succ.getInt()];
        tr.kind = verification::BehavioralActionKind::Internal;
        tr.progressMarkers = std::move(pendingMarkers);
        aut.transitions.push_back(std::move(tr));
      }
    } else if (termKind == "conditional_branch") {
      auto tSucc = bDict.getAs<IntegerAttr>("successor");
      auto fSucc = bDict.getAs<IntegerAttr>("false_successor");
      if (tSucc && fSucc && blockEntry.contains(tSucc.getInt()) &&
          blockEntry.contains(fSucc.getInt())) {
        const std::string ctrl = getDictStr(bDict, "control");
        auto bProps = bDict.getAs<DictionaryAttr>("properties");
        const bool isOfferDispatch = StringRef(ctrl).starts_with("offer:");
        const bool isOfferCont = getDictStr(bProps, "role") == "offer.next";

        if (isOfferDispatch && !isOfferCont) {
          // Reconstruct non-blocking `AutomatonOfferGroup` across `offer.next`
          // chain and mark `offer.default` fallback edge.
          DictionaryAttr offerOpDict;
          for (const auto &[oid, od] : opById) {
            (void)oid;
            if (getDictStr(od, "kind") == "offer") {
              offerOpDict = od;
              break;
            }
          }
          verification::AutomatonOfferGroup group;
          if (offerOpDict) {
            auto oProps = offerOpDict.getAs<DictionaryAttr>("properties");
            group.mode = getDictStr(oProps, "offer_mode") == "conjunctive"
                             ? graphir::OfferMode::Conjunctive
                             : graphir::OfferMode::Exclusive;
          }
          ArrayAttr armArray =
              offerOpDict ? offerOpDict.getAs<ArrayAttr>("offer_branches")
                          : ArrayAttr{};
          std::int64_t curDisp = idAttr.getInt();
          std::size_t armIdx = 0;
          while (cfgBlockById.contains(curDisp)) {
            DictionaryAttr dBlk = cfgBlockById[curDisp];
            auto dTSucc = dBlk.getAs<IntegerAttr>("successor");
            auto dFSucc = dBlk.getAs<IntegerAttr>("false_successor");
            if (!dTSucc || !dFSucc || !blockEntry.contains(dTSucc.getInt()) ||
                !blockEntry.contains(dFSucc.getInt())) {
              break;
            }
            const auto armTarget = blockEntry[dTSucc.getInt()];
            if (armArray && armIdx < armArray.size()) {
              if (auto armDict = dyn_cast<DictionaryAttr>(armArray[armIdx])) {
                auto aProps = armDict.getAs<DictionaryAttr>("properties");
                const auto armKind =
                    getDictStr(aProps, "direction") == "send"
                        ? verification::BehavioralActionKind::Send
                        : verification::BehavioralActionKind::Receive;
                const std::string armEp =
                    resolveEp(getDictStr(aProps, "endpoint"));
                const std::string armMsg = getDictStr(aProps, "message");
                group.arms.push_back(
                    {armKind, armEp, armMsg, armTarget, aut.location});

                bool armHasExplicitChOp = false;
                if (cfgBlockById.contains(dTSucc.getInt())) {
                  if (auto armOps = cfgBlockById[dTSucc.getInt()]
                                        .getAs<ArrayAttr>("operations")) {
                    for (Attribute aOpAttr : armOps) {
                      auto aInt = dyn_cast<IntegerAttr>(aOpAttr);
                      if (aInt && opById.contains(aInt.getInt())) {
                        const std::string ak =
                            getDictStr(opById[aInt.getInt()], "kind");
                        if (ak == "channel_send" || ak == "channel_recv" ||
                            ak == "channel_receive") {
                          armHasExplicitChOp = true;
                          break;
                        }
                      }
                    }
                  }
                }
                verification::ProcessTransition armTr;
                armTr.source = cursor;
                armTr.target = armTarget;
                armTr.kind = armHasExplicitChOp
                                 ? verification::BehavioralActionKind::Internal
                                 : armKind;
                armTr.endpoint = armEp;
                armTr.message = armMsg;
                armTr.progressMarkers = pendingMarkers;
                aut.transitions.push_back(std::move(armTr));
              }
            }
            const std::int64_t fbId = dFSucc.getInt();
            if (!cfgBlockById.contains(fbId)) break;
            auto fbProps =
                cfgBlockById[fbId].getAs<DictionaryAttr>("properties");
            if (getDictStr(fbProps, "role") == "offer.next") {
              curDisp = fbId;
              ++armIdx;
              continue;
            }
            const auto defTarget = blockEntry[fbId];
            group.defaultTarget = defTarget;
            verification::ProcessTransition defTr;
            defTr.source = cursor;
            defTr.target = defTarget;
            defTr.kind = verification::BehavioralActionKind::Internal;
            defTr.isOfferDefault = true;
            defTr.progressMarkers = pendingMarkers;
            aut.transitions.push_back(std::move(defTr));
            break;
          }
          aut.states[cursor].offer = std::move(group);
          continue;
        }
        if (isOfferCont) continue;

        std::optional<std::string> fairTag;
        bool justified = false;
        if (StringRef(ctrl).starts_with("fair:")) {
          fairTag = ctrl.substr(5);
          if (cfgBlockById.contains(tSucc.getInt())) {
            auto tProps =
                cfgBlockById[tSucc.getInt()].getAs<DictionaryAttr>("properties");
            justified = (getDictStr(tProps, "assumption_justified") == "true");
          }
          if (!justified) aut.unprovedFairnessAssumptions.insert(*fairTag);
        }

        verification::ProcessTransition trT;
        trT.source = cursor;
        trT.target = blockEntry[tSucc.getInt()];
        trT.kind = verification::BehavioralActionKind::Internal;
        trT.fairTag = fairTag;
        trT.fairBranchOutcome = true;
        trT.assumptionJustified = justified;
        trT.progressMarkers = pendingMarkers;
        aut.transitions.push_back(std::move(trT));

        verification::ProcessTransition trF;
        trF.source = cursor;
        trF.target = blockEntry[fSucc.getInt()];
        trF.kind = verification::BehavioralActionKind::Internal;
        trF.fairTag = fairTag;
        trF.fairBranchOutcome = false;
        trF.assumptionJustified = justified;
        trF.progressMarkers = std::move(pendingMarkers);
        aut.transitions.push_back(std::move(trF));
      }
    }
  }

  return aut;
}

} // namespace

LogicalResult verifyGPUPipelineProtocols(
    ModuleOp module, const verification::GPUBoundedQueueOptions &options) {
  std::map<std::string, verification::ProtocolAutomatonSpec> protocols;
  for (ProtocolOp protoOp : module.getOps<ProtocolOp>()) {
    verification::ProtocolAutomatonSpec pSpec;
    pSpec.name = protoOp.getSymName().str();
    pSpec.location = mlirLocToSourceLoc(protoOp.getLoc());
    if (auto rolesArr =
            protoOp.getSemanticProperties().getAs<ArrayAttr>("roles")) {
      for (Attribute r : rolesArr) {
        if (auto rs = dyn_cast<StringAttr>(r))
          pSpec.roles.push_back(rs.getValue().str());
      }
    }
    if (!protoOp.getBody().empty()) {
      for (ProtocolStateOp stOp :
           protoOp.getBody().front().getOps<ProtocolStateOp>()) {
        verification::ProtocolStateSpec stSpec;
        stSpec.name = stOp.getSymName().str();
        stSpec.isFinal =
            getDictStr(stOp.getSemanticProperties(), "final") == "true";
        stSpec.location = mlirLocToSourceLoc(stOp.getLoc());
        if (!stOp.getBody().empty()) {
          for (MessageDeclOp msgOp :
               stOp.getBody().front().getOps<MessageDeclOp>()) {
            verification::ProtocolMessageSpec mSpec;
            mSpec.name = msgOp.getSymName().str();
            mSpec.senderRole =
                getDictStr(msgOp.getSemanticProperties(), "sender");
            mSpec.receiverRole =
                getDictStr(msgOp.getSemanticProperties(), "receiver");
            if (!msgOp.getRecords().empty()) {
              if (auto rDict = dyn_cast<DictionaryAttr>(msgOp.getRecords()[0])) {
                mSpec.successorState = getDictStr(
                    rDict.getAs<DictionaryAttr>("properties"), "successor");
              }
            }
            mSpec.location = mlirLocToSourceLoc(msgOp.getLoc());
            stSpec.messages.push_back(std::move(mSpec));
          }
        }
        pSpec.states.emplace(stSpec.name, std::move(stSpec));
      }
    }
    protocols.emplace(pSpec.name, std::move(pSpec));
  }

  bool allValid = true;
  for (PipelineOp pipeOp : module.getOps<PipelineOp>()) {
    if (pipeOp.getBody().empty()) continue;
    Block &pipeBody = pipeOp.getBody().front();
    const std::string pipeName = pipeOp.getSymName().str();

    SmallVector<EdgeOp> edges(pipeBody.getOps<EdgeOp>());
    SmallVector<TopologyOp> topos(pipeBody.getOps<TopologyOp>());
    SmallVector<ScheduleOp> scheds(pipeBody.getOps<ScheduleOp>());
    SmallVector<StageOp> stages(pipeBody.getOps<StageOp>());

    bool isGPUPipeline = false;
    for (EdgeOp e : edges) {
      const std::string storage = getDictStr(e.getSemanticProperties(), "storage");
      const std::string scope = getDictStr(e.getSemanticProperties(), "scope");
      if (StringRef(storage).starts_with("gpu::") || scope == "block" ||
          scope == "warp" || scope == "device" || scope == "grid" ||
          scope == "cluster") {
        isGPUPipeline = true;
      }
    }
    for (ScheduleOp s : scheds) {
      const std::string tgt = getDictStr(s.getSemanticProperties(), "target");
      if (StringRef(tgt).starts_with("sm") || StringRef(tgt).starts_with("gfx"))
        isGPUPipeline = true;
    }
    for (StageOp st : stages) {
      if (st.getExecutionDomain() == "Gpu") isGPUPipeline = true;
    }
    if (!isGPUPipeline) continue;

    verification::GPUPipelineSystemSpec sys;
    sys.name = pipeName;
    sys.protocols = protocols;
    sys.scheduleCount = scheds.size();
    sys.location = mlirLocToSourceLoc(pipeOp.getLoc());

    std::map<std::string, std::string> portRoleMap;
    for (StageOp st : stages) {
      const std::string qStage = pipeName + "::" + st.getSymName().str();
      for (Attribute pAttr : st.getParameterAttrs()) {
        auto pDict = dyn_cast<DictionaryAttr>(pAttr);
        if (!pDict) continue;
        const std::string pName = getDictStr(pDict, "name");
        if (auto srcType = pDict.getAs<DictionaryAttr>("source_type")) {
          const std::string tName = getDictStr(srcType, "name");
          const auto sep = tName.rfind("::");
          if (sep != std::string::npos)
            portRoleMap[qStage + "." + pName] = tName.substr(sep + 2);
        }
      }
    }

    std::map<std::string, std::map<std::string, std::string>> stagePortMaps;
    std::map<std::string, std::string> edgeProdStage, edgeConsStage;
    std::map<std::string, std::string> edgeProdRole, edgeConsRole;

    auto splitEp = [](const std::string &ep)
        -> std::pair<std::string, std::string> {
      const auto dot = ep.rfind('.');
      if (dot == std::string::npos) return {ep, ""};
      return {ep.substr(0, dot), ep.substr(dot + 1)};
    };

    for (TopologyOp topo : topos) {
      for (Attribute rAttr : topo.getRecords()) {
        auto rDict = dyn_cast<DictionaryAttr>(rAttr);
        if (!rDict || getDictStr(rDict, "kind") != "connection") continue;
        auto props = rDict.getAs<DictionaryAttr>("properties");
        const std::string edgeName = getDictStr(props, "edge");
        if (edgeName.empty()) continue;
        const std::string srcFull = getDictStr(props, "source");
        const std::string dstFull = getDictStr(props, "target");
        const auto [srcSt, srcPort] = splitEp(srcFull);
        const auto [dstSt, dstPort] = splitEp(dstFull);

        if (!srcPort.empty()) stagePortMaps[srcSt][srcPort] = edgeName;
        if (!dstPort.empty()) stagePortMaps[dstSt][dstPort] = edgeName;
        const auto sep = edgeName.rfind("::");
        if (sep != std::string::npos) {
          const std::string shortEdge = edgeName.substr(sep + 2);
          stagePortMaps[srcSt][shortEdge] = edgeName;
          stagePortMaps[dstSt][shortEdge] = edgeName;
        }
        edgeProdStage[edgeName] = srcSt;
        if (portRoleMap.contains(srcFull))
          edgeProdRole[edgeName] = portRoleMap[srcFull];
        if (edgeConsStage.contains(edgeName) &&
            edgeConsStage[edgeName] != dstSt) {
          sys.hasUnconnectedDuplicateConsumer = true;
          sys.duplicateConsumerEdge = edgeName;
          sys.duplicateConsumerLoc = mlirLocToSourceLoc(topo.getLoc());
        } else {
          edgeConsStage[edgeName] = dstSt;
          if (portRoleMap.contains(dstFull))
            edgeConsRole[edgeName] = portRoleMap[dstFull];
        }
      }
    }

    for (EdgeOp edgeOp : edges) {
      verification::GPUBoundedQueueSpec q;
      q.name = pipeName + "::" + edgeOp.getSymName().str();
      q.location = mlirLocToSourceLoc(edgeOp.getLoc());
      DictionaryAttr props = edgeOp.getSemanticProperties();
      q.initialProtocolState = getDictStr(props, "initial");
      q.capacity = parseAttrUnsigned(getDictStr(props, "capacity")).value_or(0);
      const std::string seededStr = getDictStr(props, "seeded_items");
      if (!seededStr.empty()) {
        q.seededItems =
            parseAttrUnsigned(seededStr).value_or(q.capacity + 1);
      }
      const std::string stg = getDictStr(props, "storage");
      if (!stg.empty()) q.storage = stg;
      const std::string scp = getDictStr(props, "scope");
      if (!scp.empty()) q.scope = scp;
      const std::string bp = getDictStr(props, "backpressure");
      if (!bp.empty()) q.backpressure = bp;

      for (Attribute rAttr : edgeOp.getRecords()) {
        auto rDict = dyn_cast<DictionaryAttr>(rAttr);
        if (rDict && getDictStr(rDict, "kind") == "channel_protocol")
          q.protocolName = getDictStr(rDict, "name");
      }

      const bool wgLocal = (q.scope == "block" || q.scope == "warp");
      q.residency = wgLocal
                        ? verification::GPUResidencyContract::WorkgroupLocal
                        : (options.profile.allowCooperativeGridChannels
                               ? verification::GPUResidencyContract::CooperativeGrid
                               : verification::GPUResidencyContract::UnscheduledGrid);
      const std::string pubStr = getDictStr(props, "publication");
      if (!pubStr.empty()) {
        q.explicitPublication = true;
        q.publication = parseMLIRPublication(pubStr).value_or(
            verification::GPUPublicationOrder::Relaxed);
      } else {
        q.publication =
            wgLocal ? verification::GPUPublicationOrder::AcqRelWorkgroup
                    : verification::GPUPublicationOrder::AcqRelDevice;
      }

      if (edgeProdStage.contains(q.name)) q.producerStage = edgeProdStage[q.name];
      if (edgeProdRole.contains(q.name)) q.producerRole = edgeProdRole[q.name];
      if (edgeConsStage.contains(q.name)) q.consumerStage = edgeConsStage[q.name];
      if (edgeConsRole.contains(q.name)) q.consumerRole = edgeConsRole[q.name];
      sys.queues.push_back(std::move(q));
    }

    auto symRefToStageName = [&](Attribute refAttr) -> std::string {
      SymbolRefAttr sym = dyn_cast<SymbolRefAttr>(refAttr);
      if (!sym) {
        if (auto dict = dyn_cast<DictionaryAttr>(refAttr))
          sym = dict.getAs<SymbolRefAttr>("symbol");
      }
      if (!sym) return {};
      if (!sym.getNestedReferences().empty()) {
        return sym.getRootReference().str() + "::" +
               sym.getNestedReferences().back().getValue().str();
      }
      return sym.getRootReference().str();
    };

    for (ScheduleOp sched : scheds) {
      std::map<std::string, std::pair<std::size_t, std::string>> workerInfo;
      for (Attribute rAttr : sched.getRecords()) {
        auto rDict = dyn_cast<DictionaryAttr>(rAttr);
        if (!rDict || getDictStr(rDict, "kind") != "fused_kernel") continue;
        const std::string wName = getDictStr(rDict, "name");
        auto props = rDict.getAs<DictionaryAttr>("properties");
        const std::string blkStr = getDictStr(props, "blocks");
        const std::size_t blks =
            blkStr.empty() ? 1 : parseAttrUnsigned(blkStr).value_or(0);
        const std::string resMode = getDictStr(props, "residency").empty()
                                        ? "persistent"
                                        : getDictStr(props, "residency");
        workerInfo[wName] = {blks, resMode};
        if (auto refsArr = rDict.getAs<ArrayAttr>("references")) {
          for (Attribute sRef : refsArr) {
            const std::string sName = symRefToStageName(sRef);
            if (sName.empty()) continue;
            verification::GPUStagePlacementInfo pInfo;
            pInfo.stageName = sName;
            pInfo.workerName = wName;
            pInfo.workerBlocks = blks;
            pInfo.workerResidency = resMode;
            pInfo.location = mlirLocToSourceLoc(sched.getLoc());
            sys.placements[sName] = std::move(pInfo);
          }
        }
      }
      for (Attribute rAttr : sched.getRecords()) {
        auto rDict = dyn_cast<DictionaryAttr>(rAttr);
        if (!rDict || getDictStr(rDict, "kind") != "placement") continue;
        auto refsArr = rDict.getAs<ArrayAttr>("references");
        if (!refsArr || refsArr.empty()) continue;
        const std::string sName = symRefToStageName(refsArr[0]);
        if (sName.empty()) continue;
        auto &pInfo = sys.placements[sName];
        pInfo.stageName = sName;
        auto props = rDict.getAs<DictionaryAttr>("properties");
        const std::string wName = getDictStr(props, "worker");
        if (!wName.empty()) {
          pInfo.workerName = wName;
          if (workerInfo.contains(wName)) {
            pInfo.workerBlocks = workerInfo[wName].first;
            pInfo.workerResidency = workerInfo[wName].second;
          }
        }
        pInfo.hasWarpPlacement = true;
        pInfo.firstWarp =
            parseAttrUnsigned(getDictStr(props, "index")).value_or(0);
        const std::string endStr = getDictStr(props, "end");
        pInfo.lastWarp = endStr.empty()
                             ? pInfo.firstWarp
                             : parseAttrUnsigned(endStr).value_or(pInfo.firstWarp);
        pInfo.location = mlirLocToSourceLoc(sched.getLoc());
      }
    }

    for (StageOp st : stages) {
      const std::string qStage = pipeName + "::" + st.getSymName().str();
      const auto mIt = stagePortMaps.find(qStage);
      const std::map<std::string, std::string> emptyMap;
      sys.stages.push_back(extractMLIRStageAutomaton(
          st, qStage, mIt != stagePortMaps.end() ? mIt->second : emptyMap));
    }

    verification::GPUBoundedQueueReport rep;
    verification::verifyGPUPipelineSystem(sys, options, rep);
    for (const auto &v : rep.violations) {
      pipeOp.emitError("[" + v.code + "] " + v.message);
      allValid = false;
    }
    if (rep.status == verification::VerificationStatus::Unknown &&
        rep.violations.empty()) {
      pipeOp.emitError("[gpu.solver.unknown] bounded-queue verification indeterminate");
      allValid = false;
    }
  }

  return success(allValid);
}

} // namespace metreon
