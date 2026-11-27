#include "metreon/MLIR/DeviceExecution.h"
#include "CanonicalSyntax.h"
#include "ScalarLowering.h"
#include "metreon/MLIR/Dialect.h"
#include "metreon/MLIR/GPUVerification.h"
#include "metreon/Verification/GPUOwnershipAndLoans.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <type_traits>

using namespace mlir;
namespace metreon {
namespace {
using Kind = graphir::CallableOperationKind;
using Syntax = parser::CanonicalSyntax;

DictionaryAttr application(OpBuilder &builder, const ast::TypeReference &type) {
  SmallVector<Attribute> arguments;
  for (const auto &argument : type.arguments)
    arguments.push_back(application(builder, argument));
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("name", builder.getStringAttr(type.name.str())),
       builder.getNamedAttr("arguments", builder.getArrayAttr(arguments))});
}

void graphContextChecks(const graphir::Module &source,
                        ArrayRef<const graphir::CallableGraph *> graphs) {
  if (!source.resourceGraphs().empty() || !source.stageGraphs().empty() ||
      !source.protocolGraphs().empty() || !source.pipelineGraphs().empty())
    throw std::runtime_error(
        "device execution profile does not implement resource/protocol/pipeline execution");
  std::map<std::string, const graphir::CallableGraph *> procedures;
  for (const auto &graph : source.procedureGraphs())
    procedures.emplace(graph.name(), &graph);
  auto owned = [&](const graphir::ContextMetadataRef &metadata) {
    source.requireContextUse(metadata, graphir::ContextUse::MetadataAttachment);
    if (!metadata->hasIdentifier())
      throw std::runtime_error("device context must have a declared identity");
  };
  for (const auto *graph : graphs) {
    const auto ambient = graph->context();
    std::optional<verification::GPUExecutionScope> ambientScope;
    if (ambient) {
      owned(ambient);
      const auto type = Syntax::type(ambient->name()).name.str();
      ambientScope =
          verification::classifyContextExecutionScope(ambient->name());
      if (!type.starts_with("Gpu::") || type == "Gpu::Loop" || !ambientScope)
        throw std::runtime_error(
            "device execution requires a Gpu execution context");
    }
    const auto domain = graph->attributes().at("execution_domain");
    if (domain != (ambient ? "Gpu" : "context_independent") &&
        !(domain == "Gpu" && !ambient))
      throw std::runtime_error(
          "device execution cannot lower host or hardware procedures");
    for (const auto &parameter : graph->parameters())
      if (parameter.context) {
        owned(parameter.context);
        if (!ambient || !ambient->hasSameIdentity(*parameter.context))
          throw std::runtime_error(
              "parameter changes nominal execution context");
      }
    graphir::walkCallableOperationsWithContext(
        graph->body(), {}, [&](const auto &op, const auto &effective) {
          if (op.kind == Kind::AsyncIssue || op.kind == Kind::AsyncWait) {
            if (!effective)
              throw std::runtime_error(
                  "device async operation has no declared nominal context");
            owned(effective);
            if (!ambient || !ambient->hasSameIdentity(*effective))
              throw std::runtime_error(
                  "operation changes nominal execution context");
            return;
          }
          if (graphir::isBehavioralOrAsyncOperation(op.kind))
            throw std::runtime_error(
                "device execution profile does not lower standalone channel operations outside a verified pipeline");
          if (op.kind == Kind::ContextRegion && !op.context)
            throw std::runtime_error(
                "device context region has no declared context");
          if (effective) {
            owned(effective);
            if (!ambient || !ambient->hasSameIdentity(*effective))
              throw std::runtime_error(
                  "operation changes nominal execution context");
          }
          const bool loop = op.kind == Kind::For || op.kind == Kind::Loop ||
                            op.kind == Kind::While || op.kind == Kind::Until;
          if (loop || op.loopContext) {
            if (!op.loopContext)
              throw std::runtime_error("device loop has no Gpu::Loop context");
            owned(op.loopContext);
            if (op.loopContext->name() != "Gpu::Loop" ||
                op.loopContext->genericArity())
              throw std::runtime_error(
                  "device loops require Gpu::Loop evidence");
            for (const auto &node : source.nodes())
              if (node.kind == graphir::NodeKind::Grant && node.context &&
                  node.context->hasSameIdentity(*op.loopContext))
                throw std::runtime_error(
                    "loop contexts cannot grant execution authority");
          }
          if (op.kind == Kind::Call) {
            const auto kind = op.attributes.at("callee_kind");
            if (kind == "gpu_builtin") {
              if (!effective ||
                  (op.name != "gpu::global_id" && op.name != "gpu::thread_id" &&
                   op.name != "gpu::block_id" && op.name != "gpu::block_dim" &&
                   op.name != "gpu::load" && op.name != "gpu::store" &&
                   op.name != "gpu::barrier" && op.name != "gpu::warp_barrier" &&
                   op.name != "gpu::grid_barrier"))
                throw std::runtime_error(
                    "unknown or context-free GPU intrinsic");
              if (ambientScope &&
                  !verification::isIntrinsicAllowedInScope(*ambientScope,
                                                           op.name))
                throw std::runtime_error("GPU intrinsic `" + op.name +
                                         "` is forbidden in `" +
                                         ambient->name() + "` scope");
            } else if (kind == "procedure") {
              auto found = procedures.find(op.name);
              if (found == procedures.end())
                throw std::runtime_error(
                    "device call has no procedure declaration");
              const auto target = found->second->context();
              if (target && (!ambient || !target->hasSameIdentity(*ambient)))
                throw std::runtime_error(
                    "call requires a different nominal execution context");
              if (target && !effective)
                throw std::runtime_error(
                    "device call is missing nominal execution context");
            } else
              throw std::runtime_error(
                  "device execution does not implement this call kind");
          }
        });
  }
}

ContextOp resolveContext(Operation *op, SymbolRefAttr reference) {
  if (!reference) return {};
  for (Operation *scope = op; scope != nullptr; scope = scope->getParentOp()) {
    if (scope->hasTrait<OpTrait::SymbolTable>()) {
      if (auto found =
              dyn_cast_or_null<ContextOp>(SymbolTable::lookupSymbolIn(scope, reference)))
        return found;
    }
  }
  return {};
}

bool grants(ContextOp context, StringRef name) {
  for (auto value : context.getGrants())
    if (cast<DictionaryAttr>(value).getAs<StringAttr>("name").getValue() == name)
      return true;
  return false;
}

bool deviceSpan(Type type) {
  auto span = dyn_cast<MemRefType>(type);
  auto space = span ? dyn_cast_or_null<IntegerAttr>(span.getMemorySpace())
                    : IntegerAttr{};
  return span && span.getRank() == 1 && span.isDynamicDim(0) &&
         span.getElementType().isF32() && span.getLayout().isIdentity() &&
         space && space.getInt() == 1;
}

bool scalar(Type type) { return isa<IntegerType, FloatType, IndexType>(type); }

std::string dictStr(DictionaryAttr dict, StringRef key) {
  if (!dict) return {};
  if (auto s = dict.getAs<StringAttr>(key)) return s.getValue().str();
  return {};
}

std::optional<std::size_t> parseSmallUnsigned(StringRef text) {
  if (text.empty()) return std::nullopt;
  unsigned long long v = 0;
  if (text.getAsInteger(10, v)) return std::nullopt;
  return static_cast<std::size_t>(v);
}

std::string sanitizeSymbolToken(StringRef raw) {
  std::string out;
  out.reserve(raw.size());
  for (char c : raw) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  return out;
}

func::FuncOp ensureAdapterDecl(gpu::GPUModuleOp device, Location loc,
                               StringRef symName, TypeRange argTypes,
                               TypeRange resTypes) {
  if (auto existing =
          dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(device, symName))) {
    return existing;
  }
  OpBuilder b(device.getContext());
  b.setInsertionPointToStart(device.getBody());
  auto fn = b.create<func::FuncOp>(loc, symName,
                                   b.getFunctionType(argTypes, resTypes));
  fn.setPrivate();
  return fn;
}

Value createSubgroupConvergentSync(OpBuilder &b, Location loc) {
  Type i32Ty = b.getI32Type();
  Value zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
  Value width = b.create<arith::ConstantIntOp>(loc, 32, 32);
  auto shfl = b.create<gpu::ShuffleOp>(loc, i32Ty, b.getI1Type(), zero, zero,
                                       width, gpu::ShuffleMode::IDX);
  return shfl.getShuffleResult();
}

struct IntrinsicConversion : OpConversionPattern<DeviceIntrinsicOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(DeviceIntrinsicOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const auto args = adaptor.getInputs();
    const Location loc = op.getLoc();
    if (op.getKind() == "global_id") {
      auto block = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
      auto width = rewriter.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
      auto thread = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
      auto offset = rewriter.create<arith::MulIOp>(loc, block, width);
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, offset, thread);
    } else if (op.getKind() == "thread_id") {
      rewriter.replaceOpWithNewOp<gpu::ThreadIdOp>(op, gpu::Dimension::x);
    } else if (op.getKind() == "block_id") {
      rewriter.replaceOpWithNewOp<gpu::BlockIdOp>(op, gpu::Dimension::x);
    } else if (op.getKind() == "block_dim") {
      rewriter.replaceOpWithNewOp<gpu::BlockDimOp>(op, gpu::Dimension::x);
    } else if (op.getKind() == "load") {
      rewriter.replaceOpWithNewOp<memref::LoadOp>(op, args[0],
                                                  ValueRange{args[1]});
    } else if (op.getKind() == "store") {
      rewriter.create<memref::StoreOp>(loc, args[2], args[0],
                                       ValueRange{args[1]});
      rewriter.eraseOp(op);
    } else if (op.getKind() == "barrier") {
      rewriter.replaceOpWithNewOp<gpu::BarrierOp>(op);
    } else if (op.getKind() == "warp_barrier") {
      (void)createSubgroupConvergentSync(rewriter, loc);
      rewriter.eraseOp(op);
    } else {
      return rewriter.notifyMatchFailure(
          op, "device intrinsic requires target specialization");
    }
    return success();
  }
};

template <typename Contract>
struct CallableConversion : OpConversionPattern<Contract> {
  gpu::GPUModuleOp destination;
  CallableConversion(MLIRContext *context, gpu::GPUModuleOp destination)
      : OpConversionPattern<Contract>(context), destination(destination) {}
  LogicalResult matchAndRewrite(Contract op, typename Contract::Adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto targetModule = destination;
    auto body = cast<func::FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(op, op.getBodyRefAttr()));
    rewriter.setInsertionPoint(targetModule.getBody()->getTerminator());
    if constexpr (std::is_same_v<Contract, KernelOp>) {
      auto kernel = rewriter.template create<gpu::GPUFuncOp>(
          op.getLoc(), op.getSymName(), op.getFunctionType());
      kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                      rewriter.getUnitAttr());
      rewriter.inlineRegionBefore(body.getBody(), kernel.getBody(),
                                  kernel.getBody().end());
      if (kernel.getBody().front().empty())
        rewriter.eraseBlock(&kernel.getBody().front());
      SmallVector<func::ReturnOp> returns;
      kernel.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
      for (auto ret : returns) {
        rewriter.setInsertionPoint(ret);
        rewriter.template replaceOpWithNewOp<gpu::ReturnOp>(ret,
                                                            ret.getOperands());
      }
      rewriter.eraseOp(body);
    } else {
      rewriter.modifyOpInPlace(
          body, [&] { body->moveBefore(targetModule.getBody()->getTerminator()); });
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ContextConversion : OpConversionPattern<ContextOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ContextOp op, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// Materializes verified `AsyncCopyOp` and `AsyncWaitOp` operations inside live
// `func::FuncOp` bodies before `CallableConversion` moves them into `device`.
LogicalResult materializeAsyncCopyAndWaitOps(ModuleOp module,
                                             gpu::GPUModuleOp device) {
  MLIRContext *ctx = module.getContext();
  Type idxTy = IndexType::get(ctx);

  SmallVector<AsyncCopyOp> copies;
  SmallVector<AsyncWaitOp> waits;
  module.walk([&](AsyncCopyOp op) { copies.push_back(op); });
  module.walk([&](AsyncWaitOp op) { waits.push_back(op); });

  unsigned copyDeclId = 0;
  for (AsyncCopyOp copyOp : copies) {
    Location loc = copyOp.getLoc();
    Value src = copyOp.getSource();
    Value dst = copyOp.getDestination();
    auto srcType = dyn_cast<MemRefType>(src.getType());
    auto dstType = dyn_cast<MemRefType>(dst.getType());
    if (!srcType || !dstType || srcType.getRank() != 1 ||
        dstType.getRank() != 1 || !srcType.getElementType().isF32() ||
        !dstType.getElementType().isF32()) {
      return copyOp.emitOpError(
          "GPU async_copy lowering currently requires rank-1 f32 memrefs");
    }

    auto srcSpaceAttr = dyn_cast_or_null<IntegerAttr>(srcType.getMemorySpace());
    auto dstSpaceAttr = dyn_cast_or_null<IntegerAttr>(dstType.getMemorySpace());
    const int64_t srcSpace = srcSpaceAttr ? srcSpaceAttr.getInt() : 0;
    const int64_t dstSpace = dstSpaceAttr ? dstSpaceAttr.getInt() : 0;
    if (srcSpace != 1 || (dstSpace != 3 && dstSpace != 1)) {
      return copyOp.emitOpError(
          "GPU async_copy requires global source (space 1) and shared/global destination (space 3 or 1)");
    }

    ContextOp ctxOp = resolveContext(copyOp, copyOp.getContextAttr());
    bool isWarpScope = false;
    if (ctxOp) {
      const std::string nom =
          dictStr(ctxOp.getNominalType(), "name");
      isWarpScope =
          (verification::classifyContextExecutionScope(nom) ==
           verification::GPUExecutionScope::Subgroup);
    }

    OpBuilder b(copyOp);
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value count;
    if (srcType.hasStaticShape()) {
      count = b.create<arith::ConstantIndexOp>(loc, srcType.getDimSize(0));
    } else {
      count = b.create<memref::DimOp>(loc, src, c0);
    }
    if (dstType.hasStaticShape()) {
      Value dstCap =
          b.create<arith::ConstantIndexOp>(loc, dstType.getDimSize(0));
      count = b.create<arith::MinUIOp>(loc, count, dstCap);
    }

    Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value start = tid;
    Value step;
    if (isWarpScope) {
      Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
      start = b.create<arith::RemUIOp>(loc, tid, c32);
      step = c32;
    } else {
      step = b.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    }

    const std::string elemSym =
        (dstSpace == 3 ? "__gpu_async_copy_elem_g2s_f32_"
                       : "__gpu_async_copy_elem_g2g_f32_") +
        std::to_string(copyDeclId++);
    ensureAdapterDecl(device, loc, elemSym, {dstType, srcType, idxTy}, {});
    ensureAdapterDecl(device, loc, "__gpu_async_commit_group", {}, {});

    auto whileOp = b.create<scf::WhileOp>(
        loc, TypeRange{idxTy}, ValueRange{start},
        [&](OpBuilder &beforeB, Location bLoc, ValueRange args) {
          Value cmp = beforeB.create<arith::CmpIOp>(
              bLoc, arith::CmpIPredicate::ult, args[0], count);
          beforeB.create<scf::ConditionOp>(bLoc, cmp, args);
        },
        [&](OpBuilder &afterB, Location aLoc, ValueRange args) {
          afterB.create<func::CallOp>(aLoc, elemSym, TypeRange{},
                                      ValueRange{dst, src, args[0]});
          Value next = afterB.create<arith::AddIOp>(aLoc, args[0], step);
          afterB.create<scf::YieldOp>(aLoc, ValueRange{next});
        });
    (void)whileOp;
    b.create<func::CallOp>(loc, "__gpu_async_commit_group", TypeRange{},
                           ValueRange{});
    copyOp.erase();
  }

  for (AsyncWaitOp waitOp : waits) {
    Location loc = waitOp.getLoc();
    ContextOp ctxOp = resolveContext(waitOp, waitOp.getContextAttr());
    bool isWarpScope = false;
    if (ctxOp) {
      const std::string nom = dictStr(ctxOp.getNominalType(), "name");
      isWarpScope =
          (verification::classifyContextExecutionScope(nom) ==
           verification::GPUExecutionScope::Subgroup);
    }
    ensureAdapterDecl(device, loc, "__gpu_async_wait_group", {}, {});
    OpBuilder b(waitOp);
    b.create<func::CallOp>(loc, "__gpu_async_wait_group", TypeRange{},
                           ValueRange{});
    if (isWarpScope) {
      (void)createSubgroupConvergentSync(b, loc);
    } else {
      b.create<gpu::BarrierOp>(loc);
    }
    waitOp.erase();
  }

  return success();
}

Type pipelineScalarOrSpanType(OpBuilder &b, StringRef typeText,
                              bool isOutputParam) {
  auto vt = Syntax::valueType(typeText.str());
  const std::string name = vt.reference.name.str();
  if (typeText == "gpu::span<f32>") {
    return MemRefType::get({ShapedType::kDynamic}, b.getF32Type(),
                           MemRefLayoutAttrInterface{},
                           b.getI64IntegerAttr(1));
  }
  if (isOutputParam && name == "f32") {
    return MemRefType::get({ShapedType::kDynamic}, b.getF32Type(),
                           MemRefLayoutAttrInterface{},
                           b.getI64IntegerAttr(1));
  }
  if (name == "void") return b.getNoneType();
  if (name == "bool") return b.getI1Type();
  if (name == "index") return b.getIndexType();
  if (name == "f16") return b.getF16Type();
  if (name == "f32") return b.getF32Type();
  if (name == "f64") return b.getF64Type();
  if (name == "i8" || name == "u8") return b.getIntegerType(8);
  if (name == "i16" || name == "u16") return b.getIntegerType(16);
  if (name == "i32" || name == "u32") return b.getIntegerType(32);
  if (name == "i64" || name == "u64") return b.getIntegerType(64);
  return {};
}

Value emitDeferredConstant(OpBuilder &b, Location loc, StringRef srcTypeName,
                           StringRef litKind, StringRef litVal) {
  Type ty = pipelineScalarOrSpanType(b, srcTypeName, /*isOutputParam=*/false);
  if (!ty || isa<NoneType>(ty)) return {};
  if (ty.isInteger(1)) {
    return b.create<arith::ConstantIntOp>(loc, litVal == "true" ? 1 : 0, 1);
  }
  if (isa<IntegerType, IndexType>(ty)) {
    const bool neg = !litVal.empty() && litVal.front() == '-';
    StringRef mag = litVal.drop_front(neg ? 1 : 0);
    llvm::APInt val;
    if (mag.empty() || mag.getAsInteger(10, val)) return {};
    const unsigned bits =
        isa<IndexType>(ty) ? 64u : cast<IntegerType>(ty).getWidth();
    val = val.zextOrTrunc(bits);
    if (neg) val = -val;
    return b.create<arith::ConstantOp>(loc, ty, IntegerAttr::get(ty, val));
  }
  if (auto flt = dyn_cast<FloatType>(ty)) {
    llvm::APFloat fv(flt.getFloatSemantics());
    if (litKind == "infinity") {
      fv = llvm::APFloat::getInf(flt.getFloatSemantics(),
                                 !litVal.empty() && litVal.front() == '-');
    } else {
      auto st =
          fv.convertFromString(litVal, llvm::APFloat::rmNearestTiesToEven);
      if (!st) {
        llvm::consumeError(st.takeError());
        return {};
      }
    }
    return b.create<arith::ConstantOp>(loc, ty, FloatAttr::get(ty, fv));
  }
  return {};
}

struct LoweredQueueRing {
  std::string edgeName;
  std::size_t capacity = 1;
  std::size_t seededItems = 0;
  int32_t seededTagId = 1;
  bool workgroupShared = true;
  std::string ctrlGlobalSym;
  std::string dataGlobalSym;
  std::string tagGlobalSym;
  unsigned globalArgOffset = 0;
  MemRefType ctrlType;
  MemRefType dataType;
  MemRefType tagType;
  std::string loadAcqSym;
  std::string loadRlxSym;
  std::string storeRelSym;
  std::string readF32Sym;
  std::string writeF32Sym;
  std::string readI32Sym;
  std::string writeI32Sym;
};

// Lowers all verified GPU `PipelineOp`s (and erases accompanying `ProtocolOp`s)
// into warp-specialized `gpu::GPUFuncOp` kernels and shared/global bounded
// ring-buffer adapters inside `device`.
LogicalResult lowerVerifiedPipelinesToGPU(ModuleOp module,
                                          gpu::GPUModuleOp device) {
  SmallVector<PipelineOp> pipelines(module.getOps<PipelineOp>());
  if (pipelines.empty()) return success();

  MLIRContext *ctx = module.getContext();
  OpBuilder b(ctx);
  Type idxTy = b.getIndexType();
  Type i32Ty = b.getI32Type();
  Type f32Ty = b.getF32Type();

  // Assign deterministic 1-based message tag IDs across all protocols.
  std::map<std::string, int32_t> messageTagIds{{"token", 1}};
  int32_t nextMsgId = 2;
  for (ProtocolOp protoOp : module.getOps<ProtocolOp>()) {
    if (protoOp.getBody().empty()) continue;
    for (ProtocolStateOp stOp :
         protoOp.getBody().front().getOps<ProtocolStateOp>()) {
      if (stOp.getBody().empty()) continue;
      for (MessageDeclOp msgOp :
           stOp.getBody().front().getOps<MessageDeclOp>()) {
        const std::string mName = msgOp.getSymName().str();
        if (!messageTagIds.contains(mName))
          messageTagIds[mName] = nextMsgId++;
      }
    }
  }

  unsigned ringAdapterCounter = 0;

  for (PipelineOp pipeOp : pipelines) {
    if (pipeOp.getBody().empty()) continue;
    Block &pipeBody = pipeOp.getBody().front();
    const std::string pipeName = pipeOp.getSymName().str();
    const Location pipeLoc = pipeOp.getLoc();

    SmallVector<PipelineStateOp> states(pipeBody.getOps<PipelineStateOp>());
    SmallVector<EdgeOp> edges(pipeBody.getOps<EdgeOp>());
    SmallVector<TopologyOp> topos(pipeBody.getOps<TopologyOp>());
    SmallVector<ScheduleOp> scheds(pipeBody.getOps<ScheduleOp>());
    SmallVector<StageOp> stages(pipeBody.getOps<StageOp>());

    if (scheds.empty()) {
      return pipeOp.emitOpError(
          "executable GPU pipeline lowering requires a schedule block");
    }

    std::map<std::string, StageOp> stageByQualName;
    for (StageOp st : stages) {
      stageByQualName[pipeName + "::" + st.getSymName().str()] = st;
    }

    std::map<std::string, PipelineStateOp> stateByQualName;
    for (PipelineStateOp ps : states) {
      stateByQualName[pipeName + "::" + ps.getSymName().str()] = ps;
    }

    // Parse pipeline parameters (`input` / `output`).
    struct PipeParamSpec {
      std::string name;
      std::string qualName;
      std::string srcType;
      bool isOutput = false;
      Type mlirType;
    };
    std::vector<PipeParamSpec> pipeParams;
    std::map<std::string, std::size_t> pipeParamIndex;
    for (Attribute rAttr : pipeOp.getRecords()) {
      auto rDict = dyn_cast<DictionaryAttr>(rAttr);
      if (!rDict || dictStr(rDict, "kind") != "pipeline_parameter") continue;
      auto props = rDict.getAs<DictionaryAttr>("properties");
      PipeParamSpec spec;
      spec.name = dictStr(rDict, "name");
      spec.qualName = pipeName + "::" + spec.name;
      if (auto tDict = props ? props.getAs<DictionaryAttr>("type")
                             : DictionaryAttr{}) {
        spec.srcType = dictStr(tDict, "name");
        if (auto argsArr = tDict.getAs<ArrayAttr>("arguments");
            argsArr && !argsArr.empty()) {
          if (auto a0 = dyn_cast<DictionaryAttr>(argsArr[0])) {
            spec.srcType += "<" + dictStr(a0, "name") + ">";
          }
        }
      } else {
        spec.srcType = dictStr(props, "type");
      }
      spec.isOutput = (dictStr(props, "direction") == "output");
      spec.mlirType = pipelineScalarOrSpanType(b, spec.srcType, spec.isOutput);
      if (!spec.mlirType) {
        return pipeOp.emitOpError("unsupported GPU pipeline parameter type: ")
               << spec.srcType;
      }
      pipeParamIndex[spec.qualName] = pipeParams.size();
      pipeParams.push_back(std::move(spec));
    }

    // Parse topology bindings.
    std::map<std::string, std::map<std::string, std::string>> stagePortToQueue;
    std::map<std::string, std::map<std::string, std::size_t>> stagePortToPipeParam;
    std::map<std::string, std::map<std::string, PipelineStateOp>> stagePortToState;

    auto splitEp = [](const std::string &ep)
        -> std::pair<std::string, std::string> {
      const auto dot = ep.rfind('.');
      if (dot == std::string::npos) return {ep, ""};
      return {ep.substr(0, dot), ep.substr(dot + 1)};
    };

    for (TopologyOp topo : topos) {
      for (Attribute rAttr : topo.getRecords()) {
        auto rDict = dyn_cast<DictionaryAttr>(rAttr);
        if (!rDict || dictStr(rDict, "kind") != "connection") continue;
        auto props = rDict.getAs<DictionaryAttr>("properties");
        const std::string srcFull = dictStr(props, "source");
        const std::string dstFull = dictStr(props, "target");
        const std::string edgeName = dictStr(props, "edge");
        const auto [srcNode, srcPort] = splitEp(srcFull);
        const auto [dstNode, dstPort] = splitEp(dstFull);

        if (!edgeName.empty()) {
          if (!srcPort.empty()) stagePortToQueue[srcNode][srcPort] = edgeName;
          if (!dstPort.empty()) stagePortToQueue[dstNode][dstPort] = edgeName;
          const auto sep = edgeName.rfind("::");
          if (sep != std::string::npos) {
            const std::string shortEdge = edgeName.substr(sep + 2);
            stagePortToQueue[srcNode][shortEdge] = edgeName;
            stagePortToQueue[dstNode][shortEdge] = edgeName;
          }
        } else {
          if (!dstPort.empty() && pipeParamIndex.contains(srcFull)) {
            stagePortToPipeParam[dstNode][dstPort] = pipeParamIndex[srcFull];
          } else if (!dstPort.empty() && stateByQualName.contains(srcFull)) {
            stagePortToState[dstNode][dstPort] = stateByQualName[srcFull];
          } else if (!srcPort.empty() && pipeParamIndex.contains(dstFull)) {
            stagePortToPipeParam[srcNode][srcPort] = pipeParamIndex[dstFull];
          }
        }
      }
    }

    // Build queue ring-buffer descriptors and `memref.global`s in `device`.
    std::map<std::string, LoweredQueueRing> queueRings;
    for (EdgeOp edgeOp : edges) {
      LoweredQueueRing qr;
      qr.edgeName = pipeName + "::" + edgeOp.getSymName().str();
      DictionaryAttr props = edgeOp.getSemanticProperties();
      qr.capacity =
          parseSmallUnsigned(dictStr(props, "capacity")).value_or(1);
      qr.seededItems =
          parseSmallUnsigned(dictStr(props, "seeded_items")).value_or(0);
      const std::string storage = dictStr(props, "storage");
      qr.workgroupShared = (storage.empty() || storage == "gpu::shared");

      // Resolve seeded tag ID from protocol initial state if present.
      std::string protoName;
      for (Attribute rAttr : edgeOp.getRecords()) {
        if (auto rDict = dyn_cast<DictionaryAttr>(rAttr);
            rDict && dictStr(rDict, "kind") == "channel_protocol") {
          protoName = dictStr(rDict, "name");
        }
      }
      const std::string initSt = dictStr(props, "initial");
      for (ProtocolOp protoOp : module.getOps<ProtocolOp>()) {
        if (protoOp.getSymName() != protoName || protoOp.getBody().empty())
          continue;
        for (ProtocolStateOp stOp :
             protoOp.getBody().front().getOps<ProtocolStateOp>()) {
          if (stOp.getSymName() != initSt || stOp.getBody().empty()) continue;
          auto msgs = stOp.getBody().front().getOps<MessageDeclOp>();
          if (!msgs.empty()) {
            const std::string firstMsg = (*msgs.begin()).getSymName().str();
            if (messageTagIds.contains(firstMsg))
              qr.seededTagId = messageTagIds[firstMsg];
          }
        }
      }

      const unsigned id = ringAdapterCounter++;
      const std::string scopeTag = qr.workgroupShared ? "wg" : "dev";
      const std::string suf =
          scopeTag + "_" + std::to_string(id);

      if (qr.workgroupShared) {
        Attribute space3 = b.getI64IntegerAttr(3);
        qr.ctrlType =
            MemRefType::get({2}, i32Ty, MemRefLayoutAttrInterface{}, space3);
        qr.dataType = MemRefType::get({static_cast<int64_t>(qr.capacity)}, f32Ty,
                                      MemRefLayoutAttrInterface{}, space3);
        qr.tagType = MemRefType::get({static_cast<int64_t>(qr.capacity)}, i32Ty,
                                     MemRefLayoutAttrInterface{}, space3);

        const std::string baseSym =
            "__ring_" + sanitizeSymbolToken(qr.edgeName) + "_" +
            std::to_string(id);
        qr.ctrlGlobalSym = baseSym + "_ctrl";
        qr.dataGlobalSym = baseSym + "_data";
        qr.tagGlobalSym = baseSym + "_tag";

        b.setInsertionPointToStart(device.getBody());
        b.create<memref::GlobalOp>(edgeOp.getLoc(), qr.ctrlGlobalSym,
                                   b.getStringAttr("private"), qr.ctrlType,
                                   b.getUnitAttr(), /*constant=*/false,
                                   IntegerAttr{});
        b.create<memref::GlobalOp>(edgeOp.getLoc(), qr.dataGlobalSym,
                                   b.getStringAttr("private"), qr.dataType,
                                   b.getUnitAttr(), /*constant=*/false,
                                   IntegerAttr{});
        b.create<memref::GlobalOp>(edgeOp.getLoc(), qr.tagGlobalSym,
                                   b.getStringAttr("private"), qr.tagType,
                                   b.getUnitAttr(), /*constant=*/false,
                                   IntegerAttr{});
      } else {
        Attribute space1 = b.getI64IntegerAttr(1);
        qr.ctrlType = MemRefType::get({ShapedType::kDynamic}, i32Ty,
                                      MemRefLayoutAttrInterface{}, space1);
        qr.dataType = MemRefType::get({ShapedType::kDynamic}, f32Ty,
                                      MemRefLayoutAttrInterface{}, space1);
        qr.tagType = MemRefType::get({ShapedType::kDynamic}, i32Ty,
                                     MemRefLayoutAttrInterface{}, space1);
      }

      qr.loadAcqSym = "__gpu_ring_load_acquire_" + suf;
      qr.loadRlxSym = "__gpu_ring_load_relaxed_" + suf;
      qr.storeRelSym = "__gpu_ring_store_release_" + suf;
      qr.readF32Sym = "__gpu_ring_read_f32_" + suf;
      qr.writeF32Sym = "__gpu_ring_write_f32_" + suf;
      qr.readI32Sym = "__gpu_ring_read_i32_" + suf;
      qr.writeI32Sym = "__gpu_ring_write_i32_" + suf;

      ensureAdapterDecl(device, edgeOp.getLoc(), qr.loadAcqSym,
                        {qr.ctrlType, idxTy}, {i32Ty});
      ensureAdapterDecl(device, edgeOp.getLoc(), qr.loadRlxSym,
                        {qr.ctrlType, idxTy}, {i32Ty});
      ensureAdapterDecl(device, edgeOp.getLoc(), qr.storeRelSym,
                        {qr.ctrlType, idxTy, i32Ty}, {});
      ensureAdapterDecl(device, edgeOp.getLoc(), qr.readF32Sym,
                        {qr.dataType, idxTy}, {f32Ty});
      ensureAdapterDecl(device, edgeOp.getLoc(), qr.writeF32Sym,
                        {qr.dataType, idxTy, f32Ty}, {});
      ensureAdapterDecl(device, edgeOp.getLoc(), qr.readI32Sym,
                        {qr.tagType, idxTy}, {i32Ty});
      ensureAdapterDecl(device, edgeOp.getLoc(), qr.writeI32Sym,
                        {qr.tagType, idxTy, i32Ty}, {});

      queueRings.emplace(qr.edgeName, std::move(qr));
    }

    auto symRefToQualStage = [](Attribute refAttr) -> std::string {
      SymbolRefAttr sym = dyn_cast<SymbolRefAttr>(refAttr);
      if (!sym) {
        if (auto d = dyn_cast<DictionaryAttr>(refAttr))
          sym = d.getAs<SymbolRefAttr>("symbol");
      }
      if (!sym) return {};
      if (!sym.getNestedReferences().empty()) {
        return sym.getRootReference().str() + "::" +
               sym.getNestedReferences().back().getValue().str();
      }
      return sym.getRootReference().str();
    };

    // Count total fused workers across schedules to name single-worker
    // pipelines cleanly (`@PipelineName`).
    std::size_t totalWorkers = 0;
    for (ScheduleOp sched : scheds) {
      for (Attribute rAttr : sched.getRecords()) {
        auto rDict = dyn_cast<DictionaryAttr>(rAttr);
        if (rDict && dictStr(rDict, "kind") == "fused_kernel") ++totalWorkers;
      }
    }

    for (ScheduleOp sched : scheds) {
      std::map<std::string, std::pair<std::size_t, std::size_t>> stageWarps;
      for (Attribute rAttr : sched.getRecords()) {
        auto rDict = dyn_cast<DictionaryAttr>(rAttr);
        if (!rDict || dictStr(rDict, "kind") != "placement") continue;
        auto refs = rDict.getAs<ArrayAttr>("references");
        if (!refs || refs.empty()) continue;
        const std::string sName = symRefToQualStage(refs[0]);
        auto props = rDict.getAs<DictionaryAttr>("properties");
        const std::size_t firstW =
            parseSmallUnsigned(dictStr(props, "index")).value_or(0);
        const std::string endStr = dictStr(props, "end");
        const std::size_t lastW =
            endStr.empty() ? firstW
                           : parseSmallUnsigned(endStr).value_or(firstW);
        stageWarps[sName] = {firstW, lastW};
      }

      for (Attribute rAttr : sched.getRecords()) {
        auto rDict = dyn_cast<DictionaryAttr>(rAttr);
        if (!rDict || dictStr(rDict, "kind") != "fused_kernel") continue;
        const std::string wName = dictStr(rDict, "name");
        const std::string kernelName =
            totalWorkers == 1 ? pipeName : (pipeName + "__" + wName);

        SmallVector<Type> kernelArgTypes;
        for (const auto &pp : pipeParams) kernelArgTypes.push_back(pp.mlirType);
        for (auto &[qName, qr] : queueRings) {
          (void)qName;
          if (!qr.workgroupShared) {
            qr.globalArgOffset = kernelArgTypes.size();
            kernelArgTypes.push_back(qr.ctrlType);
            kernelArgTypes.push_back(qr.dataType);
            kernelArgTypes.push_back(qr.tagType);
          }
        }

        b.setInsertionPoint(device.getBody()->getTerminator());
        auto kernel = b.create<gpu::GPUFuncOp>(
            pipeLoc, kernelName, b.getFunctionType(kernelArgTypes, {}));
        kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                        b.getUnitAttr());
        Block &kEntry = kernel.getBody().front();
        b.setInsertionPointToStart(&kEntry);

        Value tid = b.create<gpu::ThreadIdOp>(pipeLoc, gpu::Dimension::x);
        Value c0 = b.create<arith::ConstantIndexOp>(pipeLoc, 0);
        Value c1 = b.create<arith::ConstantIndexOp>(pipeLoc, 1);
        Value c32 = b.create<arith::ConstantIndexOp>(pipeLoc, 32);
        Value warpId = b.create<arith::DivUIOp>(pipeLoc, tid, c32);

        bool hasSharedQueues = false;
        for (const auto &[qName, qr] : queueRings) {
          (void)qName;
          if (qr.workgroupShared) hasSharedQueues = true;
        }

        if (hasSharedQueues) {
          Value isTid0 = b.create<arith::CmpIOp>(
              pipeLoc, arith::CmpIPredicate::eq, tid, c0);
          auto initIf = b.create<scf::IfOp>(pipeLoc, isTid0, /*withElse=*/false);
          OpBuilder ib(ctx);
          ib.setInsertionPointToStart(&initIf.getThenRegion().front());
          Value zeroI32 = ib.create<arith::ConstantIntOp>(pipeLoc, 0, 32);
          Value zeroF32 = ib.create<arith::ConstantOp>(
              pipeLoc, f32Ty, ib.getF32FloatAttr(0.0f));
          for (const auto &[qName, qr] : queueRings) {
            (void)qName;
            if (!qr.workgroupShared) continue;
            Value ctrl = ib.create<memref::GetGlobalOp>(pipeLoc, qr.ctrlType,
                                                        qr.ctrlGlobalSym);
            Value data = ib.create<memref::GetGlobalOp>(pipeLoc, qr.dataType,
                                                        qr.dataGlobalSym);
            Value tag = ib.create<memref::GetGlobalOp>(pipeLoc, qr.tagType,
                                                       qr.tagGlobalSym);
            ib.create<func::CallOp>(pipeLoc, qr.storeRelSym, TypeRange{},
                                    ValueRange{ctrl, c0, zeroI32});
            Value seedTagVal =
                ib.create<arith::ConstantIntOp>(pipeLoc, qr.seededTagId, 32);
            for (std::size_t s = 0; s < qr.seededItems; ++s) {
              Value sIdx = ib.create<arith::ConstantIndexOp>(pipeLoc, s);
              ib.create<func::CallOp>(pipeLoc, qr.writeF32Sym, TypeRange{},
                                      ValueRange{data, sIdx, zeroF32});
              ib.create<func::CallOp>(pipeLoc, qr.writeI32Sym, TypeRange{},
                                      ValueRange{tag, sIdx, seedTagVal});
            }
            Value seededI32 = ib.create<arith::ConstantIntOp>(
                pipeLoc, static_cast<int64_t>(qr.seededItems), 32);
            ib.create<func::CallOp>(pipeLoc, qr.storeRelSym, TypeRange{},
                                    ValueRange{ctrl, c1, seededI32});
          }
          b.setInsertionPointAfter(initIf);
          b.create<gpu::BarrierOp>(pipeLoc);
        }

        // Emit and invoke each fused stage in its assigned warp range.
        auto refsArr = rDict.getAs<ArrayAttr>("references");
        std::size_t fallbackWarp = 0;
        if (refsArr) {
          for (Attribute sRef : refsArr) {
            const std::string qStageName = symRefToQualStage(sRef);
            auto stIt = stageByQualName.find(qStageName);
            if (stIt == stageByQualName.end()) continue;
            StageOp stageOp = stIt->second;
            const Location stLoc = stageOp.getLoc();

            const auto [firstW, lastW] =
                stageWarps.contains(qStageName)
                    ? stageWarps[qStageName]
                    : std::make_pair(fallbackWarp, fallbackWarp);
            ++fallbackWarp;

            // Synthesize private stage helper `func::FuncOp` inside `device` so
            // arbitrary `deferred_cfg` loops/branches lift cleanly via
            // `structureGPUFunctionControlFlow`.
            SmallVector<Type> stageArgTypes;
            SmallVector<Value> stageCallArgs;
            std::map<std::int64_t, unsigned> graphirParamToArgIdx;

            for (Attribute pAttr : stageOp.getParameterAttrs()) {
              auto pDict = dyn_cast<DictionaryAttr>(pAttr);
              if (!pDict) continue;
              const std::string pName = dictStr(pDict, "name");
              auto gidAttr = pDict.getAs<IntegerAttr>("graphir_id");
              if (stagePortToQueue[qStageName].contains(pName)) {
                // Channel endpoint parameter: resolved statically to ring-buffer!
                continue;
              }
              if (stagePortToPipeParam[qStageName].contains(pName)) {
                const std::size_t ppIdx = stagePortToPipeParam[qStageName][pName];
                Value argVal = kEntry.getArgument(ppIdx);
                // If pipeline `output c: f32` was promoted to `memref<?xf32, 1>`
                // while stage parameter `dst: f32` is scalar, pass `0.0 : f32`!
                auto declaredParamTy =
                    pDict.getAs<TypeAttr>("type").getValue();
                if (declaredParamTy.isF32() && isa<MemRefType>(argVal.getType())) {
                  argVal = b.create<arith::ConstantOp>(
                      stLoc, f32Ty, b.getF32FloatAttr(0.0f));
                }
                if (gidAttr)
                  graphirParamToArgIdx[gidAttr.getInt()] = stageArgTypes.size();
                stageArgTypes.push_back(argVal.getType());
                stageCallArgs.push_back(argVal);
              } else if (stagePortToState[qStageName].contains(pName)) {
                PipelineStateOp psOp = stagePortToState[qStageName][pName];
                DictionaryAttr psProps = psOp.getSemanticProperties();
                std::string psTypeName;
                if (auto tD = psProps.getAs<DictionaryAttr>("type"))
                  psTypeName = dictStr(tD, "name");
                else
                  psTypeName = dictStr(psProps, "type");
                const std::string initTxt = dictStr(psProps, "initial");
                Value initVal = emitDeferredConstant(
                    b, stLoc, psTypeName,
                    (initTxt == "true" || initTxt == "false") ? "boolean"
                                                              : "integer",
                    initTxt);
                if (!initVal)
                  initVal = b.create<arith::ConstantIntOp>(stLoc, 0, 64);
                if (gidAttr)
                  graphirParamToArgIdx[gidAttr.getInt()] = stageArgTypes.size();
                stageArgTypes.push_back(initVal.getType());
                stageCallArgs.push_back(initVal);
              } else if (auto tAttr = pDict.getAs<TypeAttr>("type")) {
                Type pt = tAttr.getValue();
                if (scalar(pt)) {
                  Value z = pt.isF32()
                                ? b.create<arith::ConstantOp>(
                                      stLoc, f32Ty, b.getF32FloatAttr(0.0f))
                                      .getResult()
                                : (pt.isIndex()
                                       ? b.create<arith::ConstantIndexOp>(stLoc, 0)
                                             .getResult()
                                       : b.create<arith::ConstantIntOp>(
                                              stLoc, 0,
                                              cast<IntegerType>(pt).getWidth())
                                             .getResult());
                  if (gidAttr)
                    graphirParamToArgIdx[gidAttr.getInt()] =
                        stageArgTypes.size();
                  stageArgTypes.push_back(pt);
                  stageCallArgs.push_back(z);
                }
              }
            }

            // Also pass any global-memory queue buffers (`!workgroupShared`) into
            // the stage helper function.
            std::map<std::string, unsigned> globalQueueStageArgOffset;
            for (const auto &[qName, qr] : queueRings) {
              if (!qr.workgroupShared) {
                globalQueueStageArgOffset[qName] = stageArgTypes.size();
                stageArgTypes.push_back(qr.ctrlType);
                stageArgTypes.push_back(qr.dataType);
                stageArgTypes.push_back(qr.tagType);
                stageCallArgs.push_back(kEntry.getArgument(qr.globalArgOffset));
                stageCallArgs.push_back(
                    kEntry.getArgument(qr.globalArgOffset + 1));
                stageCallArgs.push_back(
                    kEntry.getArgument(qr.globalArgOffset + 2));
              }
            }

            const std::string stageFnSym =
                "__stage_" + sanitizeSymbolToken(kernelName) + "_" +
                sanitizeSymbolToken(stageOp.getSymName());
            OpBuilder sb(ctx);
            sb.setInsertionPoint(kernel);
            auto stageFn = sb.create<func::FuncOp>(
                stLoc, stageFnSym, sb.getFunctionType(stageArgTypes, {}));
            stageFn.setPrivate();

            // Populate `stageFn` from `stageOp.getDeferredBody()` +
            // `stageOp.getDeferredCfg()`.
            std::map<std::int64_t, DictionaryAttr> opById;
            std::function<void(ArrayAttr)> collectOps = [&](ArrayAttr arr) {
              if (!arr) return;
              for (Attribute elem : arr) {
                auto d = dyn_cast<DictionaryAttr>(elem);
                if (!d) continue;
                if (auto idA = d.getAs<IntegerAttr>("id"))
                  opById[idA.getInt()] = d;
                for (StringRef k : {"initialization", "condition", "body",
                                    "alternative", "step", "offer_branches"}) {
                  if (auto sub = d.getAs<ArrayAttr>(k)) collectOps(sub);
                }
              }
            };
            collectOps(stageOp.getDeferredBody());

            ArrayAttr cfgBlocks = stageOp.getDeferredCfg();
            Block *fnEntry = stageFn.addEntryBlock();
            if (!cfgBlocks || cfgBlocks.empty()) {
              sb.setInsertionPointToEnd(fnEntry);
              sb.create<func::ReturnOp>(stLoc);
            } else {
              std::map<std::int64_t, Block *> blockMap;
              std::map<std::int64_t, DictionaryAttr> blockDictMap;
              bool first = true;
              for (Attribute bAttr : cfgBlocks) {
                auto bDict = dyn_cast<DictionaryAttr>(bAttr);
                if (!bDict) continue;
                const std::int64_t bid = bDict.getAs<IntegerAttr>("id").getInt();
                blockDictMap[bid] = bDict;
                if (first) {
                  blockMap[bid] = fnEntry;
                  first = false;
                } else {
                  auto *nb = new Block();
                  stageFn.getBody().push_back(nb);
                  blockMap[bid] = nb;
                }
              }

              std::map<std::int64_t, Value> ssaValues, slotAllocas;
              std::map<std::int64_t, std::string> opTypeNames;
              for (const auto &[gid, argIdx] : graphirParamToArgIdx) {
                ssaValues[gid] = fnEntry->getArgument(argIdx);
              }

              auto typeNameFromProps = [](DictionaryAttr props) -> std::string {
                if (!props) return "void";
                if (auto tDict = props.getAs<DictionaryAttr>("type")) {
                  std::string n = dictStr(tDict, "name");
                  if (auto args = tDict.getAs<ArrayAttr>("arguments");
                      args && !args.empty()) {
                    if (auto a0 = dyn_cast<DictionaryAttr>(args[0]))
                      n += "<" + dictStr(a0, "name") + ">";
                  }
                  return n;
                }
                return dictStr(props, "type");
              };

              sb.setInsertionPointToStart(fnEntry);
              for (const auto &[oid, oDict] : opById) {
                DictionaryAttr props = oDict.getAs<DictionaryAttr>("properties");
                const std::string tName = typeNameFromProps(props);
                opTypeNames[oid] = tName;
                const std::string dKind = dictStr(oDict, "kind");
                if ((dKind == "variable" || dKind == "variable_decl") &&
                    dictStr(props, "mutable_local") == "true") {
                  Type elemTy = pipelineScalarOrSpanType(sb, tName, false);
                  if (elemTy && !isa<NoneType>(elemTy)) {
                    auto alloca = sb.create<memref::AllocaOp>(
                        stLoc, MemRefType::get({}, elemTy));
                    if (dictStr(oDict, "name") == "$range_index")
                      alloca->setAttr("metreon.synthetic_iv", sb.getUnitAttr());
                    slotAllocas[oid] = alloca;
                  }
                }
              }

              auto resolveQueueHandles =
                  [&](OpBuilder &qb, const LoweredQueueRing &qr)
                  -> std::tuple<Value, Value, Value> {
                if (qr.workgroupShared) {
                  Value ctrl = qb.create<memref::GetGlobalOp>(
                      stLoc, qr.ctrlType, qr.ctrlGlobalSym);
                  Value data = qb.create<memref::GetGlobalOp>(
                      stLoc, qr.dataType, qr.dataGlobalSym);
                  Value tag = qb.create<memref::GetGlobalOp>(
                      stLoc, qr.tagType, qr.tagGlobalSym);
                  return {ctrl, data, tag};
                }
                const unsigned off = globalQueueStageArgOffset.at(qr.edgeName);
                return {fnEntry->getArgument(off), fnEntry->getArgument(off + 1),
                        fnEntry->getArgument(off + 2)};
              };

              // Emit blocks in reverse post-order so SSA definitions dominate uses.
              std::set<std::int64_t> visitedB;
              std::vector<std::int64_t> rpo;
              std::function<void(std::int64_t)> dfsB = [&](std::int64_t bid) {
                if (!visitedB.insert(bid).second || !blockDictMap.contains(bid))
                  return;
                DictionaryAttr bd = blockDictMap[bid];
                if (auto s = bd.getAs<IntegerAttr>("successor"))
                  dfsB(s.getInt());
                if (auto fs = bd.getAs<IntegerAttr>("false_successor"))
                  dfsB(fs.getInt());
                rpo.push_back(bid);
              };
              dfsB(blockMap.begin()->first);
              std::reverse(rpo.begin(), rpo.end());

              for (std::int64_t bid : rpo) {
                DictionaryAttr bd = blockDictMap[bid];
                sb.setInsertionPointToEnd(blockMap[bid]);

                if (auto opsArr = bd.getAs<ArrayAttr>("operations")) {
                  for (Attribute opIdAttr : opsArr) {
                    const std::int64_t oid =
                        cast<IntegerAttr>(opIdAttr).getInt();
                    if (!opById.contains(oid)) continue;
                    DictionaryAttr oDict = opById[oid];
                    const std::string kind = dictStr(oDict, "kind");
                    DictionaryAttr props =
                        oDict.getAs<DictionaryAttr>("properties");
                    ArrayAttr opnds = oDict.getAs<ArrayAttr>("operands");
                    auto getOpnd = [&](unsigned i) -> Value {
                      const std::int64_t refId =
                          cast<IntegerAttr>(opnds[i]).getInt();
                      return ssaValues.at(refId);
                    };

                    if (kind == "literal") {
                      Value v = emitDeferredConstant(
                          sb, stLoc, opTypeNames[oid], dictStr(props, "kind"),
                          dictStr(props, "value"));
                      if (v) ssaValues[oid] = v;
                    } else if (kind == "variable" || kind == "variable_decl") {
                      Value init;
                      if (opnds && !opnds.empty()) {
                        init = getOpnd(0);
                      } else if (dictStr(props, "initialized") == "true") {
                        init = emitDeferredConstant(
                            sb, stLoc, opTypeNames[oid],
                            dictStr(props, "initializer.kind"),
                            dictStr(props, "initializer.value"));
                      }
                      if (slotAllocas.contains(oid)) {
                        if (init)
                          sb.create<memref::StoreOp>(stLoc, init,
                                                     slotAllocas[oid],
                                                     ValueRange{});
                      } else if (init) {
                        ssaValues[oid] = init;
                      }
                    } else if (kind == "load") {
                      const std::int64_t slotId =
                          cast<IntegerAttr>(opnds[0]).getInt();
                      ssaValues[oid] = sb.create<memref::LoadOp>(
                          stLoc, slotAllocas.at(slotId), ValueRange{});
                    } else if (kind == "assign") {
                      const std::int64_t slotId =
                          cast<IntegerAttr>(opnds[0]).getInt();
                      sb.create<memref::StoreOp>(stLoc, getOpnd(1),
                                                 slotAllocas.at(slotId),
                                                 ValueRange{});
                    } else if (kind == "binary") {
                      Value lhs = getOpnd(0), rhs = getOpnd(1);
                      const std::string opStr = dictStr(props, "operator");
                      const std::int64_t lhsId =
                          cast<IntegerAttr>(opnds[0]).getInt();
                      const bool isUns =
                          opTypeNames.contains(lhsId) &&
                          StringRef(opTypeNames[lhsId]).starts_with("u");
                      if (isa<FloatType>(lhs.getType())) {
                        if (opStr == "+")
                          ssaValues[oid] =
                              sb.create<arith::AddFOp>(stLoc, lhs, rhs);
                        else if (opStr == "-")
                          ssaValues[oid] =
                              sb.create<arith::SubFOp>(stLoc, lhs, rhs);
                        else if (opStr == "*")
                          ssaValues[oid] =
                              sb.create<arith::MulFOp>(stLoc, lhs, rhs);
                        else if (opStr == "/")
                          ssaValues[oid] =
                              sb.create<arith::DivFOp>(stLoc, lhs, rhs);
                        else {
                          auto pred = opStr == "=="   ? arith::CmpFPredicate::OEQ
                                      : opStr == "!=" ? arith::CmpFPredicate::UNE
                                      : opStr == "<"  ? arith::CmpFPredicate::OLT
                                      : opStr == "<=" ? arith::CmpFPredicate::OLE
                                      : opStr == ">"  ? arith::CmpFPredicate::OGT
                                                      : arith::CmpFPredicate::OGE;
                          ssaValues[oid] =
                              sb.create<arith::CmpFOp>(stLoc, pred, lhs, rhs);
                        }
                      } else {
                        if (opStr == "+")
                          ssaValues[oid] =
                              sb.create<arith::AddIOp>(stLoc, lhs, rhs);
                        else if (opStr == "-")
                          ssaValues[oid] =
                              sb.create<arith::SubIOp>(stLoc, lhs, rhs);
                        else if (opStr == "*")
                          ssaValues[oid] =
                              sb.create<arith::MulIOp>(stLoc, lhs, rhs);
                        else if (opStr == "/")
                          ssaValues[oid] =
                              isUns ? sb.create<arith::DivUIOp>(stLoc, lhs, rhs)
                                          .getResult()
                                    : sb.create<arith::DivSIOp>(stLoc, lhs, rhs)
                                          .getResult();
                        else if (opStr == "%")
                          ssaValues[oid] =
                              isUns ? sb.create<arith::RemUIOp>(stLoc, lhs, rhs)
                                          .getResult()
                                    : sb.create<arith::RemSIOp>(stLoc, lhs, rhs)
                                          .getResult();
                        else {
                          auto pred =
                              opStr == "==" ? arith::CmpIPredicate::eq
                              : opStr == "!=" ? arith::CmpIPredicate::ne
                              : opStr == "<"
                                  ? (isUns ? arith::CmpIPredicate::ult
                                           : arith::CmpIPredicate::slt)
                              : opStr == "<="
                                  ? (isUns ? arith::CmpIPredicate::ule
                                           : arith::CmpIPredicate::sle)
                              : opStr == ">"
                                  ? (isUns ? arith::CmpIPredicate::ugt
                                           : arith::CmpIPredicate::sgt)
                                  : (isUns ? arith::CmpIPredicate::uge
                                           : arith::CmpIPredicate::sge);
                          ssaValues[oid] =
                              sb.create<arith::CmpIOp>(stLoc, pred, lhs, rhs);
                        }
                      }
                    } else if (kind == "unary") {
                      Value v = getOpnd(0);
                      const std::string opStr = dictStr(props, "operator");
                      if (opStr == "+") {
                        ssaValues[oid] = v;
                      } else if (opStr == "!") {
                        Value t = sb.create<arith::ConstantIntOp>(stLoc, 1, 1);
                        ssaValues[oid] = sb.create<arith::XOrIOp>(stLoc, v, t);
                      } else if (isa<FloatType>(v.getType())) {
                        ssaValues[oid] = sb.create<arith::NegFOp>(stLoc, v);
                      } else {
                        Value z = sb.create<arith::ConstantOp>(
                            stLoc, v.getType(),
                            sb.getZeroAttr(v.getType()));
                        ssaValues[oid] = sb.create<arith::SubIOp>(stLoc, z, v);
                      }
                    } else if (kind == "call" &&
                               dictStr(props, "callee_kind") == "gpu_builtin") {
                      const std::string bName = dictStr(oDict, "name");
                      if (bName == "gpu::thread_id") {
                        ssaValues[oid] = sb.create<gpu::ThreadIdOp>(
                            stLoc, gpu::Dimension::x);
                      } else if (bName == "gpu::block_id") {
                        ssaValues[oid] =
                            sb.create<gpu::BlockIdOp>(stLoc, gpu::Dimension::x);
                      } else if (bName == "gpu::block_dim") {
                        ssaValues[oid] = sb.create<gpu::BlockDimOp>(
                            stLoc, gpu::Dimension::x);
                      } else if (bName == "gpu::global_id") {
                        auto blk =
                            sb.create<gpu::BlockIdOp>(stLoc, gpu::Dimension::x);
                        auto dim = sb.create<gpu::BlockDimOp>(
                            stLoc, gpu::Dimension::x);
                        auto th = sb.create<gpu::ThreadIdOp>(
                            stLoc, gpu::Dimension::x);
                        auto off = sb.create<arith::MulIOp>(stLoc, blk, dim);
                        ssaValues[oid] =
                            sb.create<arith::AddIOp>(stLoc, off, th);
                      } else if (bName == "gpu::load") {
                        ssaValues[oid] = sb.create<memref::LoadOp>(
                            stLoc, getOpnd(0), ValueRange{getOpnd(1)});
                      } else if (bName == "gpu::store") {
                        sb.create<memref::StoreOp>(stLoc, getOpnd(2), getOpnd(0),
                                                   ValueRange{getOpnd(1)});
                      } else if (bName == "gpu::warp_barrier") {
                        (void)createSubgroupConvergentSync(sb, stLoc);
                      } else if (bName == "gpu::barrier") {
                        sb.create<gpu::BarrierOp>(stLoc);
                      }
                    } else if (kind == "channel_send" ||
                               kind == "channel_recv" ||
                               kind == "channel_receive") {
                      const std::string rawEp = dictStr(props, "endpoint");
                      std::string qName = rawEp;
                      if (stagePortToQueue[qStageName].contains(rawEp))
                        qName = stagePortToQueue[qStageName][rawEp];
                      if (!queueRings.contains(qName)) continue;
                      const LoweredQueueRing &qr = queueRings.at(qName);
                      auto [ctrl, data, tag] = resolveQueueHandles(sb, qr);

                      Value fnTid =
                          sb.create<gpu::ThreadIdOp>(stLoc, gpu::Dimension::x);
                      Value fnC0 = sb.create<arith::ConstantIndexOp>(stLoc, 0);
                      Value fnC1 = sb.create<arith::ConstantIndexOp>(stLoc, 1);
                      Value fnC32 =
                          sb.create<arith::ConstantIndexOp>(stLoc, 32);
                      Value lane =
                          sb.create<arith::RemUIOp>(stLoc, fnTid, fnC32);
                      Value isLeader = sb.create<arith::CmpIOp>(
                          stLoc, arith::CmpIPredicate::eq, lane, fnC0);

                      const std::string msg = dictStr(props, "message");
                      const int32_t msgId =
                          messageTagIds.contains(msg) ? messageTagIds[msg] : 1;
                      Value msgIdVal =
                          sb.create<arith::ConstantIntOp>(stLoc, msgId, 32);
                      Value capI32 = sb.create<arith::ConstantIntOp>(
                          stLoc, static_cast<int64_t>(qr.capacity), 32);
                      Value capIdx =
                          sb.create<arith::ConstantIndexOp>(stLoc, qr.capacity);

                      if (kind == "channel_send") {
                        Value payload =
                            (opnds && !opnds.empty() &&
                             ssaValues.contains(
                                 cast<IntegerAttr>(opnds[0]).getInt()) &&
                             ssaValues
                                 .at(cast<IntegerAttr>(opnds[0]).getInt())
                                 .getType()
                                 .isF32())
                                ? getOpnd(0)
                                : sb.create<arith::ConstantOp>(
                                        stLoc, f32Ty, sb.getF32FloatAttr(0.0f))
                                      .getResult();
                        auto sendIf =
                            sb.create<scf::IfOp>(stLoc, isLeader, false);
                        OpBuilder lb(ctx);
                        lb.setInsertionPointToStart(
                            &sendIf.getThenRegion().front());
                        // Spin while `(tail - acquire_load(head)) >= C`.
                        lb.create<scf::WhileOp>(
                            stLoc, TypeRange{}, ValueRange{},
                            [&](OpBuilder &wb, Location wLoc, ValueRange) {
                              Value curTail =
                                  wb.create<func::CallOp>(
                                        wLoc, qr.loadRlxSym, TypeRange{i32Ty},
                                        ValueRange{ctrl, fnC1})
                                      .getResult(0);
                              Value curHead =
                                  wb.create<func::CallOp>(
                                        wLoc, qr.loadAcqSym, TypeRange{i32Ty},
                                        ValueRange{ctrl, fnC0})
                                      .getResult(0);
                              Value occ =
                                  wb.create<arith::SubIOp>(wLoc, curTail, curHead);
                              Value full = wb.create<arith::CmpIOp>(
                                  wLoc, arith::CmpIPredicate::sge, occ, capI32);
                              wb.create<scf::ConditionOp>(wLoc, full,
                                                          ValueRange{});
                            },
                            [&](OpBuilder &ab, Location aLoc, ValueRange) {
                              ab.create<scf::YieldOp>(aLoc);
                            });
                        Value tailVal =
                            lb.create<func::CallOp>(stLoc, qr.loadRlxSym,
                                                    TypeRange{i32Ty},
                                                    ValueRange{ctrl, fnC1})
                                .getResult(0);
                        Value tailIdx =
                            lb.create<arith::IndexCastUIOp>(stLoc, idxTy, tailVal);
                        Value slot =
                            lb.create<arith::RemUIOp>(stLoc, tailIdx, capIdx);
                        lb.create<func::CallOp>(stLoc, qr.writeF32Sym,
                                                TypeRange{},
                                                ValueRange{data, slot, payload});
                        lb.create<func::CallOp>(stLoc, qr.writeI32Sym,
                                                TypeRange{},
                                                ValueRange{tag, slot, msgIdVal});
                        Value oneI32 =
                            lb.create<arith::ConstantIntOp>(stLoc, 1, 32);
                        Value nextTail =
                            lb.create<arith::AddIOp>(stLoc, tailVal, oneI32);
                        lb.create<func::CallOp>(
                            stLoc, qr.storeRelSym, TypeRange{},
                            ValueRange{ctrl, fnC1, nextTail});
                        (void)createSubgroupConvergentSync(sb, stLoc);
                      } else {
                        auto recvIf = sb.create<scf::IfOp>(
                            stLoc, TypeRange{f32Ty}, isLeader, true);
                        {
                          OpBuilder lb(ctx);
                          lb.setInsertionPointToStart(
                              &recvIf.getThenRegion().front());
                          lb.create<scf::WhileOp>(
                              stLoc, TypeRange{}, ValueRange{},
                              [&](OpBuilder &wb, Location wLoc, ValueRange) {
                                Value curHead =
                                    wb.create<func::CallOp>(
                                          wLoc, qr.loadRlxSym, TypeRange{i32Ty},
                                          ValueRange{ctrl, fnC0})
                                        .getResult(0);
                                Value curTail =
                                    wb.create<func::CallOp>(
                                          wLoc, qr.loadAcqSym, TypeRange{i32Ty},
                                          ValueRange{ctrl, fnC1})
                                        .getResult(0);
                                Value empty = wb.create<arith::CmpIOp>(
                                    wLoc, arith::CmpIPredicate::eq, curHead,
                                    curTail);
                                wb.create<scf::ConditionOp>(wLoc, empty,
                                                            ValueRange{});
                              },
                              [&](OpBuilder &ab, Location aLoc, ValueRange) {
                                ab.create<scf::YieldOp>(aLoc);
                              });
                          Value headVal =
                              lb.create<func::CallOp>(stLoc, qr.loadRlxSym,
                                                      TypeRange{i32Ty},
                                                      ValueRange{ctrl, fnC0})
                                  .getResult(0);
                          Value headIdx = lb.create<arith::IndexCastUIOp>(
                              stLoc, idxTy, headVal);
                          Value slot =
                              lb.create<arith::RemUIOp>(stLoc, headIdx, capIdx);
                          Value item =
                              lb.create<func::CallOp>(stLoc, qr.readF32Sym,
                                                      TypeRange{f32Ty},
                                                      ValueRange{data, slot})
                                  .getResult(0);
                          Value oneI32 =
                              lb.create<arith::ConstantIntOp>(stLoc, 1, 32);
                          Value nextHead =
                              lb.create<arith::AddIOp>(stLoc, headVal, oneI32);
                          lb.create<func::CallOp>(
                              stLoc, qr.storeRelSym, TypeRange{},
                              ValueRange{ctrl, fnC0, nextHead});
                          lb.create<scf::YieldOp>(stLoc, ValueRange{item});
                        }
                        {
                          OpBuilder eb(ctx);
                          eb.setInsertionPointToStart(
                              &recvIf.getElseRegion().front());
                          Value zf = eb.create<arith::ConstantOp>(
                              stLoc, f32Ty, eb.getF32FloatAttr(0.0f));
                          eb.create<scf::YieldOp>(stLoc, ValueRange{zf});
                        }
                        Value zeroLane =
                            sb.create<arith::ConstantIntOp>(stLoc, 0, 32);
                        Value width32 =
                            sb.create<arith::ConstantIntOp>(stLoc, 32, 32);
                        auto bcast = sb.create<gpu::ShuffleOp>(
                            stLoc, f32Ty, sb.getI1Type(), recvIf.getResult(0),
                            zeroLane, width32, gpu::ShuffleMode::IDX);
                        if (opTypeNames[oid] == "f32")
                          ssaValues[oid] = bcast.getShuffleResult();
                      }
                    } else if (kind == "progress_marker") {
                      // Provenance-verified in GPU-4; zero runtime instructions.
                    }
                  }
                }

                const std::string term = dictStr(bd, "terminator");
                if (term == "branch") {
                  const std::int64_t dst =
                      bd.getAs<IntegerAttr>("successor").getInt();
                  sb.create<cf::BranchOp>(stLoc, blockMap.at(dst));
                } else if (term == "conditional_branch") {
                  const std::int64_t condId =
                      bd.getAs<IntegerAttr>("condition").getInt();
                  const std::int64_t tDst =
                      bd.getAs<IntegerAttr>("successor").getInt();
                  const std::int64_t fDst =
                      bd.getAs<IntegerAttr>("false_successor").getInt();
                  Value condVal = ssaValues.contains(condId)
                                      ? ssaValues.at(condId)
                                      : sb.create<arith::ConstantIntOp>(
                                              stLoc, 1, 1)
                                            .getResult();

                  // If this conditional branch is an `offer:` dispatch, compute
                  // real atomic queue readiness across the warp!
                  const std::string ctrl = dictStr(bd, "control");
                  if (StringRef(ctrl).starts_with("offer:")) {
                    for (const auto &[oid, od] : opById) {
                      (void)oid;
                      if (dictStr(od, "kind") != "offer") continue;
                      auto arms = od.getAs<ArrayAttr>("offer_branches");
                      if (!arms || arms.empty()) break;
                      auto armDict = dyn_cast<DictionaryAttr>(arms[0]);
                      if (!armDict) break;
                      auto aProps = armDict.getAs<DictionaryAttr>("properties");
                      std::string ep = dictStr(aProps, "endpoint");
                      if (stagePortToQueue[qStageName].contains(ep))
                        ep = stagePortToQueue[qStageName][ep];
                      if (!queueRings.contains(ep)) break;
                      const LoweredQueueRing &qr = queueRings.at(ep);
                      auto [qCtrl, qData, qTag] = resolveQueueHandles(sb, qr);
                      (void)qData;
                      (void)qTag;
                      Value fnC0 = sb.create<arith::ConstantIndexOp>(stLoc, 0);
                      Value fnC1 = sb.create<arith::ConstantIndexOp>(stLoc, 1);
                      Value h = sb.create<func::CallOp>(
                                      stLoc, qr.loadAcqSym, TypeRange{i32Ty},
                                      ValueRange{qCtrl, fnC0})
                                    .getResult(0);
                      Value t = sb.create<func::CallOp>(
                                      stLoc, qr.loadAcqSym, TypeRange{i32Ty},
                                      ValueRange{qCtrl, fnC1})
                                    .getResult(0);
                      if (dictStr(aProps, "direction") == "send") {
                        Value capI32 = sb.create<arith::ConstantIntOp>(
                            stLoc, static_cast<int64_t>(qr.capacity), 32);
                        Value occ = sb.create<arith::SubIOp>(stLoc, t, h);
                        condVal = sb.create<arith::CmpIOp>(
                            stLoc, arith::CmpIPredicate::slt, occ, capI32);
                      } else {
                        condVal = sb.create<arith::CmpIOp>(
                            stLoc, arith::CmpIPredicate::ne, h, t);
                      }
                      break;
                    }
                  }

                  sb.create<cf::CondBranchOp>(stLoc, condVal, blockMap.at(tDst),
                                              ValueRange{}, blockMap.at(fDst),
                                              ValueRange{});
                } else {
                  sb.create<func::ReturnOp>(stLoc);
                }
              }
            }

            // Dispatch `stageFn` on warps `[firstW .. lastW]` inside `kernel`.
            b.setInsertionPointToEnd(&kEntry);
            Value fWVal = b.create<arith::ConstantIndexOp>(stLoc, firstW);
            Value lWVal = b.create<arith::ConstantIndexOp>(stLoc, lastW);
            Value ge = b.create<arith::CmpIOp>(
                stLoc, arith::CmpIPredicate::uge, warpId, fWVal);
            Value le = b.create<arith::CmpIOp>(
                stLoc, arith::CmpIPredicate::ule, warpId, lWVal);
            Value inWarp = b.create<arith::AndIOp>(stLoc, ge, le);
            auto warpIf = b.create<scf::IfOp>(stLoc, inWarp, false);
            OpBuilder wb(ctx);
            wb.setInsertionPointToStart(&warpIf.getThenRegion().front());
            wb.create<func::CallOp>(stLoc, stageFnSym, TypeRange{},
                                    stageCallArgs);
          }
        }

        b.setInsertionPointToEnd(&kEntry);
        b.create<gpu::ReturnOp>(pipeLoc);
      }
    }
  }

  for (PipelineOp p : pipelines) p.erase();
  SmallVector<ProtocolOp> protocols(module.getOps<ProtocolOp>());
  for (ProtocolOp pr : protocols) pr.erase();
  return success();
}

} // namespace

FailureOr<OwningOpRef<ModuleOp>>
importDeviceExecution(const graphir::Module &source, MLIRContext &context) {
  try {
    SmallVector<const graphir::CallableGraph *> graphs;
    for (const auto &graph : source.kernelGraphs()) graphs.push_back(&graph);
    std::set<std::string> selected;
    if (graphs.empty())
      for (const auto &graph : source.procedureGraphs())
        if (graph.attributes().at("execution_domain") != "Host") {
          graphs.push_back(&graph);
          selected.insert(graph.name());
        }
    for (size_t index = 0; index < graphs.size(); ++index)
      graphir::walkCallableOperations(graphs[index]->body(), [&](const auto &op) {
        if (op.kind != Kind::Call || op.attributes.at("callee_kind") != "procedure")
          return;
        for (const auto &procedure : source.procedureGraphs())
          if (procedure.name() == op.name && selected.insert(op.name).second)
            graphs.push_back(&procedure);
      });
    graphContextChecks(source, graphs);
    auto bodies = detail::lowerScalarCFG(source, graphs, context, 64, true);
    if (failed(bodies)) return failure();
    auto semantic = OwningOpRef<ModuleOp>(ModuleOp::create((*bodies)->getLoc()));
    (*semantic)->setAttrs((**bodies)->getAttrs());
    OpBuilder builder(&context);
    std::string name = "__metreon_bodies";
    for (unsigned suffix = 0;
         llvm::any_of(graphs, [&](const auto *graph) { return graph->name() == name; });
         ++suffix)
      name = "__metreon_bodies_" + std::to_string(suffix);
    (**bodies)->setAttr(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(name));
    (**bodies)->setAttr("metreon.ordinary_lowering",
                        builder.getStringAttr("primary_with_context_intrinsics"));
    semantic->getBody()->push_back(bodies->release().getOperation());
    for (const auto *graph : graphs) {
      auto function = cast<func::FuncOp>(SymbolTable::lookupSymbolIn(
          *semantic,
          SymbolRefAttr::get(&context, name,
                             {FlatSymbolRefAttr::get(&context, graph->name())})));
      SmallVector<Attribute> params, results;
      unsigned runtime = 0;
      for (const auto &parameter : graph->parameters())
        if (!parameter.context) {
          const auto sourceType = Syntax::valueType(parameter.type);
          params.push_back(builder.getDictionaryAttr(
              {builder.getNamedAttr("name", builder.getStringAttr(parameter.name)),
               builder.getNamedAttr(
                   "type", TypeAttr::get(function.getArgumentTypes()[runtime++])),
               builder.getNamedAttr("ownership", builder.getStringAttr("none")),
               builder.getNamedAttr("source_type",
                                    application(builder, sourceType.reference))}));
        }
      if (function.getNumResults()) {
        results.push_back(builder.getDictionaryAttr(
            {builder.getNamedAttr("name", builder.getStringAttr("")),
             builder.getNamedAttr(
                 "type", TypeAttr::get(function.getResultTypes()[0])),
             builder.getNamedAttr("ownership", builder.getStringAttr("none")),
             builder.getNamedAttr(
                 "source_type",
                 application(builder,
                             Syntax::valueType(
                                 graph->attributes().at("return_type"))
                                 .reference))}));
      }
      const bool kernel = std::any_of(
          source.kernelGraphs().begin(), source.kernelGraphs().end(),
          [&](const auto &item) { return &item == graph; });
      OperationState state(function.getLoc(),
                           kernel ? KernelOp::getOperationName()
                                  : ProcedureOp::getOperationName());
      state.addAttribute("sym_name", builder.getStringAttr(graph->name()));
      state.addAttribute("function_type",
                         TypeAttr::get(function.getFunctionType()));
      state.addAttribute("parameter_attrs", builder.getArrayAttr(params));
      state.addAttribute("result_attrs", builder.getArrayAttr(results));
      for (const auto *key :
           {"context_requirements", "effects", "deferred_body", "deferred_cfg"})
        state.addAttribute(key, builder.getArrayAttr({}));
      state.addAttribute(
          "execution_domain",
          builder.getStringAttr(graph->attributes().at("execution_domain")));
      state.addAttribute(
          "body_ref",
          SymbolRefAttr::get(&context, name,
                             {FlatSymbolRefAttr::get(&context, graph->name())}));
      if (graph->context())
        state.addAttribute(
            "context",
            SymbolRefAttr::get(
                &context, name,
                {FlatSymbolRefAttr::get(&context,
                                        graph->context()->identifier())}));
      builder.setInsertionPointToEnd(semantic->getBody());
      builder.create(state);
    }
    (*semantic)->setAttr("metreon.verification_profile",
                         builder.getStringAttr("contexts_only"));
    (*semantic)->setAttr("metreon.ordinary_lowering",
                         builder.getStringAttr("live_device_bodies"));
    if (failed(verify(*semantic)) || failed(verifyDeviceContexts(*semantic)))
      return failure();
    return semantic;
  } catch (const std::exception &error) {
    emitError(FileLineColLoc::get(&context, source.sourceName(), 1, 1))
        << "GraphIR device import: " << error.what();
    return failure();
  }
}

FailureOr<OwningOpRef<ModuleOp>>
importGPUPipelineExecution(const graphir::Module &source, MLIRContext &context) {
  if (source.pipelineGraphs().empty()) {
    emitError(FileLineColLoc::get(&context, source.sourceName(), 1, 1))
        << "GPU pipeline execution import requires at least one pipeline declaration";
    return failure();
  }
  auto imported = importGraphIR(source, context);
  if (failed(imported)) return failure();
  if (failed(verifyDeviceContexts(**imported))) return failure();
  return imported;
}

LogicalResult verifyDeviceContexts(ModuleOp module) {
  if (failed(verify(module))) return failure();

  // Verify any GPU pipelines (`GPU-4` bounded-queue, co-residency, and memory
  // scope checks + stage nominal GPU context checks).
  bool hasPipelines = false;
  bool pipelineCtxValid = true;
  for (PipelineOp pipeOp : module.getOps<PipelineOp>()) {
    hasPipelines = true;
    if (pipeOp.getBody().empty()) continue;
    for (StageOp stageOp : pipeOp.getBody().front().getOps<StageOp>()) {
      if (auto ctxRef = stageOp.getContextAttr()) {
        ContextOp stCtx = resolveContext(stageOp, ctxRef);
        if (!stCtx) {
          stageOp.emitOpError("stage references unknown nominal execution context");
          pipelineCtxValid = false;
          continue;
        }
        const std::string nom = dictStr(stCtx.getNominalType(), "name");
        if (!StringRef(nom).starts_with("Gpu::") || nom == "Gpu::Loop" ||
            !verification::classifyContextExecutionScope(nom)) {
          stageOp.emitOpError("GPU pipeline stage requires a Gpu execution context");
          pipelineCtxValid = false;
        }
      }
    }
  }
  if (!pipelineCtxValid) return failure();
  if (hasPipelines && failed(verifyGPUPipelineProtocols(module)))
    return failure();

  std::map<std::string, Operation *> contracts;
  std::set<Operation *> bodies;
  for (auto &op : module.getBody()->getOperations())
    if (isa<KernelOp, ProcedureOp>(op))
      contracts.emplace(cast<StringAttr>(op.getAttr("sym_name")).str(), &op);
  bool valid = true;
  auto reject = [&](Operation *op, StringRef message) {
    op->emitError(message);
    valid = false;
  };
  for (const auto &[name, contract] : contracts) {
    auto reference = contract->getAttrOfType<SymbolRefAttr>("body_ref");
    auto function =
        reference ? dyn_cast_or_null<func::FuncOp>(
                        SymbolTable::lookupNearestSymbolFrom(contract, reference))
                  : func::FuncOp{};
    if (!function) {
      if (!contract->getAttrOfType<ArrayAttr>("deferred_cfg").empty() &&
          hasPipelines)
        continue;
      reject(contract, "device contract requires one unique live function body");
      continue;
    }
    if (!bodies.insert(function).second) {
      reject(contract, "device contract requires one unique live function body");
      continue;
    }
    if (function.isExternal() || function.getSymName() != name)
      reject(contract,
             "device contract requires a defined function body with the same name");
    if (function.getFunctionType() !=
        cast<TypeAttr>(contract->getAttr("function_type")).getValue())
      reject(contract, "device body signature disagrees with contract");
    for (auto type : function.getArgumentTypes()) {
      if (!scalar(type) && !deviceSpan(type) && !isa<ResourceType, OwnedType>(type)) {
        if (auto mem = dyn_cast<MemRefType>(type)) {
          if (auto space = dyn_cast_or_null<IntegerAttr>(mem.getMemorySpace());
              !space || (space.getInt() != 1 && space.getInt() != 3))
            reject(contract,
                   "device parameters must be scalars or GPU global/shared spans");
        } else
          reject(contract,
                 "device parameters must be scalars or dynamic f32 global spans");
      }
    }
    for (auto type : function.getResultTypes())
      if (!scalar(type) && !isa<ResourceType, OwnedType>(type))
        reject(contract, "device helper results must be scalars");
    auto ambient =
        resolveContext(contract, contract->getAttrOfType<SymbolRefAttr>("context"));
    if (contract->hasAttr("context") && !ambient)
      reject(contract, "unknown nominal execution context");
    const auto domain =
        contract->getAttrOfType<StringAttr>("execution_domain").getValue();
    if (domain != (ambient ? "Gpu" : "context_independent") &&
        !(isa<KernelOp>(contract) && domain == "Gpu" && !ambient))
      reject(contract, "device body execution domain disagrees with its context");
    if (ambient) {
      const auto nominal =
          ambient.getNominalType().getAs<StringAttr>("name").getValue();
      if (!nominal.starts_with("Gpu::") || nominal == "Gpu::Loop" ||
          !verification::classifyContextExecutionScope(nominal))
        reject(contract, "device body requires a Gpu execution context");
    }
    if (auto loops = function->getAttrOfType<ArrayAttr>("metreon.loop_contexts"))
      for (auto item : loops) {
        auto loop = resolveContext(function, dyn_cast<SymbolRefAttr>(item));
        if (!loop ||
            loop.getNominalType().getAs<StringAttr>("name").getValue() !=
                "Gpu::Loop" ||
            !loop.getGrants().empty() || !loop.getGenericParameters().empty())
          reject(function,
                 "device loop evidence must reference a grant-free Gpu::Loop context");
      }
    function.walk([&](Operation *op) {
      auto require = [&](ContextOp selected, StringRef grant) {
        if (!ambient || !selected || selected != ambient) {
          reject(op,
                 "device operation requires the same nominal execution context as its body");
          return;
        }
        if (!grant.empty() && !grants(selected, grant))
          reject(op, ("device context lacks required grant: " + grant).str());
      };
      if (auto intrinsic = dyn_cast<DeviceIntrinsicOp>(op)) {
        const auto kind = intrinsic.getKind();
        StringRef req = kind == "load"           ? "gpu_load"
                        : kind == "store"        ? "gpu_store"
                        : kind == "barrier"      ? "gpu_barrier"
                        : kind == "warp_barrier" ? "gpu_warp_barrier"
                        : kind == "grid_barrier" ? "gpu_grid_sync"
                                                 : "";
        require(resolveContext(op, intrinsic.getContextAttr()), req);
      } else if (isa<AsyncCopyOp, AsyncWaitOp, TransitionOp>(op)) {
        // Verified in detail by verifyGPUContextHierarchyAndLoans below.
      } else if (auto call = dyn_cast<func::CallOp>(op)) {
        auto target = contracts.find(call.getCallee().str());
        if (target == contracts.end() || !isa<ProcedureOp>(target->second)) {
          reject(op, "device call has no procedure contract");
          return;
        }
        auto targetContext = resolveContext(
            target->second, target->second->getAttrOfType<SymbolRefAttr>("context"));
        if (targetContext) {
          require(resolveContext(
                      op, op->getAttrOfType<SymbolRefAttr>("metreon.context")),
                  "");
          if (targetContext != ambient)
            reject(op, "callee requires a different nominal execution context");
        }
      } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
        if (load.getMemRefType().getRank() != 0 &&
            !deviceSpan(load.getMemRefType())) {
          auto space =
              dyn_cast_or_null<IntegerAttr>(load.getMemRefType().getMemorySpace());
          if (!space || space.getInt() != 3)
            reject(op,
                   "device memory access requires an f32 global or shared span");
        }
        if (load.getMemRefType().getMemorySpace() ||
            !isa_and_nonnull<memref::AllocaOp>(load.getMemref().getDefiningOp()))
          require(ambient, "gpu_load");
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        if (store.getMemRefType().getRank() != 0 &&
            !deviceSpan(store.getMemRefType())) {
          auto space =
              dyn_cast_or_null<IntegerAttr>(store.getMemRefType().getMemorySpace());
          if (!space || space.getInt() != 3)
            reject(op,
                   "device memory access requires an f32 global or shared span");
        }
        if (store.getMemRefType().getMemorySpace() ||
            !isa_and_nonnull<memref::AllocaOp>(store.getMemref().getDefiningOp()))
          require(ambient, "gpu_store");
      } else if (isa<memref::GetGlobalOp, memref::DimOp>(op)) {
        // Allowed for workgroup shared buffers and dynamic span extents.
      } else if (isa<affine::AffineLoadOp>(op)) {
        require(ambient, "gpu_load");
      } else if (isa<affine::AffineStoreOp>(op)) {
        require(ambient, "gpu_store");
      } else if (isa<gpu::BlockIdOp, gpu::BlockDimOp, gpu::ThreadIdOp,
                     gpu::SubgroupIdOp, gpu::LaneIdOp>(op)) {
        require(ambient, "");
      } else if (isa<gpu::SubgroupReduceOp, gpu::ShuffleOp>(op)) {
        require(ambient, "gpu_warp_barrier");
      } else if (isa<gpu::BarrierOp>(op)) {
        require(ambient, "gpu_barrier");
      } else if (auto local = dyn_cast<memref::AllocaOp>(op)) {
        if (local.getType().getRank() || local.getType().getMemorySpace() ||
            !scalar(local.getType().getElementType()))
          reject(op,
                 "device local slots must hold scalars in the default address space");
      } else {
        const auto dialect = op->getName().getDialectNamespace();
        if (dialect != "arith" && dialect != "cf" && dialect != "scf" &&
            dialect != "affine" && dialect != "ub" &&
            !isa<func::FuncOp, func::ReturnOp>(op))
          reject(op, "operation is outside the context-only device profile");
      }
    });
  }
  module.walk([&](func::FuncOp function) {
    if (!bodies.contains(function))
      reject(function, "uncontracted function in device module");
  });
  if (!valid) return failure();
  return verifyGPUContextHierarchyAndLoans(module);
}

LogicalResult lowerMetreonDeviceToGPU(ModuleOp module) {
  // Recheck live MLIR contracts, scope hierarchy, path-sensitive ownership
  // typestates, async generation loans, and bounded-queue pipeline protocols.
  if (failed(verifyDeviceContexts(module))) return failure();

  bool hasUnloweredResource = false;
  module.walk([&](Operation *op) {
    if (isa<TransitionOp, ResourceOp>(op)) {
      op->emitError("device executable lowering requires resource physical specialization");
      hasUnloweredResource = true;
    }
  });
  if (hasUnloweredResource) return failure();

  auto *context = module.getContext();
  context->getOrLoadDialect<gpu::GPUDialect>();
  context->getOrLoadDialect<scf::SCFDialect>();
  context->getOrLoadDialect<affine::AffineDialect>();
  context->getOrLoadDialect<ub::UBDialect>();

  OpBuilder builder(context);
  builder.setInsertionPointToEnd(module.getBody());
  std::string name = "__metreon_device";
  for (unsigned suffix = 0; SymbolTable::lookupSymbolIn(module, name); ++suffix)
    name = "__metreon_device_" + std::to_string(suffix);
  auto device = builder.create<gpu::GPUModuleOp>(module.getLoc(), name);

  // Move any module-scope or `__metreon_bodies`-scope workgroup shared
  // `memref.global`s (space 3) into `device` so `memref.get_global` resolves
  // inside `gpu.module` once `CallableConversion` moves device bodies.
  SmallVector<memref::GlobalOp> topGlobals(module.getOps<memref::GlobalOp>());
  for (auto nested : module.getOps<ModuleOp>()) {
    for (memref::GlobalOp g : nested.getOps<memref::GlobalOp>())
      topGlobals.push_back(g);
  }
  for (memref::GlobalOp glob : topGlobals) {
    glob->moveBefore(device.getBody()->getTerminator());
  }

  if (failed(materializeAsyncCopyAndWaitOps(module, device))) return failure();
  if (failed(lowerVerifiedPipelinesToGPU(module, device))) return failure();

  RewritePatternSet patterns(context);
  patterns.add<IntrinsicConversion, ContextConversion>(context);
  patterns.add<CallableConversion<KernelOp>, CallableConversion<ProcedureOp>>(
      context, device);
  ConversionTarget target(*context);
  target.addIllegalDialect<MetreonDialect>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    return failure();

  SmallVector<ModuleOp> empty;
  for (auto nested : module.getOps<ModuleOp>()) {
    if (nested.getBody()->empty()) {
      empty.push_back(nested);
    } else {
      // Move any remaining workgroup globals inside `__metreon_bodies` into `device`.
      SmallVector<memref::GlobalOp> nestedGlobs(
          nested.getOps<memref::GlobalOp>());
      for (memref::GlobalOp g : nestedGlobs)
        g->moveBefore(device.getBody()->getTerminator());
      if (nested.getBody()->empty()) empty.push_back(nested);
    }
  }
  for (auto nested : empty) nested.erase();

  // Unify with GPULowering: lift device CFGs inside `gpu.module` (both kernels
  // and synthesized stage helpers) to structured `scf`/`affine` control flow.
  if (failed(structureGPUFunctionControlFlow(module, /*promoteAllScalarSlots=*/false)))
    return failure();

  module.walk([](Operation *op) {
    SmallVector<StringAttr> remove;
    for (auto attribute : op->getAttrs())
      if (attribute.getName().getValue().starts_with("metreon."))
        remove.push_back(attribute.getName());
    for (auto attrName : remove) op->removeAttr(attrName);
  });

  bool leftCFG = false;
  device.walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() == "cf") {
      op->emitError("device GPU structuring left unstructured CFG branches");
      leftCFG = true;
    }
  });
  if (leftCFG) return failure();
  if (failed(verifyGPUBarrierUniformityAndAccessProtocols(module)))
    return failure();
  return verify(module);
}

llvm::json::Object deviceLaunchManifest(ModuleOp semantic) {
  llvm::json::Array kernels;
  for (auto kernel : semantic.getOps<KernelOp>()) {
    llvm::json::Array parameters;
    for (auto item : kernel.getParameterAttrs()) {
      auto metadata = cast<DictionaryAttr>(item);
      auto type = metadata.getAs<TypeAttr>("type").getValue();
      std::string kind;
      const auto sourceType =
          metadata.getAs<DictionaryAttr>("source_type")
              .getAs<StringAttr>("name")
              .getValue();
      if (auto span = dyn_cast<MemRefType>(type)) {
        if ((sourceType != "gpu::span" && sourceType != "gpu::shared_span") ||
            span.getRank() != 1 || !span.isDynamicDim(0) ||
            !span.getElementType().isF32())
          throw std::runtime_error(
              "launch profile requires contiguous dynamic rank-one f32 spans");
        kind = "span_f32";
      } else if (type.isIndex())
        kind = "index";
      else if (type.isInteger(32) && sourceType == "i32")
        kind = "i32";
      else if (type.isF32())
        kind = "f32";
      else
        throw std::runtime_error(
            "launch profile supports gpu::span<f32>, index, i32 and f32 arguments");
      parameters.push_back(llvm::json::Object{
          {"name", metadata.getAs<StringAttr>("name").str()}, {"kind", kind}});
    }
    kernels.push_back(llvm::json::Object{
        {"name", kernel.getSymName().str()},
        {"parameters", std::move(parameters)}});
  }

  for (auto pipeOp : semantic.getOps<PipelineOp>()) {
    llvm::json::Array parameters;
    for (Attribute rAttr : pipeOp.getRecords()) {
      auto rDict = dyn_cast<DictionaryAttr>(rAttr);
      if (!rDict || dictStr(rDict, "kind") != "pipeline_parameter") continue;
      auto props = rDict.getAs<DictionaryAttr>("properties");
      const bool isOut = (dictStr(props, "direction") == "output");
      std::string tName;
      if (auto tDict = props ? props.getAs<DictionaryAttr>("type")
                             : DictionaryAttr{})
        tName = dictStr(tDict, "name");
      else
        tName = dictStr(props, "type");
      std::string kind;
      if (tName == "gpu::span" || tName == "gpu::span<f32>" ||
          (isOut && tName == "f32"))
        kind = "span_f32";
      else if (tName == "index")
        kind = "index";
      else if (tName == "i32")
        kind = "i32";
      else if (tName == "u64" || tName == "i64")
        kind = "i64";
      else if (tName == "f32")
        kind = "f32";
      else
        throw std::runtime_error("unsupported pipeline launch parameter type: " +
                                 tName);
      parameters.push_back(
          llvm::json::Object{{"name", dictStr(rDict, "name")}, {"kind", kind}});
    }
    kernels.push_back(llvm::json::Object{
        {"name", pipeOp.getSymName().str()},
        {"parameters", std::move(parameters)}});
  }

  return llvm::json::Object{{"schema_version", 1},
                            {"index_bits", 64},
                            {"memref_abi", "expanded_rank_one"},
                            {"kernels", std::move(kernels)}};
}
} // namespace metreon
