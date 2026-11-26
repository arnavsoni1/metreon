#include "metreon/MLIR/GPULowering.h"
#include "CanonicalSyntax.h"
#include "ScalarLowering.h"
#include "metreon/MLIR/Dialect.h"
#include "metreon/MLIR/GPUVerification.h"
#include "metreon/Verification/GPUOwnershipAndLoans.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

#include <map>
#include <set>
#include <stdexcept>

using namespace mlir;
namespace metreon {
namespace {
using Kind = graphir::CallableOperationKind;
using Syntax = parser::CanonicalSyntax;

struct GPULoweringIntrinsicConversion : OpConversionPattern<DeviceIntrinsicOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(DeviceIntrinsicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto args = adaptor.getInputs();
    if (op.getKind() == "global_id") {
      auto block = rewriter.create<gpu::BlockIdOp>(op.getLoc(), gpu::Dimension::x);
      auto width = rewriter.create<gpu::BlockDimOp>(op.getLoc(), gpu::Dimension::x);
      auto thread = rewriter.create<gpu::ThreadIdOp>(op.getLoc(), gpu::Dimension::x);
      auto offset = rewriter.create<arith::MulIOp>(op.getLoc(), block, width);
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, offset, thread);
    } else if (op.getKind() == "load") {
      rewriter.replaceOpWithNewOp<memref::LoadOp>(op, args[0], ValueRange{args[1]});
    } else if (op.getKind() == "store") {
      rewriter.create<memref::StoreOp>(op.getLoc(), args[2], args[0], ValueRange{args[1]});
      rewriter.eraseOp(op);
    } else if (op.getKind() == "barrier") {
      rewriter.replaceOpWithNewOp<gpu::BarrierOp>(op);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported device intrinsic in GPU control-flow lowering");
    }
    return success();
  }
};

class GPULowering {
  const graphir::Module &source;
  MLIRContext &context;
  OwningOpRef<ModuleOp> output;
  SourceLocation current;
  std::map<std::string, const graphir::CallableGraph *> procedures;
  SmallVector<const graphir::CallableGraph *> selected;
  std::set<std::string> selectedNames, kernels;
  bool usesDeviceIntrinsics = false;

  [[noreturn]] void fail(const std::string &message) { throw std::runtime_error(message); }
  Location location(SourceLocation loc) {
    return FileLineColLoc::get(&context, source.sourceName(), loc.line, loc.column);
  }
  void gpuContext(const graphir::ContextMetadataRef &metadata, bool loop) {
    if (!metadata) {
      if (loop) fail("GPU loops require Gpu::Loop evidence");
      return;
    }
    source.requireContextUse(metadata, graphir::ContextUse::MetadataAttachment);
    const auto type = Syntax::type(metadata->name());
    if (type.name.components.empty() || type.name.components.front() != "Gpu" ||
        (loop ? type.name.str() != "Gpu::Loop" : type.name.components.back() == "Loop"))
      fail(loop ? "GPU loops require Gpu::Loop evidence" : "GPU lowering cannot consume a non-GPU execution context");
    if (loop) {
      if (metadata->genericArity()) fail("GPU loop evidence cannot have generic arguments");
      for (const auto &node : source.nodes())
        if (node.kind == graphir::NodeKind::Grant && node.context && node.context->hasSameIdentity(*metadata))
          fail("GPU loop contexts cannot grant execution authority");
    } else if (!verification::classifyContextExecutionScope(metadata->name())) {
      fail("unsupported GPU execution context scope `" + metadata->name() + "`");
    }
  }
  void select(const graphir::CallableGraph &graph) {
    if (selectedNames.insert(graph.name()).second) selected.push_back(&graph);
  }
  void preflight() {
    for (const auto &graph : source.procedureGraphs()) {
      procedures.emplace(graph.name(), &graph);
      if (graph.attributes().at("execution_domain") == "Gpu") select(graph);
    }
    for (const auto &graph : source.kernelGraphs()) {
      kernels.insert(graph.name());
      select(graph);
    }
    if (selected.empty()) fail("GPU emission requires a GPU kernel or GPU-context procedure");
    for (std::size_t index = 0; index < selected.size(); ++index) {
      const auto &graph = *selected[index];
      current = graph.location();
      const auto domain = graph.attributes().at("execution_domain");
      if (domain != "Gpu" && domain != "context_independent")
        fail("GPU calls cannot cross into host or hardware execution domains");
      gpuContext(graph.context(), false);
      for (const auto &parameter : graph.parameters()) {
        if (parameter.context) gpuContext(parameter.context, false);
        if (parameter.type == "gpu::span<f32>") usesDeviceIntrinsics = true;
      }
      graphir::walkCallableOperationsWithContext(graph.body(), {}, [&](const auto &op, const auto &effective) {
        current = op.location;
        if (op.kind == Kind::ContextRegion && !op.context)
          fail("GPU context region requires an explicit GPU context");
        gpuContext(effective, false);
        if (effective && graph.context() && !graph.context()->hasSameIdentity(*effective))
          fail("GPU operation requires a different nominal execution context");
        const bool loop = op.kind == Kind::For || op.kind == Kind::While || op.kind == Kind::Until || op.kind == Kind::Loop;
        if (loop || op.loopContext) gpuContext(op.loopContext, true);
        if (graphir::isBehavioralOrAsyncOperation(op.kind))
          fail("GPU scalar control-flow lowering requires behavioral/async legalization first");
        switch (op.kind) {
        case Kind::IteratorInit: case Kind::IteratorHasNext:
        case Kind::IteratorValue: case Kind::IteratorAdvance:
          fail("GPU container iteration requires an explicit extent, layout and access contract");
        case Kind::Call: {
          const auto calleeKind = op.attributes.at("callee_kind");
          if (calleeKind == "gpu_builtin") {
            usesDeviceIntrinsics = true;
            if (!effective) fail("GPU intrinsic requires an execution context");
            const auto scope = verification::classifyContextExecutionScope(effective->name());
            if (!scope || !verification::isIntrinsicAllowedInScope(*scope, op.name))
              fail("GPU intrinsic `" + op.name + "` is not allowed in `" + effective->name() + "` scope");
            break;
          }
          if (calleeKind != "procedure")
            fail("GPU resource transitions require a checked device implementation");
          auto target = procedures.find(op.name);
          if (target == procedures.end()) fail("GPU call has no declared procedure");
          if (target->second->context() &&
              (!graph.context() || !target->second->context()->hasSameIdentity(*graph.context()) ||
               !effective || !target->second->context()->hasSameIdentity(*effective)))
            fail("GPU call requires a different nominal execution context");
          select(*target->second);
          break;
        }
        case Kind::Until:
          if (op.attributes.at("stop_value") != "true" && op.attributes.at("stop_value") != "false")
            fail("invalid GraphIR until stop value");
          break;
        default: break;
        }
      });
    }
  }

public:
  GPULowering(const graphir::Module &source, MLIRContext &context, OwningOpRef<ModuleOp> semantic)
      : source(source), context(context), output(std::move(semantic)) {}
  FailureOr<OwningOpRef<ModuleOp>> run() {
    try {
      preflight();
      context.getOrLoadDialect<scf::SCFDialect>();
      context.getOrLoadDialect<affine::AffineDialect>();
      context.getOrLoadDialect<gpu::GPUDialect>();
      context.getOrLoadDialect<ub::UBDialect>();

      auto device = detail::lowerScalarCFG(source, selected, context, 64, usesDeviceIntrinsics);
      if (failed(device)) return failure();

      if (usesDeviceIntrinsics) {
        for (const auto *graph : selected) {
          auto fn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
              **device, StringAttr::get(&context, graph->name()));
          if (fn && graph->context()) {
            fn->setAttr("metreon.context",
                        FlatSymbolRefAttr::get(&context, graph->context()->identifier()));
          }
        }
        if (failed(verifyGPUContextHierarchyAndLoans(**device))) return failure();
        RewritePatternSet intrinsicPatterns(&context);
        intrinsicPatterns.add<GPULoweringIntrinsicConversion>(&context);
        ConversionTarget intrinsicTarget(context);
        intrinsicTarget.addIllegalOp<DeviceIntrinsicOp>();
        intrinsicTarget.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
        if (failed(applyFullConversion(**device, intrinsicTarget, std::move(intrinsicPatterns))))
          return failure();
        SmallVector<ContextOp> ctxOps((*device)->getOps<ContextOp>());
        for (ContextOp ctxOp : ctxOps) ctxOp.erase();
        (**device).walk([](Operation *op) {
          SmallVector<StringAttr> remove;
          for (NamedAttribute attr : op->getAttrs()) {
            if (attr.getName().getValue().starts_with("metreon.") &&
                attr.getName().getValue() != "metreon.synthetic_iv") {
              remove.push_back(attr.getName());
            }
          }
          for (StringAttr attrName : remove) op->removeAttr(attrName);
        });
      }

      if (failed(structureGPUFunctionControlFlow(**device, /*promoteAllScalarSlots=*/true,
                                                 /*promoteAffine=*/false)))
        return failure();
      if (failed(verify(**device))) return failure();

      OpBuilder builder(&context);
      builder.setInsertionPointToEnd(output->getBody());
      std::string name = "__metreon_gpu";
      for (unsigned suffix = 0; SymbolTable::lookupSymbolIn(*output, name); ++suffix)
        name = "__metreon_gpu_" + std::to_string(suffix);
      auto gpuModule = builder.create<gpu::GPUModuleOp>(location({}), name);
      SmallVector<func::FuncOp> functions((*device)->getOps<func::FuncOp>());
      for (auto function : functions) {
        if (kernels.contains(function.getSymName().str())) {
          builder.setInsertionPoint(gpuModule.getBody()->getTerminator());
          auto kernel = builder.create<gpu::GPUFuncOp>(function.getLoc(), function.getSymName(), function.getFunctionType());
          kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), builder.getUnitAttr());
          kernel.getBody().takeBody(function.getBody());
          kernel.walk([&](func::ReturnOp ret) {
            OpBuilder atReturn(ret);
            atReturn.create<gpu::ReturnOp>(ret.getLoc(), ret.getOperands());
            ret.erase();
          });
          function.erase();
        } else {
          function->moveBefore(gpuModule.getBody()->getTerminator());
        }
      }
      promoteGPUStructuredLoopsAndBranches(gpuModule);

      for (const auto *graph : selected) {
        if (auto *contract = SymbolTable::lookupSymbolIn(*output, graph->name())) {
          contract->setAttr("metreon.device_body", SymbolRefAttr::get(&context, name,
              {FlatSymbolRefAttr::get(&context, graph->name())}));
        }
      }
      (*output)->setAttr("metreon.gpu_lowering", builder.getStringAttr("structured_scalar_control_flow"));
      (*output)->setAttr("metreon.gpu_index_bits", builder.getI64IntegerAttr(64));
      (*output)->setAttr("metreon.ordinary_lowering", builder.getStringAttr("partial_gpu_scalar"));
      bool complete = true;
      gpuModule.walk([&](Operation *op) {
        if (op->getName().getDialectNamespace() == "cf" || isa<memref::AllocaOp>(op)) {
          op->emitError("GPU scalar restructuring left CFG branches or scalar storage");
          complete = false;
        } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
          if (loadOp.getMemRefType().getRank() == 0) {
            op->emitError("GPU scalar restructuring left CFG branches or scalar storage");
            complete = false;
          }
        } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
          if (storeOp.getMemRefType().getRank() == 0) {
            op->emitError("GPU scalar restructuring left CFG branches or scalar storage");
            complete = false;
          }
        }
      });
      if (!complete || failed(verify(*output))) return failure();
      return std::move(output);
    } catch (const std::exception &error) {
      emitError(location(current)) << "GraphIR GPU control-flow lowering: " << error.what();
      return failure();
    }
  }
};
} // namespace

FailureOr<OwningOpRef<ModuleOp>> lowerGPUControlFlow(const graphir::Module &source, MLIRContext &context) {
  bool hasExecutableBuiltins = false;
  auto checkBuiltins = [&](const std::vector<graphir::CallableGraph> &graphs) {
    for (const auto &g : graphs) {
      graphir::walkCallableOperations(g.body(), [&](const auto &op) {
        if (op.kind == graphir::CallableOperationKind::Call &&
            op.attributes.contains("callee_kind")) {
          const auto &ck = op.attributes.at("callee_kind");
          if (ck == "gpu_builtin" || ck == "host_builtin")
            hasExecutableBuiltins = true;
        }
      });
    }
  };
  checkBuiltins(source.kernelGraphs());
  checkBuiltins(source.procedureGraphs());

  OwningOpRef<ModuleOp> semantic;
  if (hasExecutableBuiltins) {
    auto contextsOnly = importGraphIRContexts(source, context);
    if (failed(contextsOnly)) return failure();
    semantic = std::move(*contextsOnly);
  } else {
    auto fullSemantic = importGraphIR(source, context);
    if (failed(fullSemantic)) return failure();
    semantic = std::move(*fullSemantic);
  }
  return GPULowering(source, context, std::move(semantic)).run();
}
} // namespace metreon
