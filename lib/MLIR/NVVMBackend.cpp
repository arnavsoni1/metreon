#include "metreon/MLIR/DeviceExecution.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;
namespace metreon {
namespace {

bool isGPURuntimeAdapterSymbol(StringRef name) {
  return name.starts_with("__gpu_ring_") || name.starts_with("__gpu_async_");
}

LogicalResult specializeNVVMRingAndAsyncAdapters(ModuleOp mod) {
  MLIRContext *ctx = mod.getContext();
  Type i32Ty = IntegerType::get(ctx, 32);
  Type f32Ty = Float32Type::get(ctx);

  SmallVector<LLVM::CallOp> adapterCalls;
  mod.walk([&](LLVM::CallOp call) {
    if (auto callee = call.getCallee()) {
      if (isGPURuntimeAdapterSymbol(*callee)) adapterCalls.push_back(call);
    }
  });

  for (LLVM::CallOp call : adapterCalls) {
    const StringRef name = *call.getCallee();
    const Location loc = call.getLoc();
    OperandRange args = call.getOperands();
    OpBuilder b(call);

    auto gepRank1 = [&](unsigned memrefStart, Value idxVal, Type elemTy) -> Value {
      Value alignedPtr = args[memrefStart + 1];
      Value offset = args[memrefStart + 2];
      Value totalIdx = b.create<LLVM::AddOp>(loc, offset, idxVal);
      return b.create<LLVM::GEPOp>(loc, alignedPtr.getType(), elemTy,
                                   alignedPtr, ValueRange{totalIdx});
    };

    auto emitPTXAcqRelFence = [&](StringRef symName) {
      const StringRef ptxFence = symName.contains("_dev")
                                     ? StringRef("fence.acq_rel.gpu;")
                                     : StringRef("fence.acq_rel.cta;");
      b.create<LLVM::InlineAsmOp>(
          loc, TypeRange{}, ValueRange{}, ptxFence, "~{memory}",
          /*has_side_effects=*/true, /*is_align_stack=*/false,
          LLVM::AsmDialectAttr{}, ArrayAttr{});
    };

    if (name.starts_with("__gpu_ring_load_acquire_")) {
      if (args.size() != 6) return call.emitOpError("invalid ring acquire-load arity");
      Value ptr = gepRank1(0, args[5], i32Ty);
      Value zero = b.create<LLVM::ConstantOp>(loc, i32Ty, b.getI32IntegerAttr(0));
      auto rmw = b.create<LLVM::AtomicRMWOp>(
          loc, LLVM::AtomicBinOp::add, ptr, zero,
          LLVM::AtomicOrdering::monotonic);
      emitPTXAcqRelFence(name);
      call.getResult().replaceAllUsesWith(rmw.getRes());
      call.erase();
    } else if (name.starts_with("__gpu_ring_load_relaxed_")) {
      if (args.size() != 6) return call.emitOpError("invalid ring relaxed-load arity");
      Value ptr = gepRank1(0, args[5], i32Ty);
      Value zero = b.create<LLVM::ConstantOp>(loc, i32Ty, b.getI32IntegerAttr(0));
      auto rmw = b.create<LLVM::AtomicRMWOp>(
          loc, LLVM::AtomicBinOp::add, ptr, zero,
          LLVM::AtomicOrdering::monotonic);
      call.getResult().replaceAllUsesWith(rmw.getRes());
      call.erase();
    } else if (name.starts_with("__gpu_ring_store_release_")) {
      if (args.size() != 7) return call.emitOpError("invalid ring release-store arity");
      Value ptr = gepRank1(0, args[5], i32Ty);
      emitPTXAcqRelFence(name);
      (void)b.create<LLVM::AtomicRMWOp>(
          loc, LLVM::AtomicBinOp::xchg, ptr, args[6],
          LLVM::AtomicOrdering::monotonic);
      call.erase();
    } else if (name.starts_with("__gpu_ring_read_f32_")) {
      if (args.size() != 6) return call.emitOpError("invalid ring f32-read arity");
      Value ptr = gepRank1(0, args[5], f32Ty);
      auto ld = b.create<LLVM::LoadOp>(loc, f32Ty, ptr, /*alignment=*/4);
      call.getResult().replaceAllUsesWith(ld.getRes());
      call.erase();
    } else if (name.starts_with("__gpu_ring_write_f32_")) {
      if (args.size() != 7) return call.emitOpError("invalid ring f32-write arity");
      Value ptr = gepRank1(0, args[5], f32Ty);
      b.create<LLVM::StoreOp>(loc, args[6], ptr, /*alignment=*/4);
      call.erase();
    } else if (name.starts_with("__gpu_ring_read_i32_")) {
      if (args.size() != 6) return call.emitOpError("invalid ring i32-read arity");
      Value ptr = gepRank1(0, args[5], i32Ty);
      auto ld = b.create<LLVM::LoadOp>(loc, i32Ty, ptr, /*alignment=*/4);
      call.getResult().replaceAllUsesWith(ld.getRes());
      call.erase();
    } else if (name.starts_with("__gpu_ring_write_i32_")) {
      if (args.size() != 7) return call.emitOpError("invalid ring i32-write arity");
      Value ptr = gepRank1(0, args[5], i32Ty);
      b.create<LLVM::StoreOp>(loc, args[6], ptr, /*alignment=*/4);
      call.erase();
    } else if (name.starts_with("__gpu_async_copy_elem_g2s_f32")) {
      if (args.size() != 11) return call.emitOpError("invalid async g2s copy arity");
      Value dstPtr = gepRank1(0, args[10], f32Ty);
      Value srcPtr = gepRank1(5, args[10], f32Ty);
      b.create<NVVM::CpAsyncOp>(loc, dstPtr, srcPtr, /*size=*/4u,
                                NVVM::LoadCacheModifierKind::CA, Value{});
      call.erase();
    } else if (name.starts_with("__gpu_async_copy_elem_g2g_f32")) {
      if (args.size() != 11) return call.emitOpError("invalid async g2g copy arity");
      Value dstPtr = gepRank1(0, args[10], f32Ty);
      Value srcPtr = gepRank1(5, args[10], f32Ty);
      auto val = b.create<LLVM::LoadOp>(loc, f32Ty, srcPtr, /*alignment=*/4);
      b.create<LLVM::StoreOp>(loc, val.getRes(), dstPtr, /*alignment=*/4);
      call.erase();
    } else if (name == "__gpu_async_commit_group") {
      b.create<NVVM::CpAsyncCommitGroupOp>(loc);
      call.erase();
    } else if (name == "__gpu_async_wait_group") {
      b.create<NVVM::CpAsyncWaitGroupOp>(loc, /*n=*/0u);
      call.erase();
    } else {
      return call.emitOpError("unknown GPU runtime adapter symbol: ") << name;
    }
  }

  SmallVector<LLVM::LLVMFuncOp> adapterDecls;
  mod.walk([&](LLVM::LLVMFuncOp fn) {
    if (isGPURuntimeAdapterSymbol(fn.getName())) adapterDecls.push_back(fn);
  });
  for (LLVM::LLVMFuncOp fn : adapterDecls) fn.erase();
  return success();
}

} // namespace

FailureOr<OwningOpRef<ModuleOp>> lowerDeviceToNVVM(ModuleOp module) {
  auto *context = module.getContext();
  context->getOrLoadDialect<LLVM::LLVMDialect>();
  context->getOrLoadDialect<NVVM::NVVMDialect>();
  context->getOrLoadDialect<cf::ControlFlowDialect>();
  context->getOrLoadDialect<scf::SCFDialect>();
  context->getOrLoadDialect<affine::AffineDialect>();
  context->getOrLoadDialect<ub::UBDialect>();

  OwningOpRef<ModuleOp> working(module.clone());
  PassManager lowerStructured(context);
  lowerStructured.addPass(createLowerAffinePass());
  lowerStructured.addPass(createConvertSCFToCFPass());
  if (failed(lowerStructured.run(*working))) return failure();

  LowerToLLVMOptions options(context);
  options.overrideIndexBitwidth(64);
  LLVMTypeConverter converter(context, options);
  RewritePatternSet patterns(context);
  populateGpuToNVVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  ub::populateUBToLLVMConversionPatterns(converter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
  ConversionTarget target(*context);
  configureGpuToNVVMConversionLegality(target);
  target.addLegalDialect<LLVM::LLVMDialect, NVVM::NVVMDialect>();
  target.addLegalOp<ModuleOp, gpu::GPUModuleOp, gpu::ModuleEndOp>();
  if (failed(applyFullConversion(*working, target, std::move(patterns))) || failed(verify(*working))) return failure();
  auto output = OwningOpRef<ModuleOp>(ModuleOp::create(working->getLoc()));
  unsigned modules = 0;
  for (auto device : working->getOps<gpu::GPUModuleOp>()) {
    ++modules;
    for (auto &op : device.getBody()->without_terminator()) output->getBody()->push_back(op.clone());
  }
  if (modules != 1) { module.emitError("NVVM execution requires exactly one device module"); return failure(); }
  output->walk([](LLVM::LLVMFuncOp function) { function->removeAttr("gpu.kernel"); });
  if (failed(specializeNVVMRingAndAsyncAdapters(*output))) return failure();
  OpBuilder builder(context);
  (*output)->setAttr("llvm.target_triple", builder.getStringAttr("nvptx64-nvidia-cuda"));
  if (failed(verify(*output))) return failure();
  return output;
}

FailureOr<std::string> emitDevicePTX(ModuleOp nvvm, llvm::StringRef chip) {
  if (chip.size() < 5 || !chip.starts_with("sm_") ||
      !llvm::all_of(chip.drop_front(3), [](char c) { return c >= '0' && c <= '9'; })) {
    nvvm.emitError("CUDA architecture must be sm_<compute capability>, for example sm_89");
    return failure();
  }
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
  std::string error;
  const std::string triple = "nvptx64-nvidia-cuda";
  const auto *target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) { nvvm.emitError(error); return failure(); }
  auto supported = std::unique_ptr<llvm::MCSubtargetInfo>(target->createMCSubtargetInfo(triple, "", ""));
  if (!supported || !supported->isCPUStringValid(chip)) {
    nvvm.emitError() << "CUDA architecture is unsupported by this LLVM build: " << chip;
    return failure();
  }
  llvm::TargetOptions options;
  auto machine = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      triple, chip, "+ptx80", options, std::nullopt, std::nullopt, llvm::CodeGenOptLevel::None));
  if (!machine) { nvvm.emitError("cannot create the NVPTX target machine"); return failure(); }
  registerBuiltinDialectTranslation(*nvvm.getContext());
  registerLLVMDialectTranslation(*nvvm.getContext());
  registerNVVMDialectTranslation(*nvvm.getContext());
  llvm::LLVMContext context;
  auto ir = translateModuleToLLVMIR(nvvm, context);
  if (!ir) return failure();
  ir->setTargetTriple(triple);
  ir->setDataLayout(machine->createDataLayout());
  if (llvm::verifyModule(*ir, &llvm::errs())) { nvvm.emitError("invalid translated LLVM IR"); return failure(); }
  llvm::SmallVector<char> bytes;
  llvm::raw_svector_ostream assembly(bytes);
  llvm::legacy::PassManager emission;
  if (machine->addPassesToEmitFile(emission, assembly, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
    nvvm.emitError("NVPTX assembly emission is unavailable"); return failure();
  }
  emission.run(*ir);
  return std::string(bytes.begin(), bytes.end());
}
} // namespace metreon
