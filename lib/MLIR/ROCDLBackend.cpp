#include "metreon/MLIR/DeviceExecution.h"
#include "metreon/CodeGen/Toolchain.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;
namespace metreon {
namespace {

bool isGPURuntimeAdapterSymbol(StringRef name) {
  return name.starts_with("__gpu_ring_") || name.starts_with("__gpu_async_");
}

StringRef rocdlRingSyncScope(StringRef symbolName) {
  return symbolName.contains("_dev") ? StringRef("agent")
                                     : StringRef("workgroup");
}

LogicalResult specializeROCDLRingAndAsyncAdapters(ModuleOp mod) {
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

    if (name.starts_with("__gpu_ring_load_acquire_")) {
      if (args.size() != 6) return call.emitOpError("invalid ring acquire-load arity");
      Value ptr = gepRank1(0, args[5], i32Ty);
      const StringRef scope = rocdlRingSyncScope(name);
      Value zero = b.create<LLVM::ConstantOp>(loc, i32Ty, b.getI32IntegerAttr(0));
      auto rmw = b.create<LLVM::AtomicRMWOp>(
          loc, LLVM::AtomicBinOp::add, ptr, zero,
          LLVM::AtomicOrdering::monotonic, scope);
      b.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::acquire, scope);
      call.getResult().replaceAllUsesWith(rmw.getRes());
      call.erase();
    } else if (name.starts_with("__gpu_ring_load_relaxed_")) {
      if (args.size() != 6) return call.emitOpError("invalid ring relaxed-load arity");
      Value ptr = gepRank1(0, args[5], i32Ty);
      Value zero = b.create<LLVM::ConstantOp>(loc, i32Ty, b.getI32IntegerAttr(0));
      auto rmw = b.create<LLVM::AtomicRMWOp>(
          loc, LLVM::AtomicBinOp::add, ptr, zero,
          LLVM::AtomicOrdering::monotonic, rocdlRingSyncScope(name));
      call.getResult().replaceAllUsesWith(rmw.getRes());
      call.erase();
    } else if (name.starts_with("__gpu_ring_store_release_")) {
      if (args.size() != 7) return call.emitOpError("invalid ring release-store arity");
      Value ptr = gepRank1(0, args[5], i32Ty);
      const StringRef scope = rocdlRingSyncScope(name);
      b.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::release, scope);
      (void)b.create<LLVM::AtomicRMWOp>(
          loc, LLVM::AtomicBinOp::xchg, ptr, args[6],
          LLVM::AtomicOrdering::monotonic, scope);
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
    } else if (name.starts_with("__gpu_async_copy_elem_g2s_f32") ||
               name.starts_with("__gpu_async_copy_elem_g2g_f32")) {
      if (args.size() != 11) return call.emitOpError("invalid async copy arity");
      Value dstPtr = gepRank1(0, args[10], f32Ty);
      Value srcPtr = gepRank1(5, args[10], f32Ty);
      auto val = b.create<LLVM::LoadOp>(loc, f32Ty, srcPtr, /*alignment=*/4);
      b.create<LLVM::StoreOp>(loc, val.getRes(), dstPtr, /*alignment=*/4);
      call.erase();
    } else if (name == "__gpu_async_commit_group") {
      b.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::release, "workgroup");
      call.erase();
    } else if (name == "__gpu_async_wait_group") {
      b.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::acquire, "workgroup");
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

FailureOr<OwningOpRef<ModuleOp>> lowerDeviceToROCDL(ModuleOp module) {
  auto *context = module.getContext();
  context->getOrLoadDialect<LLVM::LLVMDialect>();
  context->getOrLoadDialect<ROCDL::ROCDLDialect>();
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
  // AMD allocas live in private address space 5. Obtain the target layout
  // before converting memref slots rather than inheriting a host layout.
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  std::string error;
  auto *amd = llvm::TargetRegistry::lookupTarget("amdgcn-amd-amdhsa", error);
  if (!amd) { module.emitError(error); return failure(); }
  auto layout = std::unique_ptr<llvm::TargetMachine>(amd->createTargetMachine(
      "amdgcn-amd-amdhsa", "gfx900", "", llvm::TargetOptions{}, llvm::Reloc::PIC_));
  if (!layout) { module.emitError("AMDGPU target layout unavailable"); return failure(); }
  options.dataLayout = layout->createDataLayout();
  working->walk([&](memref::AllocaOp local) {
    auto type = local.getType();
    if (!type.getMemorySpace()) {
      local.getResult().setType(MemRefType::get(type.getShape(), type.getElementType(), type.getLayout(),
          IntegerAttr::get(IntegerType::get(context, 64), 5)));
    }
  });
  options.overrideIndexBitwidth(64);
  LLVMTypeConverter converter(context, options);
  RewritePatternSet patterns(context);
  populateGpuToROCDLConversionPatterns(converter, patterns, gpu::amd::Runtime::HIP);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  ub::populateUBToLLVMConversionPatterns(converter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
  ConversionTarget target(*context);
  configureGpuToROCDLConversionLegality(target);
  target.addLegalDialect<LLVM::LLVMDialect, ROCDL::ROCDLDialect>();
  target.addLegalOp<ModuleOp, gpu::GPUModuleOp, gpu::ModuleEndOp>();
  if (failed(applyFullConversion(*working, target, std::move(patterns))) || failed(verify(*working))) return failure();
  // Hardware IDs and dimensions are unsigned 32-bit values. Preserve their
  // full range when realizing Metreon's 64-bit index arithmetic.
  SmallVector<LLVM::SExtOp> indexExtensions;
  working->walk([&](LLVM::SExtOp op) {
    if (isa_and_nonnull<ROCDL::BlockIdXOp, ROCDL::BlockIdYOp, ROCDL::BlockIdZOp,
        ROCDL::ThreadIdXOp, ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp,
        ROCDL::BlockDimXOp, ROCDL::BlockDimYOp, ROCDL::BlockDimZOp,
        ROCDL::GridDimXOp, ROCDL::GridDimYOp, ROCDL::GridDimZOp>(op.getArg().getDefiningOp())) indexExtensions.push_back(op);
  });
  for (auto op : indexExtensions) {
    OpBuilder builder(op);
    auto replacement = builder.create<LLVM::ZExtOp>(op.getLoc(), op.getType(), op.getArg());
    op.replaceAllUsesWith(replacement.getResult());
    op.erase();
  }
  auto output = OwningOpRef<ModuleOp>(ModuleOp::create(working->getLoc()));
  unsigned modules = 0;
  for (auto device : working->getOps<gpu::GPUModuleOp>()) {
    ++modules;
    for (auto &op : device.getBody()->without_terminator()) output->getBody()->push_back(op.clone());
  }
  if (modules != 1) { module.emitError("ROCDL emission requires exactly one device module"); return failure(); }
  output->walk([&](LLVM::LLVMFuncOp function) {
    function->removeAttr("gpu.kernel");
    if (function->hasAttr("rocdl.kernel"))
      function->setAttr("rocdl.flat_work_group_size", StringAttr::get(context, "1,1024"));
  });
  if (failed(specializeROCDLRingAndAsyncAdapters(*output))) return failure();
  (*output)->setAttr("llvm.target_triple", StringAttr::get(context, "amdgcn-amd-amdhsa"));
  if (failed(verify(*output))) return failure();
  return output;
}

FailureOr<std::string> emitDeviceAMDGPU(ModuleOp rocdl, llvm::StringRef chip, bool object) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();
  std::string error;
  const std::string triple = "amdgcn-amd-amdhsa";
  const auto *target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) { rocdl.emitError(error); return failure(); }
  auto supported = std::unique_ptr<llvm::MCSubtargetInfo>(target->createMCSubtargetInfo(triple, "gfx900", ""));
  if (!supported || !chip.starts_with("gfx") || !supported->isCPUStringValid(chip)) {
    rocdl.emitError() << "unsupported AMDGPU architecture: " << chip;
    return failure();
  }
  auto machine = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      triple, chip, "", llvm::TargetOptions{}, llvm::Reloc::PIC_, std::nullopt, llvm::CodeGenOptLevel::None));
  if (!machine) { rocdl.emitError("cannot create AMDGPU target machine"); return failure(); }
  registerBuiltinDialectTranslation(*rocdl.getContext());
  registerLLVMDialectTranslation(*rocdl.getContext());
  registerROCDLDialectTranslation(*rocdl.getContext());
  llvm::LLVMContext context;
  auto ir = translateModuleToLLVMIR(rocdl, context);
  if (!ir) return failure();
  ir->setTargetTriple(triple);
  ir->setDataLayout(machine->createDataLayout());
  ir->addModuleFlag(llvm::Module::Override, "amdgpu_code_object_version", 500);
  // Standard ROCDL workgroup-size operations translate to OCKL calls. Link the
  // actual device library and its ABI constant; do not leave unresolved device
  // calls or synthesize a different indexing contract in the source frontend.
  for (const char *path : {toolchain::ockl, toolchain::abi500}) {
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer) { rocdl.emitError() << "AMDGPU emission requires ROCm device bitcode: " << path; return failure(); }
    auto library = llvm::parseBitcodeFile((*buffer)->getMemBufferRef(), context);
    if (!library) { rocdl.emitError() << llvm::toString(library.takeError()); return failure(); }
    (*library)->setDataLayout(ir->getDataLayout());
    if (llvm::Linker::linkModules(*ir, std::move(*library), llvm::Linker::Flags::LinkOnlyNeeded)) {
      rocdl.emitError("failed to link ROCm device bitcode"); return failure();
    }
  }
  if (llvm::verifyModule(*ir, &llvm::errs())) { rocdl.emitError("invalid translated AMDGPU LLVM IR"); return failure(); }
  llvm::SmallVector<char> bytes;
  llvm::raw_svector_ostream stream(bytes);
  llvm::legacy::PassManager emission;
  if (machine->addPassesToEmitFile(emission, stream, nullptr,
          object ? llvm::CodeGenFileType::ObjectFile : llvm::CodeGenFileType::AssemblyFile)) {
    rocdl.emitError("AMDGPU code emission unavailable"); return failure();
  }
  emission.run(*ir);
  return std::string(bytes.begin(), bytes.end());
}
} // namespace metreon
