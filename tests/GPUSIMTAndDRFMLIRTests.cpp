#include "metreon/GraphIR/Lowering.h"
#include "metreon/GraphIR/Passes.h"
#include "metreon/MLIR/DeviceExecution.h"
#include "metreon/MLIR/Dialect.h"
#include "metreon/MLIR/GPULowering.h"
#include "metreon/MLIR/GPUVerification.h"
#include "metreon/Parser/Parser.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

using namespace metreon;

void check(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream os(text);
  module.print(os);
  return os.str();
}

void registerDialects(mlir::MLIRContext &context) {
  context.getOrLoadDialect<MetreonDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::gpu::GPUDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::affine::AffineDialect>();
  context.getOrLoadDialect<mlir::ub::UBDialect>();
}

mlir::OwningOpRef<mlir::ModuleOp> parseMLIR(const std::string &irText,
                                            mlir::MLIRContext &context) {
  registerDialects(context);
  auto parsed = mlir::parseSourceString<mlir::ModuleOp>(irText, &context);
  check(static_cast<bool>(parsed), "failed to parse GPU-3 test MLIR");
  check(mlir::succeeded(mlir::verify(*parsed)), "parsed GPU-3 MLIR is invalid");
  return parsed;
}

void expectProtocolAccepted(
    const std::string &irText, mlir::MLIRContext &context,
    const verification::GPUMemoryAccessProtocolOptions &opts = {}) {
  auto module = parseMLIR(irText, context);
  std::string diagText;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diag) {
    llvm::raw_string_ostream os(diagText);
    diag.print(os);
    return mlir::success();
  });
  check(mlir::succeeded(
            verifyGPUBarrierUniformityAndAccessProtocols(*module, opts)),
        "expected MLIR module to pass GPU-3 verification, got: " + diagText);
}

void expectProtocolRejected(
    const std::string &irText, mlir::MLIRContext &context,
    const std::string &expectedSubstring,
    const verification::GPUMemoryAccessProtocolOptions &opts = {}) {
  auto module = parseMLIR(irText, context);
  std::string diagText;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diag) {
    llvm::raw_string_ostream os(diagText);
    diag.print(os);
    return mlir::success();
  });
  check(mlir::failed(
            verifyGPUBarrierUniformityAndAccessProtocols(*module, opts)),
        "expected MLIR module to fail GPU-3 verification (" +
            expectedSubstring + "), but it succeeded");
  check(diagText.find(expectedSubstring) != std::string::npos,
        "expected diagnostic containing `" + expectedSubstring +
            "`, got: " + diagText);
}

void testMLIRStructuredSIMTAndSharedMemoryProtocols() {
  mlir::MLIRContext context;

  // 1. Positive: Aligned 2-barrier shared-memory 1D stencil inside `affine.for`
  //    on `gpu.func` workgroup attribution `memref<64xf32, 3>`.
  expectProtocolAccepted(
      R"(
      module {
        gpu.module @dev {
          gpu.func @stencil_ok(%out: memref<?xf32, 1>)
              workgroup(%smem: memref<64xf32, 3>) kernel {
            %tid = gpu.thread_id x
            %bid = gpu.block_id x
            %bdim = gpu.block_dim x
            %base = arith.muli %bid, %bdim : index
            %gid = arith.addi %base, %tid : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %init = arith.constant 1.0 : f32

            memref.store %init, %smem[%tid] : memref<64xf32, 3>
            gpu.barrier

            affine.for %step = 0 to 4 {
              %nbr_raw = arith.addi %tid, %c1 : index
              %nbr = arith.remsi %nbr_raw, %c64 : index
              %v = memref.load %smem[%nbr] : memref<64xf32, 3>
              gpu.barrier
              memref.store %v, %smem[%tid] : memref<64xf32, 3>
              gpu.barrier
            }

            %res = memref.load %smem[%tid] : memref<64xf32, 3>
            memref.store %res, %out[%gid] : memref<?xf32, 1>
            gpu.return
          }
        }
      })",
      context);

  // 2. Negative: Missing second `gpu.barrier` inside `affine.for` stencil loop!
  //    Must be rejected via cross-iteration loop barrier aligning!
  expectProtocolRejected(
      R"(
      module {
        gpu.module @dev {
          gpu.func @stencil_missing_second_barrier()
              workgroup(%smem: memref<64xf32, 3>) kernel {
            %tid = gpu.thread_id x
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %init = arith.constant 1.0 : f32
            memref.store %init, %smem[%tid] : memref<64xf32, 3>
            gpu.barrier

            affine.for %step = 0 to 4 {
              %nbr_raw = arith.addi %tid, %c1 : index
              %nbr = arith.remsi %nbr_raw, %c64 : index
              %v = memref.load %smem[%nbr] : memref<64xf32, 3>
              gpu.barrier
              memref.store %v, %smem[%tid] : memref<64xf32, 3>
            }
            gpu.return
          }
        }
      })",
      context, "gpu.map.read_write_data_race");

  // 3. Positive: Warp-specialized (`tid >> 5 == 0`, i.e. `tid / 32 == 0`) branch
  //    executing `gpu.subgroup_reduce` to synchronize intra-warp shared exchange,
  //    plus 2D `[%warp, %lane]` conflict-free shared tile write!
  expectProtocolAccepted(
      R"(
      module {
        gpu.module @dev {
          gpu.func @warp_spec_and_2d_tile()
              workgroup(%wbuf: memref<32xf32, 3>, %tile2d: memref<2x32xf32, 3>) kernel {
            %tid = gpu.thread_id x
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c5 = arith.constant 5 : index
            %c31 = arith.constant 31 : index
            %f1 = arith.constant 2.0 : f32
            %i0 = arith.constant 0 : i32

            %warp = arith.shrui %tid, %c5 : index
            %lane = arith.andi %tid, %c31 : index
            memref.store %f1, %tile2d[%warp, %lane] : memref<2x32xf32, 3>

            %is_warp0 = arith.cmpi eq, %warp, %c0 : index
            scf.if %is_warp0 {
              memref.store %f1, %wbuf[%lane] : memref<32xf32, 3>
              %sync = gpu.subgroup_reduce add %i0 uniform : (i32) -> (i32)
              %next_raw = arith.addi %tid, %c1 : index
              %next_lane = arith.andi %next_raw, %c31 : index
              %v = memref.load %wbuf[%next_lane] : memref<32xf32, 3>
            }
            gpu.return
          }
        }
      })",
      context);

  // 4. Negative: `gpu.barrier` inside warp-uniform `scf.if (%warp == 0)` must fail!
  expectProtocolRejected(
      R"(
      module {
        gpu.module @dev {
          gpu.func @bad_wg_barrier_in_warp0() kernel {
            %warp = gpu.subgroup_id : index
            %c0 = arith.constant 0 : index
            %cond = arith.cmpi eq, %warp, %c0 : index
            scf.if %cond {
              gpu.barrier
            }
            gpu.return
          }
        }
      })",
      context, "gpu.simt.divergent_workgroup_barrier");

  // 5. Negative: `gpu.shuffle` inside lane-divergent `scf.if (%tid % 2 == 0)` must fail!
  expectProtocolRejected(
      R"(
      module {
        gpu.module @dev {
          gpu.func @bad_divergent_shuffle(%v: f32) kernel {
            %tid = gpu.thread_id x
            %c1 = arith.constant 1 : index
            %c0 = arith.constant 0 : index
            %odd = arith.andi %tid, %c1 : index
            %is_even = arith.cmpi eq, %odd, %c0 : index
            scf.if %is_even {
              %off = arith.constant 1 : i32
              %w = arith.constant 32 : i32
              %res, %valid = gpu.shuffle xor %v, %off, %w : f32
            }
            gpu.return
          }
        }
      })",
      context, "gpu.simt.ambiguous_subgroup_participation");

  // 6. Negative: Inter-warp shared race separated ONLY by `gpu.subgroup_reduce`!
  expectProtocolRejected(
      R"(
      module {
        gpu.module @dev {
          gpu.func @bad_inter_warp_smem()
              workgroup(%smem: memref<64xf32, 3>) kernel {
            %tid = gpu.thread_id x
            %c0 = arith.constant 0 : index
            %i0 = arith.constant 0 : i32
            %f1 = arith.constant 1.0 : f32
            %is_t0 = arith.cmpi eq, %tid, %c0 : index
            scf.if %is_t0 {
              memref.store %f1, %smem[%c0] : memref<64xf32, 3>
            }
            %sync = gpu.subgroup_reduce add %i0 uniform : (i32) -> (i32)
            %r = memref.load %smem[%c0] : memref<64xf32, 3>
            gpu.return
          }
        }
      })",
      context, "gpu.map.read_write_data_race");

  // 7. Negative: Static out-of-bounds `affine.store` to `memref<64xf32, 3>`.
  expectProtocolRejected(
      R"(
      module {
        gpu.module @dev {
          gpu.func @bad_static_oob()
              workgroup(%smem: memref<64xf32, 3>) kernel {
            %tid = gpu.thread_id x
            %c0 = arith.constant 0 : index
            %f1 = arith.constant 1.0 : f32
            %is_t0 = arith.cmpi eq, %tid, %c0 : index
            scf.if %is_t0 {
              affine.store %f1, %smem[64] : memref<64xf32, 3>
            }
            gpu.return
          }
        }
      })",
      context, "gpu.map.out_of_bounds_access");
}

void testEndToEndSourceLoweringEnforcement(const char *vectorAddPath) {
  std::ifstream input(vectorAddPath);
  check(static_cast<bool>(input), "cannot read vector_add.mtr");
  const std::string vecSrc((std::istreambuf_iterator<char>(input)), {});

  mlir::MLIRContext context;
  registerDialects(context);

  {
    parser::Parser vp(vecSrc);
    auto vg = graphir::lowerModule(vp.parseModule(), "vector_add.mtr", true);
    auto vgpu = lowerGPUControlFlow(vg, context);
    check(mlir::succeeded(vgpu), "vector_add lowerGPUControlFlow failed");
    verification::GPUMemoryAccessProtocolOptions strictBounds;
    strictBounds.requireProvenBounds = true;
    check(mlir::succeeded(
              verifyGPUBarrierUniformityAndAccessProtocols(**vgpu, strictBounds)),
          "vector_add MLIR failed strict requireProvenBounds check");
  }

  // 1. Positive: `Gpu::Block` synchronized kernel lowers cleanly through BOTH
  //    `lowerGPUControlFlow` and `DeviceExecution` -> NVVM PTX (`bar.sync`) + ROCDL (`rocdl.s.barrier`)!
  const std::string syncBlockSrc = R"(
context Gpu::Block grants {gpu_load, gpu_store, gpu_barrier} block_ctx;
context Gpu::Loop gpu_loop;

kernel staged_scale(@cx: block_ctx; a: gpu::span<f32>, out: gpu::span<f32>, n: index) {
  index tid = gpu::thread_id(@cx);
  index bid = gpu::block_id(@cx);
  index bdim = gpu::block_dim(@cx);
  index i = bid * bdim + tid;
  f32 acc = 0.0;
  for<@cx: gpu_loop> (index step) in 0..2 {
    if (i < n) {
      acc = acc + gpu::load(@cx; a, i);
    }
    gpu::barrier(@cx);
  }
  if (i < n) {
    gpu::store(@cx; out, i, acc);
  }
}
)";
  for (const bool opt : {false, true}) {
    parser::Parser p1(syncBlockSrc);
    auto g1 = graphir::lowerModule(p1.parseModule(), "staged.mtr", true);
    if (opt) graphir::passes::runDefaultPipeline(g1);

    auto gpuMod = lowerGPUControlFlow(g1, context);
    check(mlir::succeeded(gpuMod), "lowerGPUControlFlow failed on staged_scale");

    auto devMod = importDeviceExecution(g1, context);
    check(mlir::succeeded(devMod) &&
              mlir::succeeded(lowerMetreonDeviceToGPU(**devMod)),
          "lowerMetreonDeviceToGPU failed on staged_scale");

    auto nvvm = lowerDeviceToNVVM(**devMod);
    check(mlir::succeeded(nvvm), "staged_scale NVVM lowering failed");
    auto ptx = emitDevicePTX(**nvvm, "sm_89");
    check(mlir::succeeded(ptx) && ptx->find("bar.sync") != std::string::npos,
          "staged_scale PTX missing bar.sync barrier instruction");

    auto rocdl = lowerDeviceToROCDL(**devMod);
    check(mlir::succeeded(rocdl) &&
              printModule(**rocdl).find("rocdl.barrier") != std::string::npos,
          "staged_scale ROCDL missing rocdl.barrier");
  }

  // 2. Negative: Divergent `gpu::barrier` in source is rejected by BOTH
  //    `lowerGPUControlFlow` and `lowerMetreonDeviceToGPU`!
  const std::string badDivSrc = R"(
context Gpu::Block grants {gpu_load, gpu_store, gpu_barrier} block_ctx;
kernel bad_div(@cx: block_ctx; n: index) {
  index i = gpu::global_id(@cx);
  if (i < n) {
    gpu::barrier(@cx);
  }
}
)";
  {
    parser::Parser p(badDivSrc);
    auto g = graphir::lowerModule(p.parseModule(), "bad_div.mtr", true);
    std::string diag1, diag2;
    {
      mlir::ScopedDiagnosticHandler h(&context, [&](mlir::Diagnostic &d) {
        llvm::raw_string_ostream os(diag1);
        d.print(os);
        return mlir::success();
      });
      check(mlir::failed(lowerGPUControlFlow(g, context)),
            "lowerGPUControlFlow accepted divergent gpu::barrier");
    }
    check(diag1.find("gpu.simt.divergent_workgroup_barrier") != std::string::npos,
          "expected divergent_workgroup_barrier diagnostic in lowerGPUControlFlow, got: " +
              diag1);

    auto dev = importDeviceExecution(g, context);
    check(mlir::succeeded(dev), "importDeviceExecution failed on bad_div");
    {
      mlir::ScopedDiagnosticHandler h(&context, [&](mlir::Diagnostic &d) {
        llvm::raw_string_ostream os(diag2);
        d.print(os);
        return mlir::success();
      });
      check(mlir::failed(lowerMetreonDeviceToGPU(**dev)),
            "lowerMetreonDeviceToGPU accepted divergent gpu::barrier");
    }
    check(diag2.find("gpu.simt.divergent_workgroup_barrier") != std::string::npos,
          "expected divergent_workgroup_barrier diagnostic in DeviceExecution, got: " +
              diag2);
  }

  // 3. Negative: Racy `gpu::store(@cx; out, 0, 1.0)` in source is rejected by BOTH
  //    `lowerGPUControlFlow` and `lowerMetreonDeviceToGPU`!
  const std::string badRaceSrc = R"(
context Gpu::Thread grants {gpu_store} thr_ctx;
kernel bad_race(@cx: thr_ctx; out: gpu::span<f32>) {
  gpu::store(@cx; out, 0, 1.0);
}
)";
  {
    parser::Parser p(badRaceSrc);
    auto g = graphir::lowerModule(p.parseModule(), "bad_race.mtr", true);
    std::string diag1, diag2;
    {
      mlir::ScopedDiagnosticHandler h(&context, [&](mlir::Diagnostic &d) {
        llvm::raw_string_ostream os(diag1);
        d.print(os);
        return mlir::success();
      });
      check(mlir::failed(lowerGPUControlFlow(g, context)),
            "lowerGPUControlFlow accepted racy out[0] store");
    }
    check(diag1.find("gpu.map.write_write_data_race") != std::string::npos,
          "expected write_write_data_race diagnostic in lowerGPUControlFlow, got: " +
              diag1);

    auto dev = importDeviceExecution(g, context);
    check(mlir::succeeded(dev), "importDeviceExecution failed on bad_race");
    {
      mlir::ScopedDiagnosticHandler h(&context, [&](mlir::Diagnostic &d) {
        llvm::raw_string_ostream os(diag2);
        d.print(os);
        return mlir::success();
      });
      check(mlir::failed(lowerMetreonDeviceToGPU(**dev)),
            "lowerMetreonDeviceToGPU accepted racy out[0] store");
    }
    check(diag2.find("gpu.map.write_write_data_race") != std::string::npos,
          "expected write_write_data_race diagnostic in DeviceExecution, got: " +
              diag2);
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    check(argc == 2, "expected vector_add.mtr fixture path");
    testMLIRStructuredSIMTAndSharedMemoryProtocols();
    testEndToEndSourceLoweringEnforcement(argv[1]);
    std::cout << "GPU Phase 3 MLIR SIMT uniformity and memory access protocol tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "GPUSIMTAndDRFMLIRTests failed: " << error.what() << '\n';
    return 1;
  }
}
