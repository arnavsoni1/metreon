#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace metreon {

// Lifts unstructured CFGs inside `func.func` and `gpu.func` operations into
// structured `scf` and `affine` control flow while preserving potentially
// non-terminating loops (`scf.while`). When `promoteAllScalarSlots` is false,
// only synthetic loop-induction slots (`metreon.synthetic_iv`) are promoted to
// SSA so counted loops become `scf.for`/`affine.for` while explicit mutable
// user locals remain rank-0 `memref.alloca` slots for target private storage.
::mlir::LogicalResult
structureGPUFunctionControlFlow(::mlir::ModuleOp module,
                                bool promoteAllScalarSlots = true,
                                bool promoteAffine = true);

void promoteGPUStructuredLoopsAndBranches(::mlir::Operation *root);

// Enforces Phase GPU-2 static guarantees directly on live MLIR:
// 1. Nominal Host::* and Gpu::{Grid, Block, Warp, Thread} context scope
//    hierarchy, capability-to-scope legality, and grant-free loop evidence.
// 2. Static host/device address-space dereference isolation across memref and
//    affine memory operations.
// 3. Path-sensitive resource/owned handle typestates (rejecting double
//    consumption, asymmetric branch leaks, loop-backedge consumption/typestate
//    drift, and unawaited `must_complete` resources at exit).
// 4. `(AllocationId, SlotId, Generation)` async-copy read/write loan tracking
//    across `scf`/`affine` regions and `cf` blocks (rejecting pre-wait buffer
//    accesses, stale-generation completion reuse, asymmetric branch waits, and
//    unawaited loop/exit transfers).
::mlir::LogicalResult verifyGPUContextHierarchyAndLoans(::mlir::ModuleOp module);

} // namespace metreon
