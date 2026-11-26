#ifndef METREON_GPU_PROFILES_TD
#define METREON_GPU_PROFILES_TD

include "MetreonHardwareProfiles.td"

// Build-time GPU verification and workgroup-pipeline contracts exported as JSON.
// Contexts and loop evidence remain compile-time metadata and never materialize
// as runtime SSA values, kernel arguments, or memory payloads.

def Metreon_GpuGlobalLoad : MetreonGrantProfile<"gpu_load", [],
    "Authorize device global or shared memory reads within a verified GPU context.">;
def Metreon_GpuGlobalStore : MetreonGrantProfile<"gpu_store", [],
    "Authorize device global or shared memory writes within a verified GPU context.">;
def Metreon_GpuWorkgroupBarrier : MetreonGrantProfile<"gpu_barrier", ["scope"],
    "Authorize uniform workgroup or subgroup synchronization in the named scope.">;
def Metreon_GpuChannelSend : MetreonGrantProfile<"channel_send", ["channel"],
    "Authorize bounded GPU queue publication with acquire/release visibility.">;
def Metreon_GpuChannelRecv : MetreonGrantProfile<"channel_recv", ["channel"],
    "Authorize bounded GPU queue consumption with acquire/release visibility.">;
def Metreon_GpuAsyncCopy : MetreonGrantProfile<"gpu_async_copy",
    ["source_address_space", "destination_address_space"],
    "Issue a generation-tracked async transfer retaining storage until wait.">;
def Metreon_GpuAsyncWait : MetreonGrantProfile<"gpu_async_wait", ["completion_kind"],
    "Complete a matching generation-tracked async transfer token.">;

def Metreon_HostGpuAlloc : MetreonGrantProfile<"gpu_alloc", [],
    "Authorize device global buffer allocation from a host execution context.">;
def Metreon_HostGpuFree : MetreonGrantProfile<"gpu_free", [],
    "Authorize device buffer release when no in-flight loan or launch remains.">;
def Metreon_HostGpuCopy : MetreonGrantProfile<"gpu_copy", [],
    "Authorize host-device memory transfers with checked span extents.">;
def Metreon_HostGpuAwait : MetreonGrantProfile<"gpu_await", [],
    "Authorize host synchronization with pending device execution.">;
def Metreon_HostGpuLaunch : MetreonGrantProfile<"gpu_launch", ["kernel_context"],
    "Authorize launching a kernel bound to the exact nominal GPU context.">;

class MetreonLoopEvidenceProfile<string name, string domain, int arityCount,
                                 list<MetreonGrantProfile> grants> {
  string context_name = name;
  string execution_domain = domain;
  int arity = arityCount;
  list<MetreonGrantProfile> supported_grants = grants;
  bit runtime_materializable = 0;
  bit grants_execution_authority = 0;
  assert !ne(name, ""), "a loop evidence profile must have a name";
  assert !eq(arityCount, 0), "loop evidence contexts cannot declare generic parameters";
  assert !empty(grants), "loop evidence contexts cannot grant execution capabilities";
}

def Metreon_GpuLoopEvidence : MetreonLoopEvidenceProfile<"Gpu::Loop", "Gpu", 0, []>;
def Metreon_HostLoopEvidence : MetreonLoopEvidenceProfile<"Host::Loop", "Host", 0, []>;

def Metreon_GpuThreadExecution : MetreonContextProfile<
    "Gpu::Thread", [],
    [Metreon_GpuGlobalLoad, Metreon_GpuGlobalStore], 0> {
  string execution_scope = "thread";
  list<int> dereferenceable_address_spaces = [1, 3, 4, 5];
  string lowering_domain = "device scalar and SIMT lane execution";
}

def Metreon_GpuWarpStageExecution : MetreonContextProfile<
    "Gpu::WarpStage", ["stage_role"],
    [Metreon_GpuGlobalLoad, Metreon_GpuGlobalStore, Metreon_GpuWorkgroupBarrier,
     Metreon_GpuChannelSend, Metreon_GpuChannelRecv,
     Metreon_GpuAsyncCopy, Metreon_GpuAsyncWait], 0> {
  string execution_scope = "subgroup";
  list<int> dereferenceable_address_spaces = [1, 3, 4, 5];
  string lowering_domain = "warp-specialized workgroup pipeline stage";
}

class MetreonGpuQueueVerificationProfile<string id, int minCap, int maxCap,
    string residencyScope, bit blockingBackpressure,
    bit verifiedCooperativeLaunch> {
  string profile_id = id;
  int min_capacity = minCap;
  int max_capacity = maxCap;
  string residency = residencyScope;
  bit blocking_backpressure = blockingBackpressure;
  bit cooperative_launch_verified = verifiedCooperativeLaunch;
  int pinned_llvm_major = 18;
  int index_bits = 64;
  int max_workgroup_threads = 1024;
  int max_workgroup_warps = 32;
  string conservation_invariant = "free + reserved + queued = capacity";
  string publication_order = "acquire_release_workgroup";
  list<int> host_dereferenceable_spaces = [0];
  list<int> device_dereferenceable_spaces = [1, 3, 4, 5];
  assert !gt(minCap, 0), "GPU bounded queue capacity must be positive";
  assert !ge(maxCap, minCap), "GPU maximum queue capacity must be at least min_capacity";
  assert !or(!ne(residencyScope, "grid"), !not(blockingBackpressure),
             verifiedCooperativeLaunch),
      "blocking grid-scope GPU channel requires a verified cooperative residency contract";
}

def Metreon_WorkgroupPipelineProfile
    : MetreonGpuQueueVerificationProfile<"gpu.workgroup_pipeline.v1", 1, 2,
                                         "workgroup", 1, 0>;

#endif // METREON_GPU_PROFILES_TD
