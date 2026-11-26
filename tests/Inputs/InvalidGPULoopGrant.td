include "MetreonGPUProfiles.td"

def InvalidGPULoopGrant
    : MetreonLoopEvidenceProfile<"Gpu::Loop", "Gpu", 0, [Metreon_GpuGlobalLoad]>;
