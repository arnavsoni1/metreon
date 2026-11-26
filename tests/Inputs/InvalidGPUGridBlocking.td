include "MetreonGPUProfiles.td"

def InvalidGPUGridBlocking
    : MetreonGpuQueueVerificationProfile<"gpu.invalid.grid_blocking", 1, 2,
                                         "grid", 1, 0>;
