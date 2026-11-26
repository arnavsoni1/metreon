include "MetreonGPUProfiles.td"

def InvalidGPUQueueCapacity
    : MetreonGpuQueueVerificationProfile<"gpu.invalid.zero_capacity", 0, 2,
                                         "workgroup", 1, 0>;
