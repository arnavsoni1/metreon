file(READ "${SCHEMA_DIR}/MetreonGPUProfiles.json" gpu_profiles)

function(expect_field variable record field expected)
    string(JSON actual GET "${${variable}}" "${record}" "${field}")
    if(NOT "${actual}" STREQUAL "${expected}")
        message(FATAL_ERROR "${record}.${field}: expected ${expected}, got ${actual}")
    endif()
endfunction()

expect_field(gpu_profiles Metreon_WorkgroupPipelineProfile profile_id
    "gpu.workgroup_pipeline.v1")
expect_field(gpu_profiles Metreon_WorkgroupPipelineProfile pinned_llvm_major 18)
expect_field(gpu_profiles Metreon_WorkgroupPipelineProfile index_bits 64)
expect_field(gpu_profiles Metreon_WorkgroupPipelineProfile min_capacity 1)
expect_field(gpu_profiles Metreon_WorkgroupPipelineProfile max_capacity 2)
expect_field(gpu_profiles Metreon_WorkgroupPipelineProfile residency "workgroup")
expect_field(gpu_profiles Metreon_WorkgroupPipelineProfile max_workgroup_threads 1024)
expect_field(gpu_profiles Metreon_WorkgroupPipelineProfile max_workgroup_warps 32)
expect_field(gpu_profiles Metreon_WorkgroupPipelineProfile conservation_invariant
    "free + reserved + queued = capacity")
expect_field(gpu_profiles Metreon_GpuLoopEvidence context_name "Gpu::Loop")
expect_field(gpu_profiles Metreon_GpuLoopEvidence runtime_materializable 0)
expect_field(gpu_profiles Metreon_GpuLoopEvidence grants_execution_authority 0)
expect_field(gpu_profiles Metreon_GpuThreadExecution context_name "Gpu::Thread")
expect_field(gpu_profiles Metreon_GpuThreadExecution runtime_materializable 0)
expect_field(gpu_profiles Metreon_GpuThreadExecution seq_eligible 0)
expect_field(gpu_profiles Metreon_GpuWarpStageExecution context_name "Gpu::WarpStage")
expect_field(gpu_profiles Metreon_GpuWarpStageExecution arity 1)
expect_field(gpu_profiles Metreon_HostGpuLaunch arity 1)

function(reject_schema input expected)
    execute_process(
        COMMAND "${TABLEGEN}" -I "${DIALECT_DIR}" -dump-json
            "${INPUT_DIR}/${input}.td" -o "${SCHEMA_DIR}/${input}.json"
        RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if(result EQUAL 0 OR NOT stderr MATCHES "${expected}")
        message(FATAL_ERROR "invalid GPU schema ${input}: ${result} ${stdout} ${stderr}")
    endif()
endfunction()

reject_schema(InvalidGPUQueueCapacity "GPU bounded queue capacity must be positive")
reject_schema(InvalidGPUGridBlocking
    "blocking grid-scope GPU channel requires a verified cooperative residency contract")
reject_schema(InvalidGPULoopGrant
    "loop evidence contexts cannot grant execution capabilities")
