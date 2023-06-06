add_subdirectory(bm_lib)

set(SOURCE_FILES cuda_benchmark.cc)

list(APPEND SOURCE_FILES baseline/baseline.cc baseline/baseline.cu
     out_sync/out_sync.cc out_sync/out_sync.cu)

cuda_add_executable(cuda_benchmark ${SOURCE_FILES})
target_link_libraries(cuda_benchmark bm_lib)

install(TARGETS cuda_benchmark RUNTIME)
