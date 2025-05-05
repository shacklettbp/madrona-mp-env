include(brt/cmake/sys-detect.cmake)
include(brt/cmake/cxx-setup.cmake)
add_subdirectory(brt EXCLUDE_FROM_ALL)
target_link_libraries(brt-libcxx INTERFACE madrona_libcxx)
add_subdirectory(gas EXCLUDE_FROM_ALL)

if (EMSCRIPTEN)
  target_compile_definitions(gas_ui PRIVATE
    GAS_USE_SDL=1
  )
endif()

set(SIMULATOR_SRCS
    sim.hpp sim.inl sim.cpp
    level_gen.hpp level_gen.cpp
    sim_flags.hpp sim_flags.inl
    utils.hpp utils.cpp
)
add_library(madrona_mp_env_cpu_impl STATIC ${SIMULATOR_SRCS})

target_include_directories(madrona_mp_env_cpu_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(madrona_mp_env_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_navmesh
)

if (EMSCRIPTEN)
    # Create a custom target to copy data directory to the runtime output directory
    add_custom_target(copy_data_files ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data
        COMMENT "Copying data files to runtime output directory"
    )
    
    # Make madrona_mp_env_cpu_impl dependent on the copy_data_files target
    add_dependencies(madrona_mp_env_cpu_impl copy_data_files)

    target_compile_definitions(madrona_mp_env_cpu_impl PRIVATE
        -DDATA_DIR="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data/"
    )
else()
    target_compile_definitions(madrona_mp_env_cpu_impl PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
    )
endif()


if (TARGET madrona_viz OR EMSCRIPTEN)
    target_sources(madrona_mp_env_cpu_impl PRIVATE
        viz.hpp viz.cpp  
        trajectory_db.hpp trajectory_db.cpp
    )

    gas_add_shaders(
        TARGET
            mpenv_shaders
        SHADER_ENUM
            ShaderID
        SHADER_CLASS
            VizShaders
        CPP_NAMESPACE
            madronaMPEnv
        SHADERS
            PostEffect post_effect.slang
            SSAO ssao.slang
            Downsample downsample.slang
            Bloom bloom.slang
            Agent agent.slang
            TeamHull analytics_team_hulls.slang
            GoalRegions goal_regions.slang
            ShotViz shot_viz.slang
            Objects objects.slang
            Map map.slang
            Paths paths.slang
    )

    target_link_libraries(madrona_mp_env_cpu_impl PRIVATE
        gas_core brt mpenv_shaders
    )
endif()


add_library(madrona_mp_env_importer STATIC
    map_importer.hpp map_importer.cpp
    mesh_bvh_builder.hpp mesh_bvh_builder.cpp
)

target_link_libraries(madrona_mp_env_importer PUBLIC
    madrona_hdrs
    madrona_common
    madrona_importer
    madrona_physics_assets
    madrona_bvh_builder
    madrona_embree
    meshoptimizer
)

add_subdirectory(nav)

find_package(SQLite3)

if (FALSE AND SQLite3_FOUND)
    add_library(madrona_mp_env_db STATIC
      db.hpp db.cpp
    )

    target_link_libraries(madrona_mp_env_db PUBLIC
      madrona_mw_core
      SQLite::SQLite3
    )

    target_compile_definitions(madrona_mp_env_cpu_impl PUBLIC DB_SUPPORT=1)
endif ()

add_library(madrona_mp_env_mgr STATIC
    mgr.hpp mgr.cpp
    dnn.hpp dnn.cpp
)

target_link_libraries(madrona_mp_env_mgr 
    PRIVATE
        madrona_mp_env_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
        madrona_mp_env_importer
        meshoptimizer
)

target_link_libraries(madrona_mp_env_mgr PUBLIC madrona_python_utils)

if (FALSE AND SQLite3_FOUND)
    target_link_libraries(madrona_mp_env_mgr PRIVATE
        madrona_mp_env_db
    )
endif ()

if (TARGET madrona_cuda)
    target_link_libraries(madrona_mp_env_mgr
        PRIVATE madrona_cuda
    )
endif ()

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            madrona_mp_env_gpu_srcs
        SOURCES_DEFN
            GPU_HIDESEEK_SRC_LIST
        FLAGS_DEFN
            GPU_HIDESEEK_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(madrona_mp_env_mgr PRIVATE
        madrona_mp_env_gpu_srcs
        madrona_mw_gpu
    )

    target_compile_definitions(madrona_mp_env_mgr PRIVATE
        -DENABLE_MWGPU=1
    )
endif ()

target_compile_definitions(madrona_mp_env_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

if (NOT EMSCRIPTEN)
    madrona_python_module(madrona_mp_env
        bindings.cpp
    )

    target_link_libraries(madrona_mp_env PRIVATE
        madrona_mp_env_mgr
    )
endif()

if (TARGET madrona_viz OR EMSCRIPTEN)
    add_executable(viewer viewer.cpp)
    target_link_libraries(viewer gas_core gas_ui gas_imgui
        madrona_mw_core madrona_mp_env_mgr madrona_mp_env_importer
    )

    target_link_libraries(madrona_mp_env_cpu_impl PUBLIC gas_core gas_ui gas_imgui)

    target_compile_definitions(madrona_mp_env_cpu_impl PRIVATE
        -DMADRONA_MP_ENV_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\"
        -DMADRONA_MP_ENV_OUT_DIR=\"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/\"
    )

    add_compile_definitions(madrona_mp_env_cpu_impl PRIVATE
        -DMADRONA_MP_ENV_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\"
        -DMADRONA_MP_ENV_OUT_DIR=\"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/\"
    )

    target_compile_definitions(viewer PRIVATE
        -DMADRONA_MP_ENV_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\"
    )

    if (TARGET madrona_cuda)
        target_link_libraries(viewer madrona_cuda)
    endif()

    if (EMSCRIPTEN)
        target_compile_definitions(viewer PRIVATE
            -DDATA_DIR="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data/"
        )
    else()
        target_compile_definitions(viewer PRIVATE
            -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
        )
    endif()

    if (NOT EMSCRIPTEN)
        add_dependencies(viewer gas_shader_compiler gas_shader_compiler_frontend)
    endif()

    if (EMSCRIPTEN)
      target_compile_options(gas_ui PRIVATE
        -sUSE_SDL=3
      )

      target_compile_options(viewer PRIVATE
        -sUSE_SDL=3
      )
      target_link_options(viewer PRIVATE
        -sUSE_SDL=3
        -sINITIAL_MEMORY=2GB
        -sALLOW_MEMORY_GROWTH=0
        -sABORTING_MALLOC=0
        #-sASSERTIONS=1
        --embed-file=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/wgsl
        --embed-file=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data/
      )
    endif()
endif()

add_executable(headless headless.cpp)
target_link_libraries(headless
  madrona_mw_core madrona_mp_env_mgr)

add_executable(convert_assets convert_assets.cpp)
target_link_libraries(convert_assets
  madrona_mw_core madrona_mp_env_mgr
  madrona_mp_env_nav_build
)

if (FALSE AND SQLite3_FOUND)
    add_executable(build_db build_db.cpp)
    target_link_libraries(build_db
      madrona_mw_core madrona_mp_env_mgr
      madrona_mp_env_db
    )
    
    add_executable(build_trajectory_curriculum
      build_trajectory_curriculum.cpp
    )
    target_link_libraries(build_trajectory_curriculum
      madrona_mw_core madrona_mp_env_mgr
      madrona_mp_env_db
    )
endif ()
