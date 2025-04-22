add_subdirectory(gas)

set(SIMULATOR_SRCS
    sim.hpp sim.inl sim.cpp
    level_gen.hpp level_gen.cpp
    sim_flags.hpp sim_flags.inl
    utils.hpp utils.cpp
)

add_library(madrona_mp_env_cpu_impl STATIC
    ${SIMULATOR_SRCS}
    viz.hpp viz.cpp  
    trajectory_db.hpp trajectory_db.cpp
)

target_link_libraries(madrona_mp_env_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_rendering_system
        madrona_navmesh
)

target_compile_definitions(madrona_mp_env_cpu_impl PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

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
    PUBLIC
        madrona_python_utils
    PRIVATE
        madrona_mp_env_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
        madrona_mp_env_importer
        meshoptimizer
)

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

madrona_python_module(madrona_mp_env
    bindings.cpp
)

target_link_libraries(madrona_mp_env PRIVATE
    madrona_mp_env_mgr
)

if (TARGET madrona_viz)
    add_executable(viewer viewer.cpp)
    target_link_libraries(viewer gas_core gas_ui gas_imgui
        madrona_mw_core madrona_mp_env_mgr madrona_mp_env_importer
    )

    target_link_libraries(madrona_mp_env_cpu_impl PUBLIC gas_core gas_ui gas_imgui)

    target_compile_definitions(madrona_mp_env_cpu_impl PRIVATE
        -DMADRONA_MP_ENV_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\"
    )

    target_compile_definitions(viewer PRIVATE
        -DMADRONA_MP_ENV_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\"
    )

    if (TARGET madrona_cuda)
        target_link_libraries(viewer madrona_cuda)
    endif()

    target_compile_definitions(viewer PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
    )
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
