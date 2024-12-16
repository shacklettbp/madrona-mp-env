add_library(madrona_mp_env_nav_build STATIC
  nav_build.hpp nav_build.inl nav_build.cpp
  # Recast sources
  Recast.h Recast.cpp
  RecastRasterization.cpp
  RecastArea.cpp
  RecastRegion.cpp
  RecastContour.cpp
  RecastMesh.cpp
)

target_link_libraries(madrona_mp_env_nav_build PUBLIC madrona_mp_env_cpu_impl)

#set(MADRONA_MP_ENV_NAV_SYS_SRCS
#    nav_sys.hpp nav_sys.cpp
#    nav_impl.hpp
#)
#
#add_library(madrona_mp_env_nav_sys STATIC
#    ${MADRONA_MP_ENV_NAV_SYS_SRCS}
#)
#
#target_link_libraries(madrona_mp_env_nav_sys PUBLIC madrona_mp_env_core)

#add_library(madrona_mp_env_nav_sys_frontend STATIC
#  nav_frontend.cpp
#)
#
#target_link_libraries(madrona_mp_env_nav_sys_frontend PUBLIC
#  madrona_mp_env_nav_sys
#)
#
#list(TRANSFORM MADRONA_MP_ENV_NAV_SYS_SRCS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
#set(MADRONA_MP_ENV_NAV_SYS_SRCS "${MADGAME_NAV_SYS_SRCS}" PARENT_SCOPE)
