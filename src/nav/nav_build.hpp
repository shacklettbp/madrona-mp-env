#pragma once

#include "../types.hpp"

#include "consts.hpp"

#include <madrona/context.hpp>
#include <madrona/taskgraph_builder.hpp>

#include <madrona/importer.hpp>

namespace madronaMPEnv {

struct NavBuildPoly {
    static constexpr inline i32 N = navMaxVertsPerPoly;

    u16 vertIndices[N];
    u16 edgeAdjacency[N];
    i32 numVerts;
};

struct NavmeshVoxelData {
  Vector3 gridOrigin;
  f32 cellSize;
  i32 gridNumCellsX;
  i32 gridNumCellsY;
  i32 gridNumCellsZ;

  static constexpr inline i32 occupancyBitXDim = 32;
  u32 *voxelOccupancy;

  inline bool isOccupied(i32 cell_x, i32 cell_y, i32 cell_z);
  inline void markOccupied(i32 cell_x, i32 cell_y, i32 cell_z);
};

struct NavmeshBuildParams {
  float agentRadius;
  i32 minRegionArea;
  i32 mergeRegionArea;
  float maxSimplificationError;
  i32 maxEdgeLen;
  float maxWalkableSlope = math::toRadians(30.f);
};

struct NavmeshBuildResult {
  Vector3 *verts;
  NavBuildPoly *polys;
  i32 numVerts;
  i32 numPolys;
};

NavmeshBuildResult buildNavmeshFromSourceObjects(
    Span<const imp::SourceObject> src_objs);

NavmeshBuildResult buildNavmeshFromVoxels(NavmeshVoxelData &data);

madrona::Navmesh createMadronaNavmesh(const NavmeshBuildResult &build_result);

}

#include "nav_build.inl"
