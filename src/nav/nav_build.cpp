#include "nav_build.hpp"

#include "Recast.h"

#include "consts.hpp"
#include "../consts.hpp"

namespace madronaMPEnv {

namespace {

struct NavmeshBuildFromVoxelsJobEntity : madrona::Archetype<NavmeshVoxelData> {};

struct NavmeshFinalizeInput {
  Vector3 *verts;
  NavBuildPoly *polys;
  i32 numVerts;
  i32 numPolys;
};

}

static inline void checkRc(bool res,
                           const char *file,
                           int line,
                           const char *funcname)
{
  if (!res) [[unlikely]] {
    FATAL("Navmesh build failed at %s:%d in %s", file, line, funcname);
  }
}

#define REQ_RC(expr)                             \
  ::madronaMPEnv::checkRc((expr), __FILE__, __LINE__, \
                     MADRONA_COMPILER_FUNCTION_NAME)

static NavmeshBuildResult buildFromHeightfield(
    rcContext &rc_ctx,
    const NavmeshBuildParams &build_params,
    rcCompactHeightfield &compact_heightfield)
{
  // Refine heightfield and build regions
  REQ_RC(rcErodeWalkableArea(rc_ctx,
      ceilf(build_params.agentRadius / compact_heightfield.cs),
      compact_heightfield));
  REQ_RC(rcBuildDistanceField(rc_ctx, compact_heightfield));
  REQ_RC(rcBuildRegions(rc_ctx, compact_heightfield, 0,
                        build_params.minRegionArea, 
                        build_params.mergeRegionArea));

  rcContourSet contour_set;
  REQ_RC(rcBuildContours(rc_ctx, compact_heightfield,
                         build_params.maxSimplificationError, build_params.maxEdgeLen,
                         contour_set));

  rcPolyMesh rc_navmesh;
  REQ_RC(rcBuildPolyMesh(rc_ctx, contour_set, navMaxVertsPerPoly, rc_navmesh));

  i32 num_verts = rc_navmesh.nverts;
  Vector3 *out_verts = (Vector3 *)rc_ctx.tmpAlloc(
    sizeof(Vector3) * (size_t)num_verts);

  i32 num_polys = rc_navmesh.npolys;
  NavBuildPoly *out_polys = (NavBuildPoly *)rc_ctx.tmpAlloc(
    sizeof(NavBuildPoly) * (size_t)num_polys);

  for (i32 vert_idx = 0; vert_idx < num_verts; vert_idx++) {
    i32 vert_base = 3 * vert_idx;

    u16 src_x = rc_navmesh.verts[vert_base + 0];
    u16 src_y = rc_navmesh.verts[vert_base + 1];
    u16 src_z = rc_navmesh.verts[vert_base + 2];

    f32 x = (f32)src_x * compact_heightfield.cs +
        compact_heightfield.bmin.x;
    f32 y = (f32)src_y * compact_heightfield.cs +
        compact_heightfield.bmin.y;
    f32 z = (f32)src_z * compact_heightfield.ch +
        compact_heightfield.bmin.z;

    out_verts[vert_idx] = { x, y, z };
  }

  for (i32 poly_idx = 0; poly_idx < num_polys; poly_idx++) {
    NavBuildPoly out_poly;
    out_poly.numVerts = 0;

    unsigned short *poly_indices =
      rc_navmesh.polys + poly_idx * 2 * navMaxVertsPerPoly;
    unsigned short *poly_neighbors = poly_indices + navMaxVertsPerPoly;

    i32 vert_idx;
    for (vert_idx = 0; vert_idx < navMaxVertsPerPoly; vert_idx++) {
      u16 idx = (u16)poly_indices[vert_idx];
      if (idx == 0xFFFF) {
        break;
      }

      out_poly.vertIndices[vert_idx] = idx;
      out_poly.edgeAdjacency[vert_idx] = (u16)poly_neighbors[vert_idx];

      out_poly.numVerts += 1;
    }

    assert(out_poly.numVerts >= 3);

    out_polys[poly_idx] = out_poly;
  }

  rcDestroyPolyMesh(rc_navmesh);
  rcDestroyContourSet(contour_set);

  return NavmeshBuildResult {
    .verts = out_verts,
    .polys = out_polys,
    .numVerts = num_verts,
    .numPolys = num_polys,
  };
}

static NavmeshBuildResult buildFromSourceObjects(
    rcContext &rc_ctx,
    const NavmeshBuildParams &build_params,
    Span<const imp::SourceObject> src_objs)
{
  AABB bounds = AABB::invalid();

  for (auto &src_obj : src_objs) {
    for (auto &src_mesh : src_obj.meshes) {
      assert(src_mesh.faceCounts == nullptr); // Must be triangle mesh

      for (uint32_t i = 0; i < src_mesh.numVertices; i++) {
        Vector3 pos = src_mesh.positions[i];
        bounds.expand(pos);
      }
    }
  }

  float cell_size = consts::agentRadius / 4.f;
  float cell_height = consts::proneHeight;

  rcHeightfield heightfield;
  REQ_RC(rcInitHeightfield(rc_ctx, heightfield, 
      ceilf((bounds.pMax.x - bounds.pMin.x) / cell_size),
      ceilf((bounds.pMax.y - bounds.pMin.y) / cell_size),
      bounds.pMin,
      bounds.pMax,
      cell_size,
      cell_height));

  for (auto &src_obj : src_objs) {
    for (auto &src_mesh : src_obj.meshes) {
      u8 *area_ids =
          (u8 *)rc_ctx.tmpAlloc(sizeof(u8) * src_mesh.numFaces);

      for (uint32_t i = 0; i < src_mesh.numFaces; i++) {
        area_ids[i] = RC_NULL_AREA;
      }

      rcMarkWalkableTriangles(rc_ctx, 0.25f * math::pi,
          src_mesh.positions,
          src_mesh.numVertices, src_mesh.indices,
          src_mesh.numFaces, area_ids);

      REQ_RC(rcRasterizeTriangles(rc_ctx, src_mesh.positions,
          src_mesh.numVertices, src_mesh.indices, area_ids,
          src_mesh.numFaces, heightfield));
    }
  }

  rcCompactHeightfield compact_heightfield;
  REQ_RC(rcBuildCompactHeightfield(
      rc_ctx, consts::standHeight, 1, heightfield, compact_heightfield));

  // Finish initializing heightfield
  REQ_RC(rcConnectCompactHeightfieldNeighbors(rc_ctx, compact_heightfield));

  return buildFromHeightfield(rc_ctx, build_params, compact_heightfield);
}

// This is really naive, only will work if gridNumCellsZ = 1
static NavmeshBuildResult buildFromVoxels(
    rcContext &rc_ctx,
    const NavmeshBuildParams &build_params,
    NavmeshVoxelData &voxel_data)
{
  assert(voxel_data.gridNumCellsZ == 1);

  Vector3 bmin = voxel_data.gridOrigin;
  Vector3 bmax = bmin;
  bmax.x += voxel_data.cellSize * voxel_data.gridNumCellsX;
  bmax.y += voxel_data.cellSize * voxel_data.gridNumCellsY;
  bmax.z += voxel_data.cellSize * voxel_data.gridNumCellsZ;

  rcCompactHeightfield compact_heightfield;
  compact_heightfield.numCellsX = voxel_data.gridNumCellsX;
  compact_heightfield.numCellsY = voxel_data.gridNumCellsY;

  const int walkable_height = 3;
  const int walkable_climb = 0;

  int num_columns = voxel_data.gridNumCellsX * voxel_data.gridNumCellsY;
  compact_heightfield.spanCount = num_columns;
  compact_heightfield.walkableHeight = walkable_height;
  compact_heightfield.walkableClimb = walkable_climb;
  compact_heightfield.borderSize = 0;
  compact_heightfield.maxDistance = 0;
  compact_heightfield.maxRegions = 0;
  compact_heightfield.bmin = bmin;
  compact_heightfield.bmax = bmax;
  compact_heightfield.bmax.z +=
      compact_heightfield.walkableHeight * voxel_data.cellSize;
  compact_heightfield.cs = voxel_data.cellSize;
  compact_heightfield.ch = voxel_data.cellSize;

  compact_heightfield.cells =
      (rcCompactCell *)rc_ctx.tmpAlloc(sizeof(rcCompactCell) * num_columns);
  memset(compact_heightfield.cells, 0, sizeof(rcCompactCell) * num_columns);

  compact_heightfield.spans =
      (rcCompactSpan *)rc_ctx.tmpAlloc(sizeof(rcCompactSpan) * num_columns);
  memset(compact_heightfield.spans, 0, sizeof(rcCompactSpan) * num_columns);

  compact_heightfield.dist = nullptr;

  compact_heightfield.areas = (u8 *)rc_ctx.tmpAlloc(sizeof(u8) * num_columns);
  memset(compact_heightfield.areas, RC_NULL_AREA, sizeof(u8) * num_columns);

  for (i32 y = 0; y < voxel_data.gridNumCellsY; y++) {
    for (i32 x = 0; x < voxel_data.gridNumCellsX; x++) {
      i32 col_idx = y * voxel_data.gridNumCellsX + x;

      rcCompactCell &cell = compact_heightfield.cells[col_idx];
      cell.index = col_idx;
      cell.count = 1;

      bool is_occupied = voxel_data.isOccupied(x, y, 0);

      if (is_occupied) {
        compact_heightfield.spans[col_idx].z = 0;
        compact_heightfield.spans[col_idx].h = 0xFF;
        compact_heightfield.areas[col_idx] = RC_NULL_AREA;
      } else {
        compact_heightfield.spans[col_idx].z = 0;
        compact_heightfield.spans[col_idx].h = 0xFF;
        compact_heightfield.areas[col_idx] = RC_WALKABLE_AREA;
      }
    }
  }

  // Finish initializing heightfield
  REQ_RC(rcConnectCompactHeightfieldNeighbors(rc_ctx, compact_heightfield));

  return buildFromHeightfield(rc_ctx, build_params, compact_heightfield);
}

static NavmeshBuildParams defaultBuildParams()
{
  return NavmeshBuildParams {
      .agentRadius = consts::agentRadius + 1.f,
      .minRegionArea = 10,
      .mergeRegionArea = 10,
      .maxSimplificationError = 1.f,
      .maxEdgeLen = 100,
  };
}

NavmeshBuildResult buildNavmeshFromSourceObjects(
    Span<const imp::SourceObject> src_objs)
{
  StackAlloc tmp_alloc;
  rcContext rc_ctx(tmp_alloc);

  NavmeshBuildParams build_params = defaultBuildParams();

  return buildFromSourceObjects(rc_ctx, build_params, src_objs);
}

NavmeshBuildResult buildNavmeshFromVoxels(NavmeshVoxelData &input)
{
  StackAlloc tmp_alloc;
  rcContext rc_ctx(tmp_alloc);

  NavmeshBuildParams build_params = defaultBuildParams();

  return buildFromVoxels(rc_ctx, build_params, input);
}

madrona::Navmesh createMadronaNavmesh(const NavmeshBuildResult &build_result)
{
  HeapArray<uint32_t> poly_idx_offsets(build_result.numPolys);
  HeapArray<uint32_t> poly_sizes(build_result.numPolys);
  HeapArray<uint32_t> poly_indices(build_result.numPolys * NavBuildPoly::N);

  i32 total_num_indices = 0;
  for (i32 i = 0; i < build_result.numPolys; i++) {
    NavBuildPoly build_poly = build_result.polys[i];

    poly_idx_offsets[i] = total_num_indices;
    poly_sizes[i] = build_poly.numVerts;

    for (i32 j = 0; j < build_poly.numVerts; j++) {
      poly_indices[total_num_indices++] = build_poly.vertIndices[j];
    }

  }

  Navmesh navmesh = Navmesh::initFromPolygons(
      build_result.verts,
      poly_indices.data(),
      poly_idx_offsets.data(),
      poly_sizes.data(),
      build_result.numVerts,
      build_result.numPolys);

  return navmesh;
}

}
