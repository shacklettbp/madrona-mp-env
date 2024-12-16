#include "nav_impl.hpp"

#include "gas/gas.hpp"

#include "renderer.hpp"

namespace madGame {

struct NavRenderState {
  ParamBlockType edgeParamBlockType;
  RasterShader triShader;
  RasterShader edgeShader;
  RasterShader edgeOverlayShader;

  ParamBlockType pathsParamBlockType;
  RasterShader pathsShader;
  RasterShader pathsOverlayShader;
};

namespace NavSystemFrontend {

using namespace NavSystem;

NavRenderState * createRenderState(Renderer &renderer)
{
  ParamBlockType edge_pb_type = renderer.gpu->createParamBlockType({
    .uuid = "navmesh_edge_param_block"_to_uuid,
    .buffers = {
      {
        .type = BufferBindingType::Storage,
        .shaderUsage = ShaderStage::Vertex,
      },
      {
        .type = BufferBindingType::Storage,
        .shaderUsage = ShaderStage::Vertex,
      },
    },
  });

  auto shader_bytecode =
      renderer.compileShaderBytecode(BSG_SRC_DIR "/nav/navmesh.slang");

  RasterShader navmesh_shader = renderer.gpu->createRasterShader({
    .byteCode = shader_bytecode,
    .vertexEntry = "triVert",
    .fragmentEntry = "triFrag",
    .rasterPass = renderer.onscreenPassInterface,
    .paramBlockTypes = { renderer.globalPassParamBlockType },
    .vertexBuffers = {{
      .stride = sizeof(Vector3), .attributes = {
        { .offset = 0, .format = VertexFormat::Vec3_F32 },
      },
    }},
    .rasterConfig = {
      .depthCompare = DepthCompare::GreaterOrEqual,
      .depthBias = 5,
      .depthBiasSlope = 1e-2f,
      .depthBiasClamp = 1e-7f,
      .cullMode = CullMode::None,
      .blending = { BlendingConfig::additiveDefault() },
    },
  });

  RasterShader edge_shader = renderer.gpu->createRasterShader({
    .byteCode = shader_bytecode,
    .vertexEntry = "edgeVert",
    .fragmentEntry = "edgeFrag",
    .rasterPass = renderer.onscreenPassInterface,
    .paramBlockTypes = {
      renderer.globalPassParamBlockType,
      edge_pb_type,
    },
    .numPerDrawBytes = sizeof(Vector4),
    .rasterConfig = {
      .depthBias = 100000,
      .depthBiasSlope = 2e-2f,
      .depthBiasClamp = 1e-4f,
      .cullMode = CullMode::None,
    },
  });

  RasterShader edge_overlay_shader = 
      renderer.gpu->createRasterShader({
    .byteCode = shader_bytecode,
    .vertexEntry = "edgeVert",
    .fragmentEntry = "edgeFrag",
    .rasterPass = renderer.onscreenPassInterface,
    .paramBlockTypes = {
      renderer.globalPassParamBlockType,
      edge_pb_type,
    },
    .numPerDrawBytes = sizeof(Vector4),
    .rasterConfig = {
      .depthCompare = DepthCompare::Disabled,
      .writeDepth = false,
      .cullMode = CullMode::None,
      .blending = { BlendingConfig::additiveDefault() },
    },
  });

  renderer.shadercAlloc.release();

  ParamBlockType paths_pb_type = renderer.gpu->createParamBlockType({
    .uuid = "navmesh_paths_param_block"_to_uuid,
    .buffers = {
      {
        .type = BufferBindingType::Storage,
        .shaderUsage = ShaderStage::Vertex,
      },
    },
  });

  shader_bytecode =
      renderer.compileShaderBytecode(BSG_SRC_DIR "/nav/paths.slang");

  RasterShader paths_shader = renderer.gpu->createRasterShader({
    .byteCode = shader_bytecode,
    .vertexEntry = "pathVert",
    .fragmentEntry = "pathFrag",
    .rasterPass = renderer.onscreenPassInterface,
    .paramBlockTypes = {
      renderer.globalPassParamBlockType,
      paths_pb_type,
    },
    .numPerDrawBytes = sizeof(Vector4),
    .rasterConfig = {
      .depthBias = 200000,
      .depthBiasSlope = 2e-2f,
      .depthBiasClamp = 1e-4f,
      .cullMode = CullMode::None,
    },
  });

  RasterShader paths_overlay_shader = 
      renderer.gpu->createRasterShader({
    .byteCode = shader_bytecode,
    .vertexEntry = "pathVert",
    .fragmentEntry = "pathFrag",
    .rasterPass = renderer.onscreenPassInterface,
    .paramBlockTypes = {
      renderer.globalPassParamBlockType,
      paths_pb_type,
    },
    .numPerDrawBytes = sizeof(Vector4),
    .rasterConfig = {
      .depthCompare = DepthCompare::Disabled,
      .writeDepth = false,
      .cullMode = CullMode::None,
      .blending = { BlendingConfig::additiveDefault() },
    },
  });


  renderer.shadercAlloc.release();

  return new NavRenderState {
    .edgeParamBlockType = edge_pb_type,
    .triShader = navmesh_shader,
    .edgeShader = edge_shader,
    .edgeOverlayShader = edge_overlay_shader,
    .pathsParamBlockType = paths_pb_type,
    .pathsShader = paths_shader,
    .pathsOverlayShader = paths_overlay_shader,
  };
}

void destroyRenderState(Renderer &renderer, NavRenderState *render_state)
{
  renderer.gpu->destroyRasterShader(render_state->pathsOverlayShader);
  renderer.gpu->destroyRasterShader(render_state->pathsShader);
  renderer.gpu->destroyParamBlockType(render_state->pathsParamBlockType);

  renderer.gpu->destroyRasterShader(render_state->edgeShader);
  renderer.gpu->destroyRasterShader(render_state->triShader);
  renderer.gpu->destroyParamBlockType(render_state->edgeParamBlockType);

  delete render_state;
}

void debugDraw(Context &ctx,
               NavRenderState *render_state,
               RasterPassEncoder &raster_enc,
               bool render_edge_overlay)
{
  Navmesh &navmesh = ctx.singleton<Navmesh>();

  if (navmesh.numPolys == 0) {
    return;
  }

  i32 total_num_verts = navmesh.numVerts;

  u32 num_tmp_vert_bytes = total_num_verts * sizeof(Vector3);
  MappedTmpBuffer tmp_verts = raster_enc.tmpBuffer(
      num_tmp_vert_bytes, 256 * 3);

  Vector3 *out_verts_ptr = (Vector3 *)tmp_verts.ptr;
  memcpy(out_verts_ptr, navmesh.verts, num_tmp_vert_bytes);

  {
    i32 total_num_tris = navmesh.numTris;
    u32 num_tmp_tri_bytes = total_num_tris * 3 * sizeof(u32);
    MappedTmpBuffer tmp_tri_idxs = raster_enc.tmpBuffer(
        num_tmp_tri_bytes, sizeof(u32));
    u32 *out_tri_idxs_ptr = (u32 *)tmp_tri_idxs.ptr;

    i32 total_num_edges = 0;
    for (i32 poly_idx = 0; poly_idx < navmesh.numPolys; poly_idx++) {
      NavPoly &poly = navmesh.polys[poly_idx];

      for (i32 tri_fan_offset = 1; tri_fan_offset < poly.numVerts - 1;
           tri_fan_offset++) {
        *out_tri_idxs_ptr++ = poly.vertIndices[0];
        *out_tri_idxs_ptr++ = poly.vertIndices[tri_fan_offset];
        *out_tri_idxs_ptr++ = poly.vertIndices[tri_fan_offset + 1];
      }

      for (i32 i = 0; i < poly.numVerts; i++) {
        u16 adjacent = poly.edgeAdjacency[i];
        if (adjacent == NavPoly::noAdjacent || poly_idx < (i32)adjacent) {
          total_num_edges += 1;
        }
      }
    }

    raster_enc.setShader(render_state->triShader);
    raster_enc.setVertexBuffer(0, tmp_verts.buffer);
    raster_enc.setIndexBufferU32(tmp_tri_idxs.buffer);

    raster_enc.drawIndexed(tmp_verts.offset / sizeof(Vector3),
                           tmp_tri_idxs.offset / sizeof(u32),
                           total_num_tris);

    u32 num_tmp_edge_bytes = total_num_edges * 2 * sizeof(u32);
    MappedTmpBuffer tmp_edge_idxs = raster_enc.tmpBuffer(
        num_tmp_edge_bytes, 256);

    u32 *out_edge_idxs_ptr = (u32 *)tmp_edge_idxs.ptr;

    for (i32 poly_idx = 0; poly_idx < navmesh.numPolys; poly_idx++) {
      NavPoly &poly = navmesh.polys[poly_idx];

      for (i32 i = 0; i < poly.numVerts; i++) {
        u16 adjacent = poly.edgeAdjacency[i];
        if (adjacent == NavPoly::noAdjacent || poly_idx < (i32)adjacent) {
          i32 idx_a = poly.vertIndices[i];
          i32 idx_b = poly.vertIndices[((i + 1) % poly.numVerts)];

          *out_edge_idxs_ptr++ = idx_a;
          *out_edge_idxs_ptr++ = idx_b;
        }
      }
    }

    ParamBlock wireframe_tmp_pb = raster_enc.createTemporaryParamBlock({
      .typeID = render_state->edgeParamBlockType,
      .buffers = {
        {
          .buffer = tmp_verts.buffer,
          .offset = tmp_verts.offset,
          .numBytes = num_tmp_vert_bytes,
        },
        {
          .buffer = tmp_edge_idxs.buffer,
          .offset = tmp_edge_idxs.offset,
          .numBytes = num_tmp_edge_bytes,
        },
      },
    });

    raster_enc.setShader(render_state->edgeShader);
    raster_enc.setParamBlock(1, wireframe_tmp_pb);

    raster_enc.drawData(Vector4 { 0, 0, 1, 1 });
    raster_enc.draw(0, total_num_edges * 2);

    if (render_edge_overlay) {
      raster_enc.setShader(render_state->edgeOverlayShader);

      raster_enc.drawData(Vector4 { 0, 0, 1, 0.3 });
      raster_enc.draw(0, total_num_edges * 2);
    }
  }

  {
    StackAlloc tmp_alloc;

    auto paths_query = ctx.query<Position, NavPathingState, NavPathingResult>();

    constexpr u32 poly_crossing_points_per_buffer = 256;
    constexpr u32 path_segment_points_per_buffer = 256;

    struct TmpPolyCrossings {
      MappedTmpBuffer buffer;
      u32 offset;
      u32 *ptr;
      TmpPolyCrossings *next;
    };

    struct TmpPaths {
      MappedTmpBuffer buffer;
      u32 offset;
      Vector3 *ptr;
      TmpPaths *next;
    };

    auto allocTmpPolyCrossings =
      [&]
    ()
    {
      TmpPolyCrossings *new_crossings = tmp_alloc.alloc<TmpPolyCrossings>();

      new_crossings->buffer = raster_enc.tmpBuffer(
        poly_crossing_points_per_buffer * sizeof(u32), 256);
      new_crossings->offset = 0;
      new_crossings->ptr = (u32 *)new_crossings->buffer.ptr;
      new_crossings->next = nullptr;

      return new_crossings;
    };

    auto allocTmpPaths =
      [&]
    ()
    {
      TmpPaths *new_paths = tmp_alloc.alloc<TmpPaths>();
      new_paths->buffer = raster_enc.tmpBuffer(
        path_segment_points_per_buffer * sizeof(Vector3), 256);
      new_paths->offset = 0;
      new_paths->ptr = (Vector3 *)new_paths->buffer.ptr;
      new_paths->next = nullptr;

      return new_paths;
    };

    TmpPolyCrossings *tmp_crossings_head = allocTmpPolyCrossings();
    TmpPolyCrossings *cur_tmp_crossings = tmp_crossings_head;

    TmpPaths *tmp_paths_head = allocTmpPaths();
    TmpPaths *cur_tmp_paths = tmp_paths_head;

    ctx.iterateQuery(paths_query,
      [&]
    (Vector3 pos, NavPathingState &state, NavPathingResult &result)
    {
      Entity path_entity = state.pathEntity;
      if (path_entity == Entity::none()) {
        return;
      }

      NavPolyPath &poly_path =
          ctx.getDirect<NavPolyPath>(2, ctx.loc(path_entity));

      NavPolyPathEdge *poly_crossings = poly_path.edges;

      for (i32 path_idx = 0; path_idx < poly_path.pathLen; path_idx++) {
        NavPolyPathEdge poly_crossing = poly_crossings[path_idx];

        if (cur_tmp_crossings->offset + 2 > poly_crossing_points_per_buffer) {
          auto new_tmp_crossings = allocTmpPolyCrossings();

          cur_tmp_crossings->next = new_tmp_crossings;
          cur_tmp_crossings = new_tmp_crossings;
        }

        cur_tmp_crossings->ptr[cur_tmp_crossings->offset++] =
            poly_crossing.leftIDX;

        cur_tmp_crossings->ptr[cur_tmp_crossings->offset++] =
            poly_crossing.rightIDX;

      }

      Vector3 cur_pos = pos;
      for (i32 i = 0; i < result.numVerts; i++) {
        if (cur_tmp_paths->offset + 2 > path_segment_points_per_buffer) {
          auto new_tmp_paths = allocTmpPaths();
          cur_tmp_paths->next = new_tmp_paths;
          cur_tmp_paths = new_tmp_paths;
        }

        Vector3 next_pos = result.pathVerts[i];

        cur_tmp_paths->ptr[cur_tmp_paths->offset++] = cur_pos;
        cur_tmp_paths->ptr[cur_tmp_paths->offset++] = next_pos;

        cur_pos = next_pos;
      }
    });

    cur_tmp_crossings = tmp_crossings_head;

    while (cur_tmp_crossings != nullptr) {
      if (cur_tmp_crossings->offset == 0) {
        break;
      }

      ParamBlock edge_tmp_pb = raster_enc.createTemporaryParamBlock({
        .typeID = render_state->edgeParamBlockType,
        .buffers = {
          {
            .buffer = tmp_verts.buffer,
            .offset = tmp_verts.offset,
            .numBytes = num_tmp_vert_bytes,
          },
          {
            .buffer = cur_tmp_crossings->buffer.buffer,
            .offset = cur_tmp_crossings->buffer.offset,
            .numBytes = cur_tmp_crossings->offset * (u32)sizeof(u32),
          },
        },
      });

      raster_enc.setShader(render_state->edgeShader);
      raster_enc.setParamBlock(1, edge_tmp_pb);
      raster_enc.drawData(Vector4 {1, 1, 0, 1});
      raster_enc.draw(0, cur_tmp_crossings->offset);

      if (render_edge_overlay) {
        raster_enc.setShader(render_state->edgeOverlayShader);
        raster_enc.drawData(Vector4 {1, 1, 0, 0.3f});
        raster_enc.draw(0, cur_tmp_crossings->offset);
      }

      cur_tmp_crossings = cur_tmp_crossings->next;
    }

    cur_tmp_paths = tmp_paths_head;

    while (cur_tmp_paths != nullptr) {
      if (cur_tmp_paths->offset == 0) {
        break;
      }

      ParamBlock paths_tmp_pb = raster_enc.createTemporaryParamBlock({
        .typeID = render_state->pathsParamBlockType,
        .buffers = {
          {
            .buffer = cur_tmp_paths->buffer.buffer,
            .offset = cur_tmp_paths->buffer.offset,
            .numBytes = cur_tmp_paths->offset * (u32)sizeof(Vector3),
          },
        },
      });

      raster_enc.setShader(render_state->pathsShader);
      raster_enc.setParamBlock(1, paths_tmp_pb);
      raster_enc.drawData(Vector4 {1, 0, 0, 1});
      raster_enc.draw(0, cur_tmp_paths->offset);

      if (render_edge_overlay) {
        raster_enc.setShader(render_state->pathsOverlayShader);
        raster_enc.drawData(Vector4 { 1, 0, 0, 0.3f });
        raster_enc.draw(0, cur_tmp_paths->offset);
      }

      cur_tmp_paths = cur_tmp_paths->next;
    }
  }
}

}

}
