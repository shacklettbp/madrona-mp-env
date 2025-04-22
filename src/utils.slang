module utils;

import viz_shader_common;

public float3 cameraPosition(ViewData view_data)
{
  return float3(view_data.camTxfm[0].w,
                view_data.camTxfm[1].w,
                view_data.camTxfm[2].w);
}

public float3 viewDirection(ViewData view_data)
{
  return view_data.camTxfm[2].xyz;
}

public float convertToLinearDepth(ViewData view_data, float proj_depth)
{
  return proj_depth == 0.0 ? 132000.f : view_data.zNear / proj_depth;
}

public float4 worldToClip(ViewData view_data, float3 pos_w)
{
  float3x4 cam = view_data.camTxfm;
  float3x3 w2c = float3x3(cam[0].xyz, cam[1].xyz, cam[2].xyz);
  float3 cam_pos = float3(cam[0].w, cam[1].w, cam[2].w);

  float3 pos_cam_relative = pos_w - cam_pos;

  return float4(
    view_data.screenScale.x * dot(pos_cam_relative, w2c[0]),
    view_data.screenScale.y * dot(pos_cam_relative, w2c[1]),
    view_data.zNear,
    dot(pos_cam_relative, w2c[2]));
}

public float3 clipToWorld(ViewData view_data, float3 clip_pos)
{
  float3x4 cam = view_data.camTxfm;
  float3 cam_pos = float3(cam[0].w, cam[1].w, cam[2].w);

  float2 inv_screen_scale = 1.0 / view_data.screenScale;

  float linear_depth = convertToLinearDepth(view_data, clip_pos.z);

  float2 unproject = float2(inv_screen_scale.x * clip_pos.x * linear_depth, 
                            inv_screen_scale.y * clip_pos.y * linear_depth);

  return cam_pos + 
        cam[0].xyz * unproject.x + 
        cam[1].xyz * unproject.y + 
        cam[2].xyz * linear_depth;
}

public float3 objectPositionToWorld(float3x4 o2w, float3 pos_o)
{
  return float3(
    dot(pos_o, o2w[0].xyz) + o2w[0].w,
    dot(pos_o, o2w[1].xyz) + o2w[1].w,
    dot(pos_o, o2w[2].xyz) + o2w[2].w);
}

public float3 objectNormalToWorld(float3x4 w2o, float3 normal_o)
{
  return w2o[0].xyz * normal_o.x +
         w2o[1].xyz * normal_o.y +
         w2o[2].xyz * normal_o.z;
}

public void cubeVert(uint id, out float3 pos_out, out float3 normal_out)
{
  uint tri = id / 3;
  uint idx = id % 3;
  uint face = tri / 2;
  uint top = tri % 2;

  uint dir = face % 3;
  uint pos = face / 3;

  uint nz = dir >> 1;
  uint ny = dir & 1;
  uint nx = 1 ^ (ny | nz);

  float3 d = {nx, ny, nz};
  float flip = 1.0 - 2.0 * pos;

  float3 n = flip * d;
  float3 u = -d.yzx;
  float3 v = flip * d.zxy;

  float mirror = -1.0 + 2.0 * top;
  pos_out = n +
    mirror * (1.0 - 2 * (idx & 1)) * u +
    mirror * (1.0 - 2 * (idx >> 1)) * v;

  normal_out = n;
}

public interface IVertexFetch {
  float3 fetch(uint idx);
};

public interface IObject2ClipTransform {
  float4 object2Clip(float3 pos_o);
};

// Taken from github.com/m-schuetz/webgpu_wireframe_thicklines
public float4 wireframeVertexShaderTransform<VertexFetchT, TransformT>(
    uint id,
    float line_width,
    uint2 fb_dims,
    VertexFetchT vertex_fetcher,
    TransformT txfm)
  where VertexFetchT : IVertexFetch
  where TransformT : IObject2ClipTransform
{
  uint tri_idx = id / 18;
  uint local_vert_idx = id % 18;
  uint local_line_idx = local_vert_idx / 6;

  uint start_elem_idx = 3 * tri_idx + local_line_idx;
  uint end_elem_idx = 3 * tri_idx + (local_line_idx + 1) % 3;

  float3 start = vertex_fetcher.fetch(start_elem_idx);
  float3 end = vertex_fetcher.fetch(end_elem_idx);

  uint local_idx = id % 6;

  float4 clip_pos;
  float2 screen_pos;
  float2 screen_dir;
  {
    float4 start_clip_pos = txfm.object2Clip(start);
    float4 end_clip_pos = txfm.object2Clip(end);

    float2 start_screen_pos = start_clip_pos.xy / start_clip_pos.w;
    float2 end_screen_pos = end_clip_pos.xy / end_clip_pos.w;

    if (local_idx == 0 || local_idx == 3 || local_idx == 5) {
      clip_pos = start_clip_pos;
      screen_pos = start_screen_pos;
    } else {
      clip_pos = end_clip_pos;
      screen_pos = end_screen_pos;
    }

    screen_dir = normalize(end_screen_pos - start_screen_pos);
  }

  {
    float2 px_offset;

    if (local_idx == 0 || local_idx == 1 || local_idx == 3) {
      px_offset = { screen_dir.y, -screen_dir.x };
    } else {
      px_offset = { -screen_dir.y, screen_dir.x };
    }

    if (local_idx == 0 || local_idx == 3 || local_idx == 5) {
      px_offset -= screen_dir;
    } else {
      px_offset += screen_dir;
    }

    float2 adjusted = screen_pos + line_width * px_offset / float2(fb_dims);
    clip_pos = float4(adjusted * clip_pos.w, clip_pos.zw);
  }

  return clip_pos;
}

public float4 lineVertexShader(ViewData view, float line_width,
   float3 start_pos_w, float3 end_pos_w, uint line_offset)
{
  bool is_start = line_offset % 2 == 0;

  float4 start_clip_pos = worldToClip(view, start_pos_w);
  float4 end_clip_pos = worldToClip(view, end_pos_w);

  float2 start_screen_pos = start_clip_pos.xy / start_clip_pos.w;
  float2 end_screen_pos = end_clip_pos.xy / end_clip_pos.w;

  float2 screen_dir = normalize(end_screen_pos - start_screen_pos);

  float4 clip_pos = is_start ? start_clip_pos : end_clip_pos;
  float2 screen_pos = is_start ? start_screen_pos : end_screen_pos;

  float2 px_offset = (line_offset >= 1 && line_offset <= 3) ?
      float2(screen_dir.y, -screen_dir.x) : float2(-screen_dir.y, screen_dir.x);

  if (is_start) {
    px_offset -= screen_dir;
  } else {
    px_offset += screen_dir;
  }

  const float line_width = 4.0;
  float2 adjusted = screen_pos + line_width * px_offset / view.fbDims;

  return float4(adjusted * clip_pos.w, clip_pos.zw);
}
