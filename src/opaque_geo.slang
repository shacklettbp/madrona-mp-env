#include "viz_shader_common.h"

import utils;

ParameterBlock<GlobalPassData> global;

ParameterBlock<OpaqueGeoPerDraw> perDraw;

struct VertInput {
  float3 pos    : Position;
  float3 normal : NORMAL0;
  float2 uv     : TEXCOORD0;
};

struct V2F {
  float4 pos    : SV_Position;
  float3 normal : NORMAL0;
  float2 uv     : TEXCOORD0;
};

[shader("vertex")]
V2F vertMain(VertInput v)
{
  float3 pos_o = v.pos;
  float3 normal_o = v.normal;

  float3x4 o2w = perDraw.txfm.o2w;
  float3x4 w2o = perDraw.txfm.w2o;

  float3 pos_w = objectPositionToWorld(o2w, pos_o);

  const float z_near = 0.01;

  V2F out;
  out.pos = worldToClip(global.view.w2c, pos_w, 0.01);
  out.normal = objectNormalToWorld(w2o, normal_o);
  out.uv = v.uv;

  return out;
}

[shader("fragment")]
float4 fragMain(V2F v2f) : SV_Target0
{
  float3 view_fwd = global.view.w2c[2].xyz;

  float3 n = normalize(v2f.normal);

  float dot_view = clamp(-dot(view_fwd, n), 0, 1);

  float brightness = lerp(0.25, 1, dot_view);

  return perDraw.baseColor * float4(float3(brightness), 1.f);
}
