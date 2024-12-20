#ifndef MADRONA_MP_ENV_SRC_VIZ_SHADER_COMMON_H_INCLUDED
#define MADRONA_MP_ENV_SRC_VIZ_SHADER_COMMON_H_INCLUDED

struct OpaqueGeoVertex {
  float3 pos;
  float3 normal;
  float2 uv;
};

struct ViewData {
  float3x4 w2c;
  uint2 fbDims;
  uint2 pad;
};

struct NonUniformScaleObjectTransform {
  float3x4 o2w;
  float3x4 w2o;
};

struct GlobalPassData {
  ViewData view;
};

struct OpaqueGeoPerDraw {
  NonUniformScaleObjectTransform txfm;
  float4 baseColor;
};

struct GoalRegionPerDraw {
  NonUniformScaleObjectTransform txfm;
  float4 color;
};

struct AnalyticsTeamHullPerDraw {
  float4 color;
};

struct ShotVizLineData {
  float3 start;
  uint pad;
  float3 end;
  uint pad2;
  float4 color;
};

struct AgentPerDraw {
  NonUniformScaleObjectTransform txfm;
  float4 color;
};

#endif
