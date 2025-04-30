#ifdef VIZ_SHADER_COMMON_HOST_INCLUDE
#define VIS
#else
module viz_shader_common;
#define VIS public
#endif

struct OpaqueGeoVertex {
  float3 pos;
  float3 normal;
  float2 uv;
};

struct MapGeoVertex {
  float3 pos;
};

VIS struct ViewData {
  VIS float3x4 camTxfm; // 3x3 rotation matrix, last column is camera position
  VIS uint2 fbDims;
  VIS float2 screenScale;
  VIS float zNear;
  float pad0;
  float pad1;
  float pad2;
};

VIS struct NonUniformScaleObjectTransform {
  VIS float3x4 o2w;
  VIS float3x4 w2o;
};

VIS struct LightData {
  VIS float3 sunPosition;
  float pad1;
  VIS float3 sunColor;
  float pad2;
};

VIS struct GlobalPassData {
  VIS ViewData view;
  VIS LightData lights;
};

VIS struct PostEffectData {
  VIS float4 params1;
  VIS uint4 params2;
  VIS float4 mapBBMin;
  VIS float4 mapBBMax;
};

VIS struct OpaqueGeoPerDraw {
  VIS NonUniformScaleObjectTransform txfm;
  VIS float4 baseColor;
};

VIS struct MapPerDraw {
  VIS float4 wireframeConfig;
  VIS uint meshVertexOffset;
  VIS uint meshIndexOffset;
  VIS float metallic;
  VIS float roughness;
};

VIS struct GoalRegionPerDraw {
  VIS NonUniformScaleObjectTransform txfm;
  VIS float4 color;
  VIS float lineWidth;
};

VIS struct AnalyticsTeamHullPerDraw {
  VIS float4 color;
};

VIS struct ShotVizLineData {
  VIS float3 start;
  VIS uint pad;
  VIS float3 end;
  VIS uint pad2;
  VIS float4 color;
};

VIS struct AgentPerDraw {
  VIS NonUniformScaleObjectTransform txfm;
  VIS float4 color;
};

#undef VIS
