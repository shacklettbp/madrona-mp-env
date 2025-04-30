import viz_shader_common;
import utils;

ParameterBlock<GlobalPassData> global;

ParameterBlock<GoalRegionPerDraw> perDraw;

struct V2F {
  float3 normal : Normal0;
};

[shader("vertex")]
float4 vertMain(uint id : SV_VertexID,
                out V2F v2f) : SV_Position
{
  float3 pos_o, normal_o;
  cubeVert(id, pos_o, normal_o);

  pos_o *= 0.5f;

  float3 pos_w = objectPositionToWorld(perDraw.txfm.o2w, pos_o);
  float4 pos_clip = worldToClip(global.view, pos_w);

  v2f.normal = objectNormalToWorld(perDraw.txfm.w2o, normal_o);

  return pos_clip;
}

[shader("fragment")]
float4 fragMain(V2F v2f) : SV_Target0
{
  float3 view_fwd = global.view.camTxfm[2].xyz;
  float3 n = normalize(v2f.normal);
  float dot_view = clamp(-dot(view_fwd, n), 0, 1);

  float brightness = lerp(0.25, 1, dot_view);

  return float4(perDraw.color) * float4(float3(brightness), 1.f);
}

struct VertexFetch : IVertexFetch {
  float3 fetch(uint idx)
  {
    float3 pos, normal;
    cubeVert(idx, pos, normal);
    return 0.5f * pos;
  }
};

struct O2CTransform : IObject2ClipTransform {
  ViewData viewData;
  NonUniformScaleObjectTransform txfmData;

  float4 object2Clip(float3 pos_o)
  {
    float3 pos_w = objectPositionToWorld(txfmData.o2w, pos_o);
    float4 pos_clip = worldToClip(viewData, pos_w);
    return pos_clip;
  }
};

[shader("vertex")]
float4 vertMainWireframe(uint id : SV_VertexID) : SV_Position
{
  VertexFetch fetcher;
  O2CTransform txfm;
  txfm.viewData = global.view;
  txfm.txfmData = perDraw.txfm;

  // Skip internal triangle edges in the cubes
  uint base_idx = id / 6;
  uint offset = id % 6;
  uint idx = base_idx * 18 + offset;

  return wireframeVertexShaderTransform<VertexFetch, O2CTransform>(
    idx, perDraw.lineWidth, global.view.fbDims, fetcher, txfm);
}

[shader("fragment")]
float4 fragMainWireframe(V2F v2f) : SV_Target0
{
  return float4(perDraw.color.rgb, 1.0);
}

[shader("fragment")]
float4 fragMainWireframeNoDepth(V2F v2f) : SV_Target0
{
  return float4(perDraw.color.rgb, 0.2);
}
