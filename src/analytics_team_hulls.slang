import viz_shader_common;
import utils;

ParameterBlock<GlobalPassData> global;

ParameterBlock<AnalyticsTeamHullPerDraw> perDraw;

struct VertInput {
  float3 position : Position;
};

[shader("vertex")]
float4 vertMain(VertInput v) : SV_Position
{
  float3 pos_w = v.position;
  float4 clip_pos = worldToClip(global.view, pos_w);

  return clip_pos;
}

[shader("fragment")]
float4 triFrag() : SV_Target0
{
  return perDraw.color;
}
