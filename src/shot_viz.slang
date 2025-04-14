import utils;
import viz_shader_common;

ParameterBlock<GlobalPassData> global;

struct ShotVizGeometry {
  StructuredBuffer<ShotVizLineData> lineData;
};

ParameterBlock<ShotVizGeometry> geo;

struct V2F {
  float4 color : Color0;
};

[shader("vertex")]
float4 vertMain(uint id : SV_VertexID, out V2F v2f) : SV_Position
{
  uint line_idx = id / 6;
  uint local_idx = id % 6;

  bool is_start = id % 2 == 0;

  ShotVizLineData line_data = geo.lineData[line_idx];

  float3 start_pos_w = line_data.start;
  float3 end_pos_w = line_data.end;

  float4 start_clip_pos = worldToClip(global.view, start_pos_w);
  float4 end_clip_pos = worldToClip(global.view, end_pos_w);

  float2 start_screen_pos = start_clip_pos.xy / start_clip_pos.w;
  float2 end_screen_pos = end_clip_pos.xy / end_clip_pos.w;

  float2 screen_dir = normalize(end_screen_pos - start_screen_pos);

  float4 clip_pos = is_start ? start_clip_pos : end_clip_pos;
  float2 screen_pos = is_start ? start_screen_pos : end_screen_pos;

  float2 px_offset = (local_idx >= 1 && local_idx <= 3) ?
    float2(screen_dir.y, -screen_dir.x) : float2(-screen_dir.y, screen_dir.x);

  if (is_start) {
    px_offset -= screen_dir;
  } else {
    px_offset += screen_dir;
  }

  const float line_width = 4.0;
  float2 adjusted = screen_pos +
      line_width * px_offset / float2(global.view.fbDims);

  v2f.color = line_data.color;

  return float4(adjusted * clip_pos.w, clip_pos.zw);
}

[shader("fragment")]
float4 fragMain(V2F v2f) : SV_Target0
{
  return v2f.color;
}
