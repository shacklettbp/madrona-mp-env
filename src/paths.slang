import utils;
import viz_shader_common;

ParameterBlock<GlobalPassData> global;

struct PathsParamBlock {
  StructuredBuffer<float4> vertices;
};

ParameterBlock<PathsParamBlock> pathsParamBlock;

ParameterBlock<float4> perDraw;

struct V2F {
  float alpha;
};

[shader("vertex")]
void pathVert(uint id : SV_VertexID,
              out float4 clip_pos_out : SV_Position,
              out V2F v2f_out)
{
  uint line_idx = id / 6;
  uint line_offset = id % 6;

  uint start_vert_idx = 2 * line_idx;
  uint end_vert_idx = 2 * line_idx + 1;

  float4 start = pathsParamBlock.vertices[start_vert_idx];
  float4 end = pathsParamBlock.vertices[end_vert_idx];

  const float line_width = 4.0;
  clip_pos_out = lineVertexShader(global.view, line_width,
      start.xyz, end.xyz, line_offset);

  v2f_out.alpha = line_offset < 3 ? start.w : end.w;
}

[shader("fragment")]
float4 pathFrag(V2F v2f) : SV_Target0
{
  return float4(perDraw.xyz, v2f.alpha);
}
