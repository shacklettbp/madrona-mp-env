import utils;

#include "shader_host.slang"

ParameterBlock<GlobalPassData> global;

struct LineParamBlock {
  StructuredBuffer<float> vertices;
  StructuredBuffer<uint> indices;
};

ParameterBlock<LineParamBlock> lineParamBlock;

ParameterBlock<float4> perDraw;

struct VertInput {
  float3 position : Position;
};

[shader("vertex")]
float4 triVert(VertInput v) : SV_Position
{
  float3 pos_w = v.position;

  float4 clip_pos = worldToClip(global.viewData.w2c, pos_w);

  return clip_pos;
}

[shader("fragment")]
float4 triFrag() : SV_Target0
{
  return float4(0, 0, 1, 0.5f);
}

[shader("vertex")]
float4 edgeVert(uint id : SV_VertexID) : SV_Position
{
  uint line_idx = id / 6;
  uint line_offset = id % 6;

  uint start_vert_idx = lineParamBlock.indices[2 * line_idx];
  uint end_vert_idx = lineParamBlock.indices[2 * line_idx + 1];

  float3 start_pos_w = {
    lineParamBlock.vertices[3 * start_vert_idx + 0],
    lineParamBlock.vertices[3 * start_vert_idx + 1],
    lineParamBlock.vertices[3 * start_vert_idx + 2],
  };

  float3 end_pos_w = {
    lineParamBlock.vertices[3 * end_vert_idx + 0],
    lineParamBlock.vertices[3 * end_vert_idx + 1],
    lineParamBlock.vertices[3 * end_vert_idx + 2],
  };

  const float line_width = 4.0;
  return lineVertexShader(line_width, global.viewData.w2c,
      global.viewData.fbDims, start_pos_w, end_pos_w, line_offset);
}

[shader("fragment")]
float4 edgeFrag() : SV_Target0
{
  return perDraw;
}
