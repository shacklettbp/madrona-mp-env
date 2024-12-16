import utils;

#include "shader_host.slang"

ParameterBlock<GlobalPassData> global;

struct PathsParamBlock {
  StructuredBuffer<float> vertices;
};

ParameterBlock<PathsParamBlock> pathsParamBlock;

ParameterBlock<float4> perDraw;

[shader("vertex")]
float4 pathVert(uint id : SV_VertexID) : SV_Position
{
  uint line_idx = id / 6;
  uint line_offset = id % 6;

  uint start_vert_idx = 2 * line_idx;
  uint end_vert_idx = 2 * line_idx + 1;

  float3 start_pos_w = {
    pathsParamBlock.vertices[3 * start_vert_idx + 0],
    pathsParamBlock.vertices[3 * start_vert_idx + 1],
    pathsParamBlock.vertices[3 * start_vert_idx + 2],
  };

  float3 end_pos_w = {
    pathsParamBlock.vertices[3 * end_vert_idx + 0],
    pathsParamBlock.vertices[3 * end_vert_idx + 1],
    pathsParamBlock.vertices[3 * end_vert_idx + 2],
  };

  const float line_width = 4.0;
  return lineVertexShader(line_width, global.viewData.w2c,
      global.viewData.fbDims, start_pos_w, end_pos_w, line_offset);
}

[shader("fragment")]
float4 pathFrag() : SV_Target0
{
  return perDraw;
}
