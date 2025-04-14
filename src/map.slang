import utils;
import viz_shader_common;

ParameterBlock<GlobalPassData> global;

struct MapGeoBindings {
  StructuredBuffer<float> positions;
  StructuredBuffer<uint32_t> indices;
};

ParameterBlock<MapGeoBindings> geoBlock;

ParameterBlock<MapPerDraw> perDraw;

struct V2F {
  float4 clipPos : SV_Position;
  nointerpolation uint primID;
};

[shader("vertex")]
V2F vertMain(uint vert_id : SV_VertexID)
{
  // Map assets are in world space
  float3 pos_w;
  {
    uint base_vert_offset = 3 * vert_id;
    pos_w = float3(geoBlock.positions[base_vert_offset],
                   geoBlock.positions[base_vert_offset + 1],
                   geoBlock.positions[base_vert_offset + 2]);
  }

  V2F v2f;
  v2f.clipPos = worldToClip(global.view, pos_w);
  v2f.primID = vert_id;

  return v2f;
}

float3 getPrimitiveColor(uint mesh_id, uint prim_id)
{
  static const float3 colors[32] =
  {
    float3(1, 0, 0),
    float3(0, 1, 0),
    float3(0, 0, 1),
    float3(1, 1, 0),
    float3(1, 0, 1),
    float3(0, 1, 1),
    float3(1, 0.5, 0),
    float3(0.5, 0, 1),
    float3(0.5, 1, 0),
    float3(0, 0.5, 1),
    float3(1, 0.75, 0.8),
    float3(0.6, 0.3, 0),
    float3(0.5, 0.5, 0),
    float3(0, 0, 0.5),
    float3(0.5, 0, 0),
    float3(1, 0.84, 0),
    float3(1, 0.5, 0.31),
    float3(0.93, 0.51, 0.93),
    float3(0.29, 0, 0.51),
    float3(0.25, 0.88, 0.82),
    float3(0.75, 0.75, 0.75),
    float3(0.9, 0.9, 0.98),
    float3(0.98, 0.5, 0.45),
    float3(0.76, 0.69, 0.57),
    float3(0.63, 0.32, 0.18),
    float3(0.5, 1, 0),
    float3(0.5, 1, 0.83),
    float3(0.85, 0.44, 0.84),
    float3(0.8, 0.52, 0.25),
    float3(0.41, 0.35, 0.8),
    float3(0.87, 0.63, 0.87),
    float3(0.74, 0.98, 0.78)
  };

  uint hash_input = mesh_id * 65536 + prim_id;

  uint hash = (hash_input * 2654435761u) & 31u;

  return colors[hash];
}

[shader("fragment")]
float4 fragMain(V2F v2f) : SV_Target0
{
  float3 view_fwd = global.view.camTxfm[2].xyz;

  float3 barys;
  {
    uint base_index_offset = perDraw.meshIndexOffset + 3 * v2f.primID;

    uint3 indices = uint3(geoBlock.indices[base_index_offset],
                          geoBlock.indices[base_index_offset + 1],
                          geoBlock.indices[base_index_offset + 2]);

    float3 a, b, c;
    {
      uint a_base_offset = 3 * indices.x;
      a = float3(geoBlock.positions[a_base_offset], 
                 geoBlock.positions[a_base_offset + 1], 
                 geoBlock.positions[a_base_offset + 2]);
    }

    {
      uint b_base_offset = 3 * indices.y;
      b = float3(geoBlock.positions[b_base_offset], 
                 geoBlock.positions[b_base_offset + 1], 
                 geoBlock.positions[b_base_offset + 2]);
    }

    {
      uint c_base_offset = 3 * indices.z;
      c = float3(geoBlock.positions[c_base_offset], 
                 geoBlock.positions[c_base_offset + 1], 
                 geoBlock.positions[c_base_offset + 2]);
    }

    
  }

  //float3 n = normalize(v2f.normal);

  //float dot_view = clamp(-dot(view_fwd, n), 0, 1);

  //float brightness = lerp(0.25, 1, dot_view);

  //float4 color = lerp(perDraw.baseColor, float4(1.0, 1.0, 0.5, 1),
  //    pow(clamp(v2f.worldZ / 500, 0.0, 1.0), 0.4));
  //return color * float4(float3(brightness), 1.f);

  float4 color = float4(getPrimitiveColor(perDraw.meshIndexOffset, v2f.primID), 1);
  return color;
}
