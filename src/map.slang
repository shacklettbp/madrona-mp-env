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
  nointerpolation uint primID;
};

struct BarycentricDeriv
{
  float3 lambda;
  float3 d_dx;
  float3 d_dy;
};

static const float PI = 3.14159265359;

// http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
BarycentricDeriv calcFullBary(float4 pt0, float4 pt1, float4 pt2,
                              float2 pixel_ndc)
{
  BarycentricDeriv ret = (BarycentricDeriv)0;

  float3 invW = rcp(float3(pt0.w, pt1.w, pt2.w));

  float2 ndc0 = pt0.xy * invW.x;
  float2 ndc1 = pt1.xy * invW.y;
  float2 ndc2 = pt2.xy * invW.z;

  float invDet = rcp(determinant(float2x2(ndc2 - ndc1, ndc0 - ndc1)));
  ret.d_dx = float3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * invDet * invW;
  ret.d_dy = float3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * invDet * invW;
  float ddxSum = dot(ret.d_dx, float3(1,1,1));
  float ddySum = dot(ret.d_dy, float3(1,1,1));

  float2 deltaVec = pixel_ndc - ndc0;
  float interpInvW = invW.x + deltaVec.x*ddxSum + deltaVec.y*ddySum;
  float interpW = rcp(interpInvW);

  ret.lambda.x = interpW * (invW[0] + deltaVec.x*ret.d_dx.x + deltaVec.y*ret.d_dy.x);
  ret.lambda.y = interpW * (0.0f    + deltaVec.x*ret.d_dx.y + deltaVec.y*ret.d_dy.y);
  ret.lambda.z = interpW * (0.0f    + deltaVec.x*ret.d_dx.z + deltaVec.y*ret.d_dy.z);

  ret.d_dx *= (2.0f/global.view.fbDims.x);
  ret.d_dy *= (2.0f/global.view.fbDims.y);
  ddxSum   *= (2.0f/global.view.fbDims.x);
  ddySum   *= (2.0f/global.view.fbDims.y);

  ret.d_dy *= -1.0f;
  ddySum   *= -1.0f;

  float interpW_ddx = 1.0f / (interpInvW + ddxSum);
  float interpW_ddy = 1.0f / (interpInvW + ddySum);

  ret.d_dx = interpW_ddx*(ret.lambda * interpInvW + ret.d_dx) - ret.lambda;
  ret.d_dy = interpW_ddy*(ret.lambda * interpInvW + ret.d_dy) - ret.lambda;  

  return ret;
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

  uint hash_input = (mesh_id << 16) + prim_id;

  uint hash = (hash_input * 2654435761u) & 31u;

  return colors[hash];
}

float edgeFactor(BarycentricDeriv barys) {
  float3 d = abs(barys.d_dx) + abs(barys.d_dy);
  float3 f = step(d * perDraw.wireframeConfig.w, barys.lambda);
  return min(min(f.x, f.y), f.z);
}

float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0); // saturate
    float NdotH2 = NdotH*NdotH;

    float num    = a2;
    float denom  = (NdotH2 * (a2 - 1.0) + 1.0);
    denom        = PI * denom * denom;

    return num / max(denom, 0.0000001); // avoid divide by zero
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / max(denom, 0.0000001); // avoid divide by zero
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

[shader("vertex")]
void vertMain(uint vert_id : SV_VertexID,
              out float4 clip_pos_out : SV_Position,
              out V2F v2f_out)
{
  // Map assets are in world space
  float3 pos_w;
  {
    uint base_vert_offset = 3 * (perDraw.meshVertexOffset + vert_id);
    pos_w = float3(geoBlock.positions[base_vert_offset],
                   geoBlock.positions[base_vert_offset + 1],
                   geoBlock.positions[base_vert_offset + 2]);
  }

  clip_pos_out = worldToClip(global.view, pos_w);
  v2f_out.primID = vert_id;
}

[shader("fragment")]
float4 fragMain(float4 window_pos : SV_Position, V2F v2f) : SV_Target0
{
  float3 view_fwd = viewDirection(global.view);

  BarycentricDeriv barys;
  float3 pos_w;
  float3 geo_normal;
  {
    uint base_index_offset = perDraw.meshIndexOffset + 3 * v2f.primID;

    uint3 indices = uint3(geoBlock.indices[base_index_offset],
                          geoBlock.indices[base_index_offset + 1],
                          geoBlock.indices[base_index_offset + 2]);

    uint3 base_vert_offsets = 3 * (indices + perDraw.meshVertexOffset);

    float3 a = float3(geoBlock.positions[base_vert_offsets.x], 
                      geoBlock.positions[base_vert_offsets.x + 1], 
                      geoBlock.positions[base_vert_offsets.x + 2]);

    float3 b = float3(geoBlock.positions[base_vert_offsets.y], 
                      geoBlock.positions[base_vert_offsets.y + 1], 
                      geoBlock.positions[base_vert_offsets.y + 2]);

    float3 c = float3(geoBlock.positions[base_vert_offsets.z], 
                      geoBlock.positions[base_vert_offsets.z + 1], 
                      geoBlock.positions[base_vert_offsets.z + 2]);

    geo_normal = normalize(cross(b - a, c - a));

    float4 a_clip = worldToClip(global.view, a);
    float4 b_clip = worldToClip(global.view, b);
    float4 c_clip = worldToClip(global.view, c);

    float2 pixel_ndc = float2(
      2.0 * (window_pos.x / global.view.fbDims.x) - 1,
      -2.0 * (window_pos.y / global.view.fbDims.y) + 1);

    barys = calcFullBary(a_clip, b_clip, c_clip, pixel_ndc);

    pos_w = a * barys.lambda.x + b * barys.lambda.y + c * barys.lambda.z;
  }

  float3 albedo = float3(1, 1, 1);

  float3 to_sun = global.lights.sunPosition - pos_w;
  float3 to_cam = cameraPosition(global.view) - pos_w;
  float3 sun_color = global.lights.sunColor;

  float3 V = normalize(to_cam);
  float3 N = dot(geo_normal, V) > 0 ? geo_normal : -geo_normal;
  float3 L = normalize(to_sun);

  float metallic = perDraw.metallic;
  float roughness = perDraw.roughness;

  float3 H = normalize(V + L);

  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  float NdotH = max(dot(N, H), 0.0);
  float VdotH = max(dot(V, H), 0.0);

  float3 F0 = float3(0.04, 0.04, 0.04); // Base reflectance for dielectrics
  F0 = lerp(F0, albedo, metallic);      // Use albedo for metallic surfaces

  float3 F = fresnelSchlick(VdotH, F0); // Use VdotH for Cook-Torrance
  float D = DistributionGGX(N, H, roughness);
  float G = GeometrySmith(N, V, L, roughness);

  float3 numerator = D * G * F;
  float denominator = 4.0 * NdotV * NdotL + 0.001;

  float3 specular = numerator / denominator;


  float3 kS = F;
  float3 kD = float3(1.0, 1.0, 1.0) - kS;
  kD *= (1.0 - metallic);


  float3 radiance = sun_color * NdotL;

  float3 Lo = (kD * albedo / PI + specular) * radiance;

  float3 ambient = float3(0.05) * albedo * (1.0 - metallic);

  float3 emissive = perDraw.wireframeConfig.xyz * (1.0 - edgeFactor(barys));
  emissive /= max(1.0, (length(to_cam) * 0.001f));

  float3 color = ambient + Lo + emissive;

  return float4(color, 1.f);
  // return float4(getPrimitiveColor(perDraw.meshIndexOffset, v2f.primID), 1);
}
