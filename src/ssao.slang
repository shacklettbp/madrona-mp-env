// post_effect.slang

import viz_shader_common;
import utils;

struct PostParams {
  Texture2D depthTexture;
  SamplerState depthSamplerState;
};

ParameterBlock<GlobalPassData> global;
ParameterBlock<PostParams> params;
ParameterBlock<PostEffectData> consts;

static int poissonDiskSize = 32;
static float2[] poissonDisk = {
float2(0.278, 0.324),
float2(-0.684, -0.305),
float2(-0.816, 0.230),
float2(-0.420, -0.060),
float2(-0.111, 0.820),
float2(-0.392, -0.365),
float2(0.187, -0.074),
float2(-0.650, -0.588),
float2(-0.568, 0.812),
float2(0.055, 0.978),
float2(0.649, -0.536),
float2(-0.558, 0.461),
float2(-0.891, -0.294),
float2(-0.255, -0.655),
float2(0.754, 0.556),
float2(0.269, -0.837),
float2(-0.160, 0.360),
float2(0.045, -0.780),
float2(0.841, -0.117),
float2(0.783, 0.199),
float2(0.351, 0.608),
float2(-0.554, 0.133),
float2(0.367, -0.396),
float2(0.285, 0.911),
float2(-0.034, -0.280),
float2(-0.348, 0.924),
float2(0.452, -0.595),
float2(-0.829, -0.039),
float2(0.084, 0.199),
float2(0.507, -0.809),
float2(0.131, 0.670),
float2(-0.783, 0.534)
};
float2 RotatedDisk(float2 uv, int sample)
{
    // Generate a random angle from the uv.
    float2 sceneResolution = (float2)global.view.fbDims;
    int angleSelect = (((int)(uv.y * sceneResolution.y) % 4) * 4 + ((int)(uv.x * sceneResolution.x) % 4));
    float angle = poissonDisk[angleSelect / 2][angleSelect % 2] * 3.1415;
    float s = sin(angle);
    float c = cos(angle);
    float2x2 rot = float2x2(c, -s, s, c);
    return mul(rot, poissonDisk[sample]);
}

struct VSOutput {
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float GaussianFactor(float x, float y, float sigma)
{
    sigma = sigma / 3.0;
    return exp(-(x * x + y * y) / (2 * sigma * sigma));
}

[shader("vertex")]
VSOutput vertMain(uint id : SV_VertexID)
{
    float2 uv = float2((id << 1) & 2, id & 2);
    VSOutput output;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = uv;
    return output;
}

float SSAO(float2 uv)
{
    float size = 5.0;
    //float power = 1.5;
    //float intensity = 3.0;
    float power = 1.0;
    float intensity = 1.0;
    float ssaoDepth = 50.0;

    float occlusion = 0.0;
    float depth = params.depthTexture.Sample(params.depthSamplerState, uv).x;
    float linearDepth = convertToLinearDepth(global.view, depth);
    for (int i = 0; i < poissonDiskSize; i++) {
        float2 offset = RotatedDisk(uv, i / 2) * (size / linearDepth);
        float sampleDepth = params.depthTexture.Sample(params.depthSamplerState, uv + offset).x;
        float sampleLinearDepth = convertToLinearDepth(global.view, sampleDepth);
        float sampleOcclusion = saturate((linearDepth - sampleLinearDepth) / ssaoDepth);
        occlusion += sampleOcclusion;
    }
    occlusion = pow(occlusion / (float)poissonDiskSize, power) * intensity;
    return saturate(1.0 - occlusion);
}
[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    input.uv.y = 1 - input.uv.y;

    float4 color;
    color.rgb = SSAO(input.uv);
    color.a = 1.0;

    return color;
}
