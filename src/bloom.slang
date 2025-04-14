// bloom.slang

import viz_shader_common;

struct PostParams {
  Texture2D inputTexture;
  SamplerState samplerState;
};

ParameterBlock<GlobalPassData> global;
ParameterBlock<PostParams> params;
ParameterBlock<PostEffectData> consts;

struct VSOutput {
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

static int bloomSize = 8;

float GaussianFactor(float x, float sigma)
{
    sigma = sigma / 3.0;
    return exp(-(x * x) / (2 * sigma * sigma));
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

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    input.uv.y = 1 - input.uv.y;

    float4 color = float4(0.0, 0.0, 0.0, 1.0);

    bool vertical = consts.params1.x > 0.0;
    
    if(vertical)
    {
        // Loop on the vertical across the specified number of steps and apply the gaussian factor.
        [unroll]
        for (int i = -bloomSize; i <= bloomSize; i++) {
            float2 offset = float2(0, i) / (float)consts.params2.y;
            float2 uv = saturate(input.uv + offset);
            float3 sample = params.inputTexture.Sample(params.samplerState, uv).rgb;
            color.rgb += sample * GaussianFactor(float(i), bloomSize);// / (float)bloomSize * 0.5;
        }
    }
    else
    {
        // Loop on the horizontal across the specified number of steps and apply the gaussian factor.
        [unroll]
        for (int i = -bloomSize; i <= bloomSize; i++) {
            float2 offset = float2(i, 0) / (float)consts.params2.x;
            float2 uv = saturate(input.uv + offset);
            float3 sample = params.inputTexture.Sample(params.samplerState, uv).rgb;
            // If the color is high saturation, it will bloom.
            float saturation = max(sample.r, max(sample.g, sample.b)) - min(sample.r, min(sample.g, sample.b));
            float value = max(sample.r, max(sample.g, sample.b));
            color.rgb += sample * saturation * value * GaussianFactor(float(i), bloomSize) / (float)bloomSize * 0.5;
        }
    }

    return color;
}
