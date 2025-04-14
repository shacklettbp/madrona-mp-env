// downsample.slang

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

    int2 ratio = (int2)consts.params1.xy;

    for( int x = 0; x < ratio.x; x++ )
    {
        for( int y = 0; y < ratio.y; y++ )
        {
            float2 offset = float2(x, y) / (float2)consts.params2.xy / (float2)ratio;
            float2 uv = saturate(input.uv + offset);
            float3 sample = params.inputTexture.Sample(params.samplerState, uv).rgb;
            color.rgb += sample / (float)(ratio.x * ratio.y);
        }
    }

    return color;
}
