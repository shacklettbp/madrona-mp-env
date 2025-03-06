// post_effect.slang

struct PostParams {
  Texture2D inputTexture;
  Texture2D depthTexture;
  SamplerState samplerState;
  SamplerState depthSamplerState;
};

ParameterBlock<PostParams> params;

struct VSOutput {
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertMain(uint id : SV_VertexID) {
    float2 uv = float2((id << 1) & 2, id & 2);
    VSOutput output;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = uv;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    input.uv.y = 1 - input.uv.y;

    // Lens distortion
    float distortionAmount = 0.05;
    input.uv = input.uv * 2.0 - 1.0;
    float2 centeredUV = input.uv;
    float distanceFromCenter = length(input.uv);
    input.uv = input.uv * (1.0 - distortionAmount + distortionAmount * distanceFromCenter * distanceFromCenter);
    input.uv = input.uv * 0.5 + 0.5;

    // Main Sample
    float4 color = params.inputTexture.Sample(params.samplerState, input.uv);

    // Chromatic Aberration
    float chromaticAberrationAmount = 0.003;
    float2 offset = centeredUV * centeredUV * chromaticAberrationAmount;
    float4 r = params.inputTexture.Sample(
        params.samplerState, input.uv + float2(offset.x, offset.y));
    float4 b = params.inputTexture.Sample(
        params.samplerState, input.uv + float2(-offset.x, -offset.y));
    color.r = r.r;
    color.b = b.b;

    // Vignette
    float vignetteAmount = 0.5;
    float brighten = 0.3;
    float vignette = saturate(1.0 - distanceFromCenter * vignetteAmount  + brighten);
    color.rgb *= vignette;

    // Sample depth.
    float depth = params.depthTexture.Sample(params.depthSamplerState, input.uv).x;
    color = sqrt(depth);

    return color;
}
