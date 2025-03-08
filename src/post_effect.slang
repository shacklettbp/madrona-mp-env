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

static float near = 0.01;
static float far = 50000.0;
float LinearDepth(float depth) {
    return near * far / (far + (1.0-depth) * (near - far));
}

float GaussianFactor(float x, float y, float sigma) {
    sigma = sigma / 3.0;
    return exp(-(x * x + y * y) / (2 * sigma * sigma));
}

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
    float distortionAmount = 0.0;//5;
    input.uv = input.uv * 2.0 - 1.0;
    float2 centeredUV = input.uv;
    float distanceFromCenter = length(input.uv);
    input.uv = input.uv * (1.0 - distortionAmount + distortionAmount * distanceFromCenter * distanceFromCenter);
    input.uv = input.uv * 0.5 + 0.5;

    // Main Sample
    float4 color = params.inputTexture.Sample(params.samplerState, input.uv);

    // Chromatic Aberration
    /*float chromaticAberrationAmount = 0.003;
    float2 offset = centeredUV * centeredUV * chromaticAberrationAmount;
    float4 r = params.inputTexture.Sample(
        params.samplerState, input.uv + float2(offset.x, offset.y));
    float4 b = params.inputTexture.Sample(
        params.samplerState, input.uv + float2(-offset.x, -offset.y));
    color.r = r.r;
    color.b = b.b;*/

    // Vignette
    float vignetteAmount = 0.5;
    float brighten = 0.3;
    float vignette = saturate(1.0 - distanceFromCenter * vignetteAmount  + brighten);
    color.rgb *= vignette;

    // Sample depth.
    float depth = params.depthTexture.Sample(params.depthSamplerState, input.uv).x;
    depth = LinearDepth(depth);
    //color.rgb = lerp( color.rgb, float3(1, 1, 1), saturate(depth / 1000) * 0.5);

    // Sampler around the pixel for various blur use cases.
    int sampleCount = 12;
    float kernelSize = 0.02;
    float3 bloomColor = float3(0, 0, 0);
    float3 blurColor = float3(0, 0, 0);
    float blurNorm = 0;
    for(int x = 0; x < sampleCount; x++) {
        for(int y = 0; y < sampleCount; y++) {
            float2 offset = float2(x, y) / (float)sampleCount * kernelSize - kernelSize.xx * 0.5;
            float2 sampleUV = input.uv + offset;
            float depthSample = params.depthTexture.Sample(params.depthSamplerState, sampleUV).x;
            depthSample = LinearDepth(depthSample);
            float3 colorSample = params.inputTexture.Sample(params.samplerState, sampleUV).rgb;

            // If the color is high saturation, it will bloom.
            float saturation = max(colorSample.r, max(colorSample.g, colorSample.b)) - min(colorSample.r, min(colorSample.g, colorSample.b));
            bloomColor += colorSample * saturate(saturation - 0.25) / 64.0 * GaussianFactor(offset.x, offset.y, kernelSize * 0.5);

            // Depth of field
            blurColor += colorSample * GaussianFactor(offset.x, offset.y, kernelSize * 0.5);
            blurNorm += GaussianFactor(offset.x, offset.y, kernelSize * 0.5);
        }
    }

    // Depth of field
    float blurAmount = saturate((depth - 1000) / 250);
    blurAmount += 1 - saturate((depth - 10) / 50);
    color.rgb = lerp(color.rgb, blurColor / blurNorm, blurAmount);

    // Bloom
    float bloomAmount = 10.0;
    color.rgb = color.rgb * 0.6 + bloomColor * bloomAmount;
    for( int i = 0; i < 3; i++ ) {
        color[i] += saturate(color[(i+1)%3] - 1) + saturate(color[(i+2)%3] - 1);
    }

    return color;
}
