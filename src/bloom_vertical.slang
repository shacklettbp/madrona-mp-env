// bloom_vertical.slang

struct PostParams {
  Texture2D inputTexture;
  SamplerState samplerState;
};

ParameterBlock<PostParams> params;

// We will pass this in eventually.
// Half on X.
//static int2 sceneResolution = int2(1920, 1080);
//static int2 sceneResolution = int2( 2730, 1536);
static int2 sceneResolution = int2( 682, 384);

struct VSOutput {
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

static int bloomSize = 32;

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

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    input.uv.y = 1 - input.uv.y;

    float4 color = float4(0.0, 0.0, 0.0, 1.0);
    
    // Loop on the horizontal across the specified number of steps and apply the gaussian factor.
    [unroll]
    for (int i = -bloomSize; i <= bloomSize; i++) {
        float2 offset = float2(0, i) / sceneResolution.y;
        float2 uv = saturate(input.uv + offset);
        float3 sample = params.inputTexture.Sample(params.samplerState, uv).rgb;
        color.rgb += sample * GaussianFactor(float(i), 0.0, 10.0);// / (float)bloomSize * 0.5;
    }

    return color;
}
