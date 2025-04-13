// post_effect.slang

#include "viz_shader_common.h"

struct PostParams {
  Texture2D ssaoTexture;
  Texture2D bloomTexture0;
  Texture2D bloomTexture1;
  Texture2D bloomTexture2;
  Texture3D heatmapTexture;
  Texture2D inputTexture;
  Texture2D depthTexture;
  SamplerState ssaoSamplerState;
  SamplerState bloomSamplerState0;
  SamplerState bloomSamplerState1;
  SamplerState bloomSamplerState2;
  SamplerState heatmapSamplerState;
  SamplerState samplerState;
  SamplerState depthSamplerState;
};

ParameterBlock<PostParams> params;
ParameterBlock<PostEffectData> consts;

struct VSOutput {
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

static float near = 0.01;
float LinearDepth(float depth)
{
    return near / depth;
}

float2 ixRayBox( float3 size, float3 origin, float3 dir ) 
{ 
    float tmin = (-size.x - origin.x) / dir.x; 
    float tmax = (size.x - origin.x) / dir.x; 
 
    if (tmin > tmax)
    {
        float temp = tmax;
        tmax = tmin;
        tmin = temp;
    }
 
    float tymin = (-size.y - origin.y) / dir.y; 
    float tymax = (size.y - origin.y) / dir.y; 
 
    if (tymin > tymax)
    {
        float temp = tymax;
        tymax = tymin;
        tymin = temp;
    }
 
    if ((tmin > tymax) || (tymin > tmax)) 
        return float2( -1.0, -1.0 );
 
    if (tymin > tmin) 
        tmin = tymin; 
 
    if (tymax < tmax) 
        tmax = tymax; 
 
    float tzmin = (-size.z - origin.z) / dir.z; 
    float tzmax = (size.z - origin.z) / dir.z; 
 
    if (tzmin > tzmax)
    {
        float temp = tzmax;
        tzmax = tzmin;
        tzmin = temp;
    }
 
    if ((tmin > tzmax) || (tzmin > tmax)) 
        return float2( -1.0, -1.0 );
 
    if (tzmin > tmin) 
        tmin = tzmin; 
 
    if (tzmax < tmax) 
        tmax = tzmax; 
    
    if( tmin < 0.0 )
        tmin = 0.0;
 
    return float2( tmin, tmax );
}

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

float2 LensDistortion(float2 uv)
{
    float distortionAmount = 0.06;
    float rounding = 0.02;
    uv = uv * 2.0 - 1.0;
    float2 centeredUV = uv;
    float distanceFromCenter = length(uv);
    uv = uv * (1.0 - distortionAmount + distortionAmount * distanceFromCenter * distanceFromCenter);

    // Very near the corners, push the uv to something invalid to get some rounding.
    if( length(saturate(abs(uv) - 1 + rounding)) > rounding )
        uv.x = 2;

    return uv * 0.5 + 0.5;
}

float3 ChromaticAberration(float2 centeredUV, float green)
{
    float chromaticAberrationAmount = 0.003;
    float2 offset = centeredUV * centeredUV * chromaticAberrationAmount;
    float4 r = params.inputTexture.Sample(
        params.samplerState, centeredUV + float2(offset.x, offset.y));
    float4 b = params.inputTexture.Sample(
        params.samplerState, centeredUV + float2(-offset.x, -offset.y));
    return float3(r.r, green, b.b);
}

float Vignette(float2 uv)
{
    float distanceFromCenter = length(uv - 0.5);
    float vignetteAmount = 0.6;
    float brighten = 0.0;
    float power = 3.0;
    return pow(saturate(1.0 - distanceFromCenter * vignetteAmount  + brighten), power);
}

float SSAO(float2 uv)
{
    // Sample the 16 closest points and average.
    float ssao = 0;
    for(int x = -1; x < 3; x++)
    {
        for( int y = -1; y < 3; y++)
        {
            float2 offset = float2(x, y) / (float2)consts.view.fbDims;
            ssao += params.ssaoTexture.Sample(params.ssaoSamplerState, uv + offset).x;
        }
    }

    return ssao / 16;
}

float3 DepthOfField(float3 color, float2 uv, inout float3 bloomColor)
{
    // Sample depth.
    float depth = params.depthTexture.Sample(params.depthSamplerState, uv).x;
    depth = LinearDepth(depth);
    //color.rgb = lerp( color.rgb, float3(1, 1, 1), saturate(depth / 1000) * 0.5);

    // Sampler around the pixel for various blur use cases.
    int sampleCount = 12;
    float kernelSize = 0.02;
    bloomColor = float3(0, 0, 0);
    float3 blurColor = float3(0, 0, 0);
    float blurNorm = 0;
    for(int x = 0; x < sampleCount; x++) {
        for(int y = 0; y < sampleCount; y++) {
            float2 offset = float2(x, y) / (float)sampleCount * kernelSize - kernelSize.xx * 0.5;
            float2 sampleUV = uv + offset;
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
    //color.rgb = lerp(color.rgb, blurColor / blurNorm, blurAmount);
    return color;
}

float3 Bloom(float3 color, float2 uv)
{
    // Bloom
    float bloomAmount = 1.0;
    float3 bloomColor = params.bloomTexture0.Sample(params.bloomSamplerState0, uv).rgb * 0.5;
    bloomColor += params.bloomTexture1.Sample(params.bloomSamplerState1, uv).rgb * 0.7;
    bloomColor += params.bloomTexture2.Sample(params.bloomSamplerState2, uv).rgb * 1.0;
    color.rgb = color.rgb * 0.8 + bloomColor * bloomAmount;
    // Overdrive to white.
    /*for( int i = 0; i < 3; i++ ) {
        color[i] += saturate(color[(i+1)%3] - 1) + saturate(color[(i+2)%3] - 1);
    }*/
    return color;
}

float3 ClipToWorld(float3 pos, float3x4 c2w, inout float3 camPos)
{
    float3 worldPos = float3(c2w[0].w, c2w[1].w, c2w[2].w);
    camPos = worldPos;
    worldPos += c2w[0].xyz * pos.x * pos.z;
    worldPos += c2w[1].xyz * pos.y * pos.z;
    worldPos += c2w[2].xyz * pos.z;

    return worldPos;
}

float HeightFog(float2 uv)
{
    // We need to reconstruct actual world position from sampled depth.
    float depth = params.depthTexture.Sample(params.depthSamplerState, uv).x;
    if( depth == 0 )
        return 1;
    depth = LinearDepth(depth);
    float3 camPos;
    float3 worldPos = ClipToWorld(float3(uv * 2 - 1, depth), consts.c2w, camPos);

    // Use the world position to get the fog amount.
    float d = consts.params1.x; // Half Distance.
    float h = consts.params1.y; // Half Height.
    float offset = consts.params1.z;

    // Ensure both points are at or below to fog plane.
    camPos.z -= offset;
    worldPos.z -= offset;
    if( camPos.z > 0 && worldPos.z > 0 )
        return 0;

    // Move each point to the fog plane.
    float3 dir = worldPos - camPos;
    if( camPos.z > 0 )
    {
        float t = camPos.z / -dir.z;
        depth = (1 - t) * depth;
        camPos += dir * t;
    }
    if( worldPos.z > 0 )
    {
        float t = worldPos.z / dir.z;
        depth = (1 - t) * depth;
        worldPos -= dir * t;
    }

    // Linear Height Fog
    /*float d1 = saturate((-camPos.z) / h / d );
    float d2 = saturate((-worldPos.z) / h / d );
    float integrated = depth * (d1 + d2) * 0.5;*/

    // Exponential Height Fog
    float d1 = (1.0 - exp( camPos.z / h * d)) / d;
    float i1 = ( h / d * exp(camPos.z * d / h) - camPos.z) / d;
    float i2 = ( h / d * exp(worldPos.z * d / h) - worldPos.z ) / d;
    float integrated = depth * d1;
    if( camPos.z != worldPos.z )
    {
        integrated = depth * (i1 - i2) / (worldPos.z - camPos.z);
    }
    float fogAmount = 1 - exp(-integrated);

    return saturate(fogAmount);
}

float SampleVolume( float3 samplePos )
{
    // Sample the volume texture.
    float3 uvw = (samplePos) / ( consts.mapBBMax.xyz - consts.mapBBMin.xyz ) + 0.5;
    return saturate(params.heatmapTexture.Sample(params.samplerState, uvw).x * 1.5 - 0.5);
}

float4 VolumetricFog(float2 uv)
{
    float density = 0.25;

    // We need to reconstruct actual world position from sampled depth.
    float depth = params.depthTexture.Sample(params.depthSamplerState, uv).x;
    bool failure = false;
    depth = depth == 0.0 ? 132000.0 : LinearDepth(depth);
    float3 camPos;
    float3 worldPos = ClipToWorld(float3(uv * 2 - 1, depth), consts.c2w, camPos);

    // Now we can transform the end points into a ray in volume space, relative to the center.
    float3 boxCenter = (consts.mapBBMin.xyz + consts.mapBBMax.xyz) * 0.5;
    float3 boxSize = (consts.mapBBMax.xyz - consts.mapBBMin.xyz) * 0.5;
    float3 rayDir = worldPos - camPos;
    float3 rayOrigin = camPos - boxCenter;
    depth = length(rayDir);
    rayDir = rayDir / depth;

    // Get the portion of the ray to iterate over.
    float2 ix = ixRayBox(boxSize, rayOrigin, rayDir);
    ix.y = min(depth, ix.y);
    if( ix.x >= ix.y || ix.y <= 0.0 )
        failure = true;
        
    // Ray march through the volume.
    float stepSize = (0.99 / 64.0) * (ix.y - ix.x);
    float alpha = 0.0;
    float3 color = float3(0.0, 0.0, 0.0);
    float t = ix.x;
    
    for(int i = 0; i < 64; i++)
    {
        float3 samplePos = rayOrigin + rayDir * t;
        float sample = SampleVolume(samplePos);

        // Calculate a color for this sample.
        float3 sampleColor = float3(0.0, 0.0, 0.0);
        if( sample < 0.5 )
        {
            sampleColor += lerp(float3(0.0, 1.0, 0.0), float3(1.0, 1.0, 0.0), sample * 2.0);
        }
        else
        {
            sampleColor += lerp(float3(1.0, 1.0, 0.0), float3(1.0, 0.0, 0.0), (sample - 0.5) * 2.0);
        }

        // Occlude this sample from all preceding samples.
        // It would be nice to early out on full alpha, but conditional texture samples crash the shader compiler.
        float density = 0.03 * stepSize * min(sample, 0.25) * (1.0f - alpha);
        if( density > 0 )
            color = lerp(color, sampleColor, 1.0 - alpha / saturate(alpha+density));
        alpha = saturate(alpha+density);

        t += stepSize;
    }
    
    
    if( failure )
        return float4(0.0, 0.0, 0.0, 0.0);
    return float4( color, saturate(alpha) );
}

float3 Scanlines( float3 color, float2 uv )
{
    float pixelf = (uv.x + 1)*0.5 * consts.view.fbDims.x;
    float vertf = (uv.y + 1)*0.5 * consts.view.fbDims.y;
    int pixel = (int)pixelf;
    int vert = (int)vertf;
    float pfrac = frac(pixelf);
    float vfrac = frac(vertf);
    float amt = 0.5;

    float dim = 1.0;
    if((pixel/3+vert)%4 == 0)
        dim = 1.0 - vfrac;
    if((pixel/3+vert)%4 == 1)
        dim = vfrac;

    if(pixel%3 == 0)
        color = lerp( color, color * float3( 1.0 - pfrac, pfrac, 0.0 ) * dim, amt);
    if(pixel%3 == 1)
        color = lerp( color, color * float3( 0.0, 1.0 - pfrac, pfrac, ) * dim, amt);
    if(pixel%3 == 2)
        color = lerp( color, color * float3( pfrac, 0.0, 1.0 - pfrac ) * dim, amt);

    return color;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    input.uv.y = 1 - input.uv.y;

    input.uv = LensDistortion(input.uv);

    // Main Sample
    float4 color = params.inputTexture.Sample(params.samplerState, input.uv);

    color.rgb = ChromaticAberration(input.uv, color.g);

    color.rgb *= SSAO(input.uv);

    color.rgb = lerp(color.rgb, float3(0.7, 0.8, 1.0), HeightFog(input.uv));

    float4 vfog = VolumetricFog(input.uv);
    color.rgb = lerp(color.rgb, vfog.rgb, vfog.a);

    //float3 blurColor;
    //color.rgb = DepthOfField(color.rgb, input.uv, blurColor);

    color.rgb = Bloom(color.rgb, input.uv);

    color.rgb *= Vignette(input.uv);

    //color.rgb = Scanlines( color.rgb, input.uv );

    if( input.uv.x > 1 || input.uv.x < 0 || input.uv.y > 1 || input.uv.y < 0 )
        color = float4(0.0, 0.0, 0.0, 1.0);

    return color;
}
