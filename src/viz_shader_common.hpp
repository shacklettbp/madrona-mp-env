#pragma once 

#include "types.hpp"

namespace madronaMPEnv {

struct float3x4 {
  Vector4 rows[3];
};

struct float3x3 {
  Vector3 rows[3];
};

namespace shader {
using float4 = Vector4;
using float3 = Vector3;
using float2 = Vector2;

using uint2 = std::array<uint32_t, 2>;
using uint = uint32_t;

#include "viz_shader_common.h"
}

using shader::OpaqueGeoVertex;
using shader::ViewData;
using shader::NonUniformScaleObjectTransform;
using shader::GlobalPassData;
using shader::OpaqueGeoPerDraw;
using shader::GoalRegionPerDraw;
using shader::AnalyticsTeamHullPerDraw;
using shader::ShotVizLineData;
using shader::AgentPerDraw;

}
