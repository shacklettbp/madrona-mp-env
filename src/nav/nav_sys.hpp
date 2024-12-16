#pragma once

#include <madrona/rand.hpp>
#include <madrona/taskgraph_builder.hpp>

#include "../types.hpp"

namespace madronaMPEnv {

constexpr inline i32 navMaxVertsPerPoly = 6;
constexpr inline i32 navMaxPolys = 8192;
constexpr inline i32 navSentinel = (i32)0xFFFF'FFFF;

struct NavNearestResult {
  Vector3 point;
  float distance2;
  u16 poly;
};

struct NavSampleResult {
  Vector3 point;
  u16 poly;
};

struct NavPathingState {
  Entity pathEntity = Entity::none();
  i32 curPathOffset = -1;
  Vector3 goalPosition = Vector3::zero();
};

struct NavPathingResult {
  static constexpr inline i32 maxNumVerts = 8;
  Vector3 pathVerts[maxNumVerts];
  i32 numVerts = 0;
};

namespace NavSystem {

i32 findContainingPoly(Context &ctx, Vector3 pos);

NavNearestResult findNearestPoly(Context &ctx, Vector3 pos);

NavSampleResult sampleRandomPointOnNavmesh(Context &ctx, RandKey rnd);

NavPathingState queueFindPath(
    Context &ctx, Vector3 start_pos, i32 start_poly,
    Vector3 end_pos, i32 end_poly);

void init(Context &ctx);

void registerTypes(ECSRegistry &registry);

TaskGraphNodeID setupFindPathsTasks(TaskGraphBuilder &builder,
                                    Span<const TaskGraphNodeID> deps);

TaskGraphNodeID setupFollowPathTasks(TaskGraphBuilder &builder,
                                     Span<const TaskGraphNodeID> deps);

}

struct NavRenderState;

namespace NavSystemFrontend {

NavRenderState * createRenderState(Renderer &renderer);
void destroyRenderState(Renderer &renderer, NavRenderState *render_state);

void debugDraw(Context &ctx, NavRenderState *render_state,
               RasterPassEncoder &raster_enc,
               bool render_edge_overlay);


}

}
