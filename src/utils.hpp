#pragma once

#include "sim.hpp"

#include <madrona/math.hpp>

namespace madronaMPEnv {

bool traceRayAgainstWorld(Engine &ctx,
                          Vector3 o,
                          Vector3 d,
                          float *hit_t_out,
                          Entity *hit_entity_out);

float sphereCastWorld(Engine &ctx,
                      Vector3 o,
                      Vector3 d,
                      float r,
                      Entity *hit_entity_out);


Aim computeAim(float yaw, float pitch);

bool isAgentVisible(Engine &ctx, Vector3 o, Aim aim,
                    Entity agent, Vector3 *out_avg_visible_pos);

void spawnAgents(Engine &ctx, bool is_respawn);

inline float viewHeight(StandState stand_state)
{
    Pose cur_pose = stand_state.curPose;

    float top_height;
    switch (cur_pose) {
    case Pose::Stand: {
        top_height = consts::standHeight;
    } break;
    case Pose::Crouch: {
        top_height = consts::crouchHeight;
    } break;
    case Pose::Prone: {
        top_height = consts::proneHeight;
    } break;
    default: MADRONA_UNREACHABLE();
    }
    return top_height - consts::agentRadius;
}

}
