#include "utils.hpp"

using namespace madrona;

namespace madronaMPEnv {


bool traceRayAgainstWorld(Engine &ctx,
                          Vector3 o,
                          Vector3 d,
                          float *hit_t_out,
                          Entity *hit_entity_out)
{
    const MeshBVH &world_bvh = ctx.data().staticMeshes[0];

    float min_hit_t = FLT_MAX;

    TraversalStack stack;
    stack.size = 0;
    MeshBVH::HitInfo hit_info;

    bool hit = world_bvh.traceRay(o, d, &hit_info, &stack);
    if (hit) {
        min_hit_t = hit_info.tHit;
    }

    Entity hit_entity = Entity::none();

    Entity *agents = ctx.data().agents;
    CountT num_agents = (CountT)ctx.data().numAgents;

    auto checkCapsuleEntity = [&ctx, &min_hit_t, &hit, &hit_entity, o, d](
            Entity e)
    {
        Vector3 entity_pos = ctx.get<Position>(e);
        Vector3 capsule_origin = entity_pos;
        // intersectRayZOriginCapsule expects the origin at the bottom
        // of the line segment, not the capsule
        capsule_origin.z += consts::agentRadius;

        Vector3 translated_o = o - capsule_origin;

        float entity_hit_t = geo::intersectRayZOriginCapsule(
            translated_o, d, consts::agentRadius,
            consts::standHeight - 2.f * consts::agentRadius);
        
        if (entity_hit_t != 0 && entity_hit_t < min_hit_t) {
            min_hit_t = entity_hit_t;
            hit = true;
            hit_entity = e;
        }
    };

    for (CountT agent_idx = 0; agent_idx < num_agents; agent_idx++) {
        Entity agent = agents[agent_idx];
        checkCapsuleEntity(agent);
    }

    Entity *turrets = ctx.data().turrets;
    CountT num_turrets = (CountT)ctx.data().numTurrets;

    for (CountT turret_idx = 0; turret_idx < num_turrets; turret_idx++) {
        Entity turret = turrets[turret_idx];
        checkCapsuleEntity(turret);
    }

    *hit_t_out = min_hit_t;
    *hit_entity_out = hit_entity;
    return hit;
}


float sphereCastWorld(Engine &ctx,
                      Vector3 o,
                      Vector3 d,
                      float r,
                      Entity *hit_entity_out)
{
    MeshBVH &world_bvh = ctx.data().staticMeshes[0];

    Vector3 hit_normal;
    float hit_t = world_bvh.sphereCast(o, d, r, &hit_normal);
    Entity hit_entity = Entity::none();

#if 0
    Entity *agents = ctx.data().agents;
    CountT num_agents = (CountT)ctx.data().numAgents;

    auto checkCapsuleEntity = [&ctx, &hit_t, &hit_entity, o, d, r](
            Entity e)
    {
        Vector3 entity_pos = ctx.get<Position>(e);
        Vector3 capsule_origin = entity_pos;
        // intersectRayZOriginCapsule expects the origin at the bottom
        // of the line segment, not the capsule
        capsule_origin.z += consts::agentRadius;

        Vector3 translated_o = o - capsule_origin;

        float entity_hit_t = geo::intersectRayZOriginCapsule(
            translated_o, d, consts::agentRadius + r,
            consts::standHeight - 2.f * consts::agentRadius);

        if (entity_hit_t != 0 && entity_hit_t < hit_t) {
            hit_t = entity_hit_t;
            hit_entity = e;
        }
    };

    for (CountT agent_idx = 0; agent_idx < num_agents; agent_idx++) {
        Entity agent = agents[agent_idx];
        checkCapsuleEntity(agent);
    }

    Entity *turrets = ctx.data().turrets;
    CountT num_turrets = (CountT)ctx.data().numTurrets;

    for (CountT turret_idx = 0; turret_idx < num_turrets; turret_idx++) {
        Entity turret = turrets[turret_idx];
        checkCapsuleEntity(turret);
    }
#endif

    *hit_entity_out = hit_entity;
    return hit_t;
}

Aim computeAim(float yaw, float pitch)
{
    if (yaw < -math::pi) {
        yaw += 2.f * math::pi;
    } else if (yaw > math::pi) {
        yaw -= 2.f * math::pi;
    }

    if (pitch < -0.25f * math::pi) {
        pitch = -0.25f * math::pi;
    }

    if (pitch > 0.25f * math::pi) {
        pitch = 0.25f * math::pi;
    }

    math::Quat aim_rot = 
        math::Quat::angleAxis(yaw, math::up) *
        math::Quat::angleAxis(pitch, math::right);

    aim_rot = aim_rot.normalize();

    return Aim {
        .yaw = yaw,
        .pitch = pitch,
        .rot = aim_rot,
    };
}

static inline bool inFrustum(Engine &ctx, Vector3 viewspace_pos)
{
    bool in_frustum = true;
    
    in_frustum = in_frustum &&
        viewspace_pos.y * ctx.data().frustumData.y -
        fabsf(viewspace_pos.x) * ctx.data().frustumData.x >
        -consts::agentRadius;
    
    in_frustum = in_frustum &&
        viewspace_pos.y * ctx.data().frustumData.w -
        fabsf(viewspace_pos.z) * ctx.data().frustumData.z >
        -consts::agentRadius;

    return in_frustum;
}

bool isAgentVisible(Engine &ctx, Vector3 o, Aim aim,
                    Entity agent, Vector3 *out_avg_visible_pos)
{
    Vector3 base_agent_pos = ctx.get<Position>(agent);

    auto testVisible = [&](Vector3 test_pos) {
        Vector3 to_test = test_pos - o;

        Vector3 test_view_pos = aim.rot.inv().rotateVec(to_test);
        if (test_view_pos.y <= 0.f) {
            return false;
        }

        if (!inFrustum(ctx, test_view_pos)) {
            return false;
        }

        float to_test_len = to_test.length();
        if (to_test_len < consts::agentRadius) {
            return false;
        }

        to_test /= to_test_len;

        float hit_t;
        Entity hit_entity;
        bool hit = traceRayAgainstWorld(ctx, o, to_test, &hit_t, &hit_entity);

        if (!hit) {
            return false;
        }

        return hit_entity == agent;
    };

    float agent_view_height = viewHeight(ctx.get<StandState>(agent));

    Vector3 aim_right = aim.rot.rotateVec(math::right);
    Vector3 delta_right = aim_right * 0.9f * consts::agentRadius;

    Vector3 bottom = base_agent_pos;
    bottom.z += consts::agentRadius;

    Vector3 top = base_agent_pos;
    top.z += agent_view_height;

    Vector3 right = base_agent_pos;
    right.z += agent_view_height;
    right += delta_right;

    Vector3 left = base_agent_pos;
    left.z += agent_view_height;
    left -= delta_right;

    Vector3 visible_avg = Vector3::zero();
    CountT num_visible = 0;

    auto runningVisibleMean = [&](Vector3 p) {
        CountT new_num_visible = num_visible + 1;

        visible_avg =
            float(num_visible) / float(new_num_visible) * visible_avg +
            1.f / float(new_num_visible) * p;

        num_visible = new_num_visible;
    };

    if (testVisible(bottom)) {
        runningVisibleMean(bottom);
    }

    if (testVisible(top)) {
        runningVisibleMean(top);
    }

    if (testVisible(left)) {
        runningVisibleMean(left);
    }

    if (testVisible(right)) {
        runningVisibleMean(right);
    }

    *out_avg_visible_pos = visible_avg;
    return num_visible > 0;
}

static inline void standardSpawnPoint(
    Engine &ctx,
    Entity spawning_agent,
    const StandardSpawns &spawns,
    const MatchInfo &match_info,
    bool is_respawn,
    bool use_middle_spawn,
    TDMEpisode &episode,
    Vector3 *out_spawn_pt,
    float *out_spawn_yaw)
{
    CombatState &combat_state = ctx.get<CombatState>(spawning_agent);

    RNG &agent_rng = combat_state.rng;
    SpawnUsageCounter &spawn_usage = ctx.singleton<SpawnUsageCounter>();
    const TeamInfo &team_info = ctx.get<TeamInfo>(spawning_agent);

    if (!is_respawn && ctx.data().taskType == Task::TDM) {
      int i;
      if (team_info.team == match_info.teamA) {
        i = 0;
      } else {
        i = 6;
      }

      i += team_info.offset;

      *out_spawn_pt = episode.startPositions[i];// + Vector3 { 0, 0, 30 };
      *out_spawn_yaw = episode.startRotations[i];
      combat_state.immitationGoalPosition =
          episode.goalPositions[i];
      combat_state.minDistToImmitationGoal =
          episode.goalPositions[i].distance(episode.startPositions[i]);
      return;
    }

    Spawn *spawn_options;
    CountT num_spawns;

    auto spawnAgent = [&](CountT spawn_idx) {

        Spawn spawn = spawn_options[spawn_idx];

        float x_rnd = agent_rng.sampleUniform();
        float y_rnd = agent_rng.sampleUniform();
        float z_rnd = agent_rng.sampleUniform();
        float yaw_rnd = agent_rng.sampleUniform();

        float x_min = spawn.region.pMin.x;
        float x_diff = spawn.region.pMax.x - x_min;
        float y_min = spawn.region.pMin.y;
        float y_diff = spawn.region.pMax.y - y_min;
        float z_min = spawn.region.pMin.z;
        float z_diff = spawn.region.pMax.z - z_min;

        Vector3 spawn_pt = {
            .x = x_min + x_rnd * x_diff,
            .y = y_min + y_rnd * y_diff,
            .z = z_min + z_rnd * z_diff,
        };
        float spawn_yaw =
            spawn.yawMin + yaw_rnd * (spawn.yawMax - spawn.yawMin);

        *out_spawn_pt = spawn_pt;
        *out_spawn_yaw = spawn_yaw;
    };

    if (!is_respawn || spawns.numCommonRespawns == 0) {
        uint32_t *init_spawn_usage_tracker;

        CountT num_default_spawns;
        CountT num_extra_spawns;
        if (team_info.team == match_info.teamA) {
            spawn_options = spawns.aSpawns;
            num_default_spawns = (CountT)spawns.numDefaultASpawns;
            num_extra_spawns = (CountT)spawns.numExtraASpawns;

            init_spawn_usage_tracker = spawn_usage.initASpawnsLastUsedTick;
        } else {
            spawn_options = spawns.bSpawns;
            num_default_spawns = (CountT)spawns.numDefaultBSpawns;
            num_extra_spawns = (CountT)spawns.numExtraBSpawns;

            init_spawn_usage_tracker = spawn_usage.initBSpawnsLastUsedTick;
        } 

        if (use_middle_spawn) {
            spawn_options += num_default_spawns;
            num_spawns = num_extra_spawns;
        } else {
            num_spawns = num_default_spawns;
        }

        assert(num_spawns <= SpawnUsageCounter::maxNumSpawns);

        CountT init_spawn_idx = -1;
        for (CountT i = 0; i < 5; i++) {
            CountT spawn_idx = (CountT)agent_rng.sampleI32(0, num_spawns);

            if (init_spawn_usage_tracker[spawn_idx] ==
                    (uint32_t)match_info.curStep) {
                continue;
            }

            init_spawn_idx = spawn_idx;
            break;
        }

        if (init_spawn_idx == -1) {
            init_spawn_idx = (CountT)agent_rng.sampleI32(0, num_spawns);
        }

        spawnAgent(init_spawn_idx);
        init_spawn_usage_tracker[init_spawn_idx] = match_info.curStep;

        return;
    } 

    spawn_options = spawns.commonRespawns;
    num_spawns = (CountT)spawns.numCommonRespawns;

    Vector3 zone_center;
    {
      if (ctx.data().taskType == Task::TDM) {
        zone_center = Vector3::zero();
      } else {
        const ZoneState &zone_state =
            ctx.singleton<ZoneState>();
        AABB zone_aabb = ctx.data().zones.bboxes[
            zone_state.curZone];

        zone_center = 0.5f * (zone_aabb.pMin + zone_aabb.pMax);
      }
    }

    float best_spawn_score = FLT_MAX;
    CountT best_spawn_idx = -1;
    for (CountT spawn_idx = 0; spawn_idx < num_spawns; spawn_idx++) {
        uint32_t last_used = spawn_usage.respawnLastUsedTick[spawn_idx];
        if (last_used == (uint32_t)match_info.curStep) {
            continue;
        }

        float score = 0.f;

        uint32_t elapsed = consts::deltaT * 
            float((uint32_t)match_info.curStep - last_used);

        float elapsed_threshold = 3.f;

        constexpr float elapsed_weight = 0.1f;
        constexpr float dist_weight = 0.01f;

        if (elapsed < 3.f) {
            score += elapsed_weight * (3.f - elapsed);
        }

        Spawn spawn = spawn_options[spawn_idx];

        Vector3 spawn_pt = 0.5f * (spawn.region.pMin + spawn.region.pMax);

        for (CountT j = 0; j < (CountT)ctx.data().numAgents; j++) {
            Entity other_agent = ctx.data().agents[j];

            if (spawning_agent == other_agent) {
                continue;
            }

            if (ctx.get<Alive>(other_agent).mask == 0.f) {
                continue;
            }


            Vector3 other_pos = ctx.get<Position>(other_agent);
            float dist = spawn_pt.distance(other_pos);

            if (dist < 4.f * consts::agentRadius) {
                score += 100000.f;
            } else {
                TeamInfo other_team = ctx.get<TeamInfo>(other_agent);
                if (other_team.team == team_info.team) {
                    continue;
                }

                score += dist_weight * (1.f / dist);
            }
        }

        {
            float dist_to_zone_center = spawn_pt.distance(zone_center);

            if (dist_to_zone_center < 100.f) {
                score += 1000000.f;
            }
        }

        if (score < best_spawn_score) {
            best_spawn_idx = spawn_idx;
            best_spawn_score = score;
        }
    }

    assert(best_spawn_idx != -1);
    spawnAgent(best_spawn_idx);
    spawn_usage.respawnLastUsedTick[best_spawn_idx] =
        (uint32_t)match_info.curStep;
}


static void hardcodedSpawnPoint(
    Engine &ctx,
    RNG &rng,
    Entity spawning_agent,
    const MatchInfo &match_info,
    Vector3 *out_spawn_pt,
    float *out_spawn_yaw,
    float *out_spawn_pitch)
{
    TeamInfo team = ctx.get<TeamInfo>(spawning_agent);

    struct HardcodedSpawn {
        Vector3 pos;
        float yaw;
        float pitch;
    };

    struct HardcodedSpawns {
        std::array<HardcodedSpawn, 6> spawns;
    };

    static const std::array<HardcodedSpawns, 1> hardcoded_spawns {{
        {{
            HardcodedSpawn {
                {510.0, 179.1, -64},
                -2.05,
                0,
            },
            HardcodedSpawn {
                {525.8, 17.1, -64},
                -0.80,
                0,
            },
            HardcodedSpawn {
                {434.3, 184.7, -64},
                -1.80,
                0,
            },
#if 0
            HardcodedSpawn {
                {1058.8, 1029.6, -64},
                2.73,
                0,
            },
#endif
            HardcodedSpawn {
                {1037.2, 449.0, -56},
                2.37,
                0,
            },
            HardcodedSpawn {
                {1094.3, 200.1, -56},
                1.41,
                0,
            },
            HardcodedSpawn {
                {1045.8, 416.8, -56},
                2.37f,
                0,
            },
        }},
    }};

    HardcodedSpawns spawns = hardcoded_spawns[ctx.data().hardcodedSpawnIdx];
    CountT base_idx = team.team == match_info.teamA ? 0 : 3;

    CountT idx = base_idx + team.offset;

    HardcodedSpawn spawn = spawns.spawns[idx];

    if (false && team.team == 0 && team.offset == 2) {
        static const auto curriculum_spawns = std::to_array({
            HardcodedSpawn {
                {420.4, 282.4, -64},
                -1.80,
                0,
            },
            HardcodedSpawn {
                {411.3, 352.0, -64},
                -1.30,
                0,
            },
            HardcodedSpawn {
                {501.8, 383.2, -64},
                -1.30,
                0,
            },
            HardcodedSpawn {
                {711.3, 463.7, -56},
                -1.30,
                0,
            },
            HardcodedSpawn {
                {833.6, 524.2, -56},
                -1.80,
                0,
            },
            HardcodedSpawn {
                {893.7, 559.1, -56},
                -2.55,
                0,
            },
            HardcodedSpawn {
                {930.3, 556.4, -56},
                -2.80,
                0,
            },
            HardcodedSpawn {
                {950.5, 606.7, -56},
                -3.02,
                0,
            },
            HardcodedSpawn {
                {904.8, 542.8, -56},
                -2.44,
                0,
            },
#if 0
            HardcodedSpawn {
                {1058.8, 1029.6, -64},
                2.73,
                0,
            },
            HardcodedSpawn {
                {1025.4, 923.6, -64},
                2.73,
                0,
            },
            HardcodedSpawn {
                {1014.1, 712.2, -64},
                2.73,
                0,
            },
            HardcodedSpawn {
                {957.0, 672.9, -56},
                2.73,
                0,
            },
            HardcodedSpawn {
                {978.3, 628.5, -56},
                2.80,
                0,
            },
            HardcodedSpawn {
                {1005.0, 531.0, -56},
                2.98,
                0,
            },
            HardcodedSpawn {
                {986.1, 533.4, -56},
                -2.80,
                0,
            },
            HardcodedSpawn {
                {959.9, 524.2, -56},
                -2.80,
                0,
            },
#endif
        });

        spawn = curriculum_spawns[
            rng.sampleI32(0, curriculum_spawns.size())];
    }

    *out_spawn_pt = spawn.pos;
    *out_spawn_yaw = spawn.yaw;
    *out_spawn_pitch = spawn.pitch;
}

static void curriculumSpawnPoint(
    Engine &ctx,
    RNG &base_rng,
    Navmesh &navmesh,
    const MatchInfo &match_info,
    const SpawnCurriculum &spawn_curriculum,
    Entity spawning_agent,
    Vector3 *out_spawn_pt,
    float *out_spawn_yaw)
{
    const TeamInfo &agent_team_info = ctx.get<TeamInfo>(spawning_agent);

    auto &curriculum_tier = spawn_curriculum.tiers[match_info.curCurriculumTier];
    NavmeshSpawn &navmesh_spawn = curriculum_tier.spawns[
        match_info.curCurriculumSpawnIdx];

    uint32_t navmesh_spawn_poly_offset;
    uint32_t num_navmesh_spawn_polys;
    float base_yaw;
    if (agent_team_info.team == match_info.teamA) {
        navmesh_spawn_poly_offset = navmesh_spawn.aOffset;
        num_navmesh_spawn_polys = navmesh_spawn.numANavmeshPolys;
        base_yaw = navmesh_spawn.aBaseYaw;
    } else {
        navmesh_spawn_poly_offset = navmesh_spawn.bOffset;
        num_navmesh_spawn_polys = navmesh_spawn.numBNavmeshPolys;
        base_yaw = navmesh_spawn.bBaseYaw;
    }

    for (CountT i = 0; i < consts::numSpawnRetries; i++) {
        uint32_t poly_selector = base_rng.sampleI32(0, num_navmesh_spawn_polys);
        uint32_t navmesh_tri = curriculum_tier.spawnPolyData[
            navmesh_spawn_poly_offset + poly_selector];

        Vector3 a, b, c;
        navmesh.getTriangleVertices(navmesh_tri, &a, &b, &c);

        Vector2 uv = rand::sample2xUniform(base_rng.randKey());
        if (uv.x + uv.y > 1.f) {
            uv.x = 1.f - uv.x;
            uv.y = 1.f - uv.y;
        }

        float w = 1.f - uv.x - uv.y;
        Vector3 spawn_pt = a * uv.x + b * uv.y + c * w;

        bool overlap = false;
        if (i < consts::numSpawnRetries - 1) {
            for (CountT j = 0; j < (CountT)ctx.data().numAgents; j++) {
                Entity other_agent = ctx.data().agents[j];

                if (spawning_agent == other_agent) {
                    continue;
                }

                if (ctx.get<Alive>(other_agent).mask == 0.f) {
                    continue;
                }

                Vector3 other_pos = ctx.get<Position>(other_agent);

                if (spawn_pt.distance2(other_pos) <
                        math::sqr(3.f * consts::agentRadius)) {
                    overlap = true;
                    break;
                }
            }
        }

        if (!overlap) {
            float yaw_rnd = base_rng.sampleUniform();
            float yaw = base_yaw + math::pi / 2.f * yaw_rnd - math::pi / 4.f;

            *out_spawn_pt = spawn_pt;
            *out_spawn_yaw = yaw;
            break;
        }
    }
}

void spawnAgents(Engine &ctx, bool is_respawn)
{
    Navmesh &navmesh = ctx.singleton<LevelData>().navmesh;
    const StandardSpawns &standard_spawns = ctx.singleton<StandardSpawns>();
    const SpawnCurriculum &spawn_curriculum = ctx.singleton<SpawnCurriculum>();
    const MatchInfo &match_info = ctx.singleton<MatchInfo>();

    const bool navmesh_spawn = (
        ctx.data().simFlags & SimFlags::NavmeshSpawn) ==
        SimFlags::NavmeshSpawn;
    
    const bool randomize_hp_magazine = (
        ctx.data().simFlags & SimFlags::RandomizeHPMagazine) ==
        SimFlags::RandomizeHPMagazine;

    const bool can_spawn_in_middle = (
        ctx.data().simFlags & SimFlags::SpawnInMiddle) ==
        SimFlags::SpawnInMiddle;

    const bool enable_spawn_curriculum =
        ((ctx.data().simFlags & SimFlags::EnableCurriculum) ==
            SimFlags::EnableCurriculum) && match_info.enableSpawnCurriculum;

    const bool enable_hardcoded_spawns =
        ((ctx.data().simFlags & SimFlags::HardcodedSpawns) ==
            SimFlags::HardcodedSpawns);

    Entity alive_agents[consts::maxTeamSize * 2];
    Entity dead_agents[consts::maxTeamSize * 2];

    CountT num_dead = 0;
    CountT num_alive = 0;
    for (CountT i = 0; i < ctx.data().numAgents; i++) {
        Entity agent = ctx.data().agents[i];

        if (ctx.get<Alive>(agent).mask == 0.f) {
            dead_agents[num_dead++] = agent;
        } else {
            alive_agents[num_alive++] = agent;
        }
    }

    if (num_dead == 0) {
        return;
    }

    RNG &base_rng = ctx.data().baseRNG;

    TDMEpisode &episode = ctx.data().episodes[
        base_rng.sampleI32(0, ctx.data().numEpisodes)];

    bool standard_use_middle_spawn;
    if (can_spawn_in_middle) {
        standard_use_middle_spawn = base_rng.sampleUniform() < 0.5f;
    } else {
        standard_use_middle_spawn = false;
    }

    for (CountT i = 0; i < num_dead; i++) {
        Entity spawning_agent = dead_agents[i];

        Vector3 spawn_pt;
        float spawn_yaw;
        float spawn_pitch = 0.f;
        if (enable_hardcoded_spawns && !is_respawn) {
            hardcodedSpawnPoint(ctx, base_rng, spawning_agent, match_info,
                                &spawn_pt, &spawn_yaw, &spawn_pitch);
        } else if (navmesh_spawn) {
            spawn_pt = navmesh.samplePoint(base_rng.randKey());
            spawn_yaw = base_rng.sampleUniform() * 2.f * math::pi;
        } else if (enable_spawn_curriculum) {
            curriculumSpawnPoint(ctx, base_rng, navmesh,
                                 match_info, spawn_curriculum,
                                 spawning_agent,
                                 &spawn_pt, &spawn_yaw);
        }  else {
            standardSpawnPoint(ctx, spawning_agent, standard_spawns, match_info,
                               is_respawn, standard_use_middle_spawn, episode,
                               &spawn_pt, &spawn_yaw);
        }

        ctx.get<Position>(spawning_agent) = spawn_pt;
        ctx.get<Rotation>(spawning_agent) =
            math::Quat::angleAxis(spawn_yaw, math::up).normalize();
        ctx.get<Aim>(spawning_agent) = computeAim(spawn_yaw, spawn_pitch);

        ctx.get<AgentVelocity>(spawning_agent) = Vector3::zero();
        ctx.get<ExternalForce>(spawning_agent) = Vector3::zero();
        ctx.get<ExternalTorque>(spawning_agent) = Vector3::zero();

        HP &hp = ctx.get<HP>(spawning_agent);
        Magazine &magazine = ctx.get<Magazine>(spawning_agent);
        CombatState &combat_state = ctx.get<CombatState>(spawning_agent);
        Alive &alive = ctx.get<Alive>(spawning_agent);

        combat_state.weaponType = base_rng.sampleI32(0, ctx.data().numWeaponTypes);

        WeaponStats &weapon_stats = 
            ctx.data().weaponTypeStats[combat_state.weaponType];

        if (randomize_hp_magazine) {
            int32_t tenth_hp = base_rng.sampleI32(1, 11);
            hp.hp = float(tenth_hp * 10);

            magazine.numBullets = base_rng.sampleI32(0, weapon_stats.magSize),
            magazine.isReloading = 0;
        } else {
            hp.hp = 100.f;

            magazine.numBullets = weapon_stats.magSize;
            magazine.isReloading = 0;
        }

        if (is_respawn) {
            combat_state.remainingRespawnSteps = 0;
        } else {
            combat_state.remainingRespawnSteps =
                consts::respawnInvincibleSteps;
        }

        combat_state.remainingStepsBeforeAutoheal = 0;
        for (int32_t j = 0; j < consts::maxFireQueueSize; j++) {
            combat_state.fireQueue[j] = false;
        }

        if (ctx.data().taskType == Task::Zone ||
                ctx.data().taskType == Task::ZoneCaptureDefend) {
            const ZoneState &zone_state =
                ctx.singleton<ZoneState>();

            combat_state.inZone = false;

            AABB zone_aabb = ctx.data().zones.bboxes[
                zone_state.curZone];
            Vector3 zone_center = (zone_aabb.pMax + zone_aabb.pMin) / 2.f;

            combat_state.minDistToZone = spawn_pt.distance(zone_center);
        }

        ctx.get<StandState>(spawning_agent) = {
            .curPose = Pose::Stand,
            .tgtPose = Pose::Stand,
            .transitionRemaining = 0,
        };

        alive.mask = 1.f;
    }

}

}
