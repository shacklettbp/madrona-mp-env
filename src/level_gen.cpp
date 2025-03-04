#include "level_gen.hpp"
#include "utils.hpp"

namespace madronaMPEnv {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

enum class RoomType : uint32_t {
    SingleButton,
    DoubleButton,
    CubeButtons,
    NumTypes,
};

// Creates floor, outer walls, and agent entities.
// All these entities persist across all episodes.
void createPersistentEntities(Engine &ctx, const TaskConfig &cfg)
{
    CountT total_num_agents;
    if (cfg.task == Task::Explore) {
        total_num_agents = 1;
    } else if (cfg.task == Task::TDM ||
               cfg.task == Task::Zone ||
               cfg.task == Task::ZoneCaptureDefend) {
        assert(cfg.eTeamSize == cfg.pTeamSize);
        total_num_agents = cfg.pTeamSize + cfg.eTeamSize;
    } else if (cfg.task == Task::Turret) {
        total_num_agents = cfg.pTeamSize;
    } else {
        assert(false);
        MADRONA_UNREACHABLE();
    }

    ctx.data().agents = (Entity *)rawAlloc(
        sizeof(Entity) * total_num_agents);
    ctx.data().numAgents = total_num_agents;

    if (cfg.task == Task::Turret) {
        ctx.data().turrets = (Entity *)rawAlloc(
            sizeof(Entity) * cfg.eTeamSize);
        ctx.data().numTurrets = cfg.eTeamSize;
    } else {
        ctx.data().turrets = nullptr;
        ctx.data().numTurrets = 0;
    }

    ctx.data().staticMeshes = cfg.staticMeshes;
    ctx.data().numStaticMeshes = cfg.numStaticMeshes;

    ctx.data().staticEntities = (Entity *)rawAlloc(
        sizeof(Entity) * cfg.numStaticMeshes);
    for (uint32_t i = 0; i < cfg.numStaticMeshes; i++) {
        Entity static_ent = ctx.data().staticEntities[i] =
            ctx.makeEntity<StaticGeometry>();

        ctx.get<Position>(static_ent) = Vector3::zero();
        ctx.get<Rotation>(static_ent) = Quat { 1, 0, 0, 0 };
        ctx.get<Scale>(static_ent) = Diag3x3 { 1, 1, 1 };
        // Static mesh object IDs are consts::numNonMapAssets ... numStaticMeshes + consts::numNonMapAssets
        ctx.get<ObjectID>(static_ent).idx = i + consts::numNonMapAssets;
    }

    ctx.data().navmeshQueryQueueData = (uint32_t *)rawAlloc(
        sizeof(uint32_t) * ctx.singleton<LevelData>().navmesh.numTris);
    ctx.data().navmeshQueryVisitedData = (bool *)rawAlloc(
        sizeof(bool) * ctx.singleton<LevelData>().navmesh.numTris);

    if (cfg.task == Task::Zone || cfg.task == Task::ZoneCaptureDefend) {
        const Zones &src_zones = ctx.data().zones;

        assert(src_zones.numZones <= consts::maxZones);
        for (CountT i = 0; i < (CountT)src_zones.numZones; i++) {
            AABB aabb = src_zones.bboxes[i];
            float rotation = src_zones.rotations[i];
            auto rot_txfm = Quat::angleAxis(rotation, math::up);

            Vector3 center = (aabb.pMax + aabb.pMin) / 2.f;
            Vector3 diff = rot_txfm.inv().rotateVec(aabb.pMax - aabb.pMin);

            Entity zone = ctx.makeEntity<ZoneViz>();
            ctx.get<Position>(zone) = center;
            ctx.get<Rotation>(zone) = rot_txfm;
            ctx.get<Scale>(zone) = Diag3x3 { diff.x, diff.y, diff.z };
            ctx.data().zoneEntities[i] = zone;
        }
    }

    if (cfg.showSpawns) {
        const StandardSpawns &spawns = cfg.standardSpawns;

        const float spawn_obj_scale = 10.f;
        for (uint32_t i = 0; i < spawns.numDefaultASpawns; i++) {
            Spawn spawn = spawns.aSpawns[i];
            Vector3 pos = 0.5f * (spawn.region.pMin + spawn.region.pMax);

            Entity e = ctx.makeEntity<StaticGeometry>();
            ctx.get<Position>(e) = pos;
            ctx.get<Rotation>(e) = Quat::id();
            ctx.get<Scale>(e) = spawn_obj_scale * Diag3x3::id();
            ctx.get<ObjectID>(e).idx = 11;
        }

        for (uint32_t i = 0; i < spawns.numDefaultBSpawns; i++) {
            Spawn spawn = spawns.bSpawns[i];
            Vector3 pos = 0.5f * (spawn.region.pMin + spawn.region.pMax);

            Entity e = ctx.makeEntity<StaticGeometry>();
            ctx.get<Position>(e) = pos;
            ctx.get<Rotation>(e) = Quat::id();
            ctx.get<Scale>(e) = spawn_obj_scale * Diag3x3::id();
            ctx.get<ObjectID>(e).idx = 12;
        }

        for (uint32_t i = 0; i < spawns.numCommonRespawns; i++) {
            Spawn spawn = spawns.commonRespawns[i];
            Vector3 pos = 0.5f * (spawn.region.pMin + spawn.region.pMax);

            Entity e = ctx.makeEntity<StaticGeometry>();
            ctx.get<Position>(e) = pos;
            ctx.get<Rotation>(e) = Quat::id();
            ctx.get<Scale>(e) = spawn_obj_scale * Diag3x3::id();
            ctx.get<ObjectID>(e).idx = 10;
        }

        for (uint32_t i = 0; i < spawns.numRespawnRegions; i++) {
            RespawnRegion region = spawns.respawnRegions[i];
            Vector3 pos = 0.5f * (region.aabb.pMin + region.aabb.pMax);

            Entity e = ctx.makeEntity<StaticGeometry>();

            auto rot_txfm = Quat::angleAxis(region.rotation, math::up);

            Vector3 diff = rot_txfm.inv().rotateVec(
                region.aabb.pMax - region.aabb.pMin);

            ctx.get<Position>(e) = pos;
            ctx.get<Rotation>(e) = rot_txfm;
            ctx.get<Scale>(e) = Diag3x3::fromVec(diff);
            ctx.get<ObjectID>(e).idx = 13;
        }
    }

    // Create agent entities. Note that this leaves a lot of components
    // uninitialized, these will be set during world generation, which is
    // called for every episode.
    for (CountT i = 0; i < total_num_agents; ++i) {
        Entity agent;
        if (cfg.task == Task::Explore) {
            agent = ctx.makeEntity<ExploreAgent>();
        } else if (cfg.task == Task::TDM ||
                   cfg.task == Task::Zone ||
                   cfg.task == Task::ZoneCaptureDefend ||
                   cfg.task == Task::Turret) {
            agent = ctx.makeEntity<PvPAgent>();
        } else {
            assert(false);
            MADRONA_UNREACHABLE();
        }

        ctx.data().agents[i] = agent;

        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };

        ExploreTracker &explore_tracker = ctx.get<ExploreTracker>(agent);
        for (int32_t y = 0; y < ExploreTracker::gridMaxY; y++) {
            for (int32_t x = 0; x < ExploreTracker::gridMaxX; x++) {
                explore_tracker.visited[y][x] = 0xFFFF'FFFFu;
            }
        }

        ctx.get<AgentPolicy>(agent).idx = 0;
        //ctx.get<AgentPolicy>(agent).idx = consts::aStarPolicyID;
        
        ctx.get<PvPAimAction>(agent) = {
          .yaw = 0.f,
          .pitch = 0.f,
        };

        ctx.get<PvPDiscreteAimAction>(agent) = {
          .yaw = consts::discreteAimNumYawBuckets / 2,
          .pitch = consts::discreteAimNumPitchBuckets / 2,
        };
        ctx.get<PvPDiscreteAimState>(agent) = {
          .yawVelocity = 0,
          .pitchVelocity = 0,
        };
    }

    if (cfg.task == Task::Explore) {
        return;
    }

    for (CountT i = 0; i < total_num_agents; i++) {
        CountT team_idx = i / cfg.pTeamSize;
        CountT team_offset = i - team_idx * cfg.pTeamSize;

        Entity cur_agent = ctx.data().agents[i];


        CamRef &cam_ref = ctx.get<CamRef>(cur_agent);
        cam_ref.camEntity = ctx.makeEntity<CamEntity>();

        TeamInfo &team_info = ctx.get<TeamInfo>(cur_agent);
        team_info.team = (int32_t)team_idx;
        team_info.offset = (int32_t)team_offset;

        Teammates &teammates = ctx.get<Teammates>(cur_agent);

        for (CountT j = 0; j < consts::maxTeamSize - 1; j++) {
            teammates.e[j] = Entity::none();
        }

        CountT out_idx = 0;
        for (CountT j = 0; j < (CountT)cfg.pTeamSize; j++) {
            if (team_offset == j) {
                continue;
            }

            Entity teammate = ctx.data().agents[j + team_idx * cfg.pTeamSize];
            teammates.e[out_idx++] = teammate;
        }

        Opponents &opponents = ctx.get<Opponents>(cur_agent);
        DamageDealt &dmg_dealt = ctx.get<DamageDealt>(cur_agent);

        for (CountT j = 0; j < consts::maxTeamSize; j++) {
            opponents.e[j] = Entity::none();
            dmg_dealt.dmg[j] = 0.f;
        }
    }

    if (cfg.task == Task::TDM ||
            cfg.task == Task::Zone ||
            cfg.task == Task::ZoneCaptureDefend) {
        for (CountT i = 0; i < total_num_agents; i++) {
            CountT team_idx = i / cfg.pTeamSize;
            CountT opponent_team_idx = team_idx == 0 ? 1 : 0;
    
            Entity cur_agent = ctx.data().agents[i];
            Opponents &opponents = ctx.get<Opponents>(cur_agent);

            for (uint32_t j = 0; j < cfg.pTeamSize; j++) {
                Entity opponent =
                    ctx.data().agents[j + opponent_team_idx * cfg.pTeamSize];
                opponents.e[j] = opponent;
            }
        }
    } else if (cfg.task == Task::Turret) {
        for (CountT i = 0; i < cfg.eTeamSize; i++) {
            Entity turret = ctx.makeEntity<Turret>();
            ctx.get<Scale>(turret) = Diag3x3 { 1, 1, 1 };
            ctx.get<ObjectID>(turret) = ObjectID { 1 };

            ctx.data().turrets[i] = turret;
        }

        for (CountT i = 0; i < total_num_agents; i++) {
            Entity cur_agent = ctx.data().agents[i];

            Opponents &opponents = ctx.get<Opponents>(cur_agent);

            for (uint32_t j = 0; j < cfg.eTeamSize; j++) {
                opponents.e[j] = ctx.data().turrets[j];
            }
        }
    } else {
        assert(false);
    }

    for (i32 i = 0; i < consts::numTeams; i++) {
      Entity full_team = ctx.data().teamInterfaces[i] =
          ctx.makeEntity<FullTeamInterface>();
      ctx.get<FullTeamID>(full_team).id = i;
    }
}

void resetPersistentEntities(Engine &ctx, RandKey episode_rand_key)
{
  Navmesh &navmesh = ctx.singleton<LevelData>().navmesh;

  RNG &base_rng = ctx.data().baseRNG;

  for (int32_t i = 0; i < (int32_t)ctx.data().numAgents; i++) {
    Entity agent_entity = ctx.data().agents[i];
    ctx.get<Position>(agent_entity) = Vector3 {
      .x = FLT_MAX,
      .y = FLT_MAX,
      .z = FLT_MAX,
    };

    CombatState combat_state;
    combat_state.rng = RNG(rand::split_i(episode_rand_key, i + 1));
    combat_state.landedShotOn = Entity::none();
    combat_state.remainingRespawnSteps = 0;
    combat_state.remainingStepsBeforeAutoheal = 0;
    combat_state.successfulKill = false;
    combat_state.wasShotCount = 0;
    combat_state.wasKilled = false;
    combat_state.firedShotT = -FLT_MAX;
    combat_state.hasDiedDuringEpisode = false;

    ctx.get<CombatState>(agent_entity) = std::move(combat_state);

    // Set agents to "dead" so the spawnAgents call below "respawns" them
    ctx.get<Alive>(agent_entity).mask = 0.f;

    auto &last_obs = ctx.get<OpponentLastKnownObservations>(agent_entity);
    auto &last_pos_obs = ctx.get<OpponentLastKnownPositionObservations>(
        agent_entity);
    for (int32_t j = 0; j < consts::maxTeamSize; j++) {
      last_obs.obs[j] = {};
      last_pos_obs.obs[j] = {};
    }

    ctx.get<BreadcrumbAgentState>(agent_entity) = BreadcrumbAgentState {
      .totalPenalty = 0.f,
      .lastBreadcrumb = Entity::none(),
      .stepsSinceLastNewBreadcrumb = 0,
    };
  }

  {
    assert(ctx.singleton<StandardSpawns>().numCommonRespawns <
           SpawnUsageCounter::maxNumSpawns);
    SpawnUsageCounter &spawn_usage = ctx.singleton<SpawnUsageCounter>();
    for (int32_t i = 0; i < SpawnUsageCounter::maxNumSpawns; i++) {
      spawn_usage.initASpawnsLastUsedTick[i] = 0xFFFF'FFFF;
      spawn_usage.initBSpawnsLastUsedTick[i] = 0xFFFF'FFFF;
      spawn_usage.respawnLastUsedTick[i] = 0xFFFF'FFFF;
    }
  }
  spawnAgents(ctx, false);

  for (int32_t i = 0; i < (int32_t)ctx.data().numAgents; i++) {
    Entity agent_entity = ctx.data().agents[i];
    
    Vector3 pos = ctx.get<Position>(agent_entity);
    ctx.get<StartPos>(agent_entity) = pos;

    if (ctx.data().taskType == Task::Explore) {
      ctx.get<ExploreAction>(agent_entity) = {
        .moveAmount = 0,
        .moveAngle = 0,
        .rotate = consts::numTurnBuckets / 2,
        .mantle = 0,
      };
    } else if (ctx.data().taskType == Task::TDM ||
               ctx.data().taskType == Task::Zone ||
               ctx.data().taskType == Task::ZoneCaptureDefend ||
               ctx.data().taskType == Task::Turret) {
      ctx.get<PvPDiscreteAction>(agent_entity) = {
        .moveAmount = 0,
        .moveAngle = 0,
        .fire = 0,
        .stand = 0,
      };

      ctx.get<PvPAimAction>(agent_entity) = {
        .yaw = 0,
        .pitch = 0,
      };

      ctx.get<CoarsePvPAction>(agent_entity) = {
        .moveAmount = 0,
        .moveAngle = 0,
        .facing = 0,
      };
    }

    ExploreTracker &explore_tracker = 
        ctx.get<ExploreTracker>(agent_entity);
    explore_tracker.numNewCellsVisited = 0;

    auto sampleCoef =
      [&base_rng]
    (float a, float b)
    {
      return a + (b - a) * base_rng.sampleUniform();
    };

    ctx.get<RewardHyperParams>(agent_entity) = {
      .teamSpirit   = sampleCoef(0, 1),
      .shotScale    = sampleCoef(0.01, 0.2),
      .exploreScale = sampleCoef(0.0001, 0.005),
      .inZoneScale  = sampleCoef(0.0, 0.05),
      .zoneTeamContestScale = sampleCoef(0.0, 0.01),
      .zoneTeamCtrlScale = sampleCoef(0.0, 0.1),
      .zoneDistScale = sampleCoef(0.0, 0.01),
      .zoneEarnedPointScale = sampleCoef(0.1, 2.0),
      .breadcrumbScale = sampleCoef(0.01, 0.5),
    };

    ctx.get<RewardHyperParams>(agent_entity) = RewardHyperParams {};
  }

  for (int32_t i = 0; i < (int32_t)ctx.data().numTurrets; i++) {
    Entity turret = ctx.data().turrets[i];
    ctx.get<HP>(turret).hp = 100.f;
    ctx.get<Magazine>(turret) = Magazine {
      .numBullets = 30,
      .isReloading = 0,
    };
    ctx.get<Alive>(turret).mask = 1.f;

    ctx.get<TurretState>(turret) = TurretState {
      .rng = RNG(1 + ctx.data().numAgents + i),
      .offset = (int32_t)i,
    };

    float yaw_rnd = base_rng.sampleUniform();

    ctx.get<Position>(turret) = navmesh.samplePoint(base_rng.randKey());

    float yaw = yaw_rnd * 2.f * math::pi;
    ctx.get<Rotation>(turret) = Quat::angleAxis(yaw, math::up).normalize();
    ctx.get<Aim>(turret) = computeAim(yaw, 0.f);
  }

  {
    GoalRegionsState &goal_regions_state = ctx.singleton<GoalRegionsState>();

    int32_t num_goal_regions = ctx.data().numGoalRegions;
    assert(num_goal_regions <= GoalRegionsState::maxRegions);

    for (int32_t i = 0; i < num_goal_regions; i++) {
      goal_regions_state.regionsActive[i] = true;
      goal_regions_state.minDistToRegions[i] = FLT_MAX;
    }

    goal_regions_state.teamStepRewards[0] = 0.f;
    goal_regions_state.teamStepRewards[1] = 0.f;
  }

  for (i32 i = 0; i < consts::numTeams; i++) {
    Entity team_iface = ctx.data().teamInterfaces[i];
    assert(ctx.get<FullTeamID>(team_iface).id == i);

    auto &enemies_last_known =
        ctx.get<FullTeamLastKnownEnemyObservations>(team_iface);
    for (i32 j = 0; j < consts::maxTeamSize; j++) {
      enemies_last_known.obs[j] = {};
    }
  }

  if (ctx.data().trajectoryCurriculum.numSnapshots > 0 &&
      base_rng.sampleUniform() < 0.5f &&
      !ctx.data().trainControl->evalMode) {
    i32 snapshot_idx = base_rng.sampleI32(
        0, ctx.data().trajectoryCurriculum.numSnapshots);

    CurriculumSnapshot snapshot =
        ctx.data().trajectoryCurriculum.snapshots[snapshot_idx];

    ZoneState &zone_state = ctx.singleton<ZoneState>();

    zone_state.curZone = snapshot.matchState.curZone;

    if (snapshot.matchState.curZoneController == -1) {
      zone_state.curControllingTeam = -1;
      zone_state.isCaptured = false;
    } else {
      zone_state.isCaptured = true;
      zone_state.curControllingTeam = snapshot.matchState.curZoneController;

      zone_state.stepsUntilPoint = snapshot.matchState.stepsUntilPoint;
      zone_state.zoneStepsRemaining = snapshot.matchState.zoneStepsRemaining;
    }

    MatchInfo &match_info = ctx.singleton<MatchInfo>();
    i32 team_a = match_info.teamA;

    match_info.curStep = snapshot.matchState.step;

    for (i32 i = 0; i < (i32)ctx.data().numAgents; i++) {
      Entity agent_entity;
      if (team_a == 0) {
        agent_entity = ctx.data().agents[i];
      } else {
        if (i < (i32)ctx.data().numAgents / 2) {
          agent_entity =
              ctx.data().agents[ctx.data().numAgents / 2 + i];
        } else {
          agent_entity =
              ctx.data().agents[i - ctx.data().numAgents / 2];
        }
      }

      PackedPlayerSnapshot player_snapshot = snapshot.players[i];

      ctx.get<Position>(agent_entity) = Vector3 {
        .x = (float)player_snapshot.pos[0],
        .y = (float)player_snapshot.pos[1],
        .z = (float)player_snapshot.pos[2],
      };

      ctx.get<Aim>(agent_entity) = computeAim(
          (float)player_snapshot.yaw * math::pi / 32768.f,
          (float)player_snapshot.pitch * math::pi / 32768.f);

      ctx.get<Rotation>(agent_entity) =
        Quat::angleAxis(ctx.get<Aim>(agent_entity).yaw, math::up).normalize();

      ctx.get<HP>(agent_entity) = {
        .hp = (float)player_snapshot.hp,
      };

      ctx.get<Magazine>(agent_entity) = {
        .numBullets = player_snapshot.magNumBullets,
        .isReloading = player_snapshot.isReloading,
      };

      if ((player_snapshot.flags & (u8)PackedPlayerStateFlags::Crouch) != 0) {
        ctx.get<StandState>(agent_entity) = {
          .curPose = Pose::Crouch,
          .tgtPose = Pose::Crouch,
          .transitionRemaining = 0,
        };
      }

      if ((player_snapshot.flags & (u8)PackedPlayerStateFlags::Prone) != 0) {
        ctx.get<StandState>(agent_entity) = {
          .curPose = Pose::Prone,
          .tgtPose = Pose::Prone,
          .transitionRemaining = 0,
        };
      }
    }
  }
}

}
