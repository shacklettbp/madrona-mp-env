#include <madrona/mw_gpu_entry.hpp>

#include <madrona/geo.hpp>

#include <algorithm>

#include "sim.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "level_gen.hpp"

#ifndef MADRONA_GPU_MODE
#include "dnn.hpp"
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace madronaMPEnv {

static inline void logEvent(Engine &ctx, const GameEvent &event)
{
  if (!ctx.data().eventGlobalState) {
    return;
  }

  ctx.getDirect<GameEvent>(2, ctx.makeTemporary<GameEventEntity>()) = event;

  AtomicU32Ref num_events_atomic(ctx.data().eventGlobalState->numEvents);
  num_events_atomic.fetch_add_relaxed(1);

  AtomicU32Ref event_logged_in_step_atomic(ctx.data().eventLoggedInStep);
  event_logged_in_step_atomic.store<sync::relaxed>(1);

  AtomicU32Ref event_mask_atomic(ctx.data().eventMask);
  event_mask_atomic.fetch_or<sync::relaxed>((u32)event.type);
}

static void writePackedStepSnapshot(Engine &ctx)
{
  u32 cur_step = ctx.singleton<MatchInfo>().curStep;

  u32 num_events = ctx.data().eventLoggedInStep;

#if 0
  if (num_events == 0 && cur_step % 20 != 0) {
    return;
  }
#endif

  u32 event_mask = ctx.data().eventMask;

  ctx.data().eventLoggedInStep = 0;
  ctx.data().eventMask = 0;

  AtomicU32Ref num_match_states_atomic(ctx.data().eventGlobalState->numStepStates);
  num_match_states_atomic.fetch_add_relaxed(1);

  PackedStepSnapshot &state_out =
    ctx.getDirect<PackedStepSnapshot>(2, ctx.makeTemporary<PackedStepSnapshotEntity>());

  PackedMatchState &match_state = state_out.matchState;
  
  state_out.numEvents = num_events;
  state_out.eventMask = event_mask;

  state_out.matchID = ctx.data().matchID;
  match_state.step = (u16)cur_step;

  {
    ZoneState &zone_state = ctx.singleton<ZoneState>();
    match_state.curZone = (u8)zone_state.curZone;
    match_state.curZoneController = (i8)(
        zone_state.isCaptured ? zone_state.curControllingTeam : -1);

    match_state.zoneStepsRemaining = u16(zone_state.zoneStepsRemaining);
    match_state.stepsUntilPoint = u16(zone_state.stepsUntilPoint);
  }

  for (CountT i = 0; i < consts::maxTeamSize * 2; i++) {
    PackedPlayerSnapshot &player_state = state_out.players[i];

    if (i >= (CountT)ctx.data().numAgents) {
      player_state = {};
      break;
    }

    player_state.flags = (u8)PackedPlayerStateFlags::None;

    Entity e = ctx.data().agents[i];

    Vector3 pos = ctx.get<Position>(e);
    player_state.pos[0] = (i16)pos.x;
    player_state.pos[1] = (i16)pos.y;
    player_state.pos[2] = (i16)pos.z;

    Aim aim = ctx.get<Aim>(e);
    player_state.yaw = (i16)(aim.yaw * 32768 / math::pi);
    player_state.pitch = (i16)(aim.pitch * 32768 / math::pi);

    Magazine mag = ctx.get<Magazine>(e);

    player_state.magNumBullets = (u16)mag.numBullets;
    player_state.isReloading = (u8)mag.isReloading;

    const CombatState &combat_state = ctx.get<CombatState>(e);

    if ((u8)(combat_state.landedShotOn != Entity::none())) {
      player_state.flags |= u8(PackedPlayerStateFlags::FiredShot);
    }

    HP hp = ctx.get<HP>(e);

    player_state.hp = (u8)hp.hp;

    StandState stand_state = ctx.get<StandState>(e);
    
    if (stand_state.curPose == Pose::Crouch) {
      player_state.flags |= u8(PackedPlayerStateFlags::Crouch);
    } else if (stand_state.curPose == Pose::Prone) {
      player_state.flags |= u8(PackedPlayerStateFlags::Prone);
    }
  }
}

static void updateFiltersState(Engine &ctx, u32 cur_step)
{
  std::array<AnalyticsFilter, 3> hardcoded_filters;
  hardcoded_filters[0].type = AnalyticsFilterType::PlayerInRegion;
  hardcoded_filters[0].playerInRegion = {
    .region = {
      .min = { -1272, -866 },
      .max = { -825, 696 },
    },
    .minNumInRegion = 5,
  };

  hardcoded_filters[1].type = AnalyticsFilterType::PlayerInRegion;
  hardcoded_filters[1].playerInRegion = {
    .region = {
      .min = { 852, -851 },
      .max = { 1280, 593 },
    },
    .minNumInRegion = 1,
  };

  hardcoded_filters[2].type = AnalyticsFilterType::PlayerShotEvent;
  hardcoded_filters[2].playerShotEvent = {};

  constexpr int filter_match_window = 0;

  for (int filter_idx = 0;
       filter_idx < (int)hardcoded_filters.size();
       filter_idx += 1) {
    AnalyticsFilter &filter = hardcoded_filters[filter_idx];

    for (int team_idx = 0; team_idx < 2; team_idx++) {
      FiltersMatchState &filter_state = ctx.data().filtersState[team_idx];
      if ((filter_state.active & (1 << filter_idx)) != 0) {
        if (cur_step - filter_state.lastMatches[filter_idx] >
            filter_match_window) {
          filter_state.active &= ~(1 << filter_idx);
        }
      }
    }

    switch (filter.type) {
    case AnalyticsFilterType::CaptureEvent: {
      assert(false);
    } break;
    case AnalyticsFilterType::ReloadEvent: {
      assert(false);
    } break;
    case AnalyticsFilterType::KillEvent: {
      auto &kill_filter = filter.killEvent;

      for (int player_idx = 0;
           player_idx < (int)ctx.data().numAgents;
           player_idx++) {
        Entity agent = ctx.data().agents[player_idx];
        int team = player_idx / ctx.data().pTeamSize;

        Vector3 attacker_pos = ctx.get<Position>(agent);
        const CombatState &combat_state = ctx.get<CombatState>(agent);

        if (!combat_state.successfulKill) {
          continue;
        }

        Entity killed = combat_state.landedShotOn;
        Vector3 target_pos = ctx.get<Position>(killed);

        if (attacker_pos.x < kill_filter.killerRegion.min.x ||
            attacker_pos.y < kill_filter.killerRegion.min.y ||
            attacker_pos.x > kill_filter.killerRegion.max.x ||
            attacker_pos.y > kill_filter.killerRegion.max.y ||
            target_pos.x < kill_filter.killedRegion.min.x ||
            target_pos.y < kill_filter.killedRegion.min.y ||
            target_pos.x > kill_filter.killedRegion.max.x ||
            target_pos.y > kill_filter.killedRegion.max.y) {
          continue;
        }

        FiltersMatchState &filter_state = ctx.data().filtersState[team];
        filter_state.active |= 1 << filter_idx;
        filter_state.lastMatches[filter_idx] = cur_step;
      }
    } break;
    case AnalyticsFilterType::PlayerShotEvent: {
      auto &shot_filter = filter.playerShotEvent;

      for (int player_idx = 0;
           player_idx < (int)ctx.data().numAgents;
           player_idx++) {
        Entity agent = ctx.data().agents[player_idx];
        int team = player_idx / ctx.data().pTeamSize;

        Vector3 attacker_pos = ctx.get<Position>(agent);
        const CombatState &combat_state = ctx.get<CombatState>(agent);

        if (combat_state.landedShotOn == Entity::none()) {
          continue;
        }

        Entity target = combat_state.landedShotOn;
        Vector3 target_pos = ctx.get<Position>(target);

        if (attacker_pos.x < shot_filter.attackerRegion.min.x ||
            attacker_pos.y < shot_filter.attackerRegion.min.y ||
            attacker_pos.x > shot_filter.attackerRegion.max.x ||
            attacker_pos.y > shot_filter.attackerRegion.max.y ||
            target_pos.x < shot_filter.targetRegion.min.x ||
            target_pos.y < shot_filter.targetRegion.min.y ||
            target_pos.x > shot_filter.targetRegion.max.x ||
            target_pos.y > shot_filter.targetRegion.max.y) {
          continue;
        }

        FiltersMatchState &filter_state = ctx.data().filtersState[team];
        filter_state.active |= 1 << filter_idx;
        filter_state.lastMatches[filter_idx] = cur_step;
      }
    } break;
    case AnalyticsFilterType::PlayerInRegion: {
      auto &in_region_filter = filter.playerInRegion;

      i32 num_in_region[2];
      num_in_region[0] = 0;
      num_in_region[1] = 0;

      for (int player_idx = 0;
           player_idx < (int)ctx.data().numAgents;
           player_idx++) {
        Entity agent = ctx.data().agents[player_idx];
        int team = player_idx / ctx.data().pTeamSize;

        Vector3 pos = ctx.get<Position>(agent);

        if (pos.x < in_region_filter.region.min.x ||
            pos.y < in_region_filter.region.min.y ||
            pos.x > in_region_filter.region.max.x ||
            pos.y > in_region_filter.region.max.y) {
          continue;
        }

        num_in_region[team] += 1;
      }

      for (int team = 0; team < 2; team++) {
        if (num_in_region[team] >= in_region_filter.minNumInRegion) {
          FiltersMatchState &filter_state = ctx.data().filtersState[team];
          filter_state.active |= 1 << filter_idx;
          filter_state.lastMatches[filter_idx] = cur_step;
        }
      }
    } break;
    default: MADRONA_UNREACHABLE(); break;
    }
  }

  for (int team = 0; team < 2; team++) {
    int num_active_filters = 
        std::popcount(ctx.data().filtersState[team].active);

    if (num_active_filters == hardcoded_filters.size()) {
      ctx.data().filtersLastMatchedStep[team] = cur_step;
    }
  }
}

static inline constexpr float discreteTurnDelta()
{
    constexpr float turn_max = 10;

    return turn_max / f32(CountT(consts::numTurnBuckets / 2));
}

#if 0
static float closestPointSegmentSegment(Vector3 p1, Vector3 q1,
                                        Vector3 p2, Vector3 q2,
                                        Vector3 *c1, Vector3 *c2)
{
    Vector3 d1 = q1 - p1;
    Vector3 d2 = q2 - p2;
    Vector3 r = p1 - p2;

    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);

    if (a <= 1e-5f && e <= 1e-5f) {
        *c1 = p1;
        *c2 = p2;

        return dot(*c1 - *c2, *c1 - *c2);
    }

    float s, t;
    if (a <= 1e-5f) {
        s = 0.f;
        t = f / e;
        t = std::clamp(t, 0.f, 1.f);
    } else {
        float c = dot(d1, r);
        if (e <= 1e-5f) {
            t = 0.f;
            s = std::clamp(c / a, 0.f, 1.f);
        } else {
            float b = dot(d1, d2);
            float denom = a * e - b * b;

            if (denom != 0.f) {
                s = std::clamp((b * f - c * e) / denom, 0.f, 1.f);
            } else {
                s = 0.f;
            }

            t = (b * s + f) / e;

            if (t < 0.f) {
                t = 0.f;
                s = std::clamp(-c / a, 0.f, 1.f);
            } else if (t > 1.f) {
                t = 1.f;
                s = std::clamp((b - c) / a, 0.f, 1.f);
            }
        }
    }

    *c1 = p1 + d1 * s;
    *c2 = p2 + d2 * t;

    return dot(*c1 - *c2, *c1 - *c2);
}

static bool projectOnlyWithinTri(Vector3 a, Vector3 b, Vector3 c,
                                 Vector3 p, Vector3 *out)
{
    Vector3 u = b - a;
    Vector3 v = c - a;
    Vector3 n = cross(u, v);
    float n_len_sq = dot(n, n);
    if (n_len_sq == 0.f) {
        return false;
    }

    Vector3 w = p - a;
    float gamma = dot(cross(u, w), n) / n_len_sq;
    float beta = dot(cross(w, v), n) / n_len_sq;
    float alpha = 1.f - gamma - beta;

    if (alpha < 0.f || beta < 0.f || gamma < 0.f &&
            alpha > 1.f || beta > 1.f || gamma > 1.f) {
        return false;
    }

    *out = alpha * a + beta * b + gamma * c;
    return true;
}

static bool capsuleTriCollision(Vector3 a, Vector3 b, Vector3 c,
                                Vector3 p, Vector3 q,
                                Vector3 *correction)
{
    float min_sep_sq = FLT_MAX;
    Vector3 closest_midline, closest_tri;
    Vector3 test_closest_midline, test_closest_tri;

    float sep_sq = closestPointSegmentSegment(
        p, q, a, b, &test_closest_midline, &test_closest_tri);

    // Edge tests
    if (sep_sq < min_sep_sq) {
        min_sep_sq = sep_sq;
        closest_midline = test_closest_midline;
        closest_tri = test_closest_tri;
    }

    sep_sq = closestPointSegmentSegment(
        p, q, b, c, &test_closest_midline, &test_closest_tri);

    if (sep_sq < min_sep_sq) {
        min_sep_sq = sep_sq;
        closest_midline = test_closest_midline;
        closest_tri = test_closest_tri;
    }

    sep_sq = closestPointSegmentSegment(
        p, q, a, a, &test_closest_midline, &test_closest_tri);

    if (sep_sq < min_sep_sq) {
        min_sep_sq = sep_sq;
        closest_midline = test_closest_midline;
        closest_tri = test_closest_tri;
    }

    Vector3 p_proj;
    bool p_proj_within = projectOnlyWithinTri(a, b, c, p, &p_proj);
    if (p_proj_within) {
        sep_sq = p.distance2(p_proj);
        if (sep_sq < min_sep_sq) {
            min_sep_sq = sep_sq;
            closest_midline = p;
            closest_tri = p_proj;
        }
    }

    Vector3 q_proj;
    bool q_proj_within = projectOnlyWithinTri(a, b, c, q, &q_proj);
    if (q_proj_within) {
        sep_sq = q.distance2(q_proj);
        if (sep_sq < min_sep_sq) {
            min_sep_sq = sep_sq;
            closest_midline = q;
            closest_tri = q_proj;
        }
    }

    if (min_sep_sq > consts::agentRadius * consts::agentRadius) {
        return false;
    }

    if (min_sep_sq == 0.f) {
        Vector3 fallback = (a + b + c) / 3.f - p;
        float fallback_dist2 = fallback.length2();
        if (fallback_dist2 == 0.f) {
            fallback = Vector3 { 0.f, 0.f, consts::agentRadius };
        } else {
            fallback *= rsqrtApprox(fallback_dist2) * consts::agentRadius;
        }

        *correction = fallback;
        return true;
    }

    float len = sqrtf(min_sep_sq);

#if 0
    printf("\n(%f %f %f) (%f %f %f) (%f %f %f) (%f %f %f) (%f %f %f) %f %f\n\n",
           a.x, a.y, a.z,
           b.x, b.y, b.z,
           c.x, c.y, c.z,
           closest_midline.x,
           closest_midline.y,
           closest_midline.z,
           closest_tri.x,
           closest_tri.y,
           closest_tri.z,
           min_sep_sq,
           len);
#endif

    *correction = (closest_midline - closest_tri) * (1.f / len) * (consts::agentRadius - len);
    return true;
}
#endif


// Register all the ECS components and archetypes that will be
// use in the simulation
void Sim::registerTypes(ECSRegistry &registry,
                        const TaskConfig &cfg)
{
    base::registerTypes(registry);
    phys::PhysicsSystem::registerTypes(registry);

    if (cfg.viz) {
#ifndef MADRONA_GPU_MODE
      VizSystem::registerTypes(registry);
#endif
    } else {
      registry.registerComponent<VizCamera>(); // hack when viz is disabled
    }

    registry.registerComponent<SelfObservation>();
    registry.registerComponent<SelfPositionObservation>();

    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<AgentPolicy>();

    registry.registerComponent<AgentVelocity>();
    registry.registerComponent<IntermediateMoveState>();

    registry.registerComponent<FwdLidar>();
    registry.registerComponent<RearLidar>();
    registry.registerComponent<AgentMap>();
    registry.registerComponent<UnmaskedAgentMap>();

    registry.registerComponent<StandState>();

    registry.registerComponent<BreadcrumbAgentState>();
    registry.registerComponent<Breadcrumb>();
    registry.registerArchetype<BreadcrumbEntity>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<WorldCurriculum>();

    registry.registerSingleton<TeamRewardState>();
    registry.registerSingleton<MatchResult>();
    registry.registerSingleton<MatchInfo>();
    registry.registerSingleton<LevelData>();
    registry.registerSingleton<StandardSpawns>();
    registry.registerSingleton<SpawnCurriculum>();
    registry.registerSingleton<CurriculumState>();
    registry.registerSingleton<SpawnUsageCounter>();
    registry.registerSingleton<GoalRegionsState>();

    registry.registerArchetype<StaticGeometry>();

    registry.registerComponent<GameEvent>();
    registry.registerArchetype<GameEventEntity>();

    registry.registerComponent<PackedStepSnapshot>();
    registry.registerArchetype<PackedStepSnapshotEntity>();

    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);

    registry.exportSingleton<WorldCurriculum>(
        (uint32_t)ExportID::WorldCurriculum);

    registry.exportSingleton<MatchResult>(
        (uint32_t)ExportID::MatchResult);

    registry.registerComponent<StartPos>();
    registry.registerComponent<ExploreTracker>();

    if (cfg.task == Task::TDM ||
        cfg.task == Task::Zone ||
        cfg.task == Task::ZoneCaptureDefend ||
        cfg.task == Task::Turret) {
        registry.registerComponent<HP>();
        registry.registerComponent<Magazine>();
        registry.registerComponent<Aim>();
        registry.registerComponent<CamRef>();
        registry.registerComponent<DamageDealt>();
        registry.registerComponent<Alive>();
        registry.registerComponent<TeamInfo>();
        registry.registerComponent<CombatState>();
        registry.registerComponent<ShotVizState>();
        registry.registerComponent<ShotVizRemaining>();

        registry.registerComponent<Teammates>();
        registry.registerComponent<Opponents>();
        registry.registerComponent<OpponentsVisibility>();
        registry.registerComponent<PvPDiscreteAction>();
        registry.registerComponent<PvPAimAction>();
        registry.registerComponent<PvPDiscreteAimAction>();
        registry.registerComponent<PvPDiscreteAimState>();
        registry.registerComponent<CoarsePvPAction>();
        registry.registerComponent<HardcodedBotAction>();

        registry.registerComponent<TeammateObservations>();
        registry.registerComponent<OpponentObservations>();
        registry.registerComponent<OpponentLastKnownObservations>();

        registry.registerComponent<TeammatePositionObservations>();
        registry.registerComponent<OpponentPositionObservations>();
        registry.registerComponent<OpponentLastKnownPositionObservations>();

        registry.registerComponent<OpponentMasks>();
        registry.registerSingleton<ZoneState>();

        registry.registerComponent<FiltersStateObservation>();
        registry.registerComponent<RewardHyperParams>();

        registry.registerArchetype<CamEntity>();
        registry.registerArchetype<PvPAgent>();
        registry.registerArchetype<ShotViz>();
        registry.registerArchetype<ZoneViz>();

        if (cfg.highlevelMove) {
            registry.exportColumn<PvPAgent, CoarsePvPAction>(
                (uint32_t)ExportID::PvPDiscreteAction);
        } else {
            registry.exportColumn<PvPAgent, PvPDiscreteAction>(
                (uint32_t)ExportID::PvPDiscreteAction);
            registry.exportColumn<PvPAgent, PvPAimAction>(
                (uint32_t)ExportID::PvPAimAction);

            registry.exportColumn<PvPAgent, PvPDiscreteAimAction>(
                (uint32_t)ExportID::PvPDiscreteAimAction);
        }

        registry.exportColumn<PvPAgent, SelfObservation>(
            (uint32_t)ExportID::SelfObservation);
        registry.exportColumn<PvPAgent, SelfPositionObservation>(
            (uint32_t)ExportID::SelfPositionObservation);

        registry.exportColumn<PvPAgent, TeammateObservations>(
            (uint32_t)ExportID::TeammateObservations);
        registry.exportColumn<PvPAgent, TeammatePositionObservations>(
            (uint32_t)ExportID::TeammatePositionObservations);

        registry.exportColumn<PvPAgent, OpponentObservations>(
            (uint32_t)ExportID::OpponentObservations);
        registry.exportColumn<PvPAgent, OpponentPositionObservations>(
            (uint32_t)ExportID::OpponentPositionObservations);

        registry.exportColumn<PvPAgent, OpponentLastKnownObservations>(
            (uint32_t)ExportID::OpponentLastKnownObservations);
        registry.exportColumn<PvPAgent, OpponentLastKnownPositionObservations>(
            (uint32_t)ExportID::OpponentLastKnownPositionObservations);

        registry.exportColumn<PvPAgent, OpponentMasks>(
            (uint32_t)ExportID::OpponentMasks);

        registry.exportColumn<PvPAgent, FwdLidar>(
            (uint32_t)ExportID::FwdLidar);
        registry.exportColumn<PvPAgent, RearLidar>(
            (uint32_t)ExportID::RearLidar);
        registry.exportColumn<PvPAgent, Reward>(
            (uint32_t)ExportID::Reward);
        registry.exportColumn<PvPAgent, RewardHyperParams>(
            (uint32_t)ExportID::RewardHyperParams);
        registry.exportColumn<PvPAgent, Done>(
            (uint32_t)ExportID::Done);
        registry.exportColumn<PvPAgent, AgentPolicy>(
            (uint32_t)ExportID::AgentPolicy);

        registry.exportColumn<PvPAgent, AgentMap>(
            (uint32_t)ExportID::AgentMap);
        registry.exportColumn<PvPAgent, UnmaskedAgentMap>(
            (uint32_t)ExportID::UnmaskedAgentMap);

        registry.exportColumn<PvPAgent, HP>(
            (uint32_t)ExportID::HP);
        registry.exportColumn<PvPAgent, Magazine>(
            (uint32_t)ExportID::Magazine);
        registry.exportColumn<PvPAgent, Alive>(
            (uint32_t)ExportID::Alive);

        registry.exportColumn<PvPAgent, FiltersStateObservation>(
            (u32)ExportID::FiltersStateObservation);
    } else if (cfg.task == Task::Explore) {
        registry.registerComponent<ExploreAction>();

        registry.registerArchetype<ExploreAgent>();

        registry.exportColumn<ExploreAgent, ExploreAction>(
            (uint32_t)ExportID::ExploreAction);
        registry.exportColumn<ExploreAgent, SelfObservation>(
            (uint32_t)ExportID::SelfObservation);
        registry.exportColumn<ExploreAgent, SelfPositionObservation>(
            (uint32_t)ExportID::SelfPositionObservation);
        registry.exportColumn<ExploreAgent, FwdLidar>(
            (uint32_t)ExportID::FwdLidar);
        registry.exportColumn<ExploreAgent, Reward>(
            (uint32_t)ExportID::Reward);
        registry.exportColumn<ExploreAgent, Done>(
            (uint32_t)ExportID::Done);
        registry.exportColumn<ExploreAgent, AgentPolicy>(
            (uint32_t)ExportID::AgentPolicy);
    }

    if (cfg.task == Task::Turret) {
        registry.registerComponent<TurretState>();
        registry.registerArchetype<Turret>();
    }

    registry.registerComponent<FullTeamID>();
    registry.registerComponent<FullTeamActions>();
    registry.registerComponent<FullTeamGlobalObservation>();
    registry.registerComponent<FullTeamPlayerObservations>();
    registry.registerComponent<FullTeamEnemyObservations>();
    registry.registerComponent<FullTeamLastKnownEnemyObservations>();
    registry.registerComponent<FullTeamFwdLidar>();
    registry.registerComponent<FullTeamRearLidar>();


    registry.registerComponent<FullTeamReward>();
    registry.registerComponent<FullTeamDone>();
    registry.registerComponent<FullTeamPolicy>();

    registry.registerArchetype<FullTeamInterface>();

    registry.exportColumn<FullTeamInterface, FullTeamActions>(
        ExportID::FullTeamActions);
    registry.exportColumn<FullTeamInterface, FullTeamGlobalObservation>(
        ExportID::FullTeamGlobal);
    registry.exportColumn<FullTeamInterface, FullTeamPlayerObservations>(
        ExportID::FullTeamPlayers);
    registry.exportColumn<FullTeamInterface, FullTeamEnemyObservations>(
        ExportID::FullTeamEnemies);
    registry.exportColumn<FullTeamInterface, FullTeamLastKnownEnemyObservations>(
        ExportID::FullTeamLastKnownEnemies);
    registry.exportColumn<FullTeamInterface, FullTeamFwdLidar>(
        ExportID::FullTeamFwdLidar);
    registry.exportColumn<FullTeamInterface, FullTeamRearLidar>(
        ExportID::FullTeamRearLidar);

    registry.exportColumn<FullTeamInterface, FullTeamReward>(
        ExportID::FullTeamReward);
    registry.exportColumn<FullTeamInterface, FullTeamDone>(
        ExportID::FullTeamDone);
    registry.exportColumn<FullTeamInterface, FullTeamPolicy>(
        ExportID::FullTeamPolicyAssignments);

    registry.exportColumn<GameEventEntity, GameEvent>(
        ExportID::EventLog);

    registry.exportColumn<PackedStepSnapshotEntity, PackedStepSnapshot>(
        ExportID::PackedStepSnapshot);
}

static inline void initWorld(Engine &ctx, bool triggered_reset)
{
    ctx.data().episodeCurriculum = ctx.singleton<WorldCurriculum>();

    ctx.data().matchID = 
      (((u64)ctx.worldID().idx) << 32) |
      ((u64)ctx.data().curEpisodeIdx);

    const TrainControl &train_ctrl = *ctx.data().trainControl;
    MatchInfo &match_info = ctx.singleton<MatchInfo>();

    RandKey episode_rand_key = rand::split_i(
        ctx.data().initRandKey,
        ctx.data().curEpisodeIdx,
        (uint32_t)ctx.worldID().idx);

    RNG &base_rng = ctx.data().baseRNG =
        RNG(rand::split_i(episode_rand_key, 0));

    bool flip_teams = false;

    if (train_ctrl.randomizeTeamSides) {
        flip_teams = base_rng.sampleUniform() < 0.5f;
    } 

    if (flip_teams) {
        match_info.teamA = 1;
    } else {
        match_info.teamA = 0;
    }
    if (triggered_reset && train_ctrl.randomizeEpisodeLengthAfterReset) {
      match_info.curStep = base_rng.sampleI32(0, consts::episodeLen - 1);
    } else {
      match_info.curStep = 0;
    }
    match_info.isFinished = false;

    auto &curriculum_state = ctx.singleton<CurriculumState>();
    match_info.enableSpawnCurriculum =
        base_rng.sampleUniform() < curriculum_state.useCurriculumSpawnProb;

    float tier_cdf[SpawnCurriculum::numCurriculumTiers];
    float running_prob_sum = 0.f;
    for (uint32_t i = 0; i < SpawnCurriculum::numCurriculumTiers; i++) {
        running_prob_sum += curriculum_state.tierProbabilities[i];
        tier_cdf[i] = running_prob_sum;
    }

    float tier_selector_rnd = running_prob_sum * base_rng.sampleUniform();
    for (uint32_t i = 0; i < SpawnCurriculum::numCurriculumTiers; i++) {
        if (tier_selector_rnd < tier_cdf[i]) {
            match_info.curCurriculumTier = i;
            break;
        }
    }

    auto &spawn_curriculum = ctx.singleton<SpawnCurriculum>();
    auto &curriculum_tier =
        spawn_curriculum.tiers[match_info.curCurriculumTier];

    match_info.curCurriculumSpawnIdx =
        (uint32_t)base_rng.sampleI32(0, curriculum_tier.numSpawns);

    if ((ctx.data().simFlags & SimFlags::HardcodedSpawns) ==
        SimFlags::HardcodedSpawns) {
        ctx.data().hardcodedSpawnIdx = base_rng.sampleI32(0, 4);
        ctx.data().hardcodedSpawnIdx = 0;
    }

    if (ctx.data().taskType == Task::Zone ||
        ctx.data().taskType == Task::ZoneCaptureDefend) {
        ZoneState &zone_state = ctx.singleton<ZoneState>();
        zone_state.curZone = base_rng.sampleI32(
            0, ctx.data().zones.numZones);
        zone_state.curControllingTeam = -1;
        zone_state.isContested = false;
        zone_state.isCaptured = false;
        zone_state.earnedPoint = false;

        zone_state.zoneStepsRemaining = consts::numStepsPerZone;
        zone_state.stepsUntilPoint = consts::zonePointInterval;
    }

    if (ctx.data().taskType == Task::ZoneCaptureDefend) {
        ZoneState &zone_state = ctx.singleton<ZoneState>();
        zone_state.curZone = 3;
    }

    resetPersistentEntities(ctx, episode_rand_key);

    for (int team = 0; team < 2; team++) {
      ctx.data().filtersState[team].active = 0;
      ctx.data().filtersLastMatchedStep[team] = 0;
    }
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    int32_t force_reset = reset.reset;
    int32_t should_reset = force_reset;

    MatchInfo &match_info = ctx.singleton<MatchInfo>();

    if (ctx.data().autoReset) {
        if (match_info.isFinished) {
            should_reset = 1;
        }
    }

    if (should_reset != 0) {
        reset.reset = 0;

        ctx.data().curEpisodeIdx = ctx.data().worldEpisodeCounter++;

        if (ctx.data().curEpisodeIdx < 50) {
          if (ctx.data().baseRNG.sampleUniform() <
              (ctx.data().curEpisodeIdx + 1) / (float)50) {
            ctx.singleton<WorldCurriculum>() = WorldCurriculum::FullMatch;
          } else {
            ctx.singleton<WorldCurriculum>() = WorldCurriculum::LearnShooting;
          }
        } else {
          ctx.singleton<WorldCurriculum>() = WorldCurriculum::FullMatch;
        }

        initWorld(ctx, force_reset == 1);
    } 
}

static Vector3 Rotate2D(Vector3 dir, float radians)
{
  float c = cosf(radians);
  float s = sinf(radians);
  return Vector3(c * dir.x - s * dir.y, s * dir.x + c * dir.y, 0);
}

// Provide forces for entities which are controlled by actions (the two agents in this case).
inline void exploreMovementSystem(Engine &,
                                  ExploreAction &, 
                                  Rotation &,
                                  Velocity &)
{
}

inline void applyVelocitySystem(Engine &ctx,
                                const Position &pos,
                                const AgentVelocity &vel,
                                StandState &stand,
                                IntermediateMoveState &move_state)
{
  Vector3 x = pos;
  Vector3 v = vel;
  v.z = 0;

  move_state.newPosition = x;
  move_state.newVelocity = Vector3::zero();

  float v_len = v.length();
  if (v_len == 0.f) {
    return;
  }
  Vector3 v_norm = v / v_len;
  float move_dist = v_len * consts::deltaT;

  // Cast a ray at the minimum step-up, and the max height.
  // If we're prone, there is no second cast.
  const float spherecast_buffer = 0.05f * consts::agentRadius;
  const float spherecast_r = consts::agentRadius;
  float top_of_capsule = consts::standHeight - spherecast_r;
  float low_check_height = consts::proneHeight;
  if (stand.curPose == Pose::Crouch) {
    top_of_capsule = consts::crouchHeight - spherecast_r;
  }
  else if (stand.curPose == Pose::Prone) {
    top_of_capsule = low_check_height;
    low_check_height = consts::proneHeight - spherecast_r + spherecast_buffer;
  }

  // Unfortunately we need a downward ray to find the slope of the ground we're on.
  // To prevent us from sliding up steep slopes.
  Vector3 ray_o = x;
  ray_o.z += top_of_capsule;
  Vector3 normal;
  Entity hit_entity;
  sphereCastWorld(ctx, ray_o, -math::up, spherecast_r, &hit_entity, normal);
  if ( normal.z > 0.0f && normal.z < 0.7 && math::dot(normal, v_norm) < 0.0f) {
    return;
  }

  // Do the two primary raycasts.
  ray_o = x + v_norm * spherecast_buffer * 0.5f;
  ray_o.z += low_check_height;
  float low_dist = sphereCastWorld(ctx, ray_o, v_norm, spherecast_r, &hit_entity, normal);
  float high_dist = low_dist;
  bool high_hit = false;
  if (stand.curPose != Pose::Prone) {
    ray_o.z = x.z + top_of_capsule;
    Vector3 high_normal;
    high_dist = sphereCastWorld(ctx, ray_o, v_norm, spherecast_r, &hit_entity, high_normal);
    if (high_dist < low_dist) {
      low_dist = high_dist;
      normal = high_normal;
      high_hit = true;
    }
  }
  // If we get literal zeros from any raycasts, it means we're stuck in something and we'll have to unstick.
  bool stuck = low_dist == 0.0f || high_dist == 0.0f;
  low_dist = fmaxf(0.0f, low_dist - spherecast_buffer);
  high_dist = fmaxf(0.0f, high_dist - spherecast_buffer);
  Vector3 hit_pos = x + v_norm * fminf(low_dist, move_dist);

  // Enable one step of wall-sliding by doing a secondary cast along the normal of the hit surface.
  // Fire a ray parallel to the hit normal on the X/Y plane.
  if (move_dist > low_dist) {
    Vector3 slide_dir = math::normalize(math::cross(math::up, normal));
    if (math::dot(slide_dir, v_norm) < 0) {
      slide_dir = -slide_dir;
    }
    ray_o = x + v_norm * low_dist;
    ray_o.z += high_hit ? top_of_capsule : low_check_height;
    float slide_dist = sphereCastWorld(ctx, ray_o, slide_dir, spherecast_r, &hit_entity);
    slide_dist = fmaxf(0.0f, slide_dist - spherecast_buffer);
    float max_move = move_dist - low_dist;
    slide_dist = fminf(slide_dist, max_move);
    if (slide_dist > 0.0f)
    {
      hit_pos = hit_pos + slide_dir * slide_dist;
    }
  }

  // Find the ground under where we ended up.
  Vector3 ground_check_pos = hit_pos;
  ground_check_pos.z += top_of_capsule;
  float ground_dist = sphereCastWorld(ctx, ground_check_pos, -math::up, spherecast_r, &hit_entity);
  if (ground_dist == FLT_MAX) {
    return;
  }

  // Uh oh, we're in something.
  if (ground_dist <= 0.0f || stuck) {
    // Try raycasting from all directions to find a way out.
    float furthest_hit = 0.0f;
    int best_dir = -1;
    for (int dir = 0; dir < 4; dir++)
    {
      Vector3 dir_vec = Rotate2D(v_norm, dir * 3.14159f * 0.5f);
      ray_o = x - dir_vec * spherecast_r * 2.0f;
      ray_o.z += low_check_height;
      float hit_dist = sphereCastWorld(ctx, ray_o, dir_vec, spherecast_r, &hit_entity);
      if (hit_dist > furthest_hit)
      {
        furthest_hit = hit_dist;
        best_dir = dir;
      }
    }
    // If we found a direction to teleport, do that, but redo the ground check.
    if (best_dir != -1)
    {
      Vector3 dir_vec = Rotate2D(v_norm, best_dir * 3.14159f * 0.5f);
      hit_pos = x + dir_vec * (fminf(furthest_hit - spherecast_r * 2.0f, -spherecast_buffer));
      ground_check_pos = hit_pos;
      ground_check_pos.z += top_of_capsule;
      ground_dist = sphereCastWorld(ctx, ground_check_pos, -math::up, spherecast_r, &hit_entity);
      if (ground_dist == FLT_MAX) {
        return;
      }
    }
  }

  // Move to the new location.
  // Don't drop to the ground, the fall system will take care of that.
  float fall_dist = fminf(ground_dist, top_of_capsule) + spherecast_r;
  Vector3 new_pos = ground_check_pos;
  new_pos.z -= fall_dist;

  Vector3 to_new_pos = new_pos - x;
  float to_new_dist = to_new_pos.length();
  if (to_new_dist == 0.f) {
    return;
  }

  move_state.newPosition = new_pos;
  move_state.newVelocity = to_new_pos / consts::deltaT;
}

inline void updateMoveStateSystem(
    Engine &ctx,
    Position &pos,
    AgentVelocity &vel,
    const IntermediateMoveState &move_state)
{
  (void)ctx;
  pos = move_state.newPosition;
  vel = move_state.newVelocity;
}

inline void fallSystem(Engine &ctx,
                       const Position &pos,
                       IntermediateMoveState &move_state,
                       const Alive alive)
{
    if (alive.mask == 0.f) {
        move_state.newPosition = pos;
        return;
    }

    const float FALL_RATE = 386.08858267717f;

    const float cast_offset = consts::agentRadius;

    Vector3 ray_o = pos;
    ray_o.z += consts::agentRadius + cast_offset;

    Entity hit_entity;

    float ground_dist = sphereCastWorld(
        ctx, ray_o, -math::up, consts::agentRadius, &hit_entity);

    if (ground_dist == FLT_MAX || ground_dist < cast_offset) {
        move_state.newPosition = pos;
        return;
    }

    float fall_dist = fminf(ground_dist - cast_offset, FALL_RATE * consts::deltaT);

    Vector3 new_pos = pos;
    new_pos.z -= fall_dist;
    move_state.newPosition = new_pos;

    (void)ctx;
#if 0
    Vector3 o = pos;
    o.z += consts::standHeight;
    float hit_dist;
    bool hit = traceRayAgainstWorld(ctx, o, -math::up, &hit_dist);

    printf("(%f %f %f): %d %f\n", o.x, o.y, o.z, hit, hit_dist);

    if (hit) {
        pos.z = o.z - hit_dist + 0.5f;
    }
#endif

#if 0
    printf("Pos: %f %f %f %u %f\n",
           pos.x, pos.y, pos.z, (uint32_t)hit, hit_dist);
#endif

    //(void)pos;
    //vel.z -= 386.08858267717f * consts::deltaT;
}

inline void updateMoveStatePostFallSystem(
    Engine &ctx,
    Position &pos,
    const IntermediateMoveState &move_state)
{
  (void)ctx;
  pos = move_state.newPosition;
}

#if 0
inline void exploreWorldCollisionSystem(Engine &ctx,
                                        Position &pos,
                                        ExploreAction &)
{
    Vector3 agent_pos = pos;

    AABB expanded_agent_aabb {
        .pMin = agent_pos + Vector3 {
            -2.f * consts::agentRadius,
            -2.f * consts::agentRadius,
            -consts::agentRadius,
        },
        .pMax = agent_pos + Vector3 {
            2.f * consts::agentRadius,
            2.f * consts::agentRadius,
            consts::standHeight + consts::agentRadius,
        },
    };

    MeshBVH &world_geo_bvh = ctx.data().staticMeshes[0];

    const Vector3 capsule_bottom = Vector3 { 0, 0, consts::agentRadius };
    const Vector3 capsule_top =
        Vector3 { 0, 0, consts::standHeight - consts::agentRadius };

    const Vector3 capsule_segment = capsule_top - capsule_bottom;

    world_geo_bvh.findOverlaps(expanded_agent_aabb, [&](
            Vector3 tri_a, Vector3 tri_b, Vector3 tri_c) {
        Vector3 a = tri_a - agent_pos;
        Vector3 b = tri_b - agent_pos;
        Vector3 c = tri_c - agent_pos;

        AABB local_tri_aabb = AABB::point(a);
        local_tri_aabb.expand(b);
        local_tri_aabb.expand(c);

        AABB tight_local_agent_aabb {
            .pMin = {
                -consts::agentRadius,
                -consts::agentRadius,
                0.f,
            },
            .pMax = {
                consts::agentRadius,
                consts::agentRadius,
                consts::standHeight,
            }
        };

        if (!tight_local_agent_aabb.overlaps(local_tri_aabb)) {
            return;
        }

#if 0
        printf("(%f %f %f) (%f %f %f)\n",
               local_tri_aabb.pMin.x,
               local_tri_aabb.pMin.y,
               local_tri_aabb.pMin.z,
               local_tri_aabb.pMax.x,
               local_tri_aabb.pMax.y,
               local_tri_aabb.pMax.z);

        printf("(%f %f %f) (%f %f %f)\n",
               tight_local_agent_aabb.pMin.x,
               tight_local_agent_aabb.pMin.y,
               tight_local_agent_aabb.pMin.z,
               tight_local_agent_aabb.pMax.x,
               tight_local_agent_aabb.pMax.y,
               tight_local_agent_aabb.pMax.z);

        printf("Checking (%f %f %f) (%f %f %f) (%f %f %f)\n",
               a.x,
               a.y,
               a.z,
               b.x,
               b.y,
               b.z,
               c.x,
               c.y,
               c.z);
#endif

        Vector3 correction;
        if (capsuleTriCollision(a, b, c, capsule_bottom, capsule_top,
                                &correction)) {
            agent_pos += correction;
        }
    });

    pos = agent_pos;
}
#endif

#if 0
inline void pvpWorldCollisionSystem(Engine &ctx,
                                    Position &pos,
                                    PvPAction &)
{
    Vector3 agent_pos = pos;

    AABB expanded_agent_aabb {
        .pMin = agent_pos + Vector3 {
            -2.f * consts::agentRadius,
            -2.f * consts::agentRadius,
            -consts::agentRadius,
        },
        .pMax = agent_pos + Vector3 {
            2.f * consts::agentRadius,
            2.f * consts::agentRadius,
            consts::standHeight + consts::agentRadius,
        },
    };

    MeshBVH &world_geo_bvh = ctx.data().staticMeshes[0];

    const Vector3 capsule_bottom = Vector3 { 0, 0, consts::agentRadius };
    const Vector3 capsule_top =
        Vector3 { 0, 0, consts::standHeight - consts::agentRadius };

    const Vector3 capsule_segment = capsule_top - capsule_bottom;

    world_geo_bvh.findOverlaps(expanded_agent_aabb, [&](
            Vector3 tri_a, Vector3 tri_b, Vector3 tri_c) {
        Vector3 a = tri_a - agent_pos;
        Vector3 b = tri_b - agent_pos;
        Vector3 c = tri_c - agent_pos;

        AABB local_tri_aabb = AABB::point(a);
        local_tri_aabb.expand(b);
        local_tri_aabb.expand(c);

        AABB tight_local_agent_aabb {
            .pMin = {
                -consts::agentRadius,
                -consts::agentRadius,
                0.f,
            },
            .pMax = {
                consts::agentRadius,
                consts::agentRadius,
                consts::standHeight,
            }
        };

        if (!tight_local_agent_aabb.overlaps(local_tri_aabb)) {
            return;
        }

#if 0
        printf("(%f %f %f) (%f %f %f)\n",
               local_tri_aabb.pMin.x,
               local_tri_aabb.pMin.y,
               local_tri_aabb.pMin.z,
               local_tri_aabb.pMax.x,
               local_tri_aabb.pMax.y,
               local_tri_aabb.pMax.z);

        printf("(%f %f %f) (%f %f %f)\n",
               tight_local_agent_aabb.pMin.x,
               tight_local_agent_aabb.pMin.y,
               tight_local_agent_aabb.pMin.z,
               tight_local_agent_aabb.pMax.x,
               tight_local_agent_aabb.pMax.y,
               tight_local_agent_aabb.pMax.z);

        printf("Checking (%f %f %f) (%f %f %f) (%f %f %f)\n",
               a.x,
               a.y,
               a.z,
               b.x,
               b.y,
               b.z,
               c.x,
               c.y,
               c.z);
#endif

        Vector3 correction;
        if (capsuleTriCollision(a, b, c, capsule_bottom, capsule_top,
                                &correction)) {
            agent_pos += correction;
        }
    });

    pos = agent_pos;
}
#endif

inline void updateCamEntitySystem(Engine &ctx,
                                  const Position &pos,
                                  const Rotation &rot,
                                  Scale &scale,
                                  const Aim &aim,
                                  const StandState &stand_state,
                                  CamRef &cam_ref)
{
    switch (stand_state.curPose) {
        case Pose::Stand: {
            scale.d2 = 1.f;
        } break;
        case Pose::Crouch: {
            scale.d2 = consts::crouchHeight / consts::standHeight;
        } break;
        case Pose::Prone: {
            scale.d2 = consts::proneHeight / consts::standHeight;
        } break;
        default: MADRONA_UNREACHABLE();
    }

    Vector3 cam_pos = pos;
    cam_pos.z += viewHeight(stand_state);

    ctx.get<Position>(cam_ref.camEntity) = cam_pos;
    ctx.get<Rotation>(cam_ref.camEntity) = aim.rot;
    //ctx.get<Rotation>(cam_ref.camEntity) = rot;
    (void)rot;
}

static inline void makeShotVizEntity(
    Engine &ctx,
    bool hit_success,
    Vector3 from,
    Vector3 dir,
    float hit_t,
    int32_t team)
{
    Entity shot_viz = ctx.makeEntity<ShotViz>();

    ctx.get<ShotVizState>(shot_viz) = {
      .from = from,
      .dir = dir,
      .hitT = hit_t,
      .team = team,
      .hit = hit_success,
    };
    ctx.get<ShotVizRemaining>(shot_viz) = ShotVizRemaining { 20 };
}

inline void hlBattleSystem(Engine &ctx,
                           Position &pos,
                           Rotation &rot,
                           Aim &aim,
                           const Opponents &opponents,
                           Magazine &magazine,
                           const TeamInfo &team_info,
                           StandState stand_state,
                           Alive alive,
                           CombatState &combat_state)
{
    (void)rot;
    (void)magazine;

    combat_state.landedShotOn = Entity::none();
    combat_state.successfulKill = false;
    combat_state.firedShotT = -FLT_MAX;

    if (alive.mask == 0.f) {
        return;
    }

    Vector3 fire_from = pos;
    fire_from.z += viewHeight(stand_state);

    Vector3 cur_fwd = aim.rot.rotateVec(math::fwd);

    constexpr float max_aim_turn =
        discreteTurnDelta() * (consts::numTurnBuckets / 2) * consts::deltaT;
    const float cos_max_aim_turn = cosf(max_aim_turn);

    Entity tgt = Entity::none();
    float max_cos_turn = 0.f;
    float min_dist = FLT_MAX;
    Vector3 fire_dir;

    const CountT team_size = ctx.data().eTeamSize;
    for (CountT i = 0; i < team_size; i++) {
        Entity opponent = opponents.e[i];
        Alive opponent_alive = ctx.get<Alive>(opponent);
        const CombatState &opponent_combat_state =
            ctx.get<CombatState>(opponent);

        if (opponent_alive.mask == 0.f) {
            continue;
        }

        Vector3 opponent_vis_pos;
        if (!isAgentVisible(ctx, fire_from, aim, opponent, &opponent_vis_pos)) {
            continue;
        }

        if (opponent_combat_state.remainingRespawnSteps > 0) {
            continue;
        }

        Vector3 to_tgt = opponent_vis_pos - fire_from;
        float to_tgt_dist = to_tgt.length();
        assert(to_tgt_dist > 0.f);

        to_tgt /= to_tgt_dist;

        float cos_angle_to_tgt = dot(cur_fwd, to_tgt);

        // Any angle that we can autocorrect to in this frame
        // is effectively 1.
        if (cos_angle_to_tgt > cos_max_aim_turn) {
            cos_angle_to_tgt = 1.f;
        }

        // If the angle is lower than any alternative so far, pick
        // Otherwise, if angle is equal, pick min dist
        if (cos_angle_to_tgt > max_cos_turn || (
                cos_angle_to_tgt == max_cos_turn && to_tgt_dist < min_dist)) {
            max_cos_turn = cos_angle_to_tgt;
            min_dist = to_tgt_dist;
            tgt = opponent;
            fire_dir = to_tgt;
        }
    }

    if (tgt == Entity::none()) {
        return;
    }

    if (ctx.data().enableVizRender) {
        makeShotVizEntity(ctx, true, fire_from, fire_dir, min_dist,
                          team_info.team);
    }

    combat_state.landedShotOn = tgt;
    combat_state.successfulKill = true;

    DamageDealt &dmg = ctx.get<DamageDealt>(tgt);
    dmg.dmg[team_info.offset] = 100.f;
}

inline void fireSystem(Engine &ctx,
                       Position &pos,
                       Rotation &rot,
                       Aim &aim,
                       PvPDiscreteAction &action,
                       const Opponents &opponents,
                       Magazine &magazine,
                       const TeamInfo &team_info,
                       StandState stand_state,
                       Alive alive,
                       CombatState &combat_state)
{
  combat_state.landedShotOn = Entity::none();
  combat_state.successfulKill = false;
  combat_state.firedShotT = -FLT_MAX;
  combat_state.reloadedFullMag = false;

  const WeaponStats &weapon_stats =
      ctx.data().weaponTypeStats[combat_state.weaponType];

  if (alive.mask == 0.f) {
    return;
  }

  if (action.fire == 2) {
    logEvent(ctx, GameEvent {
      .type = EventType::Reload,
      .matchID = ctx.data().matchID,
      .step = (u32)ctx.singleton<MatchInfo>().curStep,
      .reload = {
        .player = u8(team_info.team * consts::maxTeamSize + team_info.offset),
        .numBulletsAtReloadTime = u8(magazine.numBullets),
      },
    });

    if (magazine.numBullets == weapon_stats.magSize) {
      combat_state.reloadedFullMag = true;
    }

    magazine.numBullets = weapon_stats.magSize;
    magazine.isReloading = weapon_stats.reloadTime;
  }

  bool reload_in_progress = magazine.isReloading > 0;
  if (reload_in_progress) {
    magazine.isReloading -= 1;
  }

  bool should_fire = false;
  if (!reload_in_progress && magazine.numBullets > 0) {
    should_fire = action.fire == 1;
  }

  if (!should_fire) {
    return;
  }

  magazine.numBullets -= 1;

  Vector3 fire_from = pos;
  fire_from.z += viewHeight(stand_state);

  float u1 = combat_state.rng.sampleUniform();
  float u2 = combat_state.rng.sampleUniform();

  float z1 = sqrtf(-2.f * logf(u1)) * cosf(2.f * math::pi * u2);
  float z2 = sqrtf(-2.f * logf(u1)) * sinf(2.f * math::pi * u2);

  float accuracy_scale =
    weapon_stats.accuracyScale;

  float upwardBias = 1.5f; // Controls amount of upward recoil bias
  float up_delta = fminf(fmaxf((z1 + upwardBias) * accuracy_scale, 0.f), 4.f * accuracy_scale);

  float right_delta = fminf(fmaxf(z2 * accuracy_scale, -4.f * accuracy_scale), 4.f * accuracy_scale);

  aim.yaw += right_delta;
  aim.pitch += up_delta;

  aim = computeAim(aim.yaw, aim.pitch);

  Vector3 fire_dir = aim.rot.rotateVec(math::fwd);

  float hit_t;
  Entity hit_entity;

  bool hit = traceRayAgainstWorld(ctx, fire_from, fire_dir,
                                  &hit_t, &hit_entity);

  if (!hit) {
    combat_state.firedShotT = FLT_MAX;
  } else {
    assert(hit_t >= 0.f);
    combat_state.firedShotT = hit_t;
  }

  bool hit_success = hit;
  ResultRef<TeamInfo> hit_teaminfo = nullptr;
  if (hit_entity == Entity::none()) {
    hit_success = false;
  } else if (
    ctx.data().taskType == Task::TDM ||
    ctx.data().taskType == Task::Zone ||
    ctx.data().taskType == Task::ZoneCaptureDefend
  ) {
    hit_teaminfo = ctx.getSafe<TeamInfo>(hit_entity);
    if (!hit_teaminfo.valid()) {
      hit_success = false;
    }

    if (hit_success && hit_teaminfo.value().team == team_info.team) {
      hit_success = false;
    }

    if (hit_success) {
      const CombatState &opponent_combat_state =
          ctx.get<CombatState>(hit_entity);
      if (opponent_combat_state.remainingRespawnSteps > 0) {
        hit_success = false;
      }
    }
  } else if (ctx.data().taskType == Task::Turret) {
    auto hit_turretstate = ctx.getSafe<TurretState>(hit_entity);
    if (!hit_turretstate.valid()) {
      hit_success = false;
    }
  }

  if (ctx.data().enableVizRender) {
    makeShotVizEntity(ctx, hit_success, fire_from, fire_dir, hit_t,
                      team_info.team);
  }

  if (!hit_success) {
      return;
  }

  logEvent(ctx, GameEvent {
    .type = EventType::PlayerShot,
    .matchID = ctx.data().matchID,
    .step = (u32)ctx.singleton<MatchInfo>().curStep,
    .playerShot = {
      .attacker = 
        u8(team_info.team * consts::maxTeamSize + team_info.offset),
      .target =
        u8(hit_teaminfo.value().team * consts::maxTeamSize +
           hit_teaminfo.value().offset),
    },
  });

  combat_state.landedShotOn = hit_entity;

  HP hp = ctx.get<HP>(hit_entity);
  if (hp.hp <= weapon_stats.dmgPerBullet) {
    combat_state.successfulKill = true;

    logEvent(ctx, GameEvent {
      .type = EventType::Kill,
      .matchID = ctx.data().matchID,
      .step = (u32)ctx.singleton<MatchInfo>().curStep,
      .kill = {
        .killer =
          u8(team_info.team * consts::maxTeamSize + team_info.offset),
        .killed =
          u8(hit_teaminfo.value().team * consts::maxTeamSize +
             hit_teaminfo.value().offset),
      },
    });
  }

  DamageDealt &dmg = ctx.get<DamageDealt>(hit_entity);
  dmg.dmg[team_info.offset] = weapon_stats.dmgPerBullet;
}

inline void turretFireSystem(Engine &ctx,
                             Position &pos,
                             Rotation &rot,
                             Aim &aim,
                             Magazine &magazine,
                             const Alive &alive,
                             TurretState &turret_state)
{
    if (alive.mask == 0.f) {
        return;
    }

    Vector3 fire_from = pos;
    fire_from.z += consts::standHeight - consts::agentRadius;

    Vector3 cur_fwd = aim.rot.rotateVec(math::fwd);

    constexpr float max_aim_turn =
        discreteTurnDelta() * (consts::numTurnBuckets / 2) * consts::deltaT;
    const float cos_max_aim_turn = cosf(max_aim_turn);

    float max_cos_turn = 0.f;
    float min_dist = FLT_MAX;
    Vector3 tgt_pos = Vector3 {
        .x = FLT_MAX,
        .y = FLT_MAX,
        .z = FLT_MAX,
    };

    // First, select closest opponent
    const CountT team_size = ctx.data().pTeamSize;
    for (CountT i = 0; i < team_size; i++) {
        Entity opponent = ctx.data().agents[i];
        Alive opponent_alive = ctx.get<Alive>(opponent);

        if (opponent_alive.mask == 0.f) {
            continue;
        }

        Vector3 opponent_vis_pos;
        if (!isAgentVisible(ctx, fire_from, aim, opponent, &opponent_vis_pos)) {
            continue;
        }

        Vector3 to_tgt = opponent_vis_pos - fire_from;
        float to_tgt_dist = to_tgt.length();
        assert(to_tgt_dist > 0.f);

        to_tgt /= to_tgt_dist;

        float cos_angle_to_tgt = dot(cur_fwd, to_tgt);

        // Any angle that we can autocorrect to in this frame
        // is effectively 1.
        if (cos_angle_to_tgt > cos_max_aim_turn) {
            cos_angle_to_tgt = 1.f;
        }

        // If the angle is lower than any alternative so far, pick
        // Otherwise, if angle is equal, pick min dist
        if (cos_angle_to_tgt > max_cos_turn || (
                cos_angle_to_tgt == max_cos_turn && to_tgt_dist < min_dist)) {
            max_cos_turn = cos_angle_to_tgt;
            min_dist = to_tgt_dist;
            tgt_pos = opponent_vis_pos;
        }
    }

    bool not_centered = false;
    if (min_dist != FLT_MAX) {
        Vector3 to_tgt = tgt_pos - fire_from;
        to_tgt = to_tgt.normalize();

        float old_yaw = aim.yaw;
        float old_pitch = aim.pitch;

        float new_yaw = -atan2f(to_tgt.x, to_tgt.y);
        float new_pitch = asinf(std::clamp(to_tgt.z, -1.f, 1.f));

        float yaw_delta = new_yaw - old_yaw;
        float pitch_delta = new_pitch - old_pitch;

        if (yaw_delta > math::pi) {
            old_yaw += 2.f * math::pi;
            yaw_delta -= 2.f * math::pi;
        } else if (yaw_delta < -math::pi) {
            old_yaw -= 2.f * math::pi;
            yaw_delta += 2.f * math::pi;
        }

        if (yaw_delta < -max_aim_turn) {
            yaw_delta = -max_aim_turn;
            not_centered = true;
        } else if (yaw_delta > max_aim_turn) {
            yaw_delta = max_aim_turn;
            not_centered = true;
        }

        if (pitch_delta < -max_aim_turn) {
            pitch_delta = -max_aim_turn;
            not_centered = true;
        } else if (pitch_delta > max_aim_turn) {
            pitch_delta = max_aim_turn;
            not_centered = true;
        }

        aim = computeAim(old_yaw + yaw_delta, old_pitch + pitch_delta);
    }
    rot = Quat::angleAxis(aim.yaw, math::up).normalize();

    if (magazine.numBullets == 0) {
        magazine.numBullets = 30;
        magazine.isReloading = 15;
    }

    bool reload_in_progress = magazine.isReloading > 0;

    if (reload_in_progress) {
        magazine.isReloading -= 1;
    }

    if (min_dist == FLT_MAX || not_centered || reload_in_progress) {
        return;
    }

    magazine.numBullets -= 1;

    float u1 = turret_state.rng.sampleUniform();
    float u2 = turret_state.rng.sampleUniform();

    float z1 = sqrtf(-2.f * logf(u1)) * cosf(2.f * math::pi * u2);
    float z2 = sqrtf(-2.f * logf(u1)) * sinf(2.f * math::pi * u2);

    cur_fwd = aim.rot.rotateVec(math::fwd);
    Vector3 cur_up = aim.rot.rotateVec(math::up);
    Vector3 cur_right = aim.rot.rotateVec(math::right);

    const float turret_accuracy = 0.15f;

    float up_delta = fminf(fmaxf(z1 * turret_accuracy,
                                 -4.f * turret_accuracy), 4.f * turret_accuracy);

    float right_delta = fminf(fmaxf(z2 * turret_accuracy,
                                    -4.f * turret_accuracy), 4.f * turret_accuracy);

    Vector3 fire_dir = (
        cur_fwd + cur_up * up_delta + cur_right * right_delta).normalize();

    float hit_t;
    Entity hit_entity;
    bool hit = traceRayAgainstWorld(ctx, fire_from, fire_dir,
                                    &hit_t, &hit_entity);

    bool hit_success = hit;
    if (hit_entity == Entity::none()) {
        hit_success = false;
    } else {
        auto hit_combatstate = ctx.getSafe<CombatState>(hit_entity);

        if (!hit_combatstate.valid() ||
            hit_combatstate.value().remainingRespawnSteps > 0) {
            hit_success = false;
        }
    }

    if (ctx.data().enableVizRender) {
        makeShotVizEntity(ctx, hit_success, fire_from, fire_dir, hit_t, 1);
    }

    if (!hit_success) {
        return;
    }

    DamageDealt &dmg = ctx.get<DamageDealt>(hit_entity);
    dmg.dmg[turret_state.offset] = 10.f;
}

inline void applyDmgSystem(
    Engine &ctx,
    Position &pos,
    AgentVelocity &vel,
    HP &hp,
    DamageDealt &dmg_dealt,
    Alive &alive,
    CombatState &combat_state)
{
    combat_state.wasShotCount = 0;
    combat_state.wasKilled = false;

    if (combat_state.remainingRespawnSteps > 0) {
        combat_state.remainingRespawnSteps -= 1;
    }

    CountT teamsize = ctx.data().eTeamSize;
    for (CountT i = 0; i < teamsize; i++) {
        if (dmg_dealt.dmg[i] > 0.f) {
            combat_state.wasShotCount += 1;
            combat_state.remainingStepsBeforeAutoheal =
                consts::numOutOfCombatStepsBeforeAutoheal;
        }

        hp.hp -= dmg_dealt.dmg[i];
        dmg_dealt.dmg[i] = 0.f;
    }

    if (alive.mask == 1.f && hp.hp <= 0.f) {
        combat_state.wasKilled = true;
        combat_state.hasDiedDuringEpisode = true;
    }

    if (hp.hp <= 0.f) {
        hp.hp = 0.f;
        alive.mask = 0.f;

        pos = Vector3 {0, 0, 10000.f};
        vel = Vector3 {0, 0, 0};
    } else {
        alive.mask = 1.f;
    }
}

inline void respawnSystem(
    Engine &ctx,
    LevelData &)
{
    if ((ctx.data().simFlags & SimFlags::NoRespawn) ==
        SimFlags::NoRespawn) {
        return;
    }

    spawnAgents(ctx, true);
}

inline void applyDmgToTurretSystem(
    Engine &ctx,
    Position &pos,
    HP &hp,
    DamageDealt &dmg_dealt,
    Alive &alive)
{
    if (alive.mask == 0.f) {
        return;
    }

    CountT teamsize = ctx.data().pTeamSize;
    for (CountT i = 0; i < teamsize; i++) {
        hp.hp -= dmg_dealt.dmg[i];
        dmg_dealt.dmg[i] = 0.f;
    }

    if (hp.hp <= 0.f) {
        hp.hp = 0.f;
        alive.mask = 0.f;

        pos = Vector3 {0, 0, 10000.f};
    }
}

inline void autoHealSystem(Engine &,
                           HP &hp,
                           const Alive &alive,
                           CombatState &combat_state)
{
    if (alive.mask == 0.f) {
        return;
    }

    if (combat_state.remainingStepsBeforeAutoheal == 0 &&
        hp.hp < 100.f) {
        hp.hp = fminf(100.f, hp.hp + consts::autohealAmountPerStep);
    } else if (combat_state.remainingStepsBeforeAutoheal > 0) {
        combat_state.remainingStepsBeforeAutoheal -= 1;
    }
}

inline void zoneSystem(Engine &ctx,
                       ZoneState &zone_state)
{
    if (zone_state.curControllingTeam != -1) {
        zone_state.zoneStepsRemaining -= 1;
    }

    if (zone_state.zoneStepsRemaining == 0) {
        zone_state.curZone += 1;
        if (zone_state.curZone == (i32)ctx.data().zones.numZones) {
            zone_state.curZone = 0;
        }

        zone_state.isCaptured = false;
        zone_state.zoneStepsRemaining = consts::numStepsPerZone;
        zone_state.stepsUntilPoint = consts::zonePointInterval;

        AABB zone_aabb = ctx.data().zones.bboxes[
            zone_state.curZone];
        Vector3 zone_center = (zone_aabb.pMax + zone_aabb.pMin) / 2.f;

        for (CountT i = 0; i < (CountT)ctx.data().numAgents; i++) {
            Entity agent = ctx.data().agents[i];
            ctx.get<CombatState>(agent).minDistToZone =
                ctx.get<Position>(agent).distance(zone_center);
        }
    }

    AABB zone_aabb = ctx.data().zones.bboxes[zone_state.curZone];
    float zone_aabb_rot_angle = ctx.data().zones.rotations[zone_state.curZone];

    Quat to_zone_frame = Quat::angleAxis(zone_aabb_rot_angle, math::up).inv();

    zone_aabb.pMin = to_zone_frame.rotateVec(zone_aabb.pMin);
    zone_aabb.pMax = to_zone_frame.rotateVec(zone_aabb.pMax);

    CountT num_team_a_inside = 0;
    CountT num_team_b_inside = 0;
    for (CountT i = 0; i < (CountT)ctx.data().numAgents; i++) {
        Entity agent = ctx.data().agents[i];
        int32_t agent_team = ctx.get<TeamInfo>(agent).team;

        Vector3 pos = ctx.get<Position>(agent);
        pos.z += consts::standHeight / 2.f;

        Vector3 pos_in_zone_frame = to_zone_frame.rotateVec(pos);

        if (!zone_aabb.contains(pos_in_zone_frame)) {
            ctx.get<CombatState>(agent).inZone = false;
            continue;
        }

        ctx.get<CombatState>(agent).inZone = true;

        if (agent_team == 0) {
            num_team_a_inside += 1;
        }

        if (agent_team == 1) {
            num_team_b_inside += 1;
        }
    }

    zone_state.stepsUntilPoint -= 1;
    zone_state.isContested = num_team_a_inside > 0 && num_team_b_inside > 0;

    if (zone_state.isContested ||
        (num_team_a_inside == 0 && num_team_b_inside == 0)) {
        zone_state.curControllingTeam = -1;
        zone_state.isCaptured = false;
        zone_state.stepsUntilPoint = consts::zonePointInterval;
    } else if (num_team_a_inside > 0 && num_team_b_inside == 0) {
        if (zone_state.curControllingTeam != 0) {
            zone_state.curControllingTeam = 0;
            zone_state.isCaptured = false;
            zone_state.stepsUntilPoint = consts::zonePointInterval;
        }
    } else if (num_team_a_inside == 0 && num_team_b_inside > 0) {
        if (zone_state.curControllingTeam != 1) {
            zone_state.curControllingTeam = 1;
            zone_state.isCaptured = false;
            zone_state.stepsUntilPoint = consts::zonePointInterval;
        }
    }
}

inline void cleanupShotVizSystem(Engine &ctx,
                                 Entity e,
                                 ShotVizRemaining &remaining)
{
  int32_t num_remaining = --remaining.numStepsRemaining;
  if (num_remaining > 0) {
      return;
  }

  ctx.destroyEntity(e);
}

inline void applyBotActionsSystem(Engine &ctx,
                                  HardcodedBotAction hardcoded,
                                  PvPDiscreteAction &out_discrete,
                                  Aim aim,
                                  PvPAimAction &out_aim,
                                  AgentPolicy policy)
{
  (void)ctx;
  (void)aim;
  if (policy.idx != consts::aStarPolicyID) {
    return;
  }

  out_discrete = {
    .moveAmount = hardcoded.moveAmount,
    .moveAngle = hardcoded.moveAngle,
    .fire = hardcoded.fire,
    .stand = hardcoded.stand,
  };

  constexpr float discrete_turn_delta = discreteTurnDelta();
  int32_t discrete_yaw_rotate = hardcoded.yawRotate;
  int32_t discrete_pitch_rotate = hardcoded.pitchRotate;

  float t_z =
      discrete_turn_delta * (discrete_yaw_rotate - consts::numTurnBuckets / 2);

  float t_x =
      discrete_turn_delta * (discrete_pitch_rotate - consts::numTurnBuckets / 2);

  out_aim = {
    .yaw = t_z,
    .pitch = t_x,
  };
}

inline void pvpMovementSystem(Engine &,
                              PvPDiscreteAction &action, 
                              const Rotation &rot,
                              AgentVelocity &vel,
                              const Alive &alive,
                              CombatState &combat_state,
                              StandState &stand_state,
                              IntermediateMoveState &move_state)
{
  if (alive.mask == 0.f) {
      return;
  }

  {
    Vector3 v = vel;
    float v_len = v.length();

    if (v_len > 0.f) {
      Vector3 norm_v = v / v_len;

      v_len -= consts::deaccelerateRate * consts::deltaT;
      v_len = fmaxf(0.f, v_len);

      vel = norm_v * v_len;
    }
  }

  if (stand_state.transitionRemaining > 0) {
    stand_state.transitionRemaining -= 1;

    if (stand_state.transitionRemaining == 0) {
      stand_state.curPose = stand_state.tgtPose;
    }
  }

  Pose action_pose = (Pose)action.stand;

  if (action_pose != stand_state.tgtPose) {
    stand_state.tgtPose = action_pose;

    int dst_to_tgt =
        abs((int)stand_state.tgtPose - (int)stand_state.curPose);
    stand_state.transitionRemaining = dst_to_tgt * (
        consts::poseTransitionSpeed / 2);
  }

  int32_t discrete_move_amount = action.moveAmount;
  int32_t discrete_move_angle = action.moveAngle;

  float move_accel_max = 3000;
  if (stand_state.curPose == Pose::Crouch) {
    move_accel_max = 100;
  } else if (stand_state.curPose == Pose::Prone) {
    move_accel_max = 50;
  }

  float move_amount = discrete_move_amount *
      (move_accel_max / (consts::numMoveAmountBuckets - 1));

  constexpr float move_angle_per_bucket =
      2.f * math::pi / float(consts::numMoveAngleBuckets);

  float move_angle = float(discrete_move_angle) * move_angle_per_bucket;

  float f_x = move_amount * sinf(move_angle);
  float f_y = move_amount * cosf(move_angle);

  vel += rot.rotateVec({ f_x, f_y, 0 }) * consts::deltaT;

  if (move_amount != 0) {
      combat_state.remainingRespawnSteps = 0;
  }

  float v_len = vel.length();
  if (v_len == 0.f) {
    return;
  }

  {
    constexpr float max_vel_change_rate =510.f;
    float tgt_max_vel;
    if (stand_state.curPose == Pose::Stand) {
      if (discrete_move_amount == 2) {
        tgt_max_vel =  consts::maxRunVelocity;
      } else {
        tgt_max_vel = consts::maxWalkVelocity;
      }
    } else if (stand_state.curPose == Pose::Crouch) {
      tgt_max_vel = consts::maxCrouchVelocity;
    } else {
      tgt_max_vel = consts::maxProneVelocity;
    }

    float max_vel_diff = tgt_max_vel - move_state.maxVelocity;
    float max_vel_adjust = fmaxf(fminf(max_vel_diff, max_vel_change_rate),
                                 -max_vel_change_rate);

    move_state.maxVelocity += max_vel_adjust;
  }


  Vector3 v_norm = vel / v_len;

  v_len = fminf(v_len, move_state.maxVelocity);

  vel = v_norm * v_len;
}

inline void coarseMovementSystem(
    Engine &,
    CoarsePvPAction &action, 
    Rotation &rot,
    Aim &aim,
    AgentVelocity &vel,
    const Alive &alive,
    CombatState &combat_state,
    const StandState &stand_state)
{
  (void)action;
  (void)rot;
  (void)aim;
  (void)vel;
  (void)alive;
  (void)combat_state;
  (void)stand_state;

#if 0
    if (alive.mask == 0.f) {
        return;
    }

    vel = Vector3::zero();

    int32_t discrete_move_amount = action.moveAmount;
    int32_t discrete_move_angle = action.moveAngle;

    float move_max = 5000;
    if (stand_state.curPose == Pose::Crouch) {
        move_max = 1000;
    } else if (stand_state.curPose == Pose::Prone) {
        move_max = 100;
    }

    float move_amount = discrete_move_amount *
        (move_max / (consts::numMoveAmountBuckets - 1));

    constexpr float move_angle_per_bucket =
        2.f * math::pi / float(consts::numMoveAngleBuckets);

    float move_angle = float(discrete_move_angle) * move_angle_per_bucket;

    float f_x = move_amount * sinf(move_angle);
    float f_y = move_amount * cosf(move_angle);

    vel += rot.rotateVec({ f_x, f_y, 0 }) * consts::deltaT;

    if (move_amount != 0) {
        combat_state.remainingRespawnSteps = 0;
    }


    constexpr float discrete_turn_delta =
        2.f * math::pi / float(consts::numFacingBuckets);

    aim.yaw = discrete_turn_delta * action.facing;
    aim.pitch = 0.f;

    aim = computeAim(aim.yaw, aim.pitch);

    rot = Quat::angleAxis(aim.yaw, math::up).normalize();
#endif
}

inline void pvpContinuousAimSystem(Engine &,
                                   PvPAimAction action,
                                   Rotation &rot,
                                   Aim &aim,
                                   const Alive &alive)
{
  if (alive.mask == 0.f) {
    return;
  }

  aim.yaw += action.yaw * consts::deltaT;
  aim.pitch += action.pitch * consts::deltaT;

  aim = computeAim(aim.yaw, aim.pitch);

  rot = Quat::angleAxis(aim.yaw, math::up).normalize();
}

inline void pvpDiscreteAimSystem(Engine &,
                                 PvPDiscreteAimAction action,
                                 PvPDiscreteAimState &,
                                 Rotation &rot,
                                 Aim &aim,
                                 const Alive &alive)
{
  if (alive.mask == 0.f) {
      return;
  }

  constexpr int32_t center_yaw_bucket = consts::discreteAimNumYawBuckets / 2;
  constexpr int32_t center_pitch_bucket = consts::discreteAimNumPitchBuckets / 2;

  int32_t yaw_bucket = action.yaw - center_yaw_bucket;

  constexpr auto yaw_turn_amounts = std::to_array<float>({
    0,                 // 0
    0.00390625f * math::pi, // 1
    0.0078125f  * math::pi, // 2
    0.015625f   * math::pi, // 3
    0.03125f    * math::pi, // 4
    0.0625f     * math::pi, // 5
    0.125f      * math::pi, // 6
  });

  if (yaw_bucket < 0) {
    aim.yaw -= yaw_turn_amounts[std::abs(yaw_bucket)];
  } else {
    aim.yaw += yaw_turn_amounts[std::abs(yaw_bucket)];
  }

  int32_t pitch_bucket = action.pitch - center_pitch_bucket;

  constexpr auto pitch_turn_amounts = std::to_array<float>({
    0,                     // 0
    0.0078125f * math::pi, // 1
    0.015625f  * math::pi, // 2
    0.03125f   * math::pi, // 3
  });

  if (pitch_bucket < 0) {
    aim.pitch -= pitch_turn_amounts[std::abs(pitch_bucket)];
  } else {
    aim.pitch += pitch_turn_amounts[std::abs(pitch_bucket)];
  }

  aim = computeAim(aim.yaw, aim.pitch);

  rot = Quat::angleAxis(aim.yaw, math::up).normalize();

#if 0
  constexpr int32_t center_yaw_bucket = consts::discreteAimNumYawBuckets / 2;
  constexpr int32_t center_pitch_bucket = consts::discreteAimNumPitchBuckets / 2;

  constexpr float discrete_yaw_delta =
      (0.25f * math::pi / consts::deltaT) / center_yaw_bucket;

  constexpr float discrete_pitch_delta =
      (0.125f * math::pi / consts::deltaT) / center_pitch_bucket;

  constexpr float max_yaw_vel = 2.f * math::pi;
  constexpr float max_pitch_vel = max_yaw_vel / 2;

  float yaw_accel = discrete_yaw_delta * (action.yaw - center_yaw_bucket);
  float pitch_accel = discrete_pitch_delta * (action.pitch - center_pitch_bucket);

  aim_state.yawVelocity += yaw_accel * consts::deltaT;
  aim_state.pitchVelocity += pitch_accel * consts::deltaT;

  aim_state.yawVelocity = 
      fminf(fmaxf(aim_state.yawVelocity, -max_yaw_vel), max_yaw_vel);
  aim_state.pitchVelocity = 
      fminf(fmaxf(aim_state.pitchVelocity, -max_pitch_vel), max_pitch_vel);

  aim.yaw += aim_state.yawVelocity * consts::deltaT;

  float start_pitch = aim.pitch;
  aim.pitch += aim_state.pitchVelocity * consts::deltaT;

  aim = computeAim(aim.yaw, aim.pitch);

  aim_state.pitchVelocity = (aim.pitch - start_pitch) / consts::deltaT;

  rot = Quat::angleAxis(aim.yaw, math::up).normalize();
#endif
}

#if 0
static inline void computeRelativePositioning(
    const Aim &tgt_aim,
    const Aim &aim,
    Vector3 to_other,
    float *out_facing_yaw,
    float *out_facing_pitch,
    float *out_dist_to_other,
    float *out_yaw_to_other,
    float *out_pitch_to_other)
{
    float relative_facing_yaw = tgt_aim.yaw - aim.yaw;
    float relative_facing_pitch = tgt_aim.pitch - aim.pitch;

    if (relative_facing_yaw > math::pi) {
        relative_facing_yaw -= 2.f * math::pi;
    } else if (relative_facing_yaw < -math::pi) {
        relative_facing_yaw += 2.f * math::pi;
    }

    *out_facing_yaw = relative_facing_yaw;
    *out_facing_pitch = relative_facing_pitch;

    float dist = to_other.length();

    if (dist < 1e-5f) {
        *out_dist_to_other = 0.f;
        *out_yaw_to_other = 0.f;
        *out_pitch_to_other = 0.f;
        return;
    }

    to_other /= dist;

    float new_yaw = -atan2f(to_other.x, to_other.y);
    float new_pitch = asinf(std::clamp(to_other.z, -1.f, 1.f));

    float yaw_delta = new_yaw - aim.yaw;
    float pitch_delta = new_pitch - aim.pitch;

    if (yaw_delta > math::pi) {
        yaw_delta -= 2.f * math::pi;
    } else if (yaw_delta < -math::pi) {
        yaw_delta += 2.f * math::pi;
    }

    *out_dist_to_other = dist;
    *out_yaw_to_other = yaw_delta;
    *out_pitch_to_other = pitch_delta;
}
#endif

#if 0
static inline float computeRightAngle(Quat q)
{
    Vector3 right_vec = normalize(q.rotateVec(math::right));

    float d = dot(right_vec, math::Vector3 { q.x, q.y, q. z });

    return 2.f * atan2f(sqrtf(1.f - d * d), q.w);
}

static inline float computeZAngle(Quat q)
{
    float siny_cosp = 2.f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.f - 2.f * (q.y * q.y + q.z * q.z);
    return atan2f(siny_cosp, cosy_cosp);
}
#endif

static inline StandObservation computeStandObs(StandState stand_state)
{
    StandObservation stand_obs;

    switch (stand_state.curPose) {
        case Pose::Stand: {
            stand_obs.curStanding = 1.f;
            stand_obs.curCrouching = 0.f;
            stand_obs.curProning = 0.f;
        } break;
        case Pose::Crouch: {
            stand_obs.curStanding = 0.f;
            stand_obs.curCrouching = 1.f;
            stand_obs.curProning = 0.f;
        } break;
        case Pose::Prone: {
            stand_obs.curStanding = 0.f;
            stand_obs.curCrouching = 0.f;
            stand_obs.curProning = 1.f;
        } break;
        default: MADRONA_UNREACHABLE();
    }

    switch (stand_state.tgtPose) {
        case Pose::Stand: {
            stand_obs.tgtStanding = 1.f;
            stand_obs.tgtCrouching = 0.f;
            stand_obs.tgtProning = 0.f;
        } break;
        case Pose::Crouch: {
            stand_obs.tgtStanding = 0.f;
            stand_obs.tgtCrouching = 1.f;
            stand_obs.tgtProning = 0.f;
        } break;
        case Pose::Prone: {
            stand_obs.tgtStanding = 0.f;
            stand_obs.tgtCrouching = 0.f;
            stand_obs.tgtProning = 1.f;
        } break;
        default: MADRONA_UNREACHABLE();
    }

    stand_obs.transitionRemaining =
        (float)stand_state.transitionRemaining /
        (float)consts::poseTransitionSpeed;

    return stand_obs;
}

inline void computePairwiseVisibility(Engine &ctx,
                                      MatchInfo &)
{
    i32 num_agents = ctx.data().numAgents;

    i32 num_checks = num_agents * (num_agents - 1);

    for (i32 i = 0; i < num_checks; i++) {
        i32 a_idx = i / (num_agents - 1);
        i32 b_idx = i % (num_agents - 1);

        if (b_idx >= a_idx) {
            b_idx += 1;
        }

        Entity a = ctx.data().agents[a_idx];
        Entity b = ctx.data().agents[b_idx];

        Vector3 a_pos = ctx.get<Position>(a);
        a_pos.z += viewHeight(ctx.get<StandState>(a));

        Vector3 b_pos = ctx.get<Position>(b);
        b_pos.z += viewHeight(ctx.get<StandState>(b));

        Aim a_aim = ctx.get<Aim>(a);

        Vector3 vis_pos;
        if (isAgentVisible(ctx, a_pos, a_aim, b, &vis_pos)) {
            ctx.data().pairwiseVisibility[i] = true;
        } else {
            ctx.data().pairwiseVisibility[i] = false;
        }
    }
}

inline void opponentsWriteVisibilitySystem(Engine &ctx,
                                           Position pos,
                                           Aim aim,
                                           Alive agent_alive,
                                           StandState stand_state,
                                           const Opponents &opponents,
                                           OpponentsVisibility &opponents_viz)
{
    Vector3 ray_o = pos;
    ray_o.z += viewHeight(stand_state);

#pragma unroll
    for (CountT i = 0; i < consts::maxTeamSize; i++) {
        opponents_viz.canSee[i] = false;

        if (agent_alive.mask == 0.f) {
            continue;
        }

        Entity opponent = opponents.e[i];
        if (opponent == Entity::none()) {
            continue;
        }

        Alive opponent_alive = ctx.get<Alive>(opponent);
        if (opponent_alive.mask == 0.f) {
            continue;
        }

        Vector3 vis_pos;
        if (isAgentVisible(ctx, ray_o, aim, opponent, &vis_pos)) {
            opponents_viz.canSee[i] = true;
        }
    }
}

inline void pvpOpponentMasksSystem(
    Engine &ctx,
    Alive agent_alive,
    const Teammates &teammates,
    const Opponents &opponents,
    const OpponentsVisibility &opponents_viz,
    OpponentMasks &masks)
{
#pragma unroll
    for (CountT i = 0; i < consts::maxTeamSize; i++) {
        masks.masks[i] = 0.f;

        if (agent_alive.mask == 0.f) {
            continue;
        }

        Entity opponent = opponents.e[i];
        if (opponent == Entity::none()) {
            continue;
        }

        Alive opponent_alive = ctx.get<Alive>(opponent);
        if (opponent_alive.mask == 0.f) {
            continue;
        }

        bool can_see_opponent = opponents_viz.canSee[i];
        for (CountT teammate_idx = 0;
             teammate_idx < consts::maxTeamSize - 1; teammate_idx++) {
            Entity teammate = teammates.e[teammate_idx];
            if (teammate == Entity::none()) {
                continue;
            }

            const OpponentsVisibility &teammate_viz =
                ctx.get<OpponentsVisibility>(teammate);

            if (teammate_viz.canSee[i]) {
                can_see_opponent = true;
                break;
            }
        }

        if (can_see_opponent) {
            masks.masks[i] = 1.f;
        }

        const CombatState &opponent_combat = ctx.get<CombatState>(opponent);
        if (opponent_combat.firedShotT >= 0) {
            masks.masks[i] = 1.f;
        }
    }
}

inline void turretOpponentMasksSystem(
    Engine &ctx,
    Alive agent_alive,
    const Opponents &opponents,
    OpponentMasks &masks)
{
    for (CountT i = 0; i < consts::maxTeamSize; i++) {
        masks.masks[i] = 0.f;

        if (agent_alive.mask == 0.f) {
            continue;
        }

        Entity opponent = opponents.e[i];
        if (opponent == Entity::none()) {
            continue;
        }

        Alive opponent_alive = ctx.get<Alive>(opponent);
        if (opponent_alive.mask == 0.f) {
            continue;
        }

        masks.masks[i] = 1.f;
    }
}

// This system packages all the egocentric observations together 
// for the policy inputs.
inline void pvpObservationsSystem(
    Engine &ctx,
    Entity self_e,
    Position self_pos,
    Rotation self_rot,
    Aim self_aim,
    const Teammates &teammates,
    const TeamInfo &team_info,
    const Opponents &opponents,
    const OpponentsVisibility &opponents_vis,

    SelfObservation &self_ob,
    TeammateObservations &teammate_obs,
    OpponentObservations &opponent_obs,
    OpponentLastKnownObservations &opponent_last_known_obs,

    SelfPositionObservation &self_pos_ob,
    TeammatePositionObservations &teammate_pos_obs,
    OpponentPositionObservations &opponent_pos_obs,
    OpponentLastKnownPositionObservations &opponent_last_known_pos_obs,

    OpponentMasks &opponent_masks,
    FiltersStateObservation &filters_state_obs)
{
  {
    const MatchInfo &match_info = ctx.singleton<MatchInfo>();

    if (match_info.curStep - ctx.data().filtersLastMatchedStep[team_info.team] < 5) {
      filters_state_obs.filtersMatching = 1.f;
    } else {
      filters_state_obs.filtersMatching = 0.f;
    }
  }

  self_ob = {};
  self_pos_ob = {};

  for (i32 i = 0; i < consts::maxTeamSize - 1; i++) {
    teammate_pos_obs.obs[i] = {};
    teammate_obs.obs[i] = {};
  }

  for (i32 i = 0; i < consts::maxTeamSize; i++) {
    opponent_pos_obs.obs[i] = {};
    opponent_obs.obs[i] = {};
  }

  auto getNormalizedPos =
    [&]
  (Vector3 p) -> Vector3
  {
    float min_x = ctx.data().worldBounds.pMin.x;
    float min_y = ctx.data().worldBounds.pMin.y;
    float min_z = ctx.data().worldBounds.pMin.z;

    float max_x = ctx.data().worldBounds.pMax.x;
    float max_y = ctx.data().worldBounds.pMax.y;
    float max_z = ctx.data().worldBounds.pMax.z;

    float x_range = max_x - min_x;
    float y_range = max_y - min_y;
    float z_range = max_z - min_z;

    float x = (p.x - min_x) / x_range;
    float y = (p.y - min_y) / y_range;
    float z = (p.z - min_z) / z_range;

    x = std::clamp(x, 0.f, 1.f);
    y = std::clamp(y, 0.f, 1.f);
    z = std::clamp(z, 0.f, 1.f);

    return {x, y, z};
  };

  auto fillCommonOb =
    [&]
  (PlayerCommonObservation &ob,
   NormalizedPositionObservation &pos_ob,
   Entity agent)
  {
    ob.isValid = 1.f;

    Alive alive = ctx.get<Alive>(agent);
    if (!alive.mask) {
      // Valid but dead
      return false;
    }

    Vector3 pos = ctx.get<Position>(agent);
    Aim aim = ctx.get<Aim>(agent);

    ob.isAlive = 1.f;
    {
      Vector3 normalized_pos = getNormalizedPos(pos);
      ob.globalX = normalized_pos.x;
      ob.globalY = normalized_pos.y;
      ob.globalZ = normalized_pos.z;

      pos_ob.x = normalized_pos.x;
      pos_ob.y = normalized_pos.y;
      pos_ob.z = normalized_pos.z;
    }

    ob.facingYaw = 0.5f * ((aim.yaw / math::pi) + 1.f);
    ob.facingPitch = 0.5f * (aim.pitch / (0.25f * math::pi) + 1.f);

    AgentVelocity vel = ctx.get<AgentVelocity>(agent);

    Vector3 relative_vel = self_rot.inv().rotateVec(vel);

    ob.velocityX = relative_vel.x;
    ob.velocityY = relative_vel.y;
    ob.velocityZ = relative_vel.z;

    {
      PvPDiscreteAimState discrete_aim_state = ctx.get<PvPDiscreteAimState>(agent);
      ob.yawVelocity = discrete_aim_state.yawVelocity;
      ob.pitchVelocity = discrete_aim_state.pitchVelocity;
    }

    ob.stand = computeStandObs(ctx.get<StandState>(agent));

    const CombatState &combat_state = ctx.get<CombatState>(agent);
    ob.inZone = combat_state.inZone;

    ob.weaponTypeObs[combat_state.weaponType] = 1.f;

    return true;
  };

  auto fillCombatOb = 
    [&]
  (CombatStateObservation &ob, Entity agent)
  {
    HP hp = ctx.get<HP>(agent);
    Magazine mag = ctx.get<Magazine>(agent);
    const CombatState &combat_state = ctx.get<CombatState>(agent);

    ob.hp = (float)hp.hp / 100.f;
    ob.magazine = (float)mag.numBullets;
    ob.isReloading = mag.isReloading;

    ob.timeBeforeAutoheal =
        float(combat_state.remainingStepsBeforeAutoheal) /
        float(consts::numOutOfCombatStepsBeforeAutoheal);
  };

  {
    if (!fillCommonOb(self_ob, self_pos_ob.ob, self_e)) {
      return;
    }

    fillCombatOb(self_ob.combat, self_e);

    const MatchInfo &match_info = ctx.singleton<MatchInfo>();
    self_ob.fractionMatchRemaining =
        float(consts::episodeLen - match_info.curStep) / consts::episodeLen;

    ZoneObservation &zone_ob = self_ob.zone;
    if (ctx.data().taskType != Task::TDM) {
      const auto &zone_state = ctx.singleton<ZoneState>();

      AABB zone_aabb =
        ctx.data().zones.bboxes[zone_state.curZone];

      float zone_angle = 
        ctx.data().zones.rotations[zone_state.curZone];

      Quat zone_rot = Quat::angleAxis(zone_angle, math::up);
      (void)zone_rot;

      Vector3 zone_center = (zone_aabb.pMax + zone_aabb.pMin) / 2.f;

      Vector3 normalized_center = getNormalizedPos(zone_center);

      zone_ob.centerX = normalized_center.x;
      zone_ob.centerY = normalized_center.y;
      zone_ob.centerZ = normalized_center.z;

      {
        Vector3 to_zone_center = zone_center - self_pos;
        float dist = to_zone_center.length();

        if (dist < 1e-2f) {
          zone_ob.toCenterDist = 0.f;
          zone_ob.toCenterYaw = 0.f;
          zone_ob.toCenterPitch = 0.f;
        } else {
          to_zone_center /= dist;

          float new_yaw = -atan2f(to_zone_center.x, to_zone_center.y);
          float new_pitch = asinf(std::clamp(to_zone_center.z, -1.f, 1.f));

          float yaw_delta = new_yaw - self_aim.yaw;
          float pitch_delta = new_pitch - self_aim.pitch;

          if (yaw_delta > math::pi) {
              yaw_delta -= 2.f * math::pi;
          } else if (yaw_delta < -math::pi) {
              yaw_delta += 2.f * math::pi;
          }

          zone_ob.toCenterDist = dist;
          zone_ob.toCenterYaw = yaw_delta;
          zone_ob.toCenterPitch = pitch_delta;
        }
      }

      zone_ob.myTeamControlling =
        (zone_state.curControllingTeam == team_info.team) ? 1.f : 0.f;
      zone_ob.enemyTeamControlling = 
        (zone_state.curControllingTeam != -1 && 
         zone_state.curControllingTeam != team_info.team) ? 1.f : 0.f;

      zone_ob.isContested = zone_state.isContested ? 1.f : 0.f;
      zone_ob.isCaptured = zone_state.isCaptured ? 1.f : 0.f;
      zone_ob.stepsUntilPoint = float(zone_state.stepsUntilPoint) /
        float(consts::zonePointInterval);
      zone_ob.stepsRemaining = float(zone_state.zoneStepsRemaining) /
        float(consts::numStepsPerZone);

      if (zone_state.curZone == 0) {
        zone_ob.id = { 1, 0, 0, 0 };
      } else if (zone_state.curZone == 1) {
        zone_ob.id = { 0, 1, 0, 0 };
      } else if (zone_state.curZone == 2) {
        zone_ob.id = { 0, 0, 1, 0 };
      } else if (zone_state.curZone == 3) {
        zone_ob.id = { 0, 0, 0, 1 };
      } else {
        assert(false);
      }
    }
  }

  const i32 my_teamsize = (i32)ctx.data().pTeamSize;
  const i32 opponent_teamsize = (i32)ctx.data().eTeamSize;

  auto fillOtherPlayerCommonOb =
    [&]
  (OtherPlayerCommonObservation &ob, Entity agent)
  {
    Vector3 other_pos = ctx.get<Position>(agent);
    Aim other_aim = ctx.get<Aim>(agent);

    Vector3 to_other = other_pos - self_pos;

    float to_other_dist = to_other.length();

    if (to_other_dist < 1e-2f) {
      ob.toPlayerDist = 0.f;
      ob.toPlayerYaw = 0.f;
      ob.toPlayerPitch = 0.f;
    } else {
      ob.toPlayerDist = to_other_dist;

      to_other /= to_other_dist;

      float new_yaw = -atan2f(to_other.x, to_other.y);
      float new_pitch = asinf(std::clamp(to_other.z, -1.f, 1.f));

      float yaw_delta = new_yaw - self_aim.yaw;
      float pitch_delta = new_pitch - self_aim.pitch;

      if (yaw_delta > math::pi) {
          yaw_delta -= 2.f * math::pi;
      } else if (yaw_delta < -math::pi) {
          yaw_delta += 2.f * math::pi;
      }

      ob.toPlayerYaw = yaw_delta;
      ob.toPlayerPitch = pitch_delta;
    }

    float relative_facing_yaw = other_aim.yaw - self_aim.yaw;
    float relative_facing_pitch = other_aim.pitch - self_aim.pitch;

    if (relative_facing_yaw > math::pi) {
        relative_facing_yaw -= 2.f * math::pi;
    } else if (relative_facing_yaw < -math::pi) {
        relative_facing_yaw += 2.f * math::pi;
    }

    ob.relativeFacingYaw = relative_facing_yaw;
    ob.relativeFacingPitch = relative_facing_pitch;
  };

  for (i32 i = 0; i < my_teamsize - 1; i++) {
    Entity agent = teammates.e[i];
    TeammateObservation &ob = teammate_obs.obs[i];
    NormalizedPositionObservation &pos_ob = teammate_pos_obs.obs[i];
    if (!fillCommonOb(ob, pos_ob, agent)) {
      continue;
    }

    fillOtherPlayerCommonOb(ob, agent);

    fillCombatOb(ob.combat, agent);
  }

  for (i32 i = 0; i < opponent_teamsize; i++) {
    Entity agent = opponents.e[i];
    OpponentObservation &ob = opponent_obs.obs[i];
    OpponentObservation &last_known_ob = opponent_last_known_obs.obs[i];

    NormalizedPositionObservation &pos_ob = opponent_pos_obs.obs[i];
    NormalizedPositionObservation &last_known_pos_ob =
        opponent_last_known_pos_obs.obs[i];

    if (!fillCommonOb(ob, pos_ob, agent)) {
      last_known_ob = {};
      last_known_pos_ob = {};
      continue;
    }

    fillOtherPlayerCommonOb(ob, agent);

    const CombatState &combat_state = ctx.get<CombatState>(agent);

    if (combat_state.wasKilled) {
      last_known_ob = {};
      last_known_pos_ob = {};
    }

    ob.wasHit = (float)combat_state.wasShotCount;
    ob.firedShot = combat_state.firedShotT >= 0.f ? 1.f : 0.f;

    ob.hasLOS = opponents_vis.canSee[i];

    bool team_knows_location = opponent_masks.masks[i] == 1.f;

    if (ob.hasLOS) {
      assert(team_knows_location);
    }

    ob.teamKnowsLocation = team_knows_location ? 1.f : 0.f;

    if (team_knows_location) {
      last_known_ob = ob;
      last_known_pos_ob = pos_ob;
    }
  }
}

inline void fullTeamObservationsSystem(
  Engine &ctx,
  FullTeamID team_id,
  FullTeamGlobalObservation &global_ob,
  FullTeamPlayerObservations &player_obs,
  FullTeamEnemyObservations &enemy_obs,
  FullTeamLastKnownEnemyObservations &last_known_enemy_obs,
  FullTeamFwdLidar &team_fwd_lidar,
  FullTeamRearLidar &team_rear_lidar)
{
  for (i32 i = 0; i < consts::maxTeamSize; i++) {
    player_obs.obs[i] = {};
  }

  for (i32 i = 0; i < consts::maxTeamSize; i++) {
    enemy_obs.obs[i] = {};
  };

  for (i32 i = 0; i < consts::maxTeamSize; i++) {
    last_known_enemy_obs.obs[i] = {};
  };

  Entity my_team_agents[consts::maxTeamSize];
  Entity enemy_team_agents[consts::maxTeamSize];

  const i32 my_teamsize = (i32)ctx.data().pTeamSize;
  const i32 enemy_teamsize = (i32)ctx.data().eTeamSize;

  {
    i32 my_team_offset = 0;
    i32 enemy_team_offset = 0;

    for (i32 i = 0; i < (i32)ctx.data().numAgents; i++) {
      Entity agent = ctx.data().agents[i];
      if (ctx.get<TeamInfo>(agent).team == team_id.id) {
        my_team_agents[my_team_offset++] = agent;
      } else {
        enemy_team_agents[enemy_team_offset++] = agent;
      }
    }

    assert(my_team_offset == my_teamsize);
    assert(enemy_team_offset == enemy_teamsize);
  }

  auto getNormalizedPos =
    [&]
  (Vector3 p) -> Vector3
  {
    float min_x = ctx.data().worldBounds.pMin.x;
    float min_y = ctx.data().worldBounds.pMin.y;
    float min_z = ctx.data().worldBounds.pMin.z;

    float max_x = ctx.data().worldBounds.pMax.x;
    float max_y = ctx.data().worldBounds.pMax.y;
    float max_z = ctx.data().worldBounds.pMax.z;

    float x_range = max_x - min_x;
    float y_range = max_y - min_y;
    float z_range = max_z - min_z;

    float x = (p.x - min_x) / x_range;
    float y = (p.y - min_y) / y_range;
    float z = (p.z - min_z) / z_range;

    return {x, y, z};
  };

  {
    if (team_id.id == 0) {
      global_ob.teamID = { 0, 1 };
    } else {
      global_ob.teamID = { 1, 0 };
    }

    const MatchInfo &match_info = ctx.singleton<MatchInfo>();

    global_ob.fractionMatchRemaining = 
        float(consts::episodeLen - match_info.curStep) / consts::episodeLen;

    {
      FullTeamZoneObservation &zone_ob = global_ob.zone;
      if (ctx.data().taskType == Task::TDM) {
        zone_ob.centerX = 0.f;
        zone_ob.centerY = 0.f;
        zone_ob.centerZ = 0.f;
        zone_ob.myTeamControlling = 0.f;
        zone_ob.enemyTeamControlling = 0.f;
        zone_ob.isContested = 0.f;
        zone_ob.isCaptured = 0.f;
        zone_ob.stepsUntilPoint = 1.f;
        zone_ob.stepsRemaining = 0.f;
        zone_ob.id = { 0, 0, 0, 0 };
      } else {
        const auto &zone_state = ctx.singleton<ZoneState>();

        AABB zone_aabb =
          ctx.data().zones.bboxes[zone_state.curZone];

        float zone_angle = 
          ctx.data().zones.rotations[zone_state.curZone];

        Quat zone_rot = Quat::angleAxis(zone_angle, math::up);
        (void)zone_rot;

        Vector3 normalized_center = getNormalizedPos(
          (zone_aabb.pMax + zone_aabb.pMin) / 2.f);
        zone_ob.centerX = normalized_center.x;
        zone_ob.centerY = normalized_center.y;
        zone_ob.centerZ = normalized_center.z;

        zone_ob.myTeamControlling =
          (zone_state.curControllingTeam == team_id.id) ? 1.f : 0.f;
        zone_ob.enemyTeamControlling = 
          (zone_state.curControllingTeam != -1 && 
           zone_state.curControllingTeam != team_id.id) ? 1.f : 0.f;

        zone_ob.isContested = zone_state.isContested ? 1.f : 0.f;
        zone_ob.isCaptured = zone_state.isCaptured ? 1.f : 0.f;
        zone_ob.stepsUntilPoint = float(zone_state.stepsUntilPoint) /
          float(consts::zonePointInterval);
        zone_ob.stepsRemaining = float(zone_state.zoneStepsRemaining) /
          float(consts::numStepsPerZone);

        if (zone_state.curZone == 0) {
          zone_ob.id = { 1, 0, 0, 0 };
        } else if (zone_state.curZone == 1) {
          zone_ob.id = { 0, 1, 0, 0 };
        } else if (zone_state.curZone == 2) {
          zone_ob.id = { 0, 0, 1, 0 };
        } else if (zone_state.curZone == 3) {
          zone_ob.id = { 0, 0, 0, 1 };
        } else {
          assert(false);
        }
      }
    }
  }

  auto fillCommonOb =
    [&]
  (FullTeamCommonObservation &ob, Entity agent, i32 i)
  {
    ob.isValid = 1.f;

    ob.id[i] = 1.f;

    Alive alive = ctx.get<Alive>(agent);
    if (!alive.mask) {
      // Valid but dead
      return false;
    }

    Vector3 pos = ctx.get<Position>(agent);
    Aim aim = ctx.get<Aim>(agent);

    ob.isAlive = 1.f;
    {
      Vector3 normalized_pos = getNormalizedPos(pos);
      ob.globalX = normalized_pos.x;
      ob.globalY = normalized_pos.y;
      ob.globalZ = normalized_pos.z;
    }

    ob.facingYaw = 0.5f * ((aim.yaw / math::pi) + 1.f);
    ob.facingPitch = 0.5f * (aim.pitch / (0.25f * math::pi) + 1.f);

    AgentVelocity vel = ctx.get<AgentVelocity>(agent);

    ob.velocityX = vel.x;
    ob.velocityY = vel.y;
    ob.velocityZ = vel.z;

    ob.stand = computeStandObs(ctx.get<StandState>(agent));

    const CombatState &combat_state = ctx.get<CombatState>(agent);
    ob.inZone = combat_state.inZone;

    return true;
  };

  for (i32 i = 0; i < my_teamsize; i++) {
    Entity agent = my_team_agents[i];
    FullTeamPlayerObservation &ob = player_obs.obs[i];
    if (!fillCommonOb(ob, agent, i)) {
      continue;
    }

    HP hp = ctx.get<HP>(agent);
    Magazine mag = ctx.get<Magazine>(agent);

    ob.hp = (float)hp.hp / 100.f;
    ob.magazine = (float)mag.numBullets / 30;
    ob.isReloading = mag.isReloading;

    const CombatState &combat_state = ctx.get<CombatState>(agent);
    ob.timeBeforeAutoheal =
        float(combat_state.remainingStepsBeforeAutoheal) /
        float(consts::numOutOfCombatStepsBeforeAutoheal);
  }

  for (i32 i = 0; i < enemy_teamsize; i++) {
    Entity agent = enemy_team_agents[i];
    FullTeamEnemyObservation &ob = enemy_obs.obs[i];
    FullTeamCommonObservation &last_known_ob = last_known_enemy_obs.obs[i];
    if (!fillCommonOb(ob, agent, i)) {
      last_known_ob = {};
      continue;
    }

    const CombatState &combat_state = ctx.get<CombatState>(agent);

    if (combat_state.wasKilled) {
      last_known_ob = {};
    }

    ob.wasHit = (float)combat_state.wasShotCount;
    ob.firedShot = combat_state.firedShotT >= 0.f ? 1.f : 0.f;

    bool team_knows_location = false;
    if (ob.firedShot) {
      team_knows_location = true;
    }

    for (i32 j = 0; j < my_teamsize; j++) {
      assert(ctx.get<Opponents>(my_team_agents[j]).e[i] == agent);
      bool can_see = ctx.get<OpponentsVisibility>(my_team_agents[j]).canSee[i];

      if (can_see) {
        ob.hasLOS[j] = 1.f;
        team_knows_location = true;
      }
    }

    ob.teamKnowsLocation = team_knows_location ? 1.f : 0.f;

    if (team_knows_location) {
      last_known_ob = ob;
    }
  }

  for (i32 i = 0; i < my_teamsize; i++) {
    Entity agent = my_team_agents[i];

    team_fwd_lidar.obs[i] = ctx.get<FwdLidar>(agent);
    team_rear_lidar.obs[i] = ctx.get<RearLidar>(agent);
  }
}

inline void exploreObservationsSystem(Engine &,
                                      Position pos,
                                      Rotation rot,
                                      SelfObservation &self_obs,
                                      SelfPositionObservation &)
{
  (void)pos;
  (void)rot;
  (void)self_obs;
}

inline void exploreLidarSystem(Engine &ctx,
                               Entity e,
                               FwdLidar &fwd_lidar)
{
    (void)ctx;
    (void)e;
    (void)fwd_lidar;
    assert(false);
}

inline void pvpLidarSystem(Engine &ctx,
                           Position pos,
                           Rotation rot,
                           Aim aim,
                           StandState stand_state,
                           TeamInfo &team_info,
                           FwdLidar &fwd_lidar,
                           RearLidar &rear_lidar)
{
    Vector3 fwd_fwd = aim.rot.rotateVec(math::fwd);
    Vector3 fwd_right = aim.rot.rotateVec(math::right);

    Vector3 rear_fwd = rot.rotateVec(math::fwd);
    Vector3 rear_right = rot.rotateVec(math::right);

    auto traceRay = [&](int32_t idx,
                        int32_t num_samples,
                        Vector3 ray_o,
                        Vector3 fwd,
                        Vector3 right,
                        float theta_range,
                        float theta_offset = 0.f)
    {
        float theta = theta_range * 
            (float(idx) / float(num_samples - 1)) + theta_offset;
        float x = -cosf(theta);
        float y = sinf(theta);

        Vector3 ray_dir = (x * right + y * fwd).normalize();

        float hit_t;
        Entity hit_entity;
        bool hit = traceRayAgainstWorld(
            ctx, ray_o, ray_dir, &hit_t, &hit_entity);

        if (hit) {
            bool is_wall = false;
            bool is_teammate = false;
            bool is_opponent = false;

            if (hit_entity == Entity::none()) {
              is_wall = true;
            } else {
              if (ctx.get<TeamInfo>(hit_entity).team == team_info.team) {
                is_teammate = true;
              } else {
                is_opponent = true;
              }
            }

            return LidarData {
              .depth = fminf(hit_t, ctx.data().maxDist),
              .isWall = is_wall ? 1.f : 0.f,
              .isTeammate = is_teammate ? 1.f : 0.f,
              .isOpponent = is_opponent ? 1.f : 0.f,
            };
        } else {
            return LidarData {
              .depth = -1.f,
              .isWall = 0.f,
              .isTeammate = 0.f,
              .isOpponent = 0.f,
            };
        }
    };

    // Used for debugging
#if 0
    auto traceFwdRayPerspective = [&](int32_t w, int32_t h) {
        Vector2 screen {
            .x = (2.f * w) / consts::fwdLidarWidth - 1.f,
            .y = (2.f * h) / consts::fwdLidarHeight - 1.f,
        };

        float right_scale =
            0.5773 * float(consts::fwdLidarWidth) / consts::fwdLidarHeight;
        float up_scale = 0.5773;
        Vector3 right = fwd_right * right_scale;
        Vector3 up = aim.rot.rotateVec(math::up) * up_scale;

        Vector3 ray_o = pos;
        ray_o.z += viewHeight();
        Vector3 ray_d = right * screen.x + up * screen.y + fwd_fwd;

        float hit_t;
        Entity hit_entity;
        bool hit = traceRayAgainstWorld(
            ctx, ray_o, ray_d, &hit_t, &hit_entity);

        return hit ? hit_t : 0.f;
    };
    (void)traceFwdRayPerspective;
#endif

    static_assert(consts::fwdLidarHeight > 1);
    static_assert(consts::rearLidarHeight > 1);

    float top_height = viewHeight(stand_state) + consts::agentRadius;

    auto computeLidarFwdRayHeightOffset = [top_height](CountT h) {
        float lidar_height_range =
            top_height - 2.f * consts::agentRadius;

        return consts::agentRadius + lidar_height_range * (
            float(h) / float(consts::fwdLidarHeight - 1));
    };

    auto computeLidarRearRayHeightOffset = [top_height](CountT h) {
        float lidar_height_range =
            top_height - 2.f * consts::agentRadius;

        return consts::agentRadius + lidar_height_range * (
            float(h) / float(consts::rearLidarHeight - 1));
    };

#ifdef MADRONA_GPU_MODE
    const int32_t num_fwd_samples =
        consts::fwdLidarWidth * consts::fwdLidarHeight;
    for (int32_t i = 0; i < num_fwd_samples; i += 32) {
        int32_t idx = i + threadIdx.x % 32;

        int32_t h = idx / consts::fwdLidarWidth;
        int32_t w = idx % consts::fwdLidarWidth;

        Vector3 ray_o = pos;
        ray_o.z += computeLidarFwdRayHeightOffset(h);

        if (idx < num_fwd_samples) {
            fwd_lidar.data[h][w] = traceRay(
                w, consts::fwdLidarWidth, ray_o, fwd_fwd, fwd_right,
                0.75f * math::pi,
                0.5f * (1.f - 0.75f) * math::pi);
#if 0
            fwd_lidar.data[h][w] = traceFwdRayPerspective(w, h);
#endif
        }
    }

    const int32_t num_rear_samples =
        consts::rearLidarWidth * consts::rearLidarHeight;
    for (int32_t i = 0; i < num_rear_samples; i += 32) {
        int32_t idx = i + threadIdx.x % 32;

        int32_t h = idx / consts::rearLidarWidth;
        int32_t w = idx % consts::rearLidarWidth;

        Vector3 ray_o = pos;
        ray_o.z += computeLidarRearRayHeightOffset(h);

        if (idx < num_rear_samples) {
            rear_lidar.data[h][w] = traceRay(
                w, consts::rearLidarWidth, ray_o, rear_fwd, rear_right,
                -math::pi);
        }
    }
#else
    for (CountT h = 0; h < consts::fwdLidarHeight; h++) {
        Vector3 ray_o = pos;
        ray_o.z += computeLidarFwdRayHeightOffset(h);

        for (CountT w = 0; w < consts::fwdLidarWidth; w++) {
            fwd_lidar.data[h][w] = traceRay(
                w, consts::fwdLidarWidth, ray_o, fwd_fwd, fwd_right,
                0.75f * math::pi, 0.5f * (1.f - 0.75f) * math::pi);

#if 0
            fwd_lidar.data[h][w] = traceFwdRayPerspective(w, h);
#endif
        }
    }

    for (CountT h = 0; h < consts::rearLidarHeight; h++) {
        Vector3 ray_o = pos;
        ray_o.z += computeLidarRearRayHeightOffset(h);

        for (CountT w = 0; w < consts::rearLidarWidth; w++) {
            rear_lidar.data[h][w] = traceRay(
                w, consts::rearLidarWidth, ray_o, rear_fwd, rear_right,
                -math::pi);
        }
    }
#endif
}

inline void exploreVisitedSystem(Engine &ctx,
                                 const Position &pos,
                                 const StartPos &start_pos,
                                 ExploreTracker &explore_tracker)
{
    Vector3 delta = pos - start_pos;

    // Discretize position
    int32_t x = int32_t((delta.x + 0.5f) / (consts::agentRadius * 2.f));
    int32_t y = int32_t((delta.y + 0.5f) / (consts::agentRadius * 2.f));

    int32_t cell_x = x + ExploreTracker::gridMaxX;
    int32_t cell_y = y + ExploreTracker::gridMaxY;

    if (cell_x < 0 || cell_x >= ExploreTracker::gridWidth ||
        cell_y < 0 || cell_y >= ExploreTracker::gridHeight) {
        return;
    }

    uint32_t old = explore_tracker.visited[cell_y][cell_x];
    uint32_t cur = (uint32_t)ctx.data().curEpisodeIdx;

    if (old != cur) {
        explore_tracker.visited[cell_y][cell_x] = cur;
        if (delta.length2() > 2.f) {
            explore_tracker.numNewCellsVisited += 1;
        }
    }
}

inline void exploreRewardSystem(Engine &,
                                Position,
                                ExploreTracker &tracker,
                                Reward &out_reward)
{
    uint32_t num_new_cells = tracker.numNewCellsVisited;
    tracker.numNewCellsVisited = 0;

    if (num_new_cells > 0) {
        out_reward.v = float(num_new_cells) * 0.05f;
    } else {
        out_reward.v = -0.005f;
    }
}

#if 0
static bool anyOpponentsVisible(
    Engine &ctx,
    const Vector3 pos,
    const Aim aim,
    const Opponents &opponents)
{
    // Check if any enemies are visible
    Vector3 ray_o = pos;
    ray_o.z += viewHeight();

    const CountT team_size = ctx.data().eTeamSize;
    for (CountT i = 0; i < team_size; i++) {
        Entity opponent = opponents.e[i];

        Alive opponent_alive = ctx.get<Alive>(opponent);

        if (opponent_alive.mask == 0.f) {
            continue;
        }

        Vector3 vis_pos;
        if (isAgentVisible(ctx, ray_o, aim, opponent, &vis_pos)) {
            return true;
        }
    }

    return false;
}
#endif

#if 0
static RewardHyperParams getRewardHyperParamsForPolicy(
  Engine &ctx,
  AgentPolicy agent_policy)
{
  i32 idx = agent_policy.idx;

  if (idx < 0) {
    idx = 0;
  }

  return ctx.data().rewardHyperParams[idx];
}
#endif

inline void tdmRewardSystem(Engine &ctx,
                            Position pos,
                            AgentPolicy agent_policy,
                            Aim aim,
                            const Alive &alive,
                            const Opponents &opponents,
                            CombatState &combat_state,
                            BreadcrumbAgentState &breadcrumb_state,
                            ExploreTracker &explore_tracker,
                            RewardHyperParams reward_hyper_params,
                            Reward &out_reward)
{
    (void)aim;
    (void)opponents;
    (void)agent_policy;
    (void)ctx;

    out_reward.v = 0.f;

    if (combat_state.successfulKill) {
        out_reward.v += 1.f;
    }

    out_reward.v -= reward_hyper_params.breadcrumbScale * breadcrumb_state.totalPenalty;

    if (combat_state.reloadedFullMag) {
        out_reward.v -= 0.01f;
    }

    {
      Vector3 goal_pos = combat_state.immitationGoalPosition;
      float dist_to_goal = goal_pos.distance(pos);

      if (dist_to_goal < combat_state.minDistToImmitationGoal) {
        float dist_reduction =
            combat_state.minDistToImmitationGoal - dist_to_goal;

        out_reward.v += dist_reduction * 0.01f;
        combat_state.minDistToImmitationGoal = dist_to_goal;
      }
    }

    if (combat_state.landedShotOn != Entity::none()) {
        out_reward.v += reward_hyper_params.shotScale * 1.f;
    } else if (combat_state.firedShotT >= 0.f) {
        //out_reward.v -= 0.005f;
    }

    if (combat_state.wasKilled) {
        out_reward.v -= 1.f;
    }

    if (combat_state.wasShotCount > 0) {
        out_reward.v -= reward_hyper_params.shotScale * 1.f;
    }

    uint32_t num_new_cells = explore_tracker.numNewCellsVisited;
    explore_tracker.numNewCellsVisited = 0;

    if (num_new_cells > 0) {
        out_reward.v += float(num_new_cells) * reward_hyper_params.exploreScale;
    } else {
        //out_reward.v -= 0.0005f;
    }

    //if (alive.mask == 1.f &&
    //        anyOpponentsVisible(ctx, pos, aim, opponents)) {
    //    out_reward.v += 0.001f;
    //}

    if (alive.mask == 0.f) {
        combat_state.successfulKill = false;
        combat_state.landedShotOn = Entity::none();
        combat_state.wasKilled = false;
        combat_state.wasShotCount = 0;
        combat_state.firedShotT = -FLT_MAX;

        return;
    }

#if 0
    if (!combat_state.wasShot && !combat_state.wasKilled) {
        out_reward.v += 0.005f;
    }
#endif

#if 0
    if (combat_state.successfulShot) {
        out_reward.v += 0.5f;
    }

    if (combat_state.successfulKill) {
        out_reward.v += 5.f;
    }

    if (combat_state.wasShot) {
        out_reward.v -= 0.1f;
    }
    if (combat_state.wasKilled) {
        out_reward.v -= 2.5f;
    }

    if (!combat_state.wasShot && !combat_state.wasKilled) {
        out_reward.v += 0.005f;
    }
#endif
}

static void learnShootingRewardSystem(
    Engine &ctx,
    CombatState &combat_state,
    ExploreTracker &explore_tracker,
    Reward &out_reward)
{
  (void)ctx;

  if (combat_state.landedShotOn != Entity::none()) {
    out_reward.v += 0.5f;
  } else if (combat_state.firedShotT >= 0.f) {
    out_reward.v -= 0.05f;
  }

  if (combat_state.reloadedFullMag) {
      out_reward.v -= 0.5f;
  }

  //uint32_t num_new_cells = explore_tracker.numNewCellsVisited;
  //explore_tracker.numNewCellsVisited = 0;

  //if (num_new_cells > 0) {
  //    out_reward.v += float(num_new_cells) * 0.001f;
  //}
  (void)explore_tracker;
}

inline void zoneRewardSystem(Engine &ctx,
                             Position pos,
                             AgentPolicy agent_policy,
                             TeamInfo team_info,
                             Aim aim,
                             const Alive &alive,
                             const Teammates &teammates,
                             const Opponents &opponents,
                             CombatState &combat_state,
                             BreadcrumbAgentState &breadcrumb_state,
                             ExploreTracker &explore_tracker,
                             RewardHyperParams reward_hyper_params,
                             Reward &out_reward)
{
    out_reward.v = 0.f;

    if (ctx.singleton<WorldCurriculum>() ==
        WorldCurriculum::LearnShooting) {
      learnShootingRewardSystem(ctx, combat_state,
                                explore_tracker, out_reward);
      return;
    }

    (void)aim;
    (void)opponents;

#if 0
    const RewardHyperParams reward_hyper_params =
        getRewardHyperParamsForPolicy(ctx, agent_policy);
#endif

    out_reward.v -= reward_hyper_params.breadcrumbScale * breadcrumb_state.totalPenalty;

    if (combat_state.reloadedFullMag) {
        out_reward.v -= 0.5f;
    }

    if (combat_state.successfulKill) {
        out_reward.v += 1.f;
    }

    if (combat_state.landedShotOn != Entity::none()) {
        out_reward.v += reward_hyper_params.shotScale * 1.f;
    } else if (combat_state.firedShotT >= 0.f) {
        //out_reward.v -= 0.005f;
    }

    if (combat_state.wasKilled) {
        out_reward.v -= 1.5f;
    }

    if (combat_state.wasShotCount > 0) {
        out_reward.v -= reward_hyper_params.shotScale * 1.f;
    }

    uint32_t num_new_cells = explore_tracker.numNewCellsVisited;
    explore_tracker.numNewCellsVisited = 0;

    if (num_new_cells > 0) {
        out_reward.v += float(num_new_cells) * reward_hyper_params.exploreScale;
    } else {
        //out_reward.v -= 0.0005f;
    }

    const ZoneState &zone_state = ctx.singleton<ZoneState>();

    if (combat_state.inZone) {
        out_reward.v += reward_hyper_params.inZoneScale;
    } else {
        AABB zone_aabb = ctx.data().zones.bboxes[zone_state.curZone];
        Vector3 center = (zone_aabb.pMax + zone_aabb.pMin) / 2.f;

        float dist = center.distance(pos);

        if (dist < combat_state.minDistToZone) {
            out_reward.v += reward_hyper_params.zoneDistScale * (
                combat_state.minDistToZone - dist);
            combat_state.minDistToZone = dist;
        }
    }

    // Our team is currently contesting this point
    if (zone_state.curControllingTeam == -1) {
        if (zone_state.isContested) {
            //out_reward.v += reward_hyper_params.zoneTeamContestScale;
        }
    } else {
        if (zone_state.curControllingTeam == team_info.team) {
            out_reward.v += reward_hyper_params.zoneTeamCtrlScale;

            if (zone_state.earnedPoint) {
                out_reward.v += reward_hyper_params.zoneEarnedPointScale;
            }
        } else {
            out_reward.v -= reward_hyper_params.zoneTeamCtrlScale;

            if (zone_state.earnedPoint) {
                out_reward.v -= reward_hyper_params.zoneEarnedPointScale;
            }
        }
    }

    if (alive.mask == 0.f) {
        combat_state.successfulKill = false;
        combat_state.landedShotOn = Entity::none();
        combat_state.wasKilled = false;
        combat_state.wasShotCount = 0;
        combat_state.firedShotT = -FLT_MAX;

        return;
    }

    {
      // Bonus reward for total map area covered by team
      float poly_area2x = 0.f;
      const CountT num_teammates = ctx.data().pTeamSize - 1;
      for (CountT i = 0; i < num_teammates - 1; i++) {
          Entity t1 = teammates.e[i];
          Entity t2 = teammates.e[i + 1];

          Vector3 t1_pos = ctx.get<Position>(t1);
          Vector3 t2_pos = ctx.get<Position>(t2);

          Vector2 e1 = t1_pos.xy() - pos.xy();
          Vector2 e2 = t2_pos.xy() - pos.xy();

          float triarea2x = e1.x * e2.y - e1.y * e2.x;

          poly_area2x += fabsf(triarea2x);
      }

      Vector2 diff = ctx.data().worldBounds.pMax.xy() - ctx.data().worldBounds.pMin.xy();

      float bounds_area = diff.x * diff.y;

      float fraction_controlled = poly_area2x / (2.f * bounds_area);

      out_reward.v += fraction_controlled * 1e-2f;
    }
}

static inline float distToZOBB(ZOBB zobb, Vector3 pos)
{
    Quat to_zobb_frame = Quat::angleAxis(zobb.rotation, math::up).inv();

    Vector3 p_min = to_zobb_frame.rotateVec(zobb.pMin);
    Vector3 p_max = to_zobb_frame.rotateVec(zobb.pMax);

    Vector3 pos_in_frame = to_zobb_frame.rotateVec(pos);

    // RTCD 5.1.3
    float sq_dist = 0.f;
    MADRONA_UNROLL
    for (int i = 0; i < 3; i++) {
        float v = pos_in_frame[i];
        if (v < p_min[i]) {
            sq_dist += math::sqr(p_min[i] - v);
        }
        if (v > p_max[i]) {
            sq_dist += math::sqr(v - p_max[i]);
        }
    }

    return sqrtf(sq_dist);
}

inline void evaluateGoalRegionsSystem(
    Engine &ctx,
    GoalRegionsState &goal_regions_state)
{
    goal_regions_state.teamStepRewards[0] = 0.f;
    goal_regions_state.teamStepRewards[1] = 0.f;

    GoalRegion *goal_regions = ctx.data().goalRegions;
    int32_t num_goal_regions = ctx.data().numGoalRegions;

    int32_t num_agents  = ctx.data().numAgents;
    Entity *agents = ctx.data().agents;

    const MatchInfo &match_info = ctx.singleton<MatchInfo>();
    int32_t attacker_team = match_info.teamA;


    for (int32_t region_idx = 0; region_idx < num_goal_regions; region_idx++) {
        const GoalRegion &goal_region = goal_regions[region_idx];
        int32_t region_team = goal_region.attackerTeam ?
            attacker_team : (attacker_team ^ 1);

        float max_min_dist_to_sub_region = -FLT_MAX;
        for (int32_t sub_idx = 0; sub_idx < goal_region.numSubRegions; sub_idx++) {
            ZOBB sub_region = goal_region.subRegions[sub_idx];

            float min_dist_to_sub_region = FLT_MAX;
            for (int32_t i = 0; i < num_agents; i++) {
                Entity agent = agents[i];
                TeamInfo agent_team = ctx.get<TeamInfo>(agent);

                if (agent_team.team != region_team) {
                    continue;
                }

                Vector3 pos = ctx.get<Position>(agent);

                float dist_to_sub_region = distToZOBB(sub_region, pos);

                if (dist_to_sub_region < min_dist_to_sub_region) {
                    min_dist_to_sub_region = dist_to_sub_region;
                }
            }

            if (min_dist_to_sub_region > max_min_dist_to_sub_region) {
                max_min_dist_to_sub_region = min_dist_to_sub_region;
            }
        }

        float prev_min_dist = goal_regions_state.minDistToRegions[region_idx];
        if (prev_min_dist == FLT_MAX) {
            goal_regions_state.minDistToRegions[region_idx] =
                max_min_dist_to_sub_region;
        } else {
            float diff = prev_min_dist - max_min_dist_to_sub_region;
            if (diff > 0.f) {
                goal_regions_state.minDistToRegions[region_idx] =
                    max_min_dist_to_sub_region;

                goal_regions_state.teamStepRewards[region_team] +=
                    diff * goal_region.rewardStrength;
            }
        }
    }
}

inline void zoneCaptureDefendRewardSystem(
    Engine &ctx,
    Position pos,
    AgentPolicy agent_policy,
    TeamInfo team_info,
    Aim aim,
    const Alive &alive,
    const Opponents &opponents,
    CombatState &combat_state,
    ExploreTracker &explore_tracker,
    RewardHyperParams reward_hyper_params,
    Reward &out_reward)
{
    (void)aim;
    (void)opponents;
    (void)agent_policy;

#if 0
    const RewardHyperParams reward_hyper_params =
        getRewardHyperParamsForPolicy(ctx, agent_policy);
#endif

    out_reward.v = 0.f;

    GoalRegionsState &goal_regions_state = ctx.singleton<GoalRegionsState>();

    out_reward.v += 0.02f * goal_regions_state.teamStepRewards[team_info.team];

    if (combat_state.reloadedFullMag) {
        out_reward.v -= 0.01f;
    }

    if (combat_state.successfulKill) {
        out_reward.v += 1.f;
    }

    if (combat_state.landedShotOn != Entity::none()) {
        out_reward.v += reward_hyper_params.shotScale * 1.f;
    } else if (combat_state.firedShotT >= 0.f) {
        //out_reward.v -= 0.005f;
    }

    if (combat_state.wasKilled) {
        out_reward.v -= 1.f;
    }

    if (combat_state.wasShotCount > 0) {
        out_reward.v -= reward_hyper_params.shotScale * 1.f;
    }

    uint32_t num_new_cells = explore_tracker.numNewCellsVisited;
    explore_tracker.numNewCellsVisited = 0;

    if (num_new_cells > 0) {
        out_reward.v += float(num_new_cells) * reward_hyper_params.exploreScale;
    } else {
        //out_reward.v -= 0.0005f;
    }

    const ZoneState &zone_state = ctx.singleton<ZoneState>();

    if (combat_state.inZone) {
        //out_reward.v += reward_hyper_params.inZoneScale;
    } else {
        AABB zone_aabb = ctx.data().zones.bboxes[zone_state.curZone];
        Vector3 center = (zone_aabb.pMax + zone_aabb.pMin) / 2.f;

        float dist = center.distance(pos);

        if (dist < combat_state.minDistToZone) {
            //out_reward.v += reward_hyper_params.zoneDistScale * (
            //    combat_state.minDistToZone - dist);
            combat_state.minDistToZone = dist;
        }
    }

    // Our team is currently contesting this point
    if (zone_state.curControllingTeam == -1) {
        if (zone_state.isContested) {
            //out_reward.v += reward_hyper_params.zoneTeamContestScale;
        }
    } else {
        if (zone_state.curControllingTeam == team_info.team) {
            out_reward.v += reward_hyper_params.zoneTeamCtrlScale;

            if (zone_state.earnedPoint) {
                out_reward.v += reward_hyper_params.zoneEarnedPointScale;
            }
        } else {
#if 0
            out_reward.v -= reward_hyper_params.zoneTeamCtrlScale;

            if (zone_state.earnedPoint) {
                out_reward.v -= reward_hyper_params.zoneEarnedPointScale;
            }
#endif
        }
    }

    auto &match_info = ctx.singleton<MatchInfo>();
    auto &match_result = ctx.singleton<MatchResult>();
    if (match_info.isFinished) {
        if (match_result.winResult == 2) {
            out_reward.v -= 5.f;
        } else if (match_result.winResult == team_info.team) {
            out_reward.v += 20.f;
        } else {
            out_reward.v -= 20.f;
        }
    }

    if (alive.mask == 0.f) {
        combat_state.successfulKill = false;
        combat_state.landedShotOn = Entity::none();
        combat_state.wasKilled = false;
        combat_state.wasShotCount = 0;
        combat_state.firedShotT = -FLT_MAX;

        return;
    }
}

inline void flankRewardSystem(Engine &ctx,
                              Entity e,
                              Position pos,
                              Aim aim,
                              StandState stand_state,
                              CombatState combat_state,
                              const Teammates &teammates,
                              const Opponents &opponents,
                              ExploreTracker &explore_tracker,
                              const RewardHyperParams &reward_hyper_params,
                              Reward &out_reward)
{
  out_reward.v = 0.f;

  Vector3 vis_check_pos = pos;
  vis_check_pos.z += viewHeight(stand_state);

  constexpr float flank_dist = 100.f;

  i32 num_teammates = ctx.data().pTeamSize - 1;

  float teammate_positioning_reward = 0.f;
  for (i32 i = 0; i < num_teammates; i++) {
    Entity teammate = teammates.e[i];

    Vector3 teammate_pos = ctx.get<Position>(teammate);
    Vector3 teammate_dir = teammate_pos - pos;

    Vector3 vis_pos;
    bool is_teammate_visible = 
      isAgentVisible(ctx, vis_check_pos, aim, teammate, &vis_pos);

    if (teammate_dir.length2() >= flank_dist * flank_dist || !is_teammate_visible) {
      teammate_positioning_reward += 0.001f;
    }
  }

  out_reward.v += teammate_positioning_reward;

  i32 num_opponents = ctx.data().pTeamSize;

  float opponent_positioning_reward = 0.f;
  for (i32 i = 0; i < num_opponents; i++) {
    Entity opponent = opponents.e[i];

    Vector3 opponent_pos = ctx.get<Position>(opponent);
    opponent_pos.z += viewHeight(ctx.get<StandState>(opponent));

    Vector3 vis_pos;
    bool can_opponent_see_agent = isAgentVisible(ctx, opponent_pos, aim, e, &vis_pos);

    if (!can_opponent_see_agent) {
      opponent_positioning_reward += 0.001f;
    }
  }

  out_reward.v += opponent_positioning_reward;

  Entity tgt = combat_state.landedShotOn;
  if (tgt != Entity::none()) {
    Aim tgt_aim = ctx.get<Aim>(tgt);

    float yaw_diff = fabsf(tgt_aim.yaw - aim.yaw);

    if (yaw_diff > math::pi) {
      if (combat_state.successfulKill) {
        out_reward.v += 1.f;
      } else {
        out_reward.v += 0.2f;
      }
    }
  }

  uint32_t num_new_cells = explore_tracker.numNewCellsVisited;
  explore_tracker.numNewCellsVisited = 0;

  if (num_new_cells > 0) {
    out_reward.v += float(num_new_cells) * reward_hyper_params.exploreScale;
  }
}

inline void pvpTeamRewardSystem(Engine &ctx,
                                TeamRewardState &team_reward_state)
{
    float team_rewards[2] = { 0, 0 };
    CountT team_size[2] = { 0, 0 };

    for (CountT i = 0; i < (CountT)ctx.data().numAgents; i++) {
        Entity agent = ctx.data().agents[i];
        int32_t team = ctx.get<TeamInfo>(agent).team;

        float reward = ctx.get<Reward>(agent).v;

        team_rewards[team] += reward;
        team_size[team] += 1;
    }

    team_rewards[0] /= float(team_size[0]);
    team_rewards[1] /= float(team_size[1]);

    team_reward_state.teamRewards[0] = team_rewards[0];
    team_reward_state.teamRewards[1] = team_rewards[1];
}

inline void pvpFinalRewardSystem(Engine &ctx,
                                 TeamInfo team_info,
                                 AgentPolicy agent_policy,
                                 RewardHyperParams reward_hyper_params,
                                 Reward &reward)
{
    const TeamRewardState &team_rewards = ctx.singleton<TeamRewardState>();

    float my_reward = reward.v;

    int32_t my_team = team_info.team;
    float team_reward = team_rewards.teamRewards[my_team];
    float other_team_reward = team_rewards.teamRewards[my_team ^ 1];

    (void)agent_policy;
#if 0
    const RewardHyperParams reward_hyper_params =
        getRewardHyperParamsForPolicy(ctx, agent_policy);
#endif

    float team_spirit = reward_hyper_params.teamSpirit;

    (void)other_team_reward;
    reward.v = my_reward * (1.f - team_spirit) + team_reward * team_spirit;
}

inline void turretRewardSystem(Engine &ctx,
                               Alive alive,
                               const Teammates &teammates,
                               const CombatState &combat_state,
                               ExploreTracker &explore_tracker,
                               Reward &out_reward)
{
    (void)alive;

    out_reward.v = -0.0075f;

    uint32_t num_new_cells = explore_tracker.numNewCellsVisited;
    explore_tracker.numNewCellsVisited = 0;

    if (num_new_cells > 0) {
        out_reward.v += float(num_new_cells) * 0.005f;
    } 

    float turret_killed_reward = 0.f;
    if (combat_state.successfulKill) {
        turret_killed_reward += 1.f;
    }

    float landed_shot_reward = 0.f;
    if (combat_state.landedShotOn != Entity::none()) {
        landed_shot_reward += 0.05f;
    }

    const CountT num_teammates = ctx.data().pTeamSize - 1;
    for (CountT i = 0; i < num_teammates; i++) {
        Entity teammate = teammates.e[i];
        const CombatState &teammate_combat_state =
            ctx.get<CombatState>(teammate);

        if (teammate_combat_state.successfulKill) {
            turret_killed_reward += 1.f;
        }

        if (teammate_combat_state.landedShotOn != Entity::none()) {
            turret_killed_reward += 0.05f;
        }
    }

    out_reward.v += turret_killed_reward;
    out_reward.v += landed_shot_reward;

    if (combat_state.wasKilled) {
        out_reward.v -= 1.f;
    }

    if (combat_state.wasShotCount > 0) {
        out_reward.v -= 0.05f;
    }
}

inline void tdmMatchInfoSystem(Engine &ctx, MatchInfo &match_info)
{
    bool team_a_alive = false;
    bool team_b_alive = false;
    for (CountT i = 0; i < ctx.data().numAgents; i++) {
        Entity agent = ctx.data().agents[i];
        Alive alive = ctx.get<Alive>(agent);
        if (alive.mask == 1.f) {
            if (i < ctx.data().pTeamSize) {
                team_a_alive = true;
            } else {
                team_b_alive = true;
            }
        }
    }

    bool match_finished = !team_a_alive || !team_b_alive;

    int32_t cur_step = match_info.curStep;
    cur_step += 1;

    const WorldReset &reset = ctx.singleton<WorldReset>();

    if (cur_step >= consts::episodeLen ||
        reset.reset == 1) {
        match_finished = true;
    }

    match_info.curStep = cur_step;
    match_info.isFinished = match_finished;
}

inline void updateTDMMatchResultsSystem(Engine &ctx,
                                        MatchResult &match_result)
{
    const MatchInfo &match_info = ctx.singleton<MatchInfo>();

    if (match_info.curStep == 1) {
        match_result.winResult = -1;
        match_result.teamTotalKills[0] = 0;
        match_result.teamTotalKills[1] = 0;
        match_result.teamObjectivePoints[0] = 0;
        match_result.teamObjectivePoints[1] = 0;
    }

    for (CountT i = 0; i < ctx.data().numAgents; i++) {
        Entity agent = ctx.data().agents[i];

        const CombatState &combat_state = ctx.get<CombatState>(agent);
        const TeamInfo &team_info = ctx.get<TeamInfo>(agent);

        int32_t agent_team = team_info.team;
        int32_t opponent_team = agent_team ^ 1;

        // Track from perspective of killed player to avoid double kills
        // since player can be killed by two opponents simultaneously
        if (combat_state.wasKilled) {
            match_result.teamTotalKills[opponent_team] += 1;
        }
    }

    if (match_info.isFinished) {
        if (match_result.teamTotalKills[0] > match_result.teamTotalKills[1]) {
            match_result.winResult = 0;
        } else if (match_result.teamTotalKills[1] >
                   match_result.teamTotalKills[0]) {
            match_result.winResult = 1;
        } else {
            match_result.winResult = 2;
        }
    }
}


inline void zoneMatchInfoSystem(Engine &ctx, MatchInfo &match_info)
{
    int32_t cur_step = match_info.curStep;
    cur_step += 1;

    const WorldReset &reset = ctx.singleton<WorldReset>();

    bool match_finished = false;
    if (cur_step >= consts::episodeLen ||
        reset.reset == 1) {
        match_finished = true;
    }

    MatchResult &match_result = ctx.singleton<MatchResult>();

    if (cur_step == 1) {
        match_result.winResult = -1;
        match_result.teamTotalKills[0] = 0;
        match_result.teamTotalKills[1] = 0;
        match_result.teamObjectivePoints[0] = 0;
        match_result.teamObjectivePoints[1] = 0;
    }

    for (CountT i = 0; i < ctx.data().numAgents; i++) {
        Entity agent = ctx.data().agents[i];

        const CombatState &combat_state = ctx.get<CombatState>(agent);
        const TeamInfo &team_info = ctx.get<TeamInfo>(agent);

        int32_t agent_team = team_info.team;
        int32_t opponent_team = agent_team ^ 1;

        // Track from perspective of killed player to avoid double kills
        // since player can be killed by two opponents simultaneously
        if (combat_state.wasKilled) {
            match_result.teamTotalKills[opponent_team] += 1;
        }
    }

    ZoneState &zone_state = ctx.singleton<ZoneState>();
    zone_state.earnedPoint = false;

    bool new_captured = false;
    if (zone_state.stepsUntilPoint == 0) {
        zone_state.stepsUntilPoint = consts::zonePointInterval;

        if (zone_state.isCaptured == false) {
            zone_state.isCaptured = true;
            new_captured = true;
        }

        assert(zone_state.curControllingTeam != -1);

        match_result.teamObjectivePoints[
            zone_state.curControllingTeam] += 1;

        zone_state.earnedPoint = true;
    }

    if (match_result.teamObjectivePoints[0] >= consts::zoneWinPoints||
        match_result.teamObjectivePoints[1] >= consts::zoneWinPoints) {
        match_finished = true;
    }

    int32_t capdefend_attacker_win = 1;
    int32_t capdefend_defender_win = 8;
    int32_t attacker_team = 0;
    int32_t defender_team = 1;

    if (match_info.teamA == 1) {
        attacker_team = 1;
        defender_team = 0;
    }

    bool team_has_all_died[2];
    if (ctx.data().taskType == Task::ZoneCaptureDefend) {
        if (match_result.teamObjectivePoints[attacker_team] == capdefend_attacker_win) {
            match_finished = true;
        }

        if (match_result.teamObjectivePoints[defender_team] == capdefend_defender_win) {
            match_finished = true;
        }

        team_has_all_died[0] = true;
        team_has_all_died[1] = true;

        for (CountT i = 0; i < ctx.data().numAgents; i++) {
            Entity agent = ctx.data().agents[i];
            const CombatState &combat_state = ctx.get<CombatState>(agent);
            const TeamInfo &team_info = ctx.get<TeamInfo>(agent);

            int32_t agent_team = team_info.team;
            if (!combat_state.hasDiedDuringEpisode) {
                team_has_all_died[agent_team] = false;
            }
        }

        if (team_has_all_died[attacker_team]) {
            match_finished = true;
        }
    }

    {
      ZoneStats &cur_zone_stats = ctx.data().zoneStats[zone_state.curZone];

      cur_zone_stats.numTotalActiveSteps += 1;

      if (zone_state.isCaptured) {
        cur_zone_stats.numTeamCapturedSteps[zone_state.curControllingTeam] += 1;
      }

      if (zone_state.isContested) {
        cur_zone_stats.numContestedSteps += 1;
      }

      if (new_captured) {
        cur_zone_stats.numSwaps += 1;
      }

      updateFiltersState(ctx, cur_step);

      if (ctx.data().eventGlobalState && ctx.data().matchID != ~0_u64) {
        if (new_captured) {
          AABB zone_aabb = ctx.data().zones.bboxes[zone_state.curZone];
          float zone_aabb_rot_angle = ctx.data().zones.rotations[zone_state.curZone];

          Quat to_zone_frame = Quat::angleAxis(zone_aabb_rot_angle, math::up).inv();

          zone_aabb.pMin = to_zone_frame.rotateVec(zone_aabb.pMin);
          zone_aabb.pMax = to_zone_frame.rotateVec(zone_aabb.pMax);

          u32 in_zone_mask = 0;
          for (CountT i = 0; i < (CountT)ctx.data().numAgents; i++) {
            Entity agent = ctx.data().agents[i];
            int32_t agent_team = ctx.get<TeamInfo>(agent).team;

            if (agent_team == zone_state.curControllingTeam) {
              Vector3 pos = ctx.get<Position>(agent);
              pos.z += consts::standHeight / 2.f;

              Vector3 pos_in_zone_frame = to_zone_frame.rotateVec(pos);

              if (!zone_aabb.contains(pos_in_zone_frame)) {
                continue;
              }

              in_zone_mask |= (1_u32 << (u32)i);
            }
          }

          logEvent(ctx, {
            .type = EventType::Capture,
            .matchID = ctx.data().matchID,
            .step = (u32)match_info.curStep,
            .capture = {
              .zoneIDX = (u8)zone_state.curZone,
              .captureTeam = (u8)zone_state.curControllingTeam,
              .inZoneMask = (u16)in_zone_mask,
            }
          });
        }

        writePackedStepSnapshot(ctx);
      }
    }

    if (match_finished) {
        if (ctx.data().taskType == Task::ZoneCaptureDefend) {
            if (match_result.teamObjectivePoints[attacker_team] == capdefend_attacker_win) {
                match_result.winResult = attacker_team;
            } else if (match_result.teamObjectivePoints[defender_team] == capdefend_defender_win ||
                       team_has_all_died[attacker_team]) {
                match_result.winResult = defender_team;
            } else {
                match_result.winResult = 2;
            }
        } else {
            if (match_result.teamObjectivePoints[0] >
                match_result.teamObjectivePoints[1]) {
                match_result.winResult = 0;
            } else if (match_result.teamObjectivePoints[1] >
                       match_result.teamObjectivePoints[0]) {
                match_result.winResult = 1;
            } else {
                match_result.winResult = 2;
            }
        }

        match_result.zoneStats = ctx.data().zoneStats;

        for (CountT i = 0; i < consts::maxZones; i++) {
            ctx.data().zoneStats[i] = {
                .numSwaps = 0,
                .numTeamCapturedSteps = { 0, 0 },
                .numContestedSteps = 0,
                .numTotalActiveSteps = 0,
            };
        }
    }

    match_info.curStep = cur_step;
    match_info.isFinished = match_finished;
}

inline void turretMatchInfoSystem(Engine &ctx, MatchInfo &match_info)
{
    bool agents_alive = false;
    for (CountT i = 0; i < (CountT)ctx.data().numAgents; i++) {
        Entity agent = ctx.data().agents[i];
        Alive alive = ctx.get<Alive>(agent);
        if (alive.mask == 1.f) {
            agents_alive = true;
            break;
        }
    }

    bool turrets_alive = false;
    for (CountT i = 0; i < (CountT)ctx.data().numTurrets; i++) {
        Entity turret = ctx.data().turrets[i];
        Alive alive = ctx.get<Alive>(turret);
        if (alive.mask == 1.f) {
            turrets_alive = true;
            break;
        }
    }

    bool match_finished = !agents_alive || !turrets_alive;

    int32_t cur_step = match_info.curStep;
    cur_step += 1;

    if (cur_step >= consts::episodeLen) {
        match_finished = true;
    }

    match_info.curStep = cur_step;
    match_info.isFinished = match_finished;
}

// Notify training that an episode has completed by
// setting done = 1 on the final step of the episode
inline void doneSystem(Engine &ctx,
                       Done &done)
{
    MatchInfo match_info = ctx.singleton<MatchInfo>();
    done.v = match_info.isFinished;
}


inline void fullTeamDoneRewardSystem(
    Engine &ctx,
    FullTeamID team_id,
    FullTeamReward &reward_out,
    FullTeamDone &done_out
)
{
  reward_out.v = 0.f;

  bool agents_done = true;

  for (i32 i = 0; i < (i32)ctx.data().numAgents; i++) {
    Entity agent = ctx.data().agents[i];

    if (ctx.get<TeamInfo>(agent).team != team_id.id) {
      continue;
    }

    Reward agent_reward = ctx.get<Reward>(agent);

    reward_out.v += agent_reward.v;

    if (!ctx.get<Done>(agent).v) {
      agents_done = false;
    }
  }

  done_out.d = agents_done;
}

inline void pvpRecordSystem(Engine &ctx,
                            MatchInfo &match_info)
{
    StepLog &step_log = *ctx.data().recordLog;
    step_log.curStep = match_info.curStep;

    for (CountT i = 0; i < consts::maxTeamSize * 2; i++) {
        if (i >= (CountT)ctx.data().numAgents) {
            break;
        }

        AgentLogData &agent_log = step_log.agentData[i];

        Entity e = ctx.data().agents[i];

        agent_log.position = ctx.get<Position>(e);
        agent_log.aim = ctx.get<Aim>(e);
        agent_log.hp = ctx.get<HP>(e);
        agent_log.mag = ctx.get<Magazine>(e);
        agent_log.standState = ctx.get<StandState>(e);

        const auto &combat_state = ctx.get<CombatState>(e);

        if (combat_state.landedShotOn == Entity::none()) {
            agent_log.shotAgentIdx = -1;
        } else {
            for (CountT j = 0; j < consts::maxTeamSize * 2; j++) {
                if (j >= (CountT)ctx.data().numAgents) {
                    break;
                }

                if (combat_state.landedShotOn == ctx.data().agents[j]) {
                    agent_log.shotAgentIdx = (int32_t)j;
                    break;
                }
            }
        } 

        agent_log.firedShotT = combat_state.firedShotT;
        agent_log.wasKilled = combat_state.wasKilled;
        agent_log.successfullKill = combat_state.successfulKill;
    }
}

inline void pvpReplaySystem(Engine &ctx,
                            MatchInfo &match_info)
{
    StepLog step_log = *ctx.data().replayLog;
    match_info.curStep = step_log.curStep;

    for (CountT i = 0; i < consts::maxTeamSize * 2; i++) {
        if (i >= (CountT)ctx.data().numAgents) {
            break;
        }

        AgentLogData agent_log = step_log.agentData[i];

        Entity e = ctx.data().agents[i];

        ctx.get<Position>(e) = agent_log.position;
        ctx.get<Aim>(e) = agent_log.aim;
        ctx.get<Rotation>(e) =
            Quat::angleAxis(agent_log.aim.yaw, math::up).normalize();
        ctx.get<HP>(e) = agent_log.hp;
        ctx.get<Magazine>(e) = agent_log.mag;
        ctx.get<StandState>(e) = agent_log.standState;

        auto &combat_state = ctx.get<CombatState>(e);
        if (agent_log.shotAgentIdx == -1) {
            combat_state.landedShotOn = Entity::none();
        } else {
            combat_state.landedShotOn = ctx.data().agents[agent_log.shotAgentIdx];
        }

        combat_state.firedShotT = agent_log.firedShotT;
        combat_state.wasKilled = agent_log.wasKilled;
        combat_state.successfulKill = agent_log.successfullKill;

        if (combat_state.wasKilled) {
            combat_state.hasDiedDuringEpisode = true;
        }

        if (agent_log.firedShotT >= 0.f) {
            Vector3 fire_from = agent_log.position;
            fire_from.z += viewHeight(agent_log.standState);

            makeShotVizEntity(ctx, agent_log.shotAgentIdx != -1,
                              fire_from,
                              agent_log.aim.rot.rotateVec(math::fwd),
                              agent_log.firedShotT,
                              ctx.get<TeamInfo>(e).team);
        }
    }
}

inline void leaveBreadcrumbsSystem(Engine &ctx,
                                   Position pos,
                                   TeamInfo team_info,
                                   BreadcrumbAgentState &breadcrumb_agent_state)
{
    breadcrumb_agent_state.totalPenalty = 0.f;

    const float breadcrumb_penalty = 1.f;
    int32_t breadcrumb_frequency = 10;

    bool updated_last_breadcrumb = false;
    if (breadcrumb_agent_state.lastBreadcrumb != Entity::none()) {
        auto last_breadcrumb = ctx.getCheck<Breadcrumb>(breadcrumb_agent_state.lastBreadcrumb);
        if (last_breadcrumb.valid()) {
            if (pos.distance(last_breadcrumb.value().pos) < consts::agentRadius * 4) {
                last_breadcrumb.value().penalty = breadcrumb_penalty;
                updated_last_breadcrumb = true;

                breadcrumb_agent_state.stepsSinceLastNewBreadcrumb = 0;
            }
        }
    }

    if (!updated_last_breadcrumb) {
        breadcrumb_agent_state.stepsSinceLastNewBreadcrumb += 1;
        if (breadcrumb_agent_state.stepsSinceLastNewBreadcrumb >
            breadcrumb_frequency) {
            Entity new_breadcrumb_entity = ctx.makeEntity<BreadcrumbEntity>();

            Breadcrumb &breadcrumb = ctx.get<Breadcrumb>(new_breadcrumb_entity);
            breadcrumb.pos = pos;
            breadcrumb.penalty = breadcrumb_penalty;
            breadcrumb.teamInfo = team_info;

            breadcrumb_agent_state.stepsSinceLastNewBreadcrumb = 0;
            breadcrumb_agent_state.lastBreadcrumb = new_breadcrumb_entity;

            ctx.get<Position>(new_breadcrumb_entity) = pos;
            ctx.get<Rotation>(new_breadcrumb_entity) = Quat { 1, 0, 0, 0 };
            ctx.get<Scale>(new_breadcrumb_entity) = 10.f * Diag3x3 { 1, 1, 1 };
            //ctx.get<ObjectID>(new_breadcrumb_entity) = { 11 };
        }
    }

}


inline void accumulateBreadcrumbPenaltiesSystem(Engine &ctx,
                                                Entity e,
                                                Breadcrumb &breadcrumb)
{
    for (int team_idx = 0; team_idx < 2; team_idx++) {
        for (int offset = 0; offset < (int)ctx.data().pTeamSize; offset++) {
            if (breadcrumb.teamInfo.team != team_idx) {
                continue;
            }

            if (breadcrumb.teamInfo.offset == offset) {
                continue;
            }

            Entity agent =
                ctx.data().agents[team_idx * ctx.data().pTeamSize + offset];

            BreadcrumbAgentState &breadcrumb_agent_state =
                ctx.get<BreadcrumbAgentState>(agent);

            Vector3 agent_pos = ctx.get<Position>(agent);

            if (agent_pos.distance(breadcrumb.pos) <= consts::agentRadius * 4.f) {
                AtomicRef<float> atomic(breadcrumb_agent_state.totalPenalty);
                atomic.fetch_add_relaxed(breadcrumb.penalty);
            }
        }
    }

    breadcrumb.penalty -= 0.025f;

    if (breadcrumb.penalty <= 0.f) {
        ctx.destroyEntity(e);
    }
}

void readFullTeamActionsPolicies(Engine &ctx,
                                 FullTeamID team_id,
                                 const FullTeamActions &actions,
                                 FullTeamPolicy team_policy)
{
  assert(false);
  (void)ctx;
  (void)team_id;
  (void)actions;
  (void)team_policy;

#if 0
  i32 team_agent_offset = 0;

  for (i32 i = 0; i < (i32)ctx.data().numAgents; i++) {
    Entity agent = ctx.data().agents[i];

    if (ctx.get<TeamInfo>(agent).team != team_id.id) {
      continue;
    }

    ctx.get<PvPAction>(agent) = actions.actions[team_agent_offset];

    ctx.get<AgentPolicy>(agent).idx = team_policy.idx;

    team_agent_offset += 1;
  }
#endif
}

namespace {

namespace NavUtils {

Vector3 CenterOfTri(const LevelData &lvl_data, int tri)
{
    const Navmesh &navmesh = lvl_data.navmesh;

    Vector3 center = { 0.0f, 0.0f, 0.0f };
    for (int i = 0; i < 3; i++)
    {
        center += navmesh.vertices[navmesh.triIndices[tri * 3 + i]] / 3.0f;
    }
    return center;
}


int NearestNavTri(const LevelData &lvl_data, Vector3 pos)
{
  const Navmesh &navmesh = lvl_data.navmesh;

  // Keep track of the nearest while looking for containment.
  float closest = FLT_MAX;
  int closestIdx = -1;
  for (int tri = 0; tri < (int)navmesh.numTris; tri++)
  {
    bool contained = true;
    bool gtz = false;
    for (int i = 0; i < 3; i++)
    {
      Vector3 v1 = navmesh.vertices[navmesh.triIndices[tri * 3 + i]];
      Vector3 v2 = navmesh.vertices[navmesh.triIndices[tri * 3 + ((i + 1) % 3)]];
      Vector3 v3 = v2 - v1;
      Vector3 vp = pos - v1;
      Vector3 c = cross(v3, vp);
      if ((c.z > 0.0f) != gtz && i > 0)
        contained = false;
      gtz = c.z > 0.0f;
      float distsq = v1.distance2(pos);
      if (distsq < closest)
      {
        float dir = v3.dot(vp);
        Vector3 perp = vp * (-dir / v3.dot(v3)) + v3;
        distsq = perp.dot(perp);
        if (distsq < closest)
        {
          closest = fabs(c.z);
          closestIdx = tri;
        }
      }
    }
    if (contained)
      return tri;
  }
  return closestIdx;
}

Vector3 PathfindToPoint(const LevelData &lvl_data,
                        const Vector3& start, const Vector3& pos)
{
  const Navmesh &navmesh = lvl_data.navmesh;
  const AStarLookup &astar_lookup = lvl_data.aStarLookup;

  int start_tri = NearestNavTri(lvl_data, start);
  int goal_tri = NearestNavTri(lvl_data, pos);

  assert(start_tri != -1 && goal_tri != -1);

  int next_tri = astar_lookup.data[start_tri * navmesh.numTris + goal_tri];

  if (next_tri == -1) {
    return Vector3::zero();
  } else if (next_tri == goal_tri) {
    return pos;
  } else {
    return CenterOfTri(lvl_data, next_tri);
  }
}

}

}

inline void planAStarAISystem(Engine &ctx,
                   Position agent_pos,
                   Aim agent_aim,
                   const Magazine &magazine,
                   const HP &hp,
                   const FwdLidar &fwd_lidar,
                   const RearLidar &rear_lidar,
                   CombatState &combat_state,
                   const OpponentsVisibility &enemies,
                   HardcodedBotAction &bot_action_out,
                   AgentPolicy &policy
    )
{
  if (policy.idx != consts::aStarPolicyID) {
    return;
  }

  (void)hp;
  (void)rear_lidar;

  const LevelData &lvl_data = ctx.singleton<LevelData>();
  const Zones& zones = ctx.data().zones;
  const ZoneState &zone_mode_state = ctx.singleton<ZoneState>();

  RNG &combat_rng = combat_state.rng;

  int move_amount = combat_rng.sampleI32(0, 2);
  int move_angle = combat_rng.sampleI32(0, 2);
  int r_yaw = combat_rng.sampleI32(0, 5);
  int r_pitch = combat_rng.sampleI32(0, 2);
  int r = magazine.numBullets == 0 ? 1 : 0;
  int stand = combat_rng.sampleI32(0, 2);

  // If we can see an enemy, fire.
  int f = 0;
  int numAgents = ctx.data().numAgents;
  for (int i = 0; i < numAgents / 2; i++)
  {
    if (enemies.canSee[i])
      f = 1;
  }

  // If there's an active zone, move to it.
  int zoneIdx = zone_mode_state.curZone;
  assert(zoneIdx >= 0 && zoneIdx < (int)zones.numZones);

  // Get a target point in the zone.
  Vector3 center = zones.bboxes[zoneIdx].centroid();

  //Entity agent1 = ctx.data().agents[0];
  //  const GlobalPosObservation& c = ctx.get<GlobalPosObservation>(agent1);
  //  center = Vector3(c.globalX, c.globalY, 0.0f);

  // Pathfind to the target point.
  Vector3 pos = Vector3(agent_pos.x, agent_pos.y, 0.0f);
  center = NavUtils::PathfindToPoint(lvl_data, pos, center);
  /*int navTri = NavUtils::NearestNavTri(navmesh, center);
    if (navTri >= 0)
    center = NavUtils::CenterOfTri(navmesh, navTri);*/

  // Turn to face the target, and if we're facing the right way, move forward.
  center.z = 0.0f;
  Vector3 fwd = Vector3(-sinf(agent_aim.yaw), cosf(agent_aim.yaw));
  Vector3 tgtDir = (center - pos).normalize();
  move_amount = dot(fwd, tgtDir) > 0.6f ? 1 : 0;
  r_yaw = cross(fwd, tgtDir).z < 0.0f ? 0 + move_amount : 4 - move_amount;
  move_amount *= 2;
  move_angle = 0;
  stand = 0;

  // If we're facing a wall, and it's right in our face, push off of it.
  float collisionAng = 0.0f;
  float collisionNorm = 0.0f;
  for (int y = 0; y < consts::fwdLidarHeight; y++)
  {
      for (int x = 0; x < consts::fwdLidarWidth; x++)
      {
          if (fwd_lidar.data[y][x].depth < 16.0f)
          {
              collisionNorm++;
              collisionAng += x;
          }
      }
  }

  // If anything invades our personal space, backpedal from it.
  if (collisionNorm > 0.0f)
  {
      collisionAng /= collisionNorm;
      // The range should be devided into 8 equal segments, but we don't care about the back ones, and we only see half of the extreme side ones.
      // So we look at 16 half-segments...
      move_amount = 1;
      switch ((int)(collisionAng / consts::fwdLidarWidth * 8.0f))
      {
        case 0:
          move_angle = 2;
          break;
        case 1:
        case 2:
          move_angle = 3;
          break;
        case 3:
        case 4:
          move_angle = 4;
          move_amount = 2;
          break;
        case 5:
        case 6:
          move_angle = 5;
          break;
        case 7:
          move_angle = 6;
          break;
      }
  }
  
  // Don't try to fire while reloading, and don't try to turn while firing.
  if (r)
      f = 0;
  if (f)
      r_yaw = 2;

  bot_action_out = HardcodedBotAction {
    .moveAmount = move_amount,
    .moveAngle = move_angle,
    .yawRotate = r_yaw,
    .pitchRotate = r_pitch,
    .fire = f,
    .reload = r,
    .stand = stand,
  };
}

static void resetAndObsTasks(TaskGraphBuilder &builder, const TaskConfig &cfg,
                             Span<const TaskGraphNodeID> deps)
{
  // Conditionally reset the world if the episode is over
  auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
    resetSystem,
      WorldReset
    >>(deps);

  TaskGraphNodeID post_reset = builder.addToGraph<ResetTmpAllocNode>({reset_sys});

#ifdef MADRONA_GPU_MODE
    post_reset = builder.addToGraph<RecycleEntitiesNode>({post_reset});
#endif

    TaskGraphNodeID collect_obs, lidar;
    if (cfg.task == Task::Explore) {
        collect_obs = builder.addToGraph<ParallelForNode<Engine,
            exploreObservationsSystem,
                Position,
                Rotation,
                SelfObservation,
                SelfPositionObservation
            >>({post_reset});

#ifdef MADRONA_GPU_MODE
        lidar = builder.addToGraph<CustomParallelForNode<Engine,
            exploreLidarSystem, 32, 1,
#else
        lidar = builder.addToGraph<ParallelForNode<Engine,
            exploreLidarSystem,
#endif
                Entity,
                FwdLidar
            >>({post_reset});
    } else if (cfg.task == Task::TDM ||
               cfg.task == Task::Zone ||
               cfg.task == Task::ZoneCaptureDefend ||
               cfg.task == Task::Turret) {
        auto update_cam_sys = builder.addToGraph<ParallelForNode<Engine,
            updateCamEntitySystem,
                Position,
                Rotation,
                Scale,
                Aim,
                StandState,
                CamRef
            >>({post_reset});

        (void)update_cam_sys;

        auto check_opponent_viz_system = builder.addToGraph<ParallelForNode<Engine,
            opponentsWriteVisibilitySystem,
                Position,
                Aim,
                Alive,
                StandState,
                Opponents,
                OpponentsVisibility
            >>({post_reset});

        TaskGraphNodeID masks_sys;

        if (cfg.task == Task::TDM ||
                cfg.task == Task::Zone ||
                cfg.task == Task::ZoneCaptureDefend) {
            masks_sys = builder.addToGraph<ParallelForNode<Engine,
                pvpOpponentMasksSystem,
                    Alive,
                    Teammates,
                    Opponents,
                    OpponentsVisibility,
                    OpponentMasks
                >>({check_opponent_viz_system});
        } else if (cfg.task == Task::Turret) {
            masks_sys = builder.addToGraph<ParallelForNode<Engine,
                turretOpponentMasksSystem,
                    Alive,
                    Opponents,
                    OpponentMasks
                >>({check_opponent_viz_system});
        } else {
            assert(false);
            MADRONA_UNREACHABLE();
        }

        collect_obs = builder.addToGraph<ParallelForNode<Engine,
          pvpObservationsSystem,
            Entity,
            Position,
            Rotation,
            Aim,
            Teammates,
            TeamInfo,
            Opponents,
            OpponentsVisibility,
            SelfObservation,
            TeammateObservations,
            OpponentObservations,
            OpponentLastKnownObservations,
            SelfPositionObservation,
            TeammatePositionObservations,
            OpponentPositionObservations,
            OpponentLastKnownPositionObservations,
            OpponentMasks,
            FiltersStateObservation
          >>({masks_sys});

        collect_obs = builder.addToGraph<ParallelForNode<Engine,
          fullTeamObservationsSystem,
            FullTeamID,
            FullTeamGlobalObservation,
            FullTeamPlayerObservations,
            FullTeamEnemyObservations,
            FullTeamLastKnownEnemyObservations,
            FullTeamFwdLidar,
            FullTeamRearLidar
          >>({collect_obs});


#ifdef MADRONA_GPU_MODE
        lidar = builder.addToGraph<CustomParallelForNode<Engine,
            pvpLidarSystem, 32, 1,
#else
        lidar = builder.addToGraph<ParallelForNode<Engine,
            pvpLidarSystem,
#endif
                Position,
                Rotation,
                Aim,
                StandState,
                TeamInfo,
                FwdLidar,
                RearLidar
            >>({post_reset});
    }

#ifndef MADRONA_GPU_MODE
    if (cfg.viz) {
      VizSystem::setupGameTasks(cfg.viz, builder);
    }
#endif

  builder.addToGraph<CompactArchetypeNode<GameEventEntity>>({});
  builder.addToGraph<CompactArchetypeNode<PackedStepSnapshotEntity>>({});
}

static void setupInitTasks(TaskGraphBuilder &builder, const TaskConfig &cfg)
{
  resetAndObsTasks(builder, cfg, {});

  builder.addToGraph<CompactArchetypeNode<StaticGeometry>>({});

  if (cfg.task == Task::Explore) {
    builder.addToGraph<CompactArchetypeNode<ExploreAgent>>({});
  } else if (cfg.task == Task::TDM ||
             cfg.task == Task::Zone ||
             cfg.task == Task::ZoneCaptureDefend ||
             cfg.task == Task::Turret) {
    builder.addToGraph<CompactArchetypeNode<PvPAgent>>({});
  }

  builder.addToGraph<CompactArchetypeNode<FullTeamInterface>>({});
}

static void setupStepTasks(TaskGraphBuilder &builder, const TaskConfig &cfg)
{
  builder.addToGraph<ClearTmpNode<GameEventEntity>>({});
  builder.addToGraph<ClearTmpNode<PackedStepSnapshotEntity>>({});

  builder.addToGraph<ParallelForNode<Engine,
      planAStarAISystem,
          Position,
          Aim,
          Magazine,
          HP,
          FwdLidar,
          RearLidar,
          CombatState,
          OpponentsVisibility,
          HardcodedBotAction,
          AgentPolicy
      >>({});

#ifndef MADRONA_GPU_MODE
  if (cfg.policyWeights) {
    addPolicyEvalTasks(builder);
  }
#endif

  auto pvpGameplayLogic = [&](Span<const TaskGraphNodeID> deps) {
    if ((cfg.simFlags & SimFlags::FullTeamPolicy) ==
        SimFlags::FullTeamPolicy) {

      builder.addToGraph<ParallelForNode<Engine,
        readFullTeamActionsPolicies,
          FullTeamID,
          FullTeamActions,
          FullTeamPolicy
        >>({});
    }

    builder.addToGraph<ParallelForNode<Engine,
        applyBotActionsSystem,
            HardcodedBotAction,
            PvPDiscreteAction,
            Aim,
            PvPAimAction,
            AgentPolicy
        >>({});

    TaskGraphNodeID move_finished_sys;
    if (cfg.highlevelMove) {
        auto move_sys = builder.addToGraph<ParallelForNode<Engine,
            coarseMovementSystem,
                CoarsePvPAction,
                Rotation,
                Aim,
                AgentVelocity,
                Alive,
                CombatState,
                StandState
            >>(deps);

        move_finished_sys = move_sys;
    } else {
        auto move_sys = builder.addToGraph<ParallelForNode<Engine,
            pvpMovementSystem,
                PvPDiscreteAction,
                Rotation,
                AgentVelocity,
                Alive,
                CombatState,
                StandState,
                IntermediateMoveState
            >>(deps);

        auto turn_sys = builder.addToGraph<ParallelForNode<Engine,
            pvpContinuousAimSystem,
                PvPAimAction,
                Rotation,
                Aim,
                Alive
            >>({move_sys});

        turn_sys = builder.addToGraph<ParallelForNode<Engine,
            pvpDiscreteAimSystem,
                PvPDiscreteAimAction,
                PvPDiscreteAimState,
                Rotation,
                Aim,
                Alive
            >>({move_sys});

        move_finished_sys = turn_sys;
    }

    auto apply_velocity = builder.addToGraph<ParallelForNode<Engine,
        applyVelocitySystem,
            Position,
            AgentVelocity,
            StandState,
            IntermediateMoveState
        >>({move_finished_sys});

    apply_velocity = builder.addToGraph<ParallelForNode<Engine,
        updateMoveStateSystem,
            Position,
            AgentVelocity,
            IntermediateMoveState
        >>({apply_velocity});

    auto fall_sys = builder.addToGraph<ParallelForNode<Engine,
        fallSystem,
            Position,
            IntermediateMoveState,
            Alive
        >>({apply_velocity});

    fall_sys = builder.addToGraph<ParallelForNode<Engine,
        updateMoveStatePostFallSystem,
            Position,
            IntermediateMoveState
        >>({fall_sys});

    TaskGraphNodeID move_done = fall_sys;

    TaskGraphNodeID battle_done;
    if (cfg.highlevelMove) {
      builder.addToGraph<ParallelForNode<Engine,
          hlBattleSystem,
              Position,
              Rotation,
              Aim,
              Opponents,
              Magazine,
              TeamInfo,
              StandState,
              Alive,
              CombatState
          >>({move_done});
    } else {
      auto fire_sys = builder.addToGraph<ParallelForNode<Engine,
          fireSystem,
              Position,
              Rotation,
              Aim,
              PvPDiscreteAction,
              Opponents,
              Magazine,
              TeamInfo,
              StandState,
              Alive,
              CombatState
          >>({move_done});

      if (cfg.task == Task::Turret) {
          fire_sys = builder.addToGraph<ParallelForNode<Engine,
              turretFireSystem,
                  Position,
                  Rotation,
                  Aim,
                  Magazine,
                  Alive,
                  TurretState
              >>({fire_sys});
      }

      battle_done = fire_sys;
    }

    auto apply_dmg_sys = builder.addToGraph<ParallelForNode<Engine,
        applyDmgSystem,
            Position,
            AgentVelocity,
            HP,
            DamageDealt,
            Alive,
            CombatState
        >>({battle_done});

    if (cfg.task == Task::Turret) {
        apply_dmg_sys = builder.addToGraph<ParallelForNode<Engine,
            applyDmgToTurretSystem,
                Position,
                HP,
                DamageDealt,
                Alive
            >>({apply_dmg_sys});
    }

    auto respawn_sys = builder.addToGraph<ParallelForNode<Engine,
    respawnSystem,
    LevelData
        >>({apply_dmg_sys});

    auto autoheal_sys = builder.addToGraph<ParallelForNode<Engine,
        autoHealSystem,
            HP,
            Alive,
            CombatState
        >>({respawn_sys});

    TaskGraphNodeID sim_done = autoheal_sys;

    if (cfg.task == Task::Zone ||
        cfg.task == Task::ZoneCaptureDefend) {
        auto zone_sys = builder.addToGraph<ParallelForNode<Engine,
            zoneSystem,
                ZoneState
            >>({sim_done});

        sim_done = zone_sys;
    }

    if (cfg.recordLog != nullptr) {
        sim_done = builder.addToGraph<ParallelForNode<Engine,
            pvpRecordSystem,
                MatchInfo
            >>({sim_done});
    }

    sim_done = builder.addToGraph<ParallelForNode<Engine,
        leaveBreadcrumbsSystem,
            Position,
            TeamInfo,
            BreadcrumbAgentState
    >>({sim_done});

    sim_done = builder.addToGraph<CompactArchetypeNode<BreadcrumbEntity>>(
        {sim_done});

    sim_done = builder.addToGraph<ParallelForNode<Engine,
        accumulateBreadcrumbPenaltiesSystem,
            Entity,
            Breadcrumb
        >>({sim_done});

    sim_done = builder.addToGraph<CompactArchetypeNode<BreadcrumbEntity>>(
        {sim_done});

    return sim_done;
  };

  auto pvpReplayLogic = [&](Span<const TaskGraphNodeID> deps) {
      auto replay = builder.addToGraph<ParallelForNode<Engine,
          pvpReplaySystem,
              MatchInfo
          >>(deps);


      if (cfg.task == Task::Zone ||
              cfg.task == Task::ZoneCaptureDefend) {
          auto zone_sys = builder.addToGraph<ParallelForNode<Engine,
              zoneSystem,
                  ZoneState
              >>({replay});

          replay = zone_sys;
      }

      return replay;
  };

  TaskGraphNodeID sim_done;
  if (cfg.task == Task::Explore) {
#if 0
    auto deaccelerate = builder.addToGraph<ParallelForNode<Engine,
        deaccelerateSystem,
            AgentVelocity
        >>({});

    auto move_sys = builder.addToGraph<ParallelForNode<Engine,
        exploreMovementSystem,
            ExploreAction,
            Rotation,
            Velocity
        >>({deaccelerate});

    auto fall = builder.addToGraph<ParallelForNode<Engine,
        fallSystem,
            Position,
            Alive
        >>({move_sys});

    auto apply_velocity = builder.addToGraph<ParallelForNode<Engine,
        applyVelocitySystem,
            Position,
            AgentVelocity
        >>({fall});

    auto collision_sys = builder.addToGraph<ParallelForNode<Engine,
        exploreWorldCollisionSystem,
            Position,
            ExploreAction
        >>({apply_velocity});

    sim_done = collision_sys;
#endif
    assert(false);
    MADRONA_UNREACHABLE();
  } else if (cfg.task == Task::TDM ||
             cfg.task == Task::Zone ||
             cfg.task == Task::ZoneCaptureDefend ||
             cfg.task == Task::Turret) {
      if (cfg.replayLog != nullptr) {
          sim_done = pvpReplayLogic({});
      } else {
          sim_done = pvpGameplayLogic({});
      }

      if (cfg.viz) {
          auto cleanup_shot_viz_sys = builder.addToGraph<ParallelForNode<Engine,
              cleanupShotVizSystem,
                  Entity,
                  ShotVizRemaining
              >>({sim_done});

          sim_done = builder.addToGraph<CompactArchetypeNode<ShotViz>>({sim_done});
        }
  } else {
      assert(false);
  }

  // Track if the match is finished
  TaskGraphNodeID match_info_sys;

  if (cfg.task == Task::TDM) {
      match_info_sys = builder.addToGraph<ParallelForNode<Engine,
          tdmMatchInfoSystem,
              MatchInfo
          >>({sim_done});

      match_info_sys = builder.addToGraph<ParallelForNode<Engine,
          updateTDMMatchResultsSystem,
              MatchResult
          >>({match_info_sys});
  } else if (cfg.task == Task::Zone ||
             cfg.task == Task::ZoneCaptureDefend) {
      match_info_sys = builder.addToGraph<ParallelForNode<Engine,
          zoneMatchInfoSystem,
              MatchInfo
          >>({sim_done});
  } else if (cfg.task == Task::Turret) {
      match_info_sys = builder.addToGraph<ParallelForNode<Engine,
          turretMatchInfoSystem,
              MatchInfo
          >>({sim_done});
  } else {
      assert(false);
  }

  auto goal_regions_eval = builder.addToGraph<ParallelForNode<Engine,
      evaluateGoalRegionsSystem,
          GoalRegionsState
      >>({match_info_sys});

  auto explore_visit = builder.addToGraph<ParallelForNode<Engine,
      exploreVisitedSystem,
          Position,
          StartPos,
          ExploreTracker
      >>({goal_regions_eval});

  // Compute initial reward now that physics has updated the world state
  TaskGraphNodeID reward_sys;
  if (cfg.task == Task::Explore) {
      reward_sys = builder.addToGraph<ParallelForNode<Engine,
          exploreRewardSystem,
              Position,
              ExploreTracker,
              Reward
          >>({explore_visit});
  } else if (cfg.task == Task::TDM ||
          cfg.task == Task::Zone ||
          cfg.task == Task::ZoneCaptureDefend) {
    if (cfg.task == Task::TDM) {
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
            tdmRewardSystem,
                Position,
                AgentPolicy,
                Aim,
                Alive,
                Opponents,
                CombatState,
                BreadcrumbAgentState,
                ExploreTracker,
                RewardHyperParams,
                Reward 
            >>({explore_visit});
    } else if (cfg.task == Task::Zone) {
        if (cfg.rewardMode == RewardMode::Flank) {
            reward_sys = builder.addToGraph<ParallelForNode<Engine,
                flankRewardSystem,
                    Entity,
                    Position,
                    Aim,
                    StandState,
                    CombatState,
                    Teammates,
                    Opponents,
                    ExploreTracker,
                    RewardHyperParams,
                    Reward
                >>({explore_visit});
        } else {
            reward_sys = builder.addToGraph<ParallelForNode<Engine,
                zoneRewardSystem,
                    Position,
                    AgentPolicy,
                    TeamInfo,
                    Aim,
                    Alive,
                    Teammates,
                    Opponents,
                    CombatState,
                    BreadcrumbAgentState,
                    ExploreTracker,
                    RewardHyperParams,
                    Reward 
                >>({explore_visit});
        }
    } else if (cfg.task == Task::ZoneCaptureDefend) {
        reward_sys = builder.addToGraph<ParallelForNode<Engine,
            zoneCaptureDefendRewardSystem,
                Position,
                AgentPolicy,
                TeamInfo,
                Aim,
                Alive,
                Opponents,
                CombatState,
                ExploreTracker,
                RewardHyperParams,
                Reward 
            >>({explore_visit});
    } else {
        assert(false);
    }

    reward_sys = builder.addToGraph<ParallelForNode<Engine,
        pvpTeamRewardSystem,
            TeamRewardState
        >>({reward_sys});

    reward_sys = builder.addToGraph<ParallelForNode<Engine,
        pvpFinalRewardSystem,
            TeamInfo,
            AgentPolicy,
            RewardHyperParams,
            Reward
        >>({reward_sys});
  } else if (cfg.task == Task::Turret) {
    reward_sys = builder.addToGraph<ParallelForNode<Engine,
        turretRewardSystem,
            Alive,
            Teammates,
            CombatState,
            ExploreTracker,
            Reward
        >>({explore_visit});
  } else {
    assert(false);
  }

  // Set done values if match is finished
  auto done_sys = builder.addToGraph<ParallelForNode<Engine,
    doneSystem,
        Done
    >>({reward_sys});

  builder.addToGraph<ParallelForNode<Engine,
    fullTeamDoneRewardSystem,
      FullTeamID,
      FullTeamReward,
      FullTeamDone
    >>({});

  resetAndObsTasks(builder, cfg, {done_sys});
}

void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const TaskConfig &cfg)
{
  setupInitTasks(taskgraph_mgr.init(TaskGraphID::Init), cfg);
  setupStepTasks(taskgraph_mgr.init(TaskGraphID::Step), cfg);
}

Sim::Sim(Engine &ctx,
         const TaskConfig &cfg,
         const WorldInit &)
    : WorldBase(ctx),
      worldBounds(cfg.worldBounds),
      maxDist((cfg.worldBounds.pMax - cfg.worldBounds.pMin).length()),
      distScale(1.f / maxDist),
      taskType(cfg.task),
      pTeamSize(cfg.pTeamSize),
      eTeamSize(cfg.eTeamSize),
      numPBTPolicies(cfg.numPBTPolicies),
      policyHistorySize(cfg.policyHistorySize),
      initRandKey(cfg.initRandKey),
      curEpisodeIdx(0),
      worldEpisodeCounter(0)
{
  trainControl = cfg.trainControl;
  policyWeights = cfg.policyWeights;

  {
      float aspect = 16.f / 9.f;
      float f = 1.f / tanf(math::toRadians(90.f / 2.f));

      Vector2 w = { f / aspect, 1.f };
      Vector2 h = {          f, 1.f };
      w *= w.invLength();
      h *= h.invLength();

      frustumData = {
          w.x, w.y,
          h.x, h.y,
      };
  }

  ctx.singleton<WorldReset>() = WorldReset { 0 };

  if (cfg.viz) {
#ifndef MADRONA_GPU_MODE
    VizSystem::initWorld(ctx, cfg.viz);
#endif
    enableVizRender = true;
  } else {
    enableVizRender = false;
  }

  ctx.singleton<LevelData>() = {
      .navmesh = cfg.navmesh,
      .aStarLookup = cfg.aStarLookup,
  };

  ctx.singleton<StandardSpawns>() = cfg.standardSpawns;
  ctx.singleton<SpawnCurriculum>() = cfg.spawnCurriculum;

#if 0
  ctx.singleton<CurriculumState>() = {
      .useCurriculumSpawnProb = 0.75f,
      .tierProbabilities = {
          0.2f,
          0.2f,
          0.2f,
          0.2f,
          0.2f,
      },
  };
#endif
  ctx.singleton<CurriculumState>() = {
      .useCurriculumSpawnProb = 1.0f,
      .tierProbabilities = {
          0.f,
          0.f,
          0.3f,
          0.3f,
          0.4f,
      },
  };

  zones = cfg.zones;

  if (cfg.recordLog != nullptr) {
      recordLog = cfg.recordLog + ctx.worldID().idx;
  } else {
      recordLog = nullptr;
  }

  if (cfg.replayLog != nullptr) {
      replayLog = cfg.replayLog + ctx.worldID().idx;
  } else {
      replayLog = nullptr;
  }

  autoReset = cfg.autoReset;
  simFlags = cfg.simFlags;

  goalRegions = cfg.goalRegions;
  numGoalRegions = cfg.numGoalRegions;

  numEpisodes = cfg.numEpisodes;
  episodes = cfg.episodes;

  numWeaponTypes = cfg.numWeaponTypes;
  weaponTypeStats = cfg.weaponTypeStats;

  trajectoryCurriculum = cfg.trajectoryCurriculum;

  assert(numWeaponTypes <= consts::maxNumWeaponTypes);

  // Creates agents, walls, etc.
  createPersistentEntities(ctx, cfg);

  ctx.singleton<WorldCurriculum>() = WorldCurriculum::FullMatch;

  // Generate initial world state
  initWorld(ctx, true);

  for (CountT i = 0; i < consts::maxZones; i++) {
      ctx.data().zoneStats[i].numSwaps = 0;
      ctx.data().zoneStats[i].numTeamCapturedSteps = { 0, 0 };
      ctx.data().zoneStats[i].numContestedSteps = 0;
      ctx.data().zoneStats[i].numTotalActiveSteps = 0;
  }

  matchID = ~0_u64;

  eventGlobalState = cfg.eventGlobalState;
  eventLoggedInStep = 0;

  for (int team = 0; team < 2; team++) {
    filtersState[team].active = 0;
    filtersLastMatchedStep[team] = -1;
  }
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, TaskConfig, Sim::WorldInit);

}
