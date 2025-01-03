#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/physics.hpp>
#include <madrona/rand.hpp>
#include <madrona/navmesh.hpp>

#include "mesh_bvh.hpp"
#include "consts.hpp"
#include "sim_flags.hpp"
#include "viz.hpp"

namespace madronaMPEnv {

// Include several madrona types into the simulator namespace for convenience
using madrona::Entity;
using madrona::CountT;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;
using madrona::phys::ExternalForce;
using madrona::phys::ExternalTorque;
using madrona::math::AABB;
using madrona::math::Vector4;
using madrona::math::Vector3;
using madrona::math::Vector2;
using madrona::math::Quat;
using madrona::Navmesh;
using madrona::RandKey;

using madrona::u16;
using madrona::i16;
using madrona::u32;
using madrona::i32;

struct TDMEpisode {
  Vector3 startPositions[consts::maxTeamSize * 2];
  float startRotations[consts::maxTeamSize * 2];

  Vector3 goalPositions[consts::maxTeamSize * 2];
};

enum class Task : uint32_t {
    Explore,
    TDM,
    Zone,
    Turret,
    ZoneCaptureDefend,
};

struct WeaponStats {
    i32 magSize;
    i32 fireQueueSize;
    i32 reloadTime;
    f32 dmgPerBullet;
    f32 accuracyScale;
};

struct Spawn {
    AABB region;
    float yawMin;
    float yawMax;
};

struct NavmeshSpawn {
    uint32_t aOffset;
    uint32_t bOffset;
    uint32_t numANavmeshPolys;
    uint32_t numBNavmeshPolys;
    float aBaseYaw;
    float bBaseYaw;
};

struct RespawnRegion {
    uint32_t startOffset;
    uint32_t numSpawns;
    AABB aabb;
    float rotation;
};

struct StandardSpawns {
    Spawn *aSpawns;
    Spawn *bSpawns;
    Spawn *commonRespawns;
    uint32_t numDefaultASpawns;
    uint32_t numDefaultBSpawns;
    uint32_t numExtraASpawns;
    uint32_t numExtraBSpawns;
    uint32_t numCommonRespawns;
    RespawnRegion *respawnRegions;
    uint32_t numRespawnRegions;
};

struct SpawnUsageCounter {
    static constexpr CountT maxNumSpawns = 128;
    uint32_t initASpawnsLastUsedTick[maxNumSpawns];
    uint32_t initBSpawnsLastUsedTick[maxNumSpawns];
    uint32_t respawnLastUsedTick[maxNumSpawns];
};

struct Zones {
    AABB *bboxes;
    float *rotations;
    uint32_t numZones;
};

struct SpawnCurriculum {
    static inline constexpr uint32_t numCurriculumTiers = 5;

    struct Tier {
        uint32_t *spawnPolyData;
        NavmeshSpawn *spawns;

        uint32_t numTotalSpawnPolys;
        uint32_t numSpawns;
    };

    Tier tiers[numCurriculumTiers];
};

struct CurriculumState {
    float useCurriculumSpawnProb;
    float tierProbabilities[SpawnCurriculum::numCurriculumTiers];
};

struct MatchInfo {
    int32_t teamA;
    int32_t curStep;
    bool isFinished;
    bool enableSpawnCurriculum;
    uint32_t curCurriculumTier;
    uint32_t curCurriculumSpawnIdx;
};

struct AStarLookup {
    u32 *data;
};

struct LevelData {
    Navmesh navmesh;
    AStarLookup aStarLookup;
};

// WorldReset is a per-world singleton component that causes the current
// episode to be terminated and the world regenerated
// (Singleton components like WorldReset can be accessed via Context::singleton
// (eg ctx.singleton<WorldReset>().reset = 1)
struct WorldReset {
    int32_t reset;
};

struct TrainControl {
    int32_t randomizeEpisodeLengthAfterReset;
    int32_t randomizeTeamSides;
};

// Discrete action component. Ranges are defined by consts::numMoveBuckets (5),
// repeated here for clarity
struct ExploreAction {
    int32_t moveAmount; // [0, 3]
    int32_t moveAngle; // [0, 7]
    int32_t rotate; // [-2, 2]
    int32_t mantle;
};

struct PvPAction {
    int32_t moveAmount; // [0, 3]
    int32_t moveAngle; // [0, 7]
    int32_t yawRotate; // [-2, 2]
    int32_t pitchRotate; // [-2, 2]
    int32_t fire;
    int32_t reload;
    int32_t stand; // [0, 1, 2]
};

struct HardcodedBotAction {
    int32_t moveAmount; // [0, 3]
    int32_t moveAngle; // [0, 7]
    int32_t yawRotate; // [-2, 2]
    int32_t pitchRotate; // [-2, 2]
    int32_t fire;
    int32_t reload;
    int32_t stand; // [0, 1, 2]
};

struct CoarsePvPAction {
    int32_t moveAmount; // [0, 3]
    int32_t moveAngle; // [0, 7]
    int32_t facing; // [0, 15]
};

struct Reward {
    float v;
};

enum class Pose : int {
    Stand = 0,
    Crouch = 1,
    Prone = 2,
};

struct StandState {
    Pose curPose;
    Pose tgtPose;
    int32_t transitionRemaining;
};

struct TeamRewardState {
    float teamRewards[2];
};

// Per-agent component that indicates that the agent's episode is finished
// This is exported per-agent for simplicity in the training code
struct Done {
    // Currently bool components are not supported due to
    // padding issues, so Done is an int32_t
    int32_t v;
};

struct ZoneStats {
    int32_t numSwaps;
    std::array<int32_t, 2> numTeamCapturedSteps;
    int32_t numContestedSteps;
    int32_t numTotalActiveSteps;
};

struct MatchResult {
    int32_t winResult;
    int32_t teamTotalKills[2];
    int32_t teamObjectivePoints[2];

    std::array<ZoneStats, consts::maxZones> zoneStats;
};

struct RewardHyperParams {
    float teamSpirit = 0.f;
    float shotScale = 0.05f;
    float exploreScale = 0.001f;
    float inZoneScale = 0.05f;
    float zoneTeamContestScale = 0.01f;
    float zoneTeamCtrlScale = 0.1f;
    float zoneDistScale = 0.005f;
    float zoneEarnedPointScale = 1.f;
    float breadcrumbScale = 0.1f;
};

struct AgentPolicy {
    int32_t idx;
};

struct TeamInfo {
    int32_t team;
    int32_t offset;
};

struct StandObservation {
    float curStanding = 0.f;
    float curCrouching = 0.f;
    float curProning = 0.f;
    float tgtStanding = 0.f;
    float tgtCrouching = 0.f;
    float tgtProning = 0.f;
    float transitionRemaining = 0.f;
};

struct ZoneObservation {
    float centerX = 0.f;
    float centerY = 0.f;
    float centerZ = 0.f;
    float toCenterDist = 0.f;
    float toCenterYaw = 0.f;
    float toCenterPitch = 0.f;
    float myTeamControlling = 0.f;
    float enemyTeamControlling = 0.f;
    float isContested = 0.f;
    float isCaptured = 0.f;
    float stepsUntilPoint = 0.f;
    float stepsRemaining = 0.f;
    std::array<float, 4> id = {};
};

struct PlayerCommonObservation {
  float isValid = 0.f; // Must be first observation
  float isAlive = 0.f;
  float globalX = 0.f;
  float globalY = 0.f;
  float globalZ = 0.f;
  float facingYaw = 0.f;
  float facingPitch = 0.f;
  float velocityX = 0.f;
  float velocityY = 0.f;
  float velocityZ = 0.f;
  StandObservation stand = {};
  float inZone = 0.f;
  std::array<float, consts::maxNumWeaponTypes> weaponTypeObs = {};
};

struct OtherPlayerCommonObservation : PlayerCommonObservation {
  float toPlayerDist = 0.f;
  float toPlayerYaw = 0.f;
  float toPlayerPitch = 0.f;
  float relativeFacingYaw = 0.f;
  float relativeFacingPitch = 0.f;
};

struct CombatStateObservation {
  float hp = 0.f;
  float magazine = 0.f;
  float isReloading = 0.f;
  std::array<float, consts::maxFireQueueSize - 1> fireQueue = {};
  float timeBeforeAutoheal = 0.f;
};

// Observation state for the current agent.
// Positions are rescaled to the bounds of the play area to assist training.
struct SelfObservation : PlayerCommonObservation {
  CombatStateObservation combat;
  float fractionMatchRemaining;
  ZoneObservation zone;
};

struct TeammateObservation : public OtherPlayerCommonObservation {
  CombatStateObservation combat;
};

struct OpponentObservation : public OtherPlayerCommonObservation {
  float wasHit = 0.f;
  float firedShot = 0.f;
  float hasLOS = 0.f;
  float teamKnowsLocation = 0.f;
};

struct TeammateObservations {
    TeammateObservation obs[consts::maxTeamSize - 1];
};

struct OpponentObservations {
    OpponentObservation obs[consts::maxTeamSize];
};

struct OpponentLastKnownObservations {
    OpponentObservation obs[consts::maxTeamSize];
};

struct NormalizedPositionObservation {
    float x = -1000.f;
    float y = -1000.f;
    float z = -1000.f;
};

struct SelfPositionObservation {
  NormalizedPositionObservation ob;
};

struct TeammatePositionObservations {
    NormalizedPositionObservation obs[consts::maxTeamSize - 1];
};

struct OpponentPositionObservations {
    NormalizedPositionObservation obs[consts::maxTeamSize];
};

struct OpponentLastKnownPositionObservations {
    NormalizedPositionObservation obs[consts::maxTeamSize];
};

struct OpponentMasks {
    float masks[consts::maxTeamSize];
};

struct LidarData {
  float depth;
  float isWall;
  float isTeammate;
  float isOpponent;
};

// Linear depth values in a circle around the agent
struct FwdLidar {
    LidarData data[consts::fwdLidarHeight][consts::fwdLidarWidth];
};

struct RearLidar {
    LidarData data[consts::rearLidarHeight][consts::rearLidarWidth];
};

struct MapItem {
    float iAmPresent = 0;
    float numTeammatesPresent = 0;
    float numOpponentsPresent = 0;
    float numPastOpponentsPresent = 0;
};

struct AgentMap {
    static constexpr inline int res = 16;
    MapItem data[res][res];
};

struct UnmaskedAgentMap {
    MapItem data[AgentMap::res][AgentMap::res];
};

// Per-agent component storing Entity IDs of the other agents. Used to
// build the egocentric observations of their state.
struct Teammates {
    Entity e[consts::maxTeamSize - 1];
};

struct Opponents {
    Entity e[consts::maxTeamSize];
};

struct OpponentsVisibility {
    bool canSee[consts::maxTeamSize];
};

struct Aim {
    float yaw;
    float pitch;
    Quat rot;
};

struct CamRef {
    Entity camEntity;
};

struct AgentVelocity : madrona::math::Vector3 {
    inline AgentVelocity(madrona::math::Vector3 v)
        : Vector3(v)
    {}
};

struct IntermediateMoveState {
    Vector3 newPosition;
    Vector3 newVelocity;
};

struct HP {
    float hp;
};

struct Magazine {
    int32_t numBullets;
    int32_t isReloading;
};

struct DamageDealt {
    float dmg[consts::maxTeamSize];
};

struct Alive {
    float mask;
};

struct StartPos : public Vector3 {
    inline StartPos(Vector3 v)
        : Vector3(v)
    {}
};

struct ExploreTracker {
    static inline constexpr int32_t gridWidth = 81;
    static inline constexpr int32_t gridMaxX = gridWidth / 2;
    static inline constexpr int32_t gridHeight = 81;
    static inline constexpr int32_t gridMaxY = gridHeight / 2;

    uint32_t visited[gridHeight][gridWidth];
    uint32_t numNewCellsVisited;
};

struct CombatState {
    madrona::RNG rng;
    Entity landedShotOn;
    int32_t remainingRespawnSteps;
    int32_t remainingStepsBeforeAutoheal;
    bool fireQueue[consts::maxFireQueueSize];
    bool successfulKill;
    int32_t wasShotCount;
    bool wasKilled;
    float firedShotT;
    bool inZone;
    float minDistToZone;
    bool hasDiedDuringEpisode;
    bool reloadedFullMag;
    Vector3 immitationGoalPosition;
    float minDistToImmitationGoal;
    i32 weaponType;
};

struct ShotVizRemaining {
    int32_t numStepsRemaining;
};

struct ShotVizCleanupTracker {
    Entity e;
};

struct TurretState {
    madrona::RNG rng;
    int32_t offset;
};

struct ZoneState {
    int32_t curZone;
    int32_t curControllingTeam;
    bool isContested;
    bool isCaptured;
    bool earnedPoint;

    int32_t zoneStepsRemaining;
    int32_t stepsUntilPoint;
};

struct AgentLogData {
    Vector3 position;
    Aim aim;
    HP hp;
    Magazine mag;
    StandState standState;
    int32_t shotAgentIdx;
    float firedShotT;
    bool wasKilled;
    bool successfullKill;
};

struct StepLog {
    AgentLogData agentData[consts::maxTeamSize * 2];
};

struct EventLogGlobalState {
  u32 numEvents;
  u32 numStepStates;
};

struct EventPlayerState {
  i16 pos[3];
  i16 yaw;
  i16 pitch;
  u16 magNumBullets;
  u8 isReloading;
  u8 firedShot; 
};

enum class EventType : u32 {
  None = 0,
  Capture = 1 << 0,
  Reload = 1 << 1,
  Kill = 1 << 2,
  PlayerShot = 1 << 3,
};

struct EventStepState {
  u32 numEvents;
  u32 eventMask;
  u64 matchID;
  u32 step;
  u8 curZone;
  i8 curZoneController;
  EventPlayerState players[consts::maxTeamSize * 2];
};

struct EventStepStateEntity : madrona::Archetype<
  EventStepState
> {};

struct XYI16 {
  i16 x;
  i16 y;
};

struct TeamConvexHull {
  i16 numVerts;
  XYI16 verts[consts::maxTeamSize];
};

struct GameEvent {
  struct Capture {
    u8 zoneIDX;
    u8 captureTeam;
    u16 inZoneMask;
  };

  struct Reload {
    u8 player;
    u8 numBulletsAtReloadTime;
  };

  struct Kill {
    u8 killer;
    u8 killed;
  };

  struct PlayerShot {
    u8 attacker;
    u8 target;
  };

  EventType type;
  u64 matchID;
  u32 step;
  union {
    Capture capture;
    Reload reload;
    Kill kill;
    PlayerShot playerShot;
  };
};

struct GameEventEntity : madrona::Archetype<
    GameEvent
> {};

struct Breadcrumb {
    Vector3 pos;
    float penalty;
    TeamInfo teamInfo;
};

struct BreadcrumbAgentState {
    float totalPenalty;
    Entity lastBreadcrumb;
    int32_t stepsSinceLastNewBreadcrumb;
};

struct BreadcrumbDelete {
    Entity e;
};

struct BreadcrumbDeleteEntity : madrona::Archetype<
    BreadcrumbDelete
> {};

struct BreadcrumbEntity : madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,
    Breadcrumb
> {};

struct ZOBB {
    Vector3 pMin;
    Vector3 pMax;
    float rotation;
};

struct GoalRegion {
    static constexpr int maxSubRegions = 3;

    ZOBB subRegions[maxSubRegions];
    int32_t numSubRegions = 1;

    bool attackerTeam = true;
    float rewardStrength = 1.f;
};

struct GoalRegionsState {
    static constexpr int maxRegions = 10;

    bool regionsActive[maxRegions];
    float minDistToRegions[maxRegions];
    float teamStepRewards[2];
};

struct TaskConfig {
    bool autoReset;
    bool showSpawns;
    SimFlags simFlags;
    RandKey initRandKey;
    uint32_t numPBTPolicies;
    uint32_t policyHistorySize;
    
    AABB worldBounds;
    uint32_t pTeamSize;
    uint32_t eTeamSize;

    MeshBVH *staticMeshes;
    uint32_t numStaticMeshes;

    Navmesh navmesh;
    AStarLookup aStarLookup;

    StandardSpawns standardSpawns;
    SpawnCurriculum spawnCurriculum;

    Zones zones;

    Task task;
    bool highlevelMove;

    VizState *viz;

    RewardHyperParams *rewardHyperParams;

    StepLog *recordLog;
    StepLog *replayLog;

    EventLogGlobalState *eventGlobalState;

    GoalRegion *goalRegions;
    int32_t numGoalRegions;

    uint32_t numEpisodes;
    TDMEpisode *episodes;

    int32_t numWeaponTypes;
    WeaponStats * weaponTypeStats;
    TrainControl * trainControl;
};

/* ECS Archetypes for the game */

// There are 2 Agents in the environment trying to get to the destination
struct ExploreAgent : public madrona::Archetype<
    // Basic components required for physics. Note that the current physics
    // implementation requires archetypes to have these components first
    // in this exact order.
    Position,
    Rotation,
    Scale,
    AgentVelocity,
    ObjectID,
    ExternalForce,
    ExternalTorque,

    // Internal logic state.
    StartPos,
    ExploreTracker,

    // Input
    ExploreAction,

    // Observations
    SelfObservation,
    SelfPositionObservation,
    FwdLidar,

    // Reward, episode termination
    Reward,
    Done,

    // Visualization: In addition to the fly camera, src/viewer.cpp can
    // view the scene from the perspective of entities with this component
    VizCamera
> {};

struct CamEntity : public madrona::Archetype<
    Position,
    Rotation,
    Scale,

    // Visualization: In addition to the fly camera, src/viewer.cpp can
    // view the scene from the perspective of entities with this component
    VizCamera
> {};

struct PvPAgent : public madrona::Archetype<
    // Basic components required for physics. Note that the current physics
    // implementation requires archetypes to have these components first
    // in this exact order.
    Position,
    Rotation,
    Scale,
    AgentVelocity,
    IntermediateMoveState,
    ExternalForce,
    ExternalTorque,

    // Internal logic state.
    Teammates,
    Opponents,
    OpponentsVisibility,
    HP,
    Magazine,
    DamageDealt,
    Alive,
    Aim,
    CamRef,
    TeamInfo,
    CombatState,
    StandState,
    BreadcrumbAgentState,

    // Explore as well
    StartPos,
    ExploreTracker,

    // Input
    PvPAction,
    CoarsePvPAction,
    HardcodedBotAction,

    // Observations
    SelfObservation,
    TeammateObservations,
    OpponentObservations,
    OpponentLastKnownObservations,

    SelfPositionObservation,
    TeammatePositionObservations,
    OpponentPositionObservations,
    OpponentLastKnownPositionObservations,

    OpponentMasks,

    FwdLidar,
    RearLidar,
    AgentMap,
    UnmaskedAgentMap,

    // Training metadata
    Reward,
    Done,
    AgentPolicy
> {};

struct StaticGeometry : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    ObjectID
> {};

struct ShotVizState {
  Vector3 from;
  Vector3 dir;
  float hitT;
  int team;
  bool hit;
};

struct ShotViz : public madrona::Archetype<
    ShotVizState,
    ShotVizRemaining
> {};

struct ShotVizCleanup : public madrona::Archetype<
    ShotVizCleanupTracker
> {};

struct Turret : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,
    HP,
    Magazine,
    Alive,
    Aim,
    DamageDealt,
    TurretState
> {};

struct ZoneViz : public madrona::Archetype<
    Position,
    Rotation,
    Scale
> {};

struct FullTeamID {
  i32 id;
};

struct FullTeamActions {
  PvPAction actions[consts::maxTeamSize];
};

struct FullTeamZoneObservation {
    float centerX;
    float centerY;
    float centerZ;
    float myTeamControlling;
    float enemyTeamControlling;
    float isContested;
    float isCaptured;
    float stepsUntilPoint;
    float stepsRemaining;
    std::array<float, 4> id;
};

struct FullTeamGlobalObservation {
  std::array<float, 2> teamID;
  float fractionMatchRemaining;
  FullTeamZoneObservation zone;
};

struct FullTeamCommonObservation {
  float isValid = 0.f; // Must be first observation
  std::array<float, consts::maxTeamSize> id = {};
  float isAlive = 0.f;
  float globalX = 0.f;
  float globalY = 0.f;
  float globalZ = 0.f;
  float facingYaw = 0.f;
  float facingPitch = 0.f;
  float velocityX = 0.f;
  float velocityY = 0.f;
  float velocityZ = 0.f;
  StandObservation stand = {};
  float inZone = 0.f;
};

#ifndef MADRONA_GPU_MODE
static_assert(
    offsetof(FullTeamCommonObservation, globalX) == 8 * sizeof(float));
#endif

struct FullTeamPlayerObservation : public FullTeamCommonObservation {
  float hp = 0.f;
  float magazine = 0.f;
  float isReloading = 0.f;
  std::array<float, consts::maxFireQueueSize - 1> fireQueue = {};
  float timeBeforeAutoheal = 0.f;
};

struct FullTeamEnemyObservation : public FullTeamCommonObservation {
  float wasHit = 0.f;
  float firedShot = 0.f;
  std::array<float, consts::maxTeamSize> hasLOS = {};
  float teamKnowsLocation = 0.f; // Must be last observation
};

struct FullTeamPlayerObservations {
  FullTeamPlayerObservation obs[consts::maxTeamSize];
};

struct FullTeamEnemyObservations {
  FullTeamEnemyObservation obs[consts::maxTeamSize];
};

struct FullTeamLastKnownEnemyObservations {
  FullTeamCommonObservation obs[consts::maxTeamSize];
};

struct FullTeamFwdLidar {
  FwdLidar obs[consts::maxTeamSize];
};

struct FullTeamRearLidar {
  RearLidar obs[consts::maxTeamSize];
};

struct FullTeamReward {
  float v;
};

struct FullTeamDone {
  i32 d;
};

struct FullTeamPolicy {
  i32 idx;
};

static_assert(sizeof(FullTeamPlayerObservations) ==
              sizeof(FullTeamPlayerObservation) * consts::maxTeamSize);

static_assert(sizeof(FullTeamEnemyObservations) ==
              sizeof(FullTeamEnemyObservation) * consts::maxTeamSize);

struct FullTeamInterface : public madrona::Archetype<
  FullTeamID,
  FullTeamActions,
  FullTeamGlobalObservation,
  FullTeamPlayerObservations,
  FullTeamEnemyObservations,
  FullTeamLastKnownEnemyObservations,
  FullTeamFwdLidar,
  FullTeamRearLidar,
  FullTeamReward,
  FullTeamDone,
  FullTeamPolicy
> {};

}
