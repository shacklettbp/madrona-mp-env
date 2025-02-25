#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/custom_context.hpp>

#include "consts.hpp"
#include "types.hpp"

namespace madronaMPEnv {

class Engine;

// This enum is used by the Sim and Manager classes to track the export slots
// for each component exported to the training code.
enum class ExportID : uint32_t {
    Reset,
    WorldCurriculum,
    ExploreAction,
    PvPDiscreteAction,
    PvPAimAction,
    PvPDiscreteAimAction,
    Reward,
    Done,
    MatchResult,
    AgentPolicy,

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
    HP,
    Alive,
    Magazine,

    FullTeamActions,
    FullTeamGlobal,
    FullTeamPlayers,
    FullTeamEnemies,
    FullTeamLastKnownEnemies,
    FullTeamFwdLidar,
    FullTeamRearLidar,
    FullTeamReward,
    FullTeamDone,
    FullTeamPolicyAssignments,

    EventLog,
    PackedStepSnapshot,

    FiltersStateObservation,

    RewardHyperParams,

    NumExports,
};

enum class TaskGraphID : uint32_t {
  Init,
  Step,
  NumGraphs,
};

// The Sim class encapsulates the per-world state of the simulation.
// Sim is always available by calling ctx.data() given a reference
// to the Engine / Context object that is passed to each ECS system.
//
// Per-World state that is frequently accessed but only used by a few
// ECS systems should be put in a singleton component rather than
// in this class in order to ensure efficient access patterns.
struct Sim : public madrona::WorldBase {
    struct WorldInit {};

    // Sim::registerTypes is called during initialization
    // to register all components & archetypes with the ECS.
    static void registerTypes(madrona::ECSRegistry &registry,
                              const TaskConfig &cfg);

    // Sim::setupTasks is called during initialization to build
    // the system task graph that will be invoked by the 
    // Manager class (src/mgr.hpp) for each step.
    static void setupTasks(madrona::TaskGraphManager &taskgraph_mgr,
                           const TaskConfig &cfg);

    // The constructor is called for each world during initialization.
    // TaskConfig is global across all worlds, while WorldInit (src/init.hpp)
    // can contain per-world initialization data, created in (src/mgr.cpp)
    Sim(Engine &ctx,
        const TaskConfig &cfg,
        const WorldInit &init);

    AABB worldBounds;
    float maxDist;
    float distScale;

    Task taskType;
    uint32_t pTeamSize;
    uint32_t eTeamSize;

    uint32_t numPBTPolicies;
    uint32_t policyHistorySize;

    madrona::RandKey initRandKey;

    uint32_t curEpisodeIdx;
    uint32_t worldEpisodeCounter;
    madrona::RNG baseRNG;

    // Agent entity references. This entities live across all episodes
    // and are just reset to the start of the level on reset.
    Entity *agents;
    uint32_t numAgents;

    Entity *turrets;
    uint32_t numTurrets;

    MeshBVH *staticMeshes;
    uint32_t numStaticMeshes;

    Entity *staticEntities;

    uint32_t *navmeshQueryQueueData;
    bool *navmeshQueryVisitedData;

    Zones zones;
    // Viz only
    Entity zoneEntities[consts::maxZones];

    RewardHyperParams *rewardHyperParams;

    StepLog *recordLog;
    StepLog *replayLog;

    madrona::math::Vector4 frustumData;

    // Should the environment automatically reset (generate a new episode)
    // at the end of each episode?
    bool autoReset;

    SimFlags simFlags;

    int32_t hardcodedSpawnIdx;

    // Are we visualizing the simulation in the viewer?
    bool enableVizRender;

    std::array<ZoneStats, consts::maxZones> zoneStats;

    u64 matchID;

    EventLogGlobalState *eventGlobalState;
    u32 eventLoggedInStep;
    u32 eventMask;

    GoalRegion *goalRegions;
    int32_t numGoalRegions;

    uint32_t numEpisodes;
    TDMEpisode *episodes;

    uint32_t numWeaponTypes;
    WeaponStats * weaponTypeStats;

    Entity teamInterfaces[consts::numTeams];

    TrainControl *trainControl;
    PolicyWeights *policyWeights;

    TrajectoryCurriculum trajectoryCurriculum;

    std::array<FiltersMatchState, 2> filtersState;
    std::array<int, 2> filtersLastMatchedStep;

    WorldCurriculum episodeCurriculum = WorldCurriculum::LearnShooting;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
public:
    using CustomContext::CustomContext;
};

}

#include "sim.inl"
