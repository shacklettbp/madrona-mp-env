#pragma once

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include "types.hpp"
#include "sim_flags.hpp"
#include "viz.hpp"
#include "sim.hpp"

namespace madronaMPEnv {

struct MapConfig {
    const char *name;
    const char *collisionDataFile;
    const char *navmeshFile;
    const char *spawnDataFile;
    const char *zoneDataFile;
    madrona::math::Vector3 mapOffset;
    float mapRotation;
};

// The Manager class encapsulates the linkage between the outside training
// code and the internal simulation state (src/sim.hpp / src/sim.cpp)
//
// Manager is responsible for initializing the simulator, loading physics
// and rendering assets off disk, and mapping ECS components to tensors
// for learning
class Manager {
public:
    struct Config {
        madrona::ExecMode execMode; // CPU or CUDA
        int gpuID; // Which GPU for CUDA backend?
        uint32_t numWorlds; // Simulation batch size
        uint32_t randSeed;
        bool autoReset; // Immediately generate new world on episode end
        SimFlags simFlags;
        Task taskType;
        uint32_t teamSize;
        uint32_t numPBTPolicies;
        uint32_t policyHistorySize;
        MapConfig map;
        bool highlevelMove = true;
        const char *replayLogPath = nullptr;
        const char *recordLogPath = nullptr;
        const char *eventLogPath = nullptr;
        const char *curriculumDataPath = nullptr;
    };

    Manager(
        const Config &cfg,
        VizState *viz = nullptr);
    ~Manager();

    void init();
    void step();

    void vizStep();

    inline void cpuJAXInit(void **, void **) {};
    inline void cpuJAXStep(void **, void **) {};

#ifdef MADRONA_CUDA_SUPPORT
    void gpuStreamInit(cudaStream_t strm, void **buffers);
    void gpuStreamStep(cudaStream_t strm, void **buffers);
#endif

    // These functions export Tensor objects that link the ECS
    // simulation state to the python bindings / PyTorch tensors (src/bindings.cpp)
    madrona::py::Tensor resetTensor() const;
    madrona::py::Tensor simControlTensor() const;
    madrona::py::Tensor matchResultTensor() const;
    madrona::py::Tensor pvpDiscreteActionTensor() const;
    madrona::py::Tensor pvpAimActionTensor() const;
    madrona::py::Tensor exploreActionTensor() const;
    madrona::py::Tensor rewardTensor() const;
    madrona::py::Tensor doneTensor() const;
    madrona::py::Tensor policyAssignmentTensor() const;

    madrona::py::Tensor selfObservationTensor() const;
    madrona::py::Tensor filtersStateObservationTensor() const;
    madrona::py::Tensor teammateObservationsTensor() const;
    madrona::py::Tensor opponentObservationsTensor() const;
    madrona::py::Tensor opponentLastKnownObservationsTensor() const;

    madrona::py::Tensor selfPositionTensor() const;
    madrona::py::Tensor teammatePositionObservationsTensor() const;
    madrona::py::Tensor opponentPositionObservationsTensor() const;
    madrona::py::Tensor opponentLastKnownPositionObservationsTensor() const;


    madrona::py::Tensor opponentMasksTensor() const;

    madrona::py::Tensor fwdLidarTensor() const;
    madrona::py::Tensor rearLidarTensor() const;

    madrona::py::Tensor agentMapTensor() const;
    madrona::py::Tensor unmaskedAgentMapTensor() const;
    madrona::py::Tensor hpTensor() const;
    madrona::py::Tensor magazineTensor() const;
    madrona::py::Tensor aliveTensor() const;
    madrona::py::Tensor rewardHyperParamsTensor() const;

    madrona::py::Tensor fullTeamActionTensor() const;
    madrona::py::Tensor fullTeamGlobalObservationsTensor() const;
    madrona::py::Tensor fullTeamPlayerObservationsTensor() const;
    madrona::py::Tensor fullTeamEnemyObservationsTensor() const;
    madrona::py::Tensor fullTeamLastKnownEnemyObservationsTensor() const;
    madrona::py::Tensor fullTeamFwdLidarTensor() const;
    madrona::py::Tensor fullTeamRearLidarTensor() const;

    madrona::py::Tensor fullTeamRewardTensor() const;
    madrona::py::Tensor fullTeamDoneTensor() const;
    madrona::py::Tensor fullTeamPolicyAssignmentTensor() const;

    madrona::py::TrainInterface trainInterface() const;

    madrona::ExecMode execMode() const;

    Engine & getWorldContext(int32_t world_idx);

    // These functions are used by the viewer to control the simulation
    // with keyboard inputs in place of DNN policy actions
    void triggerReset(int32_t world_idx);

    void setExploreAction(
        int32_t world_idx,
        ExploreAction action);

    void setPvPAction(
        int32_t world_idx,
        int32_t agent_idx,
        PvPDiscreteAction discrete,
        PvPAimAction aim);

    void setCoarsePvPAction(
        int32_t world_idx,
        int32_t agent_idx,
        CoarsePvPAction action);

    void setHP(int32_t world_idx, int32_t agent_idx, int32_t hp);

    bool isReplayFinished();

    void setUniformAgentPolicy(AgentPolicy policy);

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    std::unique_ptr<Impl> impl_;
};

}
