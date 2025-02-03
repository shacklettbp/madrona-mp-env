#include "mgr.hpp"
#include "sim.hpp"
#include "map_importer.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_assets.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>

#include <array>
#include <cassert>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <set>

#ifdef MADRONA_LINUX
#include <unistd.h>
#endif

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

#include <meshoptimizer.h>

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;

namespace madronaMPEnv {

[[maybe_unused]] static inline  uint64_t numTensorBytes(const Tensor &t)
{
    uint64_t num_items = 1;
    uint64_t num_dims = t.numDims();
    for (uint64_t i = 0; i < num_dims; i++) {
        num_items *= t.dims()[i];
    }

    return num_items * (uint64_t)t.numBytesPerItem();
}

struct MapData {
    HeapArray<void *> buffers;
    HeapArray<MeshBVH> collisionBVHs;
};

struct Manager::Impl {
    Config cfg;
    WorldReset *worldResetBuffer;
    ExploreAction *exploreActionsBuffer;
    void *pvpActionsBuffer;
    uint32_t numAgentsPerWorld;
    TrainControl *trainCtrl;

    inline Impl(const Manager::Config &mgr_cfg,
                WorldReset *reset_buffer,
                ExploreAction *explore_action_buffer,
                void *pvp_action_buffer,
                TrainControl *train_ctrl)
      : cfg(mgr_cfg),
        worldResetBuffer(reset_buffer),
        exploreActionsBuffer(explore_action_buffer),
        pvpActionsBuffer(pvp_action_buffer),
        numAgentsPerWorld(
            mgr_cfg.taskType == Task::Explore ? 1 : (
                (mgr_cfg.taskType == Task::TDM || 
                 mgr_cfg.taskType == Task::Zone ||
                 mgr_cfg.taskType == Task::ZoneCaptureDefend) ? 
                (mgr_cfg.teamSize * 2) :
                mgr_cfg.teamSize)
        ),
        trainCtrl(train_ctrl)
    {}

    inline virtual ~Impl() {}

    virtual void run(TaskGraphID taskgraph_id = TaskGraphID::Step) = 0;

#ifdef MADRONA_CUDA_SUPPORT
    virtual void gpuStreamInit(cudaStream_t strm, void **buffers, Manager &) = 0;
    virtual void gpuStreamStep(cudaStream_t strm, void **buffers, Manager &) = 0;
#endif

    virtual Tensor exportTensor(ExportID slot,
                                TensorElementType type,
                                madrona::Span<const int64_t> dimensions) const = 0;

    virtual Tensor rewardHyperParamsTensor() const = 0;
    virtual Tensor simControlTensor() const = 0;

    static inline Impl * init(const Config &cfg,
                              VizState *viz);
};

static void writeGameEvents(
    std::fstream &event_log_file,
    std::fstream &step_log_file,
    const GameEvent *events, uint32_t num_events,
    const PackedStepSnapshot *step_states, uint32_t num_step_states,
    CountT num_worlds, int32_t step_idx)
{
  (void)num_worlds;
  (void)step_idx;

  step_log_file.write((char *)step_states, sizeof(PackedStepSnapshot) * num_step_states);
  event_log_file.write((char *)events, sizeof(GameEvent) * num_events);
}
    

#if 0
static void writeEventsCSV(
    std::fstream &log_file, const StepEvents *world_events, CountT num_worlds, int32_t step_idx)
{
    std::string line;

    for (CountT world_idx = 0; world_idx < num_worlds; world_idx++) {
        const GameEvent *step_events = world_events[world_idx].events;
        CountT num_step_events = (CountT)world_events[world_idx].numEvents;

        for (CountT event_idx = 0; event_idx < num_step_events; event_idx++) {
            line.clear();

            GameEvent event = step_events[event_idx];

            line += "zone_captured,";
            line += std::to_string(world_idx) + ",";
            line += std::to_string(step_idx) + ",";
            line += std::to_string(event.capture.pos.x) + ",";
            line += std::to_string(event.capture.pos.y) + ",";
            line += std::to_string(event.capture.pos.z) + ",";
            line += std::to_string(event.capture.playerIdx);
            line += "\n";

            log_file << line;
        }
    }
}
#endif

struct Manager::CPUImpl final : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, TaskConfig, Sim::WorldInit>;

    TaskGraphT cpuExec;
    RewardHyperParams *rewardHyperParams;

    StepLog *replayLogBuffer;
    StepLog *recordLogBuffer;

    GameEvent *eventLogExported;
    PackedStepSnapshot *eventLogStepStatesExported;
    EventLogGlobalState *eventLogGlobalState;
    TrajectoryCurriculum trajectoryCurriculum;

    Optional<std::fstream> replayLog;
    Optional<std::fstream> recordLog;
    Optional<std::fstream> eventLog;
    Optional<std::fstream> eventStepsLog;

    i32 stepIdx;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   WorldReset *reset_buffer,
                   ExploreAction *explore_action_buffer,
                   void *pvp_action_buffer,
                   TrainControl *train_ctrl,
                   RewardHyperParams *reward_hyper_params,
                   TaskGraphT &&cpu_exec,
                   StepLog *replay_log_buffer,
                   StepLog *record_log_buffer,
                   EventLogGlobalState *event_log_global_state,
                   TrajectoryCurriculum trajectory_curriculum,
                   const char *replay_log_path,
                   const char *record_log_path,
                   const char *event_log_path)
        : Impl(mgr_cfg, reset_buffer,
               explore_action_buffer, pvp_action_buffer, train_ctrl),
        cpuExec(std::move(cpu_exec)),
        rewardHyperParams(reward_hyper_params),
        replayLogBuffer(replay_log_buffer),
        recordLogBuffer(record_log_buffer),
        eventLogExported(
          (GameEvent *)cpuExec.getExported((CountT)ExportID::EventLog)),
        eventLogStepStatesExported(
          (PackedStepSnapshot *)cpuExec.getExported((CountT)ExportID::PackedStepSnapshot)),
        eventLogGlobalState(event_log_global_state),
        trajectoryCurriculum(trajectory_curriculum),
        replayLog(Optional<std::fstream>::none()),
        recordLog(Optional<std::fstream>::none()),
        eventLog(Optional<std::fstream>::none()),
        eventStepsLog(Optional<std::fstream>::none()),
        stepIdx(0)
    {
        if (replayLogBuffer) {
            replayLog.emplace(replay_log_path,
                              std::ios::binary | std::ios::in);

            if (!replayLog->is_open()) {
                FATAL("Failed to open replay log %s", replay_log_path);
            }
        }

        if (recordLogBuffer) {
            recordLog.emplace(record_log_path,
                              std::ios::binary | std::ios::out);

            if (!recordLog->is_open()) {
                FATAL("Failed to open record log %s", record_log_path);
            }
        }

        if (event_log_path) {
            std::string dir_str = event_log_path;
            eventLog.emplace(dir_str + "/events.bin",
                             std::ios::out | std::ios::binary);

            if (!eventLog->is_open()) {
                FATAL("Failed to open record log %s", event_log_path);
            }

            eventStepsLog.emplace(dir_str + "/steps.bin",
                                  std::ios::out | std::ios::binary);
        }
    }

    inline virtual ~CPUImpl() final
    {
        free(rewardHyperParams);
        free(trajectoryCurriculum.snapshots);
    }

    inline virtual void run(TaskGraphID graph_id)
    {
        if (graph_id == TaskGraphID::Step) {
            if (replayLogBuffer) {
                replayLog->read((char *)replayLogBuffer,
                                sizeof(StepLog) * cfg.numWorlds);
            }
        }

        cpuExec.runTaskGraph(graph_id);

        if (graph_id == TaskGraphID::Step) {
            if (recordLogBuffer) {
                recordLog->write((char *)recordLogBuffer,
                                 sizeof(StepLog) * cfg.numWorlds);
            }

            if (eventLog.has_value()) {
              writeGameEvents(*eventLog,
                              *eventStepsLog,
                              eventLogExported,
                              eventLogGlobalState->numEvents, 
                              eventLogStepStatesExported,
                              eventLogGlobalState->numStepStates, 
                              cfg.numWorlds, stepIdx);
              eventLogGlobalState->numEvents = 0;
              eventLogGlobalState->numStepStates = 0;
            }
        }

        stepIdx += 1;
    }

#ifdef MADRONA_CUDA_SUPPORT
    virtual void gpuStreamInit(cudaStream_t, void **, Manager &)
    {
        assert(false);
    }

    virtual void gpuStreamStep(cudaStream_t, void **, Manager &)
    {
        assert(false);
    }
#endif

    virtual Tensor rewardHyperParamsTensor() const final
    {
        return Tensor(rewardHyperParams, TensorElementType::Float32,
                      {
                          cfg.numPBTPolicies,
                          sizeof(RewardHyperParams) / sizeof(float),
                      }, Optional<int>::none());
    }

    virtual Tensor simControlTensor() const final
    {
        return Tensor(trainCtrl, TensorElementType::Int32,
                      {
                          sizeof(TrainControl) / sizeof(int32_t),
                      }, Optional<int>::none());
    }

    virtual inline Tensor exportTensor(ExportID slot,
                                       TensorElementType type,
                                       madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = cpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

#if defined(MADRONA_CUDA_SUPPORT) && defined(ENABLE_MWGPU)
struct Manager::CUDAImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph stepGraph;
    RewardHyperParams *rewardHyperParams;

    StepLog *replayLogBuffer;
    StepLog *recordLogBuffer;

    StepLog *replayLogStaging;
    StepLog *recordLogStaging;

    GameEvent *eventLogExported;
    PackedStepSnapshot *eventLogStepStatesExported;
    EventLogGlobalState *eventLogGlobalState;

    GameEvent *eventLogReadBack;
    PackedStepSnapshot *eventLogStepStatesReadback;
    uint32_t eventLogReadBackSize;
    uint32_t eventLogStepStatesReadBackSize;

    EventLogGlobalState *eventLogGlobalStateReadback;
    TrajectoryCurriculum trajectoryCurriculum;

    Optional<std::fstream> replayLog;
    Optional<std::fstream> recordLog;
    Optional<std::fstream> eventLog;
    Optional<std::fstream> eventStepsLog;

    RandKey staggerShuffleRND;
    int32_t stepIdx;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                    WorldReset *reset_buffer,
                    ExploreAction *explore_action_buffer,
                    void *pvp_action_buffer,
                    TrainControl *train_ctrl,
                    RewardHyperParams *reward_hyper_params,
                    MWCudaExecutor &&gpu_exec,
                    StepLog *replay_log_buffer,
                    StepLog *record_log_buffer,
                    EventLogGlobalState *event_log_global_state,
                    TrajectoryCurriculum trajectory_curriculum,
                    const char *replay_log_path,
                    const char *record_log_path,
                    const char *event_log_path,
                    RandKey stagger_shuffle_key)
        : Impl(mgr_cfg, reset_buffer,
               explore_action_buffer, pvp_action_buffer,
               train_ctrl),
        gpuExec(std::move(gpu_exec)),
        stepGraph(gpuExec.buildLaunchGraph(TaskGraphID::Step)),
        rewardHyperParams(reward_hyper_params),
        replayLogBuffer(replay_log_buffer),
        recordLogBuffer(record_log_buffer),
        replayLogStaging(nullptr),
        recordLogStaging(nullptr),
        eventLogExported(
            (GameEvent *)gpuExec.getExported((CountT)ExportID::EventLog)),
        eventLogStepStatesExported(
          (PackedStepSnapshot *)gpuExec.getExported((CountT)ExportID::PackedStepSnapshot)),
        eventLogGlobalState(event_log_global_state),
        eventLogReadBack(nullptr),
        eventLogStepStatesReadback(nullptr),
        eventLogReadBackSize(0),
        eventLogStepStatesReadBackSize(0),
        eventLogGlobalStateReadback(nullptr),
        trajectoryCurriculum(trajectory_curriculum),
        replayLog(Optional<std::fstream>::none()),
        recordLog(Optional<std::fstream>::none()),
        eventLog(Optional<std::fstream>::none()),
        eventStepsLog(Optional<std::fstream>::none()),
        staggerShuffleRND(stagger_shuffle_key),
        stepIdx(0)
    {
        if (replayLogBuffer) {
            replayLogStaging = (StepLog *)cu::allocStaging(
                sizeof(StepLog) * mgr_cfg.numWorlds);

            replayLog.emplace(replay_log_path,
                              std::ios::binary | std::ios::in);
        }

        if (recordLogBuffer) {
            recordLogStaging = (StepLog *)cu::allocReadback(
                sizeof(StepLog) * mgr_cfg.numWorlds);

            recordLog.emplace(record_log_path,
                              std::ios::binary | std::ios::out);
        }

        if (event_log_path) {
            std::string dir_str = event_log_path;
            eventLog.emplace(dir_str + "/events.bin",
                             std::ios::out | std::ios::binary);

            if (!eventLog->is_open()) {
                FATAL("Failed to open events file %s", event_log_path);
            }

            eventStepsLog.emplace(dir_str + "/steps.bin",
                                  std::ios::out | std::ios::binary);

            if (!eventStepsLog->is_open()) {
                FATAL("Failed to open event steps file %s", event_log_path);
            }

            eventLogGlobalStateReadback =
              (EventLogGlobalState *)cu::allocReadback(sizeof(EventLogGlobalState));
        }
    }

    inline virtual ~CUDAImpl() final
    {
        REQ_CUDA(cudaFree(trajectoryCurriculum.snapshots));
        REQ_CUDA(cudaFree(rewardHyperParams));
    }

    inline void saveEvents(cudaStream_t strm)
    {
      if (!eventLog.has_value()) {
        return;
      }

      cudaMemcpyAsync(eventLogGlobalStateReadback,
                 eventLogGlobalState,
                 sizeof(EventLogGlobalState),
                 cudaMemcpyDeviceToHost,
                 strm);

      REQ_CUDA(cudaStreamSynchronize(strm));

      u32 num_events = eventLogGlobalStateReadback->numEvents;
      u32 num_steps = eventLogGlobalStateReadback->numStepStates;

      if (num_events == 0 && num_steps == 0) {
        return;
      }

      cudaMemsetAsync(eventLogGlobalState, 0, sizeof(EventLogGlobalState),
                      strm);

      if (num_events > eventLogReadBackSize) {
        cu::deallocCPU(eventLogReadBack);

        eventLogReadBack = (GameEvent *)cu::allocReadback(
            sizeof(GameEvent) * num_events);
      }

      if (num_steps > eventLogStepStatesReadBackSize) {
        cu::deallocCPU(eventLogStepStatesReadback);

        eventLogStepStatesReadback = (PackedStepSnapshot *)cu::allocReadback(
            sizeof(PackedStepSnapshot) * num_steps);
      }

      cudaMemcpyAsync(eventLogReadBack, eventLogExported,
                 sizeof(GameEvent) * num_events,
                 cudaMemcpyDeviceToHost,
                 strm);

      cudaMemcpyAsync(eventLogStepStatesReadback, eventLogStepStatesExported,
                      sizeof(PackedStepSnapshot) * num_steps,
                      cudaMemcpyDeviceToHost,
                      strm);

      REQ_CUDA(cudaStreamSynchronize(strm));

      writeGameEvents(*eventLog, *eventStepsLog, eventLogReadBack, num_events, 
                      eventLogStepStatesReadback, num_steps,
                      cfg.numWorlds, stepIdx);
    }

    inline virtual void run(TaskGraphID graph_id)
    {
        if (replayLogBuffer) {
            replayLog->read((char *)replayLogStaging,
                            sizeof(StepLog) * cfg.numWorlds);

            REQ_CUDA(cudaMemcpy(replayLogBuffer, replayLogStaging,
                                sizeof(StepLog) * cfg.numWorlds,
                                cudaMemcpyHostToDevice));
        }

        assert(graph_id == TaskGraphID::Step);
        gpuExec.run(stepGraph);

        if (recordLogBuffer) {
            REQ_CUDA(cudaMemcpy(recordLogStaging, recordLogBuffer,
                                sizeof(StepLog) * cfg.numWorlds,
                                cudaMemcpyDeviceToHost));

            recordLog->write((char *)recordLogStaging,
                             sizeof(StepLog) * cfg.numWorlds);
        }

        saveEvents(nullptr);

        stepIdx++;
    }

    inline void ** copyOutObservations(cudaStream_t strm,
                                       void **buffers,
                                       Manager &mgr)
    {
        auto copyFromSim = [&strm](void *dst, const Tensor &src) {
            uint64_t num_bytes = numTensorBytes(src);

            REQ_CUDA(cudaMemcpyAsync(dst, src.devicePtr(), num_bytes,
                                     cudaMemcpyDeviceToDevice, strm));
            REQ_CUDA(cudaStreamSynchronize(strm));
        };

        // Observations
        if ((cfg.simFlags & SimFlags::FullTeamPolicy) ==
            SimFlags::FullTeamPolicy) {
            copyFromSim(*buffers++, mgr.fullTeamGlobalObservationsTensor());
            copyFromSim(*buffers++, mgr.fullTeamPlayerObservationsTensor());
            copyFromSim(*buffers++, mgr.fullTeamEnemyObservationsTensor());
            copyFromSim(*buffers++, mgr.fullTeamLastKnownEnemyObservationsTensor());
            copyFromSim(*buffers++, mgr.fullTeamFwdLidarTensor());
            copyFromSim(*buffers++, mgr.fullTeamRearLidarTensor());
        } else {
            copyFromSim(*buffers++, mgr.fwdLidarTensor());
            copyFromSim(*buffers++, mgr.rearLidarTensor());
            copyFromSim(*buffers++, mgr.hpTensor());
            copyFromSim(*buffers++, mgr.magazineTensor());
            copyFromSim(*buffers++, mgr.aliveTensor());

            copyFromSim(*buffers++, mgr.selfObservationTensor());
            copyFromSim(*buffers++, mgr.filtersStateObservationTensor());
            copyFromSim(*buffers++, mgr.teammateObservationsTensor());
            copyFromSim(*buffers++, mgr.opponentObservationsTensor());
            copyFromSim(*buffers++, mgr.opponentLastKnownObservationsTensor());

            copyFromSim(*buffers++, mgr.selfPositionTensor());
            copyFromSim(*buffers++, mgr.teammatePositionObservationsTensor());
            copyFromSim(*buffers++, mgr.opponentPositionObservationsTensor());
            copyFromSim(*buffers++, mgr.opponentLastKnownPositionObservationsTensor());

            copyFromSim(*buffers++, mgr.opponentMasksTensor());

            copyFromSim(*buffers++, mgr.agentMapTensor());
            copyFromSim(*buffers++, mgr.unmaskedAgentMapTensor());
        }

        return buffers;
    }

    virtual void gpuStreamInit(cudaStream_t strm, void **buffers, Manager &mgr)
    {
#if 0
        {
            volatile int should_breakpoint = 1;

            while (should_breakpoint) {
                printf("Loop\n");
            }
        }
#endif

        printf("Sim Stream Init Start\n");
        if (replayLogBuffer) {
            replayLog->read((char *)replayLogStaging,
                            sizeof(StepLog) * cfg.numWorlds);

            REQ_CUDA(cudaMemcpyAsync(replayLogBuffer, replayLogStaging,
                                     sizeof(StepLog) * cfg.numWorlds,
                                     cudaMemcpyHostToDevice, strm));
        }

        HeapArray<WorldReset> resets_staging(cfg.numWorlds);
        for (CountT i = 0; i < (CountT)cfg.numWorlds; i++) {
            resets_staging[i].reset = 1;
        }

        auto resetSyncStep = [&]()
        {
            cudaMemcpyAsync(worldResetBuffer, resets_staging.data(),
                            sizeof(WorldReset) * cfg.numWorlds,
                            cudaMemcpyHostToDevice, strm);
            auto init_graph = gpuExec.buildLaunchGraph(TaskGraphID::Init);
            gpuExec.runAsync(init_graph, strm);
            REQ_CUDA(cudaStreamSynchronize(strm));
        };

        resetSyncStep();
#if 0
        if ((cfg.simFlags & SimFlags::StaggerStarts) ==
            SimFlags::StaggerStarts) {
            HeapArray<int32_t> steps_before_reset(cfg.numWorlds);

            CountT cur_world_idx = 0;
            CountT step_idx;
            for (step_idx = 0;
                 step_idx < (CountT)consts::episodeLen;
                 step_idx++) {
                CountT worlds_remaining = 
                    (CountT)cfg.numWorlds - cur_world_idx;
                CountT episode_steps_remaining = consts::episodeLen - step_idx;

                CountT worlds_per_step = madrona::utils::divideRoundUp(
                    worlds_remaining, episode_steps_remaining);

                bool finished = false;
                for (CountT i = 0; i < worlds_per_step; i++) {
                    if (cur_world_idx >= (CountT)cfg.numWorlds) {
                        finished = true;
                        break;
                    }

                    steps_before_reset[cur_world_idx++] = step_idx;
                }

                if (finished || worlds_per_step == 0) {
                    break;
                }
            }

            assert(cur_world_idx == (CountT)cfg.numWorlds);

            for (int32_t i = 0; i < (int32_t)cfg.numWorlds - 1; i++) {
                int32_t j = rand::sampleI32(
                    rand::split_i(staggerShuffleRND, i), i, cfg.numWorlds);

                std::swap(steps_before_reset[i], steps_before_reset[j]);
            }


            CountT max_steps = step_idx;
            for (CountT step = 0; step < max_steps; step++) {
                for (CountT world_idx = 0; world_idx < (CountT)cfg.numWorlds;
                     world_idx++) {
                    if (steps_before_reset[world_idx] == step) {
                        resets_staging[world_idx].reset = 1;
                    } else {
                        resets_staging[world_idx].reset = 0;
                    }
                }

                resetSyncStep();
            }
        }
#endif

        copyOutObservations(strm, buffers, mgr);

        if (recordLogBuffer) {
            REQ_CUDA(cudaMemcpyAsync(recordLogStaging, recordLogBuffer,
                                     sizeof(StepLog) * cfg.numWorlds,
                                     cudaMemcpyDeviceToHost, strm));

            REQ_CUDA(cudaStreamSynchronize(strm));

            recordLog->write((char *)recordLogStaging,
                             sizeof(StepLog) * cfg.numWorlds);
        }

        if (eventLog.has_value()) {
          cudaMemsetAsync(eventLogGlobalState, 0, sizeof(EventLogGlobalState),
                      strm);
        }

        printf("Sim Stream Init Finished\n");
    }

    virtual void gpuStreamStep(cudaStream_t strm, void **buffers, Manager &mgr)
    {
        if (replayLogBuffer) {
            replayLog->read((char *)replayLogStaging,
                            sizeof(StepLog) * cfg.numWorlds);

            REQ_CUDA(cudaMemcpyAsync(replayLogBuffer, replayLogStaging,
                                     sizeof(StepLog) * cfg.numWorlds,
                                     cudaMemcpyHostToDevice, strm));
        }

        auto copyToSim = [&strm](const Tensor &dst, void *src) {
            uint64_t num_bytes = numTensorBytes(dst);

            REQ_CUDA(cudaMemcpyAsync(dst.devicePtr(), src, num_bytes,
                                     cudaMemcpyDeviceToDevice, strm));
        };

        if ((cfg.simFlags & SimFlags::FullTeamPolicy) ==
            SimFlags::FullTeamPolicy) {
            copyToSim(mgr.fullTeamActionTensor(), *buffers++);
        } else {
            copyToSim(mgr.pvpDiscreteActionTensor(), *buffers++);
            copyToSim(mgr.pvpAimActionTensor(), *buffers++);
        }
        copyToSim(mgr.resetTensor(), *buffers++);
        copyToSim(mgr.simControlTensor(), *buffers++);

        if (cfg.numPBTPolicies > 0) {
            if ((cfg.simFlags & SimFlags::FullTeamPolicy) ==
                SimFlags::FullTeamPolicy) {
                copyToSim(mgr.fullTeamPolicyAssignmentTensor(), *buffers++);
            } else {
                copyToSim(mgr.policyAssignmentTensor(), *buffers++);
            }
            copyToSim(mgr.rewardHyperParamsTensor(), *buffers++);
        }

        gpuExec.runAsync(stepGraph, strm);

        buffers = copyOutObservations(strm, buffers, mgr);

        auto copyFromSim = [&strm](void *dst, const Tensor &src) {
            uint64_t num_bytes = numTensorBytes(src);

            REQ_CUDA(cudaMemcpyAsync(dst, src.devicePtr(), num_bytes,
                                     cudaMemcpyDeviceToDevice, strm));
        };

        if ((cfg.simFlags & SimFlags::FullTeamPolicy) ==
            SimFlags::FullTeamPolicy) {
            copyFromSim(*buffers++, mgr.fullTeamRewardTensor());
            copyFromSim(*buffers++, mgr.fullTeamDoneTensor());
        } else {
            copyFromSim(*buffers++, mgr.rewardTensor());
            copyFromSim(*buffers++, mgr.doneTensor());
        }
        copyFromSim(*buffers++, mgr.matchResultTensor());

        if (recordLogBuffer) {
            REQ_CUDA(cudaMemcpyAsync(recordLogStaging, recordLogBuffer,
                                     sizeof(StepLog) * cfg.numWorlds,
                                     cudaMemcpyDeviceToHost, strm));

            REQ_CUDA(cudaStreamSynchronize(strm));

            recordLog->write((char *)recordLogStaging,
                             sizeof(StepLog) * cfg.numWorlds);
        }

        saveEvents(strm);

        stepIdx++;
    }

    virtual Tensor rewardHyperParamsTensor() const final
    {
        return Tensor(rewardHyperParams, TensorElementType::Float32,
                      {
                          cfg.numPBTPolicies,
                          sizeof(RewardHyperParams) / sizeof(float),
                      }, cfg.gpuID);
    }

    virtual Tensor simControlTensor() const final
    {
        return Tensor(trainCtrl, TensorElementType::Int32,
                      {
                          sizeof(TrainControl) / sizeof(int32_t),
                      }, cfg.gpuID);
    }

    virtual inline Tensor exportTensor(ExportID slot,
                                       TensorElementType type,
                                       madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#endif

static MapData loadMapData(const char *map_data_name, Vector3 translation,
                           float rot_around_z, AABB *world_bounds)
{
    MapCollisionAssets imported = importCollisionData(map_data_name,
        translation, rot_around_z, world_bounds);

    MeshBVH bvh;
    void *mesh_bvh_buffer = buildMeshBVH(imported, &bvh);

    HeapArray<void *> collision_buffers(1);
    HeapArray<MeshBVH> collision_bvhs(1);
    collision_buffers[0] = mesh_bvh_buffer;
    collision_bvhs[0] = bvh;

    return MapData {
        .buffers = std::move(collision_buffers),
        .collisionBVHs = std::move(collision_bvhs),
    };
}

static SpawnCurriculum::Tier buildSpawnCurriculumTier(
    const float desired_separation,
    const float desired_spawn_radius,
    Navmesh &navmesh)
{
    const float desired_spawn_radius2 = math::sqr(desired_spawn_radius);

    const float separation_threshold = desired_separation / 4.f;

    const float base_separation =
        desired_separation + 2.f * desired_spawn_radius;
    const float separation_lower_bound =
        base_separation - separation_threshold;
    const float separation_upper_bound =
        base_separation + separation_threshold;

    StackAlloc tmp_alloc;

    Navmesh::BFSState bfs_state {
        .queue = tmp_alloc.allocN<uint32_t>(navmesh.numTris),
        .visited = tmp_alloc.allocN<bool>(navmesh.numTris),
    };

    Navmesh::DijkstrasState dijkstras_state {
        .distances = tmp_alloc.allocN<float>(navmesh.numTris),
        .entryPoints = tmp_alloc.allocN<Vector3>(navmesh.numTris),
        .heap = tmp_alloc.allocN<uint32_t>(navmesh.numTris),
        .heapIndex = tmp_alloc.allocN<uint32_t>(navmesh.numTris),
    };

    uint32_t *a_navmesh_tris = tmp_alloc.allocN<uint32_t>(navmesh.numTris);
    uint32_t *b_navmesh_tris = tmp_alloc.allocN<uint32_t>(navmesh.numTris);

    struct PrioQueueData {
        float dist;
        float totalCost;
        Vector3 entryPoint;
    };

    DynArray<uint32_t> spawn_poly_data(0);
    DynArray<NavmeshSpawn> navmesh_spawns(0);

    auto triWithinRadius = [&](Vector3 a, Vector3 b, Vector3 c, Vector3 from) {
        return a.distance2(from) <= desired_spawn_radius2 &&
            b.distance2(from) <= desired_spawn_radius2 &&
            c.distance2(from) <= desired_spawn_radius2;
    };

    for (uint32_t a_start_tri = 0; a_start_tri < navmesh.numTris;
         a_start_tri++) {
        Vector3 a_start_center;
        {
            Vector3 a, b, c;
            navmesh.getTriangleVertices(a_start_tri, &a, &b, &c);

            a_start_center = (a + b + c) / 3.f;
        }

        float a_area = 0.f;
        CountT num_a_tris = 0;

        // Grow A spawn area
        navmesh.bfsFromPoly(
            a_start_tri, bfs_state,
            [&](uint32_t tri_idx)
            {
                Vector3 a, b, c;
                navmesh.getTriangleVertices(tri_idx, &a, &b, &c);

                if (tri_idx != a_start_tri &&
                    !triWithinRadius(a, b, c, a_start_center)) {
                    return false;
                }

                Vector3 ab = b - a;
                Vector3 ac = c - a;
                float tri_area = cross(ab, ac).length() / 2.f;
                a_area += tri_area;

                a_navmesh_tris[num_a_tris++] = tri_idx;

                return true;
            });

        if (a_area < 4.f * math::pi * math::sqr(consts::agentRadius)) {
            continue;
        }

        uint32_t a_offset = 0xFFFF'FFFF_u32;

        navmesh.dijkstrasFromPoly(
            a_start_tri, a_start_center, dijkstras_state,
            [&](uint32_t b_start_tri, Vector3 edge_pos, float dist_to_edge)
            {
                Vector3 b_start_center;
                {
                    Vector3 a, b, c;
                    navmesh.getTriangleVertices(b_start_tri, &a, &b, &c);

                    b_start_center = (a + b + c) / 3.f;
                }

                float dist_to_center = edge_pos.distance(b_start_center);
                float total_distance = dist_to_edge + dist_to_center;
#if 0
                printf("%f %f %f %u %u\n",
                       total_distance,
                       a_start_center.distance(b_start_center),
                       dist_to_center,
                       a_start_tri,
                       b_start_tri);
#endif

                if (total_distance < separation_lower_bound ||
                    total_distance > separation_upper_bound) {
                    return;
                }

                CountT num_b_tris = 0;
                float b_area = 0.f;

                // Grow B spawn area
                navmesh.bfsFromPoly(
                    b_start_tri, bfs_state,
                    [&](uint32_t tri_idx)
                    {
                        Vector3 a, b, c;
                        navmesh.getTriangleVertices(tri_idx, &a, &b, &c);

                        if (tri_idx != b_start_tri &&
                            !triWithinRadius(a, b, c, b_start_center)) {
                            return false;
                        }

                        Vector3 ab = b - a;
                        Vector3 ac = c - a;
                        float tri_area = cross(ab, ac).length() / 2.f;
                        b_area += tri_area;

                        b_navmesh_tris[num_b_tris++] = tri_idx;

                        return true;
                    });

                if (b_area < 4.f * math::pi * math::sqr(consts::agentRadius)) {
                    return;
                }

                if (a_offset == 0xFFFF'FFFF_u32) {
                    a_offset = spawn_poly_data.size();
                    for (CountT i = 0; i < num_a_tris; i++) {
                        uint32_t a_tri = a_navmesh_tris[i];
                        spawn_poly_data.push_back(a_tri);
                    }
                }

                uint32_t b_offset = spawn_poly_data.size();
                for (CountT i = 0; i < num_b_tris; i++) {
                    uint32_t b_tri = b_navmesh_tris[i];
                    spawn_poly_data.push_back(b_tri);
                }

                float a_to_b_yaw;
                {
                    Vector3 a_to_b = b_start_center - a_start_center;
                    a_to_b = a_to_b.normalize();

                    a_to_b_yaw = -atan2f(a_to_b.x, a_to_b.y);
                }

                navmesh_spawns.push_back({
                    .aOffset = a_offset,
                    .bOffset = b_offset,
                    .numANavmeshPolys = (uint32_t)num_a_tris,
                    .numBNavmeshPolys = (uint32_t)num_b_tris,
                    .aBaseYaw = a_to_b_yaw,
                    .bBaseYaw = a_to_b_yaw - math::pi,
                });
            });
    }

    uint32_t *out_spawn_poly_data = (uint32_t *)malloc(
        sizeof(uint32_t) * spawn_poly_data.size());
    utils::copyN<uint32_t>(out_spawn_poly_data, spawn_poly_data.data(),
                           spawn_poly_data.size());

    NavmeshSpawn *out_spawns = (NavmeshSpawn *)malloc(
        sizeof(NavmeshSpawn) * navmesh_spawns.size());
    utils::copyN<NavmeshSpawn>(out_spawns, navmesh_spawns.data(),
                               navmesh_spawns.size());

    return SpawnCurriculum::Tier {
        .spawnPolyData = out_spawn_poly_data,
        .spawns = out_spawns,
        .numTotalSpawnPolys = (uint32_t)spawn_poly_data.size(),
        .numSpawns = (uint32_t)navmesh_spawns.size(),
    };
}

static SpawnCurriculum buildSpawnCurriculum(Navmesh &navmesh)
{
    float desired_tier_distances[SpawnCurriculum::numCurriculumTiers];
    desired_tier_distances[0] = 10.f * consts::agentRadius;
    desired_tier_distances[1] = 20.f * consts::agentRadius;
    desired_tier_distances[2] = 40.f * consts::agentRadius;
    desired_tier_distances[3] = 80.f * consts::agentRadius;
    desired_tier_distances[4] = 160.f * consts::agentRadius;

    float desired_tier_spawn_radii[SpawnCurriculum::numCurriculumTiers];
    const float base_spawn_radius = 3.f * consts::agentRadius * 2.f;
    desired_tier_spawn_radii[0] = base_spawn_radius;
    desired_tier_spawn_radii[1] = base_spawn_radius * 1.5f;
    desired_tier_spawn_radii[2] = base_spawn_radius * 1.5f;
    desired_tier_spawn_radii[3] = base_spawn_radius * 1.5f;
    desired_tier_spawn_radii[4] = base_spawn_radius * 1.5f;

    SpawnCurriculum spawn_curriculum;
    for (uint32_t tier_idx = 0; tier_idx < SpawnCurriculum::numCurriculumTiers;
         tier_idx++) {
        spawn_curriculum.tiers[tier_idx] = buildSpawnCurriculumTier(
            desired_tier_distances[tier_idx],
            desired_tier_spawn_radii[tier_idx],
            navmesh);
    }

    return spawn_curriculum;
}

static DynArray<GoalRegion> hardcodedGoalRegions()
{
    DynArray<GoalRegion> goal_regions(2);

    goal_regions.push_back({ 
        .subRegions = { 
            ZOBB {
                .pMin = { 625, 510, -64},
                .pMax = { 900, 540, -56 + consts::standHeight * 1.5},
                .rotation = 0.f,
            },
        }
    });

    goal_regions.push_back({ 
        .subRegions = {
            ZOBB {
                .pMin = { 938, 440, -56},
                .pMax = { 1030, 539, -56 + consts::standHeight * 1.5},
                .rotation = 0.f,
            },
            ZOBB {
                .pMin = { 545, 102, -64},
                .pMax = { 630, 134, -56 + consts::standHeight * 1.5},
                .rotation = 0.f,
            },
        },
        .numSubRegions = 2,
    });

    return goal_regions;
}

namespace {

namespace NavUtils
{
using madrona::math::Vector3;
using madrona::Navmesh;

struct Node
{
    int idx;
    int cameFrom;
    float startDist;
    float score;

    void Clear(int index)
    {
        idx = index;
        cameFrom = -1;
        startDist = FLT_MAX;
        score = FLT_MAX;
    }
};

Vector3 CenterOfTri(const Navmesh& navmesh, int tri)
{
    Vector3 center = { 0.0f, 0.0f, 0.0f };
    for (int i = 0; i < 3; i++)
    {
        center += navmesh.vertices[navmesh.triIndices[tri * 3 + i]] / 3.0f;
    }
    return center;
}

bool VisitCell(const Navmesh& navmesh, int cell, int targetCell, std::vector<int> &path, std::vector<bool> &visited)
{
    if (visited[cell])
        return false;
    path.push_back(cell);
    visited[cell] = true;

    if (cell == targetCell)
        return true;

    for (int i = 0; i < 3; i++)
    {
        int neighbor = navmesh.triAdjacency[cell * 3 + i];
        if (neighbor < 0 )
            continue;
        if (VisitCell(navmesh, neighbor, targetCell, path, visited))
            return true;
    }
    path.pop_back();
    return false;
}

int AStarPathfindToTri(const Navmesh& navmesh, int startTri, int posTri)
{
    Vector3 start = CenterOfTri(navmesh, startTri);
    Vector3 pos = CenterOfTri(navmesh, posTri);

    static std::vector<Node> state(navmesh.numTris);
    for (int tri = 0; tri < (int)navmesh.numTris; tri++)
        state[tri].Clear(tri);
    state[startTri].startDist = 0.0f;

    auto sortHeap = [](const int &lhs, const int &rhs)
    {
        return state[lhs].score < state[rhs].score;
    };
    std::set<int, decltype(sortHeap)> heap(sortHeap);
    heap.insert(startTri);

    while (!heap.empty())
    {
        int thisTri = *heap.begin();
        if (thisTri == posTri)
        {
            int goal = thisTri;
            while (goal != startTri && goal != -1)
            {
                if (state[goal].cameFrom == startTri)
                    return goal;
                goal = state[goal].cameFrom;
            }
            return posTri;
        }

        heap.erase(heap.begin());
        Vector3 center = CenterOfTri(navmesh, thisTri);
        if (thisTri == startTri)
            center = start;
        for (int i = 0; i < 3; i++)
        {
            int neighbor = navmesh.triAdjacency[thisTri * 3 + i];
            if (neighbor == -1)
                continue;
            Vector3 neighpos = CenterOfTri(navmesh, neighbor);
            float score = state[thisTri].startDist + center.distance(neighpos);
            if (score < state[neighbor].startDist)
            {
                state[neighbor].cameFrom = thisTri;
                state[neighbor].startDist = score;
                state[neighbor].score = score + neighpos.distance(pos);
                heap.insert(neighbor);
            }
        }
    }
    return -1;
}
}

}

static AStarLookup buildAStarLookup(const Navmesh &navmesh)
{
    u32 num_tris = navmesh.numTris;

    u32 *lookup_tbl = (u32 *)malloc(sizeof(u32) * num_tris * num_tris);

    for (u32 start = 0; start < num_tris; start++) {
        for (u32 goal = 0; goal < num_tris; goal++) {
            lookup_tbl[start * num_tris + goal] =
                NavUtils::AStarPathfindToTri(navmesh, start, goal);
        }
    }

    return AStarLookup {
        .data = lookup_tbl,
    };
}

Manager::Impl * Manager::Impl::init(
    const Manager::Config &mgr_cfg,
    VizState *viz_state)
{
#if defined(MADRONA_CUDA_SUPPORT) && defined(ENABLE_MWGPU)
    CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);
#endif

    const char *collision_data_filename = mgr_cfg.map.collisionDataFile;
    const char *navmesh_filename = mgr_cfg.map.navmeshFile;
    const char *spawns_filename = mgr_cfg.map.spawnDataFile;
    const char *zones_filename = mgr_cfg.map.zoneDataFile;
    Vector3 map_offset = mgr_cfg.map.mapOffset;
    float map_rotation = mgr_cfg.map.mapRotation;

    AABB world_bounds;
    MapData map_data = loadMapData(
        collision_data_filename, map_offset, map_rotation, &world_bounds);
    MapNavmesh navmesh_data = importNavmesh(navmesh_filename, world_bounds);

    MapSpawnData map_spawn_data = loadMapSpawnData(spawns_filename);

    DynArray<Spawn> &a_spawns = map_spawn_data.aSpawns;
    DynArray<Spawn> &b_spawns = map_spawn_data.bSpawns;
    DynArray<Spawn> &common_respawns = map_spawn_data.commonRespawns;
    DynArray<RespawnRegion> &respawn_regions = map_spawn_data.respawnRegions;

    assert(a_spawns.size() > 0);
    assert(b_spawns.size() > 0);

    AABB random_spawn_region = {
        .pMin = { -280.f, -200.f, 0.5f },
        .pMax = { 280.f, 200.f, 0.5f },
    };

    uint32_t num_default_a_spawns = (uint32_t)a_spawns.size();
    uint32_t num_default_b_spawns = (uint32_t)b_spawns.size();

    if ((mgr_cfg.simFlags & SimFlags::SpawnInMiddle) == SimFlags::SpawnInMiddle) {
        Vector3 diff = random_spawn_region.pMax - random_spawn_region.pMin;

        CountT cell_dim = 20;

        float cell_width = diff.x / cell_dim;
        float cell_height = diff.y / cell_dim;

        for (CountT y = 0; y < cell_dim; y++) {
            for (CountT x = 0; x < cell_dim; x++) {
                Vector3 cell_min = random_spawn_region.pMin + Vector3 {
                    cell_width * x,
                    cell_height * y,
                    0.5f,
                };

                Vector3 cell_max = cell_min + Vector3 {
                    cell_width,
                    cell_height,
                    0.5f,
                };

                AABB cell_aabb = {
                    .pMin = cell_min,
                    .pMax = cell_max,
                };

                bool overlaps_geo = false;
                map_data.collisionBVHs[0].findOverlaps(cell_aabb, [&overlaps_geo](
                        Vector3, Vector3, Vector3) {
                    overlaps_geo = true;
                });

                if (!overlaps_geo) {
                    Spawn spawn {
                        .region = cell_aabb,
                        .yawMin = 0.f,
                        .yawMax = 2.f * math::pi,
                    };

                    if (x >= cell_dim / 2) {
                        b_spawns.push_back(spawn);
                    } else {
                        a_spawns.push_back(spawn);
                    }
                }
            }
        }
    }

    HeapArray<uint32_t> navmesh_remap(navmesh_data.verts.size());

    CountT num_deduplicated_navmesh_verts = (CountT)meshopt_generateVertexRemap(
        navmesh_remap.data(),
        nullptr, navmesh_data.verts.size(),
        navmesh_data.verts.data(), navmesh_data.verts.size(),
        sizeof(Vector3));

    HeapArray<Vector3> navmesh_dedup_verts(num_deduplicated_navmesh_verts);
    meshopt_remapVertexBuffer(navmesh_dedup_verts.data(),
                              navmesh_data.verts.data(),
                              navmesh_data.verts.size(), sizeof(Vector3),
                              navmesh_remap.data());

    HeapArray<uint32_t> navmesh_dedup_indices(navmesh_data.indices.size());
    for (CountT i = 0; i < navmesh_dedup_indices.size(); i++) {
        navmesh_dedup_indices[i] = navmesh_remap[navmesh_data.indices[i]];
    }

    MeshBVH *static_meshes;
    Navmesh navmesh = Navmesh::initFromPolygons(
        navmesh_dedup_verts.data(),
        navmesh_dedup_indices.data(),
        navmesh_data.faceStarts.data(),
        navmesh_data.faceCounts.data(),
        num_deduplicated_navmesh_verts,
        navmesh_data.faceCounts.size());

    AStarLookup astar_lookup = buildAStarLookup(navmesh);

    StandardSpawns standard_spawns;

    SpawnCurriculum spawn_curriculum = buildSpawnCurriculum(navmesh);

    ZoneData zone_data = loadMapZones(zones_filename);

    Zones zones;

    DynArray<GoalRegion> goal_regions = hardcodedGoalRegions();
    GoalRegion *goal_regions_ptr = nullptr;
    int num_goal_regions = (int)goal_regions.size();

    TDMEpisode *episodes = nullptr;
    uint32_t num_episodes = 0;
    //episodes = loadEpisodeData("", &num_episodes);

    WeaponStats *weapon_type_stats;
    i32 num_weapon_types;

#if 0
    {
        num_weapon_types = 3;

        weapon_type_stats = (WeaponStats *)malloc(
            sizeof(WeaponStats) * num_weapon_types);

        weapon_type_stats[0] = {
            .magSize = 30,
            .fireQueueSize = 4,
            .reloadTime = 30,
            .dmgPerBullet = 10.f,
            .accuracyScale = 0.15f,
        };

        weapon_type_stats[1] = {
            .magSize = 25,
            .fireQueueSize = 2,
            .reloadTime = 10,
            .dmgPerBullet = 5.f,
            .accuracyScale = 0.2f,
        };

        weapon_type_stats[2] = {
            .magSize = 1,
            .fireQueueSize = 10,
            .reloadTime = 90,
            .dmgPerBullet = 100.f,
            .accuracyScale = 0.01f,
        };
    }
#endif

    {
        num_weapon_types = 1;

        weapon_type_stats = (WeaponStats *)malloc(
            sizeof(WeaponStats) * num_weapon_types);

        weapon_type_stats[0] = {
            .magSize = 30,
            .reloadTime = 30,
            .dmgPerBullet = 10.f,
            .accuracyScale = 0.005f,
        };
    }

    TrainControl *train_ctrl;
    {
        train_ctrl = (TrainControl *)malloc(
            sizeof(TrainControl));

        *train_ctrl = TrainControl {
            .evalMode =
                (mgr_cfg.simFlags & SimFlags::SimEvalMode) ==
                SimFlags::SimEvalMode ? 1 : 0,
            .randomizeEpisodeLengthAfterReset =
                (mgr_cfg.simFlags & SimFlags::StaggerStarts) ==
                SimFlags::StaggerStarts ? 1 : 0,
            .randomizeTeamSides =
                (mgr_cfg.simFlags & SimFlags::RandomFlipTeams) ==
                SimFlags::RandomFlipTeams ? 1 : 0,
        };
    }

    RewardHyperParams *reward_hyper_params;

    StepLog *record_log = nullptr;
    StepLog *replay_log = nullptr;
    EventLogGlobalState *event_log_global_state = nullptr;

    TrajectoryCurriculum trajectory_curriculum {
      .numSnapshots = 0,
      .snapshots = nullptr,
    };

    if (mgr_cfg.curriculumDataPath != nullptr) {
      std::ifstream curriculum_data_file(mgr_cfg.curriculumDataPath,
                                         std::ios::binary);
      {
        curriculum_data_file.seekg(0, curriculum_data_file.end);
        i64 size = curriculum_data_file.tellg();
        curriculum_data_file.seekg(0, curriculum_data_file.beg);

        trajectory_curriculum.numSnapshots =
            size / sizeof(CurriculumSnapshot);
      }

      trajectory_curriculum.snapshots = (CurriculumSnapshot *)malloc(
          sizeof(CurriculumSnapshot) * trajectory_curriculum.numSnapshots);

      curriculum_data_file.read((char *)trajectory_curriculum.snapshots,
          sizeof(CurriculumSnapshot) * trajectory_curriculum.numSnapshots);
    }

    switch (mgr_cfg.execMode) {
        case ExecMode::CUDA: {
#if defined(MADRONA_CUDA_SUPPORT) && defined(ENABLE_MWGPU)
            HeapArray<MeshBVH> gpu_staging_bvhs(map_data.collisionBVHs.size());

            for (CountT i = 0; i < map_data.collisionBVHs.size(); i++) {
                const MeshBVH &src_bvh = map_data.collisionBVHs[i];
                MeshBVH &staging_bvh = gpu_staging_bvhs[i];

                MeshBVH::BVHVertex *gpu_vertices = (MeshBVH::BVHVertex *)cu::allocGPU(
                    sizeof(MeshBVH::BVHVertex) * src_bvh.numVerts);
                MeshBVH::Node *gpu_nodes = (MeshBVH::Node *)cu::allocGPU(
                    sizeof(MeshBVH::Node) * src_bvh.numNodes);
                MeshBVH::LeafMaterial *gpu_leaf_mats = 
                    (MeshBVH::LeafMaterial *)cu::allocGPU(
                        sizeof(MeshBVH::LeafMaterial) * src_bvh.numLeaves);

                cudaMemcpy(gpu_vertices, src_bvh.vertices,
                           sizeof(MeshBVH::BVHVertex) * src_bvh.numVerts,
                           cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_nodes, src_bvh.nodes,
                           sizeof(MeshBVH::Node) * src_bvh.numNodes,
                           cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_leaf_mats, src_bvh.leafMats,
                           sizeof(MeshBVH::LeafMaterial) * src_bvh.numLeaves,
                           cudaMemcpyHostToDevice);

                staging_bvh.vertices = gpu_vertices;
                staging_bvh.nodes = gpu_nodes;
                staging_bvh.leafMats = gpu_leaf_mats;
                staging_bvh.numNodes = src_bvh.numNodes;
                staging_bvh.numLeaves = src_bvh.numLeaves;
                staging_bvh.numVerts = src_bvh.numVerts;
                staging_bvh.magic = src_bvh.magic;
            }

            static_meshes = (MeshBVH *)cu::allocGPU(
                sizeof(MeshBVH) * map_data.collisionBVHs.size());
            cudaMemcpy(static_meshes, gpu_staging_bvhs.data(),
                       sizeof(MeshBVH) * map_data.collisionBVHs.size(),
                       cudaMemcpyHostToDevice);

            Navmesh gpu_navmesh;

            gpu_navmesh.vertices = (math::Vector3 *)cu::allocGPU(
                sizeof(math::Vector3) * navmesh.numVerts);
            cudaMemcpy(gpu_navmesh.vertices, navmesh.vertices,
                       sizeof(math::Vector3) * navmesh.numVerts,
                       cudaMemcpyHostToDevice);

            gpu_navmesh.triIndices = (uint32_t *)cu::allocGPU(
                sizeof(uint32_t) * navmesh.numTris * 3);
            cudaMemcpy(gpu_navmesh.triIndices, navmesh.triIndices,
                       sizeof(uint32_t) * navmesh.numTris * 3,
                       cudaMemcpyHostToDevice);

            gpu_navmesh.triAdjacency = nullptr;

            gpu_navmesh.triSampleAliasTable = (Navmesh::AliasEntry *)cu::allocGPU(
                sizeof(Navmesh::AliasEntry) * navmesh.numTris);
            cudaMemcpy(gpu_navmesh.triSampleAliasTable, navmesh.triSampleAliasTable,
                       sizeof(Navmesh::AliasEntry) * navmesh.numTris,
                       cudaMemcpyHostToDevice);

            gpu_navmesh.numVerts = navmesh.numVerts;
            gpu_navmesh.numTris = navmesh.numTris;

            free(navmesh.vertices);
            free(navmesh.triIndices);
            free(navmesh.triAdjacency);
            free(navmesh.triSampleAliasTable);

            navmesh = gpu_navmesh;

            AStarLookup gpu_astar_lookup;
            gpu_astar_lookup.data = (u32 *)cu::allocGPU(
                sizeof(u32) * navmesh.numTris * navmesh.numTris);
            cudaMemcpy(gpu_astar_lookup.data,
                       astar_lookup.data,
                       sizeof(u32) * navmesh.numTris * navmesh.numTris,
                       cudaMemcpyHostToDevice);

            free(astar_lookup.data);
            astar_lookup = gpu_astar_lookup;

            standard_spawns.aSpawns = (Spawn *)cu::allocGPU(
                sizeof(Spawn) * a_spawns.size());
            cudaMemcpy(standard_spawns.aSpawns, a_spawns.data(),
                       sizeof(Spawn) * a_spawns.size(),
                       cudaMemcpyHostToDevice);

            standard_spawns.bSpawns = (Spawn *)cu::allocGPU(
                sizeof(Spawn) * b_spawns.size());
            cudaMemcpy(standard_spawns.bSpawns, b_spawns.data(),
                       sizeof(Spawn) * b_spawns.size(),
                       cudaMemcpyHostToDevice);

            standard_spawns.commonRespawns = (Spawn *)cu::allocGPU(
                sizeof(Spawn) * common_respawns.size());
            cudaMemcpy(standard_spawns.commonRespawns, common_respawns.data(),
                       sizeof(Spawn) * common_respawns.size(),
                       cudaMemcpyHostToDevice);

            standard_spawns.numDefaultASpawns = num_default_a_spawns;
            standard_spawns.numDefaultBSpawns = num_default_b_spawns;
            standard_spawns.numExtraASpawns = (uint32_t)a_spawns.size() - num_default_a_spawns;
            standard_spawns.numExtraBSpawns = (uint32_t)b_spawns.size() - num_default_b_spawns;
            standard_spawns.numCommonRespawns = (uint32_t)common_respawns.size();

            standard_spawns.respawnRegions = (RespawnRegion *)cu::allocGPU(
                sizeof(RespawnRegion) * respawn_regions.size());
            cudaMemcpy(standard_spawns.respawnRegions, respawn_regions.data(),
                       sizeof(RespawnRegion) * respawn_regions.size(),
                       cudaMemcpyHostToDevice);
            standard_spawns.numRespawnRegions = (uint32_t)respawn_regions.size();

            for (uint32_t i = 0; i < SpawnCurriculum::numCurriculumTiers; i++) {
                auto &tier = spawn_curriculum.tiers[i];

                uint32_t *gpu_spawn_poly_data = (uint32_t *)cu::allocGPU(
                    sizeof(uint32_t) * tier.numTotalSpawnPolys);
                NavmeshSpawn *gpu_spawns = (NavmeshSpawn *)cu::allocGPU(
                    sizeof(NavmeshSpawn) * tier.numSpawns);

                cudaMemcpy(gpu_spawn_poly_data, tier.spawnPolyData,
                           sizeof(uint32_t) * tier.numTotalSpawnPolys,
                           cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_spawns, tier.spawns,
                           sizeof(NavmeshSpawn) * tier.numSpawns,
                           cudaMemcpyHostToDevice);

                tier.spawnPolyData = gpu_spawn_poly_data;
                tier.spawns = gpu_spawns;
            }

            {
                uint32_t num_zones = zone_data.aabbs.size();
                zones.bboxes = (AABB *)cu::allocGPU(sizeof(AABB) * num_zones);
                zones.rotations = (float *)cu::allocGPU(sizeof(float) * num_zones);
                zones.numZones = num_zones;

                cudaMemcpy(zones.bboxes,
                           zone_data.aabbs.data(),
                           sizeof(AABB) * num_zones,
                           cudaMemcpyHostToDevice);

                cudaMemcpy(zones.rotations,
                           zone_data.rotations.data(),
                           sizeof(float) * num_zones,
                           cudaMemcpyHostToDevice);
            }

            {
                goal_regions_ptr = (GoalRegion *)cu::allocGPU(
                    sizeof(GoalRegion) * num_goal_regions);

                cudaMemcpy(goal_regions_ptr, goal_regions.data(),
                           sizeof(GoalRegion) * num_goal_regions,
                           cudaMemcpyHostToDevice);
            }

            if (mgr_cfg.numPBTPolicies > 0) {
                reward_hyper_params = (RewardHyperParams *)cu::allocGPU(
                    sizeof(RewardHyperParams) * mgr_cfg.numPBTPolicies);
            } else {
                reward_hyper_params = (RewardHyperParams *)cu::allocGPU(
                    sizeof(RewardHyperParams));

                RewardHyperParams default_reward_hyper_params;

                REQ_CUDA(cudaMemcpy(reward_hyper_params,
                                    &default_reward_hyper_params, sizeof(RewardHyperParams),
                                    cudaMemcpyHostToDevice));
            }

            if (mgr_cfg.recordLogPath != nullptr) {
                record_log = (StepLog *)cu::allocGPU(
                    sizeof(StepLog) * mgr_cfg.numWorlds);
            }

            if (mgr_cfg.replayLogPath != nullptr) {
                replay_log = (StepLog *)cu::allocGPU(
                    sizeof(StepLog) * mgr_cfg.numWorlds);
            }

            if (mgr_cfg.eventLogPath != nullptr) {
                event_log_global_state = (EventLogGlobalState *)cu::allocGPU(
                    sizeof(EventLogGlobalState));
                cudaMemset(event_log_global_state, 0, sizeof(EventLogGlobalState));
            }

            TDMEpisode *gpu_episodes = (TDMEpisode *)cu::allocGPU(
                sizeof(TDMEpisode) * num_episodes);
            cudaMemcpy(gpu_episodes, episodes, sizeof(TDMEpisode) * num_episodes,
                       cudaMemcpyHostToDevice);

            free(episodes);
            episodes = gpu_episodes;

            WeaponStats *gpu_weapon_type_stats = (WeaponStats *)cu::allocGPU(
                sizeof(WeaponStats) * num_weapon_types);
            cudaMemcpy(gpu_weapon_type_stats, weapon_type_stats, 
                       sizeof(WeaponStats) * num_weapon_types, cudaMemcpyHostToDevice);

            weapon_type_stats = gpu_weapon_type_stats;

            TrainControl *gpu_train_ctrl = (TrainControl *)cu::allocGPU(
                sizeof(TrainControl));
            cudaMemcpy(gpu_train_ctrl, train_ctrl,
                       sizeof(TrainControl), cudaMemcpyHostToDevice);

            train_ctrl = gpu_train_ctrl;

            if (trajectory_curriculum.numSnapshots > 0) {
                CurriculumSnapshot *gpu_snapshots = (CurriculumSnapshot *)cu::allocGPU(
                  sizeof(CurriculumSnapshot) * trajectory_curriculum.numSnapshots);
                cudaMemcpy(gpu_snapshots, trajectory_curriculum.snapshots,
                  sizeof(CurriculumSnapshot) * trajectory_curriculum.numSnapshots,
                  cudaMemcpyHostToDevice);

                free(trajectory_curriculum.snapshots);
                trajectory_curriculum.snapshots = gpu_snapshots;
            }
#else
            FATAL("No CUDA");
#endif
        } break;
        case ExecMode::CPU: {
            static_meshes = (MeshBVH *)malloc(
                sizeof(MeshBVH) * map_data.collisionBVHs.size());
            memcpy(static_meshes,
                   map_data.collisionBVHs.data(),
                   sizeof(MeshBVH) * map_data.collisionBVHs.size());

            standard_spawns.aSpawns = (Spawn *)malloc(
                sizeof(Spawn) * a_spawns.size());
            memcpy(standard_spawns.aSpawns, a_spawns.data(),
                   sizeof(Spawn) * a_spawns.size());

            standard_spawns.bSpawns = (Spawn *)malloc(
                sizeof(Spawn) * b_spawns.size());
            memcpy(standard_spawns.bSpawns, b_spawns.data(),
                   sizeof(Spawn) * b_spawns.size());

            standard_spawns.commonRespawns = (Spawn *)malloc(
                sizeof(Spawn) * common_respawns.size());
            memcpy(standard_spawns.commonRespawns, common_respawns.data(),
                   sizeof(Spawn) * common_respawns.size());

            standard_spawns.numDefaultASpawns = num_default_a_spawns;
            standard_spawns.numDefaultBSpawns = num_default_b_spawns;
            standard_spawns.numExtraASpawns = (uint32_t)a_spawns.size() - num_default_a_spawns;
            standard_spawns.numExtraBSpawns = (uint32_t)b_spawns.size() - num_default_b_spawns;
            standard_spawns.numCommonRespawns = (uint32_t)common_respawns.size();

            standard_spawns.respawnRegions = (RespawnRegion *)malloc(
                sizeof(RespawnRegion) * respawn_regions.size());
            memcpy(standard_spawns.respawnRegions, respawn_regions.data(),
                   sizeof(RespawnRegion) * respawn_regions.size());
            standard_spawns.numRespawnRegions = (uint32_t)respawn_regions.size();

            {
                uint32_t num_zones = zone_data.aabbs.size();
                zones.bboxes = (AABB *)malloc(sizeof(AABB) * num_zones);
                zones.rotations = (float *)malloc(sizeof(float) * num_zones);
                zones.numZones = num_zones;

                memcpy(zones.bboxes,
                       zone_data.aabbs.data(),
                       sizeof(AABB) * num_zones);

                memcpy(zones.rotations,
                       zone_data.rotations.data(),
                       sizeof(float) * num_zones);
            }

            {
                goal_regions_ptr = (GoalRegion *)malloc(
                    sizeof(GoalRegion) * num_goal_regions);

                memcpy(goal_regions_ptr, goal_regions.data(),
                       sizeof(GoalRegion) * num_goal_regions);
            }

            if (mgr_cfg.numPBTPolicies > 0) {
                reward_hyper_params = (RewardHyperParams *)malloc(
                    sizeof(RewardHyperParams) * mgr_cfg.numPBTPolicies);
            } else {
                reward_hyper_params = (RewardHyperParams *)malloc(
                    sizeof(RewardHyperParams));

                *(reward_hyper_params) = RewardHyperParams {};
            }

            if (mgr_cfg.recordLogPath != nullptr) {
                record_log = (StepLog *)malloc(
                    sizeof(StepLog) * mgr_cfg.numWorlds);
            }

            if (mgr_cfg.replayLogPath != nullptr) {
                replay_log = (StepLog *)malloc(
                    sizeof(StepLog) * mgr_cfg.numWorlds);
            }

            if (mgr_cfg.eventLogPath != nullptr) {
                event_log_global_state = (EventLogGlobalState *)malloc(
                    sizeof(EventLogGlobalState));
                event_log_global_state->numEvents = 0;
                event_log_global_state->numStepStates = 0;
            }
        } break;
        default: {
            assert(false);
            MADRONA_UNREACHABLE();
        } break;
    }

    RandKey init_key = rand::initKey(mgr_cfg.randSeed);
    RandKey sim_init_key = rand::split_i(init_key, 0);
    RandKey stagger_shuffle_key = rand::split_i(init_key, 1);

    TaskConfig task_cfg {
        .autoReset = mgr_cfg.autoReset,
        .showSpawns = true,
        .simFlags = mgr_cfg.simFlags,
        .initRandKey = sim_init_key,
        .numPBTPolicies = mgr_cfg.numPBTPolicies,
        .policyHistorySize = mgr_cfg.policyHistorySize,
        .worldBounds = world_bounds,
        .pTeamSize = mgr_cfg.teamSize,
        .eTeamSize = mgr_cfg.taskType == Task::Turret ? 1 : mgr_cfg.teamSize,
        .staticMeshes = static_meshes,
        .numStaticMeshes = (uint32_t)map_data.collisionBVHs.size(),
        .navmesh = navmesh,
        .aStarLookup = astar_lookup,
        .standardSpawns = standard_spawns,
        .spawnCurriculum = spawn_curriculum,
        .zones = zones,
        .task = mgr_cfg.taskType,
        .highlevelMove = mgr_cfg.highlevelMove,
        .viz = viz_state,
        .rewardHyperParams = reward_hyper_params,
        .recordLog = record_log,
        .replayLog = replay_log,
        .eventGlobalState = event_log_global_state,
        .goalRegions = goal_regions_ptr,
        .numGoalRegions = (int32_t)num_goal_regions,
        .numEpisodes = num_episodes,
        .episodes = episodes,
        .numWeaponTypes = num_weapon_types,
        .weaponTypeStats = weapon_type_stats,
        .trainControl = train_ctrl,
        .trajectoryCurriculum = trajectory_curriculum,
    };

    switch (mgr_cfg.execMode) {
        case ExecMode::CUDA: {
#if defined(MADRONA_CUDA_SUPPORT) && defined(ENABLE_MWGPU)
            HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

#ifdef MADRONA_LINUX
            {
                char *env = getenv("MADRONA_MP_ENV_DEBUG_WAIT");

                if (env && env[0] == '1') {
                    volatile int done = 0;
                    while (!done) { sleep(1); }
                }
            }
#endif

            MWCudaExecutor gpu_exec({
                .worldInitPtr = world_inits.data(),
                .numWorldInitBytes = sizeof(Sim::WorldInit),
                .userConfigPtr = (void *)&task_cfg,
                .numUserConfigBytes = sizeof(TaskConfig),
                .numWorldDataBytes = sizeof(Sim),
                .worldDataAlignment = alignof(Sim),
                .numWorlds = mgr_cfg.numWorlds,
                .numTaskGraphs = (uint32_t)TaskGraphID::NumGraphs,
                .numExportedBuffers = (uint32_t)ExportID::NumExports, 
            }, {
                { GPU_HIDESEEK_SRC_LIST },
                { GPU_HIDESEEK_COMPILE_FLAGS },
                CompileConfig::OptMode::LTO,
            }, cu_ctx);

            WorldReset *world_reset_buffer = 
                (WorldReset *)gpu_exec.getExported((uint32_t)ExportID::Reset);

            ExploreAction *explore_actions_buffer = nullptr;
            void *pvp_actions_buffer = nullptr;

            if (mgr_cfg.taskType == Task::Explore) {
                explore_actions_buffer = (ExploreAction *)gpu_exec.getExported(
                    (uint32_t)ExportID::ExploreAction);
            }

            if (mgr_cfg.taskType == Task::TDM ||
                mgr_cfg.taskType == Task::Zone ||
                mgr_cfg.taskType == Task::ZoneCaptureDefend ||
                mgr_cfg.taskType == Task::Turret) {
                pvp_actions_buffer = gpu_exec.getExported(
                    (uint32_t)ExportID::PvPDiscreteAction);
            }

            return new CUDAImpl {
                mgr_cfg,
                world_reset_buffer,
                explore_actions_buffer,
                pvp_actions_buffer,
                train_ctrl,
                reward_hyper_params,
                std::move(gpu_exec),
                replay_log,
                record_log,
                event_log_global_state,
                trajectory_curriculum,
                mgr_cfg.replayLogPath,
                mgr_cfg.recordLogPath,
                mgr_cfg.eventLogPath,
                stagger_shuffle_key,
            };
#else
            FATAL("Madrona was not compiled with CUDA support");
#endif
        } break;
        case ExecMode::CPU: {
            (void)stagger_shuffle_key;

            HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

            CPUImpl::TaskGraphT cpu_exec {
                ThreadPoolExecutor::Config {
                    .numWorlds = mgr_cfg.numWorlds,
                    .numExportedBuffers = (uint32_t)ExportID::NumExports,
                },
                task_cfg,
                world_inits.data(),
                (CountT)TaskGraphID::NumGraphs,
            };

            WorldReset *world_reset_buffer = 
                (WorldReset *)cpu_exec.getExported((uint32_t)ExportID::Reset);

            ExploreAction *explore_actions_buffer = nullptr;
            void *pvp_actions_buffer = nullptr;

            if (mgr_cfg.taskType == Task::Explore) {
                explore_actions_buffer = (ExploreAction *)cpu_exec.getExported(
                    (uint32_t)ExportID::ExploreAction);
            }

            if (mgr_cfg.taskType == Task::TDM ||
                mgr_cfg.taskType == Task::Zone ||
                mgr_cfg.taskType == Task::ZoneCaptureDefend ||
                mgr_cfg.taskType == Task::Turret) {
                pvp_actions_buffer = cpu_exec.getExported(
                    (uint32_t)ExportID::PvPDiscreteAction);
            }

            auto cpu_impl = new CPUImpl {
                mgr_cfg,
                world_reset_buffer,
                explore_actions_buffer,
                pvp_actions_buffer,
                train_ctrl,
                reward_hyper_params,
                std::move(cpu_exec),
                replay_log,
                record_log,
                event_log_global_state,
                trajectory_curriculum,
                mgr_cfg.replayLogPath,
                mgr_cfg.recordLogPath,
                mgr_cfg.eventLogPath,
            };

            return cpu_impl;
        } break;
        default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(const Config &cfg,
                 VizState *viz)
    : impl_(Impl::init(cfg, viz))
{}

Manager::~Manager() {}

void Manager::init()
{
    // Currently, there is no way to populate the initial set of observations
    // without stepping the simulations in order to execute the taskgraph.
    // Therefore, after setup, we step all the simulations with a forced reset
    // that ensures the first real step will have valid observations at the
    // start of a fresh episode in order to compute actions.
    //
    // This will be improved in the future with support for multiple task
    // graphs, allowing a small task graph to be executed after initialization.

    auto &cfg = impl_->cfg;

    for (int32_t i = 0; i < (int32_t)cfg.numWorlds; i++) {
        triggerReset(i);
    }
    impl_->run(TaskGraphID::Init);

    if ((cfg.simFlags & SimFlags::StaggerStarts) ==
        SimFlags::StaggerStarts) {
        // Only supported in gpuStreamInit
        assert(false);
    }
}

void Manager::step()
{
    impl_->run(TaskGraphID::Step);
}

#ifdef MADRONA_CUDA_SUPPORT
void Manager::gpuStreamInit(cudaStream_t strm, void **buffers)
{
    impl_->gpuStreamInit(strm, buffers, *this);
}

void Manager::gpuStreamStep(cudaStream_t strm, void **buffers)
{
    impl_->gpuStreamStep(strm, buffers, *this);
}
#endif

Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(ExportID::Reset,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(WorldReset) / sizeof(int32_t),
                               });
}

Tensor Manager::simControlTensor() const
{
    return impl_->simControlTensor();
}

Tensor Manager::pvpDiscreteActionTensor() const
{
    return impl_->exportTensor(ExportID::PvPDiscreteAction,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   (int64_t)(impl_->cfg.highlevelMove ?
                                             (sizeof(CoarsePvPAction) / sizeof(uint32_t)) :
                                             (sizeof(PvPDiscreteAction) / sizeof(uint32_t))),
                               });
}

Tensor Manager::pvpAimActionTensor() const
{
    return impl_->exportTensor(ExportID::PvPAimAction,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   1,
                                   (sizeof(PvPAimAction) / sizeof(float)),
                               });
}

Tensor Manager::exploreActionTensor() const
{
    return impl_->exportTensor(ExportID::ExploreAction,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   sizeof(ExploreAction) / sizeof(uint32_t),
                               });
}

Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   1,
                               });
}

Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   1,
                               });
}

Tensor Manager::policyAssignmentTensor() const
{
    return impl_->exportTensor(ExportID::AgentPolicy,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   1,
                               });
}

Tensor Manager::selfObservationTensor() const
{
    return impl_->exportTensor(ExportID::SelfObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   sizeof(SelfObservation) / sizeof(float),
                               });
}

Tensor Manager::filtersStateObservationTensor() const
{
    return impl_->exportTensor(ExportID::FiltersStateObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   sizeof(FiltersStateObservation) / sizeof(float),
                               });
}

Tensor Manager::teammateObservationsTensor() const
{
    static_assert(sizeof(TeammateObservations) == 
                  sizeof(TeammateObservation) * (consts::maxTeamSize - 1));

    return impl_->exportTensor(ExportID::TeammateObservations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   consts::maxTeamSize - 1,
                                   sizeof(TeammateObservation) / sizeof(float),
                               });
}

Tensor Manager::opponentObservationsTensor() const
{
    return impl_->exportTensor(ExportID::OpponentObservations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   consts::maxTeamSize,
                                   sizeof(OpponentObservation) / sizeof(float),
                               });
}

Tensor Manager::opponentLastKnownObservationsTensor() const
{
    return impl_->exportTensor(ExportID::OpponentLastKnownObservations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   consts::maxTeamSize,
                                   sizeof(OpponentObservation) / sizeof(float),
                               });
}

Tensor Manager::selfPositionTensor() const
{
    return impl_->exportTensor(ExportID::SelfPositionObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   sizeof(SelfPositionObservation) / sizeof(float),
                               });
}

Tensor Manager::teammatePositionObservationsTensor() const
{
    return impl_->exportTensor(ExportID::TeammatePositionObservations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   consts::maxTeamSize - 1,
                                   sizeof(NormalizedPositionObservation) / sizeof(float),
                               });
}

Tensor Manager::opponentPositionObservationsTensor() const
{
    return impl_->exportTensor(ExportID::OpponentPositionObservations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   consts::maxTeamSize,
                                   sizeof(NormalizedPositionObservation) / sizeof(float),
                               });
}


Tensor Manager::opponentLastKnownPositionObservationsTensor() const
{
    return impl_->exportTensor(ExportID::OpponentLastKnownPositionObservations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   consts::maxTeamSize,
                                   sizeof(NormalizedPositionObservation) / sizeof(float),
                               });
}


Tensor Manager::opponentMasksTensor() const
{
    return impl_->exportTensor(ExportID::OpponentMasks,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   consts::maxTeamSize,
                                   1,
                               });
}

Tensor Manager::fwdLidarTensor() const
{
    return impl_->exportTensor(ExportID::FwdLidar,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   consts::fwdLidarHeight,
                                   consts::fwdLidarWidth,
                                   sizeof(LidarData) / sizeof(float),
                               });
}

Tensor Manager::rearLidarTensor() const
{
    return impl_->exportTensor(ExportID::RearLidar,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   consts::rearLidarHeight,
                                   consts::rearLidarWidth,
                                   sizeof(LidarData) / sizeof(float),
                               });
}

Tensor Manager::agentMapTensor() const
{
    return impl_->exportTensor(ExportID::AgentMap,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   AgentMap::res,
                                   AgentMap::res,
                                   sizeof(MapItem) / sizeof(float),
                               });
}

Tensor Manager::unmaskedAgentMapTensor() const
{
    return impl_->exportTensor(ExportID::UnmaskedAgentMap,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   AgentMap::res,
                                   AgentMap::res,
                                   sizeof(MapItem) / sizeof(float),
                               });
}

Tensor Manager::hpTensor() const
{
    return impl_->exportTensor(ExportID::HP,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   1,
                               });
}

Tensor Manager::magazineTensor() const
{
    return impl_->exportTensor(ExportID::Magazine,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   2,
                               });
}

Tensor Manager::aliveTensor() const
{
    return impl_->exportTensor(ExportID::Alive,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                                   1,
                               });
}

Tensor Manager::rewardHyperParamsTensor() const
{
    return impl_->rewardHyperParamsTensor();
}

Tensor Manager::matchResultTensor() const
{
    return impl_->exportTensor(ExportID::MatchResult,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   sizeof(MatchResult) / sizeof(int32_t),
                               });
}


Tensor Manager::fullTeamActionTensor() const
{
    return impl_->exportTensor(ExportID::FullTeamActions,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * consts::numTeams,
                                   consts::maxTeamSize,
                                   (sizeof(PvPDiscreteAction) / sizeof(uint32_t)),
                               });
}

Tensor Manager::fullTeamGlobalObservationsTensor() const
{
    return impl_->exportTensor(ExportID::FullTeamGlobal,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numTeams,
                                   sizeof(FullTeamGlobalObservation) / sizeof(float),
                               });
}

Tensor Manager::fullTeamPlayerObservationsTensor() const
{
    return impl_->exportTensor(ExportID::FullTeamPlayers,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numTeams,
                                   consts::maxTeamSize,
                                   sizeof(FullTeamPlayerObservation) / sizeof(float),
                               });
}

Tensor Manager::fullTeamEnemyObservationsTensor() const
{
    return impl_->exportTensor(ExportID::FullTeamEnemies,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numTeams,
                                   consts::maxTeamSize,
                                   sizeof(FullTeamEnemyObservation) / sizeof(float),
                               });
}

Tensor Manager::fullTeamLastKnownEnemyObservationsTensor() const
{
    return impl_->exportTensor(ExportID::FullTeamLastKnownEnemies,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numTeams,
                                   consts::maxTeamSize,
                                   sizeof(FullTeamCommonObservation) / sizeof(float),
                               });
}

Tensor Manager::fullTeamFwdLidarTensor() const
{
    return impl_->exportTensor(ExportID::FullTeamFwdLidar,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numTeams,
                                   consts::maxTeamSize,
                                   consts::fwdLidarHeight,
                                   consts::fwdLidarWidth,
                                   sizeof(LidarData) / sizeof(float),
                               });
}


Tensor Manager::fullTeamRearLidarTensor() const
{
    return impl_->exportTensor(ExportID::FullTeamRearLidar,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numTeams,
                                   consts::maxTeamSize,
                                   consts::rearLidarHeight,
                                   consts::rearLidarWidth,
                                   sizeof(LidarData) / sizeof(float),
                               });
}


Tensor Manager::fullTeamRewardTensor() const
{
    return impl_->exportTensor(ExportID::FullTeamReward, TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numTeams,
                                   1,
                               });
}

Tensor Manager::fullTeamDoneTensor() const
{
    return impl_->exportTensor(ExportID::FullTeamDone, TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * consts::numTeams,
                                   1,
                               });
}

Tensor Manager::fullTeamPolicyAssignmentTensor() const
{
    return impl_->exportTensor(ExportID::FullTeamPolicyAssignments,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * consts::numTeams,
                                   1,
                               });
}

TrainInterface Manager::trainInterface() const
{
    if ((impl_->cfg.simFlags & SimFlags::FullTeamPolicy) ==
        SimFlags::FullTeamPolicy) {
        auto pbt_inputs = std::to_array<NamedTensorInterface>({
            { "policy_assignments", fullTeamPolicyAssignmentTensor().interface() },
            { "reward_hyper_params", rewardHyperParamsTensor().interface() },
        });

        return TrainInterface {
            {
                .actions = {
                    { "discrete", fullTeamActionTensor().interface() },
                    { "aim", pvpAimActionTensor().interface() },
                },
                .resets = resetTensor().interface(),
                .simCtrl = simControlTensor().interface(),
                .pbt = impl_->cfg.numPBTPolicies > 0 ?
                    pbt_inputs : Span<const NamedTensorInterface>(nullptr, 0),
            },
            {
                .observations = {
                    { "full_team_global", fullTeamGlobalObservationsTensor().interface() },
                    { "full_team_players", fullTeamPlayerObservationsTensor().interface() },
                    { "full_team_enemies", fullTeamEnemyObservationsTensor().interface() },
                    { "full_team_last_known_enemies",
                        fullTeamLastKnownEnemyObservationsTensor().interface() },
                    { "full_team_fwd_lidar", fullTeamFwdLidarTensor().interface() },
                    { "full_team_rear_lidar", fullTeamRearLidarTensor().interface() },
                },
                .rewards = fullTeamRewardTensor().interface(),
                .dones = fullTeamDoneTensor().interface(),
                .pbt = {
                    { "episode_results", matchResultTensor().interface() },
                },
            },
        };
    } else {
        auto pbt_inputs = std::to_array<NamedTensorInterface>({
            { "policy_assignments", policyAssignmentTensor().interface() },
            { "reward_hyper_params", rewardHyperParamsTensor().interface() },
        });

        return TrainInterface {
            {
                .actions = {
                  { "discrete", pvpDiscreteActionTensor().interface() },
                  { "aim", pvpAimActionTensor().interface() },
                },
                .resets = resetTensor().interface(),
                .simCtrl = simControlTensor().interface(),
                .pbt = impl_->cfg.numPBTPolicies > 0 ?
                    pbt_inputs : Span<const NamedTensorInterface>(nullptr, 0),
            },
            {
                .observations = {
                    { "fwd_lidar", fwdLidarTensor().interface() },
                    { "rear_lidar", rearLidarTensor().interface() },
                    { "hp", hpTensor().interface() },
                    { "magazine", magazineTensor().interface() },
                    { "alive", aliveTensor().interface() },

                    { "self", selfObservationTensor().interface() },
                    { "filters_state", filtersStateObservationTensor().interface() },
                    { "teammates", teammateObservationsTensor().interface() },
                    { "opponents", opponentObservationsTensor().interface() },
                    { "opponents_last_known", opponentLastKnownObservationsTensor().interface() },

                    { "self_pos", selfPositionTensor().interface() },
                    { "teammate_positions",
                        teammatePositionObservationsTensor().interface() },
                    { "opponent_positions",
                        opponentPositionObservationsTensor().interface() },
                    { "opponent_last_known_positions", 
                        opponentLastKnownPositionObservationsTensor().interface() },

                    { "opponent_masks", opponentMasksTensor().interface() },

                    { "agent_map", agentMapTensor().interface() },
                    { "unmasked_agent_map", agentMapTensor().interface() },
                },
                .rewards = rewardTensor().interface(),
                .dones = doneTensor().interface(),
                .pbt = {
                    { "episode_results", matchResultTensor().interface() },
                },
            },
        };
    }
}

ExecMode Manager::execMode() const
{
    return impl_->cfg.execMode;
}

Engine & Manager::getWorldContext(int32_t world_idx)
{
    assert(impl_->cfg.execMode == ExecMode::CPU);

    return ((CPUImpl *)impl_.get())->cpuExec.getWorldContext(world_idx);
}

void Manager::triggerReset(int32_t world_idx)
{
    WorldReset reset {
        1,
    };

    auto *reset_ptr = impl_->worldResetBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(reset_ptr, &reset, sizeof(WorldReset),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *reset_ptr = reset;
    }
}

void Manager::setExploreAction(int32_t world_idx,
                               ExploreAction action)
{
    auto *action_ptr = (ExploreAction *)impl_->exploreActionsBuffer +
        world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(ExploreAction),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

void Manager::setPvPAction(int32_t world_idx,
                           int32_t agent_idx,
                           PvPDiscreteAction discrete,
                           PvPAimAction aim)
{
  (void)aim;

  {
    auto *action_ptr = (PvPDiscreteAction *)impl_->pvpActionsBuffer +
        world_idx * impl_->numAgentsPerWorld + agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT 
        cudaMemcpy(action_ptr, &discrete, sizeof(PvPDiscreteAction),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = discrete;
    }
  }
}

void Manager::setCoarsePvPAction(int32_t world_idx,
                                 int32_t agent_idx,
                                 CoarsePvPAction action)
{
    auto *action_ptr = (CoarsePvPAction *)impl_->pvpActionsBuffer +
        world_idx * impl_->numAgentsPerWorld + agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT 
        cudaMemcpy(action_ptr, &action, sizeof(CoarsePvPAction),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

void Manager::setHP(int32_t world_idx, int32_t agent_idx, int32_t hp)
{
    Tensor hp_tensor = hpTensor();
    HP *agent_hp_ptr = (HP *)hp_tensor.devicePtr() +
        world_idx * impl_->numAgentsPerWorld + agent_idx;

    HP hp_staging { (float)hp };

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(agent_hp_ptr, &hp_staging, sizeof(HP),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *agent_hp_ptr = hp_staging;
    }
}

bool Manager::isReplayFinished()
{
    if (impl_->cfg.execMode == ExecMode::CUDA) {
#if defined(MADRONA_CUDA_SUPPORT) and defined(ENABLE_MWGPU)
        auto *impl = static_cast<CUDAImpl *>(impl_.get());
        return impl->replayLog->eof();
#else
        assert(false);
        MADRONA_UNREACHABLE();
#endif
    } else {
        auto *impl = static_cast<CPUImpl *>(impl_.get());
        return impl->replayLog->eof();
    }
}

void Manager::setUniformAgentPolicy(AgentPolicy policy)
{
    const Tensor &policy_tensor = policyAssignmentTensor();

    u32 total_num_agents = 
        impl_->cfg.numWorlds * impl_->numAgentsPerWorld;

    AgentPolicy *policies_tmp = (AgentPolicy *)malloc(
        sizeof(AgentPolicy) * total_num_agents);
    for (u32 i = 0; i < total_num_agents; i++) {
        policies_tmp[i] = policy;
    }

    AgentPolicy *dst = (AgentPolicy *)policy_tensor.devicePtr();

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#if defined(MADRONA_CUDA_SUPPORT) and defined(ENABLE_MWGPU)
        cudaMemcpy(dst, policies_tmp,
                   sizeof(AgentPolicy) * total_num_agents,
                   cudaMemcpyHostToDevice);
#else
        assert(false);
        MADRONA_UNREACHABLE();
#endif
    } else {
        memcpy(dst, policies_tmp,
               sizeof(AgentPolicy) * total_num_agents);
    }

    free(policies_tmp);
}

}
