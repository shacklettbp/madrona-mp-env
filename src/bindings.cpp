#include "mgr.hpp"

#include <madrona/py/bindings.hpp>

#include <filesystem>

namespace nb = nanobind;

namespace madronaMPEnv {

NB_MODULE(madrona_mp_env, m) {
    madrona::py::setupMadronaSubmodule(m);

    nb::enum_<Task>(m, "Task")
        .value("Explore", Task::Explore)
        .value("TDM", Task::TDM)
        .value("Zone", Task::Zone)
        .value("Turret", Task::Turret)
        .value("ZoneCaptureDefend", Task::ZoneCaptureDefend)
    ;

    nb::enum_<SimFlags>(m, "SimFlags", nb::is_arithmetic())
        .value("Default", SimFlags::Default)
        .value("SpawnInMiddle", SimFlags::SpawnInMiddle)
        .value("RandomizeHPMagazine", SimFlags::RandomizeHPMagazine)
        .value("NavmeshSpawn", SimFlags::NavmeshSpawn)
        .value("NoRespawn", SimFlags::NoRespawn)
        .value("StaggerStarts", SimFlags::StaggerStarts)
        .value("EnableCurriculum", SimFlags::EnableCurriculum)
        .value("HardcodedSpawns", SimFlags::HardcodedSpawns)
        .value("RandomFlipTeams", SimFlags::RandomFlipTeams)
        .value("StaticFlipTeams", SimFlags::StaticFlipTeams)
        .value("FullTeamPolicy", SimFlags::FullTeamPolicy)
    ;

    nb::class_<Manager> (m, "SimManager")
        .def("__init__", [](Manager *self,
                            madrona::py::PyExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t rand_seed,
                            bool auto_reset,
                            uint32_t sim_flags,
                            Task task,
                            uint32_t team_size,
                            uint32_t num_pbt_policies,
                            uint32_t policy_history_size,
                            const char *scene_path,
                            nb::handle replay_log_path,
                            nb::handle record_log_path,
                            nb::handle event_log_path) {
            std::filesystem::path scene_path_dir(scene_path);

            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .randSeed = (uint32_t)rand_seed,
                .autoReset = auto_reset,
                .simFlags = SimFlags(sim_flags),
                .taskType = task,
                .teamSize = team_size,
                .numPBTPolicies = num_pbt_policies,
                .policyHistorySize = policy_history_size,
                .map = MapConfig {
                  .name = scene_path_dir.filename().string().c_str(),
                  .collisionDataFile =
                    (scene_path_dir / "collisions.bin").string().c_str(),
                  .navmeshFile =
                    (scene_path_dir / "navmesh.bin").string().c_str(),
                  .spawnDataFile =
                    (scene_path_dir / "spawns.bin").string().c_str(),
                  .zoneDataFile =
                    (scene_path_dir / "zones.bin").string().c_str(),
                  .mapOffset = Vector3::zero(),
                  .mapRotation = 0.f,
                },
                .highlevelMove = false,
                .replayLogPath = replay_log_path.is_none() ? nullptr :
                    nb::cast<const char*>(replay_log_path),
                .recordLogPath = record_log_path.is_none() ? nullptr :
                    nb::cast<const char*>(record_log_path),
                .eventLogPath = event_log_path.is_none() ? nullptr :
                    nb::cast<const char*>(event_log_path),
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("rand_seed"),
           nb::arg("auto_reset"),
           nb::arg("sim_flags"),
           nb::arg("task_type"),
           nb::arg("team_size"),
           nb::arg("num_pbt_policies"),
           nb::arg("policy_history_size"),
           nb::arg("scene_path"),
           nb::arg("replay_log_path") = nb::none(),
           nb::arg("record_log_path") = nb::none(),
           nb::arg("event_log_path") = nb::none())
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("explore_action_tensor", &Manager::exploreActionTensor)
        .def("pvp_action_tensor", &Manager::pvpActionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("policy_assignment_tensor", &Manager::policyAssignmentTensor)
        .def("self_observation_tensor", &Manager::selfObservationTensor)
        .def("opponent_masks_tensor", &Manager::opponentMasksTensor)
        .def("fwd_lidar_tensor", &Manager::fwdLidarTensor)
        .def("rear_lidar_tensor", &Manager::rearLidarTensor)
        .def("hp_tensor", &Manager::hpTensor)
        .def("magazine_tensor", &Manager::magazineTensor)
        .def("alive_tensor", &Manager::aliveTensor)
        .def("jax", madrona::py::JAXInterface::buildEntry<
                &Manager::trainInterface,
                &Manager::cpuJAXInit,
                &Manager::cpuJAXStep
#ifdef MADRONA_CUDA_SUPPORT
                ,
                &Manager::gpuStreamInit,
                &Manager::gpuStreamStep
#endif
             >())
    ;
}

}
