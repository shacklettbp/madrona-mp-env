#include <madrona/viz/viewer.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"
#include "map_importer.hpp"
#include "viz.hpp"

#include <filesystem>
#include <fstream>

#include <imgui.h>

#include <madrona/math.hpp>

using namespace madronaMPEnv;
using madrona::ExecMode;
using madrona::Span;
using madrona::math::Vector3;

int main(int argc, char *argv[])
{
    const Task task = Task::Zone;

    int64_t num_views;
    int64_t team_size;
    if (task == Task::Zone || task == Task::ZoneCaptureDefend) {
        team_size = 6;
        num_views = team_size * 2;
    } else if (task == Task::TDM) {
        team_size = 6;
        num_views = team_size * 2;
    } else if (task == Task::Turret) {
        team_size = 2;
        num_views = team_size;
    } else if (task == Task::Explore) {
        team_size = 1;
        num_views = 1;
    } else {
        assert(false);
        MADRONA_UNREACHABLE();
    }

    uint32_t num_worlds = 1;
    ExecMode exec_mode = ExecMode::CPU;

    auto usageErr = [argv]() {
        fprintf(stderr, "%s --scene SCENE_NAME [NUM_WORLDS] [--backend cpu|cuda] [--record path] [--replay path]\n", argv[0]);
        exit(EXIT_FAILURE);
    };

    bool num_worlds_set = false;

    char *record_log_path = nullptr;
    char *replay_log_path = nullptr;
    std::string scene_dir;
    bool doAITeam1 = false;
    bool doAITeam2 = false;

    char *analytics_db_path = nullptr;
    char *trajectories_db_path = nullptr;
    char *curriculum_data_path = nullptr;
    char *policy_weights_path = nullptr;

    for (int i = 1; i < argc; i++) {
      char *arg = argv[i];

      if (arg[0] == '-' && arg[1] == '-') {
        arg += 2;

        if (!strcmp("backend", arg)) {
          i += 1;

          if (i == argc) {
            usageErr();
          }

          char *value = argv[i];
          if (!strcmp("cpu", value)) {
            exec_mode = ExecMode::CPU;
          } else if (!strcmp("cuda", value)) {
            exec_mode = ExecMode::CUDA;
          } else {
            usageErr();
          }
        } else if (!strcmp("record", arg)) {
          if (record_log_path != nullptr) {
            usageErr();
          }

          i += 1;

          if (i == argc) {
            usageErr();
          }

          record_log_path = argv[i];
        } else if (!strcmp("replay", arg)) {
          if (replay_log_path != nullptr) {
            usageErr();
          }

          i += 1;

          if (i == argc) {
            usageErr();
          }

          replay_log_path = argv[i];
        } else if (!strcmp("scene", arg)) {
          if (!scene_dir.empty()) {
            usageErr();
          }

          i += 1;

          if (i == argc) {
            usageErr();
          }

          scene_dir = argv[i];
        } else if (!strcmp("doaiteam1", arg)) {
          doAITeam1 = true;
        } else if (!strcmp("doaiteam2", arg)) {
          doAITeam2 = true;
        } else if (!strcmp("analytics-db", arg)) {
          if (analytics_db_path != nullptr) {
            usageErr();
          }

          i += 1;

          if (i == argc) {
            usageErr();
          }

          analytics_db_path = argv[i];
        } else if (!strcmp("trajectories-db", arg)) {
          if (trajectories_db_path != nullptr) {
            usageErr();
          }
          
          i += 1;

          if (i == argc) {
            usageErr();
          }

          trajectories_db_path = argv[i];
        } else if (!strcmp("curriculum-data", arg)) {
          if (curriculum_data_path != nullptr) {
            usageErr();
          }
          
          i += 1;

          if (i == argc) {
            usageErr();
          }

          curriculum_data_path = argv[i];
        } else if (!strcmp("dnn-weights", arg)) {
          if (policy_weights_path != nullptr) {
            usageErr();
          }

          i += 1;
          if (i == argc) {
            usageErr();
          }

          policy_weights_path = argv[i];
        }
      } else {
        if (num_worlds_set) {
          usageErr();
        }

        num_worlds_set = true;

        num_worlds = (uint32_t)atoi(arg);
      }
    }

    if (scene_dir.empty()) {
        usageErr();
    }

    std::string collision_data_file = scene_dir + "/collisions.bin";
    std::string navmesh_file = scene_dir + "/navmesh.bin";
    std::string spawn_data_file = scene_dir + "/spawns.bin";
    std::string zone_data_file = scene_dir + "/zones.bin";

    Vector3 map_offset = Vector3::zero();
    float map_rotation = 0.f;

    VizState *viz = VizSystem::init(VizConfig {
      .windowWidth = 2730,
      .windowHeight = 1536,
      //.windowWidth = 3840,
      //.windowHeight = 2160,

      .numWorlds = (uint32_t)num_worlds,
      .numViews = (uint32_t)num_views,
      .teamSize = (uint32_t)team_size,

      .mapDataFilename = collision_data_file.c_str(),
      .mapOffset = map_offset,
      .mapRotation = map_rotation,
      .doAITeam1 = doAITeam1,
      .doAITeam2 = doAITeam2,
      .analyticsDBPath = analytics_db_path,
      .trajectoriesDBPath = trajectories_db_path,
    });

    const bool highlevel_move = false;

    SimFlags sim_flags = SimFlags::Default;
    if (task == Task::ZoneCaptureDefend) {
      sim_flags = SimFlags::HardcodedSpawns;
    }

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .randSeed = 10,
        .autoReset = replay_log_path != nullptr,
        //.simFlags = SimFlags::HardcodedSpawns /*| SimFlags::StaticFlipTeams*/,
        .simFlags = sim_flags,
        .taskType = task,
        .teamSize = (uint32_t)team_size,
        .numPBTPolicies = 0,
        .policyHistorySize = 1,
        .map = {
          .name = scene_dir.c_str(),
          .collisionDataFile = collision_data_file.c_str(),
          .navmeshFile = navmesh_file.c_str(),
          .spawnDataFile = spawn_data_file.c_str(),
          .zoneDataFile = zone_data_file.c_str(),
          .mapOffset = map_offset,
          .mapRotation = map_rotation,
        },
        .highlevelMove = highlevel_move,
        .replayLogPath = replay_log_path,
        .recordLogPath = record_log_path,
        .curriculumDataPath = curriculum_data_path,
        .policyWeightsPath = policy_weights_path,
    }, viz);
    mgr.init();

#if 0
    auto printObs = [&mgr,
                     &global_pos_tensor,
                     &self_obs_tensor,
                     &hp_tensor,
                     &magazine_tensor,
                     &fwd_lidar_tensor,
                     &rear_lidar_tensor,
                     &match_result_tensor,
                     task]() {
        return;
        printf("\n");

        if (task == Task::Zone ||
            task == Task::ZoneCaptureDefend ||
            task == Task::TDM ||
            task == Task::Turret) {
            auto global_pos_printer = global_pos_tensor.makePrinter();
            auto self_printer = self_obs_tensor.makePrinter();

            auto teammates_pos_printer =
                mgr.teammatePositionObservationsTensor().makePrinter();
            auto opponents_pos_printer =
                mgr.opponentPositionObservationsTensor().makePrinter();

            auto teammates_state_printer =
                mgr.teammateStateObservationsTensor().makePrinter();
            auto opponents_state_printer =
                mgr.opponentStateObservationsTensor().makePrinter();

            auto opponents_masks_printer = mgr.opponentMasksTensor().makePrinter();
            auto reward_printer = mgr.rewardTensor().makePrinter();
            auto done_printer = mgr.doneTensor().makePrinter();

            auto magazine_printer = magazine_tensor.makePrinter();
            auto hp_printer = hp_tensor.makePrinter();

            auto match_result_printer = match_result_tensor.makePrinter();

            printf("Global Pos\n");
            global_pos_printer.print();

            printf("Self\n");
            self_printer.print();

            printf("Teammates\n");
            teammates_pos_printer.print();
            teammates_state_printer.print();

            printf("Opponents\n");
            opponents_pos_printer.print();
            opponents_state_printer.print();

            printf("Opponent Masks\n");
            opponents_masks_printer.print();

            printf("Magazine\n");
            magazine_printer.print();

            printf("HP\n");
            hp_printer.print();

            printf("Reward\n");
            reward_printer.print();

            printf("Done\n");
            done_printer.print();

            printf("Match Result\n");
            match_result_printer.print();
        } else if (task == Task::Explore) {
            auto self_printer = self_obs_tensor.makePrinter();
            auto fwd_lidar_printer = mgr.fwdLidarTensor().makePrinter();
            auto reward_printer = mgr.rewardTensor().makePrinter();

            printf("Self\n");
            self_printer.print();

            printf("Fwd Lidar\n");
            fwd_lidar_printer.print();

            printf("Reward\n");
            reward_printer.print();
        }
    };

    printObs();
#endif

    VizSystem::loop(viz, mgr);

    VizSystem::shutdown(viz);
}
