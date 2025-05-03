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

struct InitArgs {
  std::string scene_dir;
  Task task;
  int64_t num_views;
  int64_t team_size;
  uint32_t num_worlds;
  ExecMode exec_mode;
  char *record_log_path;
  char *replay_log_path;
  char *curriculum_data_path;
  char *policy_weights_path;
  char *recorded_data_path;
  char *trajectory_db_path;
};

void postDeviceCreate(VizState *viz, void *data_ptr)
{
  InitArgs args;
  {
    InitArgs *init_args_ptr = (InitArgs *)data_ptr;
    args = *init_args_ptr;
    delete (InitArgs *)data_ptr;
  }

  if (args.scene_dir.empty()) {
    args.scene_dir = VizSystem::bootMenu(viz);
    if (args.scene_dir.empty()) {
      return;
    }
  }

  std::string collision_data_file = args.scene_dir + "/collisions.bin";
  std::string navmesh_file = args.scene_dir + "/navmesh.bin";
  std::string spawn_data_file = args.scene_dir + "/spawns.bin";
  std::string zone_data_file = args.scene_dir + "/zones.bin";

  VizSystem::loadMapAssets(viz, collision_data_file.c_str());

  const bool highlevel_move = false;

  SimFlags sim_flags = SimFlags::Default;
  if (args.task == Task::ZoneCaptureDefend) {
    sim_flags = SimFlags::HardcodedSpawns;
  }

  //sim_flags |= SimFlags::SubZones;
  sim_flags |= SimFlags::SimEvalMode;

  Manager mgr({
      .execMode = args.exec_mode,
      .gpuID = 0,
      .numWorlds = args.num_worlds,
      .randSeed = 10,
      .autoReset = args.replay_log_path != nullptr,
      //.simFlags = SimFlags::HardcodedSpawns /*| SimFlags::StaticFlipTeams*/,
      .simFlags = sim_flags,
      .taskType = args.task,
      .teamSize = (uint32_t)args.team_size,
      .numPBTPolicies = 0,
      .policyHistorySize = 1,
      .map = {
        .name = args.scene_dir.c_str(),
        .collisionDataFile = collision_data_file.c_str(),
        .navmeshFile = navmesh_file.c_str(),
        .spawnDataFile = spawn_data_file.c_str(),
        .zoneDataFile = zone_data_file.c_str(),
        .mapOffset = Vector3::zero(),
        .mapRotation = 0.f,
      },
      .highlevelMove = highlevel_move,
      .replayLogPath = args.replay_log_path,
      .recordLogPath = args.record_log_path,
      .curriculumDataPath = args.curriculum_data_path,
      .policyWeightsPath = args.policy_weights_path,
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
    bool skip_main_menu = false;

    char *analytics_db_path = nullptr;
    char *trajectory_db_path = nullptr;
    char *curriculum_data_path = nullptr;
    char *policy_weights_path = nullptr;
    char *recorded_data_path = nullptr;

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
          skip_main_menu = true;
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
        } else if (!strcmp("load-recorded-data", arg)) {
          if (recorded_data_path != nullptr) {
            usageErr();
          }

          i += 1;
          if (i == argc) {
            usageErr();
          }

          recorded_data_path = argv[i];
        } else if (!strcmp("trajectory-db", arg)) {
          if (trajectory_db_path != nullptr) {
            usageErr();
          }

          i += 1;
          if (i == argc) {
            usageErr();
          }

          trajectory_db_path = argv[i];
        }
      } else {
        if (num_worlds_set) {
          usageErr();
        }

        num_worlds_set = true;

        num_worlds = (uint32_t)atoi(arg);
      }
    }

    InitArgs *init_args = new InitArgs {
        .scene_dir = std::move(scene_dir),
        .task = task,
        .num_views = num_views,
        .team_size = team_size,
        .num_worlds = num_worlds,
        .exec_mode = exec_mode,
        .record_log_path = record_log_path,
        .replay_log_path = replay_log_path,
        .curriculum_data_path = curriculum_data_path,
        .policy_weights_path = policy_weights_path,
        .recorded_data_path = recorded_data_path,
        .trajectory_db_path = trajectory_db_path,
    };

    VizSystem::init(VizConfig {
      .windowWidth = 2730,
      .windowHeight = 1536,
      //.windowWidth = 3840,
      //.windowHeight = 2160,

      .numWorlds = (uint32_t)num_worlds,
      .numViews = (uint32_t)num_views,
      .teamSize = (uint32_t)team_size,

      .doAITeam1 = doAITeam1,
      .doAITeam2 = doAITeam2,
      .skipMainMenu = skip_main_menu,
      .analyticsDBPath = analytics_db_path,
      .recordedDataPath = recorded_data_path,
      .trajectoryDBPath = trajectory_db_path,
    }, postDeviceCreate, init_args);
}