#include "mgr.hpp"

#include <cstdio>
#include <chrono>
#include <string>
#include <filesystem>
#include <fstream>
#include <random>

using namespace madrona;

[[maybe_unused]] static void saveWorldActions(
    const HeapArray<int32_t> &action_store,
    int32_t total_num_steps,
    int32_t world_idx)
{
    const int32_t *world_base = action_store.data() + world_idx * total_num_steps * 2 * 3;

    std::ofstream f("/tmp/actions", std::ios::binary);
    f.write((char *)world_base,
            sizeof(uint32_t) * total_num_steps * 2 * 3);
}

int main(int argc, char *argv[])
{
    using namespace madronaMPEnv;

    if (argc < 4) {
        fprintf(stderr, "%s TYPE NUM_WORLDS NUM_STEPS SCENE [--rand-actions]\n", argv[0]);
        return -1;
    }
    std::string type(argv[1]);

    ExecMode exec_mode;
    if (type == "CPU") {
        exec_mode = ExecMode::CPU;
    } else if (type == "CUDA") {
        exec_mode = ExecMode::CUDA;
    } else {
        fprintf(stderr, "Invalid ExecMode\n");
        return -1;
    }

    uint64_t num_worlds = std::stoul(argv[2]);
    uint64_t num_steps = std::stoul(argv[3]);

    HeapArray<int32_t> action_store(
        num_worlds * 2 * num_steps * 3);

    bool rand_actions = false;
    if (argc >= 6) {
        if (std::string(argv[5]) == "--rand-actions") {
            rand_actions = true;
        }
    }

    std::string scene_dir = "data/simple_map";

    std::string collision_data_file = scene_dir + "/collisions.bin";
    std::string navmesh_file = scene_dir + "/navmesh.bin";
    std::string spawn_data_file = scene_dir + "/spawns.bin";
    std::string zone_data_file = scene_dir + "/zones.bin";

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = (uint32_t)num_worlds,
        .randSeed = 10,
        .autoReset = false,
        //.simFlags = /*SimFlags::Default,*/ SimFlags::HardcodedSpawns,
        .simFlags = SimFlags::Default,
        .taskType = Task::Zone,
        .teamSize = (uint32_t)6,
        .numPBTPolicies = 0,
        .policyHistorySize = 1,
        .map = {
          .name = scene_dir.c_str(),
          .collisionDataFile = collision_data_file.c_str(),
          .navmeshFile = navmesh_file.c_str(),
          .spawnDataFile = spawn_data_file.c_str(),
          .zoneDataFile = zone_data_file.c_str(),
          .mapOffset = Vector3::zero(),
          .mapRotation = 0.f,
        },
        .highlevelMove = false,
        .policyWeightsPath = "converted_ckpt/0",
    });
    mgr.init();

    mgr.step();

    return 0;

    std::random_device rd;
    std::mt19937 rand_gen(rd());
    std::uniform_int_distribution<int32_t> act_rand(0, 4);

    auto start = std::chrono::system_clock::now();

    for (CountT i = 0; i < (CountT)num_steps; i++) {
        if (rand_actions) {
            for (CountT j = 0; j < (CountT)num_worlds; j++) {
                int32_t x = act_rand(rand_gen);
                int32_t y = act_rand(rand_gen);
                int32_t r = act_rand(rand_gen);

                mgr.setExploreAction(j, ExploreAction {
                    .moveAmount = x,
                    .moveAngle = y,
                    .rotate = r,
                    .mantle = 0,
                });
                
                int64_t base_idx = j * num_steps * 3 + i * 3;
                action_store[base_idx] = x;
                action_store[base_idx + 1] = y;
                action_store[base_idx + 2] = r;
            }
        }
        mgr.step();
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    float fps = (double)num_steps * (double)num_worlds / elapsed.count();
    printf("FPS %f\n", fps);
}
