#pragma once

#include <madrona/components.hpp>
#include <madrona/context.hpp>
#include <madrona/registry.hpp>
#include <madrona/taskgraph_builder.hpp>

namespace madronaMPEnv {

using madrona::Context;
using madrona::Entity;
using madrona::ECSRegistry;
using madrona::math::Vector3;
using madrona::math::AABB;

struct VizState;
class Manager;

struct VizConfig {
  uint32_t windowWidth;
  uint32_t windowHeight;

  uint32_t numWorlds;
  uint32_t numViews;
  uint32_t teamSize;

  bool doAITeam1;
  bool doAITeam2;

  bool skipMainMenu;

  const char *analyticsDBPath;
  const char *recordedDataPath;
  const char *trajectoryDBPath;
};

struct VizCamera {};

#ifndef MADRONA_GPU_MODE

namespace VizSystem {

void init(const VizConfig &cfg, void (*cb)(VizState *, void *), void *data_ptr);

void bootMenu(VizState *viz, void (*cb)(VizState *, std::string scene_dir, void *), void *data_ptr);
void loadMapAssets(VizState *viz, const char *map_assets_path);

void shutdown(VizState *viz);

void initWorld(Context &ctx, VizState *viz);

void registerTypes(ECSRegistry &registry);

void setupGameTasks(VizState *viz, madrona::TaskGraphBuilder &builder);

void loop(VizState *viz, Manager &mgr);

};

#endif

}
