#pragma once

#include "sim.hpp"

namespace madronaMPEnv {

// Creates agents, outer walls and floor. Entities that will persist across
// all episodes.
void createPersistentEntities(Engine &ctx, const TaskConfig &cfg);

void resetPersistentEntities(Engine &ctx, madrona::RandKey episode_rand_key);

}
