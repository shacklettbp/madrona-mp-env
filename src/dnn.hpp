#pragma once

#include "types.hpp"
#include "sim.hpp"

namespace madronaMPEnv {

struct PolicyWeights;

PolicyWeights * loadPolicyWeights(const char *path);
void addPolicyEvalTasks(TaskGraphBuilder &builder);

}
