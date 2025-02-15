#pragma once

#include "types.hpp"

namespace madronaMPEnv {

struct PolicyWeights;

PolicyWeights * loadPolicyWeights(const char *path);
void addPolicyEvalTasks(TaskGraphBuilder &builder);

}
