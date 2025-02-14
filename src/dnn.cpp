#include "dnn.hpp"

namespace madronaMPEnv {

struct NormParam {
  float mean;
  float std;
};

struct SelfObNormalize {
  NormParam combatHP;
  NormParam combatMagazine;
  NormParam combatReloading;
  NormParam combatAutohealTime;
};

struct NormalizeParameters {
};

struct PolicyWeights {
};

void evalAgentPolicy(
  Engine &ctx,
  const SelfObservation &self_ob,
  const TeammateObservations &teammate_obs,
  const OpponentObservations &opponent_obs,
  const OpponentLastKnownObservations &opponent_last_known_obs,

  const SelfPositionObservation &self_pos_ob,
  const TeammatePositionObservations &teammate_pos_obs,
  const OpponentPositionObservations &opponent_pos_obs,
  const OpponentLastKnownPositionObservations &opponent_last_known_pos_obs,

  const OpponentMasks &opponent_masks,
  const FiltersStateObservation &filters_state_obs,
  const FwdLidar &fwd_lidar,
  const RearLidar &read_lidar)
{

}

void addPolicyEvalTasks(TaskGraphBuilder &builder)
{
  builder.addToGraph<ParallelForNode<Engine,
    evalAgentPolicy,
      SelfObservation,
      TeammateObservations,
      OpponentObservations,
      OpponentLastKnownObservations,

      SelfPositionObservation,
      TeammatePositionObservations,
      OpponentPositionObservations,
      OpponentLastKnownPositionObservations,

      OpponentMasks,
      FiltersStateObservation,
      FwdLidar,
      RearLidar
    >>({});
}

}
