#include "dnn.hpp"
#include "sim.hpp"

#include <filesystem>
#include <fstream>

namespace madronaMPEnv {

struct NormParam {
  float mean;
  float std;
};

struct StandObNormalize {
  NormParam curStanding;
  NormParam curCrouching;
  NormParam curProning;
  NormParam tgtStanding;
  NormParam tgtCrouching;
  NormParam tgtProning;
  NormParam transitionRemaining;
};

struct PlayerCommonObNormalize {
  NormParam isValid; // Must be first observation
  NormParam isAlive;
  NormParam globalX;
  NormParam globalY;
  NormParam globalZ;
  NormParam facingYaw;
  NormParam facingPitch;
  NormParam velocityX;
  NormParam velocityY;
  NormParam velocityZ;
  StandObNormalize stand;
  NormParam inZone;
  std::array<NormParam, consts::maxNumWeaponTypes> weaponTypeObs;
};

struct CombatStateObNormalize {
  NormParam hp;
  NormParam magazine;
  NormParam reloading;
  NormParam autohealTime;
};

struct ZoneObNormalize {
  NormParam centerX;
  NormParam centerY;
  NormParam centerZ;
  NormParam toCenterDist;
  NormParam toCenterYaw;
  NormParam toCenterPitch;
  NormParam myTeamControlling;
  NormParam enemyTeamControlling;
  NormParam isContested;
  NormParam isCaptured;
  NormParam stepsUntilPoint;
  NormParam stepsRemaining;
  std::array<NormParam, 4> id = {};
};

struct SelfObNormalize {
  CombatStateObNormalize combat;
  NormParam fractionRemaining;
  ZoneObNormalize zone;
};

struct NormalizeObParameters {
  SelfObNormalize self;

};

struct PolicyWeights {
  NormalizeObParameters normOb;
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

static PlayerCommonObNormalize setupPlayerCommonObNormal(
  float **self_mu, float **self_std)
{
  NormParam isValid; // Must be first observation
  NormParam isAlive;
  NormParam globalX;
  NormParam globalY;
  NormParam globalZ;
  NormParam facingYaw;
  NormParam facingPitch;
  NormParam velocityX;
  NormParam velocityY;
  NormParam velocityZ;
  StandObNormalize stand;
  NormParam inZone;
  std::array<NormParam, consts::maxNumWeaponTypes> weaponTypeObs;
}

static SelfObNormalize setupSelfNormOb(float *self_mu, float *self_std)
{
  float *self_mu_start = self_mu;

  SelfObNormalize norm;

  norm.combat.hp = { *self_mu++, *self_std++ };
  norm.combat.magazine = { *self_mu++, *self_std++ };
  norm.combat.reloading = { *self_mu++, *self_std++ };
  norm.combat.autohealTime = { *self_mu++, *self_std++ };

  norm.fractionRemaining = { *self_mu++, *self_std++ };

  norm.zone.centerX = { *self_mu++, *self_std++ };
  norm.zone.centerY = { *self_mu++, *self_std++ };
  norm.zone.centerZ = { *self_mu++, *self_std++ };
  norm.zone.toCenterDist = { *self_mu++, *self_std++ };
  norm.zone.toCenterYaw = { *self_mu++, *self_std++ };
  norm.zone.toCenterPitch = { *self_mu++, *self_std++ };
  norm.zone.myTeamControlling = { *self_mu++, *self_std++ };
  norm.zone.enemyTeamControlling = { *self_mu++, *self_std++ };
  norm.zone.isContested = { *self_mu++, *self_std++ };
  norm.zone.isCaptured = { *self_mu++, *self_std++ };
  norm.zone.stepsUntilPoint = { *self_mu++, *self_std++ };
  norm.zone.stepsRemaining = { *self_mu++, *self_std++ };
  norm.zone.id[0] = { *self_mu++, *self_std++ };
  norm.zone.id[1] = { *self_mu++, *self_std++ };
  norm.zone.id[2] = { *self_mu++, *self_std++ };
  norm.zone.id[3] = { *self_mu++, *self_std++ };

  assert(self_mu - self_mu_start == sizeof(SelfObservation) / sizeof(float));

  return norm;
}

static int loadWeightFile(const char *dir, const char *name,
                          float **weights, int32_t **shape)
{
  std::filesystem::path path = std::filesystem::path(dir) / name;

  std::ifstream file(path, std::ios::binary);

  int32_t ndim;
  file.read((char *)&ndim, sizeof(int32_t));

  *shape = (int32_t *)malloc(sizeof(int32_t) * ndim);

  file.read((char *)*shape, sizeof(int32_t) * ndim);

  int shape_product = (*shape)[0];
  for (int i = 1; i < ndim; i++) {
    shape_product *= (*shape)[i];
  }

  *weights = (float *)malloc(sizeof(float) * shape_product);
  file.read((char *)*weights, sizeof(float) * shape_product);

  return ndim;
}

PolicyWeights * loadPolicyWeights(const char *path)
{
  PolicyWeights *weights = new PolicyWeights {};

  {
    float *self_mu;
    int32_t *self_mu_shape;
    int self_mu_ndim = loadWeightFile(
      path, "obs_preprocess_state_self_mu", &self_mu, &self_mu_shape);

    float *self_std;
    int32_t *self_std_shape;
    int self_std_ndim = loadWeightFile(
      path, "obs_preprocess_state_self_sigma", &self_std, &self_std_shape);

    assert(self_mu_ndim == 1 && self_mu_shape[0] == sizeof(SelfObservation) / sizeof(float));
    assert(self_std_ndim == 1 && self_std_shape[0] == self_mu_shape[0]);

    weights->normOb.self = setupSelfNormOb(self_mu, self_std);

    free(self_mu);
    free(self_mu_shape);

    free(self_std);
    free(self_std_shape);
  }

  return weights;
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
