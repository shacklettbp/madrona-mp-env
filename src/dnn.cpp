#include "dnn.hpp"
#include "sim.hpp"

#include <filesystem>
#include <fstream>

namespace madronaMPEnv {

struct NormParam {
  float mean;
  float invStd;
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

struct OtherPlayerCommonObNormalize : PlayerCommonObNormalize {
  NormParam toPlayerDist;
  NormParam toPlayerYaw;
  NormParam toPlayerPitch;
  NormParam relativeFacingYaw;
  NormParam relativeFacingPitch;
};

struct CombatStateObNormalize {
  NormParam hp;
  NormParam magazine;
  NormParam reloading;
  NormParam autohealTime;
};

struct TeammateObNormalize : OtherPlayerCommonObNormalize {
  CombatStateObNormalize combat;
};

struct OpponentObNormalize : OtherPlayerCommonObNormalize {
  NormParam wasHit;
  NormParam firedShot;
  NormParam hasLOS;
  NormParam teamKnowsLocation;
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

struct SelfObNormalize : PlayerCommonObNormalize {
  CombatStateObNormalize combat;
  NormParam fractionRemaining;
  ZoneObNormalize zone;
};

struct LidarDataNormalize {
  NormParam depth;
  NormParam isWall;
  NormParam isTeammate;
  NormParam isOpponent;
};

struct RewardHyperParamsNorm {
  NormParam teamSpirit;
  NormParam shotScale;
  NormParam exploreScale;
  NormParam inZoneScale;
  NormParam zoneTeamContestScale;
  NormParam zoneTeamCtrlScale;
  NormParam zoneDistScale;
  NormParam zoneEarnedPointScale;
  NormParam breadcrumbScale;
};

struct NormalizeObParameters {
  SelfObNormalize self;
  TeammateObNormalize teammate;
  OpponentObNormalize opponent;
  OpponentObNormalize opponentLastKnown;
  LidarDataNormalize fwdLidar;
  LidarDataNormalize rearLidar;
  RewardHyperParamsNorm rewardCoefs;
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

static RewardHyperParamsNorm setupRewardCoefsNorm(
    float *mu, float *inv_std)
{
  float *mu_start = mu;

  RewardHyperParamsNorm norm;
  norm.teamSpirit = { *mu++, *inv_std++ };
  norm.shotScale = { *mu++, *inv_std++ };
  norm.exploreScale = { *mu++, *inv_std++ };
  norm.inZoneScale = { *mu++, *inv_std++ };
  norm.zoneTeamContestScale = { *mu++, *inv_std++ };
  norm.zoneTeamCtrlScale = { *mu++, *inv_std++ };
  norm.zoneDistScale = { *mu++, *inv_std++ };
  norm.zoneEarnedPointScale = { *mu++, *inv_std++ };
  norm.breadcrumbScale = { *mu++, *inv_std++ };

  assert(mu - mu_start == sizeof(RewardHyperParams) / sizeof(float));

  return norm;
}

static LidarDataNormalize setupLidarNormOb(
    float *mu, float *inv_std)
{
  LidarDataNormalize norm;

  norm.depth = { *mu++, *inv_std++ };
  norm.isWall = { *mu++, *inv_std++ };
  norm.isTeammate = { *mu++, *inv_std++ };
  norm.isOpponent = { *mu++, *inv_std++ };

  return norm;
}

static void setupPlayerCommonObNormal(
  PlayerCommonObNormalize &norm,
  float **self_mu_inout, float **self_inv_std_inout)
{
  float *self_mu = *self_mu_inout;
  float *self_inv_std = *self_inv_std_inout;

  norm.isValid = { *self_mu++, *self_inv_std++ };

  norm.isAlive = { *self_mu++, *self_inv_std++ };
  norm.globalX = { *self_mu++, *self_inv_std++ };
  norm.globalY = { *self_mu++, *self_inv_std++ };
  norm.globalZ = { *self_mu++, *self_inv_std++ };
  norm.facingYaw = { *self_mu++, *self_inv_std++ };
  norm.facingPitch = { *self_mu++, *self_inv_std++ };
  norm.velocityX = { *self_mu++, *self_inv_std++ };
  norm.velocityY = { *self_mu++, *self_inv_std++ };
  norm.velocityZ = { *self_mu++, *self_inv_std++ };
  norm.stand.curStanding = { *self_mu++, *self_inv_std++ };
  norm.stand.curCrouching = { *self_mu++, *self_inv_std++ };
  norm.stand.curProning = { *self_mu++, *self_inv_std++ };
  norm.stand.tgtStanding = { *self_mu++, *self_inv_std++ };
  norm.stand.tgtCrouching = { *self_mu++, *self_inv_std++ };
  norm.stand.tgtProning = { *self_mu++, *self_inv_std++ };
  norm.stand.transitionRemaining = { *self_mu++, *self_inv_std++ };
  norm.inZone = { *self_mu++, *self_inv_std++ };

  for (int i = 0; i < consts::maxNumWeaponTypes; i++) {
    norm.weaponTypeObs[i] = { *self_mu++, *self_inv_std++ };
  }

  *self_mu_inout = self_mu;
  *self_inv_std_inout = self_inv_std;
}

static CombatStateObNormalize setupCombatNormOb(float **mu_inout,
                                                float **inv_std_inout)
{
  float *mu = *mu_inout;
  float *inv_std = *inv_std_inout;

  CombatStateObNormalize combat;

  combat.hp = { *mu++, *inv_std++ };
  combat.magazine = { *mu++, *inv_std++ };
  combat.reloading = { *mu++, *inv_std++ };
  combat.autohealTime = { *mu++, *inv_std++ };

  *mu_inout = mu;
  *inv_std_inout = inv_std;

  return combat;
}

static void setupOtherPlayerCommonObNorm(OtherPlayerCommonObNormalize &norm,
                                         float **mu_inout, float **inv_std_inout)
{
  float *mu = *mu_inout;
  float *inv_std = *inv_std_inout;

  setupPlayerCommonObNormal(norm, &mu, &inv_std);

  norm.toPlayerDist = { *mu++, *inv_std++ };
  norm.toPlayerYaw = { *mu++, *inv_std++ };
  norm.toPlayerPitch = { *mu++, *inv_std++ };
  norm.relativeFacingYaw = { *mu++, *inv_std++ };
  norm.relativeFacingPitch = { *mu++, *inv_std++ };

  *mu_inout = mu;
  *inv_std_inout = inv_std;
}

static SelfObNormalize setupSelfNormOb(float *mu, float *inv_std)
{
  float *mu_start = mu;

  SelfObNormalize norm;
  setupPlayerCommonObNormal(norm, &mu, &inv_std);

  norm.combat = setupCombatNormOb(&mu, &inv_std);

  norm.fractionRemaining = { *mu++, *inv_std++ };

  norm.zone.centerX = { *mu++, *inv_std++ };
  norm.zone.centerY = { *mu++, *inv_std++ };
  norm.zone.centerZ = { *mu++, *inv_std++ };
  norm.zone.toCenterDist = { *mu++, *inv_std++ };
  norm.zone.toCenterYaw = { *mu++, *inv_std++ };
  norm.zone.toCenterPitch = { *mu++, *inv_std++ };
  norm.zone.myTeamControlling = { *mu++, *inv_std++ };
  norm.zone.enemyTeamControlling = { *mu++, *inv_std++ };
  norm.zone.isContested = { *mu++, *inv_std++ };
  norm.zone.isCaptured = { *mu++, *inv_std++ };
  norm.zone.stepsUntilPoint = { *mu++, *inv_std++ };
  norm.zone.stepsRemaining = { *mu++, *inv_std++ };
  norm.zone.id[0] = { *mu++, *inv_std++ };
  norm.zone.id[1] = { *mu++, *inv_std++ };
  norm.zone.id[2] = { *mu++, *inv_std++ };
  norm.zone.id[3] = { *mu++, *inv_std++ };

  assert(mu - mu_start == sizeof(SelfObservation) / sizeof(float));

  return norm;
}

static TeammateObNormalize setupTeammateNormOb(float *mu, float *inv_std)
{
  float *mu_start = mu;

  TeammateObNormalize norm;
  setupOtherPlayerCommonObNorm(norm, &mu, &inv_std);

  norm.combat = setupCombatNormOb(&mu, &inv_std);


  assert(mu - mu_start == sizeof(TeammateObservation) / sizeof(float));

  return norm;
}

static OpponentObNormalize setupOpponentNormOb(float *mu, float *inv_std)
{
  float *mu_start = mu;

  OpponentObNormalize norm;
  setupOtherPlayerCommonObNorm(norm, &mu, &inv_std);

  norm.wasHit = { *mu++, *inv_std++ };
  norm.firedShot = { *mu++, *inv_std++ };
  norm.hasLOS = { *mu++, *inv_std++ };
  norm.teamKnowsLocation = { *mu++, *inv_std++ };

  assert(mu - mu_start == sizeof(OpponentObservation) / sizeof(float));

  return norm;
}

static int loadWeightFile(const char *dir, const char *name,
                          float **weights, int32_t **shape)
{
  std::filesystem::path path = std::filesystem::path(dir) / name;

  std::ifstream file(path, std::ios::binary);
  assert(file.is_open());

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

    float *self_inv_std;
    int32_t *self_inv_std_shape;
    int self_inv_std_ndim = loadWeightFile(
      path, "obs_preprocess_state_self_inv_sigma", &self_inv_std, &self_inv_std_shape);

    assert(self_mu_ndim == 1 && self_mu_shape[0] == sizeof(SelfObservation) / sizeof(float));
    assert(self_inv_std_ndim == 1 && self_inv_std_shape[0] == self_mu_shape[0]);

    weights->normOb.self = setupSelfNormOb(self_mu, self_inv_std);

    free(self_mu);
    free(self_mu_shape);

    free(self_inv_std);
    free(self_inv_std_shape);
  }

  {
    float *teammate_mu;
    int32_t *teammate_mu_shape;
    int teammate_mu_ndim = loadWeightFile(
        path, "obs_preprocess_state_teammates_mu",
        &teammate_mu, &teammate_mu_shape);

    float *teammate_inv_std;
    int32_t *teammate_inv_std_shape;
    int teammate_inv_std_ndim = loadWeightFile(
        path, "obs_preprocess_state_teammates_inv_sigma",
        &teammate_inv_std, &teammate_inv_std_shape);

    assert(teammate_mu_ndim == 1 && teammate_mu_shape[0] ==
           sizeof(TeammateObservation) / sizeof(float));
    assert(teammate_inv_std_ndim == 1 &&
           teammate_inv_std_shape[0] == teammate_mu_shape[0]);

    weights->normOb.teammate = setupTeammateNormOb(teammate_mu, teammate_inv_std);

    free(teammate_mu);
    free(teammate_mu_shape);

    free(teammate_inv_std);
    free(teammate_inv_std_shape);
  }

  {
    float *opponent_mu;
    int32_t *opponent_mu_shape;
    int opponent_mu_ndim = loadWeightFile(
        path, "obs_preprocess_state_opponents_mu",
        &opponent_mu, &opponent_mu_shape);

    float *opponent_inv_std;
    int32_t *opponent_inv_std_shape;
    int opponent_inv_std_ndim = loadWeightFile(
        path, "obs_preprocess_state_opponents_inv_sigma",
        &opponent_inv_std, &opponent_inv_std_shape);

    assert(opponent_mu_ndim == 1 && opponent_mu_shape[0] ==
           sizeof(OpponentObservation) / sizeof(float));
    assert(opponent_inv_std_ndim == 1 &&
           opponent_inv_std_shape[0] == opponent_mu_shape[0]);

    weights->normOb.opponent = setupOpponentNormOb(opponent_mu, opponent_inv_std);

    free(opponent_mu);
    free(opponent_mu_shape);

    free(opponent_inv_std);
    free(opponent_inv_std_shape);
  }

  {
    float *opponent_mu;
    int32_t *opponent_mu_shape;
    int opponent_mu_ndim = loadWeightFile(
        path, "obs_preprocess_state_opponents_last_known_mu",
        &opponent_mu, &opponent_mu_shape);

    float *opponent_inv_std;
    int32_t *opponent_inv_std_shape;
    int opponent_inv_std_ndim = loadWeightFile(
        path, "obs_preprocess_state_opponents_last_known_inv_sigma",
        &opponent_inv_std, &opponent_inv_std_shape);

    assert(opponent_mu_ndim == 1 && opponent_mu_shape[0] ==
           sizeof(OpponentObservation) / sizeof(float));
    assert(opponent_inv_std_ndim == 1 &&
           opponent_inv_std_shape[0] == opponent_mu_shape[0]);

    weights->normOb.opponentLastKnown =
        setupOpponentNormOb(opponent_mu, opponent_inv_std);

    free(opponent_mu);
    free(opponent_mu_shape);

    free(opponent_inv_std);
    free(opponent_inv_std_shape);
  }

  {
    float *mu;
    int32_t *mu_shape;
    int mu_ndim = loadWeightFile(
        path, "obs_preprocess_state_fwd_lidar_mu",
        &mu, &mu_shape);

    float *inv_std;
    int32_t *inv_std_shape;
    int inv_std_ndim = loadWeightFile(
        path, "obs_preprocess_state_fwd_lidar_inv_sigma",
        &inv_std, &inv_std_shape);

    assert(mu_ndim == 1 && mu_shape[0] ==
           sizeof(LidarData) / sizeof(float));
    assert(inv_std_ndim == 1 &&
           inv_std_shape[0] == mu_shape[0]);

    weights->normOb.fwdLidar =
        setupLidarNormOb(mu, inv_std);

    free(mu);
    free(mu_shape);

    free(inv_std);
    free(inv_std_shape);
  }

  {
    float *mu;
    int32_t *mu_shape;
    int mu_ndim = loadWeightFile(
        path, "obs_preprocess_state_rear_lidar_mu",
        &mu, &mu_shape);

    float *inv_std;
    int32_t *inv_std_shape;
    int inv_std_ndim = loadWeightFile(
        path, "obs_preprocess_state_rear_lidar_inv_sigma",
        &inv_std, &inv_std_shape);

    assert(mu_ndim == 1 && mu_shape[0] ==
           sizeof(LidarData) / sizeof(float));
    assert(inv_std_ndim == 1 &&
           inv_std_shape[0] == mu_shape[0]);

    weights->normOb.rearLidar =
        setupLidarNormOb(mu, inv_std);

    free(mu);
    free(mu_shape);

    free(inv_std);
    free(inv_std_shape);
  }

  {
    float *mu;
    int32_t *mu_shape;
    int mu_ndim = loadWeightFile(
        path, "obs_preprocess_state_reward_coefs_mu",
        &mu, &mu_shape);

    float *inv_std;
    int32_t *inv_std_shape;
    int inv_std_ndim = loadWeightFile(
        path, "obs_preprocess_state_reward_coefs_inv_sigma",
        &inv_std, &inv_std_shape);

    assert(mu_ndim == 1 && mu_shape[0] ==
           sizeof(RewardHyperParams) / sizeof(float));
    assert(inv_std_ndim == 1 &&
           inv_std_shape[0] == mu_shape[0]);

    weights->normOb.rewardCoefs =
        setupRewardCoefsNorm(mu, inv_std);

    free(mu);
    free(mu_shape);

    free(inv_std);
    free(inv_std_shape);
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
