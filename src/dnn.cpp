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
  NormParam yawVelocity;
  NormParam pitchVelocity;
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

struct FullyConnectedParams {
  float *weights;
  float *bias;
  int numInputs;
  int numFeatures;
};

struct LayerNormParams {
  float *bias;
  float *scale;
  int numFeatures;
};

struct FullyConnectedLayerWithLayerNormWithActivation {
  FullyConnectedParams params;
  LayerNormParams layerNorm;
};

struct EmbedModule {
  FullyConnectedLayerWithLayerNormWithActivation fc;
};

struct PolicyWeights {
  NormalizeObParameters normOb;

  EmbedModule selfEmbed;

  EmbedModule fwdLidarEmbed;
  EmbedModule rearLidarEmbed;

  EmbedModule teammatesEmbed;
  EmbedModule opponentsEmbed;
  EmbedModule opponentsLastKnownEmbed;

  std::array<FullyConnectedLayerWithLayerNormWithActivation, 3> mlp;

  static constexpr inline auto discreteActionsNumBuckets = std::to_array<int32_t>({
      3, // moveAmount
      8, // moveAngle
      3, // fire
      3, // stand
  });

  static constexpr inline auto discreteAimNumBuckets = std::to_array<int32_t>({
      15, // yaw
      7, // pitch
  });

  FullyConnectedParams discreteHead;
  FullyConnectedParams aimHead;
};

constexpr int totalNumDiscreteActionBuckets()
{
  int sum = 0;
  for (int i = 0; i < (int)PolicyWeights::discreteActionsNumBuckets.size();
       i++) {
    sum += PolicyWeights::discreteActionsNumBuckets[i];
  }

  return sum;
}

constexpr int totalNumDiscreteAimActionBuckets()
{
  int sum = 0;
  for (int i = 0; i < (int)PolicyWeights::discreteAimNumBuckets.size();
       i++) {
    sum += PolicyWeights::discreteAimNumBuckets[i];
  }

  return sum;
}

static float * normalizeCombatOb(CombatStateObservation &ob,
                                 CombatStateObNormalize &params,
                                 float *out)
{
  *out++ = params.hp.invStd * (ob.hp - params.hp.mean);
  *out++ = params.magazine.invStd * (ob.magazine - params.magazine.mean);
  *out++ = params.reloading.invStd * (ob.isReloading - params.reloading.mean);
  *out++ = params.autohealTime.invStd *
      (ob.timeBeforeAutoheal - params.autohealTime.mean);

  return out;
}

static float * normalizeStandOb(StandObservation &ob,
                                StandObNormalize &params,
                                float *out)
{
  *out++ = params.curStanding.invStd * (ob.curStanding - params.curStanding.mean);
  *out++ = params.curCrouching.invStd * (ob.curCrouching - params.curCrouching.mean);
  *out++ = params.curProning.invStd * (ob.curProning - params.curProning.mean);
  *out++ = params.tgtStanding.invStd * (ob.tgtStanding - params.tgtStanding.mean);
  *out++ = params.tgtCrouching.invStd * (ob.tgtCrouching - params.tgtCrouching.mean);
  *out++ = params.tgtProning.invStd * (ob.tgtProning - params.tgtProning.mean);
  *out++ = params.transitionRemaining.invStd *
      (ob.transitionRemaining - params.transitionRemaining.mean);

  return out;
}

static float * normalizePlayerCommonOb(PlayerCommonObservation &ob,
                                       PlayerCommonObNormalize &params,
                                       float *out)
{
  *out++ = params.isValid.invStd * (ob.isValid - params.isValid.mean);
  *out++ = params.isAlive.invStd * (ob.isAlive - params.isAlive.mean);
  *out++ = params.globalX.invStd * (ob.globalX - params.globalX.mean);
  *out++ = params.globalY.invStd * (ob.globalY - params.globalY.mean);
  *out++ = params.globalZ.invStd * (ob.globalZ - params.globalZ.mean);
  *out++ = params.facingYaw.invStd * (ob.facingYaw - params.facingYaw.mean);
  *out++ = params.facingPitch.invStd * (ob.facingPitch - params.facingPitch.mean);
  *out++ = params.velocityX.invStd * (ob.velocityX - params.velocityX.mean);
  *out++ = params.velocityY.invStd * (ob.velocityY - params.velocityY.mean);
  *out++ = params.velocityZ.invStd * (ob.velocityZ - params.velocityZ.mean);
  *out++ = params.yawVelocity.invStd * (ob.yawVelocity - params.yawVelocity.mean);
  *out++ = params.pitchVelocity.invStd * (ob.pitchVelocity - params.pitchVelocity.mean);

  out = normalizeStandOb(ob.stand, params.stand, out);

  *out++ = params.inZone.invStd * (ob.inZone - params.inZone.mean);

  for (int i = 0; i < (int)consts::maxNumWeaponTypes; i++) {
    *out++ = params.weaponTypeObs[i].invStd *
        (ob.weaponTypeObs[i] - params.weaponTypeObs[i].mean);
  }

  return out;
}

static float * normalizeZoneOb(ZoneObservation &ob,
                               ZoneObNormalize &params,
                               float *out)
{
  *out++ = params.centerX.invStd * (ob.centerX - params.centerX.mean);
  *out++ = params.centerY.invStd * (ob.centerY - params.centerY.mean);
  *out++ = params.centerZ.invStd * (ob.centerZ - params.centerZ.mean);
  *out++ = params.toCenterDist.invStd * (ob.toCenterDist - params.toCenterDist.mean);
  *out++ = params.toCenterYaw.invStd * (ob.toCenterYaw - params.toCenterYaw.mean);
  *out++ = params.toCenterPitch.invStd * (
      ob.toCenterPitch - params.toCenterPitch.mean);
  *out++ = params.myTeamControlling.invStd * (
      ob.myTeamControlling - params.myTeamControlling.mean);
  *out++ = params.enemyTeamControlling.invStd * (
      ob.enemyTeamControlling - params.enemyTeamControlling.mean);
  *out++ = params.isContested.invStd * (
      ob.isContested - params.isContested.mean);
  *out++ = params.isCaptured.invStd * (ob.isCaptured - params.isCaptured.mean);
  *out++ = params.stepsUntilPoint.invStd * (ob.stepsUntilPoint - params.stepsUntilPoint.mean);
  *out++ = params.stepsRemaining.invStd * (ob.stepsRemaining - params.stepsRemaining.mean);

  for (int i = 0; i < (int)ob.id.size(); i++) {
    *out++ = params.id[i].invStd * (ob.id[i] - params.id[i].mean);
  }

  return out;
}

static float * normalizeOtherPlayerCommonOb(
    OtherPlayerCommonObservation &ob,
    OtherPlayerCommonObNormalize &params,
    float *out)
{
  out = normalizePlayerCommonOb(ob, params, out);

  *out++ = params.toPlayerDist.invStd * (
      ob.toPlayerDist - params.toPlayerDist.mean);
  *out++ = params.toPlayerYaw.invStd * (
      ob.toPlayerYaw - params.toPlayerYaw.mean);
  *out++ = params.toPlayerPitch.invStd * (
      ob.toPlayerPitch - params.toPlayerPitch.mean);
  *out++ = params.relativeFacingYaw.invStd * (
      ob.relativeFacingYaw - params.relativeFacingYaw.mean);
  *out++ = params.relativeFacingPitch.invStd * (
      ob.relativeFacingPitch - params.relativeFacingPitch.mean);

  return out;
}

static void normalizeSelfOb(SelfObservation &ob,
                            SelfObNormalize &params,
                            float *out)
{
  float *out_start = out;

  out = normalizePlayerCommonOb(ob, params, out);
  out = normalizeCombatOb(ob.combat, params.combat, out);
  *out++ = params.fractionRemaining.invStd * (
      ob.fractionMatchRemaining - params.fractionRemaining.mean);

  out = normalizeZoneOb(ob.zone, params.zone, out);

  assert(out - out_start == sizeof(SelfObservation) / sizeof(float));
}

static void normalizeLidarOb(LidarData *ob,
                             int width, int height,
                             LidarDataNormalize &params,
                             float *out)
{
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      LidarData in = ob[y * width + x];

      *out++ = params.depth.invStd * (in.depth - params.depth.mean);
      *out++ = params.isWall.invStd * (in.isWall - params.isWall.mean);
      *out++ =
          params.isTeammate.invStd * (in.isTeammate - params.isTeammate.mean);
      *out++ =
          params.isOpponent.invStd * (in.isOpponent - params.isOpponent.mean);
    }
  }
}

static void normalizeTeammateOb(TeammateObservation &ob,
                                TeammateObNormalize &params,
                                float *out)
{
  float *out_start = out;

  out = normalizeOtherPlayerCommonOb(ob, params, out);
  out = normalizeCombatOb(ob.combat, params.combat, out);

  assert(out - out_start == sizeof(TeammateObservation) / sizeof(float));
}

static void normalizeOpponentOb(OpponentObservation &ob,
                                OpponentObNormalize &params,
                                float *out)
{
  float *out_start = out;

  out = normalizeOtherPlayerCommonOb(ob, params, out);

  *out++ = params.wasHit.invStd * (ob.wasHit - params.wasHit.mean);
  *out++ = params.firedShot.invStd * (ob.firedShot - params.firedShot.mean);
  *out++ = params.hasLOS.invStd * (ob.hasLOS - params.hasLOS.mean);
  *out++ = params.teamKnowsLocation.invStd * (
      ob.teamKnowsLocation - params.teamKnowsLocation.mean);

  assert(out - out_start == sizeof(OpponentObservation) / sizeof(float));
}

static void normalizeRewardCoefs(RewardHyperParams &coefs,
                                 RewardHyperParamsNorm &params,
                                 float *out)
{
  float *out_start = out;

  *out++ = params.teamSpirit.invStd * (coefs.teamSpirit - params.teamSpirit.mean);
  *out++ = params.shotScale.invStd * (coefs.shotScale - params.shotScale.mean);
  *out++ = params.exploreScale.invStd * (coefs.exploreScale - params.exploreScale.mean);
  *out++ = params.inZoneScale.invStd * (coefs.inZoneScale - params.inZoneScale.mean);
  *out++ = params.zoneTeamContestScale.invStd * (coefs.zoneTeamContestScale - params.zoneTeamContestScale.mean);
  *out++ = params.zoneTeamCtrlScale.invStd * (coefs.zoneTeamCtrlScale - params.zoneTeamCtrlScale.mean);
  *out++ = params.zoneDistScale.invStd * (coefs.zoneDistScale - params.zoneDistScale.mean);
  *out++ = params.zoneEarnedPointScale.invStd * (coefs.zoneEarnedPointScale - params.zoneEarnedPointScale.mean);
  *out++ = params.breadcrumbScale.invStd * (coefs.breadcrumbScale - params.breadcrumbScale.mean);

  assert(out - out_start == sizeof(RewardHyperParams) / sizeof(float));
}

static void positionFrequencyEmbedding(NormalizedPositionObservation p,
                                          int num_freqs,
                                          float *out)
{
  for (int i = 0; i < num_freqs; i++) {
    float x_scaled = p.x * float(1 << i) * math::pi;
    float x_sin_embedding = sinf(x_scaled);
    float x_cos_embedding = cosf(x_scaled);

    float y_scaled = p.y * float(1 << i) * math::pi;
    float y_sin_embedding = sinf(y_scaled);
    float y_cos_embedding = cosf(y_scaled);

    float z_scaled = p.z * float(1 << i) * math::pi;
    float z_sin_embedding = sinf(z_scaled);
    float z_cos_embedding = cosf(z_scaled);
    
    out[(2 * i) * 3 + 0] = x_sin_embedding;
    out[(2 * i) * 3 + 1] = y_sin_embedding;
    out[(2 * i) * 3 + 2] = z_sin_embedding;

    out[(2 * i + 1) * 3 + 0] = x_cos_embedding;
    out[(2 * i + 1) * 3 + 1] = y_cos_embedding;
    out[(2 * i + 1) * 3 + 2] = z_cos_embedding;
  }
}

static void evalFullyConnectedStandalone(
    float *output,
    float *input,
    FullyConnectedParams &params)
{
  int num_inputs = params.numInputs;
  int num_features = params.numFeatures;

  for (int out_idx = 0; out_idx < num_features; out_idx++) {
    float y = 0;
    for (int in_idx = 0; in_idx < num_inputs; in_idx++) {
      float a = params.weights[num_inputs * out_idx + in_idx];
      y += a * input[in_idx];
    }

    y += params.bias[out_idx];
    output[out_idx] = y;
  }
}

static void evalFullyConnectedLayerWithLayerNormWithActivation(
    float *output,
    float *input,
    FullyConnectedLayerWithLayerNormWithActivation &layer)
{
  int num_inputs = layer.params.numInputs;
  int num_features = layer.params.numFeatures;

  float mean = 0.f;
  float m2 = 0.f;

  for (int out_idx = 0; out_idx < num_features; out_idx++) {
    float y = 0;
    for (int in_idx = 0; in_idx < num_inputs; in_idx++) {
      float x = input[in_idx];

      float a = layer.params.weights[num_inputs * out_idx + in_idx];
      y += a * x;
    }

    y += layer.params.bias[out_idx];

    output[out_idx] = y;

    float delta = y - mean;
    mean = fmaf(delta, 1.f / (out_idx + 1), mean);
    float delta2 = y - mean;
    m2 = fmaf(delta, delta2, m2);
  }

  float sigma = sqrtf(m2 / num_features);

  for (int out_idx = 0; out_idx < num_features; out_idx++) {
    float rescaled = fmaf(
        layer.layerNorm.scale[out_idx] / sigma,
        output[out_idx] - mean,
        layer.layerNorm.bias[out_idx]);

    constexpr float leaky_relu_slope = 1e-2f;

    float activation =
        rescaled >= 0.f ? rescaled : (leaky_relu_slope * rescaled);

    output[out_idx] = activation;
  }
}

static void embedInput(float *output,
                       float *input,
                       EmbedModule &module)
{
  evalFullyConnectedLayerWithLayerNormWithActivation(output, input, module.fc);
}

// Gumbel-max trick
static int32_t sampleLogits(float *logits, int32_t num_buckets, RandKey rnd)
{
  int32_t sampled = 0;
  float max_score = -INFINITY;
    
  for (int32_t i = 0; i < num_buckets; i++) {
    float u = rand::sampleUniform(rand::split_i(rnd, i));
    float g = -logf(-logf(u));

    float score = logits[i] + g;
    if (score > max_score) {
        max_score = score;
        sampled = i;
    }
  }

  return sampled;
}

void evalAgentPolicy(
  Engine &ctx,
  SelfObservation &self_ob,
  RewardHyperParams &reward_coefs,
  TeammateObservations &teammate_obs,
  OpponentObservations &opponent_obs,
  OpponentLastKnownObservations &opponent_last_known_obs,

  SelfPositionObservation &self_pos_ob,
  TeammatePositionObservations &teammate_pos_obs,
  OpponentPositionObservations &opponent_pos_obs,
  OpponentLastKnownPositionObservations &opponent_last_known_pos_obs,

  OpponentMasks &opponent_masks,
  FiltersStateObservation &filters_state_obs,
  FwdLidar &fwd_lidar,
  RearLidar &rear_lidar,
  CombatState &combat_state,
  PvPDiscreteAction &discrete_action,
  PvPDiscreteAimAction &aim_action)
{
  (void)teammate_pos_obs;
  (void)opponent_pos_obs;
  (void)opponent_last_known_pos_obs;
  (void)filters_state_obs;

  PolicyWeights *weights = ctx.data().policyWeights;

  constexpr int EMBED_SIZE = 64;
  constexpr int NUM_POSITION_EMBEDDING_FREQUENCIES = 8;

  std::array<float, 384> mlp_input_buffer;
  float *cur_mlp_input = mlp_input_buffer.data();
  {
    constexpr int SELF_INPUT_SIZE =
        sizeof(SelfObservation) / sizeof(float) +
        sizeof(RewardHyperParams) / sizeof(float) +
        3 * NUM_POSITION_EMBEDDING_FREQUENCIES * 2;
        
    std::array<float, SELF_INPUT_SIZE> self_input;
    float *cur_self_input = self_input.data();
    normalizeSelfOb(self_ob, weights->normOb.self, cur_self_input);

    cur_self_input += sizeof(SelfObservation) / sizeof(float);

    normalizeRewardCoefs(reward_coefs, weights->normOb.rewardCoefs,
                         cur_self_input);

    cur_self_input += sizeof(RewardHyperParams) / sizeof(float);

    positionFrequencyEmbedding(
        self_pos_ob.ob,
        NUM_POSITION_EMBEDDING_FREQUENCIES,
        cur_self_input);

    cur_self_input += 2 * NUM_POSITION_EMBEDDING_FREQUENCIES * 3;

    assert(cur_self_input - self_input.data() == SELF_INPUT_SIZE);

    embedInput(
        cur_mlp_input, self_input.data(), weights->selfEmbed);

    cur_mlp_input += EMBED_SIZE;
  }

  {
    std::array<float, sizeof(FwdLidar) / sizeof(float)> lidar_norm;
    normalizeLidarOb(&fwd_lidar.data[0][0],
                     consts::fwdLidarWidth, consts::fwdLidarHeight,
                     weights->normOb.fwdLidar, lidar_norm.data());

    embedInput(
        cur_mlp_input, lidar_norm.data(), weights->fwdLidarEmbed);

    cur_mlp_input += EMBED_SIZE;
  }

  {
    std::array<float, sizeof(RearLidar) / sizeof(float)> lidar_norm;
    normalizeLidarOb(&rear_lidar.data[0][0],
                     consts::rearLidarWidth, consts::rearLidarHeight,
                     weights->normOb.rearLidar, lidar_norm.data());

    embedInput(
        cur_mlp_input, lidar_norm.data(), weights->rearLidarEmbed);

    cur_mlp_input += EMBED_SIZE;
  }

  {
    for (int i = 0; i < EMBED_SIZE; i++) {
      cur_mlp_input[i] = -INFINITY;
    }

    for (int teammate_idx = 0; teammate_idx < consts::maxTeamSize - 1;
         teammate_idx++) {
      std::array<float, sizeof(TeammateObservation) / sizeof(float)>
          team_ob_norm;
      normalizeTeammateOb(
          teammate_obs.obs[teammate_idx],
          weights->normOb.teammate, team_ob_norm.data());

      std::array<float, EMBED_SIZE> embed_out;
      embedInput(embed_out.data(), team_ob_norm.data(), weights->teammatesEmbed);

      for (int i = 0; i < EMBED_SIZE; i++) {
        cur_mlp_input[i] = fmaxf(cur_mlp_input[i], embed_out[i]);
      }
    }

    cur_mlp_input += EMBED_SIZE;
  }

  {
    for (int i = 0; i < EMBED_SIZE; i++) {
      cur_mlp_input[i] = -INFINITY;
    }

    for (int opponent_idx = 0; opponent_idx < consts::maxTeamSize;
         opponent_idx++) {
      std::array<float, sizeof(OpponentObservation) / sizeof(float)>
          opponent_ob_norm;

      normalizeOpponentOb(
          opponent_obs.obs[opponent_idx],
          weights->normOb.opponent, opponent_ob_norm.data());

      std::array<float, EMBED_SIZE> embed_out;
      embedInput(embed_out.data(), opponent_ob_norm.data(),
                 weights->opponentsEmbed);

      float mask = opponent_masks.masks[opponent_idx];

      for (int i = 0; i < EMBED_SIZE; i++) {
        cur_mlp_input[i] = fmaxf(cur_mlp_input[i], mask * embed_out[i]);
      }
    }

    cur_mlp_input += EMBED_SIZE;
  }

  {
    for (int i = 0; i < EMBED_SIZE; i++) {
      cur_mlp_input[i] = -INFINITY;
    }

    for (int opponent_idx = 0; opponent_idx < consts::maxTeamSize;
         opponent_idx++) {
      std::array<float, sizeof(OpponentObservation) / sizeof(float)>
          opponent_ob_norm;

      normalizeOpponentOb(
          opponent_last_known_obs.obs[opponent_idx],
          weights->normOb.opponentLastKnown, opponent_ob_norm.data());

      std::array<float, EMBED_SIZE> embed_out;
      embedInput(embed_out.data(), opponent_ob_norm.data(),
                 weights->opponentsLastKnownEmbed);

      for (int i = 0; i < EMBED_SIZE; i++) {
        cur_mlp_input[i] = fmaxf(cur_mlp_input[i], embed_out[i]);
      }
    }

    cur_mlp_input += EMBED_SIZE;
  }

  assert(cur_mlp_input - mlp_input_buffer.data() == mlp_input_buffer.size());

  constexpr int MLP_BUFFER_SIZE = 512;
  std::array<float, MLP_BUFFER_SIZE * 2> mlp_buffer;
  {
    assert(weights->mlp[0].params.numInputs == mlp_input_buffer.size());
    assert(weights->mlp[0].params.numFeatures == MLP_BUFFER_SIZE);

    evalFullyConnectedLayerWithLayerNormWithActivation(
        mlp_buffer.data(), mlp_input_buffer.data(), weights->mlp[0]);
  }

  cur_mlp_input = mlp_buffer.data();
  float *cur_mlp_output = mlp_buffer.data() + MLP_BUFFER_SIZE;
  for (int i = 1; i < (int)weights->mlp.size(); i++) {
    assert(weights->mlp[i].params.numInputs == MLP_BUFFER_SIZE);
    assert(weights->mlp[i].params.numFeatures == MLP_BUFFER_SIZE);

    evalFullyConnectedLayerWithLayerNormWithActivation(
        cur_mlp_output, cur_mlp_input, weights->mlp[i]);
    std::swap(cur_mlp_input, cur_mlp_output);
  }
  cur_mlp_output = cur_mlp_input;

  {
    std::array<float, totalNumDiscreteActionBuckets()> logits;
    assert(MLP_BUFFER_SIZE == weights->discreteHead.numInputs);
    assert(logits.size() == weights->discreteHead.numFeatures);

    evalFullyConnectedStandalone(
        logits.data(), cur_mlp_output, weights->discreteHead);

    std::array<int32_t, sizeof(PvPDiscreteAction) / sizeof(int32_t)> int_actions;
    float *cur_logits = logits.data();
    for (int i = 0; i < (int)PolicyWeights::discreteActionsNumBuckets.size();
         i++) {
      int num_buckets = PolicyWeights::discreteActionsNumBuckets[i];

      int_actions[i] = sampleLogits(
          cur_logits, num_buckets, combat_state.rng.randKey());

      cur_logits += num_buckets;
    }

    assert(cur_logits - logits.data() == logits.size());

    discrete_action.moveAmount = int_actions[0];
    discrete_action.moveAngle = int_actions[1];
    discrete_action.fire = int_actions[2];
    discrete_action.stand = int_actions[3];
  }

  {
    std::array<float, totalNumDiscreteAimActionBuckets()> logits;
    assert(MLP_BUFFER_SIZE == weights->aimHead.numInputs);
    assert(logits.size() == weights->aimHead.numFeatures);

    evalFullyConnectedStandalone(
        logits.data(), cur_mlp_output, weights->discreteHead);

    std::array<int32_t, sizeof(PvPDiscreteAimAction) / sizeof(int32_t)> int_actions;
    float *cur_logits = logits.data();
    for (int i = 0; i < (int)PolicyWeights::discreteAimNumBuckets.size();
         i++) {
      int num_buckets = PolicyWeights::discreteAimNumBuckets[i];

      int_actions[i] = sampleLogits(
          cur_logits, num_buckets, combat_state.rng.randKey());

      cur_logits += num_buckets;
    }

    assert(cur_logits - logits.data() == logits.size());

    aim_action.yaw = int_actions[0];
    aim_action.pitch = int_actions[1];
  }

#if 0
  {
    std::array<float, 4> aim_out;
    assert(aim_out.size() == weights->aimHead.numFeatures);
    assert(MLP_BUFFER_SIZE == weights->aimHead.numInputs);

    evalFullyConnectedStandalone(
        aim_out.data(), cur_mlp_output, weights->aimHead);

    auto sampleGaussian2D =
      [&combat_state]
    (Vector2 mean, Vector2 std, float lo, float hi)
    {
      auto sigmoid = []
      (float v)
      {
        return 1.f / (1.f + expf(-v));
      };

      mean.x = tanhf(mean.x);
      mean.y = tanhf(mean.y);

      std.x = (hi - lo) * sigmoid(std.x + 2.f) + lo;
      std.y = (hi - lo) * sigmoid(std.y + 2.f) + lo;

      float u1 = combat_state.rng.sampleUniform();
      float u2 = combat_state.rng.sampleUniform();

      float z1 = sqrtf(-2.f * logf(u1)) * cosf(2.f * math::pi * u2);
      float z2 = sqrtf(-2.f * logf(u1)) * sinf(2.f * math::pi * u2);

      return Vector2 {
        .x = z1 * std.x + mean.x,
        .y = z2 * std.y + mean.y,
      };
    };

    Vector2 mean { aim_out[0], aim_out[1] };
    Vector2 std { aim_out[2], aim_out[3] };

    constexpr float MIN_STDDEV = 0.001;
    constexpr float MAX_STDDEV = 1.f;

    Vector2 sample = sampleGaussian2D(mean, std, MIN_STDDEV, MAX_STDDEV);

    aim_action.yaw = sample.x;
    aim_action.pitch = sample.y;
  }
#endif
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
  norm.yawVelocity = { *self_mu++, *self_inv_std++ };
  norm.pitchVelocity = { *self_mu++, *self_inv_std++ };
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

static FullyConnectedParams loadFullyConnectedParams(
  const char *dir, const char *path)
{
  std::string kernel_path = path;
  kernel_path += "_kernel";

  std::string bias_path = path;
  bias_path += "_bias";

  FullyConnectedParams params;

  int32_t *shape;
  int ndim = loadWeightFile(
    dir, kernel_path.c_str(), &params.weights, &shape);

  assert(ndim == 2);

  params.numInputs = shape[0];
  params.numFeatures = shape[1];

  ndim = loadWeightFile(
    dir, bias_path.c_str(), &params.bias, &shape);

  assert(ndim == 1);
  assert(shape[0] == params.numFeatures);

  // Transpose weights to be output, input row major
  float *transposed = (float *)malloc(
      sizeof(float) * params.numFeatures * params.numInputs);
  for (int row = 0; row < params.numInputs; row++) {
    for (int col = 0; col < params.numFeatures; col++) {
      transposed[col * params.numInputs + row] =
          params.weights[row * params.numFeatures + col];
    }
  }

  free(params.weights);
  params.weights = transposed;

  return params;
}

static LayerNormParams loadLayerNormParams(
  const char *dir, const char *path)
{
  std::string scale_path = path;
  scale_path += "_scale";

  std::string bias_path = path;
  bias_path += "_bias";

  LayerNormParams params;

  int32_t *shape;
  int ndim = loadWeightFile(
    dir, scale_path.c_str(), &params.scale, &shape);

  assert(ndim == 1);

  params.numFeatures = shape[0];

  ndim = loadWeightFile(
    dir, bias_path.c_str(), &params.bias, &shape);

  assert(ndim == 1);
  assert(shape[0] == params.numFeatures);

  return params;
}

static void loadNormParams(PolicyWeights *weights, const char *path)
{
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

}

PolicyWeights * loadPolicyWeights(const char *path)
{
  PolicyWeights *weights = new PolicyWeights {};
  loadNormParams(weights, path);

  {
    weights->selfEmbed.fc.params = loadFullyConnectedParams(
        path, "params_backbone_prefix_self_embed");

    weights->selfEmbed.fc.layerNorm = loadLayerNormParams(
        path, "params_backbone_prefix_LayerNorm_0_impl");
  }

  {
    weights->fwdLidarEmbed.fc.params = loadFullyConnectedParams(
        path, "params_backbone_prefix_fwd_lidar_embed");

    weights->fwdLidarEmbed.fc.layerNorm = loadLayerNormParams(
        path, "params_backbone_prefix_LayerNorm_1_impl");
  }

  {
    weights->rearLidarEmbed.fc.params = loadFullyConnectedParams(
        path, "params_backbone_prefix_rear_lidar_embed");

    weights->rearLidarEmbed.fc.layerNorm = loadLayerNormParams(
        path, "params_backbone_prefix_LayerNorm_2_impl");
  }

  {
    weights->teammatesEmbed.fc.params = loadFullyConnectedParams(
        path, "params_backbone_prefix_teammates_embed");

    weights->teammatesEmbed.fc.layerNorm = loadLayerNormParams(
        path, "params_backbone_prefix_LayerNorm_3_impl");
  }

  {
    weights->opponentsEmbed.fc.params = loadFullyConnectedParams(
        path, "params_backbone_prefix_opponents_embed");

    weights->opponentsEmbed.fc.layerNorm = loadLayerNormParams(
        path, "params_backbone_prefix_LayerNorm_4_impl");
  }

  {
    weights->opponentsLastKnownEmbed.fc.params = loadFullyConnectedParams(
        path, "params_backbone_prefix_opponents_last_known_embed");

    weights->opponentsLastKnownEmbed.fc.layerNorm = loadLayerNormParams(
        path, "params_backbone_prefix_LayerNorm_5_impl");
  }

  for (int i = 0; i < (int)weights->mlp.size(); i++) {
    std::string param_path =
        "params_backbone_actor_encoder_net_MaxPoolNet_0_MLP_0_Dense_";

    param_path += std::to_string(i);
    std::string layernorm_path =
        "params_backbone_actor_encoder_net_MaxPoolNet_0_MLP_0_LayerNorm_";
    layernorm_path += std::to_string(i) + "_impl";
        
    weights->mlp[i].params = loadFullyConnectedParams(
        path, param_path.c_str());
    weights->mlp[i].layerNorm = loadLayerNormParams(
        path, layernorm_path.c_str());
  }


  {
    weights->discreteHead = loadFullyConnectedParams(
        path, "params_actor_DenseLayerDiscreteActor_0_impl");
    weights->aimHead = loadFullyConnectedParams(
        path, "params_actor_DenseLayerDiscreteActor_1_impl");
  }

  {
    printf("selfEmbed %d %d\n",
           weights->selfEmbed.fc.params.numInputs, 
           weights->selfEmbed.fc.params.numFeatures);

    printf("fwdLidarEmbed %d %d\n",
           weights->fwdLidarEmbed.fc.params.numInputs, 
           weights->fwdLidarEmbed.fc.params.numFeatures);

    printf("rearLidarEmbed %d %d\n",
           weights->rearLidarEmbed.fc.params.numInputs, 
           weights->rearLidarEmbed.fc.params.numFeatures);

    printf("teammatesEmbed %d %d\n",
           weights->teammatesEmbed.fc.params.numInputs, 
           weights->teammatesEmbed.fc.params.numFeatures);

    printf("opponentsEmbed %d %d\n",
           weights->opponentsEmbed.fc.params.numInputs, 
           weights->opponentsEmbed.fc.params.numFeatures);

    printf("opponentsLastKnownEmbed %d %d\n",
           weights->opponentsLastKnownEmbed.fc.params.numInputs, 
           weights->opponentsLastKnownEmbed.fc.params.numFeatures);

    for (int i = 0; i < (int)weights->mlp.size(); i++) {
      printf("mlp_%d %d %d\n", i,
        weights->mlp[i].params.numInputs, 
        weights->mlp[i].params.numFeatures);
    }

    printf("discreteHead %d %d\n",
           weights->discreteHead.numInputs, 
           weights->discreteHead.numFeatures );

    printf("aimHead %d %d\n",
           weights->aimHead.numInputs, 
           weights->aimHead.numFeatures);
  }

  return weights;
}

void addPolicyEvalTasks(TaskGraphBuilder &builder)
{
  builder.addToGraph<ParallelForNode<Engine,
    evalAgentPolicy,
      SelfObservation,
      RewardHyperParams,
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
      RearLidar,
      CombatState,
      PvPDiscreteAction,
      PvPDiscreteAimAction
    >>({});
}

}
