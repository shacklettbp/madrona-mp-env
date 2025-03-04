#pragma once

#include <madrona/types.hpp>

namespace madronaMPEnv {

namespace consts {
inline constexpr madrona::CountT maxTeamSize = 6;
inline constexpr madrona::CountT numTeams = 2;
inline constexpr madrona::CountT maxZones = 5;
inline constexpr madrona::CountT numStepsPerZone = 600;
inline constexpr madrona::CountT zonePointInterval = 20;
inline constexpr madrona::CountT zoneWinPoints = 125;
inline constexpr madrona::CountT poseTransitionSpeed = 10;


// Various world / entity size parameters
inline constexpr float agentRadius = 15.f;
inline constexpr float standHeight = 65.f;
inline constexpr float crouchHeight = 47.f;
inline constexpr float proneHeight = 30.f;

inline constexpr float maxRunVelocity = 400.f;
inline constexpr float maxWalkVelocity = 200.f;
inline constexpr float maxCrouchVelocity = 50.f;
inline constexpr float maxProneVelocity = 20.f;
inline constexpr float deaccelerateRate = 1000.f;

//inline constexpr float accuracyScale = 0.08f;

inline constexpr int32_t numSpawnRetries = 10;
inline constexpr int32_t respawnInvincibleSteps = 5;
inline constexpr int32_t numOutOfCombatStepsBeforeAutoheal = 150; // consider 300
 
inline constexpr float autohealAmountPerStep = 5.f;

// Each unit of distance forward (+ y axis) rewards the agents by this amount
inline constexpr float rewardPerDist = 0.05f;
// Each step that the agents don't make additional progress they get a small
// penalty reward
inline constexpr float slackReward = -0.005f;

// Steps per episode
inline constexpr int32_t episodeLen = 1800; //240;//1800; // 400;

// How many discrete options for actions
inline constexpr madrona::CountT numMoveAmountBuckets = 3;
inline constexpr madrona::CountT numMoveAngleBuckets = 8;
inline constexpr madrona::CountT numTurnBuckets = 5;
inline constexpr madrona::CountT numFacingBuckets = 16;

// Number of lidar samples, arranged in circle around agent
inline constexpr madrona::CountT fwdLidarWidth = 32;
inline constexpr madrona::CountT fwdLidarHeight = 2;

inline constexpr madrona::CountT rearLidarWidth = 8;
inline constexpr madrona::CountT rearLidarHeight = 2;

inline constexpr int32_t maxNumWeaponTypes = 3;
inline constexpr int32_t maxFireQueueSize = 10;

// Time (seconds) per step
//inline constexpr float deltaT = 0.5f;
inline constexpr float deltaT = 0.05f;

inline constexpr madrona::CountT numNonMapAssets = 14;

inline constexpr int32_t aStarPolicyID = -1;

inline constexpr int32_t discreteAimNumYawBuckets = 13;
inline constexpr int32_t discreteAimNumPitchBuckets = 7;

}

}
