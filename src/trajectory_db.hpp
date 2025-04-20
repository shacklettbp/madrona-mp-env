#pragma once

#include "types.hpp"

namespace madronaMPEnv {

enum class TrajectoryType : u32 {
  Human,
  RL,
  Hardcoded,
};

struct AgentTrajectoryStep {
  Vector3 pos;
  float yaw;
  float pitch;

  PvPDiscreteAction discreteAction;
  PvPAimAction continuousAimAction;

  SelfObservation selfObs;
  TeammateObservation teammateObs;
  OpponentObservation opponentObs;
  OpponentLastKnownObservations opponentLastKnownObs;
  FwdLidar fwdLidarObs;
  RearLidar rearLidarObs;
};

struct TrajectoryDB;

TrajectoryDB * openTrajectoryDB(const char *path);
void closeTrajectoryDB(TrajectoryDB *db);

bool saveTrajectory(TrajectoryDB *db, TrajectoryType type, i64 id,
                    const char *tag,
                    Span<const AgentTrajectoryStep> trajectory);

}
