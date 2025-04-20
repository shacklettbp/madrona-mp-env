#pragma once

#include "types.hpp"

namespace madronaMPEnv {

enum class TrajectoryType : u32 {
  Human,
  RL,
  Hardcoded,
  NUM_TYPES,
};

struct AgentTrajectoryStep {
  Vector3 pos;
  float yaw;
  float pitch;

  CombatState combatState;

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

i64 saveTrajectory(TrajectoryDB *db, TrajectoryType type, i64 id,
                   const char *tag,
                   Span<const AgentTrajectoryStep> trajectory);

void removeTrajectory(TrajectoryDB *db, i64 id);

i64 numTrajectories(TrajectoryDB *db);

i64 advanceNTrajectories(TrajectoryDB *db, i64 cur_id, i64 n = 1);

Span<const AgentTrajectoryStep> getTrajectorySteps(TrajectoryDB *db, i64 id);
TrajectoryType getTrajectoryType(TrajectoryDB *db, i64 id);
const char * getTrajectoryTag(TrajectoryDB *db, i64 id);

}
