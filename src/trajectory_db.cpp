#include "trajectory_db.hpp"

namespace madronaMPEnv {

struct TrajectoryRef {
  i32 chunk;
  i32 offset;
};

struct Trajectory {
  TrajectoryType type;
  std::string tag;
  i32 numSteps;
};

struct TrajectoryChunk {
  std::array<Trajectory, 1024> trajectories;
  std::array<AgentTrajectoryStep, 1024> steps;
};

struct TrajectoryDB {
  const char *path;

  std::vector<Trajectory> trajectories;

  std::vector<TrajectoryRef> byID;
  i32 freeIDHead;
};

TrajectoryDB * openTrajectoryDB(const char *path)
{

  return new TrajectoryDB {
    .path = path,
    .freeIDHead = -1,
  };
}

void closeTrajectoryDB(TrajectoryDB *db)
{
}

i64 saveTrajectory(TrajectoryDB *db, TrajectoryType type,
                   i64 id, const char *tag,
                   Span<const AgentTrajectoryStep> trajectory)
{
}

void removeTrajectory(TrajectoryDB *db, i64 id)
{
  assert(id != -1);
}

}
