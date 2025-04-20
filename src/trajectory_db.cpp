#include "trajectory_db.hpp"

namespace madronaMPEnv {

struct TrajectoryDB {
};

TrajectoryDB * openTrajectoryDB(const char *path)
{
}

void closeTrajectoryDB(TrajectoryDB *db)
{
}

bool saveTrajectory(TrajectoryDB *db, TrajectoryType type, i64 id,
                    const char *tag,
                    Span<const AgentTrajectoryStep> trajectory)
{
}

}
