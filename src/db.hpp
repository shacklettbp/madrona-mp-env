#pragma once

#include <sqlite3.h>

#include <madrona/crash.hpp>

#include "types.hpp"

namespace madronaMPEnv {

struct PlayerSnapshot {
  Vector3 pos;
  float yaw;
  float pitch;
  Magazine mag;
  bool firedShot;
  HP hp;
  Pose standPose;
};

struct StepSnapshot {
  PlayerSnapshot players[consts::maxTeamSize * 2];
  int curZone;
  int curZoneController;
};

sqlite3_stmt * initLoadStepSnapshotStatement(sqlite3 *db);
sqlite3_stmt * initLoadMatchZoneStatement(sqlite3 *db);

StepSnapshot loadStepSnapshot(sqlite3 *db_hdl,
                              sqlite3_stmt *step_stmt,
                              sqlite3_stmt *players_stmt,
                              i64 step_id);

inline void checkSQL(sqlite3 *db, int res, const char *file, int line,
                     const char *funcname)
{
  if (res == SQLITE_OK) [[likely]] {
    return;
  }

  FATAL("DB Error: '%s' @ %s:%d in %s",
        sqlite3_errmsg(db), file, line, funcname);
}

#define REQ_SQL(db, r) ::madronaMPEnv::checkSQL(db, (r), __FILE__, __LINE__,\
                                            MADRONA_COMPILER_FUNCTION_NAME)

void execSQL(sqlite3 *db, const char *sql);

void execResetStmt(sqlite3 *db, sqlite3_stmt *stmt);

}
