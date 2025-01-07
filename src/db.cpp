#include "db.hpp"

namespace madronaMPEnv {

void execSQL(sqlite3 *db, const char *sql)
{
  char * err_msg;
  int res = sqlite3_exec(db, sql, nullptr, nullptr, &err_msg);

  if (res == SQLITE_OK) [[likely]] {
    return;
  }

  FATAL("SQL error executing '%s': %s", sql, err_msg);
}

void execResetStmt(sqlite3 *db, sqlite3_stmt *stmt)
{
  if (sqlite3_step(stmt) != SQLITE_DONE) {
    FATAL("Failed to execute statement: %s", sqlite3_errmsg(db));
  }

  REQ_SQL(db, sqlite3_reset(stmt));
}

sqlite3_stmt * initLoadStepSnapshotStatement(sqlite3 *db)
{
  sqlite3_stmt *stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
SELECT
  ps.pos_x, ps.pos_y, ps.pos_z, ps.yaw, ps.pitch,
  ps.num_bullets, ps.is_reloading, ps.fired_shot,
  ps.hp, ps.stand_state
FROM player_states AS ps
WHERE
  ps.step_id = ?
)", -1, &stmt, nullptr));

  return stmt;
}

sqlite3_stmt * initLoadMatchZoneStatement(sqlite3 *db)
{
  sqlite3_stmt *stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
SELECT ms.cur_zone, ms.cur_zone_controller
FROM match_steps AS ms
WHERE
  ms.id = ?
)", -1, &stmt, nullptr));

  return stmt;
}

StepSnapshot loadStepSnapshot(sqlite3 *db_hdl, 
                              sqlite3_stmt *step_stmt,
                              sqlite3_stmt *players_stmt,
                              i64 step_id)
{
  StepSnapshot snapshot;

  sqlite3_bind_int64(players_stmt, 1, step_id);

  i64 cur_player_idx = 0;
  while (sqlite3_step(players_stmt) == SQLITE_ROW) {
    i64 pos_x = sqlite3_column_int(players_stmt, 0);
    i64 pos_y = sqlite3_column_int(players_stmt, 1);
    i64 pos_z = sqlite3_column_int(players_stmt, 2);
    i64 yaw = sqlite3_column_int(players_stmt, 3);
    i64 pitch = sqlite3_column_int(players_stmt, 4);

    i64 num_bullets = sqlite3_column_int(players_stmt, 5);
    i64 is_reloading = sqlite3_column_int(players_stmt, 6);
    i64 fired_shot = sqlite3_column_int(players_stmt, 7);

    i64 hp = sqlite3_column_int(players_stmt, 8);
    i64 stand_state = sqlite3_column_int(players_stmt, 9);

    assert(cur_player_idx < consts::maxTeamSize * 2);
    snapshot.players[cur_player_idx++] = {
      .pos = { (float)pos_x, (float)pos_y, (float)pos_z },
      .yaw = (float)yaw * math::pi / 32768.f,
      .pitch = (float)pitch * math::pi / 32768.f,
      .mag = {
        .numBullets = (i32)num_bullets,
        .isReloading = (i32)is_reloading,
      },
      .firedShot = fired_shot > 0,
      .hp = { (float)hp },
      .standPose = stand_state == 0 ? Pose::Stand : (
          stand_state == 1 ? Pose::Crouch : Pose::Prone),
    };
  }

  assert(cur_player_idx == consts::maxTeamSize * 2);

  REQ_SQL(db_hdl, sqlite3_reset(players_stmt));

  sqlite3_bind_int64(step_stmt, 1, step_id);

  assert(sqlite3_step(step_stmt) == SQLITE_ROW);

  snapshot.curZone = sqlite3_column_int(step_stmt, 0);
  snapshot.curZoneController = sqlite3_column_int(step_stmt, 1);

  REQ_SQL(db_hdl, sqlite3_reset(step_stmt));

  return snapshot;
}

}
