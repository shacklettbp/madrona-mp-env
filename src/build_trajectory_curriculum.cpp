#include "types.hpp"

#include <fstream>

#include "db.hpp"

using namespace madronaMPEnv;

static CurriculumSnapshot loadCurriculumSnapshot(sqlite3 *db,
                                                 sqlite3_stmt *step_stmt,
                                                 sqlite3_stmt *player_stmt,
                                                 i64 step_id)
{
  CurriculumSnapshot snapshot;

  sqlite3_bind_int64(player_stmt, 1, step_id);

  i64 cur_player_idx = 0;
  while (sqlite3_step(player_stmt) == SQLITE_ROW) {
    i64 pos_x = sqlite3_column_int(player_stmt, 0);
    i64 pos_y = sqlite3_column_int(player_stmt, 1);
    i64 pos_z = sqlite3_column_int(player_stmt, 2);
    i64 yaw = sqlite3_column_int(player_stmt, 3);
    i64 pitch = sqlite3_column_int(player_stmt, 4);

    i64 num_bullets = sqlite3_column_int(player_stmt, 5);
    i64 is_reloading = sqlite3_column_int(player_stmt, 6);
    i64 fired_shot = sqlite3_column_int(player_stmt, 7);

    i64 hp = sqlite3_column_int(player_stmt, 8);
    i64 stand_state = sqlite3_column_int(player_stmt, 9);

    u8 flags = 0;

    if (fired_shot != 0) {
      flags |= (u8)PackedPlayerStateFlags::FiredShot;
    }

    if (stand_state == 1) {
      flags |= (u8)PackedPlayerStateFlags::Crouch;
    } else if (stand_state == 2) {
      flags |= (u8)PackedPlayerStateFlags::Prone;
    }

    assert(cur_player_idx < consts::maxTeamSize * 2);
    snapshot.players[cur_player_idx++] = {
      .pos = { (i16)pos_x, (i16)pos_y, (i16)pos_z },
      .yaw = (i16)yaw,
      .pitch = (i16)pitch,
      .magNumBullets = (u8)num_bullets,
      .isReloading = (u8)is_reloading,
      .hp = (u8)hp,
      .flags = flags,
    };
  }

  assert(cur_player_idx == consts::maxTeamSize * 2);

  REQ_SQL(db, sqlite3_reset(player_stmt));

  sqlite3_bind_int64(step_stmt, 1, step_id);
  assert(sqlite3_step(step_stmt) == SQLITE_ROW);
  int step_idx = sqlite3_column_int(step_stmt, 0);
  int cur_zone = sqlite3_column_int(step_stmt, 1);
  int cur_zone_controller = sqlite3_column_int(step_stmt, 2);
  int zone_steps_remaining = sqlite3_column_int(step_stmt, 3);
  int zone_steps_until_point = sqlite3_column_int(step_stmt, 4);

  REQ_SQL(db, sqlite3_reset(step_stmt));

  snapshot.matchState = {
    .step = (u16)step_idx,
    .curZone = (u8)cur_zone,
    .curZoneController = (i8)cur_zone_controller,
    .zoneStepsRemaining = (u16)zone_steps_remaining,
    .stepsUntilPoint = (u16)zone_steps_until_point,
  };

  return snapshot;
}

int main(int argc, char *argv[])
{
  if (argc != 4) {
    fprintf(stderr, "%s SQL_DB TRAJECTORIES CURRICULUM_OUT", argv[0]);
    exit(1);
  }

  sqlite3 *db = nullptr;
  REQ_SQL(db, sqlite3_open(argv[1], &db));

  std::ifstream trajectories_file(argv[2], std::ios::binary);
  assert(trajectories_file.is_open());

  std::ofstream curriculum_file(argv[3], std::ios::binary);
  assert(curriculum_file.is_open());

  auto fileNumElems =
    []<typename T>
  (std::ifstream &f)
  {
    f.seekg(0, f.end);
    i64 size = f.tellg();
    f.seekg(0, f.beg);

    assert(size % sizeof(T) == 0);

    return size / sizeof(T);
  };

  const i64 trajectory_len = 100;
  const i64 subsample = 20;

  i64 num_trajectories;
  i64 *trajectory_steps = nullptr;
  {
    i64 num_steps = fileNumElems.template operator()<i64>(trajectories_file);

    assert(num_steps % trajectory_len == 0);

    num_trajectories = num_steps / trajectory_len;

    trajectory_steps = (i64 *)malloc(sizeof(i64) * num_steps);
    trajectories_file.read(
        (char *)trajectory_steps, sizeof(i64) * num_steps);
  }

  sqlite3_stmt *load_step_players_stmt = initLoadStepSnapshotStatement(db);

  sqlite3_stmt *load_step_match_data_stmt;
  {
    const char *sql = R"(
      SELECT ms.step_idx, ms.cur_zone, ms.cur_zone_controller,
             ms.zone_steps_remaining, ms.zone_steps_until_point
      FROM match_steps AS ms
      WHERE ms.id = ?
    )";
    REQ_SQL(db, sqlite3_prepare_v2(db, sql, -1, &load_step_match_data_stmt,
                                   nullptr));
  }

  i64 num_dumped = 0;

  for (i64 trajectory_idx = 0; trajectory_idx < num_trajectories;
       trajectory_idx++) {
    i64 *trajectory_start = trajectory_steps + trajectory_idx * trajectory_len;

    for (i64 trajectory_offset = 0; trajectory_offset < trajectory_len;
         trajectory_offset += subsample) {
      i64 step_id = trajectory_start[trajectory_offset];

      CurriculumSnapshot snapshot = loadCurriculumSnapshot(
          db, load_step_match_data_stmt, load_step_players_stmt, step_id);

      curriculum_file.write((char *)&snapshot, sizeof(CurriculumSnapshot));

      num_dumped += 1;
    }
  }

  printf("%ld\n", (long)num_dumped);

  REQ_SQL(db, sqlite3_finalize(load_step_match_data_stmt));
  REQ_SQL(db, sqlite3_finalize(load_step_players_stmt));
  REQ_SQL(db, sqlite3_close(db));
}
