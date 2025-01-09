#include "types.hpp"

#include <fstream>

#include "db.hpp"

using namespace madronaMPEnv;

static inline i32 convexHull(XYI16 *pts, i32 n, XYI16 *hull) {
  if (n <= 1) {
    if (n == 1) {
      hull[0] = pts[0];
    }
    return n;
  }

  // Sort the points
  for (i32 i = 0; i < n; i++) {
    for (i32 j = i + 1; j < n; j++) {
      XYI16 a = pts[i];
      XYI16 b = pts[j];

      if (a.x < b.x || (a.x == b.x && a.y < b.y)) {
        continue;
      }

      std::swap(pts[i], pts[j]);
    }
  }

  auto xyint_cross = [](XYI16 o, XYI16 a, XYI16 b)
  {
    return ((i64)a.x - (i64)o.x) * ((i64)b.y - (i64)o.y) -
        ((i64)a.y - (i64)o.y) * ((i64)b.x - (i64)o.x);
  };

  // Build lower hull
  XYI16 lower[6];
  i32 lower_size = 0;
  for (i32 i = 0; i < n; i++) {
    while (lower_size >= 2 && xyint_cross(
        lower[lower_size - 2], lower[lower_size - 1], pts[i]) <= 0) {
      lower_size--;
    }
    lower[lower_size++] = pts[i];
  }

  // Build upper hull
  XYI16 upper[6];
  i32 upper_size = 0;
  for (i32 i = n; i > 0; i--) {
    while (upper_size >= 2 && xyint_cross(
        upper[upper_size - 2], upper[upper_size - 1], pts[i - 1]) <= 0) {
      upper_size--;
    }
    upper[upper_size++] = pts[i-1];
  }

  // Remove the last point of each list (it's the start point of the other list)
  upper_size--;
  lower_size--;

  // Concatenate lower and upper hull
  i32 hull_size = 0;
  for (i32 i = 0; i < lower_size; i++) hull[hull_size++] = lower[i];
  for (i32 i = 0; i < upper_size; i++) hull[hull_size++] = upper[i];

  assert(hull_size <= consts::maxTeamSize);

  return hull_size;
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    fprintf(stderr, "%s EVENTS DB", argv[0]);
    exit(1);
  }

  std::string in_dir = argv[1];

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

  std::ifstream events_file(in_dir + "/events.bin", std::ios::binary);
  assert(events_file.is_open());

  i64 num_events = fileNumElems.template operator()<GameEvent>(events_file);

  std::ifstream steps_file(in_dir + "/steps.bin", std::ios::binary);
  assert(steps_file.is_open());

  i64 num_steps = fileNumElems.template operator()<PackedStepSnapshot>(steps_file);

  printf("%ld %ld\n", (long)num_events, (long)num_steps);

  remove(argv[2]);

  sqlite3 *db = nullptr;
  REQ_SQL(db, sqlite3_open(argv[2], &db));

  {
    const char* pragmas[] = {
      "PRAGMA synchronous = OFF;",
      "PRAGMA journal_mode = MEMORY;",  // Could also try WAL or OFF
      "PRAGMA temp_store = MEMORY;",
      "PRAGMA locking_mode = EXCLUSIVE;",
      // Increase cache size (negative means # of KB, here ~100MB)
      "PRAGMA cache_size = -100000;",
    };

    for (auto p : pragmas) {
      execSQL(db, p);
    }
  }

  execSQL(db, R"(
CREATE TABLE matches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  orig_id INTEGER NOT NULL,
  num_steps INTEGER NOT NULL
);


CREATE TABLE match_steps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id INTEGER NOT NULL,
  step_idx INTEGER NOT NULL,
  cur_zone INTEGER NOT NULL,
  cur_zone_controller INTEGER NOT NULL,
  zone_steps_remaining INTEGER NOT NULL,
  zone_steps_until_point INTEGER NOT NULL,
  num_events INTEGER NOT NULL,
  event_mask INTEGER NOT NULL,
  
  UNIQUE(match_id, step_idx)
);

CREATE TABLE team_states (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  step_id INTEGER NOT NULL,
  team_idx INTEGER NOT NULL,
  centroid_x INTEGER NOT NULL,
  centroid_y INTEGER NOT NULL,
  extent_x INTEGER NOT NULL,
  extent_y INTEGER NOT NULL,
  hull_data BLOB NOT_NULL,
  
  UNIQUE(step_id, team_idx)
);

CREATE TABLE player_states (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  step_id INTEGER NOT NULL,
  player_idx INTEGER NOT NULL,
  pos_x INTEGER NOT NULL,
  pos_y INTEGER NOT NULL,
  pos_z INTEGER NOT NULL,
  yaw INTEGER NOT NULL,
  pitch INTEGER NOT NULL,
  num_bullets INTEGER NOT NULL,
  is_reloading INTEGER NOT NULL,
  fired_shot INTEGER NOT NULL,
  hp INTEGER NOT NULL,
  stand_state INTEGER NOT NULL,
  flags INTEGER NOT NULL,
  
  UNIQUE(step_id, player_idx)
);

CREATE TABLE capture_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  step_id INTEGER NOT NULL,
  zone_idx INTEGER NOT NULL,
  capture_team_idx INTEGER NOT NULL,
  in_zone_mask INTEGER NOT NULL,
  num_in_zone INTEGER NOT NULL,
  
  UNIQUE(step_id, zone_idx)
);

CREATE TABLE reload_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  step_id INTEGER NOT NULL,
  player_state_id INTEGER NOT NULL,
  num_bullets INTEGER NOT NULL,
  
  UNIQUE(step_id, player_state_id)
);

CREATE TABLE kill_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  step_id INTEGER NON NULL,
  killer_id INTEGER NOT NULL,
  killed_id INTEGER NOT NULL,
  
  UNIQUE(step_id, killer_id, killed_id)
);

CREATE TABLE player_shot_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  step_id INTEGER NON NULL,
  attacker_id INTEGER NOT NULL,
  target_id INTEGER NOT NULL,

  UNIQUE(step_id, attacker_id, target_id)
);

CREATE TABLE step_tokens (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id INTEGER NOT NULL,
  tick INTEGER NOT NULL,
  token INTEGER NOT NULL
);

CREATE UNIQUE INDEX idx_find_match_by_orig_id ON matches (orig_id);
)");

  sqlite3_stmt *insert_match_stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
INSERT INTO matches 
  (orig_id, num_steps)
VALUES
  (?, ?);
)", -1, &insert_match_stmt, nullptr));

  sqlite3_stmt *insert_match_step_stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
INSERT INTO match_steps
  (match_id, step_idx,
   cur_zone, cur_zone_controller, zone_steps_remaining, zone_steps_until_point,
   event_mask, num_events)
VALUES
  (?, ?, ?, ?, ?, ?, ?, ?);
)", -1, &insert_match_step_stmt, nullptr));

  sqlite3_stmt *insert_team_state_stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
INSERT INTO team_states
  (step_id, team_idx, centroid_x, centroid_y, extent_x, extent_y,
   hull_data)
VALUES
  (?, ?, ?, ?, ?, ?, ?);
)", -1, &insert_team_state_stmt, nullptr));

  sqlite3_stmt *insert_player_state_stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
INSERT INTO player_states (
  step_id, player_idx, pos_x, pos_y, pos_z,
  yaw, pitch, num_bullets, is_reloading, fired_shot, hp, stand_state, flags)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)", -1, &insert_player_state_stmt, nullptr));

  sqlite3_stmt *insert_capture_event_stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
INSERT INTO capture_events (
  step_id, zone_idx, capture_team_idx, in_zone_mask, num_in_zone
)
VALUES (?, ?, ?, ?, ?);
)", -1, &insert_capture_event_stmt, nullptr));

  sqlite3_stmt *insert_reload_event_stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
INSERT INTO reload_events (
  step_id, player_state_id, num_bullets
)
VALUES (?, ?, ?);
)", -1, &insert_reload_event_stmt, nullptr));

  sqlite3_stmt *insert_kill_event_stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
INSERT INTO kill_events (
  step_id, killer_id, killed_id
)
VALUES (?, ?, ?);
)", -1, &insert_kill_event_stmt, nullptr));

  sqlite3_stmt *insert_player_shot_event_stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
INSERT INTO player_shot_events (
  step_id, attacker_id, target_id
)
VALUES (?, ?, ?);
)", -1, &insert_player_shot_event_stmt, nullptr));

  sqlite3_stmt *find_player_stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
SELECT id FROM player_states WHERE step_id = ? AND player_idx = ?
)", -1, &find_player_stmt, nullptr));

  sqlite3_stmt *find_step_stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
SELECT id FROM match_steps WHERE match_id = ? AND step_idx = ?
)", -1, &find_step_stmt, nullptr));

  sqlite3_stmt *find_match_stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
SELECT id FROM matches WHERE orig_id = ?
)", -1, &find_match_stmt, nullptr));

  HeapArray<PackedStepSnapshot> steps(num_steps);
  steps_file.read((char *)steps.data(), sizeof(PackedStepSnapshot) * num_steps);

  execSQL(db, "BEGIN TRANSACTION");
  for (i64 i = 0; i < num_steps; i++) {
    PackedStepSnapshot step = steps[i];

    i64 match_id = -1;
    {
      sqlite3_bind_int64(find_match_stmt, 1, step.matchID);
      if (sqlite3_step(find_match_stmt) == SQLITE_ROW) {
        match_id = sqlite3_column_int64(find_match_stmt, 0);
      } else {
        sqlite3_bind_int64(insert_match_stmt, 1, step.matchID);
        sqlite3_bind_int(insert_match_stmt, 2, 0);
        execResetStmt(db, insert_match_stmt);
        match_id = sqlite3_last_insert_rowid(db);
      }
      REQ_SQL(db, sqlite3_reset(find_match_stmt));
    }

    sqlite3_bind_int64(insert_match_step_stmt, 1, match_id);
    sqlite3_bind_int(insert_match_step_stmt, 2, step.matchState.step);
    sqlite3_bind_int(insert_match_step_stmt, 3, step.matchState.curZone);
    sqlite3_bind_int(insert_match_step_stmt, 4, step.matchState.curZoneController);
    sqlite3_bind_int(insert_match_step_stmt, 5, step.matchState.zoneStepsRemaining);
    sqlite3_bind_int(insert_match_step_stmt, 6, step.matchState.stepsUntilPoint);
    sqlite3_bind_int(insert_match_step_stmt, 7, step.eventMask);
    sqlite3_bind_int(insert_match_step_stmt, 8, step.numEvents);

    execResetStmt(db, insert_match_step_stmt);

    int64_t step_id = sqlite3_last_insert_rowid(db);

    XYI16 convex_in[consts::maxTeamSize * 2];
    for (int player = 0; player < consts::maxTeamSize * 2; player++) {
      PackedPlayerSnapshot player_state = step.players[player];

      sqlite3_bind_int64(insert_player_state_stmt, 1, step_id);
      sqlite3_bind_int(insert_player_state_stmt, 2, player);
      sqlite3_bind_int(insert_player_state_stmt, 3, (i16)player_state.pos[0]);
      sqlite3_bind_int(insert_player_state_stmt, 4, (i16)player_state.pos[1]);
      sqlite3_bind_int(insert_player_state_stmt, 5, (i16)player_state.pos[2]);
      sqlite3_bind_int(insert_player_state_stmt, 6, player_state.yaw);
      sqlite3_bind_int(insert_player_state_stmt, 7, player_state.pitch);
      sqlite3_bind_int(insert_player_state_stmt, 8, player_state.magNumBullets);
      sqlite3_bind_int(insert_player_state_stmt, 9, player_state.isReloading);
      sqlite3_bind_int(insert_player_state_stmt, 10,
          (player_state.flags & (u8)PackedPlayerStateFlags::FiredShot) != 0);
      sqlite3_bind_int(insert_player_state_stmt, 11, player_state.hp);

      u8 stand_state = 0;
      if ((player_state.flags & (u8)PackedPlayerStateFlags::Crouch) != 0) {
        stand_state = 1;
      } else if ((player_state.flags & (u8)PackedPlayerStateFlags::Prone) != 0) {
        stand_state = 2;
      }

      sqlite3_bind_int(insert_player_state_stmt, 12, stand_state);
      sqlite3_bind_int(insert_player_state_stmt, 13, player_state.flags);

      execResetStmt(db, insert_player_state_stmt);

      convex_in[player].x = player_state.pos[0];
      convex_in[player].y = player_state.pos[1];
    }

    auto computeAndBindTeamStateFeatures =
      [&]
    (XYI16 *inputs)
    {
      Vector2 centroid = { 0, 0 };
      XYI16 min = { 32767, 32767 };
      XYI16 max = { -32768, -32768 };
      for (i32 p = 0; p < consts::maxTeamSize; p++) {
        XYI16 pos = inputs[p];

        centroid += Vector2 { (f32)pos.x, (f32)pos.y };

        if (pos.x < min.x) {
          min.x = pos.x;
        }

        if (pos.y < min.y) {
          min.y = pos.y;
        }

        if (pos.x > max.x) {
          max.x = pos.x;
        }

        if (pos.y > max.y) {
          max.y = pos.y;
        }
      }

      centroid *= (1.f / consts::maxTeamSize);

      sqlite3_bind_int(insert_team_state_stmt, 3, (i16)centroid.x);
      sqlite3_bind_int(insert_team_state_stmt, 4, (i16)centroid.y);
      sqlite3_bind_int(insert_team_state_stmt, 5, (i32)max.x - (i32)min.x);
      sqlite3_bind_int(insert_team_state_stmt, 6, (i32)max.y - (i32)min.y);
    };

    {
      TeamConvexHull hull;
      hull.numVerts = convexHull(
          convex_in, consts::maxTeamSize, hull.verts);

      sqlite3_bind_int64(insert_team_state_stmt, 1, step_id);
      sqlite3_bind_int(insert_team_state_stmt, 2, 0);

      computeAndBindTeamStateFeatures(convex_in);

      sqlite3_bind_blob(insert_team_state_stmt, 7,
                        &hull, sizeof(TeamConvexHull), SQLITE_STATIC);

      execResetStmt(db, insert_team_state_stmt);
    }

    {
      TeamConvexHull hull;
      hull.numVerts = convexHull(
          convex_in + consts::maxTeamSize, consts::maxTeamSize, hull.verts);

      sqlite3_bind_int64(insert_team_state_stmt, 1, step_id);
      sqlite3_bind_int(insert_team_state_stmt, 2, 1);

      computeAndBindTeamStateFeatures(convex_in + consts::maxTeamSize);

      sqlite3_bind_blob(insert_team_state_stmt, 7,
                        &hull, sizeof(TeamConvexHull), SQLITE_STATIC);

      execResetStmt(db, insert_team_state_stmt);
    }

  }
  execSQL(db, "COMMIT TRANSACTION");

  execSQL(db, R"(
CREATE UNIQUE INDEX idx_find_match ON match_steps (match_id, step_idx);
CREATE UNIQUE INDEX idx_find_player ON player_states (step_id, player_idx);
CREATE INDEX idx_find_player_by_pos ON player_states (pos_x, pos_y);
)");

  printf("Begin inserting events\n");

  HeapArray<GameEvent> events(num_events);
  events_file.read((char *)events.data(), sizeof(GameEvent) * num_events);

  auto lookupPlayerID =
    [&]
  (i64 step_id, i32 player_idx)
  {
    sqlite3_bind_int64(find_player_stmt, 1, step_id);
    sqlite3_bind_int(find_player_stmt, 2, player_idx);

    assert(sqlite3_step(find_player_stmt) == SQLITE_ROW);
    i64 player_id = sqlite3_column_int64(find_player_stmt, 0);
    assert(sqlite3_step(find_player_stmt) == SQLITE_DONE);

    REQ_SQL(db, sqlite3_reset(find_player_stmt));

    return player_id;
  };

  execSQL(db, "BEGIN TRANSACTION");
  for (i64 i = 0; i < num_events; i++) {
    if (i % 10000 == 0) {
      printf("E %ld\n", (long)i);
    }

    GameEvent &event = events[i];

    i64 step_id;
    {
      i64 match_id;
      {
        sqlite3_bind_int64(find_match_stmt, 1, event.matchID);
        assert(sqlite3_step(find_match_stmt) == SQLITE_ROW);
        match_id = sqlite3_column_int64(find_match_stmt, 0);
        REQ_SQL(db, sqlite3_reset(find_match_stmt));
      }

      sqlite3_bind_int64(find_step_stmt, 1, match_id);
      sqlite3_bind_int(find_step_stmt, 2, event.step);

      assert(sqlite3_step(find_step_stmt) == SQLITE_ROW);
      step_id = sqlite3_column_int64(find_step_stmt, 0);
      assert(sqlite3_step(find_step_stmt) == SQLITE_DONE);
      REQ_SQL(db, sqlite3_reset(find_step_stmt));
    }

    switch (event.type) {
    case EventType::Capture: {
      i32 zone_idx = event.capture.zoneIDX;
      i32 capture_team = event.capture.captureTeam;
      u16 in_zone_mask = event.capture.inZoneMask;

      int num_in_zone = std::popcount(in_zone_mask);

      sqlite3_bind_int64(insert_capture_event_stmt, 1, step_id);
      sqlite3_bind_int(insert_capture_event_stmt, 2, zone_idx);
      sqlite3_bind_int(insert_capture_event_stmt, 3, capture_team);
      sqlite3_bind_int(insert_capture_event_stmt, 4, in_zone_mask);
      sqlite3_bind_int(insert_capture_event_stmt, 5, num_in_zone);

      execResetStmt(db, insert_capture_event_stmt);
    } break;
    case EventType::Reload: {
      i64 reloader = lookupPlayerID(step_id, event.reload.player);
      sqlite3_bind_int64(insert_reload_event_stmt, 1, step_id);
      sqlite3_bind_int64(insert_reload_event_stmt, 2, reloader);
      sqlite3_bind_int(insert_reload_event_stmt, 3,
                       event.reload.numBulletsAtReloadTime);

      execResetStmt(db, insert_reload_event_stmt);
    } break;
    case EventType::Kill: {
      i64 killer_id = lookupPlayerID(step_id, event.kill.killer);
      i64 killed_id = lookupPlayerID(step_id, event.kill.killed);

      sqlite3_bind_int64(insert_kill_event_stmt, 1, step_id);
      sqlite3_bind_int64(insert_kill_event_stmt, 2, killer_id);
      sqlite3_bind_int64(insert_kill_event_stmt, 3, killed_id);
      execResetStmt(db, insert_kill_event_stmt);
    } break;
    case EventType::PlayerShot: {
      i16 attacker_id = lookupPlayerID(step_id, event.playerShot.attacker);
      i16 target_id = lookupPlayerID(step_id, event.playerShot.target);

      sqlite3_bind_int64(insert_player_shot_event_stmt, 1, step_id);
      sqlite3_bind_int64(insert_player_shot_event_stmt, 2, attacker_id);
      sqlite3_bind_int64(insert_player_shot_event_stmt, 3, target_id);
      execResetStmt(db, insert_player_shot_event_stmt);
    } break;
    default: {
      FATAL("Unknown event type");
    } break;
    }
  }
  execSQL(db, "COMMIT TRANSACTION");

  execSQL(db, R"(
CREATE INDEX idx_find_captures ON capture_events (
  num_in_zone, zone_idx, capture_team_idx);
)");

  REQ_SQL(db, sqlite3_finalize(find_match_stmt));
  REQ_SQL(db, sqlite3_finalize(find_step_stmt));
  REQ_SQL(db, sqlite3_finalize(find_player_stmt));

  REQ_SQL(db, sqlite3_finalize(insert_match_stmt));
  REQ_SQL(db, sqlite3_finalize(insert_team_state_stmt));
  REQ_SQL(db, sqlite3_finalize(insert_player_state_stmt));
  REQ_SQL(db, sqlite3_finalize(insert_match_step_stmt));
  REQ_SQL(db, sqlite3_finalize(insert_capture_event_stmt));
  REQ_SQL(db, sqlite3_finalize(insert_player_shot_event_stmt));

  REQ_SQL(db, sqlite3_finalize(insert_reload_event_stmt));
  REQ_SQL(db, sqlite3_finalize(insert_kill_event_stmt));
  REQ_SQL(db, sqlite3_close(db));
}
