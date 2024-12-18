#include "types.hpp"

#include <fstream>
#include <unordered_map>

#include "db.hpp"

using namespace madronaMPEnv;

struct StepKey {
  u64 matchID;
  u32 step;
};

inline bool operator==(StepKey a, StepKey b)
{
  return a.matchID == b.matchID && a.step == b.step;
}

namespace std {

template <>
struct hash<StepKey> {
  size_t operator()(StepKey v) const
  {
    uint64_t a = v.matchID;
    uint64_t b = v.step;
    a ^= b + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2);
    return a;
  }
};

}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    fprintf(stderr, "%s EVENTS DB", argv[0]);
    exit(1);
  }

  std::string in_dir = argv[1];

  std::ifstream events_file(in_dir + "/events.bin", std::ios::binary);
  assert(events_file.is_open());

  std::ifstream steps_file(in_dir + "/steps.bin", std::ios::binary);
  assert(steps_file.is_open());

  remove(argv[2]);

  sqlite3 *db = nullptr;
  REQ_SQL(db, sqlite3_open(argv[2], &db));

  execSQL(db, R"(
CREATE TABLE match_steps (
id INTEGER PRIMARY KEY AUTOINCREMENT,
match_id INTEGER,
step_idx INTEGER
);

CREATE TABLE player_states (
id INTEGER PRIMARY KEY AUTOINCREMENT,
step_id INTEGER,
player_idx INTEGER,
pos_x INTEGER,
pos_y INTEGER,
pos_z INTEGER,
yaw INTEGER,
num_bullets INTEGER,
is_reloading INTEGER
);

CREATE TABLE capture_events (
id INTEGER PRIMARY KEY AUTOINCREMENT,
step_id INTEGER,
zone_idx INTEGER,
capture_team_idx INTEGER,
num_in_zone INTEGER
);
)");

  sqlite3_stmt *insert_match_step_stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
INSERT INTO match_steps (match_id, step_idx) VALUES (?, ?);
)", -1, &insert_match_step_stmt, nullptr));

  sqlite3_stmt *insert_player_state_stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
INSERT INTO player_states (
  step_id, player_idx, pos_x, pos_y, pos_z,
  yaw, num_bullets, is_reloading)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);
)", -1, &insert_player_state_stmt, nullptr));

  sqlite3_stmt *insert_capture_event_stmt;
  REQ_SQL(db, sqlite3_prepare_v2(db, R"(
INSERT INTO capture_events (
  step_id, zone_idx, capture_team_idx, num_in_zone
)
VALUES (?, ?, ?, ?);
)", -1, &insert_capture_event_stmt, nullptr));

  while (events_file.peek() != EOF && steps_file.peek() != EOF) {
    u32 num_steps;
    steps_file.read((char *)&num_steps, sizeof(u32));
    HeapArray<EventStepState> steps(num_steps);
    steps_file.read((char *)steps.data(), sizeof(EventStepState) * num_steps);

    std::unordered_map<StepKey, i64> step_id_lookup;

    execSQL(db, "BEGIN TRANSACTION");
    for (u32 i = 0; i < num_steps; i++) {
      EventStepState step = steps[i];
      sqlite3_bind_int(insert_match_step_stmt, 1, step.matchID);
      sqlite3_bind_int(insert_match_step_stmt, 2, step.step);

      execResetStmt(db, insert_match_step_stmt);

      int64_t step_id = sqlite3_last_insert_rowid(db);

      step_id_lookup.emplace(StepKey {
        .matchID = step.matchID,
        .step = step.step,
      }, step_id);

      for (int player = 0; player < consts::maxTeamSize * 2; player++) {
        EventPlayerState player_state = step.players[player];

        sqlite3_bind_int(insert_player_state_stmt, 1, step_id);
        sqlite3_bind_int(insert_player_state_stmt, 2, player_state.playerID);
        sqlite3_bind_int(insert_player_state_stmt, 3, player_state.pos[0]);
        sqlite3_bind_int(insert_player_state_stmt, 4, player_state.pos[1]);
        sqlite3_bind_int(insert_player_state_stmt, 5, player_state.pos[2]);
        sqlite3_bind_int(insert_player_state_stmt, 6, player_state.yaw);
        sqlite3_bind_int(insert_player_state_stmt, 7, player_state.magNumBullets);
        sqlite3_bind_int(insert_player_state_stmt, 8, player_state.isReloading);

        execResetStmt(db, insert_player_state_stmt);
      }
    }
    execSQL(db, "COMMIT TRANSACTION");

    u32 num_events;
    events_file.read((char *)&num_events, sizeof(u32));
    HeapArray<GameEvent> events(num_events);
    events_file.read((char *)events.data(), sizeof(GameEvent) * num_events);

    execSQL(db, "BEGIN TRANSACTION");
    for (u32 i = 0; i < num_events; i++) {
      GameEvent &event = events[i];

      i64 step_id;
      {
        auto iter = step_id_lookup.find(StepKey {
          .matchID = event.matchID,
          .step = event.step,
        });

        assert(iter != step_id_lookup.end());

        step_id = iter->second;
      }

      switch (event.type) {
      case EventType::Capture: {
        i32 zone_idx = event.capture.zoneIDX;
        i32 capture_team = event.capture.captureTeam;
        u16 in_zone_mask = event.capture.inZoneMask;

        int num_in_zone = std::popcount(in_zone_mask);

        sqlite3_bind_int(insert_capture_event_stmt, 1, step_id);
        sqlite3_bind_int(insert_capture_event_stmt, 2, zone_idx);
        sqlite3_bind_int(insert_capture_event_stmt, 3, capture_team);
        sqlite3_bind_int(insert_capture_event_stmt, 4, num_in_zone);

        execResetStmt(db, insert_capture_event_stmt);
      } break;
      default: {
        FATAL("Unknown event type");
      } break;
      }
    }
    execSQL(db, "COMMIT TRANSACTION");
  }

  REQ_SQL(db, sqlite3_finalize(insert_player_state_stmt));
  REQ_SQL(db, sqlite3_finalize(insert_match_step_stmt));
  REQ_SQL(db, sqlite3_finalize(insert_capture_event_stmt));
  REQ_SQL(db, sqlite3_close(db));
}
