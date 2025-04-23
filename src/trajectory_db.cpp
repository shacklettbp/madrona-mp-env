#include "trajectory_db.hpp"

#include <fstream>
#include <filesystem>

namespace madronaMPEnv {

struct Trajectory {
  TrajectoryType type;
  std::string tag;
  i32 listNext;
  std::vector<AgentTrajectoryStep> steps;
};

struct TrajectoryDB {
  const char *path;

  i32 freeIDHead;

  std::vector<Trajectory> trajectories;
  i64 numTrajectories;
};

TrajectoryDB * openTrajectoryDB(const char *path)
{
  // If file doesn't exist, create empty db
  if (!std::filesystem::exists(path)) {
    return new TrajectoryDB {
      .path = path,
      .freeIDHead = -1,
      .trajectories = {},
      .numTrajectories = 0,
    };
  }

  std::ifstream db_file(path, std::ios::binary);
  assert(db_file.is_open());

  i32 free_id_head;
  db_file.read((char *)&free_id_head, sizeof(i32));

  i32 trajectories_capacity;
  db_file.read((char *)&trajectories_capacity, sizeof(i32));

  i32 num_trajectories;
  db_file.read((char *)&num_trajectories, sizeof(i32));

  std::vector<Trajectory> trajectories(trajectories_capacity);
  for (i32 i = 0; i < trajectories_capacity; i++) {
    TrajectoryType type;
    db_file.read((char *)&type, sizeof(TrajectoryType));

    i32 tag_len;
    db_file.read((char *)&tag_len, sizeof(i32));

    std::string tag(tag_len, '\0');
    db_file.read(&tag[0], tag_len);

    i32 list_next;
    db_file.read((char *)&list_next, sizeof(i32));

    i32 num_steps;
    db_file.read((char *)&num_steps, sizeof(i32));

    std::vector<AgentTrajectoryStep> steps(num_steps);
    db_file.read((char *)steps.data(), sizeof(AgentTrajectoryStep) * num_steps);

    trajectories[i] = {
      .type = type,
      .tag = tag,
      .listNext = list_next,
      .steps = std::move(steps),
    };
  }

  return new TrajectoryDB {
    .path = path,
    .freeIDHead = free_id_head,
    .trajectories = std::move(trajectories),
    .numTrajectories = num_trajectories,
  };
}

void closeTrajectoryDB(TrajectoryDB *db)
{
  // Copy old db to backup
  std::string backup_path = std::string(db->path) + ".bak";
  std::rename(db->path, backup_path.c_str());

  // Write new db
  std::ofstream db_file(db->path, std::ios::binary);
  db_file.write((char *)&db->freeIDHead, sizeof(i32));

  i32 trajectories_capacity = db->trajectories.size();
  db_file.write((char *)&trajectories_capacity, sizeof(i32));

  i32 num_trajectories = db->numTrajectories;
  db_file.write((char *)&num_trajectories, sizeof(i32));

  for (const auto &trajectory : db->trajectories) {
    db_file.write((char *)&trajectory.type, sizeof(TrajectoryType));

    i32 tag_len = trajectory.tag.length();
    db_file.write((char *)&tag_len, sizeof(i32));
    db_file.write(trajectory.tag.c_str(), tag_len);

    db_file.write((char *)&trajectory.listNext, sizeof(i32));

    i32 num_steps = trajectory.steps.size();
    db_file.write((char *)&num_steps, sizeof(i32));

    db_file.write((char *)trajectory.steps.data(), sizeof(AgentTrajectoryStep) * num_steps);
  }

  delete db;
}

i64 saveTrajectory(TrajectoryDB *db, TrajectoryType type,
                   i64 id, const char *tag,
                   Span<const AgentTrajectoryStep> trajectory)
{
  if (id == -1) {
    id = db->freeIDHead;
    if (id == -1) {
      db->trajectories.push_back({});
      id = db->trajectories.size() - 1;
    } else {
      db->freeIDHead = db->trajectories[id].listNext;
    }
  }

  db->trajectories[id] = {
    .type = type,
    .tag = std::string(tag),
    .listNext = -1,
    .steps = std::vector<AgentTrajectoryStep>(trajectory.begin(), trajectory.end()),
  };

  db->numTrajectories++;
  return id;
}

void removeTrajectory(TrajectoryDB *db, i64 id)
{
  assert(id != -1);
  db->trajectories[id].type = TrajectoryType::NUM_TYPES;
  db->trajectories[id].listNext = db->freeIDHead;
  db->freeIDHead = id;
  db->numTrajectories--;
}

i64 numTrajectories(TrajectoryDB *db)
{
  return db->numTrajectories;
}

i64 advanceNTrajectories(TrajectoryDB *db, i64 cur_id, i64 n)
{
  if (cur_id == -1) {
    cur_id = 0;
    n--;
  }

  i64 id_increment = 1;
  if (n < 0) {
    id_increment = -1;
    n = -n;
  }

  i64 next_id = cur_id;
  while (n > 0 && next_id < (i64)db->trajectories.size() && next_id >= 0) {
    if (db->trajectories[next_id].type != TrajectoryType::NUM_TYPES) {
      n--;
    }
    next_id += id_increment;
  }

  if (next_id == (i64)db->trajectories.size()) {
    return -1;
  }

  return next_id;
}

Span<const AgentTrajectoryStep> getTrajectorySteps(TrajectoryDB *db, i64 id)
{
  assert(id != -1);
  return db->trajectories[id].steps;
}

TrajectoryType getTrajectoryType(TrajectoryDB *db, i64 id)
{
  assert(id != -1);
  return db->trajectories[id].type;
}

const char * getTrajectoryTag(TrajectoryDB *db, i64 id)
{
  assert(id != -1);
  return db->trajectories[id].tag.c_str();
}

void buildTrajectoryTrainingSet(TrajectoryDB *db, Span<const i64> trajectory_ids,
                                const char *output_path)
{
  (void)db;
  (void)trajectory_ids;
  (void)output_path;
}

}
