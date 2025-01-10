#!/usr/bin/env python

import sys
import sqlite3
import numpy as np

sql_db_path = sys.argv[1]
trajectory_db_path = sys.argv[2]

con = sqlite3.connect(sys.argv[1])
db = con.cursor()

with open(trajectory_db_path, 'rb') as f:
    trajectories = np.fromfile(f, dtype=np.int64)

trajectories = trajectories.reshape(-1, 10)

print(trajectories)
print(trajectories.shape)

def load_players_for_step(step_id):
    res = db.execute(f"""
        SELECT
            ps.pos_x, ps.pos_y, ps.pos_z, ps.yaw, ps.pitch,
            ps.num_bullets, ps.is_reloading, ps.fired_shot,
            ps.hp, ps.stand_state
        FROM player_states AS ps
        WHERE ps.step_id = {step_id}
""")

    rows = res.fetchall()

    return rows

def load_match_state_for_step(step_id):
    res = db.execute(f"""
      SELECT ms.step_idx, ms.cur_zone, ms.cur_zone_controller,
             ms.zone_steps_remaining, ms.zone_steps_until_point
      FROM match_steps AS ms
      WHERE ms.id = {step_id}
""")

    return res.fetchone()


for trajectory_idx in range(trajectories.shape[0]):
    for trajectory_offset in range(trajectories.shape[1]):
        step_id = trajectories[trajectory_idx, trajectory_offset]
        players = load_players_for_step(step_id)
        match_state = load_match_state_for_step(step_id)
        print(players)
        print(match_state)
