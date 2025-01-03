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
        SELECT pos_x, pos_y, pos_z, yaw 
        FROM player_states
        WHERE player_states.step_id = {step_id}
""")

    rows = res.fetchall()

    print(rows)
    print(len(rows))

for trajectory_idx in range(trajectories.shape[0]):
    for trajectory_offset in range(trajectories.shape[1]):
        step_id = trajectories[trajectory_idx, trajectory_offset]
        players = load_players_for_step(step_id)
        print(players)
