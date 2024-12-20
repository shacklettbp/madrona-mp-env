import sys
import sqlite3

con = sqlite3.connect(sys.argv[1])
cur = con.cursor()

res = cur.execute("SELECT match_id,step_idx,event_mask FROM match_steps ORDER BY match_id,step_idx;")

rows = res.fetchall()

res = cur.execute("DELETE FROM step_tokens;")

cur_match_id = rows[0][0]
cur_step_idx = 0
cur_mask = 0

freq = 40

for match_id, step_idx, event_mask in rows:
    if step_idx - cur_step_idx >= freq or match_id != cur_match_id:
        print(cur_match_id, cur_step_idx, cur_mask)
        cur.execute(f"INSERT INTO step_tokens (match_id, tick, token) VALUES ({cur_match_id}, {cur_step_idx}, {cur_mask});")

        cur_match_id = match_id
        cur_step_idx = step_idx
        cur_mask = 0

    cur_mask |= event_mask

con.commit()
