import sys
import sqlite3

con = sqlite3.connect(sys.argv[1])
cur = con.cursor()

res = cur.execute(
    "SELECT DISTINCT match_id FROM step_tokens")
matches = res.fetchall()

ngram_length = 5

ngrams = []

for match_id, in matches:
    res = cur.execute(
        f"SELECT token FROM step_tokens WHERE match_id = {match_id} ORDER BY tick;")

    rows = res.fetchall()
    num_rows = len(rows)

    for start in range(0, num_rows - ngram_length):
        ngram = []
        for i in range(0, ngram_length):
            ngram.append(rows[start + i])

        ngrams.append(tuple(ngram))

counts = {}

for ngram in ngrams:
    counts[ngram] = counts.get(ngram, 0) + 1

sorted_items = sorted(counts.items(), key=lambda item: abs(item[1]))

top = list(reversed(sorted_items))[0:30]

print(len(ngrams))

for ngram, count in top:
    print(f"{ngram}: {count}")
