from itertools import product
import sys
import numpy as jnp

num_worlds = int(sys.argv[1])
num_policies = int(sys.argv[2])

num_world_digits = len(str(num_worlds))

assignments = list(product(range(num_policies), repeat=2))
num_repeats = num_worlds // len(assignments)

cur_world_idx = 0
for assignment in assignments:
    for i in range(num_repeats):
        print(f"{cur_world_idx:{num_world_digits}}: {assignment}")
        cur_world_idx += 1
