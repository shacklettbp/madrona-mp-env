import jax
from jax import numpy as jnp
import orbax.checkpoint
import numpy as np
import os
import sys
from functools import partial

# Path to your checkpoint directory/file
checkpoint_path = os.path.realpath(sys.argv[1])
out_dir = sys.argv[2]

# Create a checkpointer (if you don't have a custom one, you can pass None for the spec)
checkpointer = orbax.checkpoint.PyTreeCheckpointer()

# Restore the state (this will be a nested dict of weights)
state = checkpointer.restore(checkpoint_path)

# Function to recursively save arrays from a nested dictionary.
def save_arrays(d, prefix=''):
    for key, value in d.items():
        new_prefix = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            save_arrays(value, prefix=new_prefix)
        elif value is None:
            continue
        else:
            value = np.array(value)
            shape = value.shape

            if len(shape) < 2:
                continue

            for i in range(shape[0]):
                file_dir = os.path.join(out_dir, str(i))
                os.makedirs(file_dir, exist_ok=True)
                filename = os.path.join(file_dir, new_prefix)
                if i == 0:
                    print(f"Saving {filename}...")
                    print(value[i].shape, value.dtype)
                with open(filename, 'w') as f:
                    np.array([len(shape) - 1], dtype=np.int32).tofile(f)
                    np.array([*shape[1:]], dtype=np.int32).tofile(f)
                    value[i].tofile(f)

policy_states = state['policy_states']

save_arrays(policy_states)
