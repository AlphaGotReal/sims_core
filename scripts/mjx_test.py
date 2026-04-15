#! /usr/bin/env python

import sys
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp

# 1. Load the standard CPU MuJoCo model and data
xml_path = sys.argv[1] 
cpu_model = mujoco.MjModel.from_xml_path(xml_path)
cpu_data = mujoco.MjData(cpu_model)

# 2. Push the model and data to the device for MJX
m = mjx.put_model(cpu_model)
d = mjx.put_data(cpu_model, cpu_data)

# 3. JIT-compile the step function
# This is crucial: without @jax.jit, the simulation will run entirely in Python 
# dispatch overhead and will be significantly slower than standard CPU MuJoCo.
@jax.jit
def step_fn(model, data):
    # Optional: Apply control inputs here
    # ctrl = jnp.array([1.0, -0.5, ...]) 
    # data = data.replace(ctrl=ctrl)
    
    return mjx.step(model, data)

# 4. Run the simulation loop
num_steps = 1000

# Execute one step to compile before the loop (optional but good for profiling)
d = step_fn(m, d) 

for _ in range(num_steps - 1):
    d = step_fn(m, d)

print(f"Simulation completed. Final time: {d.time:.3f} seconds.")
