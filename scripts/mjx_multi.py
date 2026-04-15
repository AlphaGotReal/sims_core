#!/usr/bin/env python3

import argparse
import time
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp

def main():
    parser = argparse.ArgumentParser(description="Run parallelized MJX simulations for RL.")
    parser.add_argument("xml_path", type=str, help="Path to the MuJoCo XML model file.")
    parser.add_argument("--batch-size", "-b", type=int, default=4096, help="Number of parallel environments.")
    parser.add_argument("--steps", "-s", type=int, default=1000, help="Number of simulation steps per rollout.")
    args = parser.parse_args()

    # 1. Load CPU model/data and push to MJX
    print(f"Loading {args.xml_path}...")
    cpu_model = mujoco.MjModel.from_xml_path(args.xml_path)
    cpu_data = mujoco.MjData(cpu_model)

    m = mjx.put_model(cpu_model)
    d = mjx.put_data(cpu_model, cpu_data)

    # 2. Batch the data structure
    # Replicates the MJX data across the specified batch dimension
    batched_d = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[None, ...], args.batch_size, axis=0), 
        d
    )

    # 3. Define the core batched step function
    @jax.vmap
    def batched_step(model, data, action):
        # Apply the control actions (shape: batch_size, nu)
        data = data.replace(ctrl=action)
        return mjx.step(model, data)

    # 4. Define the rollout loop using jax.lax.scan
    # This compiles the entire simulation loop into a single GPU/TPU operation.
    def rollout_fn(model, initial_data, rng_key, num_steps):
        def scan_body(carry_data, _):
            # In a real RL setup, your policy network would generate actions here.
            # For this script, we sample random continuous actions.
            key = jax.random.fold_in(rng_key, carry_data.time[0].astype(jnp.int32))
            actions = jax.random.uniform(
                key, 
                shape=(args.batch_size, model.nu), 
                minval=-1.0, 
                maxval=1.0
            )
            
            # Step the physics
            next_data = batched_step(model, carry_data, actions)
            
            # scan requires returning (carry, accumulated_output)
            # We return next_data as the carry, and None for the output since we aren't logging trajectories here.
            return next_data, None

        # Execute the scan loop
        final_data, _ = jax.lax.scan(scan_body, initial_data, None, length=num_steps)
        return final_data

    # JIT compile the entire rollout
    jitted_rollout = jax.jit(rollout_fn, static_argnums=(3,))

    # 5. Execution
    rng = jax.random.PRNGKey(42)

    print(f"Compiling and running first step (Batch: {args.batch_size}, Steps: {args.steps})...")
    start_time = time.time()
    
    # The first call triggers XLA compilation
    final_d = jitted_rollout(m, batched_d, rng, args.steps)
    
    # Block until computation finishes (JAX is asynchronous)
    final_d.time.block_until_ready() 
    
    compilation_time = time.time() - start_time
    print(f"Compilation + First Run Time: {compilation_time:.3f}s")

    # 6. Profile the actual runtime without compilation overhead
    print("Running profiled rollout...")
    start_time = time.time()
    
    final_d = jitted_rollout(m, batched_d, rng, args.steps)
    final_d.time.block_until_ready()
    
    run_time = time.time() - start_time
    total_env_steps = args.batch_size * args.steps
    sps = total_env_steps / run_time

    print(f"\n--- Results ---")
    print(f"Rollout Time:       {run_time:.4f} seconds")
    print(f"Total Env Steps:    {total_env_steps}")
    print(f"Steps Per Second:   {sps:,.0f} SPS")

if __name__ == "__main__":
    main()
