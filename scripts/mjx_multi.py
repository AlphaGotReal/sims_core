#!/usr/bin/env python3

import argparse
import time
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp

from _mj_utils import load_model_with_cameras, CameraStreams


def main():
    parser = argparse.ArgumentParser(
        description="Run parallelized MJX simulations for RL."
    )
    parser.add_argument("xml_path", type=str)
    parser.add_argument("--batch-size", "-b", type=int, default=4096)
    parser.add_argument("--steps",      "-s", type=int, default=1000)
    parser.add_argument("--render",    action="store_true",
                        help="stream cameras for env 0 (slows things down)")
    parser.add_argument("--cams",      action="store_true")
    parser.add_argument("--cam-every", type=int, default=50)
    args = parser.parse_args()

    print(f"loading {args.xml_path}...")
    if args.cams:
        cpu_model, cams = load_model_with_cameras(args.xml_path)
    else:
        cpu_model = mujoco.MjModel.from_xml_path(args.xml_path)
        cams      = []
    cpu_data = mujoco.MjData(cpu_model)

    m = mjx.put_model(cpu_model)
    d = mjx.put_data(cpu_model, cpu_data)

    batched_d = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[None, ...], args.batch_size, axis=0), d
    )

    @jax.vmap
    def batched_step(model, data, action):
        data = data.replace(ctrl=action)
        return mjx.step(model, data)

    rng = jax.random.PRNGKey(42)

    # When rendering, we can't wrap the whole rollout in lax.scan because we
    # need to pull state back to CPU every --cam-every steps.
    if args.render and args.cams:
        streams = CameraStreams(cpu_model, cams, show=True, every=1)

        jit_step = jax.jit(batched_step)
        start = time.time()
        for i in range(args.steps):
            key = jax.random.fold_in(rng, i)
            a   = jax.random.uniform(
                key, shape=(args.batch_size, cpu_model.nu),
                minval=-1.0, maxval=1.0,
            )
            batched_d = jit_step(m, batched_d, a)

            if i % args.cam_every == 0:
                # pull env 0 state to CPU for rendering
                env0 = jax.tree_util.tree_map(lambda x: x[0], batched_d)
                mjx.get_data_into(cpu_data, cpu_model, env0)
                streams.update(cpu_data)

        batched_d.time.block_until_ready()
        streams.close()
        run_time = time.time() - start
        total    = args.batch_size * args.steps
        print(f"\n--- results ---")
        print(f"rollout time   : {run_time:.4f} s")
        print(f"total env steps: {total}")
        print(f"steps / sec    : {total / run_time:,.0f}")
        return

    # Fast path: fused rollout via lax.scan, no CPU pullbacks.
    def rollout_fn(model, initial_data, rng_key, num_steps):
        def body(carry, _):
            key = jax.random.fold_in(
                rng_key, carry.time[0].astype(jnp.int32)
            )
            actions = jax.random.uniform(
                key, shape=(args.batch_size, model.nu),
                minval=-1.0, maxval=1.0,
            )
            return batched_step(model, carry, actions), None
        final, _ = jax.lax.scan(body, initial_data, None, length=num_steps)
        return final

    jitted_rollout = jax.jit(rollout_fn, static_argnums=(3,))

    print(f"compiling (batch={args.batch_size}, steps={args.steps})...")
    t0 = time.time()
    final_d = jitted_rollout(m, batched_d, rng, args.steps)
    final_d.time.block_until_ready()
    print(f"compile + first run: {time.time() - t0:.3f} s")

    print("running profiled rollout...")
    t0 = time.time()
    final_d = jitted_rollout(m, batched_d, rng, args.steps)
    final_d.time.block_until_ready()
    run_time = time.time() - t0

    total = args.batch_size * args.steps
    print(f"\n--- results ---")
    print(f"rollout time   : {run_time:.4f} s")
    print(f"total env steps: {total}")
    print(f"steps / sec    : {total / run_time:,.0f}")


if __name__ == "__main__":
    main()
