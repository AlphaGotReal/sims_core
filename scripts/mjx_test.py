#! /usr/bin/env python

import argparse
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp

from _mj_utils import load_model_with_cameras, CameraStreams


def main(args):

    if args.cams:
        cpu_model, cams = load_model_with_cameras(args.xml)
    else:
        cpu_model = mujoco.MjModel.from_xml_path(args.xml)
        cams      = []
    cpu_data = mujoco.MjData(cpu_model)

    m = mjx.put_model(cpu_model)
    d = mjx.put_data(cpu_model, cpu_data)

    @jax.jit
    def step_fn(model, data):
        return mjx.step(model, data)

    # Rendering on MJX requires pulling state back to CPU MjData. Doing so
    # every step defeats the purpose; we do it every --cam-every steps.
    streams = (CameraStreams(cpu_model, cams, show=args.render,
                             every=1)
               if args.cams else None)

    d = step_fn(m, d)
    for i in range(args.steps - 1):
        d = step_fn(m, d)

        if streams is not None and (i % args.cam_every == 0):
            mjx.get_data_into(cpu_data, cpu_model, d)
            streams.update(cpu_data)

    if streams is not None:
        streams.close()

    print(f"simulation completed. final time: {float(d.time):.3f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xml")
    parser.add_argument("--render",    action="store_true",
                        help="stream camera feeds (MJX has no native GUI)")
    parser.add_argument("--cams",      action="store_true")
    parser.add_argument("--steps",     type=int, default=1000)
    parser.add_argument("--cam-every", type=int, default=5)
    args = parser.parse_args()
    main(args)
