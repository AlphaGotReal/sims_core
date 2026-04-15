#! /usr/bin/env python

import os
import time
import argparse

import mujoco

from _mj_utils import load_model_with_cameras, CameraStreams


def main(args):
    xml_path = os.path.abspath(args.xml)
    os.chdir(os.path.dirname(xml_path))

    model, cams = load_model_with_cameras(xml_path)
    data        = mujoco.MjData(model)

    streams = CameraStreams(model, cams, show=args.render,
                            every=args.cam_every) if args.render else None

    start = time.time()
    while time.time() - start < args.duration:
        t0 = time.time()
        mujoco.mj_step(model, data)

        if streams is not None:
            streams.update(data)

        dt = model.opt.timestep - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)

    if streams is not None:
        streams.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xml")
    parser.add_argument("--render",    action="store_true",
                        help="show cv2 camera windows")
    parser.add_argument("--cam-every", type=int, default=1)
    parser.add_argument("--duration",  type=float, default=30.0)
    args = parser.parse_args()
    main(args)
