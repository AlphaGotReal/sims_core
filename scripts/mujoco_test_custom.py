#! /usr/bin/env python

import os
import time
import argparse
import threading
import tkinter as tk
from tkinter import ttk

import mujoco

from _mj_utils import load_model_with_cameras, CameraStreams


ARM_JOINTS = {
    "left": [
        ("left_joint1",   0),
        ("left_joint2",   1),
        ("left_joint3",   2),
        ("left_joint4",   3),
        ("left_joint5",   4),
        ("left_joint6",   5),
        ("left_joint7.1", 6),
        ("left_joint7.2", 7),
    ],
    "right": [
        ("right_joint1",   8),
        ("right_joint2",   9),
        ("right_joint3",  10),
        ("right_joint4",  11),
        ("right_joint5",  12),
        ("right_joint6",  13),
        ("right_joint7.1",14),
        ("right_joint7.2",15),
    ],
}


def build_gui(root, model, data, ctrl_lock):
    sliders = {"left": {}, "right": {}}

    for arm_name, arm_label in [("left", "LEFT ARM"), ("right", "RIGHT ARM")]:
        frame = ttk.LabelFrame(root, text=arm_label, padding=10)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        for joint_name, act_id in ARM_JOINTS[arm_name]:
            var = tk.DoubleVar(value=0.0)
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=3)

            ttk.Label(row, text=joint_name, width=14).pack(side="left", padx=5)

            def on_change(v, aid=act_id):
                lo, hi    = model.actuator_ctrlrange[aid]
                mid       = (lo + hi) / 2.0
                half_rng  = (hi - lo) / 2.0
                with ctrl_lock:
                    data.ctrl[aid] = mid + float(v) * half_rng

            ttk.Scale(row, from_=-1, to=1, variable=var,
                      command=on_change).pack(side="left", fill="x",
                                              expand=True, padx=5)
            val_label = ttk.Label(row, text="0.00", width=6)
            val_label.pack(side="left", padx=5)
            var.trace_add("write",
                          lambda *a, sv=var, lbl=val_label:
                          lbl.config(text=f"{float(sv.get()):.2f}"))

            sliders[arm_name][act_id] = var

    btns = ttk.Frame(root); btns.pack(fill="x", padx=5, pady=10)
    def reset_all():
        for arm in sliders:
            for v in sliders[arm].values():
                v.set(0)
    ttk.Button(btns, text="Reset All",
               command=reset_all).pack(side="left", expand=True)


def main(args):
    xml_path = os.path.abspath(args.xml)
    os.chdir(os.path.dirname(xml_path))

    model, cams = load_model_with_cameras(xml_path)
    data        = mujoco.MjData(model)

    ctrl_lock   = threading.Lock()
    sim_running = [True]
    streams     = (CameraStreams(model, cams, show=args.render,
                                 every=args.cam_every)
                   if args.render else None)

    def sim_loop():
        while sim_running[0]:
            t0 = time.time()
            with ctrl_lock:
                mujoco.mj_step(model, data)
            if streams is not None:
                streams.update(data)
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    threading.Thread(target=sim_loop, daemon=True).start()

    root = tk.Tk()
    root.title("Arm Control")
    root.geometry("420x640")
    build_gui(root, model, data, ctrl_lock)
    def on_closing():
        sim_running[0] = False
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

    if streams is not None:
        streams.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xml")
    parser.add_argument("--render",    action="store_true",
                        help="show cv2 camera windows")
    parser.add_argument("--cam-every", type=int, default=2)
    args = parser.parse_args()
    main(args)
