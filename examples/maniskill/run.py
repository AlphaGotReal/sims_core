#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from maniskill.scene              import load as load_scene
from maniskill.hardware           import load as load_hardware
from maniskill.controller_manager import ControllerManager


parser = argparse.ArgumentParser()
parser.add_argument("--scene",    "-s",  required=True, help="path to scene description YAML")
parser.add_argument("--hardware", "-hw",                help="path to hardware config YAML", default=None)
parser.add_argument("--actors",   "-a",                 help="path to actors YAML",          default=None)
args = parser.parse_args()

scene_cfg = load_scene(args.scene)
hw_cfg    = load_hardware(args.hardware) if args.hardware else None

cm = ControllerManager(scene_cfg, hw_cfg=hw_cfg, actors_path=args.actors)
cm.run()
cm.close()
