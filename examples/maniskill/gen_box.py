#!/usr/bin/env python3
"""Generate a cardboard box URDF with 4 articulated top flaps (4 DOFs).

RSC (Regular Slotted Container) closing order:
  1. minor flaps (±X sides, depth = W/2) fold in first
  2. major flaps (±Y sides, depth = W/2) fold on top

Joint angles at open (default) = 0  (flaps vertical)
Joint angles at closed         = ±π/2 (flaps horizontal, covering the top)

  right_minor_joint : axis +Y,  close = −π/2
  left_minor_joint  : axis +Y,  close = +π/2
  front_major_joint : axis +X,  close = +π/2
  back_major_joint  : axis +X,  close = −π/2

Usage:
    python gen_box.py --length 0.3 --width 0.2 --height 0.2 --out box.urdf
"""
import argparse
import math


def generate(L: float, W: float, H: float, t: float = 0.003) -> str:
    fd     = W / 2           # flap depth (standard RSC)
    hp     = math.pi / 2
    color  = "0.76 0.60 0.42 1.0"

    def visual_collision(sx, sy, sz, ox, oy, oz):
        geom = f"<box size=\"{sx} {sy} {sz}\"/>"
        orig = f"<origin xyz=\"{ox} {oy} {oz}\" rpy=\"0 0 0\"/>"
        mat  = f"<material name=\"cardboard\"><color rgba=\"{color}\"/></material>"
        return (
            f"    <collision><geometry>{geom}</geometry>{orig}</collision>\n"
            f"    <visual><geometry>{geom}</geometry>{orig}{mat}</visual>"
        )

    def flap_link(name, sx, sy, sz):
        shapes = visual_collision(sx, sy, sz, 0, 0, sz / 2)
        return (
            f"  <link name=\"{name}\">\n"
            f"    <inertial><mass value=\"0.02\"/>"
            f"<inertia ixx=\"1e-5\" ixy=\"0\" ixz=\"0\" iyy=\"1e-5\" iyz=\"0\" izz=\"1e-5\"/>"
            f"</inertial>\n"
            f"{shapes}\n"
            f"  </link>"
        )

    def joint(name, parent, child, ox, oy, oz, ax, ay, az, lo, hi):
        return (
            f"  <joint name=\"{name}\" type=\"revolute\">\n"
            f"    <parent link=\"{parent}\"/><child link=\"{child}\"/>\n"
            f"    <origin xyz=\"{ox} {oy} {oz}\" rpy=\"0 0 0\"/>\n"
            f"    <axis xyz=\"{ax} {ay} {az}\"/>\n"
            f"    <limit lower=\"{lo:.6f}\" upper=\"{hi:.6f}\" effort=\"10\" velocity=\"3\"/>\n"
            f"    <dynamics damping=\"0.1\" friction=\"0.05\"/>\n"
            f"  </joint>"
        )

    bottom   = visual_collision(L,       W,       t, 0,          0,          -H/2 + t/2)
    wall_px  = visual_collision(t,       W,       H, L/2 - t/2,  0,          0         )
    wall_nx  = visual_collision(t,       W,       H, -L/2 + t/2, 0,          0         )
    wall_py  = visual_collision(L,       t,       H, 0,          W/2 - t/2,  0         )
    wall_ny  = visual_collision(L,       t,       H, 0,          -W/2 + t/2, 0         )

    base = (
        "  <link name=\"base\">\n"
        "    <inertial><mass value=\"0.3\"/>"
        "<inertia ixx=\"1e-3\" ixy=\"0\" ixz=\"0\" iyy=\"1e-3\" iyz=\"0\" izz=\"1e-3\"/>"
        "</inertial>\n"
        f"{bottom}\n{wall_px}\n{wall_nx}\n{wall_py}\n{wall_ny}\n"
        "  </link>"
    )

    return "\n".join([
        "<robot name=\"cardboard_box\">",
        base,
        # minor flaps (close first, underneath major)
        flap_link("right_minor_flap", t,  W,  fd),
        joint("right_minor_joint", "base", "right_minor_flap",
               L/2,  0,    H/2,  0, 1, 0, -hp, 0),
        flap_link("left_minor_flap",  t,  W,  fd),
        joint("left_minor_joint",  "base", "left_minor_flap",
              -L/2,  0,    H/2,  0, 1, 0,   0, hp),
        # major flaps (close on top of minors)
        flap_link("front_major_flap", L,  t,  fd),
        joint("front_major_joint", "base", "front_major_flap",
               0,    W/2,  H/2,  1, 0, 0,   0, hp),
        flap_link("back_major_flap",  L,  t,  fd),
        joint("back_major_joint",  "base", "back_major_flap",
               0,   -W/2,  H/2,  1, 0, 0, -hp, 0),
        "</robot>",
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--length",    type=float, default=0.30)
    ap.add_argument("--width",     type=float, default=0.20)
    ap.add_argument("--height",    type=float, default=0.20)
    ap.add_argument("--thickness", type=float, default=0.003)
    ap.add_argument("--out",       default="cardboard_box.urdf")
    args = ap.parse_args()

    with open(args.out, "w") as f:
        f.write(generate(args.length, args.width, args.height, args.thickness))
    print(f"Written {args.out}  (L={args.length} W={args.width} H={args.height})")
