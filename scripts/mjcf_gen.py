#! /usr/bin/env python

import os
import tempfile
from urdf2mjcf.convert import convert_urdf_to_mjcf

SIMS_CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

urdf_template = open(os.path.join(SIMS_CORE, "robot.urdf")).read()
urdf_content  = urdf_template.format(SIMS_CORE=SIMS_CORE)

tmp_path = os.path.join(SIMS_CORE, "_robot_tmp.urdf")
with open(tmp_path, "w") as tmp:
    tmp.write(urdf_content)

mjcf_path = os.path.join(SIMS_CORE, "robot.xml")
convert_urdf_to_mjcf(tmp_path, mjcf_path, copy_meshes=False)
os.unlink(tmp_path)
