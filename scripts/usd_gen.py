#! /usr/bin/env python

import os
import tempfile
import urdf_usd_converter
import usdex.core
from pxr import Sdf, Usd

SIMS_CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

urdf_template = open(os.path.join(SIMS_CORE, "robot.urdf")).read()
urdf_content  = urdf_template.format(SIMS_CORE=SIMS_CORE)

with tempfile.NamedTemporaryFile(
    suffix=".urdf", mode="w", delete=False
) as tmp:
    tmp.write(urdf_content)
    tmp_path = tmp.name

converter = urdf_usd_converter.Converter()
asset: Sdf.AssetPath = converter.convert(tmp_path, "robot_usd")
os.unlink(tmp_path)

stage: Usd.Stage = Usd.Stage.Open(asset.path)
# modify further using Usd or usdex.core functionality
usdex.core.saveStage(stage, comment="modified after conversion")
