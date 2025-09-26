# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import numpy as np

from isaacsim import SimulationApp

CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": False,
}

simulation_app = SimulationApp({'headless': False})
simulation_app.set_setting("/app/useFabricSceneDelegate", True)
simulation_app.set_setting("/app/usdrt/scene_delegate/enableProxyCubes", False)
simulation_app.set_setting("/app/usdrt/scene_delegate/geometryStreaming/enabled", False)
simulation_app.set_setting("/omnihydra/parallelHydraSprimSync", False)

simulation_app.update()

import omni.usd
from pxr import Sdf, UsdLux, Gf, UsdGeom, Usd
from isaacsim.core.utils.stage import add_reference_to_stage
import isaacsim.core.utils.math as math_utils
import isaacsim.core.utils.prims as prim_utils
import omni.kit.commands
from isaacsim.core.utils.viewports import set_camera_view

from isaacsim.gui.components import Frame, Button, StringField, TextBlock


import warp as wp

print(wp.__version__)

from earth import EarthScene

#usdContext = omni.usd.get_context()
#cesiumTilesetPath = "C:\\isaac-sim4_5\\standalone_examples\\Forerunner\\CesiumTilesets.usd"
#usdContext.open_stage(cesiumTilesetPath)

# Create a sim world
world = EarthScene(stage_units_in_meters=1.0, physics_dt=1/30, rendering_dt=1/30, backend="warp", device="cuda")

# Reset world (this sets simulation to 'playing' state)
world.reset()
groups = [
        ("cubesat",     Gf.Vec3f(1, 0, 0)),
        ("gnss",        Gf.Vec3f(0, 1, 0)),
        ("geo",         Gf.Vec3f(1, 1, 0)),
        ("starlink",    Gf.Vec3f(48/255, 1, 1)),
        ("iridium-NEXT",    Gf.Vec3f(1, 0, 1)), 
        #("active", Gf.Vec3f(48/255, 1, 1))    
    ]
world.loadSatellites(groups)

cameraDistance = EarthScene.wgs84semiMajor * 5

world.initializeCamera(cameraDistance)

import omni.ui as ui
window = ui.Window("Viewport")
simTimeBlock = None
with window.frame:
    with ui.VStack():
        simTimeBlock = ui.Label("----.--", alignment=ui.Alignment.LEFT_TOP)
        simTimeBlock.visible = True
        simTimeBlock.height = ui.Length(30)
        simTimeBlock.width = ui.Length(100)        
        simTimeBlock.set_style({"background_color": 0xFF0000FF, "font_size": 20})

# start the simulator
while simulation_app.is_running():

    # Step simulation
    world.step()

    # Let camera orbit
    #world.updateCameraOrbit(world.current_time, distance=cameraDistance, speed=10)
    #world.updateCameraFollowSatellite(world.satellites[0], 300.)

    # Update UI with some time infos
    simTimeBlock.text = f"{world.simTime.utc_strftime(format='%Y-%m-%d %H:%M:%S UTC')}"

# shutdown the simulator automatically
print("Closing...")

simulation_app.close()
