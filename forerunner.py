import os
import json
import math
import numpy as np

from isaacsim import SimulationApp

FR_EXPERIENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'forerunner.exp.kit')

simulation_app = SimulationApp(launch_config={'headless': False}, experience=FR_EXPERIENCE)
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

from earth import EarthScene

# Create a sim world
world = EarthScene(stage_units_in_meters=1.0, physics_dt=1/60, rendering_dt=1/60, backend="numpy", device="cuda")

# Reset world (this sets simulation to 'playing' state)
world.reset()

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
