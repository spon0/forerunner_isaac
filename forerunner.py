import os
import json
import math
import numpy as np

from isaacsim import SimulationApp

FR_EXPERIENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'forerunner.exp.kit')

simulation_app = SimulationApp(launch_config={'headless': False}, experience=FR_EXPERIENCE)
simulation_app.set_setting("/app/useFabricSceneDelegate", True)
simulation_app.set_setting("persistent/app/viewport/displayOptions", 0)
# simulation_app.set_setting("/app/usdrt/scene_delegate/enableProxyCubes", False)
# simulation_app.set_setting("/app/usdrt/scene_delegate/geometryStreaming/enabled", False)
# simulation_app.set_setting("/omnihydra/parallelHydraSprimSync", False)
simulation_app.update()

from earth import EarthScene
from simInfoGui import ForerunnerUI

# Create a sim world
world = EarthScene(stage_units_in_meters=1.0, physics_dt=1/60, rendering_dt=1/60, backend="numpy", device="cuda")

# Create the UI
gui = ForerunnerUI(world)

# Reset world (this sets simulation to 'playing' state)
world.reset()


# start the simulator
while simulation_app.is_running():

    # Step simulation
    world.step()

    # Let camera orbit
    #world.updateCameraOrbit(world.current_time, distance=cameraDistance, speed=10)
    if gui.selectedSat != None:        
        world.updateCameraFollowSatellite(gui.selectedSatIdx, 1000.)

    # Update UI
    gui.updateSimulationInfo()

# shutdown the simulator automatically
print("Closing...")

simulation_app.close()
