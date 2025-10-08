import asyncio
import omni.ui as ui
import omni.ui.scene as sc
from omni.kit.viewport.utility import get_active_viewport_window
import omni.kit.app

from earth import EarthScene

class ForerunnerUI():

    def __init__(self, sim : EarthScene):
        '''We can only initialize this object after `KitApp.startup()` is called with `["--enable", "omni.ui"]`.'''

        self.sim = sim

        #         # Add a
        #         f = ui.FloatField()

        #         def clicked(f=f):
        #             print("clicked")
        #             f.model.set_value(f.model.get_value_as_float() + 1)

        #         ui.Button("Plus One", clicked_fn=clicked)

        self.simInfoWindow = ui.Window("Simulation Information", width=300, height=100)        
        with self.simInfoWindow.frame:
            with ui.VStack():
                self.simInfoLabel = ui.Label("")                
                self.simInfoLabel.visible = True
                self.simInfoLabel.set_style({"font_size": 18})

        asyncio.ensure_future(self._dock())

    async def _dock(self) -> None:
        for i in range(5):
            await omni.kit.app.get_app().next_update_async()
        viewportWindow = ui.Workspace.get_window("Viewport")
        self.simInfoWindow.dock_in(viewportWindow, ui.DockPosition.TOP, self.simInfoWindow.height / viewportWindow.height)

        for windowHandle in ui.Workspace.get_windows():
            print(windowHandle.title)

    def updateSimulationInfo(self):
        '''Generates the simulation info string and applies it to the `self.simInfoWindow.text` field.'''

        n = len(self.sim.satellites)

        dateString = self.sim.simTime.utc_strftime(format='%Y-%m-%d %H:%M:%S UTC')
        infos = f"Simulation datetime:\n{dateString}"
        infos += f"\nNumber of Satellites: {n}"

        self.simInfoLabel.text = infos

