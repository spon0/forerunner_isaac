import asyncio
import carb

import omni.ui as ui
import omni.ui.scene as sc
from omni.kit.viewport.utility import get_active_viewport_window
from isaacsim.core.utils.viewports import set_camera_view
import omni.kit.app
from omni.kit.widget.searchable_combobox import build_searchable_combo_widget, ComboBoxListDelegate

from earth import EarthScene

class ForerunnerUI():

    EMPTY_COMBO_VAL = "Search..."

    def __init__(self, sim : EarthScene):
        '''We can only initialize this object after `KitApp.startup()` is called with `["--enable", "omni.ui"]`.'''

        self.sim = sim

        self.simInfoLabel : ui.Label = None
        self.simInfoWindow = self.createSimulationInfoWindow()
        self.selectSatWindow = self.createSelectSatWindow()
        self.selectedSatIdx = -1
        self.selectedSat = None
        eventsInterface = carb.events.acquire_events_interface()
        #         # Add a
        #         f = ui.FloatField()

        #         def clicked(f=f):
        #             print("clicked")
        #             f.model.set_value(f.model.get_value_as_float() + 1)

        #         ui.Button("Plus One", clicked_fn=clicked)

        

        asyncio.ensure_future(self._dock())

    async def _dock(self) -> None:
        '''Dock all of our GUI windows in the viewport window.'''  

        await omni.kit.app.get_app().next_update_async()
        viewportWindow = ui.Workspace.get_window("Viewport")

        height = viewportWindow.height
        width = viewportWindow.width

        # Dock simulation info window
        self.simInfoWindow.dock_in(viewportWindow, ui.DockPosition.TOP, self.simInfoWindow.height / height)

        # Dock select satellite window
        self.selectSatWindow.dock_in(viewportWindow, ui.DockPosition.RIGHT, self.selectSatWindow.width / width)

    def createSimulationInfoWindow(self) -> ui.Window:
        simInfoWindow = ui.Window("Simulation Information", width=300, height=100)        
        with simInfoWindow.frame:
            with ui.VStack():
                self.simInfoLabel = ui.Label("")                
                self.simInfoLabel.visible = True
                self.simInfoLabel.set_style({"font_size": 18})

        return simInfoWindow

    def updateSimulationInfo(self):
        '''Generates the simulation info string and applies it to the `self.simInfoWindow.text` field.'''

        n = len(self.sim.satellites)

        dateString = self.sim.simTime.utc_strftime(format='%Y-%m-%d %H:%M:%S UTC')
        infos = f"Simulation datetime:\n{dateString}"
        infos += f"\nNumber of Satellites: {n}"

        self.simInfoLabel.text = infos

    def satelliteComboClick(self, model):
        selected_item = model.get_value_as_string()

        if selected_item == ForerunnerUI.EMPTY_COMBO_VAL:
            self.selectedSat.selected = False
            self.selectedSat = None
            self.selectedSatIdx = -1
            self.sim.clearOrbitCurve()        

        # Get norad cat id and set selectedSat
        ssc = selected_item[-5:]
        for i, sat in enumerate(self.sim.satellites):
            if sat.id == ssc:
                sat.selected = True
                self.selectedSat = sat
                self.selectedSatIdx = i
                self.sim.showOrbitCurve(sat)

                # set initial camera tether
                prim_path = "/OmniverseKit_Persp"
                distance = 30.0
                pos, vel = self.sim.satPositions[self.selectedSatIdx], self.sim.satVelocities[self.selectedSatIdx]
                eye = pos - (vel * distance)
                target = self.sim.satPositions[self.selectedSatIdx]
                set_camera_view(
                    eye=eye, target=target, camera_prim_path=prim_path
                )
                return

    def createSelectSatWindow(self) -> ui.Window:
        # Run the build_ui function to display the UI
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.kit.widget.searchable_combobox", True)
        satelliteSelectWindow = ui.Window("Satellite Selection", width=300, height=100)
        with satelliteSelectWindow.frame:
            with ui.VStack():
                # Define the list of items for the combo box
                itemList = []
                for sat in self.sim.satellites:
                    item = f'{sat.name} -- {sat.id}'
                    itemList.append(item)

                # Add the searchable combo box to the UI
                # Create the searchable combo box with the specified items and callback
                searchable_combo_widget = build_searchable_combo_widget(
                    combo_list=itemList,
                    combo_index=-1,  # Start with no item selected
                    combo_click_fn=self.satelliteComboClick,
                    widget_height=18,
                    default_value=ForerunnerUI.EMPTY_COMBO_VAL,  # Placeholder text when no item is selected
                    window_id="SearchableComboBoxWindow",
                    delegate=ComboBoxListDelegate()  # Use the default delegate for item rendering
                )

        return satelliteSelectWindow