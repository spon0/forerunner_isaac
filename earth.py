import os
import json
import numpy as np
import math

import omni.usd
import omni.kit.commands
from omni.isaac.core import World
# from omni.isaac.core.objects import VisualSphere
# from isaacsim.core.api.objects import VisualSphere
# from isaacsim.core.prims import GeometryPrim, SdfShapePrim
import isaacsim.core.utils.prims as prim_utils
# from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.viewports import set_camera_view
from omni.physx.scripts import physicsUtils
from pxr import Sdf, UsdLux, UsdGeom, Gf, UsdPhysics

from skyfield.api import EarthSatellite, load, Timescale, Time, Distance
from skyfield import framelib

import warp as wp

class EarthScene(World):

    wgs84semiMajor = 6378137.0 / 1000.
    wgs84semiMinor = 6356752.314245 / 1000.

    def __init__(self, physics_dt: float | None = None, rendering_dt: float | None = None, stage_units_in_meters: float | None = None, physics_prim_path: str = "/physicsScene", sim_params: dict = None, set_defaults: bool = True, backend: str = "numpy", device: str | None = None) -> None:
        super().__init__(physics_dt, rendering_dt, stage_units_in_meters, physics_prim_path, sim_params, set_defaults, backend, device)

        # Initialize empty list of space objects to simulate
        self.satellites : list[EarthSatellite] = []

        # Intialize earth rotation timescale
        self.timescale : Timescale = load.timescale()

        self.simEpoch : Time = self.timescale.now()
        self.simTime : Time = None # type: ignore

        self.timestep = 0
        self.speed = 50.0
        self.timestepsPerUpdate = 20

        # Get the stage
        stage = omni.usd.get_context().get_stage()

        # scaling factor determined to be 20 based on original earth.usd having a bounding box of:
        # [(-10.003646850585938, -10.021764755249023, -10.020329475402832)...(10.003656387329102, 10.02176570892334, 10.020326614379883)]
        scalingFactor = 10.02

        usd_file = "earth.usd"
        script_path = os.path.dirname(os.path.abspath(__file__))
        usd_path = os.path.join(script_path, usd_file)

        
        #omni.usd.get_context().open_stage(usd_path)

        omni.kit.commands.execute(
            "IsaacSimSpawnPrim",
            usd_path=usd_path,
            prim_path="/World/earth",
            translation=(0, 0, 0),
            rotation=(0.0, 0.0, 0.0, 0.0),
        )

        omni.kit.commands.execute(
            "IsaacSimScalePrim",
            prim_path="/World/earth",
            scale=(EarthScene.wgs84semiMajor / scalingFactor, EarthScene.wgs84semiMajor / scalingFactor, EarthScene.wgs84semiMinor / scalingFactor)
        )
        success = stage.GetPrimAtPath("/World/earth").GetAttribute("xformOp:orient").Set(Gf.Quatd(0.5, 0.5, 0.5, 0.5), 0)
        print("Changed the rotation of the prim: ", success)

        # Create the sun

        eph = load('de421.bsp')
        self.sun = eph['sun']
        self.earth = eph['earth']
        self.sunDist = EarthScene.wgs84semiMajor * 50

        sunPos = self.getSunPosition(self.simEpoch)
        sunPrim = prim_utils.create_prim(
            "/World/SunLight/Sun",
            "SphereLight",
            position=np.array([sunPos[0], sunPos[1], sunPos[2]]),  # Set the desired XYZ position # type: ignore
            attributes={
                "inputs:radius": 2.5E5,
                "inputs:intensity": 5E3,
                "inputs:color": (1.0, 0.9, 0.9)
            }
        )
        
        self.sunPrim = sunPrim
        self.sunXformApi = UsdGeom.Xformable(sunPrim)
        shadow_api = UsdLux.ShadowAPI.Apply(sunPrim)
        shadow_api.CreateShadowEnableAttr().Set(False)

        self.satellitesPrim = None

        self.satPositions : np.ndarray = None
        self.satVelocities : np.ndarray = None


        # bb = omni.usd.get_context().compute_path_world_bounding_box(usd_path)
        # print(bb)

    def step(self, render: bool = True, step_sim: bool = True) -> None:

        secPerDay = 86400
        self.simTime = self.simEpoch + (self.current_time * self.speed / secPerDay)

        if len(self.satellites) > 0:
            # Update any pos/vel for whose turn it is
            for i, sat in enumerate(self.satellites):
                if self.timestep % self.timestepsPerUpdate == sat.updateIdx:
                    geocentric = sat.at(self.simTime)
                    self.satPositions[i,:] = geocentric.xyz.km
                    self.satVelocities[i, :] = geocentric.velocity.km_per_s * self.speed

            # Get dimension for warp kernel
            n = len(self.satellites)  
            # Pack all pos/vels
            pos = wp.from_numpy(self.satPositions, dtype=wp.vec3, device="cuda")
            vel = wp.from_numpy(self.satVelocities, dtype=wp.vec3, device="cuda")
            s = self.get_physics_dt()
            out = wp.empty(shape=n, dtype=wp.vec3, device="cuda")

            wp.launch(sgp4kernel, dim=n, inputs=[pos, vel, s, out], device="cuda")

            self.satPositions = out.numpy()
            self.satellitesPrim.GetPositionsAttr().Set(self.satPositions)

        # Update sun position
        sunPos = self.getSunPosition(self.simTime)
        physicsUtils.set_or_add_translate_op(self.sunXformApi, sunPos)

        self.timestep += 1

        return super().step(render, step_sim)

    def loadSatellites(self, groups : list[tuple[str, list]], maxDaysOld : float = 7.0):

        sscs = set()

        for group in groups:
            name = f'{group[0]}.json'
            url = f'https://celestrak.org/NORAD/elements/gp.php?GROUP={group[0]}&FORMAT=json'

            if not load.exists(name) or load.days_old(name) >= maxDaysOld:
                    load.download(url, filename=name)

            satGroup = []
            with load.open(name) as f:
                data = json.load(f)

                ts = load.timescale()
                for fields in data:
                    sat = EarthSatellite.from_omm(ts, fields)
                    sat.color = group[1] # type: ignore
                    geocentric = sat.at(self.simEpoch)
                    sat.pos = geocentric.xyz.km
                    sat.vel = geocentric.velocity.km_per_s * self.speed
                    # Ensure we don't intialize duplicates
                    if sat.model.satnum in sscs:
                        continue
                    else:
                        sscs.add(sat.model.satnum)
                    satGroup.append(sat)

                print('Loaded', group[0], len(satGroup), 'satellites')

                self.satellites.extend(satGroup)

        protoPath = "/World/Prototypes/Sphere"
        protoPrim = UsdGeom.Sphere.Define(omni.usd.get_context().get_stage(), protoPath)
        protoPrim.GetRadiusAttr().Set(15)

        ptInstancePath = "/World/satellites"
        self.satellitesPrim = UsdGeom.PointInstancer.Define(omni.usd.get_context().get_stage(), ptInstancePath)
        self.satellitesPrim.GetPrototypesRel().AddTarget(protoPath)

        positions = []
        velocities = []
        oris = []
        scales = []
        indices = []
        colors = []
        for sat in self.satellites:            
            
            positions.append(sat.pos)
            velocities.append(sat.vel)
            indices.append(0)
            scale = np.clip(np.linalg.norm(sat.pos), EarthScene.wgs84semiMajor*2, EarthScene.wgs84semiMajor*6) / EarthScene.wgs84semiMajor
            scales.append(Gf.Vec3f(scale,scale,scale))
            oris.append(Gf.Quath(1, 0, 0, 0))
            colors.append(sat.color)

        self.satellitesPrim.GetPositionsAttr().Set(positions)
        self.satellitesPrim.GetOrientationsAttr().Set(oris)
        self.satellitesPrim.GetScalesAttr().Set(scales)
        self.satellitesPrim.GetProtoIndicesAttr().Set(indices)
        primvarApi = UsdGeom.PrimvarsAPI(self.satellitesPrim)
        diffuse_color_primvar = primvarApi.CreatePrimvar(
            "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.varying
        )
        diffuse_color_primvar.Set(colors)

        self.satPositions = np.array(positions)
        self.satVelocities = np.array(velocities)

        for sat in self.satellites:
            sat.updateIdx = np.random.randint(self.timestepsPerUpdate) # type: ignore

    def initializeCamera(self, distance):
        prim_path = "/OmniverseKit_Persp"

        from isaacsim.sensors.camera import Camera

        camera = Camera(prim_path=prim_path)
        camera.set_clipping_range(100.0, EarthScene.wgs84semiMajor * 50)

        angle = 0
        base = [1, 0, 0]
        x = base[0] * math.cos(angle) - base[1] * math.sin(angle)
        y = base[0] * math.sin(angle) + base[1] * math.cos(angle)
        z = base[2]

        # Move to zoom level
        x *= distance
        y *= distance
        z *= distance

        # Move camera up a bit for now
        z += EarthScene.wgs84semiMinor

        set_camera_view(
            eye=[x,y,z], target=[0.00, 0.00, 0.00], camera_prim_path=prim_path
        )

    def updateCameraOrbit(self, ts : float, distance : float, speed: float = 2):
        # Simple orbit around the equator
        prim_path = "/OmniverseKit_Persp"

        # Rotation around the z-axis... need to make a class for vector functions
        s = ts % 360.0
        angle = math.radians(s * speed)
        base = [1, 0, 0]
        x = base[0] * math.cos(angle) - base[1] * math.sin(angle)
        y = base[0] * math.sin(angle) + base[1] * math.cos(angle)
        z = base[2]

        # Move to zoom level
        x *= distance
        y *= distance
        z *= distance

        # Move camera up a bit for now
        z += EarthScene.wgs84semiMinor * 6

        set_camera_view(
            eye=[x,y,z], target=[0., 0., 0.], camera_prim_path=prim_path
        )

    def updateCameraFollowSatellite(self, sat:EarthSatellite, distance: float):
        prim_path = "/OmniverseKit_Persp"

        eye = sat.pos - (sat.vel * distance)

        set_camera_view(
            eye=eye, target=sat.pos, camera_prim_path=prim_path
        )

    def getSunPosition(self, time: Time) -> Gf.Vec3d:
        apparent = self.earth.at(time).observe(self.sun).apparent()
        v = apparent.frame_xyz(framelib.itrs).m

        # Normalize and extent to our 'sun distance'
        v = v / np.linalg.norm(v) * self.sunDist

        return Gf.Vec3d(v[0], v[1], v[2])
    
@wp.kernel
def sgp4kernel(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), s: float, out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    x = pos[tid]
    v = vel[tid]
    out[tid] = x + v * s