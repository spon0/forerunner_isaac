import os
import json
import numpy as np
import math

import urllib3

import omni.usd
import omni.kit.commands
from omni.isaac.core import World
from isaacsim.sensors.camera import Camera
# from omni.isaac.core.objects import VisualSphere
# from isaacsim.core.api.objects import VisualSphere
# from isaacsim.core.prims import GeometryPrim, SdfShapePrim
import isaacsim.core.utils.prims as prim_utils
# from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.viewports import set_camera_view
from omni.physx.scripts import physicsUtils
from pxr import Sdf, UsdLux, UsdGeom, Gf, UsdPhysics, Vt, Usd

from skyfield.api import EarthSatellite, load, Timescale, Time, Distance
from skyfield import framelib

import warp as wp

class EarthScene(World):

    wgs84semiMajor = 6378137.0 / 1000.
    wgs84semiMinor = 6356752.314245 / 1000.

    SATTYPE_COLOR_MAPPING = {
        'ROCKET BODY': Gf.Vec3f(1, 0, 1),
        'DEBRIS': Gf.Vec3f(0, 1, 1),
        'PAYLOAD': Gf.Vec3f(0, 1, 0),
        'UNKNOWN': Gf.Vec3f(1, 0, 0)
    }

    def __init__(self, physics_dt: float | None = None, rendering_dt: float | None = None, stage_units_in_meters: float | None = None, physics_prim_path: str = "/physicsScene", sim_params: dict = None, set_defaults: bool = True, backend: str = "numpy", device: str | None = None) -> None:
        

        # Initialize empty list of space objects to simulate
        self.satellites : list[EarthSatellite] = []

        # Intialize earth rotation timescale
        self.timescale : Timescale = load.timescale()

        self.simEpoch : Time = self.timescale.now()
        self.simTime : Time = None # type: ignore

        self.timestep = 0
        self.speed = 50.0
        self.timestepsPerUpdate = 50

        # Get the stage
        stage = omni.usd.get_context().get_stage()
        self._stage = stage

        # scaling factor determined to be 20 based on original earth.usd having a bounding box of:
        # [(-10.003646850585938, -10.021764755249023, -10.020329475402832)...(10.003656387329102, 10.02176570892334, 10.020326614379883)]
        scalingFactor = 10.02

        usd_file = "earth.usd"
        #usd_file = "D:\\Projects\\SpaceInteractions\\Earth2\\earth2-weather-analytics\\e2cc\\source\\extensions\\omni.earth_2_command_center.app.globe_view\\data\\dynamic_texture_tests\\test_002\\test_002.usda"
        #usd_file = "D:\\Projects\\SpaceInteractions\\Earth2\\earth2-weather-analytics\\e2cc\\source\\extensions\\omni.earth_2_command_center.app.globe_view\\data\\stages\\diamond_globe_2.usd"
        script_path = os.path.dirname(os.path.abspath(__file__))
        usd_path = os.path.join(script_path, usd_file)
        #stage = Usd.Stage.CreateNew(usd_file)

        
        # Get the USD context
        # usd_context = omni.usd.get_context()
        # # Open the USD stage
        # result = usd_context.open_stage(usd_file)
        # stage = usd_context.get_stage()
        # print(f"Opened stage {stage} with result {result}")

        super().__init__(physics_dt, rendering_dt, stage_units_in_meters, physics_prim_path, sim_params, set_defaults, backend, device)
        #omni.usd.get_context().open_stage(usd_path)
        # bb = omni.usd.get_context().compute_path_world_bounding_box(usd_path)
        # print(bb)

        # globe_xform = UsdGeom.Xform(stage.GetPrimAtPath(Sdf.Path('/World/earth_xform/diamond_globe')))
        # if globe_xform:
        #     scale_attr = globe_xform.GetPrim().GetAttribute('xformOp:scale')
        #     if scale_attr:
        #         scale = scale_attr.Get()
        #         print(f'Setting Earth Radius to: {scale[0]}')

        omni.kit.commands.execute(
            "IsaacSimSpawnPrim",
            usd_path=usd_path,
            prim_path="/World/earth",
            translation=(0, 0, 0),
            rotation=(0.0, 0.0, 0.0, 0.0),
        )

        omni.kit.commands.execute(
            "IsaacSimScalePrim",
            prim_path='/World/earth',
            scale=(EarthScene.wgs84semiMajor/scalingFactor, EarthScene.wgs84semiMajor/scalingFactor, EarthScene.wgs84semiMinor/scalingFactor)
        )
        success = stage.GetPrimAtPath("/World/earth").GetAttribute("xformOp:orient").Set(Gf.Quatd(0.5, 0.5, 0.5, 0.5), 0)
        print("Changed the rotation of the prim: ", success)

        eph = load('de421.bsp')
        self.sun = eph['sun']
        self.earth = eph['earth']
        self.sunDist = EarthScene.wgs84semiMajor * 50
        
        # Create the sun
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
        self.satScales : np.ndarray = None
        self.cameraPrim : Camera = None
        self.satDistanceScaler : float = 0.00012

        # Define the prim path for the Dome Light
        domeLightPrimPath = Sdf.Path("/World/DomeLight")
        # Create the Dome Light prim
        domeLight = UsdLux.DomeLight.Define(stage, domeLightPrimPath)
        # Set attributes for the Dome Light (optional)
        # Example: Set intensity
        domeLight.CreateIntensityAttr(10)

        
        cameraDistance = EarthScene.wgs84semiMajor * 5

        self.initializeCamera(cameraDistance)
        self.loadSpaceTrack(7.0)
        self.initializeSatellitesGeoms()

    def getCameraPosition(self) -> wp.vec3:

        transformMat = omni.usd.get_world_transform_matrix(self.stage.GetPrimAtPath("/OmniverseKit_Persp"))
        pos = transformMat.ExtractTranslation()
        
        return wp.vec3(pos[0], pos[1], pos[2])

    def step(self, render: bool = True, step_sim: bool = True) -> None:

        secPerDay = 86400
        self.simTime = self.simEpoch + (self.current_time * self.speed / secPerDay)

        if len(self.satellites) > 0:
            # Update any pos/vel for whose turn it is
            for i, sat in enumerate(self.satellites):
                if self.timestep % self.timestepsPerUpdate == sat.updateIdx or sat.selected:
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

            # Position calc
            wp.launch(sgp4kernel, dim=n, inputs=[pos, vel, s, out], device="cuda")

            self.satPositions = out.numpy()
            self.satellitesPrim.GetPositionsAttr().Set(self.satPositions)

            # Scale calc
            scalesOut = wp.empty(shape=n, dtype=wp.vec3, device="cuda")

            wp.launch(cameraDistKernel, dim=n, inputs=[pos, self.getCameraPosition(), self.satDistanceScaler, scalesOut], device="cuda")
            self.satScales = scalesOut.numpy()
            self.satellitesPrim.GetScalesAttr().Set(self.satScales)

        # Update sun position
        sunPos = self.getSunPosition(self.simTime)
        physicsUtils.set_or_add_translate_op(self.sunXformApi, sunPos)

        self.timestep += 1

        return super().step(render, step_sim)

    def initializeSatellitesFromTles(self, tles):
        ts = load.timescale()
        for tle in tles:
            sat = EarthSatellite(line1=tle['TLE_LINE1'], line2=tle['TLE_LINE2'], name=tle['OBJECT_NAME'], ts=ts)
            sat.color = EarthScene.SATTYPE_COLOR_MAPPING[tle['OBJECT_TYPE']]
            geocentric = sat.at(self.simEpoch)
            sat.pos = geocentric.xyz.km
            sat.vel = geocentric.velocity.km_per_s * self.speed
            sat.id = tle['NORAD_CAT_ID'].rjust(5, '0')
            sat.selected = False
            self.satellites.append(sat)

    def initializeSatellitesFromOmms(self, ommData):
        ts = load.timescale()
        for fields in ommData:
            print(fields)
            sat = EarthSatellite.from_omm(ts, fields)
            sat.color = group[1] # type: ignore
            geocentric = sat.at(self.simEpoch)
            sat.pos = geocentric.xyz.km
            sat.vel = geocentric.velocity.km_per_s * self.speed
            #sat.id = tle['NORAD_CAT_ID'].rjust(5, '0')
            sat.selected = False
            self.satellites.append(sat)
        
    def loadSpaceTrack(self, maxDaysOld: float=7.0):

        import requests
        # d:\Projects\Space Interactions\Omniverse Forerunner\kit\python\Lib\datetime.py is overriding the datetime stdlib
        import datetime

        now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        start = now - datetime.timedelta(days=maxDaysOld)
        end = now
        tles = []
        try:
            # Hardcoded for prototyping - TODO
            url = f"http://127.0.0.1:5000/get_tles_between?aDate={start.isoformat()}&bDate={end.isoformat()}"
            response = requests.get(url)

            if response.status_code == 200:

                tles = json.loads(response.text)

            else:

                raise requests.exceptions.ConnectionError
            
        except requests.exceptions.ConnectionError:
            print("localhost:5000 is offline, defaulting to backup.tle")
            # Server is offline, default to backup.tle   
            backupTles = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backup.tle')         
            tles = json.load(open(backupTles))

        print("Received", len(tles), "TLEs")
        self.initializeSatellitesFromTles(tles=tles)

    def loadCelesTrack(self, groups : list[tuple[str, list]], maxDaysOld : float = 7.0):

        sscs = set()

        for group in groups:
            name = f'{group[0]}.json'
            url = f'https://celestrak.org/NORAD/elements/gp.php?GROUP={group[0]}&FORMAT=json'

            if not load.exists(name) or load.days_old(name) >= maxDaysOld:
                    load.download(url, filename=name)

            satGroup = []
            with load.open(name) as f:
                data = json.load(f)

                self.initializeSatellitesFromOmms(ommData=data)
                
    def initializeSatellitesGeoms(self):
        protoPath = "/World/Prototypes/Sphere"
        protoPrim = UsdGeom.Sphere.Define(omni.usd.get_context().get_stage(), protoPath)
        protoPrim.GetRadiusAttr().Set(15)

        ptInstancePath = "/World/satellites"
        self.satellitesPrim = UsdGeom.PointInstancer.Define(omni.usd.get_context().get_stage(), ptInstancePath)
        self.satellitesPrim.GetPrototypesRel().AddTarget(protoPath)

        if len(self.satellites) == 0: return

        positions = []
        velocities = []
        oris = []
        indices = []
        colors = []
        for sat in self.satellites:            
            
            positions.append(sat.pos)
            velocities.append(sat.vel)
            indices.append(0)
            oris.append(Gf.Quath(1, 0, 0, 0))
            colors.append(sat.color)

        self.satPositions = np.array(positions)
        self.satVelocities = np.array(velocities)

        # Get dimension for warp kernel
        n = len(self.satellites)  
        # Pack all pos
        pos = wp.from_numpy(self.satPositions, dtype=wp.vec3, device="cuda")        
        out = wp.empty(shape=n, dtype=wp.vec3, device="cuda")

        wp.launch(cameraDistKernel, dim=n, inputs=[pos, self.getCameraPosition(), self.satDistanceScaler, out], device="cuda")
        self.satScales = out.numpy()
        self.satellitesPrim.GetPositionsAttr().Set(positions)
        self.satellitesPrim.GetOrientationsAttr().Set(oris)
        self.satellitesPrim.GetScalesAttr().Set(self.satScales)
        self.satellitesPrim.GetProtoIndicesAttr().Set(indices)
        primvarApi = UsdGeom.PrimvarsAPI(self.satellitesPrim)
        diffuse_color_primvar = primvarApi.CreatePrimvar(
            "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.varying
        )
        diffuse_color_primvar.Set(colors)

        # Assign timestep for SGP4 update call
        for sat in self.satellites:
            sat.updateIdx = np.random.randint(self.timestepsPerUpdate) # type: ignore

    def initializeCamera(self, distance):
        prim_path = "/OmniverseKit_Persp"

        self.cameraPrim = Camera(prim_path=prim_path)
        self.cameraPrim.set_clipping_range(10.0, EarthScene.wgs84semiMajor * 50)
        
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

    def updateCameraFollowSatellite(self, satIdx: int, distance: float):
        prim_path = "/OmniverseKit_Persp"

        # pos, vel = self.satPositions[satIdx], self.satVelocities[satIdx]
        # eye = pos - (vel * distance)

        target = self.satPositions[satIdx]
        eye = target + (target / np.linalg.norm(target) * distance)

        # camera = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)        
        # world_transform_matrix = omni.usd.get_world_transform_matrix(camera)
        # print(world_transform_matrix)
        # ori = world_transform_matrix.ExtractRotation()
        # print(ori)

        # xform = UsdGeom.Xformable(camera)
        
        # new_rotation_eulers = [90.0, 0.0, 0.0] # X, Y, Z angles in degrees

        # omni.kit.commands.execute(
        #     "TransformMultiPrimsSRTCpp",
        #     count=1,
        #     paths=[prim_path],
        #     new_rotation_eulers=[new_rotation_eulers],
        #     # Add other transform properties like position and scale if needed
        #     new_translations=[eye],
        #     new_scales=[[1.0, 1.0, 1.0]],
        #     time_code=0.0
        # )

        set_camera_view(
            eye=eye, target=target, camera_prim_path=prim_path
        )


    def getSunPosition(self, time: Time) -> Gf.Vec3d:
        apparent = self.earth.at(time).observe(self.sun).apparent()
        v = apparent.frame_xyz(framelib.itrs).m

        # Normalize and extent to our 'sun distance'
        v = v / np.linalg.norm(v) * self.sunDist

        return Gf.Vec3d(v[0], v[1], v[2])
    
    def showOrbitCurve(self, sat) -> None:

        points = []
        widths = []
        times = np.linspace(0, 120 * 60.0 , 360)
        for t in times:
            pos = sat.at(self.simTime + (t/86400)).xyz.km
            points.append(Gf.Vec3f(pos[0], pos[1], pos[2]))
            widths.append(10.0)

        curve_path = "/World/orbit/curve"
        curve = UsdGeom.NurbsCurves.Define(self._stage, curve_path)

        # Set the points attribute
        curve.CreatePointsAttr().Set(Vt.Vec3fArray(points))

        # Set the widths
        curve.CreateWidthsAttr(Vt.FloatArray(widths))

        # Set the color
        curve.CreateDisplayColorAttr(Vt.Vec3fArray(1, Gf.Vec3f(1.0, 1.0, 0.0)), writeSparsely=False)

        # Set the curve vertex counts attribute
        curve.CreateCurveVertexCountsAttr().Set([len(points)])

    def clearOrbitCurve(self) -> None:
        self._stage.RemovePrim("/World/orbit/curve")

    
@wp.kernel
def sgp4kernel(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), s: float, out: wp.array(dtype=wp.vec3)): # type: ignore
    tid = wp.tid()
    x = pos[tid]
    v = vel[tid]
    out[tid] = x + v * s

@wp.kernel
def cameraDistKernel(pos: wp.array(dtype=wp.vec3), camPos: wp.vec3, s: float, out: wp.array(dtype=wp.vec3)): # type: ignore
    tid = wp.tid()
    x = pos[tid]    
    dx = x[0] - camPos[0]
    dy = x[1] - camPos[1]
    dz = x[2] - camPos[2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    scale = dist * s
    out[tid] = wp.vec3(scale, scale, scale)
