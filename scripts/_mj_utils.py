"""Shared MuJoCo helpers: camera injection + stream viewer.

Three RealSense-like cameras are injected onto URDF-derived mount bodies:
    - center_cam_link  (D455, overhead)
    - left_cam_link    (D405, left wrist)
    - right_cam_link   (D405, right wrist)

MuJoCo's <camera> fovy is in degrees (vertical). We pick values matching
the horizontal FOV of the real sensors at 848x480.
"""

from dataclasses import dataclass
import numpy as np
import mujoco


# 848x480, h-FOV 87 deg -> v-FOV:
# vfov = 2 * atan( tan(hfov/2) * H / W )
def _vfov_deg(hfov_deg: float, W: int, H: int) -> float:
    hfov = np.deg2rad(hfov_deg)
    vfov = 2.0 * np.arctan(np.tan(hfov / 2.0) * H / W)
    return float(np.rad2deg(vfov))


# urdf2mjcf collapses fixed-joint links into their parents, so the URDF
# camera frames (*_cam_link) don't survive as MJCF bodies. We attach the
# cameras to the nearest real bodies with explicit pose offsets instead.
#
# quat = [w, x, y, z]. MuJoCo cameras look down local -Z; we pick a quat
# that points the sensor at the workspace.

@dataclass
class CamSpec:
    name      : str
    mount     : str              # body name, or "worldbody"
    pos       : tuple            # (x, y, z) in mount frame (meters)
    quat      : tuple            # (w, x, y, z)
    width     : int   = 848
    height    : int   = 480
    hfov_deg  : float = 87.0
    near      : float = 0.07
    far       : float = 6.0


DEFAULT_CAMS = [
    # Wrist cams (D405): mounted on last arm link, looking forward.
    CamSpec("left_wrist_d405",  "left_link6",
            pos=(0.0, 0.0, 0.13), quat=(0.0, 1.0, 0.0, 0.0),
            near=0.07, far=0.5),
    CamSpec("right_wrist_d405", "right_link6",
            pos=(0.0, 0.0, 0.13), quat=(0.0, 1.0, 0.0, 0.0),
            near=0.07, far=0.5),
    # Overhead cam (D455): fixed in world, looking down at the table.
    CamSpec("center_d455",      "worldbody",
            pos=(0.0, 0.0, 1.6),  quat=(0.0, 1.0, 0.0, 0.0),
            near=0.10, far=6.0),
]


def load_model_with_cameras(xml_path: str, cams=DEFAULT_CAMS):
    """Load MJCF via MjSpec and inject <camera> elements on mount bodies.

    Falls back to the plain model if a mount body is missing.
    """
    spec = mujoco.MjSpec.from_file(xml_path)

    injected = []
    for cam in cams:
        if cam.mount == "worldbody":
            body = spec.worldbody
        elif _has_body(spec, cam.mount):
            body = spec.body(cam.mount)
        else:
            print(f"[cams] skip '{cam.name}': body '{cam.mount}' not found")
            continue

        body.add_camera(
            name = cam.name,
            fovy = _vfov_deg(cam.hfov_deg, cam.width, cam.height),
            pos  = list(cam.pos),
            quat = list(cam.quat),
        )
        injected.append(cam)

    model = spec.compile()
    return model, injected


def _depth_to_color(cv2, depth, near, far):
    """Colormap a float32 depth image (meters) into an 8-bit BGR image.

    Background pixels (depth >= far or == 0) are rendered black.
    """
    d = depth.copy()
    valid = (d > near) & (d < far)
    d[~valid] = far
    d_norm = np.clip((d - near) / max(far - near, 1e-6), 0.0, 1.0)
    d_u8   = (d_norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _has_body(spec, name: str) -> bool:
    try:
        spec.body(name)
        return True
    except Exception:
        return False


class CameraStreams:
    """Render RGB + depth for a set of cameras and show via cv2 windows.

    Usage:
        streams = CameraStreams(model, cams, show=True)
        while sim_loop:
            streams.update(data)   # reads current physics state
    """

    def __init__(self, model, cams, show: bool = True, every: int = 1):
        import cv2
        self.cv2       = cv2
        self.model     = model
        self.cams      = cams
        self.show      = show
        self.every     = every
        self._step     = 0
        self._renderer = {}
        for c in cams:
            r = mujoco.Renderer(model, height=c.height, width=c.width)
            self._renderer[c.name] = r

    def update(self, data) -> dict:
        self._step += 1
        out = {}
        for c in self.cams:
            r = self._renderer[c.name]

            r.update_scene(data, camera=c.name)
            rgb = r.render()                        # (H, W, 3) uint8

            r.enable_depth_rendering()
            r.update_scene(data, camera=c.name)
            depth = r.render()                      # (H, W) float32, meters
            r.disable_depth_rendering()

            out[c.name] = dict(rgb=rgb, depth=depth)

            if self.show and (self._step % self.every == 0):
                bgr     = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
                d_color = _depth_to_color(self.cv2, depth, c.near, c.far)
                panel   = np.concatenate([bgr, d_color], axis=1)
                self.cv2.imshow(c.name, panel)
                self.cv2.waitKey(1)

        return out

    def close(self):
        for r in self._renderer.values():
            r.close()
        if self.show:
            self.cv2.destroyAllWindows()
