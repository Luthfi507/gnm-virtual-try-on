import numpy as np
import trimesh
import pyrender


def create_procedural_glasses(width: float = 0.145) -> trimesh.Trimesh:
    """Fallback 3D glasses mesh (frame + bridge + temples) if no .glb asset is provided."""
    parts = []
    lens_r = width * 0.21
    bridge_w = width * 0.14
    temple_len = width * 0.85

    # Left & Right rims
    for sign in (-1.0, 1.0):
        rim = trimesh.creation.torus(major_radius=lens_r, minor_radius=0.004)
        rim.apply_translation([sign * (lens_r + bridge_w / 2), 0.0, 0.0])
        parts.append(rim)

        # Temples (gagang kacamata mengarah ke -Z menuju telinga)
        temple = trimesh.creation.box(extents=[0.005, 0.006, temple_len])
        temple.apply_translation([
            sign * (2 * lens_r + bridge_w / 2),
            0.005,
            -temple_len / 2,
        ])
        parts.append(temple)

    # Nose bridge
    bridge = trimesh.creation.box(extents=[bridge_w, 0.005, 0.005])
    bridge.apply_translation([0.0, 0.004, 0.0])
    parts.append(bridge)

    glasses = trimesh.util.concatenate(parts)
    glasses.visual.vertex_colors = [25, 25, 30, 255]  # Matte dark frame
    return glasses


class GlassesVTORenderer:
    def __init__(
        self,
        width: int,
        height: int,
        fov_y: float,
        head_trimesh: trimesh.Trimesh,
        glasses_trimesh: trimesh.Trimesh,
        glasses_offset: list[float],
        glasses_scale: float = 1.0,
        occlusion_eps: float = 0.004,
    ):
        self.width = width
        self.height = height
        self.occlusion_eps = occlusion_eps
        self.renderer = pyrender.OffscreenRenderer(width, height)

        camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=width / height)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

        # Precompute local glasses transform relative to head
        self.glasses_local_tf = np.eye(4, dtype=np.float32)
        self.glasses_local_tf[:3, :3] *= glasses_scale
        self.glasses_local_tf[:3, 3] = glasses_offset

        # Scene 1: Head Occluder only (depth pass)
        self.head_scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
        self.head_scene.add(camera, pose=np.eye(4))
        self.head_node = self.head_scene.add(pyrender.Mesh.from_trimesh(head_trimesh))

        # Scene 2: Glasses only (RGBA + depth pass)
        self.glasses_scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.4, 0.4, 0.4])
        self.glasses_scene.add(camera, pose=np.eye(4))
        self.glasses_scene.add(light, pose=np.eye(4))
        self.glasses_node = self.glasses_scene.add(pyrender.Mesh.from_trimesh(glasses_trimesh))

    def render_overlay(self, frame_bgr: np.ndarray, head_pose: np.ndarray) -> np.ndarray:
        """Renders 3D glasses occluded by GNM head and composites onto the OpenCV frame."""
        self.head_scene.set_pose(self.head_node, pose=head_pose)
        self.glasses_scene.set_pose(self.glasses_node, pose=head_pose @ self.glasses_local_tf)

        # Pass 1: Depth of GNM head
        _, head_depth = self.renderer.render(self.head_scene)

        # Pass 2: RGBA and depth of 3D glasses
        glasses_rgba, glasses_depth = self.renderer.render(
            self.glasses_scene, flags=pyrender.RenderFlags.RGBA
        )

        # Depth test: keep glasses pixels that are in front of the GNM head mesh
        has_glasses = glasses_depth > 0.0
        not_occluded = (head_depth == 0.0) | (glasses_depth <= head_depth + self.occlusion_eps)
        visible_mask = has_glasses & not_occluded

        if not np.any(visible_mask):
            return frame_bgr

        # Alpha-blend RGB (convert RGBA -> BGR for OpenCV)
        out = frame_bgr.copy()
        alpha = (glasses_rgba[:, :, 3:4].astype(np.float32) / 255.0) * visible_mask[:, :, None]
        glasses_bgr = glasses_rgba[:, :, :3][:, :, ::-1].astype(np.float32)

        blended = alpha * glasses_bgr + (1.0 - alpha) * out.astype(np.float32)
        return blended.astype(np.uint8)

    def close(self):
        self.renderer.delete()