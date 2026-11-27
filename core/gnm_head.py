import numpy as np
import trimesh
from gnm.shape import gnm_numpy

class GNMHeadModel:
    def __init__(self, scale: float = 1.0):
        self.gnm = gnm_numpy.GNM.from_remote(
            version=gnm_numpy.GNMMajorVersion.V3,
            variant=gnm_numpy.GNMVariant.HEAD,
        )
        self.scale = scale
        self.identity = np.zeros(self.gnm.identity_dim, dtype=np.float32)
        self.expression = np.zeros(self.gnm.expression_dim, dtype=np.float32)

        self.mesh = self._build_mesh()
        self.head_width = float(self.mesh.bounds[1][0] - self.mesh.bounds[0][0])

    def _build_mesh(self) -> trimesh.Trimesh:
        vertices = self.gnm(identity=self.identity, expression=self.expression)
        vertices = np.asarray(vertices, dtype=np.float32) * self.scale

        # Center mesh roughly around the face center so it aligns with MediaPipe pose
        centroid = vertices.mean(axis=0)
        vertices -= centroid

        return trimesh.Trimesh(
            vertices=vertices,
            faces=self.gnm.triangles,
            process=False,
        )

    def update_identity(self, new_identity: np.ndarray):
        """Hook to plug in a least-squares identity solver (e.g. from gnm-webcam-puppet)."""
        self.identity = new_identity.astype(np.float32)
        self.mesh = self._build_mesh()
        self.head_width = float(self.mesh.bounds[1][0] - self.mesh.bounds[0][0])