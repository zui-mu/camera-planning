"""OpenCV projection and replaceable geometry queries; world coordinates are meters."""

from typing import Protocol

import numpy as np

from .contracts import Box, Camera, Intrinsics


def unit(x):
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        raise ValueError("undefined direction")
    return x / norm


def look_at(center, focus, up, intrinsics: Intrinsics, camera_id: str) -> Camera:
    forward = unit(np.asarray(focus) - center)
    right = unit(np.cross(forward, up))
    down = unit(np.cross(forward, right))
    r = np.stack([right, down, forward])
    return Camera(
        camera_id=camera_id,
        width=intrinsics.width,
        height=intrinsics.height,
        K=intrinsics.matrix().tolist(),
        R=r.tolist(),
        T=(-r @ center).tolist(),
    )


def project(camera: Camera, points):
    xyz = np.asarray(points) @ np.asarray(camera.R).T + camera.T
    homogeneous = xyz @ np.asarray(camera.K).T
    uv = np.full((len(xyz), 2), np.nan)
    np.divide(
        homogeneous[:, :2],
        homogeneous[:, 2, None],
        out=uv,
        where=np.abs(homogeneous[:, 2, None]) > 1e-12,
    )
    return uv, xyz[:, 2]


def projection_jacobian(camera: Camera, points):
    """d(pixel coordinates)/d(world point), shape N x 2 x 3; not neural Fisher."""
    xyz = np.asarray(points) @ np.asarray(camera.R).T + camera.T
    k = np.asarray(camera.K)
    homogeneous = xyz @ k.T
    denominator = np.where(np.abs(homogeneous[:, 2]) > 1e-8, homogeneous[:, 2], 1e-8)
    jac = (
        k[None, :2, :] * denominator[:, None, None] - homogeneous[:, :2, None] * k[None, 2:3, :]
    ) / denominator[:, None, None] ** 2
    return jac @ np.asarray(camera.R)


def blender_camera_matrix(camera: Camera):
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(camera.R).T @ np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = camera.center
    return matrix


def inside_box(points, box: Box, margin=0.0):
    return np.all(
        (np.asarray(points) >= np.array(box.minimum) - margin)
        & (np.asarray(points) <= np.array(box.maximum) + margin),
        axis=-1,
    )


class GeometryBackend(Protocol):
    name: str

    def blocked(
        self,
        origin: np.ndarray,
        endpoints: np.ndarray,
        exclude_object_id: str | None = None,
        exclude_object_ids: set[str] | frozenset[str] | None = None,
    ) -> np.ndarray: ...
    def occupied(self, points: np.ndarray, clearance: float = 0.0) -> np.ndarray: ...
    def object_vertices(self, object_id: str) -> np.ndarray: ...
    def sample_surface(
        self,
        object_id: str,
        samples: int,
        seed: int,
        up_index: int,
        underside_normal_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def distance_to_object(self, points: np.ndarray, object_id: str) -> np.ndarray: ...


class AABBGeometry:
    """Conservative solid-box proxy: use decomposed boxes, never a whole-room solid box."""

    name = "solid_aabb_proxy_v1"

    def __init__(self, boxes: list[Box]):
        self.boxes = boxes
        self._lows = np.asarray([box.minimum for box in boxes], dtype=float).reshape(-1, 3)
        self._highs = np.asarray([box.maximum for box in boxes], dtype=float).reshape(-1, 3)

    def _box(self, object_id):
        for box in self.boxes:
            if box.object_id == object_id:
                return box
        raise KeyError(f"unknown geometry object: {object_id}")

    def object_vertices(self, object_id):
        box = self._box(object_id)
        low, high = np.asarray(box.minimum), np.asarray(box.maximum)
        return np.array(
            [
                [x, y, z]
                for x in (low[0], high[0])
                for y in (low[1], high[1])
                for z in (low[2], high[2])
            ],
            dtype=float,
        )

    def sample_surface(self, object_id, samples, seed, up_index, underside_normal_threshold):
        box = self._box(object_id)
        low, high = np.asarray(box.minimum), np.asarray(box.maximum)
        extent = high - low
        faces = []
        for axis in range(3):
            other = [index for index in range(3) if index != axis]
            area = extent[other[0]] * extent[other[1]]
            for side, normal_sign in ((low[axis], -1.0), (high[axis], 1.0)):
                normal = np.zeros(3)
                normal[axis] = normal_sign
                if normal[up_index] < underside_normal_threshold:
                    continue
                faces.append((axis, side, other, area, normal))
        weights = np.asarray([face[3] for face in faces], dtype=float)
        weights /= weights.sum()
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(faces), size=samples, p=weights)
        points = np.empty((samples, 3), dtype=float)
        normals = np.empty((samples, 3), dtype=float)
        for i, face_index in enumerate(selected):
            axis, side, other, _, normal = faces[face_index]
            points[i] = rng.uniform(low, high)
            points[i, axis] = side
            normals[i] = normal
        return points, normals

    def distance_to_object(self, points, object_id):
        box = self._box(object_id)
        points = np.asarray(points, dtype=float)
        low, high = np.asarray(box.minimum), np.asarray(box.maximum)
        outside = np.maximum(np.maximum(low - points, points - high), 0)
        return np.linalg.norm(outside, axis=1)

    def occupied(self, points, clearance=0.0):
        points = np.asarray(points, float)
        result = np.zeros(len(points), dtype=bool)
        for start in range(0, len(points), 1024):
            batch = points[start : start + 1024, None, :]
            result[start : start + len(batch)] = np.any(
                np.all(
                    (batch >= self._lows - clearance) & (batch <= self._highs + clearance),
                    axis=2,
                ),
                axis=1,
            )
        return result

    def blocked(
        self,
        origin,
        endpoints,
        exclude_object_id=None,
        exclude_object_ids=None,
    ):
        excluded = set(exclude_object_ids or ())
        if exclude_object_id is not None:
            excluded.add(exclude_object_id)
        if not excluded:
            lows, highs = self._lows, self._highs
        else:
            keep = np.asarray([box.object_id not in excluded for box in self.boxes])
            lows, highs = self._lows[keep], self._highs[keep]
            if not len(lows):
                return np.zeros(len(endpoints), dtype=bool)
        directions = np.asarray(endpoints) - origin
        result = np.zeros(len(endpoints), dtype=bool)
        for start in range(0, len(endpoints), 1024):
            batch = directions[start : start + 1024, None, :]
            parallel = abs(batch) < 1e-12
            possible = np.all(
                ~parallel | ((origin >= lows) & (origin <= highs)), axis=2
            )
            denominator = np.where(parallel, 1.0, batch)
            a = (lows - origin) / denominator
            b = (highs - origin) / denominator
            near = np.max(np.where(parallel, -np.inf, np.minimum(a, b)), axis=2)
            far = np.min(np.where(parallel, np.inf, np.maximum(a, b)), axis=2)
            result[start : start + len(batch)] = np.any(
                possible & (far >= np.maximum(near, 1e-7)) & (near < 1 - 1e-7) & (far > 1e-7),
                axis=1,
            )
        return result


def load_geometry(request):
    if request.geometry.backend == "aabb":
        return AABBGeometry(
            [request.target, *request.supported_objects, *request.obstacles]
        )
    from .mesh_geometry import TriangleMeshGeometry

    return TriangleMeshGeometry.from_npz(
        request.geometry.mesh_file,
        expected_sha256=request.geometry.sha256,
    )
