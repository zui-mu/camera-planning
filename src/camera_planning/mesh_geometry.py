"""Triangle-mesh visibility backend with a small dependency-free BVH.

The implementation favors auditable correctness over production throughput.  It
is suitable for pilot scenes; large assets should later use Embree/Blender BVH.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import file_hash


@dataclass(slots=True)
class _Node:
    minimum: np.ndarray
    maximum: np.ndarray
    left: int = -1
    right: int = -1
    triangles: np.ndarray | None = None


def _ray_box(origin, direction, minimum, maximum, maximum_t=1.0):
    near, far = 0.0, maximum_t
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if origin[axis] < minimum[axis] or origin[axis] > maximum[axis]:
                return False
            continue
        a = (minimum[axis] - origin[axis]) / direction[axis]
        b = (maximum[axis] - origin[axis]) / direction[axis]
        near = max(near, min(a, b))
        far = min(far, max(a, b))
        if far < near:
            return False
    return far >= 1e-8 and near < maximum_t - 1e-8


def _ray_triangle_t(origin, direction, triangles, maximum_t=1.0):
    """Möller–Trumbore segment hits. Returns parameter t in origin+t*direction."""
    if not len(triangles):
        return np.empty(0)
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, pvec)
    valid = abs(determinant) > 1e-12
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    tvec = origin - triangles[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inverse
    qvec = np.cross(tvec, edge1)
    v = qvec @ direction * inverse
    t = np.einsum("ij,ij->i", edge2, qvec) * inverse
    valid &= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    valid &= (t > 1e-8) & (t < maximum_t - 1e-8)
    return t[valid]


def _point_segment_distance(point, starts, ends):
    direction = ends - starts
    length2 = np.einsum("ij,ij->i", direction, direction)
    parameter = np.divide(
        np.einsum("ij,ij->i", point - starts, direction),
        length2,
        out=np.zeros_like(length2),
        where=length2 > 1e-18,
    )
    parameter = np.clip(parameter, 0, 1)
    closest = starts + parameter[:, None] * direction
    return np.linalg.norm(closest - point, axis=1)


def _point_triangle_distance(point, triangles):
    """Exact distance using plane projection plus all three boundary segments."""
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    normal = np.cross(edge1, edge2)
    normal2 = np.einsum("ij,ij->i", normal, normal)
    signed = np.einsum("ij,ij->i", point - triangles[:, 0], normal)
    projected = point - np.divide(
        signed[:, None] * normal,
        normal2[:, None],
        out=np.zeros_like(normal),
        where=normal2[:, None] > 1e-18,
    )
    v0 = edge1
    v1 = edge2
    v2 = projected - triangles[:, 0]
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denominator = d00 * d11 - d01 * d01
    bary_v = np.divide(
        d11 * d20 - d01 * d21,
        denominator,
        out=np.full_like(denominator, np.inf),
        where=abs(denominator) > 1e-18,
    )
    bary_w = np.divide(
        d00 * d21 - d01 * d20,
        denominator,
        out=np.full_like(denominator, np.inf),
        where=abs(denominator) > 1e-18,
    )
    inside = (bary_v >= 0) & (bary_w >= 0) & (bary_v + bary_w <= 1)
    plane_distance = np.divide(
        abs(signed),
        np.sqrt(normal2),
        out=np.full_like(signed, np.inf),
        where=normal2 > 1e-18,
    )
    boundaries = np.minimum.reduce(
        [
            _point_segment_distance(point, triangles[:, 0], triangles[:, 1]),
            _point_segment_distance(point, triangles[:, 1], triangles[:, 2]),
            _point_segment_distance(point, triangles[:, 2], triangles[:, 0]),
        ]
    )
    return np.where(inside, plane_distance, boundaries)


class TriangleMeshGeometry:
    name = "triangle_mesh_bvh_v1"

    def __init__(
        self,
        vertices,
        faces,
        face_object_indices=None,
        object_ids=None,
        solid_object_indices=None,
        leaf_size=16,
    ):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be Nx3")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3 or not len(self.faces):
            raise ValueError("faces must be nonempty Mx3")
        if np.any(self.faces < 0) or np.any(self.faces >= len(self.vertices)):
            raise ValueError("face index outside vertices")
        if not np.all(np.isfinite(self.vertices)):
            raise ValueError("non-finite mesh vertices")
        self.triangles = self.vertices[self.faces]
        area2 = np.linalg.norm(
            np.cross(
                self.triangles[:, 1] - self.triangles[:, 0],
                self.triangles[:, 2] - self.triangles[:, 0],
            ),
            axis=1,
        )
        if np.any(area2 < 1e-14):
            raise ValueError("mesh contains degenerate triangles")
        if face_object_indices is None:
            face_object_indices = np.zeros(len(self.faces), dtype=np.int64)
        self.face_object_indices = np.asarray(face_object_indices, dtype=np.int64)
        if self.face_object_indices.shape != (len(self.faces),):
            raise ValueError("face_object_indices must have one item per face")
        self.object_ids = tuple(
            str(x) for x in (object_ids if object_ids is not None else ["mesh"])
        )
        if np.any(self.face_object_indices < 0) or np.any(
            self.face_object_indices >= len(self.object_ids)
        ):
            raise ValueError("invalid face object index")
        self.solid_object_indices = frozenset(
            int(x) for x in (solid_object_indices if solid_object_indices is not None else [])
        )
        if any(x < 0 or x >= len(self.object_ids) for x in self.solid_object_indices):
            raise ValueError("invalid solid object index")
        self.leaf_size = int(leaf_size)
        self._by_object = {
            i: self.triangles[self.face_object_indices == i] for i in range(len(self.object_ids))
        }
        self.nodes = []
        self.root = self._build(np.arange(len(self.triangles)))

    @classmethod
    def from_npz(cls, path, expected_sha256=None):
        path = Path(path)
        if expected_sha256 is not None and file_hash(path) != expected_sha256:
            raise ValueError(f"geometry hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "format_version",
                "vertices",
                "faces",
                "face_object_indices",
                "object_ids",
                "solid_object_indices",
                "length_unit",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"geometry archive missing fields: {sorted(missing)}")
            if str(archive["format_version"].item()) != "camera_scene_geometry_v1":
                raise ValueError("unsupported geometry format")
            if str(archive["length_unit"].item()) != "meter":
                raise ValueError("geometry must use meters")
            return cls(
                archive["vertices"],
                archive["faces"],
                archive["face_object_indices"],
                archive["object_ids"],
                archive["solid_object_indices"],
            )

    def _build(self, indices):
        triangles = self.triangles[indices]
        node_index = len(self.nodes)
        node = _Node(triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1)))
        self.nodes.append(node)
        if len(indices) <= self.leaf_size:
            node.triangles = indices
            return node_index
        centroids = triangles.mean(axis=1)
        axis = int(np.argmax(np.ptp(centroids, axis=0)))
        order = indices[np.argsort(centroids[:, axis], kind="stable")]
        midpoint = len(order) // 2
        node.left = self._build(order[:midpoint])
        node.right = self._build(order[midpoint:])
        return node_index

    def _segment_hits(self, origin, endpoint, exclude_object_indices=None):
        direction = endpoint - origin
        if np.linalg.norm(direction) < 1e-12:
            return np.empty(0)
        stack, hits = [self.root], []
        while stack:
            node = self.nodes[stack.pop()]
            if not _ray_box(origin, direction, node.minimum, node.maximum):
                continue
            if node.triangles is not None:
                indices = node.triangles
                if exclude_object_indices:
                    indices = indices[
                        ~np.isin(self.face_object_indices[indices], list(exclude_object_indices))
                    ]
                hits.extend(_ray_triangle_t(origin, direction, self.triangles[indices]))
            else:
                stack.extend((node.left, node.right))
        return np.asarray(hits)

    def blocked(
        self,
        origin,
        endpoints,
        exclude_object_id=None,
        exclude_object_ids=None,
    ):
        origin = np.asarray(origin, dtype=float)
        names = set(exclude_object_ids or ())
        if exclude_object_id is not None:
            names.add(exclude_object_id)
        excluded = set()
        for name in names:
            try:
                excluded.add(self.object_ids.index(name))
            except ValueError:
                pass
        return np.array(
            [
                bool(len(self._segment_hits(origin, endpoint, excluded)))
                for endpoint in np.asarray(endpoints)
            ],
            dtype=bool,
        )

    def _inside_object(self, point, object_index):
        triangles = self._by_object[object_index]
        if not len(triangles):
            return False
        if np.any(point < triangles.min(axis=(0, 1))) or np.any(point > triangles.max(axis=(0, 1))):
            return False
        extent = np.ptp(triangles.reshape(-1, 3), axis=0)
        distance = max(float(np.linalg.norm(extent)) * 3, 1.0)
        direction = np.array([1.0, 0.371390676, 0.694847539])
        direction /= np.linalg.norm(direction)
        t = _ray_triangle_t(point, direction * distance, triangles)
        if not len(t):
            return False
        # Two coplanar triangles of one polygon may report the same crossing.
        unique = np.unique(np.round(t, 9))
        return len(unique) % 2 == 1

    def occupied(self, points, clearance=0.0):
        points = np.asarray(points, dtype=float)
        result = np.zeros(len(points), dtype=bool)
        for i, point in enumerate(points):
            result[i] = any(self._inside_object(point, obj) for obj in self.solid_object_indices)
            if not result[i] and clearance > 0:
                result[i] = bool(self._nearest_distance(point) <= clearance)
        return result

    def _nearest_distance(self, point, object_index=None):
        def lower(node):
            delta = np.maximum(np.maximum(node.minimum - point, point - node.maximum), 0)
            return float(np.linalg.norm(delta))

        queue, best = [(lower(self.nodes[self.root]), self.root)], np.inf
        while queue:
            bound, index = heapq.heappop(queue)
            if bound >= best:
                break
            node = self.nodes[index]
            if node.triangles is None:
                for child in (node.left, node.right):
                    distance = lower(self.nodes[child])
                    if distance < best:
                        heapq.heappush(queue, (distance, child))
            else:
                ids = node.triangles
                if object_index is not None:
                    ids = ids[self.face_object_indices[ids] == object_index]
                if len(ids):
                    best = min(
                        best, float(np.min(_point_triangle_distance(point, self.triangles[ids])))
                    )
        return best

    def _object_index(self, object_id):
        try:
            return self.object_ids.index(object_id)
        except ValueError as error:
            raise KeyError(f"unknown geometry object: {object_id}") from error

    def _object_triangles(self, object_id):
        index = self._object_index(object_id)
        triangles = self.triangles[self.face_object_indices == index]
        if not len(triangles):
            raise ValueError(f"geometry object has no triangles: {object_id}")
        return triangles

    def object_vertices(self, object_id):
        return self._object_triangles(object_id).reshape(-1, 3)

    def sample_surface(self, object_id, samples, seed, up_index, underside_normal_threshold):
        triangles = self._object_triangles(object_id)
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        area2 = np.linalg.norm(cross, axis=1)
        normals = cross / area2[:, None]
        keep = normals[:, up_index] >= underside_normal_threshold
        triangles, normals, area2 = triangles[keep], normals[keep], area2[keep]
        if not len(triangles):
            raise ValueError("underside filtering removed every target surface triangle")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(triangles), samples, p=area2 / area2.sum())
        u = rng.random(samples)
        v = rng.random(samples)
        root_u = np.sqrt(u)
        barycentric = np.column_stack((1 - root_u, root_u * (1 - v), root_u * v))
        points = np.einsum("ni,nij->nj", barycentric, triangles[chosen])
        return points, normals[chosen]

    def distance_to_object(self, points, object_id):
        index = self._object_index(object_id)
        return np.array([self._nearest_distance(point, index) for point in points])
