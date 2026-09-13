import numpy as np
import pytest

from camera_planning.artifacts import file_hash
from camera_planning.mesh_geometry import TriangleMeshGeometry


def cube_mesh():
    vertices = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def test_triangle_mesh_segment_and_solid_queries():
    vertices, faces = cube_mesh()
    geometry = TriangleMeshGeometry(
        vertices,
        faces,
        face_object_indices=np.zeros(len(faces), dtype=np.int64),
        object_ids=["cube"],
        solid_object_indices=[0],
        leaf_size=2,
    )
    assert geometry.blocked(np.array([-2.0, 0, 0]), np.array([[2.0, 0, 0]])).tolist() == [True]
    assert geometry.blocked(
        np.array([-2.0, 0, 0]),
        np.array([[2.0, 0, 0]]),
        exclude_object_id="cube",
    ).tolist() == [False]
    assert geometry.blocked(np.array([-2.0, 2, 0]), np.array([[2.0, 2, 0]])).tolist() == [False]
    assert geometry.blocked(np.array([-2.0, 0, 0]), np.array([[-1.0, 0, 0]])).tolist() == [False]
    assert geometry.occupied(np.array([[0, 0, 0], [2, 0, 0]])).tolist() == [True, False]
    assert geometry.occupied(np.array([[1.05, 0, 0]]), clearance=0.1).tolist() == [True]
    assert geometry.occupied(np.array([[1.2, 0, 0]]), clearance=0.1).tolist() == [False]


def test_triangle_archive_contract_and_hash(tmp_path):
    vertices, faces = cube_mesh()
    path = tmp_path / "geometry.npz"
    np.savez_compressed(
        path,
        format_version="camera_scene_geometry_v1",
        length_unit="meter",
        vertices=vertices,
        faces=faces,
        face_object_indices=np.zeros(len(faces), dtype=np.int64),
        object_ids=np.array(["cube"]),
        solid_object_indices=np.array([0], dtype=np.int64),
    )
    loaded = TriangleMeshGeometry.from_npz(path, file_hash(path))
    assert loaded.object_ids == ("cube",)
    with pytest.raises(ValueError, match="hash mismatch"):
        TriangleMeshGeometry.from_npz(path, "0" * 64)


def test_rejects_degenerate_triangles():
    with pytest.raises(ValueError, match="degenerate"):
        TriangleMeshGeometry(np.zeros((3, 3)), np.array([[0, 1, 2]]))
