"""Blender-free I/O for DeformingThings4D `.anime` files.

`anime_renderer.py` defines `anime_read`, but it does `import bpy` at module
top, so it cannot be imported outside Blender. This module extracts the pure
(numpy-only) reader plus a small convenience that reconstructs full per-frame
vertex positions, so the parser can be reused from any Python environment.

Vertex topology is fixed across frames, so vertex index == ground-truth
correspondence between any two frames of a sequence.
"""
from __future__ import annotations

import numpy as np


def anime_read(filename):
    """Read a `.anime` file.

    Returns:
        nf:           number of frames in the animation
        nv:           number of vertices (topology fixed across frames)
        nt:           number of triangle faces
        vert_data:    (nv, 3) float32 vertex positions of the 1st frame
        face_data:    (nt, 3) int32 triangle indices
        offset_data:  (nf-1, nv, 3) float32 vertex offsets for frames 2..nf,
                      each measured relative to frame 1.
    """
    with open(filename, "rb") as f:
        nf = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        nv = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        nt = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
        face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
        offset_data = np.fromfile(f, dtype=np.float32, count=-1)

    if len(offset_data) != (nf - 1) * nv * 3:
        raise ValueError(f"data inconsistent error: {filename}")

    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data


def load_anime(filename):
    """Load a `.anime` file as full per-frame vertex positions.

    Returns:
        verts_seq: (nf, nv, 3) float32 absolute vertex positions per frame
        faces:     (nt, 3) int64 triangle indices
    """
    nf, nv, nt, vert0, faces, offsets = anime_read(filename)
    # offsets are relative to frame 0; prepend a zero offset for frame 0.
    offsets_full = np.concatenate(
        [np.zeros((1, nv, 3), dtype=np.float32), offsets], axis=0
    )  # (nf, nv, 3)
    verts_seq = (vert0[None] + offsets_full).astype(np.float32)
    return verts_seq, faces.astype(np.int64)
