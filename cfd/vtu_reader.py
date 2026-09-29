"""Minimal, dependency-light VTU reader for OpenFOAM's foamToVTK 'binary' (base64,
UInt64-length-prefixed) format. Built because meshio's VTU parser failed
unpredictably across different mesh sizes on this system's OpenFOAM output,
and a custom parser is easy to verify correct for the one thing we need:
cell-centre coordinates and the cell-centred U field."""
import base64
import numpy as np
import xml.etree.ElementTree as ET

_TYPE_MAP = {"Float32": np.float32, "Float64": np.float64, "Int32": np.int32,
             "Int64": np.int64, "UInt8": np.uint8}


def _decode(data_array_elem):
    dtype = _TYPE_MAP[data_array_elem.get("type")]
    ncomp = int(data_array_elem.get("NumberOfComponents", "1"))
    raw = base64.b64decode(data_array_elem.text.strip())
    nbytes = int(np.frombuffer(raw[:8], dtype=np.uint64)[0])  # UInt64 length prefix
    arr = np.frombuffer(raw[8:8 + nbytes], dtype=dtype)
    if ncomp > 1:
        arr = arr.reshape(-1, ncomp)
    return arr


def read_cell_centers_and_U(vtu_path):
    """Returns (cell_centers (N,3), U (N,3)) for an OpenFOAM-written internal.vtu."""
    tree = ET.parse(vtu_path)
    root = tree.getroot()
    piece = root.find(".//Piece")
    points_da = piece.find("Points/DataArray")
    points = _decode(points_da)

    cells_elem = piece.find("Cells")
    conn = offs = None
    for da in cells_elem.findall("DataArray"):
        if da.get("Name") == "connectivity":
            conn = _decode(da)
        elif da.get("Name") == "offsets":
            offs = _decode(da)
    # cell centre = mean of the points referenced by each cell (works for any cell shape)
    starts = np.concatenate([[0], offs[:-1]])
    centers = np.empty((len(offs), 3), dtype=np.float64)
    for i, (s, e) in enumerate(zip(starts, offs)):
        centers[i] = points[conn[s:e]].mean(axis=0)

    U = None
    celldata = piece.find("CellData")
    if celldata is not None:
        for da in celldata.findall("DataArray"):
            if da.get("Name") == "U":
                U = _decode(da)
    return centers, U
