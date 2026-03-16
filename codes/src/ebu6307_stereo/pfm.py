from __future__ import annotations

from pathlib import Path
from typing import Any


def read_pfm(path: Path) -> Any:
    import numpy as np

    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").strip()
        if header not in {"Pf", "PF"}:
            raise ValueError(f"Unsupported PFM header in {path}: {header!r}")

        dims_line = handle.readline().decode("ascii").strip()
        while dims_line.startswith("#"):
            dims_line = handle.readline().decode("ascii").strip()
        width_str, height_str = dims_line.split()
        width = int(width_str)
        height = int(height_str)

        scale = float(handle.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        channels = 3 if header == "PF" else 1
        data = np.fromfile(handle, f"{endian}f")

    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"PFM data size mismatch in {path}: expected {expected}, found {data.size}")

    if channels == 3:
        image = data.reshape((height, width, 3))
    else:
        image = data.reshape((height, width))
    return np.flipud(image.copy())


def write_pfm(path: Path, image: Any, scale: float = 1.0) -> None:
    import numpy as np

    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        header = b"Pf\n"
    elif array.ndim == 3 and array.shape[2] == 3:
        header = b"PF\n"
    else:
        raise ValueError("PFM expects an HxW grayscale image or HxWx3 color image.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(f"{array.shape[1]} {array.shape[0]}\n".encode("ascii"))
        endian_scale = -abs(scale) if array.dtype.byteorder in ("<", "=") else abs(scale)
        handle.write(f"{endian_scale}\n".encode("ascii"))
        np.flipud(array).tofile(handle)
