import numpy as np


def polygon_area(points: np.ndarray) -> np.ndarray:
    """Compute the signed area of a polygon.

    Parameters
    ----------
    points : (..., n, 2)
        Row major collections of n-point polygons.

    Returns
    -------
    The signed area of all of the polygons.
    """
    px, py = np.moveaxis(points, -1, 0)
    ox, oy = np.moveaxis(np.roll(points, -1, -2), -1, 0)
    cross = px * oy - py * ox
    return cross.sum(-1) / 2
