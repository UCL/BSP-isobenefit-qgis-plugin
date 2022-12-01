from __future__ import annotations

import numpy as np


# TODO: this function doesn't appear to be used?
def get_circular_coords(radius: int, center_x: int, center_y: int, n_amenities: int) -> list[tuple[int, int]]:
    """ """
    return [
        (
            center_x + int(radius * np.sin(2 * t * np.pi)),
            center_y + int(radius * np.cos(2 * t * np.pi)),
        )
        for t in np.linspace(0, 1, n_amenities)
    ]


# TODO: this function doesn't appear to be used?
def get_random_coordinates(size_x: int, size_y: int, n_amenities: int, seed: int = 42) -> list[tuple[int, int]]:
    """ """
    np.random.seed(seed)
    return [(int(size_x * np.random.random()), int(size_y * np.random.random())) for _ in range(n_amenities)]


def get_central_coord(size_x: int, size_y: int) -> list[tuple[int, int]]:
    """Returns centre coordinates given the x, y size."""
    return [(int(size_x / 2), int(size_y / 2))]
