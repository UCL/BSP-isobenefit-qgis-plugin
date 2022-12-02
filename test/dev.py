import numpy as np
import pytest
from isobenefit import land_map


def test_land_map():
    """ """
    span = 2000
    granularity_m = 20
    walk_dist_m = 1000
    cells = int(span / granularity_m)
    land = land_map.Land(
        granularity_m=granularity_m,
        walk_dist_m=walk_dist_m,
        bounds=(0, 0, span, span),
        extents_arr=np.full((cells, cells), 0, dtype=np.int_),
        centre_seeds=[(1000, 1000)],
    )


if __name__ == "__main__":
    test_land_map()
