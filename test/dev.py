import matplotlib.pyplot as plt
import numpy as np
import pytest
from isobenefit import land_map


def test_plot(land: land_map.Land):
    """ """
    fig, axes = plt.subplots(2, 2, squeeze=True, sharex=True, sharey=True)
    axes[0][0].imshow(land.state_arr)
    axes[0][1].imshow(land.green_itx_arr)
    axes[1][0].imshow(land.cent_acc_arr)
    axes[1][1].imshow(land.green_acc_arr)
    plt.show()


def test_land_map():
    """ """
    span = 10000
    granularity_m = 50
    walk_dist_m = 1000
    cells = int(span / granularity_m)
    extents_arr = np.full((cells, cells), 0, dtype=np.int_)
    # snip out corner of extents for testing out of bounds
    for x_idx, y_idx in np.ndindex(extents_arr.shape):
        dist = np.hypot(x_idx * granularity_m, y_idx * granularity_m)
        if dist <= 1000:
            extents_arr[x_idx, y_idx] = -1
    land = land_map.Land(
        granularity_m=granularity_m,
        walk_dist_m=walk_dist_m,
        bounds=(0, 0, span, span),
        extents_arr=extents_arr,
        centre_seeds=[(1000, 1000)],
    )
    for iter in range(50):
        print(iter)
        land.iter_land_isobenefit()
        if iter % 10 == 0:
            test_plot(land)
    test_plot(land)
    print("here")


if __name__ == "__main__":
    test_land_map()
