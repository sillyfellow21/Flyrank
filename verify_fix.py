# Verification script for MONAI issue #9184
# Ensures EnsureChannelFirst(channel_dim="no_channel") keeps spatial_ndim in sync
# with the affine, so downstream spatial crops preserve all affine offsets.
import numpy as np
import sys

sys.path.insert(0, r"e:\Bdtorrent\CHikni\monai")

from monai.data import MetaTensor
from monai.transforms import EnsureChannelFirstd, SpatialCropd

FAILED = False


def check(name, actual, expected):
    global FAILED
    status = "PASS" if actual == expected else "FAIL"
    if actual != expected:
        FAILED = True
    print(f"[{status}] {name}: got {actual}, expected {expected}")


# 3D case from the issue
affine = np.eye(4)
affine[:3, 3] = [10, 20, 30]
d = {"img": MetaTensor(np.zeros((40, 40, 40), dtype=np.float32), affine=affine)}
d = EnsureChannelFirstd("img", channel_dim="no_channel")(d)
check("3D spatial_ndim after EnsureChannelFirst", d["img"].spatial_ndim, 3)
d = SpatialCropd("img", roi_start=[5, 5, 5], roi_end=[20, 20, 20])(d)
check("3D crop affine[:3,3]", d["img"].affine[:3, 3].tolist(), [15, 25, 35])

# 2D case (same class of bug, mentioned in the issue)
affine = np.eye(3)
affine[:2, 2] = [10, 20]
d = {"img": MetaTensor(np.zeros((40, 40), dtype=np.float32), affine=affine)}
d = EnsureChannelFirstd("img", channel_dim="no_channel")(d)
check("2D spatial_ndim after EnsureChannelFirst", d["img"].spatial_ndim, 2)
d = SpatialCropd("img", roi_start=[5, 5], roi_end=[20, 20])(d)
check("2D crop affine[:2,2]", d["img"].affine[:2, 2].tolist(), [15, 25])

print()
print("ALL CHECKS PASSED" if not FAILED else "SOME CHECKS FAILED")
sys.exit(1 if FAILED else 0)
