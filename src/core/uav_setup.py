import numpy as np


def UAV_SetUp():
    UAV = {}
    UAV["S"] = np.array([2, 2, 25])  # Start position (x,y,z)
    UAV["G"] = np.array([28, 28, 5])  # End position (x,y,z)
    UAV["PointNum"] = 4  # Number of navigation points for the UAV

    UAV["PointDim"] = UAV["S"].shape[0]

    # Updated no-fly zones (x, y, height, radius)
    UAV["NoFlyZones"] = np.array(
        [
            [5, 5, 22, 2],
            [15, 7, 17, 2],
            [10, 12, 18, 2.5],
            [8, 15, 26, 2.5],
            [12, 3, 11, 2],
            [3, 10, 15, 2],
            [7, 7, 20, 2],
            [13, 9, 17, 2],
            [16, 13, 20, 2],
            [20, 20, 11, 2],
            [24, 19, 15, 2],
            [18, 22, 20, 2],
            [18, 18, 27, 2],
            [26, 26, 27, 2],
            [23, 19, 17, 2],
            [25, 26, 20, 2],
        ]
    )

    # UAV constraint settings
    UAV["limt"] = {
        "x": [0, 30],
        "y": [0, 30],
        "z": [0, 30],
    }

    # Ensure start and end positions are not inside any cylinder
    if not is_position_valid(UAV["S"], UAV["NoFlyZones"]):
        raise ValueError("Start position is inside a no-fly zone!")
    if not is_position_valid(UAV["G"], UAV["NoFlyZones"]):
        raise ValueError("Goal position is inside a no-fly zone!")

    return UAV


def is_position_valid(position, no_fly_zones):
    for zone in no_fly_zones:
        x, y, height, radius = zone
        dx = position[0] - x
        dy = position[1] - y
        distance_2d = np.sqrt(dx**2 + dy**2)
        if distance_2d <= radius and 0 <= position[2] <= height:
            return False
    return True
