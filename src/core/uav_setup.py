import numpy as np


def UAV_SetUp():
    UAV = {}
    UAV["S"] = np.array([10, 10, 25])  # Start position (x,y,z)
    UAV["G"] = np.array([500, 500, 300])  # End position (x,y,z)
    UAV["PointNum"] = 10  # Number of navigation points for the UAV

    UAV["PointDim"] = UAV["S"].shape[0]

    # Updated no-fly zones (x, y, height, radius)
    UAV["NoFlyZones"] = np.array(
        [
            [20, 50, 30, 10],
            [40, 30, 30, 10],
            [65, 20, 20, 10],
            [80, 80, 60, 10],
            [70, 75, 99, 10],
            [95, 88, 50, 10],
            [100, 110, 120, 10],
            [120, 150, 100, 10],
            [125, 140, 50, 10],
            [140, 130, 120, 10],
            [180, 180, 150, 10],
            [200, 190, 50, 10],
            [250, 230, 120, 10],
            [270, 233, 450, 10],
            [299, 320, 500, 10],
            [323, 320, 466, 10],
            [340, 340, 470, 10],
            [380, 360, 500, 10],
            [340, 340, 480, 10],
            [380, 360, 400, 10],
            [380, 360, 410, 10],
            [440, 440, 480, 10],
            [480, 460, 500, 10],
            [480, 460, 500, 10],
        ]
    )

    # UAV constraint settings
    UAV["limt"] = {
        "x": [0, 1000],
        "y": [0, 1000],
        "z": [0, 500],
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
