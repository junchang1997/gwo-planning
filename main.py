import numpy as np
import random
from src.core.gwo import GWO
from src.core.uav_setup import UAV_SetUp
from src.core.obj_fun import ObjFun
from src.utils.export import export_animation_data
from src.visualization.animation import PathAnimator, save_animation


def set_seed(seed=None):
    """Set random seed for reproducibility."""
    if seed is None:
        seed = random.randint(0, 100000)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def main():
    # Algorithm parameters
    SearchAgents = 20
    Max_iter = 200

    # Setup
    UAV = UAV_SetUp()
    seed = set_seed(None)  # Change to an integer for a fixed seed
    # print(f"Using seed: {seed}")

    is_normal = True
    # Run optimization
    solution = GWO(UAV, SearchAgents, Max_iter, seed, is_normal=is_normal)

    # Export data
    animation_filename = "normal_gwo.json" if is_normal else "imporve_gwo.json"
    export_animation_data(solution, UAV, filename=animation_filename)

    # Create and save animation
    animator = PathAnimator(UAV, ObjFun)
    animation, total_frames = animator.create_animation(solution["all_paths"])
    video_filename = "normal_gwo.mp4" if is_normal else "imporve_gwo.mp4"
    save_animation(animation, total_frames, filename=video_filename)

    is_normal = False
    # Run optimization
    solution = GWO(
        UAV, SearchAgents, Max_iter, seed, is_normal=is_normal, dynamic_g=50
    )

    # Export data
    animation_filename = "normal_gwo.json" if is_normal else "imporve_gwo.json"
    export_animation_data(solution, UAV, filename=animation_filename)

    # Create and save animation
    animator = PathAnimator(UAV, ObjFun)
    animation, total_frames = animator.create_animation(solution["all_paths"])
    video_filename = "normal_gwo.mp4" if is_normal else "imporve_gwo.mp4"
    save_animation(animation, total_frames, filename=video_filename)

    # print(f"Optimization completed and animation saved. Seed used: {seed}")


if __name__ == "__main__":
    main()
