import numpy as np
from .obj_fun import ObjFun
import time
import matplotlib.pyplot as plt


def GWO(UAV, SearchAgents, Max_iter, seed, is_normal=True, dynamic_g=100):
    # Set the seed for reproducibility
    np.random.seed(seed)

    dim = UAV["PointNum"] * UAV["PointDim"]

    # Initialize positions
    Positions = None
    if is_normal:
        Positions = np.random.uniform(
            low=np.tile(
                [UAV["limt"]["x"][0], UAV["limt"]["y"][0], UAV["limt"]["z"][0]],
                UAV["PointNum"],
            ),
            high=np.tile(
                [UAV["limt"]["x"][1], UAV["limt"]["y"][1], UAV["limt"]["z"][1]],
                UAV["PointNum"],
            ),
            size=(SearchAgents, dim),
        )
    else:
        Positions = np.random.uniform(
            low=np.tile(
                [UAV["S"][0], UAV["S"][1], UAV["S"][2]],
                UAV["PointNum"],
            ),
            high=np.tile(
                [UAV["G"][0], UAV["G"][1], UAV["G"][2]],
                UAV["PointNum"],
            ),
            size=(SearchAgents, dim),
        )

    # Initialize Alpha, Beta, and Delta
    Alpha_pos, Beta_pos, Delta_pos = np.zeros((3, dim))
    Alpha_score, Beta_score, Delta_score = np.full(3, np.inf)

    Fitness_list = np.zeros(Max_iter)
    all_paths = []

    # Main loop
    start_time = time.time()
    text = ""
    if is_normal:
        text = "Normal GWO"
    else:
        text = "Imporve GWO"
    print(f">>{text} Optimization in progress    00.00%", end="", flush=True)
    for iter in range(Max_iter):
        # Store current paths
        all_paths.append(
            Positions.reshape(SearchAgents, UAV["PointNum"], UAV["PointDim"]).tolist()
        )

        for i in range(SearchAgents):
            # Evaluate fitness
            fitness = ObjFun(Positions[i], UAV)

            # Used for calculating dynamic weighted average
            Alpha = Alpha_score
            Beta = Beta_score
            Delta = Delta_score
            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Alpha_score, Alpha_pos = fitness, np.copy(Positions[i])
            elif fitness < Beta_score:
                Beta_score, Beta_pos = fitness, np.copy(Positions[i])
            elif fitness < Delta_score:
                Delta_score, Delta_pos = fitness, np.copy(Positions[i])

        # Update positions
        a = 0
        if is_normal:  # Linear decrease: 2 - iter * (2/Max_iter)
            a = 2 - iter * (2 / Max_iter)
        else:  # Non-linear decrease: 2cos((iter/Max_iter)*(π/2))
            a = 2 * np.cos((iter / Max_iter) * (np.pi / 2))
        for i in range(SearchAgents):
            for j in range(dim):
                r1, r2 = np.random.rand(2)
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = np.abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(2)
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = np.abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(2)
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = np.abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                # Update position with
                if is_normal:  # static average
                    Positions[i, j] = (X1 + X2 + X3) / 3
                else:
                    g = dynamic_g  # NOTE: dynamic nubmer
                    q = g * a  # threshold for dynamic weighted average
                    if abs(Alpha - Delta) > q:  # dynamic weighted average
                        vr = Alpha + Beta + Delta
                        Positions[i, j] = (Alpha * X1 + Beta * X2 + Delta * X3) / vr
                    else:  # static average
                        Positions[i, j] = (X1 + X2 + X3) / 3

        # Save iteration image
        # save_iteration_image(iter, Positions, UAV)

        # Enforce bounds
        Positions = np.clip(
            Positions,
            np.tile(
                [UAV["limt"]["x"][0], UAV["limt"]["y"][0], UAV["limt"]["z"][0]],
                UAV["PointNum"],
            ),
            np.tile(
                [UAV["limt"]["x"][1], UAV["limt"]["y"][1], UAV["limt"]["z"][1]],
                UAV["PointNum"],
            ),
        )

        # Store best fitness
        Fitness_list[iter] = Alpha_score

        # Print progress
        progress = (iter + 1) / Max_iter * 100
        print(
            f"\r>>{text} Optimization in progress    {progress:.2f}% | Best fitness: {Alpha_score:.4f}",
            end="",
            flush=True,
        )

    print("\n\n>>Calculation complete!")
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    # Prepare output
    solution = {
        "best_path": Alpha_pos.reshape(UAV["PointNum"], UAV["PointDim"]),
        "Fitness_list": Fitness_list,
        "all_paths": all_paths,
        "seed": seed,  # Include the seed in the solution for reference
    }

    return solution


def save_iteration_image(iteration, positions, uav_setup):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    positions_reshaped = positions.reshape(
        -1, uav_setup["PointNum"], uav_setup["PointDim"]
    )
    for path in positions_reshaped:
        ax.plot(path[:, 0], path[:, 1], path[:, 2], marker="o")

    # Mark start and goal positions
    ax.scatter(
        uav_setup["S"][0],
        uav_setup["S"][1],
        uav_setup["S"][2],
        color="green",
        label="Start",
        s=100,  # Increase the size of the point
        edgecolors="k",  # Add border color
    )
    ax.scatter(
        uav_setup["G"][0],
        uav_setup["G"][1],
        uav_setup["G"][2],
        color="red",
        label="Goal",
        s=100,  # Increase the size of the point
        edgecolors="k",  # Add border color
    )

    # Draw no-fly zones
    for zone in uav_setup["NoFlyZones"]:
        x, y, height, radius = zone
        z = np.linspace(0, height, 50)
        theta = np.linspace(0, 2 * np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid) + x
        y_grid = radius * np.sin(theta_grid) + y
        ax.plot_surface(x_grid, y_grid, z_grid, color="r", alpha=0.3)

    ax.legend()
    ax.set_title(f"Iteration {iteration}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(f"images/iteration_{iteration}.png")
    plt.close()
