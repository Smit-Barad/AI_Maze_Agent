import json
import numpy as np


# ============================================================
# LOAD TRAINING HISTORY
# ============================================================

with open(
    "training_history.json",
    "r"
) as file:

    history = json.load(file)


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW = 25


# ============================================================
# EXTRACT DATA
# ============================================================

episodes = []

rewards = []

steps = []

successes = []

epsilons = []


for episode in history:

    episodes.append(
        episode["episode"]
    )

    rewards.append(
        episode["total_reward"]
    )

    steps.append(
        len(episode["steps"])
    )

    successes.append(
        1 if episode["success"] else 0
    )

    epsilons.append(
        episode["epsilon"]
    )


# ============================================================
# MOVING AVERAGES
# ============================================================

reward_average = []

step_average = []

success_average = []


for i in range(
    len(history)
):

    start = max(
        0,
        i - WINDOW + 1
    )

    reward_window = rewards[
        start:i + 1
    ]

    step_window = steps[
        start:i + 1
    ]

    success_window = successes[
        start:i + 1
    ]


    reward_average.append(
        np.mean(
            reward_window
        )
    )

    step_average.append(
        np.mean(
            step_window
        )
    )

    success_average.append(
        np.mean(
            success_window
        ) * 100
    )


# ============================================================
# PRINT HEADER
# ============================================================

print()

print("=" * 70)

print(
    "             AI LEARNING ANALYSIS"
)

print("=" * 70)

print()


# ============================================================
# FIRST SUCCESS
# ============================================================

first_success_index = None


for i, success in enumerate(
    successes
):

    if success == 1:

        first_success_index = i

        break


if first_success_index is not None:

    print(
        "FIRST SUCCESS"
    )

    print(
        f"Episode : "
        f"{episodes[first_success_index]}"
    )

    print(
        f"Steps   : "
        f"{steps[first_success_index]}"
    )

    print(
        f"Reward  : "
        f"{rewards[first_success_index]}"
    )

    print()


# ============================================================
# FIRST OPTIMAL SOLUTION
# ============================================================

optimal_index = None


for i in range(
    len(history)
):

    if (
        successes[i] == 1
        and steps[i] == 10
    ):

        optimal_index = i

        break


if optimal_index is not None:

    print(
        "FIRST OPTIMAL SOLUTION"
    )

    print(
        f"Episode : "
        f"{episodes[optimal_index]}"
    )

    print(
        f"Steps   : "
        f"{steps[optimal_index]}"
    )

    print(
        f"Reward  : "
        f"{rewards[optimal_index]}"
    )

    print()


# ============================================================
# FIND STABLE LEARNING POINT
# ============================================================

stable_index = None


for i in range(
    WINDOW,
    len(history)
):

    recent_success = successes[
        i - WINDOW:i
    ]

    recent_steps = steps[
        i - WINDOW:i
    ]


    success_rate = (
        np.mean(
            recent_success
        )
        * 100
    )


    average_steps = np.mean(
        recent_steps
    )


    # We define "stable" as:
    #
    # > 95% recent success
    # AND
    # average path <= 12 steps

    if (
        success_rate >= 95
        and average_steps <= 12
    ):

        stable_index = i

        break


if stable_index is not None:

    print(
        "STABLE LEARNING POINT"
    )

    print(
        f"Episode : "
        f"{episodes[stable_index]}"
    )

    print(
        f"Recent success rate : "
        f"{success_average[stable_index]:.1f}%"
    )

    print(
        f"Recent average steps : "
        f"{step_average[stable_index]:.2f}"
    )

    print()


# ============================================================
# FINAL PERFORMANCE
# ============================================================

print(
    "FINAL PERFORMANCE"
)

print(
    f"Success rate : "
    f"{np.mean(successes) * 100:.2f}%"
)

print(
    f"Average reward : "
    f"{np.mean(rewards):.2f}"
)

print(
    f"Average steps : "
    f"{np.mean(steps):.2f}"
)

print(
    f"Final epsilon : "
    f"{epsilons[-1]:.3f}"
)

print()


# ============================================================
# VIDEO MILESTONES
# ============================================================

print("=" * 70)

print(
    "             🎬 VIDEO MILESTONES"
)

print("=" * 70)

print()


if first_success_index is not None:

    print(
        f"[NORMAL SPEED] "
        f"Episode {episodes[first_success_index]}"
    )

    print(
        "First successful goal discovery"
    )

    print()


if optimal_index is not None:

    print(
        f"[NORMAL SPEED] "
        f"Episode {episodes[optimal_index]}"
    )

    print(
        "First 10-step optimal solution"
    )

    print()


if stable_index is not None:

    print(
        f"[NORMAL SPEED] "
        f"Episode {episodes[stable_index]}"
    )

    print(
        "Stable learned behavior"
    )

    print()


print(
    "[FAST SPEED] "
    "Everything between major milestones"
)

print()


# ============================================================
# DONE
# ============================================================

print("=" * 70)

print(
    "             ANALYSIS COMPLETE"
)

print("=" * 70)

print()