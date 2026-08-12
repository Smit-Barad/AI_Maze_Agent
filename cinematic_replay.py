import json
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_FILE = "training_history.json"

# ------------------------------------------------------------
# Replay speeds
# ------------------------------------------------------------

FAST_DELAY = 0.015

NORMAL_DELAY = 0.18

FINAL_DELAY = 0.25

EPISODE_PAUSE = 0.05


# ------------------------------------------------------------
# Your actual learning milestones
# ------------------------------------------------------------

FIRST_SUCCESS_EPISODE = 4

FIRST_OPTIMAL_EPISODE = 156

STABLE_EPISODE = 324

FINAL_EPISODE = 1000


# ============================================================
# LOAD HISTORY
# ============================================================

with open(
    HISTORY_FILE,
    "r"
) as file:

    history = json.load(file)


print()
print("=" * 70)
print("           AI MAZE AGENT — CINEMATIC REPLAY")
print("=" * 70)
print()

print(
    f"Loaded {len(history)} episodes."
)

print()

print(
    "Milestones:"
)

print(
    "  Episode 4   → First successful discovery"
)

print(
    "  Episode 156 → First 10-step optimal solution"
)

print(
    "  Episode 324 → Stable learned behavior"
)

print(
    "  Episode 1000 → Final performance"
)

print()


# ============================================================
# MAZE
# ============================================================

MAZE = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
]


ROWS = len(MAZE)

COLS = len(MAZE[0])


START = (0, 0)

GOAL = (5, 5)


# ============================================================
# ACTION NAMES
# ============================================================

ACTION_NAMES = {

    0: "UP",

    1: "DOWN",

    2: "LEFT",

    3: "RIGHT"

}


# ============================================================
# FIGURE
# ============================================================

plt.ion()


fig = plt.figure(
    figsize=(14, 8)
)


# ============================================================
# AXES
# ============================================================

ax_maze = fig.add_axes(
    [0.04, 0.20, 0.43, 0.68]
)


ax_graph = fig.add_axes(
    [0.54, 0.55, 0.42, 0.32]
)


ax_info = fig.add_axes(
    [0.54, 0.20, 0.42, 0.25]
)


ax_info.axis(
    "off"
)


# ============================================================
# DRAW MAZE
# ============================================================

def draw_maze(
    agent_position,
    path
):

    ax_maze.clear()


    # --------------------------------------------------------
    # Walls
    # --------------------------------------------------------

    for row in range(ROWS):

        for col in range(COLS):

            if MAZE[row][col] == 1:

                ax_maze.add_patch(

                    Rectangle(

                        (
                            col,
                            ROWS - row - 1
                        ),

                        1,

                        1

                    )

                )


    # --------------------------------------------------------
    # Path
    # --------------------------------------------------------

    for position in path:

        row, col = position

        ax_maze.plot(

            col + 0.5,

            ROWS - row - 0.5,

            marker=".",

            markersize=5

        )


    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    start_row, start_col = START

    ax_maze.text(

        start_col + 0.5,

        ROWS - start_row - 0.5,

        "S",

        ha="center",

        va="center",

        fontsize=18

    )


    # --------------------------------------------------------
    # Goal
    # --------------------------------------------------------

    goal_row, goal_col = GOAL

    ax_maze.text(

        goal_col + 0.5,

        ROWS - goal_row - 0.5,

        "★",

        ha="center",

        va="center",

        fontsize=22

    )


    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    row, col = agent_position

    ax_maze.text(

        col + 0.5,

        ROWS - row - 0.5,

        "A",

        ha="center",

        va="center",

        fontsize=20

    )


    # --------------------------------------------------------
    # Maze settings
    # --------------------------------------------------------

    ax_maze.set_xlim(
        0,
        COLS
    )

    ax_maze.set_ylim(
        0,
        ROWS
    )


    ax_maze.set_xticks(
        range(COLS + 1)
    )

    ax_maze.set_yticks(
        range(ROWS + 1)
    )


    ax_maze.grid(
        True
    )


    ax_maze.set_aspect(
        "equal"
    )


    ax_maze.set_title(
        "AGENT NAVIGATION",
        fontsize=14
    )


# ============================================================
# DRAW LEARNING GRAPH
# ============================================================

def draw_graph(
    current_episode
):

    ax_graph.clear()


    episode_numbers = []

    rewards = []


    for episode in history:

        if episode["episode"] > current_episode:

            break


        episode_numbers.append(

            episode["episode"]

        )


        rewards.append(

            episode["total_reward"]

        )


    ax_graph.plot(

        episode_numbers,

        rewards

    )


    # --------------------------------------------------------
    # Milestone markers
    # --------------------------------------------------------

    if current_episode >= FIRST_SUCCESS_EPISODE:

        ax_graph.axvline(

            FIRST_SUCCESS_EPISODE,

            linestyle="--",

            alpha=0.5

        )


    if current_episode >= FIRST_OPTIMAL_EPISODE:

        ax_graph.axvline(

            FIRST_OPTIMAL_EPISODE,

            linestyle="--",

            alpha=0.5

        )


    if current_episode >= STABLE_EPISODE:

        ax_graph.axvline(

            STABLE_EPISODE,

            linestyle="--",

            alpha=0.5

        )


    ax_graph.set_title(

        "LEARNING CURVE",

        fontsize=14

    )


    ax_graph.set_xlabel(

        "Episode"

    )


    ax_graph.set_ylabel(

        "Reward"

    )


    ax_graph.grid(
        True
    )


    ax_graph.set_xlim(

        1,

        len(history)

    )


# ============================================================
# MILESTONE TEXT
# ============================================================

def get_milestone_text(
    episode_number
):

    if episode_number == FIRST_SUCCESS_EPISODE:

        return (

            "FIRST SUCCESS\n"

            "The agent discovered that the goal is reachable."

        )


    if episode_number == FIRST_OPTIMAL_EPISODE:

        return (

            "BREAKTHROUGH\n"

            "The agent discovered the 10-step optimal route."

        )


    if episode_number == STABLE_EPISODE:

        return (

            "STABLE POLICY\n"

            "The agent can now repeatedly solve the maze efficiently."

        )


    if episode_number == FINAL_EPISODE:

        return (

            "LEARNING COMPLETE\n"

            "The learned policy has converged to stable behavior."

        )


    return ""


# ============================================================
# DRAW INFORMATION
# ============================================================

def draw_information(

    episode,

    step_number,

    total_steps,

    action,

    reward

):

    ax_info.clear()

    ax_info.axis(
        "off"
    )


    episode_number = episode["episode"]

    total_reward = episode["total_reward"]

    epsilon = episode["epsilon"]

    success = episode["success"]


    milestone = get_milestone_text(

        episode_number

    )


    # --------------------------------------------------------
    # Main information
    # --------------------------------------------------------

    text = (

        f"EPISODE  {episode_number} / {len(history)}\n\n"

        f"Step             {step_number} / {total_steps}\n"

        f"Action           {ACTION_NAMES.get(action, 'UNKNOWN')}\n"

        f"Step reward      {reward:+.0f}\n"

        f"Episode reward   {total_reward:+.0f}\n"

        f"Exploration ε    {epsilon:.3f}\n"

        f"Status           "
        f"{'SUCCESS' if success else 'EXPLORING'}"

    )


    ax_info.text(

        0.02,

        0.95,

        text,

        fontsize=12,

        verticalalignment="top"

    )


    # --------------------------------------------------------
    # Milestone
    # --------------------------------------------------------

    if milestone:

        ax_info.text(

            0.52,

            0.90,

            milestone,

            fontsize=13,

            fontweight="bold",

            verticalalignment="top"

        )


# ============================================================
# DETERMINE SPEED
# ============================================================

def get_delay(
    episode_number
):

    if (

        episode_number == FIRST_SUCCESS_EPISODE

        or

        episode_number == FIRST_OPTIMAL_EPISODE

        or

        episode_number == STABLE_EPISODE

    ):

        return NORMAL_DELAY


    if episode_number == FINAL_EPISODE:

        return FINAL_DELAY


    return FAST_DELAY


# ============================================================
# REPLAY
# ============================================================

print(
    "Starting cinematic replay..."
)

print()

print(
    "Episodes between milestones will play FAST."
)

print(
    "Milestones will slow down automatically."
)

print()

print(
    "Close the window to stop."
)

print()


# ============================================================
# EPISODE LOOP
# ============================================================

for episode in history:

    episode_number = episode["episode"]

    steps = episode["steps"]


    path = [
        START
    ]


    delay = get_delay(

        episode_number

    )


    # --------------------------------------------------------
    # Step loop
    # --------------------------------------------------------

    for step_index, step_data in enumerate(
        steps
    ):

        state = tuple(

            step_data["state"]

        )


        next_state = tuple(

            step_data["next_state"]

        )


        action = step_data["action"]

        reward = step_data["reward"]


        # ----------------------------------------------------
        # Track path
        # ----------------------------------------------------

        if state not in path:

            path.append(
                state
            )


        if next_state not in path:

            path.append(
                next_state
            )


        # ----------------------------------------------------
        # Draw everything
        # ----------------------------------------------------

        draw_maze(

            next_state,

            path

        )


        draw_graph(

            episode_number

        )


        draw_information(

            episode,

            step_index + 1,

            len(steps),

            action,

            reward

        )


        # ----------------------------------------------------
        # Main title
        # ----------------------------------------------------

        fig.suptitle(

            "AI MAZE AGENT",

            fontsize=20,

            fontweight="bold"

        )


        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------

        plt.pause(

            delay

        )


    # --------------------------------------------------------
    # Pause after important milestones
    # --------------------------------------------------------

    if episode_number in [

        FIRST_SUCCESS_EPISODE,

        FIRST_OPTIMAL_EPISODE,

        STABLE_EPISODE,

        FINAL_EPISODE

    ]:

        time.sleep(
            1.5
        )

    else:

        time.sleep(
            EPISODE_PAUSE
        )


# ============================================================
# COMPLETE
# ============================================================

plt.ioff()

plt.show()


print()

print("=" * 70)

print(
    "              CINEMATIC REPLAY COMPLETE"
)

print("=" * 70)

print()