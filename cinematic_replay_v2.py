import json
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# SETTINGS
# ============================================================

HISTORY_FILE = "training_history.json"

# Fast episode animation
FAST_STEP_DELAY = 0.001

# Normal milestone animation
NORMAL_STEP_DELAY = 0.12

# Final animation
FINAL_STEP_DELAY = 0.20

# How often fast episodes are visually shown
FAST_EPISODE_INTERVAL = 5


# ============================================================
# MILESTONES
# ============================================================

FIRST_SUCCESS = 4
FIRST_OPTIMAL = 156
STABLE = 324
FINAL = 1000


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
print("        AI MAZE AGENT — COMPRESSED CINEMATIC REPLAY")
print("=" * 70)
print()

print(
    f"Loaded {len(history)} episodes."
)

print()

print(
    "Fast mode      : compressed"
)

print(
    "Milestones     : detailed"
)

print(
    "Final episode  : detailed"
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
# FIGURE
# ============================================================

plt.ion()

fig = plt.figure(
    figsize=(14, 8)
)


ax_maze = fig.add_axes(
    [0.04, 0.20, 0.43, 0.68]
)


ax_graph = fig.add_axes(
    [0.54, 0.55, 0.42, 0.32]
)


ax_info = fig.add_axes(
    [0.54, 0.20, 0.42, 0.25]
)

ax_info.axis("off")


# ============================================================
# DRAW MAZE
# ============================================================

def draw_maze(
    position,
    path
):

    ax_maze.clear()


    # Walls

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


    # Path

    for p in path:

        row, col = p

        ax_maze.plot(

            col + 0.5,

            ROWS - row - 0.5,

            marker=".",

            markersize=5

        )


    # Start

    row, col = START

    ax_maze.text(

        col + 0.5,

        ROWS - row - 0.5,

        "S",

        ha="center",

        va="center",

        fontsize=18

    )


    # Goal

    row, col = GOAL

    ax_maze.text(

        col + 0.5,

        ROWS - row - 0.5,

        "★",

        ha="center",

        va="center",

        fontsize=22

    )


    # Agent

    row, col = position

    ax_maze.text(

        col + 0.5,

        ROWS - row - 0.5,

        "🤖",

        ha="center",

        va="center",

        fontsize=18

    )


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
# DRAW GRAPH
# ============================================================

def draw_graph(
    current_episode
):

    ax_graph.clear()


    episode_numbers = []

    rewards = []


    for ep in history:

        if ep["episode"] > current_episode:

            break

        episode_numbers.append(
            ep["episode"]
        )

        rewards.append(
            ep["total_reward"]
        )


    ax_graph.plot(
        episode_numbers,
        rewards
    )


    # Milestones

    if current_episode >= FIRST_SUCCESS:

        ax_graph.axvline(
            FIRST_SUCCESS,
            linestyle="--",
            alpha=0.5
        )


    if current_episode >= FIRST_OPTIMAL:

        ax_graph.axvline(
            FIRST_OPTIMAL,
            linestyle="--",
            alpha=0.5
        )


    if current_episode >= STABLE:

        ax_graph.axvline(
            STABLE,
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
# INFORMATION
# ============================================================

def show_information(
    episode,
    current_step
):

    ax_info.clear()
    ax_info.axis("off")


    number = episode["episode"]

    total_steps = len(
        episode["steps"]
    )

    reward = episode["total_reward"]

    epsilon = episode["epsilon"]

    success = episode["success"]


    status = (
        "SUCCESS"
        if success
        else
        "EXPLORING"
    )


    text = (

        f"EPISODE   {number} / {len(history)}\n\n"

        f"Step             {current_step} / {total_steps}\n"

        f"Episode reward   {reward:+.0f}\n"

        f"Exploration ε    {epsilon:.3f}\n"

        f"Status           {status}"

    )


    ax_info.text(

        0.02,
        0.95,

        text,

        fontsize=12,

        verticalalignment="top"

    )


    # Milestone message

    if number == FIRST_SUCCESS:

        message = (

            "🎯 FIRST SUCCESS\n\n"

            "The agent reached the goal\n"

            "for the first time."

        )


    elif number == FIRST_OPTIMAL:

        message = (

            "🔥 BREAKTHROUGH\n\n"

            "First 10-step optimal solution."

        )


    elif number == STABLE:

        message = (

            "🧠 STABLE POLICY\n\n"

            "The agent is now solving\n"

            "the maze consistently."

        )


    elif number == FINAL:

        message = (

            "🏆 LEARNING COMPLETE\n\n"

            "1000 episodes completed."

        )


    else:

        message = ""


    if message:

        ax_info.text(

            0.52,
            0.90,

            message,

            fontsize=13,

            fontweight="bold",

            verticalalignment="top"

        )


# ============================================================
# DRAW COMPLETE EPISODE
# ============================================================

def draw_episode_summary(
    episode
):

    steps = episode["steps"]

    position = START

    path = [START]


    for step in steps:

        position = tuple(
            step["next_state"]
        )

        if position not in path:

            path.append(
                position
            )


    draw_maze(
        position,
        path
    )

    draw_graph(
        episode["episode"]
    )

    show_information(

        episode,

        len(steps)

    )

    fig.suptitle(

        "AI MAZE AGENT — LEARNING",

        fontsize=20,

        fontweight="bold"

    )

    plt.pause(
        FAST_STEP_DELAY
    )


# ============================================================
# DETAILED EPISODE
# ============================================================

def replay_episode(
    episode,
    delay
):

    path = [START]


    for index, step in enumerate(
        episode["steps"]
    ):

        position = tuple(
            step["next_state"]
        )


        if position not in path:

            path.append(
                position
            )


        draw_maze(
            position,
            path
        )

        draw_graph(
            episode["episode"]
        )

        show_information(

            episode,

            index + 1

        )


        fig.suptitle(

            "AI MAZE AGENT — LEARNING",

            fontsize=20,

            fontweight="bold"

        )


        plt.pause(
            delay
        )


# ============================================================
# START
# ============================================================

print(
    "Starting replay..."
)

print()

time.sleep(
    1
)


# ============================================================
# MAIN LOOP
# ============================================================

for episode in history:

    number = episode["episode"]


    # --------------------------------------------------------
    # IMPORTANT EPISODES
    # --------------------------------------------------------

    if number in [

        FIRST_SUCCESS,
        FIRST_OPTIMAL,
        STABLE,
        FINAL

    ]:

        print(
            f"▶ Detailed replay: Episode {number}"
        )


        replay_episode(

            episode,

            NORMAL_STEP_DELAY

            if number != FINAL

            else FINAL_STEP_DELAY

        )


        time.sleep(
            1.5
        )


    # --------------------------------------------------------
    # FAST MODE
    # --------------------------------------------------------

    elif (

        number % FAST_EPISODE_INTERVAL == 0

    ):

        draw_episode_summary(
            episode
        )


# ============================================================
# FINISHED
# ============================================================

print()

print("=" * 70)

print(
    "           CINEMATIC REPLAY COMPLETE"
)

print("=" * 70)

print()

print(
    "Close the matplotlib window to exit."
)

plt.ioff()

plt.show()