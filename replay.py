import json
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_FILE = "training_history.json"

# Speed of individual actions during replay.
STEP_DELAY = 0.08

# Pause between episodes.
EPISODE_DELAY = 0.15


# ============================================================
# LOAD TRAINING HISTORY
# ============================================================

with open(HISTORY_FILE, "r") as file:
    history = json.load(file)


print()
print("=" * 60)
print("              AI MAZE TRAINING REPLAY")
print("=" * 60)
print()

print(
    f"Loaded {len(history)} recorded episodes."
)

print()


# ============================================================
# EXACT MAZE FROM environment.py
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


# ============================================================
# START AND GOAL
# ============================================================

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
# CREATE FIGURE
# ============================================================

plt.ion()

fig = plt.figure(
    figsize=(12, 7)
)


# ============================================================
# MAZE AXIS
# ============================================================

ax_maze = fig.add_axes(
    [0.05, 0.15, 0.42, 0.72]
)


# ============================================================
# LEARNING GRAPH
# ============================================================

ax_graph = fig.add_axes(
    [0.55, 0.55, 0.40, 0.32]
)


# ============================================================
# INFORMATION PANEL
# ============================================================

ax_info = fig.add_axes(
    [0.55, 0.15, 0.40, 0.30]
)

ax_info.axis("off")


# ============================================================
# DRAW MAZE
# ============================================================

def draw_maze(
    agent_position,
    path
):

    ax_maze.clear()


    # --------------------------------------------------------
    # DRAW CELLS
    # --------------------------------------------------------

    for row in range(ROWS):

        for col in range(COLS):

            # Wall
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
    # DRAW AGENT PATH
    # --------------------------------------------------------

    for state in path:

        row, col = state

        ax_maze.plot(
            col + 0.5,
            ROWS - row - 0.5,
            marker=".",
            markersize=4
        )


    # --------------------------------------------------------
    # DRAW START
    # --------------------------------------------------------

    start_row, start_col = START

    ax_maze.text(
        start_col + 0.5,
        ROWS - start_row - 0.5,
        "S",
        ha="center",
        va="center",
        fontsize=16
    )


    # --------------------------------------------------------
    # DRAW GOAL
    # --------------------------------------------------------

    goal_row, goal_col = GOAL

    ax_maze.text(
        goal_col + 0.5,
        ROWS - goal_row - 0.5,
        "G",
        ha="center",
        va="center",
        fontsize=18
    )


    # --------------------------------------------------------
    # DRAW AGENT
    # --------------------------------------------------------

    row, col = agent_position

    ax_maze.text(
        col + 0.5,
        ROWS - row - 0.5,
        "A",
        ha="center",
        va="center",
        fontsize=18
    )


    # --------------------------------------------------------
    # GRID
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
        "Agent Navigation"
    )


# ============================================================
# DRAW LEARNING GRAPH
# ============================================================

def draw_learning_graph(
    current_episode
):

    ax_graph.clear()


    episode_numbers = []

    rewards = []


    # Only show information that has
    # happened so far.

    for episode in history[
        :current_episode
    ]:

        episode_numbers.append(
            episode["episode"]
        )

        rewards.append(
            episode["total_reward"]
        )


    # --------------------------------------------------------
    # DRAW REWARD CURVE
    # --------------------------------------------------------

    ax_graph.plot(
        episode_numbers,
        rewards
    )


    ax_graph.set_title(
        "Learning Progress"
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


    # --------------------------------------------------------
    # FIX X AXIS
    # --------------------------------------------------------

    ax_graph.set_xlim(
        1,
        len(history)
    )


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

    epsilon = episode["epsilon"]

    total_reward = episode["total_reward"]


    # --------------------------------------------------------
    # INFORMATION TEXT
    # --------------------------------------------------------

    text = (

        f"EPISODE {episode_number}\n\n"

        f"Step: "
        f"{step_number} / {total_steps}\n"

        f"Action: "
        f"{ACTION_NAMES.get(action, 'UNKNOWN')}\n"

        f"Step reward: "
        f"{reward:+.0f}\n"

        f"Episode reward: "
        f"{total_reward:+.0f}\n\n"

        f"Epsilon: "
        f"{epsilon:.3f}\n"

        f"Success: "
        f"{'YES' if episode['success'] else 'NO'}"

    )


    ax_info.text(

        0.02,

        0.95,

        text,

        verticalalignment="top",

        fontsize=12

    )


# ============================================================
# REPLAY
# ============================================================

print(
    "Starting replay..."
)

print()

print(
    "Close the visualization window "
    "to stop the replay."
)

print()


# ============================================================
# EPISODE LOOP
# ============================================================

for episode_index, episode in enumerate(
    history
):


    episode_number = episode["episode"]

    steps = episode["steps"]


    # Start path at starting position.

    path = [
        list(START)
    ]


    # --------------------------------------------------------
    # STEP LOOP
    # --------------------------------------------------------

    for step_index, step_data in enumerate(
        steps
    ):


        # ----------------------------------------------------
        # GET DATA
        # ----------------------------------------------------

        state = tuple(
            step_data["state"]
        )

        next_state = tuple(
            step_data["next_state"]
        )

        action = step_data["action"]

        reward = step_data["reward"]


        # ----------------------------------------------------
        # ADD CURRENT STATE
        # ----------------------------------------------------

        if list(state) not in path:

            path.append(
                list(state)
            )


        # ----------------------------------------------------
        # ADD NEXT STATE
        # ----------------------------------------------------

        if list(next_state) not in path:

            path.append(
                list(next_state)
            )


        # ----------------------------------------------------
        # DRAW MAZE
        # ----------------------------------------------------

        draw_maze(

            next_state,

            path

        )


        # ----------------------------------------------------
        # DRAW GRAPH
        # ----------------------------------------------------

        draw_learning_graph(

            episode_number

        )


        # ----------------------------------------------------
        # DRAW INFORMATION
        # ----------------------------------------------------

        draw_information(

            episode,

            step_index + 1,

            len(steps),

            action,

            reward

        )


        # ----------------------------------------------------
        # GLOBAL TITLE
        # ----------------------------------------------------

        fig.suptitle(

            "AI Maze Agent — "
            "Reinforcement Learning Journey",

            fontsize=16

        )


        # ----------------------------------------------------
        # UPDATE WINDOW
        # ----------------------------------------------------

        plt.pause(
            STEP_DELAY
        )


    # --------------------------------------------------------
    # EPISODE PAUSE
    # --------------------------------------------------------

    time.sleep(
        EPISODE_DELAY
    )


# ============================================================
# REPLAY COMPLETE
# ============================================================

plt.ioff()

plt.show()


print()

print("=" * 60)

print(
    "              REPLAY COMPLETE"
)

print("=" * 60)

print()