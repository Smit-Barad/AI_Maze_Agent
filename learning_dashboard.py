import json
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_FILE = "training_history.json"
SNAPSHOT_FILE = "q_learning_snapshots.npz"


EPISODES = [
    1,
    4,
    50,
    100,
    156,
    200,
    324,
    500,
    750,
    1000
]


# How long each snapshot remains visible
NORMAL_PAUSE = 1.2

# Extra pause at important moments
MILESTONE_PAUSE = 2.5


# ============================================================
# MAZE
# ============================================================

MAZE = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0]
])


ROWS, COLS = MAZE.shape

START = (0, 0)
GOAL = (5, 5)


# ============================================================
# ACTION ARROWS
# ============================================================

ARROWS = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→"
}


# ============================================================
# LOAD TRAINING HISTORY
# ============================================================

with open(
    HISTORY_FILE,
    "r"
) as file:

    history = json.load(file)


history_by_episode = {

    item["episode"]: item

    for item in history

}


# ============================================================
# LOAD Q-TABLE SNAPSHOTS
# ============================================================

q_data = np.load(
    SNAPSHOT_FILE
)


# ============================================================
# FIGURE
# ============================================================

plt.ion()

fig = plt.figure(
    figsize=(15, 9)
)


# ============================================================
# AXES
# ============================================================

ax_maze = fig.add_axes(
    [0.03, 0.36, 0.42, 0.55]
)


ax_policy = fig.add_axes(
    [0.55, 0.36, 0.42, 0.55]
)


ax_graph = fig.add_axes(
    [0.08, 0.07, 0.58, 0.22]
)


ax_info = fig.add_axes(
    [0.70, 0.06, 0.27, 0.24]
)

ax_info.axis("off")


# ============================================================
# GET EPISODE PATH
# ============================================================

def get_episode_path(
    episode_data
):

    path = [START]

    for step in episode_data["steps"]:

        next_state = tuple(
            step["next_state"]
        )

        path.append(
            next_state
        )

    return path


# ============================================================
# DRAW MAZE
# ============================================================

def draw_maze(
    episode_data,
    current_step=None
):

    ax_maze.clear()

    path = get_episode_path(
        episode_data
    )


    if current_step is not None:

        path = path[
            :current_step + 1
        ]


    if len(path) == 0:

        position = START

    else:

        position = path[-1]


    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    for row in range(ROWS):

        for col in range(COLS):

            y = ROWS - row - 1


            if MAZE[row, col] == 1:

                ax_maze.add_patch(

                    Rectangle(

                        (col, y),
                        1,
                        1

                    )

                )

            else:

                ax_maze.add_patch(

                    Rectangle(

                        (col, y),
                        1,
                        1,
                        fill=False,
                        alpha=0.35

                    )

                )


    # --------------------------------------------------------
    # Path
    # --------------------------------------------------------

    if len(path) > 1:

        xs = [
            p[1] + 0.5
            for p in path
        ]

        ys = [
            ROWS - p[0] - 0.5
            for p in path
        ]

        ax_maze.plot(
            xs,
            ys,
            linewidth=3
        )


    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    row, col = START

    ax_maze.text(

        col + 0.5,
        ROWS - row - 0.5,

        "S",

        fontsize=20,
        ha="center",
        va="center",
        fontweight="bold"

    )


    # --------------------------------------------------------
    # Goal
    # --------------------------------------------------------

    row, col = GOAL

    ax_maze.text(

        col + 0.5,
        ROWS - row - 0.5,

        "★",

        fontsize=26,
        ha="center",
        va="center"

    )


    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    row, col = position

    ax_maze.text(

        col + 0.5,
        ROWS - row - 0.5,

        "🤖",

        fontsize=21,
        ha="center",
        va="center"

    )


    ax_maze.set_xlim(
        0,
        COLS
    )

    ax_maze.set_ylim(
        0,
        ROWS
    )

    ax_maze.set_aspect(
        "equal"
    )

    ax_maze.set_xticks(
        range(COLS + 1)
    )

    ax_maze.set_yticks(
        range(ROWS + 1)
    )

    ax_maze.grid(
        True,
        alpha=0.3
    )

    ax_maze.set_title(
        "WHAT THE AGENT DID",
        fontsize=15,
        fontweight="bold"
    )


# ============================================================
# DRAW POLICY
# ============================================================

def draw_policy(
    q_table,
    episode
):

    ax_policy.clear()


    for row in range(ROWS):

        for col in range(COLS):

            y = ROWS - row - 1


            # ------------------------------------------------
            # Wall
            # ------------------------------------------------

            if MAZE[row, col] == 1:

                ax_policy.add_patch(

                    Rectangle(

                        (col, y),
                        1,
                        1

                    )

                )

                continue


            # ------------------------------------------------
            # Cell
            # ------------------------------------------------

            ax_policy.add_patch(

                Rectangle(

                    (col, y),
                    1,
                    1,
                    fill=False,
                    alpha=0.35

                )

            )


            # ------------------------------------------------
            # Goal
            # ------------------------------------------------

            if (
                row,
                col
            ) == GOAL:

                ax_policy.text(

                    col + 0.5,
                    y + 0.5,

                    "★",

                    fontsize=26,
                    ha="center",
                    va="center"

                )

                continue


            # ------------------------------------------------
            # Q-values
            # ------------------------------------------------

            values = q_table[
                row,
                col
            ]


            best_action = np.argmax(
                values
            )


            best_value = values[
                best_action
            ]


            # ------------------------------------------------
            # Arrow
            # ------------------------------------------------

            ax_policy.text(

                col + 0.5,
                y + 0.56,

                ARROWS[
                    best_action
                ],

                fontsize=27,
                ha="center",
                va="center"

            )


            # ------------------------------------------------
            # Q-value
            # ------------------------------------------------

            ax_policy.text(

                col + 0.5,
                y + 0.15,

                f"{best_value:.1f}",

                fontsize=8,
                ha="center",
                va="center"

            )


    ax_policy.set_xlim(
        0,
        COLS
    )

    ax_policy.set_ylim(
        0,
        ROWS
    )

    ax_policy.set_aspect(
        "equal"
    )

    ax_policy.set_xticks(
        range(COLS + 1)
    )

    ax_policy.set_yticks(
        range(ROWS + 1)
    )

    ax_policy.grid(
        True,
        alpha=0.3
    )

    ax_policy.set_title(

        f"WHAT THE AGENT LEARNED — EPISODE {episode}",

        fontsize=15,
        fontweight="bold"

    )


# ============================================================
# DRAW LEARNING GRAPH
# ============================================================

def draw_learning_graph(
    current_episode
):

    ax_graph.clear()


    episodes = []

    rewards = []


    for item in history:

        if item["episode"] <= current_episode:

            episodes.append(
                item["episode"]
            )

            rewards.append(
                item["total_reward"]
            )


    ax_graph.plot(
        episodes,
        rewards,
        linewidth=1.8
    )


    # Current episode marker

    current_item = history_by_episode[
        current_episode
    ]


    ax_graph.scatter(

        [current_episode],

        [current_item["total_reward"]],

        s=70

    )


    # Milestone lines

    milestones = [

        (4, "First success"),

        (156, "First optimal"),

        (324, "Stable")

    ]


    for ep, label in milestones:

        if current_episode >= ep:

            ax_graph.axvline(

                ep,

                linestyle="--",

                alpha=0.45

            )


    ax_graph.set_xlim(
        1,
        len(history)
    )


    ax_graph.set_title(

        "LEARNING CURVE — REWARD OVER TIME",

        fontsize=13,
        fontweight="bold"

    )


    ax_graph.set_xlabel(
        "Training Episode"
    )


    ax_graph.set_ylabel(
        "Reward"
    )


    ax_graph.grid(
        True,
        alpha=0.25
    )


# ============================================================
# DRAW INFORMATION PANEL
# ============================================================

def draw_information(
    episode_data
):

    ax_info.clear()

    ax_info.axis("off")


    episode = episode_data[
        "episode"
    ]

    reward = episode_data[
        "total_reward"
    ]

    steps = len(
        episode_data["steps"]
    )

    epsilon = episode_data[
        "epsilon"
    ]

    success = episode_data[
        "success"
    ]


    if episode == 4:

        status = "🎯 FIRST SUCCESS"

    elif episode == 156:

        status = "🔥 BREAKTHROUGH"

    elif episode == 324:

        status = "🧠 STABLE POLICY"

    elif episode == 1000:

        status = "🏆 TRAINING COMPLETE"

    elif success:

        status = "Goal reached"

    else:

        status = "Exploring"


    text = (

        f"EPISODE\n"

        f"{episode} / {len(history)}\n\n"

        f"REWARD\n"

        f"{reward:+.0f}\n\n"

        f"STEPS\n"

        f"{steps}\n\n"

        f"EPSILON\n"

        f"{epsilon:.3f}\n\n"

        f"STATUS\n"

        f"{status}"

    )


    ax_info.text(

        0.05,
        0.98,

        text,

        fontsize=13,

        verticalalignment="top",

        fontweight="bold"

    )


# ============================================================
# UPDATE DASHBOARD
# ============================================================

def update_dashboard(
    episode
):

    episode_data = history_by_episode[
        episode
    ]


    q_table = q_data[
        f"episode_{episode}"
    ]


    draw_maze(
        episode_data
    )


    draw_policy(
        q_table,
        episode
    )


    draw_learning_graph(
        episode
    )


    draw_information(
        episode_data
    )


    fig.suptitle(

        "AI MAZE AGENT — FROM EXPLORATION TO LEARNING",

        fontsize=21,

        fontweight="bold"

    )


    plt.pause(
        0.1
    )


# ============================================================
# START
# ============================================================

print()

print("=" * 70)

print(
    "          AI MAZE AGENT — LEARNING DASHBOARD"
)

print("=" * 70)

print()

print(
    "Connecting:"
)

print(
    "  Agent behavior"
)

print(
    "  Q-learning policy"
)

print(
    "  Learning curve"
)

print(
    "  Training statistics"
)

print()

time.sleep(
    1
)


# ============================================================
# PLAYBACK
# ============================================================

for episode in EPISODES:

    print(
        f"Showing Episode {episode}"
    )


    update_dashboard(
        episode
    )


    if episode in [
        4,
        156,
        324,
        1000
    ]:

        plt.pause(
            MILESTONE_PAUSE
        )

    else:

        plt.pause(
            NORMAL_PAUSE
        )


# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 70)

print(
    "              DASHBOARD COMPLETE"
)

print("=" * 70)

print()

print(
    "Close the dashboard window to finish."
)

plt.ioff()

plt.show()