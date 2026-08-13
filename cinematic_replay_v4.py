import json
import math
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


# ============================================================
# AI MAZE AGENT — CINEMATIC REPLAY V4
# ============================================================
#
# Presentation layer only.
#
# RL algorithm:
#     Q-Learning
#
# This program does NOT retrain the agent.
# It reads training_history.json and turns the learning journey
# into a compressed cinematic visualization.
#
# Main idea:
#
#     CHAOS
#       ↓
#     DISCOVERY
#       ↓
#     LEARNING
#       ↓
#     BREAKTHROUGH
#       ↓
#     STABILITY
#       ↓
#     MASTERY
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_FILE = "training_history.json"

TOTAL_EPISODES = 1000

# Number of frames used for ordinary episodes.
# Smaller = faster.
NORMAL_FRAMES = 5

# Frames for important episodes.
MILESTONE_FRAMES = 22

# Pause after milestone.
MILESTONE_PAUSE = 0.35

# Pause after final result.
FINAL_PAUSE = 2.5


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
# CINEMATIC MILESTONES
# ============================================================

MILESTONES = {
    1: "INITIALIZATION",
    4: "FIRST SUCCESS",
    156: "BREAKTHROUGH",
    324: "POLICY STABILIZED",
    1000: "TRAINING COMPLETE"
}


# ============================================================
# LOAD HISTORY
# ============================================================

print()
print("=" * 72)
print("             AI MAZE AGENT — CINEMATIC V4")
print("=" * 72)
print()

try:
    with open(HISTORY_FILE, "r") as file:
        history = json.load(file)

except FileNotFoundError:

    print("ERROR:")
    print("training_history.json was not found.")
    print()
    print("Run your training program first.")
    raise SystemExit


history_by_episode = {
    item["episode"]: item
    for item in history
}


# ============================================================
# DATA HELPERS
# ============================================================

def state_to_xy(state):

    row, col = state

    return (
        col + 0.5,
        ROWS - row - 0.5
    )


def interpolate(start, end, amount):

    x1, y1 = state_to_xy(start)
    x2, y2 = state_to_xy(end)

    return (
        x1 + (x2 - x1) * amount,
        y1 + (y2 - y1) * amount
    )


def get_path(item):

    path = [START]

    for step in item["steps"]:

        next_state = step["next_state"]

        path.append(
            tuple(next_state)
        )

    return path


# ============================================================
# FIGURE
# ============================================================

plt.ion()

fig = plt.figure(
    figsize=(15, 9),
    facecolor="#03050a"
)


# ------------------------------------------------------------
# MAZE
# ------------------------------------------------------------

ax_maze = fig.add_axes(
    [0.035, 0.17, 0.57, 0.72]
)


# ------------------------------------------------------------
# HUD
# ------------------------------------------------------------

ax_hud = fig.add_axes(
    [0.64, 0.17, 0.33, 0.72]
)


# ------------------------------------------------------------
# GRAPH
# ------------------------------------------------------------

ax_graph = fig.add_axes(
    [0.06, 0.045, 0.55, 0.075]
)


# ============================================================
# GLOBAL VISUAL SETTINGS
# ============================================================

for ax in [
    ax_maze,
    ax_hud,
    ax_graph
]:

    ax.set_facecolor(
        "#070a11"
    )


# ============================================================
# LEARNING STATE
# ============================================================

current_episode = 1

current_reward = 0

current_steps = 0

current_epsilon = 1.0

current_success = False

current_status = "INITIALIZING"

current_agent_position = START

trail = []

trail_strength = []


# ============================================================
# MAZE DRAW
# ============================================================

def draw_maze():

    ax_maze.clear()

    ax_maze.set_facecolor(
        "#070a11"
    )


    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    for row in range(ROWS):

        for col in range(COLS):

            y = ROWS - row - 1


            if MAZE[row, col] == 1:

                rect = Rectangle(

                    (col, y),

                    1,
                    1,

                    facecolor="#171d2a",

                    edgecolor="#344056",

                    linewidth=1.8

                )

            else:

                rect = Rectangle(

                    (col, y),

                    1,
                    1,

                    facecolor="#070a11",

                    edgecolor="#172236",

                    linewidth=1

                )

            ax_maze.add_patch(rect)


    # --------------------------------------------------------
    # Subtle grid pulse
    # --------------------------------------------------------

    pulse = (
        0.35
        + 0.12
        * math.sin(
            time.time() * 2
        )
    )

    ax_maze.grid(
        True,
        color="#153047",
        alpha=pulse * 0.25,
        linewidth=0.5
    )


    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    sx, sy = state_to_xy(START)

    ax_maze.scatter(

        sx,
        sy,

        s=210,

        facecolors="none",

        edgecolors="#00e5ff",

        linewidths=2.2,

        zorder=8

    )

    ax_maze.text(

        sx,
        sy,

        "S",

        color="#00e5ff",

        fontsize=11,

        fontweight="bold",

        ha="center",

        va="center",

        zorder=9

    )


    # --------------------------------------------------------
    # Goal pulse
    # --------------------------------------------------------

    gx, gy = state_to_xy(GOAL)

    goal_size = (

        220
        + 45
        * math.sin(
            time.time() * 5
        )

    )

    ax_maze.scatter(

        gx,
        gy,

        s=goal_size,

        marker="*",

        color="#ffd166",

        zorder=8

    )

    ax_maze.text(

        gx,
        gy - 0.35,

        "GOAL",

        color="#ffd166",

        fontsize=8,

        ha="center",

        fontweight="bold"

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

    ax_maze.set_xticks([])
    ax_maze.set_yticks([])


# ============================================================
# DRAW TRAIL
# ============================================================

def draw_trail():

    if len(trail) < 2:
        return


    for i in range(
        len(trail) - 1
    ):

        x1, y1 = state_to_xy(
            trail[i]
        )

        x2, y2 = state_to_xy(
            trail[i + 1]
        )


        strength = (

            (i + 1)
            / len(trail)

        )


        ax_maze.plot(

            [x1, x2],

            [y1, y2],

            color="#00e5ff",

            linewidth=1.5 + 3 * strength,

            alpha=0.10 + 0.75 * strength,

            solid_capstyle="round",

            zorder=4

        )


# ============================================================
# DRAW AGENT
# ============================================================

def draw_agent(position):

    x, y = position


    pulse = (

        0.17
        + 0.045
        * math.sin(
            time.time() * 8
        )

    )


    # Glow ring

    ax_maze.add_patch(

        Circle(

            (x, y),

            pulse * 2.1,

            facecolor="none",

            edgecolor="#00e5ff",

            linewidth=2,

            alpha=0.15,

            zorder=7

        )

    )


    # Agent

    ax_maze.scatter(

        [x],

        [y],

        s=150,

        color="#00e5ff",

        edgecolors="white",

        linewidths=1.3,

        zorder=10

    )


    ax_maze.text(

        x,

        y + 0.34,

        "AI",

        color="white",

        fontsize=7,

        fontweight="bold",

        ha="center",

        zorder=11

    )


# ============================================================
# DRAW HUD
# ============================================================

def draw_hud():

    ax_hud.clear()

    ax_hud.axis("off")

    ax_hud.set_facecolor(
        "#070a11"
    )


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    ax_hud.text(

        0.03,
        0.96,

        "AI LAB",

        transform=ax_hud.transAxes,

        fontsize=25,

        fontweight="bold",

        color="#00e5ff"

    )


    ax_hud.text(

        0.03,
        0.915,

        "Q-LEARNING / MAZE NAVIGATION",

        transform=ax_hud.transAxes,

        fontsize=9,

        color="#6f7e96"

    )


    ax_hud.plot(

        [0.03, 0.97],

        [0.88, 0.88],

        color="#1d2b40",

        linewidth=1,

        transform=ax_hud.transAxes

    )


    # --------------------------------------------------------
    # Episode
    # --------------------------------------------------------

    ax_hud.text(

        0.03,
        0.80,

        "EPISODE",

        transform=ax_hud.transAxes,

        fontsize=9,

        color="#6f7e96"

    )


    ax_hud.text(

        0.03,
        0.735,

        f"{current_episode:04d}",

        transform=ax_hud.transAxes,

        fontsize=31,

        fontweight="bold",

        color="white"

    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    progress = (

        current_episode
        / TOTAL_EPISODES

    )


    ax_hud.text(

        0.55,
        0.80,

        "TRAINING",

        transform=ax_hud.transAxes,

        fontsize=9,

        color="#6f7e96"

    )


    ax_hud.text(

        0.55,
        0.735,

        f"{progress * 100:5.1f}%",

        transform=ax_hud.transAxes,

        fontsize=20,

        fontweight="bold",

        color="#00e5ff"

    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    values = [

        ("REWARD", f"{current_reward:+.0f}"),

        ("STEPS", f"{current_steps}"),

        ("EPSILON", f"{current_epsilon:.3f}")

    ]


    ys = [
        0.61,
        0.51,
        0.41
    ]


    for (
        label,
        value
    ), y in zip(values, ys):

        ax_hud.text(

            0.03,
            y,

            label,

            transform=ax_hud.transAxes,

            fontsize=9,

            color="#6f7e96"

        )


        ax_hud.text(

            0.55,
            y,

            value,

            transform=ax_hud.transAxes,

            fontsize=16,

            fontweight="bold",

            color="white"

        )


    # --------------------------------------------------------
    # Exploration / Exploitation
    # --------------------------------------------------------

    ax_hud.text(

        0.03,
        0.30,

        "EXPLORATION",

        transform=ax_hud.transAxes,

        fontsize=8,

        color="#ff4d6d"

    )


    ax_hud.text(

        0.78,
        0.30,

        "EXPLOITATION",

        transform=ax_hud.transAxes,

        fontsize=8,

        color="#00e5ff",

        ha="right"

    )


    # Background

    ax_hud.add_patch(

        Rectangle(

            (0.03, 0.245),

            0.94,
            0.028,

            transform=ax_hud.transAxes,

            facecolor="#151d2b",

            edgecolor="none"

        )

    )


    # Exploration

    exploration = max(

        0,

        min(
            1,
            current_epsilon
        )

    )


    ax_hud.add_patch(

        Rectangle(

            (0.03, 0.245),

            0.94 * exploration,

            0.028,

            transform=ax_hud.transAxes,

            facecolor="#ff4d6d",

            edgecolor="none"

        )

    )


    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if current_success:

        status_color = "#00e5ff"

    else:

        status_color = "#ff4d6d"


    ax_hud.text(

        0.03,
        0.145,

        "SYSTEM STATUS",

        transform=ax_hud.transAxes,

        fontsize=8,

        color="#6f7e96"

    )


    ax_hud.text(

        0.03,
        0.085,

        current_status,

        transform=ax_hud.transAxes,

        fontsize=15,

        fontweight="bold",

        color=status_color

    )


# ============================================================
# DRAW LEARNING GRAPH
# ============================================================

def draw_graph():

    ax_graph.clear()

    ax_graph.set_facecolor(
        "#070a11"
    )


    visible = [

        item

        for item in history

        if item["episode"]
        <= current_episode

    ]


    if len(visible) > 1:

        episodes = [

            item["episode"]

            for item in visible

        ]

        rewards = [

            item["total_reward"]

            for item in visible

        ]


        ax_graph.plot(

            episodes,

            rewards,

            color="#00e5ff",

            linewidth=1.2,

            alpha=0.85

        )


        # Moving average

        if len(rewards) >= 20:

            window = min(
                20,
                len(rewards)
            )

            moving = np.convolve(

                rewards,

                np.ones(window)
                / window,

                mode="valid"

            )


            moving_x = episodes[
                window - 1:
            ]


            ax_graph.plot(

                moving_x,

                moving,

                color="#ffd166",

                linewidth=2,

                alpha=0.9

            )


    ax_graph.set_xlim(
        1,
        TOTAL_EPISODES
    )


    ax_graph.tick_params(

        colors="#53627a",

        labelsize=7

    )


    ax_graph.set_title(

        "LEARNING SIGNAL  /  REWARD",

        fontsize=8,

        color="#6f7e96",

        loc="left"

    )


    ax_graph.grid(

        True,

        alpha=0.10

    )


    for spine in ax_graph.spines.values():

        spine.set_color(
            "#182236"
        )


# ============================================================
# REFRESH SCREEN
# ============================================================

def refresh():

    draw_maze()

    draw_trail()

    draw_agent(
        current_agent_position
    )

    draw_hud()

    draw_graph()

    fig.canvas.draw_idle()

    fig.canvas.flush_events()


# ============================================================
# MILESTONE INTRO
# ============================================================

def show_milestone(
    episode,
    title
):

    overlay = fig.add_axes(
        [0, 0, 1, 1],
        zorder=100
    )

    overlay.set_facecolor(
        "#020308"
    )

    overlay.axis("off")


    # Horizontal scan line

    for i in range(3):

        y = 0.40 + i * 0.03

        overlay.plot(

            [0.20, 0.80],

            [y, y],

            color="#00e5ff",

            alpha=0.12,

            linewidth=1

        )


    overlay.text(

        0.5,
        0.61,

        title,

        ha="center",

        va="center",

        fontsize=28,

        fontweight="bold",

        color="#00e5ff"

    )


    overlay.text(

        0.5,
        0.51,

        f"EPISODE {episode:04d}",

        ha="center",

        va="center",

        fontsize=12,

        color="#7c8ca6"

    )


    descriptions = {

        1: "AGENT HAS NO KNOWLEDGE",

        4: "THE GOAL HAS BEEN DISCOVERED",

        156: "OPTIMAL ROUTE DISCOVERED",

        324: "BEHAVIOR HAS STABILIZED",

        1000: "AGENT HAS COMPLETED TRAINING"

    }


    overlay.text(

        0.5,
        0.43,

        descriptions.get(
            episode,
            ""
        ),

        ha="center",

        va="center",

        fontsize=10,

        color="white"

    )


    fig.canvas.draw()

    fig.canvas.flush_events()


    if episode in [4, 156, 324]:

        time.sleep(
            0.75
        )

    elif episode == 1:

        time.sleep(
            0.45
        )

    else:

        time.sleep(
            1.2
        )


    overlay.remove()


# ============================================================
# FAST TRAINING SIMULATION
# ============================================================

def simulate_episode(
    item,
    frames
):

    global current_episode
    global current_reward
    global current_steps
    global current_epsilon
    global current_success
    global current_status
    global current_agent_position
    global trail
    global trail_strength


    current_episode = item[
        "episode"
    ]

    current_reward = item[
        "total_reward"
    ]

    current_epsilon = item[
        "epsilon"
    ]

    current_success = item[
        "success"
    ]


    path = get_path(item)


    current_steps = len(path) - 1


    # --------------------------------------------------------
    # Determine behavior
    # --------------------------------------------------------

    if current_episode < 4:

        current_status = "EXPLORING"

    elif current_episode < 156:

        current_status = "LEARNING"

    elif current_episode < 324:

        current_status = "OPTIMIZING"

    else:

        current_status = "EXPLOITING"


    # --------------------------------------------------------
    # Compress path into frames
    # --------------------------------------------------------

    if len(path) <= 1:

        positions = [START]

    else:

        indices = np.linspace(

            0,

            len(path) - 1,

            frames

        ).astype(int)


        positions = [

            path[i]

            for i in indices

        ]


    for position in positions:

        current_agent_position = state_to_xy(
            position
        )


        # Recent path only

        trail = path[
            :path.index(position) + 1
        ] if position in path else []


        refresh()

        time.sleep(
            0.012
        )


# ============================================================
# TRAINING TITLE
# ============================================================

print(
    "Loading training history..."
)

time.sleep(
    0.4
)

print(
    "Compressing 1000 episodes into cinematic timeline..."
)

time.sleep(
    0.4
)

print(
    "Initializing visualization..."
)

time.sleep(
    0.5
)


# ============================================================
# INITIAL SCREEN
# ============================================================

current_episode = 0

current_reward = 0

current_steps = 0

current_epsilon = 1.0

current_success = False

current_status = "SYSTEM READY"

current_agent_position = state_to_xy(
    START
)

trail = []

refresh()

time.sleep(
    0.8
)


# ============================================================
# PLAY TRAINING
# ============================================================

for item in history:

    episode = item[
        "episode"
    ]


    # --------------------------------------------------------
    # IMPORTANT EPISODES
    # --------------------------------------------------------

    if episode in MILESTONES:

        show_milestone(

            episode,

            MILESTONES[
                episode
            ]

        )


    # --------------------------------------------------------
    # FAST / NORMAL MODE
    # --------------------------------------------------------

    if episode in MILESTONES:

        frames = MILESTONE_FRAMES

    else:

        # Most episodes are compressed aggressively.

        frames = NORMAL_FRAMES


    simulate_episode(

        item,

        frames

    )


    # --------------------------------------------------------
    # Console status
    # --------------------------------------------------------

    if (

        episode in MILESTONES

        or episode % 100 == 0

    ):

        print(

            f"Episode {episode:4d} "

            f"| Reward {item['total_reward']:4d} "

            f"| Steps {len(item['steps']):3d} "

            f"| Epsilon {item['epsilon']:.3f}"

        )


# ============================================================
# FINAL CINEMATIC RESULT
# ============================================================

current_episode = 1000

last = history_by_episode[
    1000
]

current_reward = last[
    "total_reward"
]

current_steps = len(
    last["steps"]
)

current_epsilon = last[
    "epsilon"
]

current_success = True

current_status = "POLICY MASTERED"

current_agent_position = state_to_xy(
    GOAL
)

trail = get_path(last)


refresh()

time.sleep(
    0.8
)


# ============================================================
# FINAL OVERLAY
# ============================================================

overlay = fig.add_axes(
    [0, 0, 1, 1],
    zorder=200
)

overlay.set_facecolor(
    "#020308"
)

overlay.axis("off")


overlay.text(

    0.5,
    0.68,

    "TRAINING COMPLETE",

    ha="center",

    fontsize=32,

    fontweight="bold",

    color="#00e5ff"

)


overlay.text(

    0.5,
    0.59,

    "1000 EPISODES",

    ha="center",

    fontsize=14,

    color="#7c8ca6"

)


overlay.text(

    0.5,
    0.49,

    "10 STEPS",

    ha="center",

    fontsize=25,

    fontweight="bold",

    color="white"

)


overlay.text(

    0.5,
    0.42,

    "OPTIMAL ROUTE DISCOVERED",

    ha="center",

    fontsize=11,

    color="#ffd166"

)


overlay.text(

    0.5,
    0.31,

    "EXPLORATION",

    ha="right",

    fontsize=9,

    color="#ff4d6d"

)


overlay.text(

    0.5,
    0.31,

    "  →  EXPLOITATION",

    ha="left",

    fontsize=9,

    color="#00e5ff"

)


fig.canvas.draw()

fig.canvas.flush_events()


print()
print("=" * 72)
print("                 CINEMATIC V4 COMPLETE")
print("=" * 72)
print()
print("Training journey visualized.")
print("Close the graph window to finish.")
print()


plt.ioff()

plt.show()