import json
import time
import math
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


# ============================================================
# AI MAZE AGENT — CINEMATIC REPLAY V3
# ============================================================
#
# This version focuses on:
#
#   • Futuristic AI laboratory UI
#   • Smooth agent movement
#   • Glowing learned path
#   • Live episode / reward / steps / epsilon HUD
#   • Exploration → discovery → optimization → stability
#   • Milestone cinematic effects
#
# IMPORTANT:
# This file does NOT change the RL algorithm.
# It only changes how the learning journey is presented.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_FILE = "training_history.json"

TOTAL_EPISODES = 1000

FRAME_DELAY = 0.035

FAST_DELAY = 0.006

MILESTONE_DELAY = 0.10

CELL_SIZE = 1.0


# Episodes that receive cinematic emphasis
MILESTONES = {

    1: "INITIALIZATION",

    4: "FIRST SUCCESS",

    156: "BREAKTHROUGH",

    324: "POLICY STABILIZED",

    1000: "TRAINING COMPLETE"

}


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
# LOAD TRAINING HISTORY
# ============================================================

print()

print("=" * 70)

print(
    "       AI MAZE AGENT — CINEMATIC V3"
)

print("=" * 70)

print()


try:

    with open(
        HISTORY_FILE,
        "r"
    ) as file:

        history = json.load(file)

except FileNotFoundError:

    print(
        "ERROR: training_history.json not found."
    )

    print(
        "Run your training program first."
    )

    raise SystemExit


history_by_episode = {

    item["episode"]: item

    for item in history

}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def state_to_xy(
    state
):

    row, col = state

    x = col + 0.5

    y = ROWS - row - 0.5

    return x, y


def interpolate(
    start,
    end,
    amount
):

    x1, y1 = state_to_xy(
        start
    )

    x2, y2 = state_to_xy(
        end
    )

    x = x1 + (
        x2 - x1
    ) * amount

    y = y1 + (
        y2 - y1
    ) * amount

    return x, y


# ============================================================
# FIGURE
# ============================================================

plt.ion()

fig = plt.figure(
    figsize=(15, 9)
)


# Main maze area
ax_maze = fig.add_axes(
    [0.04, 0.16, 0.57, 0.74]
)


# HUD
ax_hud = fig.add_axes(
    [0.65, 0.16, 0.31, 0.74]
)


# Bottom learning graph
ax_graph = fig.add_axes(
    [0.08, 0.035, 0.52, 0.08]
)


# ============================================================
# FUTURISTIC BACKGROUND
# ============================================================

fig.patch.set_facecolor(
    "#05070d"
)

ax_maze.set_facecolor(
    "#080b12"
)

ax_hud.set_facecolor(
    "#080b12"
)

ax_graph.set_facecolor(
    "#080b12"
)


# ============================================================
# DRAW MAZE BASE
# ============================================================

def draw_maze_base():

    ax_maze.clear()

    ax_maze.set_facecolor(
        "#080b12"
    )


    for row in range(ROWS):

        for col in range(COLS):

            y = ROWS - row - 1


            if MAZE[row, col] == 1:

                rect = Rectangle(

                    (
                        col,
                        y
                    ),

                    CELL_SIZE,
                    CELL_SIZE,

                    facecolor="#161c29",

                    edgecolor="#39445a",

                    linewidth=1.5

                )

                ax_maze.add_patch(
                    rect
                )

            else:

                rect = Rectangle(

                    (
                        col,
                        y
                    ),

                    CELL_SIZE,
                    CELL_SIZE,

                    facecolor="#080b12",

                    edgecolor="#1b2536",

                    linewidth=0.8

                )

                ax_maze.add_patch(
                    rect
                )


    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    sx, sy = state_to_xy(
        START
    )

    ax_maze.scatter(

        [sx],
        [sy],

        s=180,

        marker="s",

        facecolors="none",

        edgecolors="#00e5ff",

        linewidths=2

    )

    ax_maze.text(

        sx,
        sy,

        "S",

        ha="center",

        va="center",

        fontsize=11,

        fontweight="bold",

        color="#00e5ff"

    )


    # --------------------------------------------------------
    # Goal
    # --------------------------------------------------------

    gx, gy = state_to_xy(
        GOAL
    )

    ax_maze.scatter(

        [gx],
        [gy],

        s=250,

        marker="*",

        color="#ffd166",

        zorder=5

    )


    ax_maze.text(

        gx,
        gy - 0.32,

        "GOAL",

        ha="center",

        fontsize=8,

        color="#ffd166"

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
        []
    )

    ax_maze.set_yticks(
        []
    )


# ============================================================
# DRAW PATH
# ============================================================

def draw_path(
    path,
    progress=1.0
):

    if len(path) < 2:

        return


    visible_count = max(

        1,

        int(
            len(path)
            * progress
        )

    )


    visible_path = path[
        :visible_count
    ]


    xs = []

    ys = []


    for state in visible_path:

        x, y = state_to_xy(
            state
        )

        xs.append(
            x
        )

        ys.append(
            y
        )


    ax_maze.plot(

        xs,
        ys,

        linewidth=5,

        alpha=0.15,

        color="#00e5ff",

        solid_capstyle="round"

    )


    ax_maze.plot(

        xs,
        ys,

        linewidth=2.2,

        alpha=0.95,

        color="#00e5ff",

        solid_capstyle="round"

    )


# ============================================================
# DRAW AGENT
# ============================================================

def draw_agent(
    x,
    y
):

    # Outer pulse

    pulse = (
        0.24
        + 0.05
        * math.sin(
            time.time() * 7
        )
    )


    circle = Circle(

        (
            x,
            y
        ),

        pulse,

        facecolor="none",

        edgecolor="#00e5ff",

        linewidth=2,

        alpha=0.4

    )


    ax_maze.add_patch(
        circle
    )


    # Core

    ax_maze.scatter(

        [x],
        [y],

        s=130,

        color="#00e5ff",

        edgecolors="white",

        linewidths=1.2,

        zorder=10

    )


    # AI label

    ax_maze.text(

        x,
        y + 0.38,

        "AI",

        ha="center",

        va="center",

        fontsize=8,

        color="#ffffff",

        fontweight="bold",

        zorder=11

    )


# ============================================================
# DRAW HUD
# ============================================================

def draw_hud(
    episode,
    reward,
    steps,
    epsilon,
    success,
    status
):

    ax_hud.clear()

    ax_hud.set_facecolor(
        "#080b12"
    )

    ax_hud.axis(
        "off"
    )


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    ax_hud.text(

        0.05,
        0.96,

        "AI  LAB",

        fontsize=24,

        fontweight="bold",

        color="#00e5ff",

        transform=ax_hud.transAxes

    )


    ax_hud.text(

        0.05,
        0.91,

        "REINFORCEMENT LEARNING SYSTEM",

        fontsize=9,

        color="#8c9ab3",

        transform=ax_hud.transAxes

    )


    # --------------------------------------------------------
    # Divider
    # --------------------------------------------------------

    ax_hud.plot(

        [0.05, 0.95],

        [0.87, 0.87],

        color="#26344b",

        linewidth=1,

        transform=ax_hud.transAxes

    )


    # --------------------------------------------------------
    # Episode
    # --------------------------------------------------------

    ax_hud.text(

        0.05,
        0.80,

        "EPISODE",

        fontsize=9,

        color="#7f8da6",

        transform=ax_hud.transAxes

    )


    ax_hud.text(

        0.05,
        0.735,

        f"{episode:04d}",

        fontsize=30,

        fontweight="bold",

        color="white",

        transform=ax_hud.transAxes

    )


    ax_hud.text(

        0.55,
        0.80,

        "PROGRESS",

        fontsize=9,

        color="#7f8da6",

        transform=ax_hud.transAxes

    )


    ax_hud.text(

        0.55,
        0.735,

        f"{episode / TOTAL_EPISODES * 100:5.1f}%",

        fontsize=20,

        fontweight="bold",

        color="#00e5ff",

        transform=ax_hud.transAxes

    )


    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    stats = [

        ("REWARD", f"{reward:+.0f}"),

        ("STEPS", f"{steps}"),

        ("EPSILON", f"{epsilon:.3f}")

    ]


    y_positions = [

        0.60,

        0.50,

        0.40

    ]


    for (
        label,
        value
    ), y in zip(

        stats,
        y_positions

    ):

        ax_hud.text(

            0.05,
            y,

            label,

            fontsize=9,

            color="#7f8da6",

            transform=ax_hud.transAxes

        )


        ax_hud.text(

            0.55,
            y,

            value,

            fontsize=16,

            fontweight="bold",

            color="white",

            transform=ax_hud.transAxes

        )


    # --------------------------------------------------------
    # Exploration meter
    # --------------------------------------------------------

    ax_hud.text(

        0.05,
        0.29,

        "EXPLORATION",

        fontsize=9,

        color="#7f8da6",

        transform=ax_hud.transAxes

    )


    ax_hud.text(

        0.55,
        0.29,

        "EXPLOITATION",

        fontsize=9,

        color="#7f8da6",

        transform=ax_hud.transAxes

    )


    # Background bar

    ax_hud.add_patch(

        Rectangle(

            (
                0.05,
                0.235
            ),

            0.90,
            0.025,

            transform=ax_hud.transAxes,

            facecolor="#182234",

            edgecolor="none"

        )

    )


    # Exploration width

    exploration_width = min(

        0.90,

        max(
            0.0,
            epsilon * 0.90
        )

    )


    ax_hud.add_patch(

        Rectangle(

            (
                0.05,
                0.235
            ),

            exploration_width,
            0.025,

            transform=ax_hud.transAxes,

            facecolor="#ff4d6d",

            edgecolor="none"

        )

    )


    # --------------------------------------------------------
    # Status box
    # --------------------------------------------------------

    if success:

        status_color = "#00e5ff"

    else:

        status_color = "#ff4d6d"


    ax_hud.text(

        0.05,
        0.13,

        "STATUS",

        fontsize=9,

        color="#7f8da6",

        transform=ax_hud.transAxes

    )


    ax_hud.text(

        0.05,
        0.07,

        status,

        fontsize=13,

        fontweight="bold",

        color=status_color,

        transform=ax_hud.transAxes

    )


# ============================================================
# DRAW LEARNING GRAPH
# ============================================================

def draw_learning_graph(
    current_episode
):

    ax_graph.clear()

    ax_graph.set_facecolor(
        "#080b12"
    )


    visible = [

        item

        for item in history

        if item["episode"]
        <= current_episode

    ]


    episodes = [

        item["episode"]

        for item in visible

    ]


    rewards = [

        item["total_reward"]

        for item in visible

    ]


    if len(
        episodes
    ) > 1:

        ax_graph.plot(

            episodes,

            rewards,

            linewidth=1.4,

            color="#00e5ff"

        )


    ax_graph.scatter(

        [current_episode],

        [rewards[-1]],

        s=40,

        color="#ffd166",

        zorder=5

    )


    ax_graph.set_xlim(

        1,

        TOTAL_EPISODES

    )


    ax_graph.set_title(

        "LIVE LEARNING SIGNAL  •  REWARD",

        fontsize=9,

        color="#7f8da6",

        loc="left"

    )


    ax_graph.tick_params(

        colors="#53627a",

        labelsize=7

    )


    for spine in ax_graph.spines.values():

        spine.set_color(
            "#1b2536"
        )


    ax_graph.grid(

        True,

        alpha=0.12

    )


# ============================================================
# MILESTONE SCREEN
# ============================================================

def milestone_screen(
    episode,
    title
):

    # Dark overlay

    overlay = fig.add_axes(

        [0, 0, 1, 1],

        zorder=100

    )

    overlay.set_facecolor(
        "#020307"
    )

    overlay.axis(
        "off"
    )


    overlay.text(

        0.5,
        0.58,

        title,

        ha="center",

        va="center",

        fontsize=30,

        fontweight="bold",

        color="#00e5ff"

    )


    overlay.text(

        0.5,
        0.48,

        f"EPISODE {episode:04d}",

        ha="center",

        va="center",

        fontsize=13,

        color="#8c9ab3"

    )


    if episode == 4:

        subtitle = (
            "GOAL DISCOVERED"
        )

    elif episode == 156:

        subtitle = (
            "OPTIMAL 10-STEP ROUTE FOUND"
        )

    elif episode == 324:

        subtitle = (
            "POLICY STABILIZED"
        )

    elif episode == 1000:

        subtitle = (
            "TRAINING COMPLETE"
        )

    else:

        subtitle = (
            "SYSTEM INITIALIZED"
        )


    overlay.text(

        0.5,
        0.41,

        subtitle,

        ha="center",

        va="center",

        fontsize=11,

        color="white"

    )


    fig.canvas.draw()

    fig.canvas.flush_events()


    time.sleep(
        1.5
    )


    overlay.remove()


# ============================================================
# EPISODE REPLAY
# ============================================================

def replay_episode(
    episode_data,
    speed
):

    episode = episode_data[
        "episode"
    ]

    reward = episode_data[
        "total_reward"
    ]

    epsilon = episode_data[
        "epsilon"
    ]

    steps_data = episode_data[
        "steps"
    ]

    success = episode_data[
        "success"
    ]


    # --------------------------------------------------------
    # Build path
    # --------------------------------------------------------

    path = [START]

    for step in steps_data:

        path.append(

            tuple(
                step["next_state"]
            )

        )


    total_steps = len(
        path
    ) - 1


    # --------------------------------------------------------
    # Draw initial dashboard
    # --------------------------------------------------------

    draw_maze_base()

    draw_path(
        path,
        0
    )

    draw_hud(

        episode,

        reward,

        total_steps,

        epsilon,

        success,

        "EXPLORING..."

    )

    draw_learning_graph(
        episode
    )


    fig.canvas.draw()

    fig.canvas.flush_events()


    # --------------------------------------------------------
    # Move through path
    # --------------------------------------------------------

    for index in range(
        len(path) - 1
    ):

        current_state = path[
            index
        ]

        next_state = path[
            index + 1
        ]


        # ----------------------------------------------------
        # Smooth movement
        # ----------------------------------------------------

        interpolation_steps = 8


        for frame in range(
            interpolation_steps
        ):

            amount = (

                frame
                / interpolation_steps

            )


            x, y = interpolate(

                current_state,

                next_state,

                amount

            )


            draw_maze_base()

            draw_path(

                path,

                (
                    index + 1
                )
                / max(
                    1,
                    len(path)
                )

            )

            draw_agent(
                x,
                y
            )


            draw_hud(

                episode,

                reward,

                index + 1,

                epsilon,

                success,

                "NAVIGATING..."

            )


            draw_learning_graph(
                episode
            )


            fig.canvas.draw()

            fig.canvas.flush_events()


            time.sleep(
                speed
            )


    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------

    final_x, final_y = state_to_xy(
        path[-1]
    )


    draw_maze_base()

    draw_path(
        path,
        1.0
    )

    draw_agent(
        final_x,
        final_y
    )


    final_status = (

        "GOAL REACHED"

        if success

        else

        "EXPLORATION ENDED"

    )


    draw_hud(

        episode,

        reward,

        total_steps,

        epsilon,

        success,

        final_status

    )


    draw_learning_graph(
        episode
    )


    fig.canvas.draw()

    fig.canvas.flush_events()


    if success:

        time.sleep(
            0.7
        )


# ============================================================
# INTRO
# ============================================================

print(
    "Initializing AI laboratory..."
)

time.sleep(
    1
)

print(
    "Loading training history..."
)

time.sleep(
    0.5
)

print(
    "Loading agent behavior..."
)

time.sleep(
    0.5
)

print(
    "Starting cinematic replay..."
)

time.sleep(
    1
)


# ============================================================
# PLAY TRAINING JOURNEY
# ============================================================

for episode in history:

    episode_number = episode[
        "episode"
    ]


    # --------------------------------------------------------
    # We only show selected episodes.
    #
    # This keeps the presentation cinematic rather than
    # showing 1000 episodes one by one.
    # --------------------------------------------------------

    if (

        episode_number == 1

        or episode_number == 4

        or episode_number == 50

        or episode_number == 100

        or episode_number == 156

        or episode_number == 200

        or episode_number == 250

        or episode_number == 300

        or episode_number == 324

        or episode_number == 500

        or episode_number == 750

        or episode_number == 1000

    ):

        # ----------------------------------------------------
        # Milestone
        # ----------------------------------------------------

        if episode_number in MILESTONES:

            milestone_screen(

                episode_number,

                MILESTONES[
                    episode_number
                ]

            )


        # ----------------------------------------------------
        # Determine playback speed
        # ----------------------------------------------------

        if episode_number in [

            1,
            4,
            156,
            324,
            1000

        ]:

            speed = MILESTONE_DELAY

        else:

            speed = FRAME_DELAY


        # ----------------------------------------------------
        # Replay
        # ----------------------------------------------------

        print(

            f"Episode {episode_number:4d} "

            f"| Reward: "

            f"{episode['total_reward']:4d} "

            f"| Steps: "

            f"{len(episode['steps']):3d}"

        )


        replay_episode(

            episode,

            speed

        )


# ============================================================
# FINAL SCREEN
# ============================================================

ax_maze.clear()

ax_hud.clear()

ax_graph.clear()


ax_maze.set_facecolor(
    "#080b12"
)

ax_hud.set_facecolor(
    "#080b12"
)

ax_graph.set_facecolor(
    "#080b12"
)


ax_maze.axis(
    "off"
)

ax_hud.axis(
    "off"
)

ax_graph.axis(
    "off"
)


fig.text(

    0.5,
    0.63,

    "TRAINING COMPLETE",

    ha="center",

    fontsize=30,

    fontweight="bold",

    color="#00e5ff"

)


fig.text(

    0.5,
    0.54,

    "1000 EPISODES",

    ha="center",

    fontsize=15,

    color="#8c9ab3"

)


fig.text(

    0.5,
    0.45,

    "EXPLORATION  →  DISCOVERY  →  OPTIMIZATION",

    ha="center",

    fontsize=12,

    color="white"

)


fig.text(

    0.5,
    0.39,

    "→  STABLE POLICY",

    ha="center",

    fontsize=12,

    color="#00e5ff"

)


fig.text(

    0.5,
    0.27,

    "Q-LEARNING AGENT",

    ha="center",

    fontsize=10,

    color="#53627a"

)


fig.canvas.draw()

fig.canvas.flush_events()


print()

print("=" * 70)

print(
    "             CINEMATIC V3 COMPLETE"
)

print("=" * 70)

print()

print(
    "Close the window to finish."
)

print()


plt.ioff()

plt.show()