import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# CONFIGURATION
# ============================================================

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
# LOAD SNAPSHOTS
# ============================================================

data = np.load(
    SNAPSHOT_FILE
)


# ============================================================
# FIGURE
# ============================================================

plt.ion()


fig, ax = plt.subplots(
    figsize=(8, 8)
)


# ============================================================
# DRAW POLICY
# ============================================================

def draw_policy(
    q_table,
    episode
):

    ax.clear()


    # --------------------------------------------------------
    # Find maximum Q value for color intensity
    # --------------------------------------------------------

    max_abs_q = np.max(
        np.abs(q_table)
    )


    if max_abs_q == 0:

        max_abs_q = 1


    # --------------------------------------------------------
    # Draw grid
    # --------------------------------------------------------

    for row in range(ROWS):

        for col in range(COLS):

            y = ROWS - row - 1


            # ------------------------------------------------
            # Wall
            # ------------------------------------------------

            if MAZE[row, col] == 1:

                ax.add_patch(

                    Rectangle(

                        (col, y),

                        1,

                        1

                    )

                )

                continue


            # ------------------------------------------------
            # Cell boundary
            # ------------------------------------------------

            ax.add_patch(

                Rectangle(

                    (col, y),

                    1,

                    1,

                    fill=False

                )

            )


            # ------------------------------------------------
            # Start
            # ------------------------------------------------

            if (
                row,
                col
            ) == START:

                ax.text(

                    col + 0.18,

                    y + 0.80,

                    "START",

                    fontsize=8

                )


            # ------------------------------------------------
            # Goal
            # ------------------------------------------------

            if (
                row,
                col
            ) == GOAL:

                ax.text(

                    col + 0.50,

                    y + 0.50,

                    "★",

                    fontsize=25,

                    ha="center",

                    va="center"

                )

                continue


            # ------------------------------------------------
            # Q values
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

            ax.text(

                col + 0.50,

                y + 0.55,

                ARROWS[
                    best_action
                ],

                fontsize=28,

                ha="center",

                va="center"

            )


            # ------------------------------------------------
            # Q value
            # ------------------------------------------------

            ax.text(

                col + 0.50,

                y + 0.16,

                f"{best_value:.1f}",

                fontsize=8,

                ha="center",

                va="center"

            )


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(

        f"Q-LEARNING POLICY — EPISODE {episode}",

        fontsize=16,

        fontweight="bold"

    )


    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        COLS
    )

    ax.set_ylim(
        0,
        ROWS
    )


    ax.set_xticks(
        range(COLS + 1)
    )

    ax.set_yticks(
        range(ROWS + 1)
    )


    ax.grid(
        True
    )


    ax.set_aspect(
        "equal"
    )


    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    ax.text(

        0,

        -0.45,

        "Arrow = action with highest learned Q-value",

        fontsize=10

    )


    plt.pause(
        1.0
    )


# ============================================================
# MAIN ANIMATION
# ============================================================

print()

print("=" * 70)

print(
    "             Q-POLICY ANIMATION"
)

print("=" * 70)

print()


for episode in EPISODES:

    key = f"episode_{episode}"


    if key not in data:

        print(
            f"Missing snapshot: {key}"
        )

        continue


    print(
        f"Showing learned policy at Episode {episode}"
    )


    q_table = data[
        key
    ]


    draw_policy(

        q_table,

        episode

    )


# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 70)

print(
    "          POLICY VISUALIZATION COMPLETE"
)

print("=" * 70)

print()

print(
    "Close the graph window to finish."
)

plt.ioff()

plt.show()