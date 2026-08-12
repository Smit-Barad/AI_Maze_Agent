import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def draw_maze(environment, agent_position=None, title="AI Maze"):

    maze = environment.maze

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(
        maze,
        cmap="Greys",
        vmin=0,
        vmax=1
    )

    # Grid lines
    ax.set_xticks(
        np.arange(-0.5, maze.shape[1], 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, maze.shape[0], 1),
        minor=True
    )

    ax.grid(
        which="minor",
        linewidth=1
    )

    ax.set_xticks([])
    ax.set_yticks([])

    # Start
    start_row, start_col = environment.start

    ax.text(
        start_col,
        start_row,
        "START",
        ha="center",
        va="center",
        fontsize=9
    )

    # Goal
    goal_row, goal_col = environment.goal

    ax.text(
        goal_col,
        goal_row,
        "GOAL",
        ha="center",
        va="center",
        fontsize=9
    )

    # Agent
    if agent_position is not None:

        row, col = agent_position

        ax.plot(
            col,
            row,
            marker="o",
            markersize=18
        )

        ax.text(
            col,
            row,
            "AI",
            ha="center",
            va="center",
            fontsize=8
        )

    ax.set_title(title)

    plt.show()


def get_best_action(agent, state):

    row, col = state

    return int(
        np.argmax(
            agent.q_table[row, col]
        )
    )


def get_learned_path(
    environment,
    agent,
    max_steps=100
):

    state = environment.reset()

    path = [state]

    for _ in range(max_steps):

        action = get_best_action(
            agent,
            state
        )

        next_state, reward, done = environment.step(
            action
        )

        # Agent hit wall/boundary
        if next_state == state:
            break

        state = next_state

        path.append(state)

        if done:
            break

    return path


def animate_path(
    environment,
    path,
    interval=400
):

    maze = environment.maze

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw maze
    ax.imshow(
        maze,
        cmap="Greys",
        vmin=0,
        vmax=1
    )

    # Grid
    ax.set_xticks(
        np.arange(-0.5, maze.shape[1], 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, maze.shape[0], 1),
        minor=True
    )

    ax.grid(
        which="minor",
        linewidth=1
    )

    ax.set_xticks([])
    ax.set_yticks([])

    # Start
    start_row, start_col = environment.start

    ax.text(
        start_col,
        start_row,
        "START",
        ha="center",
        va="center",
        fontsize=9
    )

    # Goal
    goal_row, goal_col = environment.goal

    ax.text(
        goal_col,
        goal_row,
        "GOAL",
        ha="center",
        va="center",
        fontsize=9
    )

    # Agent marker
    agent_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=18
    )

    # Title
    title = ax.set_title(
        "AI is navigating..."
    )

    def update(frame):

        row, col = path[frame]

        agent_marker.set_data(
            [col],
            [row]
        )

        title.set_text(
            f"AI Navigation — Step {frame + 1}/{len(path)}"
        )

        return agent_marker, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(path),
        interval=interval,
        repeat=False
    )

    plt.show()

    return animation


def plot_training_results(
    rewards_history,
    steps_history
):

    episodes = range(
        1,
        len(rewards_history) + 1
    )

    # ----------------------------
    # Reward graph
    # ----------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        episodes,
        rewards_history
    )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.title(
        "AI Learning Progress — Reward"
    )

    plt.grid(True)

    plt.show()

    # ----------------------------
    # Steps graph
    # ----------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        episodes,
        steps_history
    )

    plt.xlabel("Episode")
    plt.ylabel("Steps")

    plt.title(
        "AI Learning Progress — Steps per Episode"
    )

    plt.grid(True)

    plt.show()