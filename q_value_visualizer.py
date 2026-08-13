import json
import numpy as np
import matplotlib.pyplot as plt

from environment import MazeEnvironment
from agent import QLearningAgent


# ============================================================
# CONFIGURATION
# ============================================================

TOTAL_EPISODES = 1000

SNAPSHOT_EPISODES = [
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
# CREATE ENVIRONMENT
# ============================================================

env = MazeEnvironment()


# ============================================================
# CREATE AGENT
# ============================================================

agent = QLearningAgent(

    rows=6,

    cols=6,

    learning_rate=0.1,

    discount_factor=0.9,

    epsilon=1.0,

    epsilon_decay=0.995,

    epsilon_min=0.01

)


# ============================================================
# STORAGE
# ============================================================

q_snapshots = {}

training_rewards = []

training_steps = []

training_success = []


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("          Q-VALUE LEARNING VISUALIZATION")
print("=" * 70)
print()

print(
    "Training a fresh agent and capturing Q-table snapshots..."
)

print()


for episode_number in range(
    1,
    TOTAL_EPISODES + 1
):

    state = env.reset()

    total_reward = 0

    steps = 0

    success = False


    # --------------------------------------------------------
    # Episode
    # --------------------------------------------------------

    for _ in range(100):

        action = agent.choose_action(
            state
        )


        next_state, reward, done = env.step(
            action
        )


        agent.learn(

            state,

            action,

            reward,

            next_state,

            done

        )


        state = next_state

        total_reward += reward

        steps += 1


        if done:

            success = True

            break


    # --------------------------------------------------------
    # Epsilon decay
    # --------------------------------------------------------

    agent.decay_epsilon()


    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    training_rewards.append(
        total_reward
    )

    training_steps.append(
        steps
    )

    training_success.append(
        success
    )


    # --------------------------------------------------------
    # Snapshot
    # --------------------------------------------------------

    if episode_number in SNAPSHOT_EPISODES:

        q_snapshots[
            episode_number
        ] = agent.q_table.copy()


    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    if episode_number % 100 == 0:

        print(

            f"Episode {episode_number:4d} | "

            f"Reward: {total_reward:4d} | "

            f"Steps: {steps:3d} | "

            f"Epsilon: {agent.epsilon:.3f}"

        )


# ============================================================
# SAVE SNAPSHOTS
# ============================================================

np.savez(

    "q_learning_snapshots.npz",

    **{

        f"episode_{episode}":
        q_table

        for episode, q_table
        in q_snapshots.items()

    }

)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "episodes": SNAPSHOT_EPISODES,

    "learning_rate":
    agent.learning_rate,

    "discount_factor":
    agent.discount_factor,

    "epsilon_decay":
    agent.epsilon_decay,

    "epsilon_min":
    agent.epsilon_min

}


with open(

    "q_learning_metadata.json",

    "w"

) as file:

    json.dump(

        metadata,

        file,

        indent=4

    )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()

print("=" * 70)

print(
    "             Q-TABLE SNAPSHOTS READY"
)

print("=" * 70)

print()

print(
    "Saved:"
)

print(
    "  q_learning_snapshots.npz"
)

print(
    "  q_learning_metadata.json"
)

print()

print(
    "Captured episodes:"
)

print(
    SNAPSHOT_EPISODES
)

print()