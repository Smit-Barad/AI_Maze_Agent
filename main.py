from environment import MazeEnvironment
from agent import QLearningAgent

from training_recorder import TrainingRecorder


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_EPISODES = 1000

MAX_STEPS = 100


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
# CREATE RECORDER
# ============================================================

recorder = TrainingRecorder()


# ============================================================
# TRAINING
# ============================================================

print()

print("=" * 60)

print(
    "       AI MAZE AGENT — RECORDING TRAINING"
)

print("=" * 60)

print()


for episode in range(
    TRAINING_EPISODES
):


    # --------------------------------------------------------
    # RESET ENVIRONMENT
    # --------------------------------------------------------

    state = env.reset()


    # --------------------------------------------------------
    # RECORD EPISODE START
    # --------------------------------------------------------

    recorder.start_episode(

        episode_number=episode + 1,

        epsilon=agent.epsilon

    )


    total_reward = 0

    success = False


    # --------------------------------------------------------
    # EPISODE LOOP
    # --------------------------------------------------------

    for step in range(
        MAX_STEPS
    ):


        # ----------------------------------------------------
        # SAVE CURRENT Q VALUES
        # ----------------------------------------------------

        row, col = state


        q_values = agent.q_table[
            row,
            col
        ]


        # ----------------------------------------------------
        # CHOOSE ACTION
        # ----------------------------------------------------

        action = agent.choose_action(
            state
        )


        # ----------------------------------------------------
        # ENVIRONMENT RESPONSE
        # ----------------------------------------------------

        next_state, reward, done = env.step(
            action
        )


        # ----------------------------------------------------
        # RECORD STEP
        # ----------------------------------------------------

        recorder.record_step(

            state=state,

            action=action,

            reward=reward,

            next_state=next_state,

            q_values=q_values

        )


        # ----------------------------------------------------
        # LEARN
        # ----------------------------------------------------

        agent.learn(

            state,

            action,

            reward,

            next_state,

            done

        )


        # ----------------------------------------------------
        # UPDATE
        # ----------------------------------------------------

        state = next_state

        total_reward += reward


        # ----------------------------------------------------
        # GOAL
        # ----------------------------------------------------

        if done:

            success = True

            break


    # --------------------------------------------------------
    # FINISH EPISODE
    # --------------------------------------------------------

    recorder.finish_episode(
        success
    )


    # --------------------------------------------------------
    # EPSILON DECAY
    # --------------------------------------------------------

    agent.decay_epsilon()


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        episode + 1
    ) % 100 == 0:

        print(

            f"Episode "
            f"{episode + 1:4d} | "

            f"Reward: "
            f"{total_reward:4d} | "

            f"Steps: "
            f"{step + 1:3d} | "

            f"Epsilon: "
            f"{agent.epsilon:.3f}"

        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()

print("=" * 60)

print(
    "             TRAINING COMPLETE"
)

print("=" * 60)

print()


print(
    "Recorded episodes:",
    recorder.total_episodes()
)


# ============================================================
# SHOW SOME RECORDED INFORMATION
# ============================================================

print()

print(
    "Checking recorded history..."
)


for episode_number in [
    1,
    10,
    50,
    100,
    250,
    500,
    750,
    1000
]:

    episode = recorder.get_episode(
        episode_number
    )


    if episode is None:

        continue


    print()

    print(
        f"Episode {episode_number}"
    )

    print(
        f"Steps: "
        f"{len(episode['steps'])}"
    )

    print(
        f"Reward: "
        f"{episode['total_reward']}"
    )

    print(
        f"Epsilon: "
        f"{episode['epsilon']:.3f}"
    )

    print(
        f"Success: "
        f"{episode['success']}"
    )


# ============================================================
# DONE
# ============================================================

print()

print("=" * 60)

print(
    "         TRAINING HISTORY READY"
)

print("=" * 60)

print()

print(
    "Next step: visualize the complete "
    "learning journey."
)
recorder.save("training_history.json")