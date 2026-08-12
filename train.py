from environment import MazeEnvironment
from agent import QLearningAgent


# Create environment
env = MazeEnvironment()

# Create AI agent
agent = QLearningAgent(
    rows=6,
    cols=6
)


# Training settings
episodes = 1000
max_steps = 100


# Store training results
rewards_history = []
steps_history = []


print("Starting training...\n")


for episode in range(episodes):

    # Reset maze
    state = env.reset()

    total_reward = 0
    steps = 0

    for step in range(max_steps):

        # AI chooses action
        action = agent.choose_action(state)

        # Environment responds
        next_state, reward, done = env.step(action)

        # AI learns from experience
        agent.learn(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Move to next state
        state = next_state

        total_reward += reward
        steps += 1

        # Stop if goal reached
        if done:
            break

    # Reduce exploration over time
    agent.decay_epsilon()

    rewards_history.append(total_reward)
    steps_history.append(steps)

    # Print progress
    if (episode + 1) % 100 == 0:

        print(
            f"Episode {episode + 1:4d} | "
            f"Reward: {total_reward:4d} | "
            f"Steps: {steps:3d} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )


print("\nTraining complete!")