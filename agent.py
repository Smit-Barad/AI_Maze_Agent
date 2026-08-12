import numpy as np
import random


class QLearningAgent:

    def __init__(
        self,
        rows,
        cols,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    ):
        self.rows = rows
        self.cols = cols

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # 4 actions:
        # 0 = UP
        # 1 = DOWN
        # 2 = LEFT
        # 3 = RIGHT
        self.q_table = np.zeros((rows, cols, 4))

    def choose_action(self, state):

        row, col = state

        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, 3)

        # Exploitation
        return np.argmax(self.q_table[row, col])

    def learn(self, state, action, reward, next_state, done):

        row, col = state
        next_row, next_col = next_state

        current_q = self.q_table[row, col, action]

        if done:
            target_q = reward

        else:
            best_future_q = np.max(
                self.q_table[next_row, next_col]
            )

            target_q = reward + self.discount_factor * best_future_q

        # Q-learning update
        self.q_table[row, col, action] += (
            self.learning_rate *
            (target_q - current_q)
        )

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )