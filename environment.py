import numpy as np


class MazeEnvironment:

    def __init__(self):
        self.maze = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        self.start = (0, 0)
        self.goal = (5, 5)

        self.position = self.start

    def reset(self):
        self.position = self.start
        return self.position

    def step(self, action):

        row, col = self.position

        if action == 0:      # UP
            new_row, new_col = row - 1, col

        elif action == 1:    # DOWN
            new_row, new_col = row + 1, col

        elif action == 2:    # LEFT
            new_row, new_col = row, col - 1

        elif action == 3:    # RIGHT
            new_row, new_col = row, col + 1

        else:
            raise ValueError("Invalid action")

        # Check boundaries
        if (
            new_row < 0
            or new_row >= self.maze.shape[0]
            or new_col < 0
            or new_col >= self.maze.shape[1]
        ):
            return self.position, -5, False

        # Check wall
        if self.maze[new_row, new_col] == 1:
            return self.position, -5, False

        # Move
        self.position = (new_row, new_col)

        # Goal reached
        if self.position == self.goal:
            return self.position, 100, True

        # Normal movement
        return self.position, -1, False