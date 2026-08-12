import copy
import json


class TrainingRecorder:

    def __init__(self):

        self.episodes = []


    def start_episode(
        self,
        episode_number,
        epsilon
    ):

        self.current_episode = {

            "episode": episode_number,

            "epsilon": epsilon,

            "steps": [],

            "total_reward": 0,

            "success": False

        }


    def record_step(
        self,
        state,
        action,
        reward,
        next_state,
        q_values
    ):

        step_data = {

            "state": list(state),

            "action": int(action),

            "reward": float(reward),

            "next_state": list(next_state),

            "q_values": copy.deepcopy(
                q_values.tolist()
            )

        }

        self.current_episode[
            "steps"
        ].append(
            step_data
        )

        self.current_episode[
            "total_reward"
        ] += reward


    def finish_episode(
        self,
        success
    ):

        self.current_episode[
            "success"
        ] = success

        self.episodes.append(
            self.current_episode
        )


    def get_episode(
        self,
        episode_number
    ):

        index = episode_number - 1

        if (
            index < 0
            or index >= len(self.episodes)
        ):

            return None

        return self.episodes[index]


    def total_episodes(self):

        return len(
            self.episodes
        )


    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    def save(
        self,
        filename="training_history.json"
    ):

        with open(
            filename,
            "w"
        ) as file:

            json.dump(
                self.episodes,
                file,
                indent=2
            )


        print()

        print(
            f"Training history saved to: "
            f"{filename}"
        )


    # ========================================================
    # LOAD TRAINING HISTORY
    # ========================================================

    def load(
        self,
        filename="training_history.json"
    ):

        with open(
            filename,
            "r"
        ) as file:

            self.episodes = json.load(
                file
            )


        print()

        print(
            f"Training history loaded from: "
            f"{filename}"
        )