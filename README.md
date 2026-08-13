# 🤖 AI Maze Agent — Learning Through Q-Learning

A visual Reinforcement Learning experiment where an AI agent learns to navigate a maze using **Q-Learning**.

The agent starts without knowing the correct path.

Through repeated interaction with the environment, it learns which actions lead toward better outcomes and gradually develops an effective navigation policy.

---

## 🎥 Project Demo

The experiment visualizes the agent's learning journey:

**Random Exploration → Discovery → Learning → Optimization → Stable Policy**

The agent begins by exploring the environment and eventually learns the shortest path to the goal.

---

## 🧠 What is the Agent Learning?

The agent interacts with the environment using:

```text
State → Action → Reward → Next State

For every action, the environment gives the agent a reward.

Example
Reach goal       → +100
Normal movement  → -1
Hit wall         → -5

The agent uses these rewards to update its Q-table.

Over time, actions that lead to better outcomes become more valuable.

🧮 Q-Learning

The project uses the standard Q-Learning update rule:

Q(s,a) ← Q(s,a) + α[
    r + γ max Q(s',a') - Q(s,a)
]

Where:

s = current state
a = current action
r = reward
s' = next state
α = learning rate
γ = discount factor

The agent also uses ε-greedy exploration.

At the beginning, it explores heavily.

As training progresses, epsilon decreases and the agent increasingly exploits what it has learned.

🗺️ Environment

The agent navigates a small grid-based maze.

S · · · · ·
· █ █ · █ ·
· · · · █ ·
█ █ · · · ·
· · · █ █ ·
· · · · · G

Where:

S = Start
G = Goal
█ = Wall
· = Free cell

The agent has four possible actions:

0 = UP
1 = DOWN
2 = LEFT
3 = RIGHT
📈 Learning Journey

The agent's behavior changes during training.

Early training

The agent explores randomly.

High exploration
      ↓
Random actions
      ↓
Wall collisions
      ↓
Negative rewards
During training

The Q-table begins to develop useful values.

Experience
    ↓
Q-value updates
    ↓
Better action selection
    ↓
Improved navigation
Later training

The agent increasingly follows a learned policy.

Exploration ↓
Exploitation ↑
      ↓
Stable navigation
🔬 What We Measure

The project records several aspects of learning:

Reward

How well the agent performed during an episode.

Steps

How many actions the agent needed to reach the goal.

Epsilon

The balance between exploration and exploitation.

Q-values

The learned value of taking each action from each state.

Policy

The preferred action at each state based on learned Q-values.

🎯 Learning Milestones

The training history can be analyzed to identify important moments such as:

First successful goal discovery
          ↓
Improving navigation
          ↓
First optimal solution
          ↓
Policy stabilization
          ↓
Final learned behavior

This makes it possible to visualize not just the final result, but the process of learning itself.

📊 Visualization Tools

The project includes several visualization scripts.

Training History
python main.py

Runs the main training experiment.

Learning Milestones
python milestones.py

Analyzes important moments in the training history.

Learning Analysis
python learning_analysis.py

Analyzes:

First successful episode
First optimal solution
Stable learning point
Final performance
Video milestones
Q-Value Visualization
python q_value_visualizer.py

Captures Q-table snapshots throughout training.

Policy Evolution
python q_policy_animation.py

Visualizes how the agent's preferred actions evolve as it learns.

Cinematic Replay
python cinematic_replay_v2.py

Creates a visual replay of the agent's learning journey.

The goal is to make the transition from random exploration to learned behavior easier to observe.

🛠️ Technologies
Python
NumPy
Matplotlib
Reinforcement Learning
Q-Learning
ε-greedy exploration
🚀 Installation
1. Clone the repository
git clone (https://github.com/Smit-Barad/AI_Maze_Agent.git)

Move into the project:

cd AI_Maze_Agent
2. Create a virtual environment

Windows:

python -m venv .venv

Activate it:

.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
▶️ Run the Project

Start training:

python main.py

Then explore the visualizations:

python milestones.py
python learning_analysis.py
python q_value_visualizer.py
python q_policy_animation.py
python cinematic_replay_v2.py
📁 Project Structure
AI_Maze_Agent/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── environment.py
├── agent.py
├── main.py
│
├── milestones.py
├── learning_analysis.py
│
├── q_value_visualizer.py
├── q_policy_animation.py
│
├── cinematic_replay_v2.py
│
├── training_history.json
├── q_learning_snapshots.npz
└── q_learning_metadata.json
🤝 Built With ChatGPT

This project was built with ChatGPT as an AI development and learning partner.

I used ChatGPT throughout the project to help with:

Understanding reinforcement learning concepts
Designing the environment
Implementing Q-Learning
Debugging Python code
Analyzing training behavior
Building visualization tools
Designing the learning replay
Iterating on the experiment

The goal was not simply to generate code, but to learn by building, testing, debugging, and iterating.
