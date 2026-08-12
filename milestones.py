import json


# ============================================================
# LOAD HISTORY
# ============================================================

with open(
    "training_history.json",
    "r"
) as file:

    history = json.load(file)


# ============================================================
# BASIC STATISTICS
# ============================================================

total_episodes = len(history)


successful_episodes = [

    episode

    for episode in history

    if episode["success"]
]


# ============================================================
# FIRST SUCCESS
# ============================================================

first_success = None

if successful_episodes:

    first_success = successful_episodes[0]


# ============================================================
# BEST EPISODE
# ============================================================

best_episode = max(

    history,

    key=lambda episode:
        episode["total_reward"]

)


# ============================================================
# SHORTEST SUCCESSFUL EPISODE
# ============================================================

shortest_success = None

if successful_episodes:

    shortest_success = min(

        successful_episodes,

        key=lambda episode:
            len(episode["steps"])

    )


# ============================================================
# FIRST 10-STEP SOLUTION
# ============================================================

first_10_step = None

for episode in successful_episodes:

    if len(episode["steps"]) <= 10:

        first_10_step = episode

        break


# ============================================================
# PRINT REPORT
# ============================================================

print()

print("=" * 60)

print(
    "          AI LEARNING MILESTONES"
)

print("=" * 60)

print()


print(
    f"Total episodes recorded : "
    f"{total_episodes}"
)

print()


# ------------------------------------------------------------
# FIRST SUCCESS
# ------------------------------------------------------------

if first_success:

    print(
        "🎯 FIRST SUCCESS"
    )

    print(
        f"Episode : "
        f"{first_success['episode']}"
    )

    print(
        f"Steps   : "
        f"{len(first_success['steps'])}"
    )

    print(
        f"Reward  : "
        f"{first_success['total_reward']}"
    )

else:

    print(
        "No successful episode found."
    )


print()


# ------------------------------------------------------------
# BEST REWARD
# ------------------------------------------------------------

print(
    "🏆 BEST REWARD"
)

print(
    f"Episode : "
    f"{best_episode['episode']}"
)

print(
    f"Reward  : "
    f"{best_episode['total_reward']}"
)

print(
    f"Steps   : "
    f"{len(best_episode['steps'])}"
)


print()


# ------------------------------------------------------------
# SHORTEST SUCCESS
# ------------------------------------------------------------

if shortest_success:

    print(
        "⚡ SHORTEST SUCCESSFUL PATH"
    )

    print(
        f"Episode : "
        f"{shortest_success['episode']}"
    )

    print(
        f"Steps   : "
        f"{len(shortest_success['steps'])}"
    )

    print(
        f"Reward  : "
        f"{shortest_success['total_reward']}"
    )


print()


# ------------------------------------------------------------
# FIRST 10-STEP SOLUTION
# ------------------------------------------------------------

if first_10_step:

    print(
        "🔥 FIRST 10-STEP SOLUTION"
    )

    print(
        f"Episode : "
        f"{first_10_step['episode']}"
    )

    print(
        f"Reward  : "
        f"{first_10_step['total_reward']}"
    )

else:

    print(
        "No 10-step solution found."
    )


print()


# ============================================================
# SUCCESS RATE OVER TIME
# ============================================================

print(
    "📈 SUCCESS RATE CHECKPOINTS"
)

print()


checkpoints = [

    50,
    100,
    200,
    300,
    500,
    750,
    1000

]


for checkpoint in checkpoints:

    if checkpoint > total_episodes:

        continue


    subset = history[
        :checkpoint
    ]


    successes = sum(

        1

        for episode in subset

        if episode["success"]

    )


    success_rate = (
        successes
        / checkpoint
        * 100
    )


    print(

        f"Episodes 1-{checkpoint:4d} "
        f"| Success rate: "
        f"{success_rate:6.2f}%"

    )


print()


# ============================================================
# COMPLETE
# ============================================================

print("=" * 60)

print(
    "              ANALYSIS COMPLETE"
)

print("=" * 60)

print()