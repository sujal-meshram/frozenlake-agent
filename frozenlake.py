import gymnasium as gym
import numpy as np
import random
import time

# Create FrozenLake environment
env = gym.make("FrozenLake-v1")


# Initialize the Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))


# Hyperparameters
no_episodes = 10000
max_steps = 100

alpha = 0.1         # Learning rate
gamma = 0.99        # Discount factor


# Exploration parameters
exploration_rate = 1
min_exploration_rate = 0.01
max_exploration_rate = 1
exploration_decay_rate = 0.001


# Track rewards
overall_rewards = []


# Training loop
for episode in range(no_episodes):
    state, _ = env.reset()
    done = False
    episode_rewards = 0
    
    for step in range(max_steps):

        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])   # Exploit
        else:
            action = env.action_space.sample()      # Explore
        
        # Take action
        new_state, reward, done, truncated, info = env.step(action)

        # Update Q-table
        q_table[state, action] = (1-alpha) * q_table[state, action] + alpha * (reward+gamma*np.max(q_table[new_state, :]))

        # Move to next state
        state = new_state
        episode_rewards += reward

        if done:
            break

    # Exponentially decay exploration rate
    exploration_rate = min_exploration_rate + (max_exploration_rate-min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    # Add rewards
    overall_rewards.append(episode_rewards)



# Training results
reward_per_thousand_eps = np.split(np.array(overall_rewards), no_episodes/1000)
count = 1000
print("****Average rewards per thousand episodes****\n")
for r in reward_per_thousand_eps:
    print(count, ": ", str(sum(r/1000)))
    count += 1000


# Testing loop
no_eval_episodes = 1000
rewards = 0

for episode in range(no_eval_episodes):
    state, _ = env.reset()
    done = False
    episode_rewards = 0

    for step in range(max_steps):
        # Use Q-table to select action
        action = np.argmax(q_table[state, :])
        new_state, reward, done, truncated, info = env.step(action)

        episode_rewards += reward

        if done:
            break

        state = new_state
    
    rewards += episode_rewards


# Testing results
print("\n\nAverage rewards over thousand episodes after training: ", str(rewards/1000))


# Close environment
env.close() 