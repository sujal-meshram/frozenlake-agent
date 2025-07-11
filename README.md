
# Q-Learning on FrozenLake-v1

A simple implementation of Q-learning to solve OpenAI Gym’s `FrozenLake-v1` environment using Python and NumPy.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Algorithm & Hyperparameters](#algorithm--hyperparameters)  
5. [Results](#results)  
6. [Project Structure](#project-structure)  
7. [Contributing](#contributing)  
8. [License](#license)  

---

## Introduction

FrozenLake is a grid world where the agent seeks to navigate from start (S) to goal (G), avoiding holes (H) on slippery ice (F). This repo implements the classic tabular Q-learning algorithm:

1. **Initialize** Q-table to zeros.  
2. **For** each episode:  
   - Select actions via ε-greedy.  
   - Update Q-values with the Bellman equation.  
   - Decay ε over time to shift from exploration to exploitation.  
3. **Evaluate** the learned policy over 1,000 episodes.

---

## Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/sujal-meshram/frozenlake-agent.git
   cd frozenlake-agent
   ```

2. **Create & activate** a Python virtual environment (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # on macOS/Linux
   venv\Scripts\activate.bat       # on Windows
   ```

3. **Install dependencies**

   ```bash
   pip install gymnasium numpy
   ```

---

## Usage

Run the training & evaluation script:

```bash
python frozenlake.py
```

This will:

* Train for 10,000 episodes (configurable).
* Print average rewards per 1,000-episode blocks.
* Evaluate the learned policy over 1,000 episodes and print the final average reward.

---

## Algorithm & Hyperparameters

| Parameter               | Value  | Description                                   |
| ----------------------- | ------ | --------------------------------------------- |
| Episodes                | 10,000 | Total training episodes                       |
| Max steps per episode   | 100    | Max time‐steps per episode                    |
| Learning rate (α)       | 0.1    | Step‐size for updating Q-values               |
| Discount factor (γ)     | 0.99   | Future reward discount                        |
| Initial ε (exploration) | 1.0    | Starting exploration rate                     |
| Min ε                   | 0.01   | Minimum exploration rate                      |
| ε decay rate            | 0.001  | Controls exponential decay of ε over episodes |

You can tweak these in `q_learning_frozenlake.py` to see how they affect learning speed and stability.

---

## Results

After training, you should see an output like:

```
****Average rewards per thousand episodes****

1000 :  0.034
2000 :  0.078
3000 :  0.112
...
10000 : 0.694

Average rewards over thousand episodes after training: 0.72
```

These numbers will vary due to randomness and ε-greedy exploration. A higher final reward indicates a better‐learned policy.