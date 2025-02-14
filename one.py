import gymnasium as gym
import numpy as np
import random
import time
import os


# --- CONSTANTS
EPSILON_MAX:float = 1.0
EPSILON_MIN:float = 0.1    # not 0, so it has the chance to escape local maxima
EPSILON_DECAY:float = 0.005

MAX_STEPS_PER_EPISODE:int = 99

SAVE_TABLE_INTERVAL:int = 1000
N_TRAINING_EPISODES = 10000

DISCOUNT_RATE_GAMMA:float = 0.99 # 0.9 - 0.99
LEARNING_RATE:float = 0.5 # 0.5 - 0.9

REWARD_GOAL:int = 1
REWARD_HOLE:int = 0
REWARD_WALL:int = 0
REWARD_MOVE:int = 0

APPROACH = "Q-LEARNING"


# ---
resp:str = input("Do you want to train (1) or evaluate (2)? > ")

if resp == "1":
    print("Entering training mode...")
    training:bool = True
    render_mode = None
    
    n_episodes = N_TRAINING_EPISODES
    
elif resp == "2":
    print("Entering eval mode...")
    training:bool = False
    n_episodes:int = 10
    # "human" will use pygame to display it (requires `pip install "gymnasium[toy-text]"`)
    # otherwise we can do "rgb_array" to get an array by calling `.render()` which we can feed into for example "pyvirtualdisplay" 
    # see https://gymnasium.farama.org/api/env/#gymnasium.Env.render
    render_mode = "human"
    
else: 
    raise ValueError()


# --- constructing the environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode=render_mode, max_episode_steps=MAX_STEPS_PER_EPISODE)
    
# these return Discrete Type
print(env.action_space) # possible actions - 0:left, 1:down, 2:right, 3:up
print(env.observation_space) # observable states - one for each field


# --- Q-table (our value function)
table_path:str = "./q_table.npy"

if training:
    print("Initializing new weights...")
    
    # random initialization encourages more exploration
    # q_table = np.random.uniform(low=0.5, high=1, size=(env.observation_space.n, env.action_space.n))
    
    # zeros
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    
else:
    if not os.path.exists(table_path):
        print("No weights exists, train an agent first...")
        raise FileNotFoundError()
        
    else:
        print("Loading existing weights...")
        q_table = np.load(table_path)


print(q_table, "\n\n")
    

# --- loop
total_steps:int = 0
total_reward:float = 0.0

for current_episode in range(n_episodes):
    state, info = env.reset()
    episode_over:bool = False
    
    terminated:bool = False
    truncated:bool = False
    action = None
    

    # iterates for each step, trains and saves table to disk if training is True
    while not (terminated or truncated):
        last_action = action

        # decide whether to take predicted best or random action
        if training:
            epsilon = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * np.exp(-EPSILON_DECAY * current_episode)
        else:
            epsilon = 0
            
        if random.uniform(0.0, 1.0) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        
        old_state = state
        state, reward, terminated, truncated, info = env.step(action)
        
        
        # --- Adjust reward structure
        if terminated and (reward == 1):
            reward = REWARD_GOAL
        elif terminated and (reward == 0):
            reward = REWARD_HOLE
        elif state == old_state:
            reward = REWARD_WALL
        else:
            reward = REWARD_MOVE

        # --- update table
        if training and (last_action is not None):
            if terminated:
                discounted_value_estimation_next = 0 
            else:
                if APPROACH == "Q-LEARNING":
                    predicted_best_action = int(np.argmax(q_table[state][:]))
                    discounted_value_estimation_next = DISCOUNT_RATE_GAMMA * q_table[state][predicted_best_action]
                elif APPROACH == "SARSA":
                    discounted_value_estimation_next = DISCOUNT_RATE_GAMMA * q_table[state][action]
                else:
                    raise ValueError(f"Approach {APPROACH} is not valid.")

            value_estimation_last = q_table[old_state][last_action]
            td_target = reward + discounted_value_estimation_next
            q_table[old_state][last_action] = ((1 - LEARNING_RATE) * value_estimation_last) + (LEARNING_RATE * td_target)


        # --- record metrics
        total_steps += 1
        total_reward += reward
        
        
    # if in training and episode is dividible by INTERVAL (except for 0) then save weights and print metrics
    if (training) and ((current_episode % SAVE_TABLE_INTERVAL) == 0) and (current_episode != 0):
        np.save(table_path, q_table)
        print(f"Epoch {str(current_episode)} had an average reward of {str(total_reward / SAVE_TABLE_INTERVAL)} and setps {str(total_steps / SAVE_TABLE_INTERVAL)} per episode, currently at epsilon {epsilon}, saving...")
        total_reward = 0
        total_steps = 0
    
    
print("-- THE END, saving and existing... --")
np.save(table_path, q_table)
print("\n\n", q_table)
time.sleep(1)
env.close()