import gymnasium as gym
import numpy as np
import random
import time
import os


# --- CONSTANTS
EPSILON_MAX:float = 1.0
EPSILON_MIN:float = 0.1    # not 0, so it has the chance to escape local maxima
EPSILON_DECAY:float = 0.0005

SAVE_TABLE_INTERVAL:int = 1000

MAX_STEPS_PER_EPISODE:int = 99
N_TRAINING_EPISODES = 20000

DISCOUNT_RATE_GAMMA:float = 0.9
LEARNING_RATE:float = 0.9


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
    q_table = np.random.uniform(low=0.5, high=1, size=(env.observation_space.n, env.action_space.n))
    
    
else:
    if not os.path.exists(table_path):
        print("No weights exists, train an agent first...")
        raise FileNotFoundError()
        
    else:
        print("Loading existing weights...")
        q_table = np.load(table_path)


print(q_table, q_table.shape)


# --- update function
def train_table(old_state:int, current_state:int, last_action:int|None, chosen_action:int, reward_current:float, q_table:np.ndarray, approach:str="Q-LEARNING"):
    """
    This is called after our agent took an action (chosen_action) based on a state (last_state). 

    old_state: S_t (the state of the state-action pair we are training)
    current_state: S_{t+1} (the state we got to by taking a)
    last_action: A_t (the action of the state-action pair we are training - the one that brough us from old_state to current_state)
    chosen_action: a (the action our policy just took in current_state - used for SARSA)
    reward_current: R_{t+1} (the reward we got from the current_state)
    q_table
    approach: either "SARSA" or "Q-LEARNING" - latter one by default
    
    If last_action is None, we know that this is the first step - we can't train on that and thus return the same table.
    
    Returns the updated q_table based on the TD method for n=1.
    """
    if last_action == None:
        return q_table
    
    
    # whether to assume that agent will follow value-function (Q-LEARNING) or account for random actions (SARSA)
    if approach == "Q-LEARNING":
        predicted_best_action:int = int(np.argmax(q_table[current_state][:]))
        discounted_value_estimation_next:float = DISCOUNT_RATE_GAMMA * q_table[current_state][predicted_best_action]
    elif approach == "SARSA":
        discounted_value_estimation_next:float = DISCOUNT_RATE_GAMMA * q_table[current_state][chosen_action]
    else:
        raise ValueError(f"Approach {approach} is not valid.")
        
        
    # updata and return
    value_estimation_last:float = q_table[old_state][last_action] 
    td_target:float = reward_current + discounted_value_estimation_next
    q_table[old_state][last_action] = ((1 - LEARNING_RATE) * value_estimation_last) + (LEARNING_RATE * td_target)
    return q_table



# --- action function
def take_action(state:int, q_table:np.ndarray, epsilon:float) -> int:
    """
    Based on the value of `episolon`, we take a random action for exploration.
    Otherwise we take the argmax of our `q_table` for exploitation.
    
    Epsilon is calculated by subtracting EPSILON_DECAY current_episode times, however we never return anything
    lower thatn EPSILON_MIN (ensured by max function over them both)
    """
    
    random_sigmoid:float = random.uniform(0.0, 1.0)
    
    if random_sigmoid < epsilon:
        # includes low number, excludes high number - gymnasium actions start at 0
        return int(np.random.randint(0, q_table.shape[1])) 
    
    else:
        # gets the index of the highest value in an array
        return int(np.argmax(q_table[state][:]))
    

# --- loop
total_steps:int = 0
total_reward:float = 0.0

for current_episode in range(n_episodes):
    state, info = env.reset()
    episode_over:bool = False
    action = None
    
    terminated:bool = False
    truncated:bool = False
    

    # iterates for each step, trains and saves table to disk if training is True
    while not (terminated or truncated):
        last_action = action
        
        
        # in training, we use epsilon to take random acitons, otherwise always the one with highest predicted value
        if training:
            epsilon:float = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * np.exp(-EPSILON_DECAY * current_episode)
        else:
            epsilon:float = 0
            
        # take an action, can be random or not
        action = take_action(state, q_table, epsilon)
       

        # `.step` takes an "ActType", which can be an integer corresponding to the position in the `action_space`
        old_state:int = state
        state, reward, terminated, truncated, info = env.step(action)
        
        
        # --- reward structure: default reward is 1 one reaching and 0 for everything else
        # reached gaol
        if terminated and (reward == 1):
            reward = 10  
             
        # hole
        elif terminated and (reward == 0):
                reward = -5
        
        # hit wall
        elif state == old_state:
            reward = -1
            
        # normal movement - small penalty to encourage efficiency
        else:
            reward = -0.01
            
        
        # update table
        if training:
            q_table = train_table(old_state, state, last_action, action, reward, q_table, "Q-LEARNING")
            
            
        # for metrics
        total_steps += 1
        total_reward += reward
        
        
    if (training) and (current_episode % SAVE_TABLE_INTERVAL) == 0:
        np.save(table_path, q_table)
        print(f"Epoch {str(current_episode)} had a total reward of {str(total_reward)} over the last {str(total_steps)} steps, saving...")
        total_reward = 0
        total_steps = 0
    
    
print("-- THE END, saving and existing... --")
np.save(table_path, q_table)
time.sleep(1)
env.close()