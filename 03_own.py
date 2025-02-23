"""
REINFORCE algorithm on the CartPole environment
"""

import gymnasium
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# --- constants
N_EPISODES:int = 5000
MAX_STEPS_PER_EPISODE:int = 100
PRINT_INTERVAL:int = 500

DISCOUNT_RATE: float = 0.99
LEARNING_RATE: float = 0.01

# --- 
resp:str = input("Do you want to train (1) or evaluate (2)? > ")
training:bool = True if resp == "1" else False
render_mode = None if training else "human" 
weights_path = "./weights/03.pt"

# --- environment
env = gymnasium.make("CartPole-v1", max_episode_steps=MAX_STEPS_PER_EPISODE, render_mode=render_mode)
print(env.action_space) # 0 = push_left, 1= push_right
print(env.observation_space) # np.ndarray with shape (4,) of float32s - [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

# --- our action-function we want to optimize
class Policy(nn.Module):
    def __init__(self, observation_size:int, hidden_size:int, action_size:int):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(observation_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x =  self.l2(F.relu(self.l1(x)))
        return F.softmax(x, dim=1)
    
    def act(self, observation:np.ndarray):
        # convert to tensor
        # turn to float32 - already is here, but still done to ensure input are ALWAYS in the datatype our layers expect (see Datatypes in "low-level ML operations")
        # turn (4,) into (1, 4) - adds an "empty dimension" at index 0
        # move to device only when available - otherwise we keep on the CPU and profit from the shared memory
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(device) # (1, 4)
        
        # push it back to CPU (if needed), because we will need an np.ndarray again
        action_probabilites = self.forward(observation).cpu() # (2,)
        
        # (see Probability Distributions in "low-level ML operations")
        action_distribution = Categorical(action_probabilites)
        action = action_distribution.sample() # tensor of 0 or 1
        
        # turns the sample into a normal scalar
        # 
        return action.item(), action_distribution.log_prob(action)


# --- loop
policy = Policy(env.observation_space.shape[0], 16, env.action_space.n)
optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

if not training:
    policy.load_state_dict(torch.load(weights_path, weights_only=True))
    policy.eval()  # Set the policy to evaluation mode
    
    with torch.no_grad():
        num_eval_episodes = 10  # Number of episodes to evaluate
        eval_rewards = []
        
        for episode in range(num_eval_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action, _ = policy.act(state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
        
        print(f"Evaluation completed. Average reward over {num_eval_episodes} episodes: {np.mean(eval_rewards)}")

else:
    policy.train()
    
    # collects total losses and rewards for each episode - later used for printing it
    # instead of having this as an ndarray, which will be recreated with each new item
    # we only convert it to an array everytime we need to calculate the mean
    collected_losses:float = 0
    collected_rewards:float = 0

    for episode in range(1, N_EPISODES+1):
        state, _ = env.reset()
        step_data:list = [] # [[log_prob:tensor, reward:float], ...]
        
        terminated:bool = False
        truncated:bool = False
        
        
        while not (terminated or truncated):
            action, log_probs = policy.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            step_data.append([log_probs, reward])


        total_steps:int = len(step_data)    # T
        
        # collects the losses for each step in this episode - we then combine it to optimize on all steps in an episode at the same time
        episode_loss:float = 0
        for t in range(0, total_steps):
            
            # calculating $G_t$  
            cumilitive_return:float = sum([(DISCOUNT_RATE ** (k - t) * step_data[k][1]) for k in range(t, total_steps)])
            
            #  $loss = -G_t * \pi_\theta(a_t|s_t)$
            step_loss:float = -cumilitive_return * step_data[t][0]
            episode_loss += step_loss
            
        optimizer.zero_grad()
        episode_loss.backward()
        optimizer.step()
            
        collected_losses += episode_loss.item()
        collected_rewards += sum([reward for _, reward in step_data])
            
        if (episode != 0) and (episode % PRINT_INTERVAL == 0):
            print(f"Episode {episode} - average loss of {str(collected_losses / PRINT_INTERVAL)} and reward of {str(collected_rewards / PRINT_INTERVAL)} over the last {str(PRINT_INTERVAL)} episodes")
            collected_losses = 0
            collected_rewards = 0
            

    torch.save(policy.state_dict(), weights_path)