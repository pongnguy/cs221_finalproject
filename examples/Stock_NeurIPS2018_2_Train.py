#!/usr/bin/env python
# coding: utf-8

# # Stock NeurIPS2018 Part 2. Train
# This series is a reproduction of *the process in the paper Practical Deep Reinforcement Learning Approach for Stock Trading*. 
# 
# This is the second part of the NeurIPS2018 series, introducing how to use FinRL to make data into the gym form environment, and train DRL agents on it.
# 
# Other demos can be found at the repo of [FinRL-Tutorials]((https://github.com/AI4Finance-Foundation/FinRL-Tutorials)).

# 
# # Part 1. Install Packages

# In[1]:


## install finrl library
get_ipython().system('pip install git+https://github.com/AI4Finance-Foundation/FinRL.git')


# In[2]:


import pandas as pd
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

check_and_make_directories([TRAINED_MODEL_DIR])


# # Part 2. Build A Market Environment in OpenAI Gym-style

# ![rl_diagram_transparent_bg.png](data:image/png;base64,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)

# The core element in reinforcement learning are **agent** and **environment**. You can understand RL as the following process: 
# 
# The agent is active in a world, which is the environment. It observe its current condition as a **state**, and is allowed to do certain **actions**. After the agent execute an action, it will arrive at a new state. At the same time, the environment will have feedback to the agent called **reward**, a numerical signal that tells how good or bad the new state is. As the figure above, agent and environment will keep doing this interaction.
# 
# The goal of agent is to get as much cumulative reward as possible. Reinforcement learning is the method that agent learns to improve its behavior and achieve that goal.

# To achieve this in Python, we follow the OpenAI gym style to build the stock data into environment.
# 
# state-action-reward are specified as follows:
# 
# * **State s**: The state space represents an agent's perception of the market environment. Just like a human trader analyzing various information, here our agent passively observes the price data and technical indicators based on the past data. It will learn by interacting with the market environment (usually by replaying historical data).
# 
# * **Action a**: The action space includes allowed actions that an agent can take at each state. For example, a ∈ {−1, 0, 1}, where −1, 0, 1 represent
# selling, holding, and buying. When an action operates multiple shares, a ∈{−k, ..., −1, 0, 1, ..., k}, e.g.. "Buy 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or −10, respectively
# 
# * **Reward function r(s, a, s′)**: Reward is an incentive for an agent to learn a better policy. For example, it can be the change of the portfolio value when taking a at state s and arriving at new state s',  i.e., r(s, a, s′) = v′ − v, where v′ and v represent the portfolio values at state s′ and s, respectively
# 
# 
# **Market environment**: 30 constituent stocks of Dow Jones Industrial Average (DJIA) index. Accessed at the starting date of the testing period.

# ## Read data
# 
# We first read the .csv file of our training data into dataframe.

# In[4]:


train = pd.read_csv('train_data.csv')

# If you are not using the data generated from part 1 of this tutorial, make sure 
# it has the columns and index in the form that could be make into the environment. 
# Then you can comment and skip the following two lines.
train = train.set_index(train.columns[0])
train.index.names = ['']


# ## Construct the environment

# Calculate and specify the parameters we need for constructing the environment.

# In[5]:


stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


# In[6]:


buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}


e_train_gym = StockTradingEnv(df = train, **env_kwargs)


# ## Environment for training

# In[7]:


env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))


# # Part 3: Train DRL Agents
# * Here, the DRL algorithms are from **[Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/)**. It's a library that implemented popular DRL algorithms using pytorch, succeeding to its old version: Stable Baselines.
# * Users are also encouraged to try **[ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL)** and **[Ray RLlib](https://github.com/ray-project/ray)**.

# In[8]:


agent = DRLAgent(env = env_train)

# Set the corresponding values to 'True' for the algorithms that you want to use
if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True


# ## Agent Training: 5 algorithms (A2C, DDPG, PPO, TD3, SAC)
# 

# ### Agent 1: A2C
# 

# In[9]:


agent = DRLAgent(env = env_train)
model_a2c = agent.get_model("a2c")

if if_using_a2c:
  # set up logger
  tmp_path = RESULTS_DIR + '/a2c'
  new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_a2c.set_logger(new_logger_a2c)


# In[10]:


trained_a2c = agent.train_model(model=model_a2c, 
                             tb_log_name='a2c',
                             total_timesteps=50000) if if_using_a2c else None


# In[ ]:


trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None


# ### Agent 2: DDPG

# In[ ]:


agent = DRLAgent(env = env_train)
model_ddpg = agent.get_model("ddpg")

if if_using_ddpg:
  # set up logger
  tmp_path = RESULTS_DIR + '/ddpg'
  new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_ddpg.set_logger(new_logger_ddpg)


# In[ ]:


trained_ddpg = agent.train_model(model=model_ddpg, 
                             tb_log_name='ddpg',
                             total_timesteps=50000) if if_using_ddpg else None


# In[ ]:


trained_ddpg.save(TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None


# ### Agent 3: PPO

# In[11]:


agent = DRLAgent(env = env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)

if if_using_ppo:
  # set up logger
  tmp_path = RESULTS_DIR + '/ppo'
  new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_ppo.set_logger(new_logger_ppo)


# In[ ]:


trained_ppo = agent.train_model(model=model_ppo, 
                             tb_log_name='ppo',
                             total_timesteps=200000) if if_using_ppo else None


# In[ ]:


trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None


# ### Agent 4: TD3

# In[12]:


agent = DRLAgent(env = env_train)
TD3_PARAMS = {"batch_size": 100, 
              "buffer_size": 1000000, 
              "learning_rate": 0.001}

model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)

if if_using_td3:
  # set up logger
  tmp_path = RESULTS_DIR + '/td3'
  new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_td3.set_logger(new_logger_td3)


# In[14]:


trained_td3 = agent.train_model(model=model_td3, 
                             tb_log_name='td3',
                             total_timesteps=50000) if if_using_td3 else None


# In[15]:


trained_td3.save(TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None


# ### Agent 5: SAC

# In[16]:


agent = DRLAgent(env = env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)

if if_using_sac:
  # set up logger
  tmp_path = RESULTS_DIR + '/sac'
  new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_sac.set_logger(new_logger_sac)


# In[ ]:


trained_sac = agent.train_model(model=model_sac, 
                             tb_log_name='sac',
                             total_timesteps=70000) if if_using_sac else None


# In[ ]:


trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None


# ## Save the trained agent
# Trained agents should have already been saved in the "trained_models" drectory after you run the code blocks above.
# 
# For Colab users, the zip files should be at "./trained_models" or "/content/trained_models".
# 
# For users running on your local environment, the zip files should be at "./trained_models".
