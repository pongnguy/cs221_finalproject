#%% md
# # Stock NeurIPS2018 Part 2. Train
# This series is a reproduction of *the process in the paper Practical Deep Reinforcement Learning Approach for Stock Trading*. 
# 
# This is the second part of the NeurIPS2018 series, introducing how to use FinRL to make data into the gym form environment, and train DRL agents on it.
# 
# Other demos can be found at the repo of [FinRL-Tutorials]((https://github.com/AI4Finance-Foundation/FinRL-Tutorials)).

import pandas as pd
from stable_baselines3.common.logger import configure
import logging

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

check_and_make_directories([TRAINED_MODEL_DIR])




#%% md
# # Part 3: Train DRL Agents
# * Here, the DRL algorithms are from **[Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/)**. It's a library that implemented popular DRL algorithms using pytorch, succeeding to its old version: Stable Baselines.
# * Users are also encouraged to try **[ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL)** and **[Ray RLlib](https://github.com/ray-project/ray)**.




#%% md
# ## Agent Training: 5 algorithms (A2C, DDPG, PPO, TD3, SAC)
# 

def train_model(model_type: str, model_parameters: dict, env_train) -> DRLAgent:

    if model_type not in ['a2c', 'ddpg', 'ppo', 'td3', 'sac']:
        raise NotImplementedError("model_type not implemented")

    # match model_type:
    #     case 'ppo':
    #         PARAMS = {
    #             "n_steps": 2048,
    #             "ent_coef": 0.01,
    #             "learning_rate": 0.00025,
    #             "batch_size": 128,
    #         }
    #     case 'td3':
    #         PARAMS = {
    #             "batch_size": 100,
    #             "buffer_size": 1000000,
    #             "learning_rate": 0.001
    #         }
    #     case 'sac':
    #         PARAMS = {
    #             "batch_size": 128,
    #             "buffer_size": 100000,
    #             "learning_rate": 0.0001,
    #             "learning_starts": 100,
    #             "ent_coef": "auto_0.1",
    #         }
    #     case _:
    #         PARAMS = {}

    agent = DRLAgent(env=env_train)
    model = agent.get_model(model_type, model_kwargs=model_parameters)

    # set up logger
    tmp_path = RESULTS_DIR + '/' + model_type
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    # Alfred this logs results in text files
    model.set_logger(new_logger)

    # TODO Alfred report intermediate rewards value for early stopping (example below)
    #  intermediate_value = model.score(X_valid, y_valid)
    #  trial.report(intermediate_value, step)
    TOTAL_TIMESTEPS = 50000
    logging.info(f'total timesteps are {TOTAL_TIMESTEPS}')
    trained_agent = agent.train_model(model=model,
                                    tb_log_name=model_type,
                                    # TODO Alfred how best to determine total_timesteps?
                                    total_timesteps=TOTAL_TIMESTEPS)
    return trained_agent


# For users running on your local environment, the zip files should be at "./trained_models".