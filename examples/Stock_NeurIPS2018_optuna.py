#%%

from __future__ import annotations

#import argparse
#import contextlib
#import functools
import logging

import subprocess
import docker
import numpy as np
import matplotlib.pyplot as plt
import json

import pandas as pd

# Alfred uses env_stocktrading_np
from Stock_NeurIPS2018_2_Train1 import train_model
from Stock_NeurIPS2018_3_Backtest import test
#from finrl.config import INDICATORS
import random

# Import Dow Jones 30 Symbols
#from finrl.config_tickers import DOW_30_TICKER, LOCAL_TICKER
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


#import io
#import torch
#from diskcache import Cache
from tempfile import NamedTemporaryFile

#cache = Cache('/mnt/diskcache/cs221_finrl', EVICTION_POLICY='none')

#ticker_list = LOCAL_TICKER
#ticker_list = DOW_30_TICKER
#env = StockTradingEnv
# if you want to use larger datasets (change to longer period), and it raises error, please try to increase "target_step". It should be larger than the episode steps.


# Set up sliding window of 6 days training and 2 days testing
#import datetime
#from pandas.tseries.offsets import BDay  # BDay is business day, not birthday...

# Alfred hardcoded to match training and testing data set
TRAIN_START_DATE = '2009-01-02'
TRAIN_END_DATE = '2020-06-30'
TEST_START_DATE = '2020-07-01'
TEST_END_DATE = '2021-10-27'

import os
from optuna import Trial, create_study, visualization
# TODO Alfred upload artifacts in minio local object storage
from optuna.artifacts import FileSystemArtifactStore #, Boto3ArtifactStore, GCSArtifactStore
from optuna.artifacts import upload_artifact
base_path = "./artifacts"
os.makedirs(base_path, exist_ok=True)
artifact_store = FileSystemArtifactStore(base_path=base_path)
#artifact_store = GCSArtifactStore("optuna_artifacts")

# ## Read data
#
# We first read the .csv file of our training data into dataframe.
train_data = pd.read_csv('train_data.csv')
trade_data = pd.read_csv('trade_data.csv')
train_data = train_data.set_index(train_data.columns[0])
train_data.index.names = ['']
trade_data = trade_data.set_index(trade_data.columns[0])
trade_data.index.names = ['']



## Construct the environment for training
# Calculate and specify the parameters we need for constructing the environment.

# # Part 2. Build A Market Environment in OpenAI Gym-style
# The core element in reinforcement learning are **agent** and **environment**. You can understand RL as the following process:

stock_dimension = len(train_data.tic.unique())
# Alfred state space depends on number of indicators used
# Alfred why state_space is 2*stock_dimension?
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension
# Alfred would adding some compound actions on groups of stock help the agent learn better strategies?
action_dim = stock_dimension + 2

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    # Alfred action_space is the stock_dimension since each stock can be sold/held/bought
    "action_space": action_dim,
    "reward_scaling": 1e-4
}


e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)

# Alfred what does this line do?
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))


## Construct the environment for testing
# # Part 2. Backtesting
# To backtest the agents, upload trade_data.csv in the same directory of this notebook. For Colab users, just upload trade_data.csv to the default directory.

# # Load the agents
#
# trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
# trained_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None
# trained_ppo = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None
# trained_td3 = TD3.load(TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None
# trained_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None

# ### Trading (Out-of-sample Performance)
#
# We update periodically in order to take full advantage of the data, e.g., retrain quarterly, monthly or weekly. We also tune the parameters along the way, in this notebook we use the in-sample data from 2009-01 to 2020-07 to tune the parameters once, so there is some alpha decay here as the length of trade date extends.
#
# Numerous hyperparameters – e.g. the learning rate, the total number of samples to train on – influence the learning process and are usually determined by testing some variations.

stock_dimension = len(trade_data.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension
action_dim = stock_dimension + 2

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": action_dim,
    "reward_scaling": 1e-4
}

#e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)
e_trade_gym = StockTradingEnv(df=trade_data, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()


# 1. Define the Objective Function
def objective(trial: Trial, env_train, e_trade_gym):
    # Define hyperparameters and model
    # Train and evaluate the model
    # Return the evaluation metric
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3)
    batch_size_power = trial.suggest_int("batch_size_power", 1, 5, 1)
    batch_size = 128 * 2**batch_size_power
    gamma = trial.suggest_float("gamma", 0.80, 0.99)
    buffer_size_power = trial.suggest_int("buffer_size_power", 4, 7, 1)
    buffer_size = 10**buffer_size_power
    learning_starts_power = trial.suggest_int("learning_starts_power", 1, 3, 1)
    learning_starts = 10**learning_starts_power
    #max_stock = trial.suggest_int("max_stock", 5, 50, 1)


    with NamedTemporaryFile(suffix=".log") as output_file:
        # initialize new logger for the trial
        handlers = [logging.FileHandler(output_file.name), logging.StreamHandler()]
        logging.basicConfig(#filename=output_file.name,
                            #filemode='w+',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            #datefmt='%H:%M:%S',
                            level=logging.INFO,
                            handlers=handlers)
        # Alfred print to screen all info that is logged
        #console = logging.StreamHandler()
        #console.setLevel(logging.INFO)
        #logging.getLogger().addHandler(console)
        #logging.info('testing printing to screen')

        #initial_capital_power = trial.suggest_int("initial_capital_power", 2, 6)
        #initial_capital = 10**initial_capital_power
        # TODO Alfred add time_interval to the search space

        TRAIN_TIMEINTERVAL = "1D"
        TRAINED_MODEL_DIR = "trained_models"

        # Alfred choices must stay the same over all trials
        #model_type = 'td3'
        model_type = trial.suggest_categorical("model_type", ['a2c', 'ddpg','ppo', 'td3', 'sac'])  #['a2c', 'ddpg','ppc', 'td3', 'sac']
        match model_type:
            case 'a2c':
                PARAMS = {
                    #"n_steps": 2048,
                    #"ent_coef": 0.01,
                    "learning_rate": learning_rate,
                    #"batch_size": batch_size, # does not work for A2C
                    #"buffer_size": buffer_size, # does not work for A2C
                    #"learning_starts": learning_starts, # does not work for A2C
                    "gamma": gamma,
                    #"random_exploration": 0.1, # TD3 and SAC
                }
            case 'ppo':
                logging.info('selected ppo!')
                PARAMS = {
                    #"n_steps": 2048,
                    #"ent_coef": 0.01,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size, # does not work for A2C
                    #"buffer_size": buffer_size, # does not work for A2C
                    #"learning_starts": learning_starts, # does not work for A2C
                    "gamma": gamma,
                    #"random_exploration": 0.1, # TD3 and SAC
                }
            case _:
                PARAMS = {
                    #"n_steps": 2048,
                    #"ent_coef": 0.01,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size, # does not work for A2C
                    "buffer_size": buffer_size, # does not work for A2C
                    "learning_starts": learning_starts, # does not work for A2C
                    "gamma": gamma,
                    #"random_exploration": 0.1, # TD3 and SAC
                }
        logging.info(PARAMS)

        # ERL_PARAMS = {
        #     "learning_rate": learning_rate,
        #     "batch_size": batch_size,
        #     "gamma": gamma,
        #     "seed": 312,
        #     # Alfred what does this do?  Looks like there are sizes of inner layers of actor/critic network
        #     "net_dimension": tuple(net_dimension),
        #     "target_step": 5000,
        #     "eval_gap": 30,
        #     "eval_times": 1,
        #     "initial_capital": initial_capital,
        #     "max_stock": max_stock
        # }

        MODEL_DESCRIPTION = f'model type {model_type} of stablebaselines3 with some selected hyperparameters'
        # save the generated actor with the trial

        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
        print(git_branch)
        trial.set_user_attr(
            "git_branch", git_branch
        )
        git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        print(git_commit)
        trial.set_user_attr(
            "git_commit", subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        )
        trial.set_user_attr(
            "model_description", MODEL_DESCRIPTION
        )

        # trial.set_user_attr(
        #     "initial_capital", initial_capital
        # )
        match os.getenv('PLATFORM'):
            case 'vastai':
                platform = 'vastai:' + os.getenv('VAST_CONTAINERLABEL')
            case None:
                platform = 'PLATFORM variable is not set'
            case _:
                platform = os.getenv('PLATFORM')

        trial.set_user_attr(
            "platform", platform
        )
        trial.set_user_attr(
            "train_timeinterval", TRAIN_TIMEINTERVAL
        )
        trial.set_user_attr(
            "train_start_date", TRAIN_START_DATE
        )
        trial.set_user_attr(
            "train_end_date", TRAIN_END_DATE
        )

        # Alfred train with local data without specifying start/end data
        trained_agent = train_model(
            # start_date='2009-01-02',
            # end_date='2020-06-30',
            # ticker_list=ticker_list,
            #data=train_data,
            # time_interval=TRAIN_TIMEINTERVAL,
            # technical_indicator_list=INDICATORS,
            # drl_lib="elegantrl",
            # env=env,
            model_type=model_type,
            # API_KEY=DATA_API_KEY,
            # API_SECRET=DATA_API_SECRET,
            # API_BASE_URL=DATA_API_BASE_URL,
            model_parameters=PARAMS,
            env_train=env_train,
            #cwd="./papertrading_erl",  # current_working_dir
            #break_step=1e5,
        )

        #trial.set_user_attr(
        #    "artifact_id_output", artifact_id_output
        #)  # Save the ID in RDB so that it can be referenced later

        # save actor to temporary file
        # Alfred save to file with random prefix to allow multiple processes on the same computer without likely collision
        file_prefix = random.randint(0, int(1e5))
        trained_agent.save(TRAINED_MODEL_DIR + "/" + str(file_prefix))
        file_path = TRAINED_MODEL_DIR + "/" + str(file_prefix) + ".zip"
        artifact_id_actor = upload_artifact(
            trial, file_path, artifact_store
        )
        trial.set_user_attr(
            "artifact_id_actor", artifact_id_actor
        )  # Save the ID in RDB so that it can be referenced later


        TEST_TIMEINTERVAL = "1D"
        # Alfred test with local data without specifying start/end dates
        account_value = test(
            trained_agent=trained_agent,
            e_trade_gym=e_trade_gym,
            #train_data=train_data,
            #trade_data=trade_data,
            # start_date='2020-07-01',
            # end_date='2021-10-27',
            # ticker_list=ticker_list,
            # data_source="local",
            # time_interval=TEST_TIMEINTERVAL,
            # technical_indicator_list=INDICATORS,
            # drl_lib="elegantrl",
            # env=env,
            # model_name="ppo",
            # if_vix=True,
            # # API_KEY=DATA_API_KEY,
            # # API_SECRET=DATA_API_SECRET,
            # # API_BASE_URL=DATA_API_BASE_URL,
            # # TODO Alfred pass in file-like object so that can be hashed with custom function?
            # # TODO Alfred might need to create wrapper class and then implement hash function which takes into account file contents
            # actor_statedict=actor_statedict,
            # #cwd="./papertrading_erl",
            # net_dimension=ERL_PARAMS["net_dimension"],
            # initial_capital=ERL_PARAMS["initial_capital"],
            # max_stock=ERL_PARAMS["max_stock"]
        )
        trial.set_user_attr(
            "test_timeinterval", TEST_TIMEINTERVAL
        )
        trial.set_user_attr(
            "test_start_date", TEST_START_DATE
        )
        trial.set_user_attr(
            "test_end_date", TEST_END_DATE
        )

        # print out all the output after finished training
        #with open(output_file.name, 'r') as f:
        #    print(f.read())
        #print(output_file.readlines())
        #file_path = 'output.txt'
        output_file.flush()
        artifact_id_output = upload_artifact(
            trial, output_file.name, artifact_store
        )  # The return value is the artifact ID.
        print(f'artifact_id_output is {artifact_id_output}')
        trial.set_user_attr(
            "artifact_id_output", artifact_id_output
        )  # Save the ID in RDB so that it can be referenced later
        logging.info(f'end of trial {trial._trial_id}')

    return account_value['account_value'][-1] / account_value['account_value'][0]


# 2. Create a Study Object
try:
    # try as if inside docker compose network
    storage_name = "postgresql://alfred:Cc17931793@postgres:5432/optuna_db"
    study = create_study(direction='maximize', study_name='cs221_finrl2', storage=storage_name, load_if_exists=True)
except:
    # otherwise assume postgres on local machine
    storage_name = "postgresql://alfred:Cc17931793@127.0.0.1:5432/optuna_db"
    study = create_study(direction='maximize', study_name='cs221_finrl2', storage=storage_name, load_if_exists=True)

def objective_wrapper(trial):
    return objective(trial, env_train=env_train, e_trade_gym=e_trade_gym)

N_TRIALS = os.getenv('N_TRIALS')
if N_TRIALS is None:
    N_TRIALS = 5
else:
    N_TRIALS = int(N_TRIALS)
#if N_TRIALS == None:
#    N_TRIALS = 100
print(f'running for {N_TRIALS} trials')
# 3. Run the Optimization Process
study.optimize(objective_wrapper, n_trials=N_TRIALS)
