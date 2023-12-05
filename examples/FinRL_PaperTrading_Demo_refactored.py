#%%
# Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Many platforms exist for simulated trading (paper trading) which can be used for building and developing the methods discussed. Please use common sense and always first consult a professional before trading or investing.
# install finrl library
# %pip install --upgrade git+https://github.com/AI4Finance-Foundation/FinRL.git
# Alpaca keys
from __future__ import annotations

import argparse
import contextlib
import functools

import subprocess
import docker
import numpy as np
import matplotlib.pyplot as plt
import json

#client = docker.from_env()
#client.containers.run("ubuntu", "echo hello world")

# parser = argparse.ArgumentParser()
# parser.add_argument("data_key", help="data source api key")
# parser.add_argument("data_secret", help="data source api secret")
# parser.add_argument("data_url", help="data source api base url")
# parser.add_argument("trading_key", help="trading api key")
# parser.add_argument("trading_secret", help="trading api secret")
# parser.add_argument("trading_url", help="trading api base url")
# args = parser.parse_args()
# DATA_API_KEY = args.data_key
# DATA_API_SECRET = args.data_secret
# DATA_API_BASE_URL = args.data_url
# TRADING_API_KEY = args.trading_key
# TRADING_API_SECRET = args.trading_secret
# TRADING_API_BASE_URL = args.trading_url
#
# print("DATA_API_KEY: ", DATA_API_KEY)
# #print("DATA_API_SECRET: ", DATA_API_SECRET)
# print("DATA_API_BASE_URL: ", DATA_API_BASE_URL)
# print("TRADING_API_KEY: ", TRADING_API_KEY)
# #print("TRADING_API_SECRET: ", TRADING_API_SECRET)
# print("TRADING_API_BASE_URL: ", TRADING_API_BASE_URL)

# Alfred uses env_stocktrading_np
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.meta.paper_trading.common import train, test, alpaca_history, DIA_history, HashableDict
from finrl.config import INDICATORS

# Import Dow Jones 30 Symbols
from finrl.config_tickers import DOW_30_TICKER, LOCAL_TICKER

import io
import torch
#from diskcache import Cache
from tempfile import NamedTemporaryFile

#cache = Cache('/mnt/diskcache/cs221_finrl', EVICTION_POLICY='none')

ticker_list = LOCAL_TICKER
#ticker_list = DOW_30_TICKER
env = StockTradingEnv
# if you want to use larger datasets (change to longer period), and it raises error, please try to increase "target_step". It should be larger than the episode steps.


# Set up sliding window of 6 days training and 2 days testing
import datetime
from pandas.tseries.offsets import BDay  # BDay is business day, not birthday...

today = datetime.datetime.today()

# TEST_END_DATE = (today - BDay(1)).to_pydatetime().date()
# TEST_START_DATE = (TEST_END_DATE - BDay(10)).to_pydatetime().date()
# TRAIN_END_DATE = (TEST_START_DATE - BDay(1)).to_pydatetime().date()
# TRAIN_START_DATE = (TRAIN_END_DATE - BDay(100)).to_pydatetime().date()
# TRAINFULL_START_DATE = TRAIN_START_DATE
# TRAINFULL_END_DATE = TEST_END_DATE
#
# TRAIN_START_DATE = str(TRAIN_START_DATE)
# TRAIN_END_DATE = str(TRAIN_END_DATE)
# TEST_START_DATE = str(TEST_START_DATE)
# TEST_END_DATE = str(TEST_END_DATE)
# TRAINFULL_START_DATE = str(TRAINFULL_START_DATE)
# TRAINFULL_END_DATE = str(TRAINFULL_END_DATE)

TRAIN_START_DATE = '2009-01-02'
TRAIN_END_DATE = '2020-06-30'
TEST_START_DATE = '2020-07-01'
TEST_END_DATE = '2021-10-27'

#TRAINFULL_START_DATE = '2015-01-01'
#TRAINFULL_END_DATE = '2015-01-22'

# print("TRAIN_START_DATE: ", TRAIN_START_DATE)
# print("TRAIN_END_DATE: ", TRAIN_END_DATE)
# print("TEST_START_DATE: ", TEST_START_DATE)
# print("TEST_END_DATE: ", TEST_END_DATE)
# print("TRAINFULL_START_DATE: ", TRAINFULL_START_DATE)
# print("TRAINFULL_END_DATE: ", TRAINFULL_END_DATE)

import os
from optuna import Trial, create_study, visualization
# TODO Alfred upload artifacts in minio local object storage
from optuna.artifacts import FileSystemArtifactStore, Boto3ArtifactStore, GCSArtifactStore
from optuna.artifacts import upload_artifact
base_path = "/mnt/artifacts"
os.makedirs(base_path, exist_ok=True)
artifact_store = FileSystemArtifactStore(base_path=base_path)
#artifact_store = GCSArtifactStore("optuna_artifacts")

# 1. Define the Objective Function
def objective(trial: Trial):
    # Define hyperparameters and model
    # Train and evaluate the model
    # Return the evaluation metric
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-6)
    batch_size_power = trial.suggest_int("batch_size_power", 1, 3, 1)
    batch_size = 1024 * 2**batch_size_power
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    max_stock = trial.suggest_int("max_stock", 5, 50, 1)
    number_layers = trial.suggest_int("number_layers", 4, 10, 1)
    net_dimension = []
    for i in range(0, number_layers):
        net_dimension.append(128)
    net_dimension.append(64)
    initial_capital_power = trial.suggest_int("initial_capital_power", 2, 6)
    initial_capital = 10**initial_capital_power
    # TODO Alfred add time_interval to the search space

    TRAIN_TIMEINTERVAL = "1D"
    ERL_PARAMS = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gamma": gamma,
        "seed": 312,
        # Alfred what does this do?  Looks like there are sizes of inner layers of actor/critic network
        "net_dimension": tuple(net_dimension),
        "target_step": 5000,
        "eval_gap": 30,
        "eval_times": 1,
        "initial_capital": initial_capital,
        "max_stock": max_stock
    }

    # Alfred train with local data without specifying start/end data
    #with NamedTemporaryFile(suffix=".txt") as output_file:
    #with open('output.txt', "w") as h, contextlib.redirect_stdout(h):
    actor_statedict = train(
        start_date='2009-01-02',
        end_date='2020-06-30',
        ticker_list=ticker_list,
        data_source="local",
        time_interval=TRAIN_TIMEINTERVAL,
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        # API_KEY=DATA_API_KEY,
        # API_SECRET=DATA_API_SECRET,
        # API_BASE_URL=DATA_API_BASE_URL,
        erl_params=ERL_PARAMS,
        #cwd="./papertrading_erl",  # current_working_dir
        break_step=1e5,
    )

        # print out all the output after finished training
        #with open("output.txt", 'r') as f:
        #    print(f.read())
        #print(output_file.readlines())
        #file_path = 'output.txt'
        #artifact_id_output = upload_artifact(
        #    trial, output_file.name, artifact_store
        #)  # The return value is the artifact ID.
        #trial.set_user_attr(
        #    "artifact_id_output", artifact_id_output
        #)  # Save the ID in RDB so that it can be referenced later

    MODEL_DESCRIPTION='PPO-standard'
    # save the generated actor with the trial

    #file_path = 'papertrading_erl/actor.pth'
    # save actor to temporary file
    with NamedTemporaryFile(suffix=".pth") as actor_file:
        torch.save(actor_statedict, actor_file)
        artifact_id_actor = upload_artifact(
            trial, actor_file.name, artifact_store
        )  # The return value is the artifact ID.
    trial.set_user_attr(
        "artifact_id_actor", artifact_id_actor
    )  # Save the ID in RDB so that it can be referenced later

    trial.set_user_attr(
        "git_branch", subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
    )
    trial.set_user_attr(
        "git_commit", subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    )
    trial.set_user_attr(
        "model_description", MODEL_DESCRIPTION
    )
    trial.set_user_attr(
        "initial_capital", initial_capital
    )
    if os.getenv('PLATFORM') == 'vastai':
        platform = 'vastai:' + os.getenv('VAST_CONTAINERLABEL')
    else:
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

    TEST_TIMEINTERVAL = "1D"
    # Alfred test with local data without specifying start/end dates
    account_value_erl, dates, data = test(
        start_date='2020-07-01',
        end_date='2021-10-27',
        ticker_list=ticker_list,
        data_source="local",
        time_interval=TEST_TIMEINTERVAL,
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        if_vix=True,
        # API_KEY=DATA_API_KEY,
        # API_SECRET=DATA_API_SECRET,
        # API_BASE_URL=DATA_API_BASE_URL,
        # TODO Alfred pass in file-like object so that can be hashed with custom function?
        # TODO Alfred might need to create wrapper class and then implement hash function which takes into account file contents
        actor_statedict=actor_statedict,
        #cwd="./papertrading_erl",
        net_dimension=ERL_PARAMS["net_dimension"],
        initial_capital=ERL_PARAMS["initial_capital"],
        max_stock=ERL_PARAMS["max_stock"]
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
    return account_value_erl[-1] / account_value_erl[0]


# 2. Create a Study Object
storage_name = "postgresql://alfred:Cc17931793@postgres:5432/optuna_db"
study = create_study(direction='maximize', study_name='cs221_finrl', storage=storage_name, load_if_exists=True)

N_TRIALS = int(os.getenv('N_TRIALS'))
#if N_TRIALS == None:
#    N_TRIALS = 100
print(f'running for {N_TRIALS} trials')
# 3. Run the Optimization Process
study.optimize(objective, n_trials=N_TRIALS)
