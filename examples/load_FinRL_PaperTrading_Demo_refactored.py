#%%
# Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Many platforms exist for simulated trading (paper trading) which can be used for building and developing the methods discussed. Please use common sense and always first consult a professional before trading or investing.
# install finrl library
# %pip install --upgrade git+https://github.com/AI4Finance-Foundation/FinRL.git
# Alpaca keys
from __future__ import annotations

import argparse
import functools
import psycopg2
from optuna import create_study
from optuna.artifacts import FileSystemArtifactStore #, GCSArtifactStore
import os


import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("data_key", help="data source api key")
parser.add_argument("data_secret", help="data source api secret")
parser.add_argument("data_url", help="data source api base url")
parser.add_argument("trading_key", help="trading api key")
parser.add_argument("trading_secret", help="trading api secret")
parser.add_argument("trading_url", help="trading api base url")
args = parser.parse_args()
DATA_API_KEY = args.data_key
DATA_API_SECRET = args.data_secret
DATA_API_BASE_URL = args.data_url
TRADING_API_KEY = args.trading_key
TRADING_API_SECRET = args.trading_secret
TRADING_API_BASE_URL = args.trading_url

print("DATA_API_KEY: ", DATA_API_KEY)
print("DATA_API_SECRET: ", DATA_API_SECRET)
print("DATA_API_BASE_URL: ", DATA_API_BASE_URL)
print("TRADING_API_KEY: ", TRADING_API_KEY)
print("TRADING_API_SECRET: ", TRADING_API_SECRET)
print("TRADING_API_BASE_URL: ", TRADING_API_BASE_URL)

from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.meta.paper_trading.common import train, test, alpaca_history, DIA_history, HashableDict
from finrl.config import INDICATORS

# Import Dow Jones 30 Symbols
from finrl.config_tickers import DOW_30_TICKER, LOCAL_TICKER

from optuna import visualization

#ticker_list = DOW_30_TICKER
ticker_list = LOCAL_TICKER
env = StockTradingEnv
# if you want to use larger datasets (change to longer period), and it raises error, please try to increase "target_step". It should be larger than the episode steps.
ERL_PARAMS = {
    "learning_rate": 3e-6,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": tuple([128, 64]),
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 1,
}

# Set up sliding window of 6 days training and 2 days testing
import datetime
from pandas.tseries.offsets import BDay  # BDay is business day, not birthday...

today = datetime.datetime.today()

TEST_END_DATE = (today - BDay(1)).to_pydatetime().date()
TEST_START_DATE = (TEST_END_DATE - BDay(10)).to_pydatetime().date()
TRAIN_END_DATE = (TEST_START_DATE - BDay(1)).to_pydatetime().date()
TRAIN_START_DATE = (TRAIN_END_DATE - BDay(100)).to_pydatetime().date()
TRAINFULL_START_DATE = TRAIN_START_DATE
TRAINFULL_END_DATE = TEST_END_DATE

TRAIN_START_DATE = str(TRAIN_START_DATE)
TRAIN_END_DATE = str(TRAIN_END_DATE)
TEST_START_DATE = str(TEST_START_DATE)
TEST_END_DATE = str(TEST_END_DATE)
TRAINFULL_START_DATE = str(TRAINFULL_START_DATE)
TRAINFULL_END_DATE = str(TRAINFULL_END_DATE)

#TRAIN_START_DATE = '2015-01-01'
#TRAIN_END_DATE = '2015-01-20'
#TEST_START_DATE = '2015-01-21'
#TEST_END_DATE = '2015-01-22'
#TRAINFULL_START_DATE = '2015-01-01'
#TRAINFULL_END_DATE = '2015-01-22'

print("TRAIN_START_DATE: ", TRAIN_START_DATE)
print("TRAIN_END_DATE: ", TRAIN_END_DATE)
print("TEST_START_DATE: ", TEST_START_DATE)
print("TEST_END_DATE: ", TEST_END_DATE)
print("TRAINFULL_START_DATE: ", TRAINFULL_START_DATE)
print("TRAINFULL_END_DATE: ", TRAINFULL_END_DATE)

# Get best study through optuna and artifact store
STUDY_NAME = 'cs221_finrl2'
base_path = "./artifacts"
os.makedirs(base_path, exist_ok=True)
artifact_store = FileSystemArtifactStore(base_path=base_path)
#artifact_store = GCSArtifactStore("optuna_artifacts")

storage_name = "postgresql://alfred:Cc17931793@postgres:5432/optuna_db"
study = create_study(direction='maximize', study_name=STUDY_NAME, storage=storage_name, load_if_exists=True)

# Loading and displaying artifacts associated with the best trial.
best_artifact_id = study.best_trial.user_attrs.get("artifact_id")
with artifact_store.open_reader(best_artifact_id) as f_artifact:
    with open('papertrading_erl/actor.pth', 'wb') as f_agent:
        f_agent.write(f_artifact.read())

# get best study through direct query to Postgres trials database and GCS object store
# TODO

# recreate net_dimensions variable
number_layers = study.best_trial.params["number_layers"]
print(f'using best trial_id {study.best_trial._trial_id} with return of {study.best_trial.values}')
net_dimension = []
for i in range(0, number_layers):
    net_dimension.append(128)
net_dimension.append(64)

#print(content)

# TODO Alfred not sure if this is correct, but model dimensions now match
action_dim = len(LOCAL_TICKER)
state_dim = (
    1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim
)  # Calculate the DRL state dimension manually for paper trading. amount + (turbulence, turbulence_bool) + (price, shares, cd (holding time)) * stock_dim + tech_dim


paper_trading_erl = PaperTradingAlpaca(
    ticker_list=LOCAL_TICKER,
    #ticker_list=DOW_30_TICKER,
    time_interval="1Min",
    drl_lib="elegantrl",
    agent="ppo",
    cwd="./papertrading_erl",
    net_dim=tuple(net_dimension),
    state_dim=state_dim,
    action_dim=action_dim,
    API_KEY=TRADING_API_KEY,
    API_SECRET=TRADING_API_SECRET,
    API_BASE_URL=TRADING_API_BASE_URL,
    tech_indicator_list=INDICATORS,
    turbulence_thresh=30,
    max_stock=study.best_trial.params["max_stock"],
)
print('end of paper_trading_erl')

paper_trading_erl.run()
print('end of paper_trading_erl.run()')
