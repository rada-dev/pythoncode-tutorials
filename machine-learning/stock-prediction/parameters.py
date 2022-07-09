import os
import time
from tensorflow.keras.layers import LSTM

# Window size or the sequence length
N_STEPS = 50
# Lookup step, 1 is the next day
LOOKUP_STEP = 15

# whether to scale feature columns & output price as well
SCALE = True
# whether to shuffle the dataset
SHUFFLE = True
# whether to split the training/testing set by date
SPLIT_BY_DATE = False
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")

### model parameters

N_LAYERS = 2
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 128
# 25% dropout
DROPOUT = 0.25
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 500


# ticker data to save path
def ticker_data_filename(ticker):
    return os.path.join("data", f"{ticker}_{date_now}.csv")


# model name to save, making it as unique as possible based on parameters
def model_name_info(now, ticker, lookup_step, sh, sc, sbd, loss, opt, nn, n_steps, n_layers, units, bidir):
    scale_str = f"sc-{int(sc)}"
    shuffle_str = f"sh-{int(sh)}"
    split_by_date_str = f"sbd-{int(sbd)}"
    return f"{now}_{ticker}-step-{lookup_step:02d}-{shuffle_str}-{scale_str}-{split_by_date_str}" \
           f"-{loss}-{opt}-{nn.__name__}-seq-{n_steps}-layers-{n_layers}-units-{units}"+bool(bidir)*"-b"


# Amazon stock market
tickers = ["AMZN"]
# ticker_data_filenames = map(ticker_data_filename, tickers)
# model_names = map(model_name_info, tickers)
