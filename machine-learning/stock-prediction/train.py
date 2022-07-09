from stock_prediction import create_model, load_data
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import pandas as pd
from parameters import *


def makedirs():
    # create these folders if they does not exist
    if not os.path.isdir("results"):
        os.mkdir("results")

    if not os.path.isdir("logs"):
        os.mkdir("logs")

    if not os.path.isdir("data"):
        os.mkdir("data")


def train(stock, stock_data_filename, model_name, n_steps, scale, split_by_date, shuffle, lookup_step, test_size, feature_columns):
    makedirs()
    # load the data
    data = load_data(stock, n_steps, scale, split_by_date, shuffle, lookup_step, test_size, feature_columns)

    # save the dataframe
    data["df"].to_csv(stock_data_filename)

    # construct the model
    model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                        dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

    # some tensorflow callbacks
    checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    # train the model and save the weights whenever we see
    # a new optimal model using ModelCheckpoint
    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        callbacks=[checkpointer, tensorboard],
                        verbose=1)


if __name__ == '__main__':
    for i_lookup_step in range(1, LOOKUP_STEP+1):
        for ticker in tickers:
            tdf = ticker_data_filename(ticker)
            mn = model_name_info(date_now, ticker, i_lookup_step, SHUFFLE, SCALE, SPLIT_BY_DATE, LOSS,
                                 OPTIMIZER, CELL, N_STEPS, N_LAYERS, UNITS, BIDIRECTIONAL)
            train(ticker, tdf, mn, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                  shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                  feature_columns=FEATURE_COLUMNS)
