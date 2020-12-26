import torch
from math import sqrt
from torch import nn
from curateData import curateData, standardizeData, getDL, get_preds, getData
from train_val_split import train_val_test_split
from extractData import extractHistory
from datetime import date
from RNN_forecaster import forecasterModel, LSTM
import streamlit as st
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

# Display info about the app
st.title("PM2.5 forecasting using DGI and CVAE")

with open("./instructions.md", "r") as f:
    info = "".join(f.readlines())
st.markdown(info)

# Hyperparameters
col1, col2 = st.beta_columns(2)
sequence_length = col1.slider(label = "Select sequence length", 
                        value = 14,
                        min_value = 1,
                        max_value = 200,
                        step = 1)
horizon = col2.slider(label = "Select horizon", 
                        value = 1,
                        min_value = 1,
                        max_value = 200,
                        step = 1)

col3, col4 = st.beta_columns(2)
input_dim = col3.slider(label = "Select input dim", 
                        value = 10,
                        min_value = 1,
                        max_value = 100,
                        step = 1)
output_dim = col4.slider(label = "Select output dim", 
                        value = 1,
                        min_value = 1,
                        max_value = 100,
                        step = 1)

col4, col5, col6 = st.beta_columns(3)

hidden_dim = col4.slider(label = "Neurons in hidden layer of the LSTM",
                        value = 80,
                        min_value = 20,
                        max_value = 150,
                        step = 5)

rnn_layers = col5.slider(label = "Number of RNN hidden layers",
                        value = 2,
                        min_value = 1,
                        max_value = 5,
                        step = 1)

dropout = col6.slider(label = "Dropout percentage",
                        value = 0.1,
                        min_value = 0.0,
                        max_value = 0.5,
                        step = 0.01)

# Provide sliders for configuring training hyperparameters
col7, col8 = st.beta_columns(2)
n_epochs = col7.slider(label = "Number of epochs to train",
                        value = 100,
                        min_value = 10,
                        max_value = 300,
                        step = 10)

batch_sz = col8.slider(label = "Minibatch size",
                        value = 16,
                        min_value = 8,
                        max_value = 64,
                        step = 4)

col8, col9 = st.beta_columns(2)
n_lags = col8.slider(label = "Number of historical timesteps to consider",
                        value = 8,
                        min_value = 1,
                        max_value = 10,
                        step = 1)

learning_rate = col9.slider(label = "Learning rate for the model",
                        value = 5e-2,
                        min_value = 1e-2,
                        max_value = 1e-1,
                        step = 1e-2)


params = {"batch_size": batch_sz,
         "shuffle": False,
         "num_workers": 4}

train_pct = 0.6
valid_pct = 0.2
device = "cpu"

if __name__ == '__main__':
    st.write(f"Runninggggg")
    # Extract data for the ticker mentioned above

    # Get the inputs and outputs from the extracted ticker data
    pm_dataset = pd.read_csv('./data/pm.csv')
    pm_dataset = pm_dataset.replace("**", 0)
    pm_dataset = pm_dataset.to_numpy()
    
    # Perform the train validation split
    train_data, val_data, test_data = train_val_test_split(pm_dataset, train_pct, valid_pct)

    # Standardize the data to bring the inputs on a uniform scale
    normalized_train, sc = standardizeData(train_data, train = True)
    normalized_val, _ = standardizeData(val_data, sc)
    normalized_test, _ = standardizeData(test_data, sc)

    trainX, trainY = getData(normalized_train, sequence_length, horizon, output_dim)
    valX, valY = getData(normalized_val, sequence_length, horizon, output_dim)
    testX, testY = getData(normalized_test, sequence_length, horizon, output_dim)
    trainY = trainY[:, 0, :]
    valY = valY[:, 0, :]
    testY = testY[:, 0, :]
    trainX = torch.FloatTensor(trainX)
    trainY = torch.FloatTensor(trainY)
    valX = torch.FloatTensor(valX)
    valY = torch.FloatTensor(valY)
    testX = torch.FloatTensor(testX)
    testY = torch.FloatTensor(testY)


    # Create the model
    num_epochs = 1
    learning_rate = 0.01

    input_size = 69
    hidden_size = 2
    num_layers = 1

    num_classes = 1

    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(trainX)
        optimizer.zero_grad()
        loss = criterion(outputs, trainY)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    prediction = lstm(testX)
    prediction = prediction.data.numpy()
    groundtruth = testY.data.numpy()
    rest_of_features = normalized_test[:(-sequence_length-horizon), 0:-1]
    prediction = sc.inverse_transform(np.concatenate((rest_of_features,prediction), axis=1))
    groundtruth = sc.inverse_transform(np.concatenate((rest_of_features,groundtruth), axis=1))

    plt.axvline(x=train_pct, c='r', linestyle='--')
    plt.plot(groundtruth)
    plt.plot(prediction)
    plt.suptitle('Time-Series Prediction')
    plt.show()