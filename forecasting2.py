import torch
from math import sqrt
from torch import nn
from curateData import curateData, standardizeData, getDL, get_preds, getData
from train_val_split import train_val_test_split
from extractData import extractHistory
from datetime import date
from RNN_forecaster import forecasterModel
import streamlit as st
import pandas as pd
import sys

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
    normalized_train, SS_ = standardizeData(train_data, train = True)
    normalized_val, _ = standardizeData(val_data, SS_)
    normalized_test, _ = standardizeData(test_data, SS_)

    trainX, valX = getData(normalized_train, sequence_length, horizon, output_dim)
    trainY, valY = getData(normalized_val, sequence_length, horizon, output_dim)
    trainZ, valZ = getData(normalized_test, sequence_length, horizon, output_dim)

    # Create dataloaders for both training and validation datasets
    # train Y này trước đó không được normalized
    training_generator = getDL(trainX, trainY, params)
    validation_generator = getDL(trainY, valY, params)

    # Create the model
    model = forecasterModel(n_lags, hidden_dim, rnn_layers, dropout).to(device)

    # Define the loss function and the optimizer
    loss_func = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # Track the losses across epochs
    train_losses = []
    valid_losses = []
    st.write("Extracted data, now training the model...")
    # Training loop 
    for epoch in range(1, n_epochs + 1):
        ls = 0
        valid_ls = 0
        # Train for one epoch
        for xb, yb in training_generator:
            # Perform the forward pass operation
            ips = xb.unsqueeze(0)
            targs = yb
            op = model(ips)
            
            # Backpropagate the errors through the network
            optim.zero_grad()
            loss = loss_func(op, targs)
            loss.backward()
            optim.step()
            ls += (loss.item() / ips.shape[1])
        
        # Check the performance on valiation data
        for xb, yb in validation_generator:
            ips = xb.unsqueeze(0)
            ops = model.predict(ips)
            vls = loss_func(ops, yb)
            valid_ls += (vls.item() / xb.shape[1])

        rmse = lambda x: round(sqrt(x * 1.000), 3)
        train_losses.append(str(rmse(ls)))
        valid_losses.append(str(rmse(valid_ls)))
        
        # Print the total loss for every tenth epoch
        if (epoch % 10 == 0) or (epoch == 1):
            st.write(f"Epoch {str(epoch):<4}/{str(n_epochs):<4} | Train Loss: {train_losses[-1]:<8}| Validation Loss: {valid_losses[-1]:<8}")

    # Make predictions on train, validation and test data and plot 
    # the predictions along with the true values 
    to_numpy = lambda x, y: (x.squeeze(0).numpy(), y.squeeze(0).numpy())
    train_preds, train_labels = get_preds(training_generator, model)
    train_preds, train_labels = to_numpy(train_preds, train_labels)

    val_preds, val_labels = get_preds(validation_generator, model)
    val_preds, val_labels = to_numpy(val_preds, val_labels)

    # visualize_results((train_preds, val_preds), (train_labels, val_labels), SYMBOL, 
    #                 f"./img/{SYMBOL}_predictions.png", f"./predictions/{SYMBOL}_predictions.csv", dates)