def train_val_test_split(data, test_size, valid_size):
    
    train_len = int(data.shape[0] * (1 - test_size - valid_size))
    valid_len = int(data.shape[0] * valid_size)

    train_set = data[0:train_len]
    valid_set = data[train_len: train_len + valid_len]
    test_set = data[train_len + valid_len:]

    return train_set, valid_set, test_set

def train_val_split(x, y, train_pct):
    """Given the input x and output labels y, splits the dataset into train, validation and test datasets
    Args:
        x ([list]): [A list of all the input sequences]
        y ([list]): [A list of all the outputs (floats)]
        train_pct ([float]): [% of data in the test set]
    """
    # Perform a train test split (It will be sequential here since we're working with time series data)
    N = len(x)
    
    trainX = x[:int(train_pct * N)]
    trainY = y[:int(train_pct * N)]

    valX = x[int(train_pct * N):]
    valY = y[int(train_pct * N):]

    return (trainX, trainY, valX, valY)