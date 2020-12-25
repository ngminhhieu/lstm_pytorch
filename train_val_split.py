def train_val_test_split(data, test_size, valid_size):
    
    train_len = int(data.shape[0] * (1 - test_size - valid_size))
    valid_len = int(data.shape[0] * valid_size)

    train_set = data[0:train_len]
    valid_set = data[train_len: train_len + valid_len]
    test_set = data[train_len + valid_len:]

    return train_set, valid_set, test_set