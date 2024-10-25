import numpy as np


def evaluate(_network, _test_data):
    # Count the number of correct predictions
    correct_predictions = sum(
        int(np.argmax(_network.feedforward(d)) == np.argmax(d['labels']))
        for d in _test_data
    )
    return correct_predictions
