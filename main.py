import numpy as np
import random


class NeuralNetwork:
    def __init__(self, layer_sizes: list, learning_rate: float):

        self.loss = None
        self.batch_labels = None
        self.a_list = None
        self.z_list = None

        self.layer_count = len(layer_sizes)
        self.learning_rate = learning_rate

        self.weights: list = []

        for layerdx in range(self.layer_count - 1):
            self.weights.append(xavier_normal(layer_sizes[layerdx], layer_sizes[layerdx + 1]))

        self.bias: list = [np.zeros_like(self.weights[_][0]) for _ in range(len(self.weights))]

        self.bias = np.array(self.bias, dtype=object)
        self.weights = np.array(self.weights, dtype=object)

    def feedforward(self, batch):
        a = batch['inputs']
        self.batch_labels = batch['labels']

        self.z_list: list = []
        self.a_list: list = [a]

        for layerdx in range(self.layer_count - 1):
            z = np.add(np.dot(a, self.weights[layerdx]), self.bias[layerdx])
            a = sigmoid_func(z)
            self.z_list.append(z)
            self.a_list.append(a)

        self.loss = mse_loss_func(a, self.batch_labels)

        return a

    def train(self):

        rev_a = self.a_list[::-1]
        rev_deriv_sigmoid = [deriv_sigmoid_func(z) for z in self.z_list[::-1]]
        rev_weights = self.weights[::-1]

        # output layer to loss function derivitive

        l2zX = np.multiply(4, np.subtract(rev_a[0], self.batch_labels)).T
        rev_a.pop(0)

        delta_weights: np.ndarray = np.empty_like(self.weights)
        delta_biases: np.ndarray = np.empty_like(self.bias)

        for index, (weight, deriv, layer_output) in enumerate(zip(rev_weights, rev_deriv_sigmoid, rev_a)):
            delta_weights[index] = np.dot(l2zX, layer_output).T
            delta_biases[index] = l2zX.sum(axis=1)

            l2zX = np.dot(weight, np.multiply(l2zX, deriv.T))

        self.weights = np.subtract(self.weights, np.multiply(delta_weights[::-1], self.learning_rate))
        self.bias = np.subtract(self.bias, np.multiply(delta_biases[::-1], self.learning_rate))

    def get_loss(self):
        return self.loss.sum()

    def export_model(self):
        pass


def sigmoid_func(x: np.ndarray) -> np.ndarray:
    pos_mask = (x >= 0)
    neg_mask = ~pos_mask
    result = np.zeros_like(x)

    result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    result[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))
    return result


def deriv_sigmoid_func(z: np.ndarray) -> np.ndarray:
    return np.subtract(sigmoid_func(z), np.square(sigmoid_func(z)))


def mse_loss_func(predict: np.ndarray, label: np.ndarray) -> np.array:
    return np.multiply(2, np.square(np.subtract(predict, label)))


def init_expected(label: any, node_count: int):
    return np.array([1 if i == label else 0 for i in range(node_count)])


def xavier_normal(n_in, n_out) -> np.ndarray:
    stddev = np.sqrt(2 / (n_in + n_out))
    return np.random.normal(0, stddev, size=(n_in, n_out))


class BatchManager:
    def __init__(self, src: str):
        self.data = np.genfromtxt(src, delimiter=',', skip_header=1)
        self.batches = None

    def create(self, batch_size: int, output_size: int):
        labels: list = [self.data[row, 0].astype(np.uint8) for row in range(len(self.data))]
        values: list = [np.array(self.data[row, 1:]) for row in range(len(self.data))]

        batches: list = []

        for batch_index in range(batch_size):
            index: int = np.multiply(batch_index, batch_size)
            lbs: list = []
            for ldx in labels[index: index + batch_size]:
                lbs.append(np.array([1 if lb == ldx else 0 for lb in range(output_size)]))

            batch: dict = {
                'labels': np.array(lbs),
                'inputs': np.stack(values[index: index + batch_size])
            }
            batches.append(batch)

        self.batches = batches

        return batches

    def shuffle(self):
        batch_list = self.batches
        for batch_index in range((len(batch_list) - 1), 0, -1):
            randx: int = random.randint(0, batch_index)
            batch_list[batch_index], batch_list[randx] = batch_list[randx], batch_list[batch_index]

        return batch_list


def main():
    epoch_count = 25
    batch_size = 5
    output_size = 10
    mngr = BatchManager(src='datasets/mnist_train.csv')

    batches = mngr.create(batch_size, output_size)
    input_size: int = len(batches[0]['inputs'][0])

    layer_sizes: list = [
        input_size,
        128, 64,
        output_size
    ]

    print('layer sizes:', layer_sizes)

    test = NeuralNetwork(layer_sizes, 0.01)

    for epoch_num in range(epoch_count):
        shuffled_batches = mngr.shuffle()

        for batch in shuffled_batches:
            test.feedforward(batch)
            test.train()

    print(test.feedforward(shuffled_batches[0]))
    print(test.get_loss() / batch_size, shuffled_batches[0]['labels'])

    print('\nfinished')


if __name__ == '__main__':
    main()
