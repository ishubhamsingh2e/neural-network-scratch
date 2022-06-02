import numpy as np

class network:
    def __init__(self, X, Y, iterations, alpha) -> None:
        self.X = X
        self.Y = Y
        self.m, self.n = self.X.shape
        self.iterations = iterations
        self.alpha = alpha

        self.W1 = np.random.rand(10, 784) - 0.5
        self.b1 = np.random.rand(10, 1) - 0.5
        self.W2 = np.random.rand(10, 10) - 0.5
        self.b2 = np.random.rand(10, 1) - 0.5

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def ReLU_deriv(self, Z):
        return Z > 0

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def forward_propragation(self):
        Z1 = self.W1.dot(self.X) + self.b1
        A1 = self.ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.softmax(Z2)

        return Z1, A1, Z2, A2

    def one_hot(self):
        one_hot_Y = np.zeros((self.Y.size, self.Y.max() + 1))
        one_hot_Y[np.arange(self.Y.size), self.Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_propragation(self, Z1, A1, Z2, A2):
        one_hot_Y = self.one_hot()

        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)

        dZ1 = self.W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(self.X.T)
        db1 = 1 / m * np.sum(dZ1)

        return dW1, db1, dW2, db2

    def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2):
        self.W1 = W1 - self.alpha * dW1
        self.b1 = b1 - self.alpha * db1
        self.W2 = W2 - self.alpha * dW2
        self.b2 = b2 - self.alpha * db2

        return W1, b1, W2, b2

    def get_predictions(self, A2):
        return np.argmax(A2, 0)

    def get_accuracy(self, predictions):
        print(predictions, self.Y)
        return np.sum(predictions == self.Y) / self.Y.size

    def gradient_descent(self):
        for i in range(self.iterations):
            Z1, A1, Z2, A2 = self.forward_propragation()
            dW1, db1, dW2, db2 = self.backward_propragation(Z1, A1, Z2, A2)
            W1, b1, W2, b2 = self.update_params(self.W1, self.b1, self.W2, self.b2, dW1, db1, dW2, db2)
            if i % 10 == 0:
                print("ITERATION: ", i)
                print("PREDICTION: ", self.get_predictions(A2))
                print("ACCURACY: ", self.get_accuracy(self.get_predictions(A2)))

        return self.W1, self.b1, self.W2, self.b2
    
    def make_predictions(self, X):
        self.X = X
        _, _, _, A2 = self.forward_propragation()
        preduction = self.get_predictions(A2)
        return preduction
    
    def predict(self, X):
        predict = self.make_predictions(X)
        return predict

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    data = pd.read_csv("train.csv")
    data = np.array(data)
    m, n = data.shape
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.
    _,m_train = X_train.shape

    model = network(X_train, Y_train, 5000, 0.1)
    model.gradient_descent()

    print(model.predict(X_dev))
    print(Y_dev)