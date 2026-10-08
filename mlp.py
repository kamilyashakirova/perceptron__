
import numpy as np
import pickle

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(np.float64)
def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.lr = learning_rate
        self.weights = []
        self.biases = []
        for i in range(self.L):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        A = X

        for i in range(self.L):
            Z = A @ self.weights[i] + self.biases[i]
            self.z_values.append(Z)

            if i < self.L - 1:
                A = relu(Z)
            else:
                A = softmax(Z)
            self.activations.append(A)

        return A
    def cross_entropy_loss(self, y_pred, y_true):
        eps = 1e-12
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + eps)) / m
        return loss
    def backward(self, y_true):
        m = y_true.shape[0]
        self.grad_weights = []
        self.grad_biases = []
        dA = self.activations[-1] - y_true

        for i in reversed(range(self.L)):
            A_prev = self.activations[i]

            dW = (A_prev.T @ dA) / m
            db = np.sum(dA, axis=0, keepdims=True) / m

            self.grad_weights.insert(0, dW)
            self.grad_biases.insert(0, db)

            if i > 0:
                # Передаём градиент на предыдущий слой
                dA = dA @ self.weights[i].T
                dZ = dA * relu_derivative(self.z_values[i - 1])
                dA = dZ
    def update_weights(self):
        for i in range(self.L):
            self.weights[i] -= self.lr * self.grad_weights[i]
            self.biases[i] -= self.lr * self.grad_biases[i]

    def train_batch(self, X_batch, y_batch):
        y_pred = self.forward(X_batch)
        loss = self.cross_entropy_loss(y_pred, y_batch)
        self.backward(y_batch)
        self.update_weights()
        return loss
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    def accuracy(self, X, y_true_classes):
        preds = self.predict(X)
        return np.mean(preds == y_true_classes)
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "layer_sizes": self.layer_sizes,
                "lr": self.lr,
                "weights": self.weights,
                "biases": self.biases
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(data["layer_sizes"], data["lr"])
        model.weights = data["weights"]
        model.biases = data["biases"]
        return model