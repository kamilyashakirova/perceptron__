import numpy as np
import matplotlib.pyplot as plt
import json
import os

from data_preprocessing import load_config, load_and_split
from mlp import MLP


def train(config):
    # Загрузка данных
    X_train, X_test, y_train, y_test, y_train_cls, y_test_cls, _, _ = load_and_split(config)

    # Создание модели
    model = MLP(config["architecture"], config["learning_rate"])

    # История метрик
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    m = X_train.shape[0]

    print(f"обучение: {config['architecture']}")
    print(f" скока фаз: {epochs}, LR: {config['learning_rate']}, Batch: {batch_size}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(m)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        epoch_loss = 0
        n_batches = 0

        for i in range(0, m, batch_size):
            X_b = X_shuf[i:i + batch_size]
            y_b = y_shuf[i:i + batch_size]
            loss = model.train_batch(X_b, y_b)
            epoch_loss += loss
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        y_pred_train = model.forward(X_train)
        train_loss = model.cross_entropy_loss(y_pred_train, y_train)
        train_acc = model.accuracy(X_train, y_train_cls)

        y_pred_test = model.forward(X_test)
        test_loss = model.cross_entropy_loss(y_pred_test, y_test)
        test_acc = model.accuracy(X_test, y_test_cls)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if epoch % 50 == 0 or epoch == 1:
            print(f"фаза {epoch:4d} | "
                  f"потеря на тренирочных данных: {train_loss:.4f} accuracy: {train_acc:.4f} | "
                  f"потеря на тестовых данных: {test_loss:.4f} accuracy: {test_acc:.4f}")

    # Сохранение модели
    os.makedirs(os.path.dirname(config["model_path"]), exist_ok=True)
    model.save(config["model_path"])
    print(f"\n модель сохранена в {config['model_path']}")

    # Построение графиков
    plot_training_curves(train_losses, test_losses, train_accs, test_accs, config["plot_path"])

    return model


def plot_training_curves(train_loss, test_loss, train_acc, test_acc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_loss) + 1)

    ax1.plot(epochs, train_loss, label="train loss", color="blue")
    ax1.plot(epochs, test_loss, label="test loss", color="red")
    ax1.set_xlabel("фаза")
    ax1.set_ylabel("функция потерь")
    ax1.set_title("кривая потерь")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_acc, label="train acc", color="blue")
    ax2.plot(epochs, test_acc, label="test acc", color="red")
    ax2.set_xlabel("фаза")
    ax2.set_ylabel("accuracy")
    ax2.set_title("кривая accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)