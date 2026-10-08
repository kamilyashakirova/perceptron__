
import numpy as np
import pandas as pd
import json
import os


def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


def load_and_split(config):
    """Загрузка данных и разделение на train/test с фиксированным seed."""
    df = pd.read_csv(config["data_path"])

    # Первый столбец - ID, второй - целевая переменная (M/B)
    X = df.iloc[:, 2:].values.astype(np.float64)  # признаки
    y_raw = df.iloc[:, 1].values  # метки

    # Кодирование: M -> 1 (malignant), B -> 0 (benign)
    y = (y_raw == "M").astype(np.int64)

    # One-hot encoding для softmax (2 класса)
    y_onehot = np.zeros((y.size, 2))
    y_onehot[np.arange(y.size), y] = 1

    # Фиксированный seed для воспроизводимости
    np.random.seed(config["seed"])
    indices = np.random.permutation(X.shape[0])
    X, y_onehot, y = X[indices], y_onehot[indices], y[indices]

    # Разделение
    split = int(X.shape[0] * (1 - config["test_size"]))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_onehot[:split], y_onehot[split:]
    y_train_cls, y_test_cls = y[:split], y[split:]

    # Нормализация (z-score) по train-статистикам
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test, y_train_cls, y_test_cls, mean, std


if __name__ == "__main__":
    cfg = load_config()
    X_tr, X_te, y_tr, y_te, y_tr_c, y_te_c, _, _ = load_and_split(cfg)
    print(f"Train shape: {X_tr.shape}, Test shape: {X_te.shape}")
    print(f"Class distribution (train): {np.bincount(y_tr_c)}")