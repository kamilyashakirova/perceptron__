
import numpy as np
import json

from data_preprocessing import load_config, load_and_split
from mlp import MLP


def predict(config):
    _, X_test, _, y_test, _, y_test_cls, _, _ = load_and_split(config)

    model = MLP.load(config["model_path"])
    print(f"Модель загружена из {config['model_path']}")

    y_pred_probs = model.forward(X_test)
    y_pred_classes = model.predict(X_test)

    loss = model.cross_entropy_loss(y_pred_probs, y_test)
    acc = model.accuracy(X_test, y_test_cls)

    print("РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 50)
    print(f"Binary Cross-Entropy Loss: {loss:.4f}")
    print(f"Accuracy:                  {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Всего образцов:            {X_test.shape[0]}")
    print(f"Верных предсказаний:       {int(acc * X_test.shape[0])}")

    # Матрица ошибок
    print("\nМатрица ошибок (Confusion Matrix):")
    print(f"{'':>15} {'Pred B':>10} {'Pred M':>10}")

    tp = np.sum((y_pred_classes == 1) & (y_test_cls == 1))
    tn = np.sum((y_pred_classes == 0) & (y_test_cls == 0))
    fp = np.sum((y_pred_classes == 1) & (y_test_cls == 0))
    fn = np.sum((y_pred_classes == 0) & (y_test_cls == 1))

    print(f"{'Actual B':>15} {tn:>10} {fp:>10}")
    print(f"{'Actual M':>15} {fn:>10} {tp:>10}")

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\nPrecision (M): {precision:.4f}")
    print(f"Recall (M):    {recall:.4f}")
    print(f"F1-score (M):  {f1:.4f}")


if __name__ == "__main__":
    cfg = load_config()
    predict(cfg)