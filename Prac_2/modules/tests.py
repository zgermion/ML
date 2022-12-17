from modules.losses import BinaryLogisticLoss
from modules.linear_model import LinearModel
from modules.utils import get_numeric_grad

import numpy as np
import numpy.testing as npt
import time
import pytest


def test_function():
    loss_function = BinaryLogisticLoss(l2_coef=1.0)
    X = np.array([
        [1, 2],
        [3, 4],
        [-5, 6]
    ])
    y = np.array([-1, 1, 1])
    w = np.array([1, 2, 3])
    npt.assert_almost_equal(loss_function.func(X, y, w), 16.00008, decimal=5)

def test_function_negative_inf_values():
    loss_function = BinaryLogisticLoss(l2_coef=0.0)
    X = np.array([
        [10 ** 5],
        [-10 ** 5],
        [10 ** 5]
    ])
    y = np.array([1, -1, 1])
    w = np.array([1, 100])
    npt.assert_almost_equal(loss_function.func(X, y, w), 0, decimal=5)

def test_function_positive_inf_values():
    loss_function = BinaryLogisticLoss(l2_coef=0.0)
    X = np.array([
        [10 ** 2],
        [-10 ** 2],
        [10 ** 2]
    ])
    y = np.array([-1, 1, -1])
    w = np.array([1, 100])
    npt.assert_almost_equal(loss_function.func(X, y, w), 10000.333334, decimal=5)


def test_gradient():
    loss_function = BinaryLogisticLoss(l2_coef=1.0)
    X = np.array([
        [1, 2],
        [3, 4],
        [-5, 6]
    ])
    y = np.array([-1, 1, 1])
    w = np.array([1, 2, 3])
    right_gradient = np.array([0.33325, 4.3335 , 6.66634])
    npt.assert_almost_equal(loss_function.grad(X, y, w), right_gradient, decimal=5)


def test_numeric_grad():
    result = get_numeric_grad(lambda x: (x ** 2).sum(), np.array([1, 2, 3]), 1e-6)
    npt.assert_almost_equal(result, np.array([2, 4, 6]), decimal=5)


def create_simple_dataset():
    X1 = np.random.randint(1, 4, (1000, 10))
    X2 = np.random.randint(-4, 0, (1000, 10))
    X = np.vstack((X1, X2))
    y = np.array([-1] * 1000 + [1] * 1000)
    return X, y


def test_simple_classification_task():
    X, y = create_simple_dataset()
    loss_function = BinaryLogisticLoss(l2_coef=0.1)
    linear_model = LinearModel(
        loss_function=loss_function,
        batch_size=100,
        step_alpha=1,
        step_beta=0,
        tolerance=1e-4,
        max_iter=1000,
    )
    linear_model.fit(X, y)
    predictions = linear_model.predict(X, 0.5)
    npt.assert_equal(predictions, y)


def test_logging():
    X, y = create_simple_dataset()
    loss_function = BinaryLogisticLoss(l2_coef=0.1)
    linear_model = LinearModel(
        loss_function=loss_function,
        batch_size=None,
        step_alpha=1,
        step_beta=0,
        tolerance=1e-100,
        max_iter=5,
    )
    history = linear_model.fit(X, y, trace=True, X_val=X, y_val=y)
    for key in ['time', 'func', 'func_val']:
        assert key in history
        assert len(history[key]) == 5


@pytest.mark.parametrize("step_alpha, step_beta, answer", [
    (1e-1, 0.5, 0.713865),
    (0.6, 1, 15.134696),
    (0.6, 1.1,  1.436495),
])
def test_full_gd(step_alpha, step_beta, answer):
    X = np.array([
        [0, 0, 2, 5, 0.9],
        [5, 1, 3, 1, 0.1],
        [0, 0, 2, 1, 0.5],
        [5, 1, 4, 3, 0.32],
        [0, 2, 3, 2, 0.1],
        [5, 2, 5, 4, 0.10],
        [0, 0, 6, 6, 0.28],
        [5, 1, 3, 2, 0.7],
    ])

    y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    w_0 = np.array([0.5, 0.1, 0.3, 0.5, 0.3, 0.5])

    loss_function = BinaryLogisticLoss(l2_coef=5)
    lm = LinearModel(
        loss_function=loss_function,
        step_alpha=step_alpha,
        step_beta=step_beta,
        tolerance=1e-5,
        max_iter=5,
    )
    lm.fit(X, y, w_0=w_0)
    npt.assert_almost_equal(lm.loss_function.func(X, y, lm.w), answer, decimal=5)
