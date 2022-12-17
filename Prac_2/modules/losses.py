import numpy as np
import scipy
from scipy.special import expit
from scipy.special import logsumexp


class BaseLoss:
    """
    Base class for loss function.
    """

    def func(self, X, y, w):
        """
        Get loss function value at w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, X, y, w):
        """
        Get loss function gradient value at w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogisticLoss(BaseLoss):
    """
    Loss function for binary logistic regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : 2d numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
        """
        N = X.shape[0]
        L = np.sum(np.logaddexp(0, -y*(X.dot(w[1:])+w[0])))
        return L/N + self.l2_coef*(np.linalg.norm(w[1:])**2)

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : 2d numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : 1d numpy.ndarray
        """
        N = X.shape[0]
        A = -y * expit(-y * (X.dot(w[1:]) + w[0]))
        w0_grad = A.mean()
        w_grad = X.T.dot(A)/N + 2*self.l2_coef*w[1:]
        return np.r_[w0_grad, w_grad]



