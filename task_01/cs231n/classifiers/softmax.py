import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # Compute the loss and derivative.
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i] @ W
        p = np.exp(scores) / np.sum(np.exp(scores))
        loss -= np.log(p[y[i]])
        p[y[i]] -= 1
        for j in range(num_classes):
            dW[:, j] += X[i, :] * p[j]

    # Average.
    loss /= num_train
    dW /= num_train

    # Regularization.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the gradient to zero.
    dW = np.zeros_like(W)

    num_train = X.shape[0]

    # Compute loss.
    scores = X @ W
    p = np.exp(scores) / np.sum(np.exp(scores), axis=1)[:, np.newaxis]
    loss = -np.sum(np.log(p[np.arange(num_train), y]))

    # Compute derivative.
    p[np.arange(num_train), y] -= 1
    dW = X.T @ p

    # Average.
    loss /= num_train
    dW /= num_train

    # Regularization.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW
