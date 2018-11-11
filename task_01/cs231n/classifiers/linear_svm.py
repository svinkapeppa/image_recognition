import numpy as np


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    # Initialize the gradient to zero.
    dW = np.zeros(W.shape)

    # Compute the loss and derivative.
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Average.
    loss /= num_train
    dW /= num_train

    # Regularization.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    num_train = X.shape[0]

    # Compute loss.
    scores = X @ W
    correct_scores = scores[np.arange(num_train), y]
    mask = np.maximum(0, scores - correct_scores[:, np.newaxis] + 1)
    mask[np.arange(num_train), y] = 0
    loss = mask.sum()

    # Compute derivative.
    mask[mask > 0] = 1
    counts = np.sum(mask, axis=1)
    mask[np.arange(num_train), y] = -counts.T
    dW = X.T @ mask

    # Average.
    loss /= num_train
    dW /= num_train

    # Regularization.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW
