from __future__ import annotations

import tensorflow as tf


def _valid_mask(y_true: tf.Tensor) -> tf.Tensor:
    """
    Takes in a tensor of true labels (y_true) of shape (batch_size, 64,64,1)
    Creates a mask where the pixels with -1, are set to 0 and the pixels with 0 or 1 are set to 1. 
    We will use this mask later when computing the loss, so that the pixels with -1 (unlabeled pixels) do not contribute to the loss and do not affect the training of the model.
    """
    return tf.cast(tf.not_equal(y_true, -1.0), tf.float32)

def weighted_masked_binary_crossentropy(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    pos_weight: float = 200.0,
    neg_weight: float = 1.0,
) -> tf.Tensor:
    """
    Weighted binary cross-entropy that ignores unlabeled pixels marked as -1.

    - valid pixels are 0 or 1
    - unlabeled pixels are -1 and do not contribute to the loss
    - fire pixels (1) get a larger weight than non-fire pixels (0)
        - The intiution is that if the model keeps predicting no fire then it can get a low loss by just predicting 0 everywhere, 
        so we need to assign a higher weight to the fire pixels to encourage the model to learn to predict them correctly.
        - So now if the model predicts 0 for a fire pixel, it will get a higher loss than if it predicts 0 for a non-fire pixel, which encourages the model to learn to predict fire pixels correctly.
    """
    mask = _valid_mask(y_true)

    # replace -1 with 0 only so BCE can be computed
    y_true_clean = tf.where(tf.equal(y_true, -1.0), 0.0, y_true)
    y_true_clean = tf.cast(y_true_clean, tf.float32)

    bce = tf.keras.backend.binary_crossentropy(y_true_clean, y_pred)
    bce = tf.cast(bce, tf.float32)

    # assign higher weight to fire pixels
    weights = tf.where(
        tf.equal(y_true_clean, 1.0),
        pos_weight,
        neg_weight,
    )
    weights = tf.cast(weights, tf.float32)

    # apply both class weights and valid-pixel mask
    weighted_masked_bce = bce * weights * mask

    return tf.reduce_sum(weighted_masked_bce) / (tf.reduce_sum(weights * mask) + 1e-8)


def masked_binary_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Accuracy computed only on labeled pixels.
    """
    mask = _valid_mask(y_true)
    y_true_clean = tf.where(tf.equal(y_true, -1.0), 0.0, y_true)

    y_pred_bin = tf.cast(y_pred >= 0.5, tf.float32)
    correct = tf.cast(tf.equal(y_true_clean, y_pred_bin), tf.float32)

    correct = correct * mask
    return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-8)