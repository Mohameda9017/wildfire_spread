from __future__ import annotations

import tensorflow as tf


def _valid_mask(y_true: tf.Tensor) -> tf.Tensor:
    """
    Valid pixels are those that are not -1.
    Returns a float mask of 1s and 0s.
    """
    return tf.cast(tf.not_equal(y_true, -1.0), tf.float32)


def _prepare_tensors(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    threshold: float = 0.6,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Prepares masked binary tensors for metric computation.

    Returns:
        y_true_clean: true labels with -1 replaced by 0
        y_pred_bin: thresholded predictions (0 or 1)
        mask: valid-pixel mask (1 for valid, 0 for unlabeled)
    """
    mask = _valid_mask(y_true)

    # Replace unlabeled pixels with 0 for numerical safety
    y_true_clean = tf.where(tf.equal(y_true, -1.0), 0.0, y_true)

    # Convert probabilities to binary predictions
    y_pred_bin = tf.cast(y_pred >= threshold, tf.float32)

    return y_true_clean, y_pred_bin, mask


def masked_precision(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Precision over valid pixels only.
    """
    y_true_clean, y_pred_bin, mask = _prepare_tensors(y_true, y_pred)

    y_true_clean = y_true_clean * mask
    y_pred_bin = y_pred_bin * mask

    true_positives = tf.reduce_sum(y_pred_bin * y_true_clean)
    predicted_positives = tf.reduce_sum(y_pred_bin)

    return true_positives / (predicted_positives + 1e-8)


def masked_recall(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Recall over valid pixels only.
    """
    y_true_clean, y_pred_bin, mask = _prepare_tensors(y_true, y_pred)

    y_true_clean = y_true_clean * mask
    y_pred_bin = y_pred_bin * mask

    true_positives = tf.reduce_sum(y_pred_bin * y_true_clean)
    actual_positives = tf.reduce_sum(y_true_clean)

    return true_positives / (actual_positives + 1e-8)


def masked_f1(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    F1 score over valid pixels only.
    """
    precision = masked_precision(y_true, y_pred)
    recall = masked_recall(y_true, y_pred)

    return 2.0 * precision * recall / (precision + recall + 1e-8)


def masked_iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Intersection over Union (IoU) over valid pixels only.
    """
    y_true_clean, y_pred_bin, mask = _prepare_tensors(y_true, y_pred)

    y_true_clean = y_true_clean * mask
    y_pred_bin = y_pred_bin * mask

    intersection = tf.reduce_sum(y_true_clean * y_pred_bin)
    union = tf.reduce_sum(y_true_clean) + tf.reduce_sum(y_pred_bin) - intersection

    return intersection / (union + 1e-8)