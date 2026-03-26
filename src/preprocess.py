# src/preprocess.py

import re
from typing import Dict, List, Text, Tuple

import tensorflow as tf

from src.config import (
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    DATA_STATS,
    DATA_SIZE,
    NUM_INPUT_CHANNELS,
)

# this function is used to extract the base key from the feature name, which is used to look up the data statistics for that feature.
# Example: "elevation_0" -> "elevation"
def _get_base_key(key: Text) -> Text:
    match = re.match(r"([a-zA-Z]+)", key)
    if match:
        return match.group(1)
    raise ValueError(f"Unexpected key format: {key}")

# gets the min and max values for the feature, then clips the input to that range and rescales it to [0, 1]. 
def _clip_and_rescale(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    base_key = _get_base_key(key)
    if base_key not in DATA_STATS:
        raise ValueError(f"No data statistics found for key: {key}")
    min_val, max_val, _, _ = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    return tf.math.divide_no_nan(inputs - min_val, max_val - min_val)

# clips the input to the min and max values for the feature, then normalizes it by subtracting the mean and dividing by the standard deviation.
# normalizing is different from rescaling because it centers the data around 0 and scales it based on the spread of the data, rather than just scaling it to a fixed range.
def _clip_and_normalize(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    base_key = _get_base_key(key)
    if base_key not in DATA_STATS:
        raise ValueError(f"No data statistics found for key: {key}")
    min_val, max_val, mean, std = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    inputs = inputs - mean
    return tf.math.divide_no_nan(inputs, std)

# this function tells tenserflow how to read the data from the TFRecord files and how to preprocess it. 
# It tells tensorflow the schema of each feature such as its shape, feature name and data type. 
def _get_features_dict(sample_size, features):
    sample_shape = [sample_size, sample_size]
    result = {}

    for feature in features:
        result[feature] = tf.io.FixedLenFeature(
            shape=sample_shape,
            dtype=tf.float32
        )

    return result

def parse_example(
    example_proto: tf.train.Example,
    data_size: int = DATA_SIZE,
    clip_and_normalize: bool = False,
    clip_and_rescale: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor]:
    if clip_and_normalize and clip_and_rescale:
        raise ValueError("Cannot enable both normalization and rescaling.")

    feature_names = INPUT_FEATURES + OUTPUT_FEATURES
    features_dict = _get_features_dict(data_size, feature_names)

    # parses the binary serialized exmaple into useable tensors. 
    # it will be in a dictionary format where the keys are the feature names and the values are the tensors containing the data for those features.
    # the values are tensor object which stores the 2d grid values. 
    features = tf.io.parse_single_example(example_proto, features_dict)

    if clip_and_normalize:
        inputs_list = [
            _clip_and_normalize(features[key], key)
            for key in INPUT_FEATURES
        ]
    elif clip_and_rescale:
        inputs_list = [
            _clip_and_rescale(features[key], key)
            for key in INPUT_FEATURES
        ]
    else:
        # inputs_list stores the tensors of each input feature separted by commas. 
        inputs_list = [features[key] for key in INPUT_FEATURES]

    # Stack inputs into H x W x C
    inputs_stacked = tf.stack(inputs_list, axis=0)
    input_img = tf.transpose(inputs_stacked, [1, 2, 0]) # now we are able to specific row, col and get all the 12 feature values for that pixel.

    # Output mask -> H x W x 1
    outputs_list = [features[key] for key in OUTPUT_FEATURES]
    outputs_stacked = tf.stack(outputs_list, axis=0)
    output_img = tf.transpose(outputs_stacked, [1, 2, 0])

    return input_img, output_img

def build_dataset(
    file_pattern: str,
    batch_size: int = 16,
    clip_and_normalize: bool = False,
    clip_and_rescale: bool = False,
    shuffle: bool = False,
    repeat: bool = False,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=None),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.map(
        lambda x: parse_example(
            x,
            data_size=DATA_SIZE,
            clip_and_normalize=clip_and_normalize,
            clip_and_rescale=clip_and_rescale,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        dataset = dataset.shuffle(1000)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

