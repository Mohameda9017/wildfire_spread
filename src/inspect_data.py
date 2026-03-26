import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_dataset


tfrecord_path = "data/raw/next_day_wildfire_spread/next_day_wildfire_spread_train_00.tfrecord"
dataset = load_dataset(tfrecord_path)


for raw_record in dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    prev_mask = example.features.feature["elevation"].float_list.value
    fire_mask = example.features.feature["FireMask"].float_list.value

    print("PrevFireMask length:", len(prev_mask))
    print("FireMask length:", len(fire_mask))

    print("\nFirst 20 PrevFireMask values:")
    print(list(prev_mask[:20]))

    print("\nFirst 20 FireMask values:")
    print(list(fire_mask[:20]))

    print("\nUnique values in PrevFireMask:")
    print(sorted(set(prev_mask)))

    print("\nUnique values in FireMask:")
    print(sorted(set(fire_mask)))