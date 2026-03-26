import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np 

iris = load_iris()
X = iris.data
y = iris.target

# mask out class 2 so this becomes binary classification
mask = y != 2
X = X[mask]
y = y[mask]

# turn into a tf dataset
dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(X), seed=42) # creates (x,y) pairs and shuffles them.

# splitting into train and test
train_ds, test_ds = tf.keras.utils.split_dataset(
    dataset, left_size=0.8, shuffle=False
)

# batching
train_ds = train_ds.batch(16)
test_ds = test_ds.batch(16)

# build normalization layer
normalizer = tf.keras.layers.Normalization(axis=-1)

# pulls out only the features and normalizes them. 
feature_ds = train_ds.map(lambda x, y: x)
normalizer.adapt(feature_ds)


# logitic regression model
# sequential is just a way to build the model layer by layer. its like a stack of layers. 
# input → layer1 → layer2 → output 
model = tf.keras.Sequential([
    # inside here you define the layers of the model.
    # defines how many neurons in the layer, what acitvation funciton the nueron will use. and then the input shape is how many features we have, which is 4 in this case

    normalizer, # first layer
    # A dense layer just combines the inputs linearlly so its weighted sum of inputs + bias. 
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(4,)) # second layer
])


model.compile(
    optimizer='adam', # optimizer is the algorithm that adjusts the weights during training to minimize the loss function. Its like regular GD. 
    loss='binary_crossentropy', # loss function measures how well the model's predictions match the true labels.
    metrics=['accuracy'] # allows us to treck the accuracy of the model during training and evaluation. 
)

# 7. Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    verbose=1
)

# train directly on dataset
model.fit(train_ds, epochs=50, verbose=1)

# evaluate directly on dataset
test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# predict on test features only
test_features = test_ds.map(lambda x, y: x)
predicted_probabilities = model.predict(test_features, verbose=0)
predicted_classes = (predicted_probabilities > 0.5).astype(int).flatten()

print("\nPredicted probabilities (first 5):")
print(predicted_probabilities[:5])

print("\nPredicted classes (first 5):")
print(predicted_classes[:5])