""" Deep learnign """

from tensorflow import keras


def model(X, y):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)), # this layer changes format of input data
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    model.fit(X, y, epochs=10)
    return model