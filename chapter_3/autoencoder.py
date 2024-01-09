from tensorflow import keras
from keras import datasets,utils,optimizers,layers, models
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

(x_train,y_train), (x_test,y_test) = datasets.fashion_mnist.load_data()

def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs
x_train = preprocess(x_train)
x_test = preprocess(x_test)

encoder_input = layers.Input(
shape=(32, 32, 1), name = "encoder_input"
)
x = layers.Conv2D(32, (3, 3), strides = 2, activation = 'relu', padding="same")(
encoder_input
)
x = layers.Conv2D(64, (3, 3), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides = 2, activation = 'relu', padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]
x = layers.Flatten()(x)
encoder_output = layers.Dense(2, name="encoder_output")(x)
encoder = models.Model(encoder_input, encoder_output)

decoder_input = layers.Input(shape=(2,), name="decoder_input")
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(
128, (3, 3), strides=2, activation = 'relu', padding="same"
)(x)
x = layers.Conv2DTranspose(
64, (3, 3), strides=2, activation = 'relu', padding="same"
)(x)
x = layers.Conv2DTranspose(
32, (3, 3), strides=2, activation = 'relu', padding="same"
)(x)
decoder_output = layers.Conv2D(
1,
(3, 3),
strides = 1,
activation="sigmoid",
padding="same",
name="decoder_output"
)(x)
decoder = models.Model(decoder_input, decoder_output)

autoencoder = models.Model(encoder_input, decoder(encoder_output))

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

autoencoder.fit(
    x_train,
    x_train,
    epochs=20,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test),
)

autoencoder.evaluate(x_test,x_test)

example_images = x_test[:5000]
predictions = autoencoder.predict(example_images)

embeddings = encoder.predict(example_images)
mins, maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)
sample = np.random.uniform(mins, maxs, size=(18, 2))
reconstructions = decoder.predict(sample)


n = 18  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original images
    # ax = plt.subplot(2, n, i + 1)
    # plt.imshow(example_images[i].reshape(32, 32), cmap='gray')
    # plt.title("Original")
    # plt.axis('off')

    # Display reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructions[i].reshape(32, 32), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()
