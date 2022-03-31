import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "10"

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
import tensorflow_datasets as tfds
import timeit

from fast_augment import center_crop, augment


# benchmark on a synthetic input
batch = tf.cast(tf.random.uniform((128, 224, 224, 3), maxval=255), tf.uint8)
labels = tf.zeros((batch.shape[0], 1000))
number = 1000
time = timeit.timeit(lambda: augment(batch, labels), number=number)
print("Sampling %d batches of %s size in %0.3f s" % (number, batch.shape, time))


# render an example on a real data
print("Getting tf_flowers dataset...")
data, info = tfds.load("tf_flowers", split="train", with_info=True)
class_names = info.features["label"].names

# make batches
data = data.map(
    lambda x: [
        center_crop(x["image"], size=(224, 224)),  # image
        tf.one_hot(x["label"], 37),  # probabilities
    ]
)

# apply augmentation
batch_size = 20
data = data.batch(batch_size).map(lambda x, y: augment(x, y, mixup=0.5))

# take a batch
samples = data.take(1).unbatch().as_numpy_iterator()

# plot images
plt.figure(1)
for i in range(batch_size):
    image, proba = next(samples)

    # show image
    plt.subplot(4, 5, i + 1)
    plt.axis(False)
    plt.imshow(image)

    # get labels and probabilities
    idx = numpy.where(proba > 0)[0]
    assert len(idx) <= 2  # expecting at most 2 classes per image
    proba = 100 * proba[idx]
    label = [class_names[_] for _ in idx]

    # add a title
    if len(proba) == 1:
        plt.gca().set_title(label[0])
    else:
        plt.gca().set_title(
            "%s %d%% / %s %d%%" % (label[0], proba[0], label[1], proba[1])
        )

plt.show()
