import sys
import os
import PIL.Image as Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from svhn_model import classification_head


WEIGHTS_FILE = "classifier.ckpt"


def detect(img_path, saved_model_weights):
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()

    # Load the previously saved model to load the vars into.
    X = tf.placeholder(tf.float32, shape=(1, 32, 32, 3))
    prediction = tf.nn.softmax(classification_head(X))

    # Create a loader
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print("hello")
        print("Loading Saved Checkpoints From:", WEIGHTS_FILE)
        # place the weights into the model.
        saver.restore(sess, saved_model_weights)
        print("Model restored.")

        pix = np.array(img)
        exp = np.expand_dims(pix, axis=0)
        norm_img = (255-pix)*1.0/255.0

        feed_dict = {X: exp}
        predictions = sess.run(prediction, feed_dict=feed_dict)
        print("Best Prediction is:", np.argmax(predictions))

if __name__ == "__main__":
    img_path = None
    if len(sys.argv) > 1:
        print("Reading Image file:", sys.argv[1])
        if os.path.isfile(sys.argv[1]):
            img_path = sys.argv[1]
        else:
            raise EnvironmentError("Cannot open image file.")
    else:
        raise EnvironmentError("You must pass an image file to process")

    if os.path.isfile(WEIGHTS_FILE):
        saved_model_weights = WEIGHTS_FILE
    else:
        raise IOError("Cannot find checkpoint file. Please run train_classifier.py")

    detect(img_path, saved_model_weights)
