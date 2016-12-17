import sys
import os
import numpy as np
import tensorflow as tf
import PIL.Image as Image
import matplotlib.pyplot as plt

from svhn_model import regression_head
from svhn_data import load_svhn_data
import time

test_dataset, test_labels = load_svhn_data("test", "full")
WEIGHTS_FILE = "regression.ckpt"


def prediction_to_string(pred_array):
    pred_str = ""
    for i in range(len(pred_array)):
        if pred_array[i] != 10:
            pred_str += str(pred_array[i])
        else:
            return pred_str
    return pred_str


def detect(img_path, saved_model_weights):
    sample_img = Image.open(img_path)
    plt.imshow(sample_img)
    plt.show()

    pix = np.array(sample_img)
    norm_pix = (255-pix)*1.0/255.0
    exp = np.expand_dims(norm_pix, axis=0)

    X = tf.placeholder(tf.float32, shape=(1, 64, 64, 3))
    [logits_1, logits_2, logits_3, logits_4, logits_5] = regression_head(X)

    predict = tf.pack([tf.nn.softmax(logits_1),
                      tf.nn.softmax(logits_2),
                      tf.nn.softmax(logits_3),
                      tf.nn.softmax(logits_4),
                      tf.nn.softmax(logits_5)])

    best_prediction = tf.transpose(tf.argmax(predict, 2))

    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, "regression.ckpt")
        print "Model restored."

        print "Initialized"
        feed_dict = {X: exp}
        start_time = time.time()
        predictions = session.run(best_prediction, feed_dict=feed_dict)
        pred = prediction_to_string(predictions[0])
        end_time = time.time()
        print "Best Prediction", pred, "made in", end_time - start_time


if __name__ == "__main__":
    img_path = None
    if len(sys.argv) > 1:
        print("Reading Image file:", sys.argv[1])
        if os.path.isfile(sys.argv[1]):
            img_path = sys.argv[1]
        else:
            raise EnvironmentError("Image file cannot be opened.")
    else:
        raise EnvironmentError("You must pass an image file to process")
    if os.path.isfile(WEIGHTS_FILE):
        saved_model_weights = WEIGHTS_FILE
    else:
        raise IOError("Cannot find checkpoint file. Please run train_regressor.py")
    detect(img_path, saved_model_weights)
