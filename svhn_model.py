import tensorflow as tf
from pdb import set_trace as bp

# Image Parameters
NUM_CHANNELS = 3
CL_NUM_LABELS = 10
NUM_LABELS = CL_NUM_LABELS + 1  # 0-9, + 1 blank


# Hyper Parameters
PATCH_SIZE = 5
DEPTH_1 = 48
DEPTH_2 = 64
DEPTH_3 = 128
DEPTH_4 = 160
LOCAL = 192

DROPOUT = 0.85


# Convolution Weight and Bias Variables
conv1_weights = tf.get_variable("Weights_1", shape=[PATCH_SIZE, PATCH_SIZE,
                                NUM_CHANNELS, DEPTH_1])
conv1_biases = tf.Variable(tf.constant(0.0, shape=[DEPTH_1]), name='Biases_1')

conv2_weights = tf.get_variable("Weights_2", shape=[PATCH_SIZE, PATCH_SIZE,
                                DEPTH_1, DEPTH_2])
conv2_biases = tf.Variable(tf.constant(0.1, shape=[DEPTH_2]), name='Biases_2')

conv3_weights = tf.get_variable("Weights_3", shape=[PATCH_SIZE, PATCH_SIZE,
                                DEPTH_2, DEPTH_3])
conv3_biases = tf.Variable(tf.constant(0.1, shape=[DEPTH_3]), name='Biases_3')

conv4_weights = tf.get_variable("Weights_4", shape=[PATCH_SIZE,
                                PATCH_SIZE, DEPTH_3, DEPTH_4])
conv4_biases = tf.Variable(tf.constant(0.1, shape=[DEPTH_4]), name='Biases_4')


# Regression Weight and Bias Variables
reg1_weights = tf.get_variable("WS1", shape=[DEPTH_4, NUM_LABELS])
reg1_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS1')

reg2_weights = tf.get_variable("WS2", shape=[DEPTH_4, NUM_LABELS])
reg2_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS2')

reg3_weights = tf.get_variable("WS3", shape=[DEPTH_4, NUM_LABELS])
reg3_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS3')

reg4_weights = tf.get_variable("WS4", shape=[DEPTH_4, NUM_LABELS])
reg4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS4')

reg5_weights = tf.get_variable("WS5", shape=[DEPTH_4, NUM_LABELS])
reg5_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS5')

# Classification Weight and Bias Variables

cl_l3_weights = tf.get_variable("Classifer_Weights_1", shape=[DEPTH_3, DEPTH_4])
cl_l3_biases = tf.Variable(tf.constant(0.05, shape=[DEPTH_4]),
                           name='Classifer_Biases_1')

# cl_l4_weights = tf.get_variable("Classifer_Weights_2", shape=[384, 192])
# cl_l4_biases = tf.Variable(tf.constant(0.0, shape=[192]),
#                            name='Classifer_Biases_2')

cl_out_weights = tf.get_variable("Classifer_Weights_3",
                                 shape=[DEPTH_4, CL_NUM_LABELS])
cl_out_biases = tf.Variable(tf.constant(0.05, shape=[CL_NUM_LABELS]),
                            name='Classifer_Biases_3')


def activation_summary(x):
    tensor_name = x.op.name
    # tf.histogram_summary(tensor_name + '/activations', x)
    # tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def convolution_model(data):
    with tf.variable_scope('Layer_1', reuse=True) as scope:
        con = tf.nn.conv2d(data, conv1_weights,
                           [1, 1, 1, 1], 'VALID', name='C1')
        hid = tf.nn.relu(con + conv1_biases)
        activation_summary(hid)

    pol = tf.nn.max_pool(hid,
                         [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='Pool_1')
    lrn = tf.nn.local_response_normalization(pol, name="Normalize_1")

    with tf.variable_scope('Layer_2') as scope:
        con = tf.nn.conv2d(lrn, conv2_weights,
                           [1, 1, 1, 1], padding='VALID', name='C3')
        hid = tf.nn.relu(con + conv2_biases)
        activation_summary(hid)

    pol = tf.nn.max_pool(hid,
                         [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='Pool_2')
    lrn = tf.nn.local_response_normalization(pol, name="Normalize_2")

    with tf.variable_scope('Layer_3') as scope:
        con = tf.nn.conv2d(lrn, conv3_weights,
                           [1, 1, 1, 1], padding='VALID', name='C5')
        hid = tf.nn.relu(con + conv3_biases)
        lrn = tf.nn.local_response_normalization(hid)

        if lrn.get_shape().as_list()[1] is 1:  # Is already reduced.
            sub = tf.nn.max_pool(lrn,
                                 [1, 1, 1, 1], [1, 1, 1, 1], 'SAME', name='S5')
        else:
            sub = tf.nn.max_pool(lrn,
                                 [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='S5')

        activation_summary(sub)

    return sub


def classification_head(data, keep_prob=1.0, train=False):
    conv_layer = convolution_model(data)
    shape = conv_layer.get_shape().as_list()
    dim = shape[1] * shape[2] * shape[3]

    if train is True:
        print("Using drop out")
        conv_layer = tf.nn.dropout(conv_layer, DROPOUT)
    else:
        print("Not using dropout")

    # Fully Connected Layer 1
    with tf.variable_scope('fully_connected_1') as scope:
        fc1 = tf.reshape(conv_layer, [shape[0], -1])
        fc1 = tf.add(tf.matmul(fc1, cl_l3_weights), cl_l3_biases)
        fc_out = tf.nn.relu(fc1, name=scope.name)
        activation_summary(fc_out)

    with tf.variable_scope("softmax_linear") as scope:
        logits = tf.matmul(fc_out, cl_out_weights) + cl_out_biases
        activation_summary(logits)

    # Output class scores
    return logits


def regression_head(data, train=False):
    conv_layer = convolution_model(data)

    # with tf.name_scope('dropout'):
    # if train is True:
    #     print("Using drop out")
    #     conv_layer = tf.nn.dropout(conv_layer, DROPOUT)
    # else:
    #     print("Not using dropout")
    
    with tf.variable_scope('full_connected_1') as scope:
        con = tf.nn.conv2d(conv_layer, conv4_weights, [1, 2, 2, 1], padding='VALID', name='C5')
        hid = tf.nn.relu(con + conv4_biases)
        activation_summary(hid)
    shape = hid.get_shape().as_list()
    reshape = tf.reshape(hid, [shape[0], shape[1] * shape[2] * shape[3]])

    with tf.variable_scope('Output') as scope:
        logits_1 = tf.matmul(reshape, reg1_weights) + reg1_biases
        logits_2 = tf.matmul(reshape, reg2_weights) + reg2_biases
        logits_3 = tf.matmul(reshape, reg3_weights) + reg3_biases
        logits_4 = tf.matmul(reshape, reg4_weights) + reg4_biases
        logits_5 = tf.matmul(reshape, reg5_weights) + reg5_biases

    return [logits_1, logits_2, logits_3, logits_4, logits_5]