import tensorflow as tf
# from tensorflow.contrib.quantize.python import quant_ops

DECAY_STEPS = 10000
DECAY_RATE = 0.8


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_training_ops(global_step, class_count, final_tensor_name, bottleneck_tensor,
                           bottleneck_tensor_size, quantize_layer, learning_rate):
    """Adds a new softmax and fully-connected layer for training.

    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.

    The set up for the softmax and fully-connected layers is based on:
    https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

    Args:
      class_count: Integer of how many categories of things we're trying to
          recognize.
      final_tensor_name: Name string for the new final node that produces results.
      bottleneck_tensor: The output of the main CNN graph.
      bottleneck_tensor_size: How many entries in the bottleneck vector.
      quantize_layer: Boolean, specifying whether the newly added layer should be
          quantized.

    Returns:
      The tensors for the training and cross entropy results, and tensors for the
      bottleneck input and ground truth input.
    """
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[None, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(
            tf.int64, [None], name='GroundTruthInput')

    # Organizing the following ops as `final_training_ops` so they're easier
    # to see in TensorBoard
    layer_name = 'final_training_ops'
    n_weights1_colunm = 512
    with tf.name_scope(layer_name):
        with tf.name_scope('weights1'):
            initial_value = tf.truncated_normal(
                [bottleneck_tensor_size, n_weights1_colunm], stddev=0.01)
            layer_weights1 = tf.Variable(initial_value, name='final_weights1')
            variable_summaries(layer_weights1)
        with tf.name_scope('biases1'):
            initial_value = tf.truncated_normal(
                [n_weights1_colunm], stddev=0.01)
            layer_biases1 = tf.Variable(initial_value, name='final_biases1')
            variable_summaries(layer_biases1)
        with tf.name_scope('Wx_plus_b1'):
            logits1 = tf.nn.relu(tf.matmul(
                bottleneck_input, layer_weights1) + layer_biases1)
            tf.summary.histogram('pre_activations1', logits1)

        with tf.name_scope('weights2'):
            initial_value = tf.truncated_normal(
                [n_weights1_colunm, class_count], stddev=0.01)
            layer_weights2 = tf.Variable(initial_value, name='final_weights2')
            variable_summaries(layer_weights2)
        with tf.name_scope('biases2'):
            initial_value = tf.truncated_normal([class_count], stddev=0.01)
            layer_biases2 = tf.Variable(initial_value, name='final_biases2')
            variable_summaries(layer_biases2)
        with tf.name_scope('Wx_plus_b2'):
            logits2 = tf.matmul(logits1, layer_weights2) + layer_biases2
            tf.summary.histogram('pre_activations2', logits2)

    final_tensor = tf.nn.softmax(logits2, name=final_tensor_name)

    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
            labels=ground_truth_input, logits=logits2)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    # Decay the learning rate exponentially based on the number of steps.
    exp_learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   DECAY_STEPS,
                                                   DECAY_RATE)
    tf.summary.scalar('learning_rate', exp_learning_rate)
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(exp_learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor)
