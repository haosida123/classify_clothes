import tensorflow as tf
# from tensorflow.contrib.quantize.python import quant_ops

DECAY_STEPS = 10000
DECAY_RATE = 0.9


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


def add_layer(name_scope, inputs, weight_shape, stddev):
    with tf.name_scope(name_scope):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(weight_shape, stddev=stddev)
            layer_weights = tf.Variable(initial_value, name='weights')
            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            initial_value = tf.truncated_normal(
                [weight_shape[1]], stddev=stddev)
            layer_biases = tf.Variable(initial_value, name='biases')
            variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.nn.relu(tf.matmul(
                inputs, layer_weights) + layer_biases)
            tf.summary.histogram(name_scope + '_logits', logits)
    return logits


def add_final_training_ops(global_step, class_count, final_tensor_name, bottleneck_tensor,
                           bottleneck_tensor_size, learning_rate):
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

    with tf.name_scope(layer_name):
        logits1 = add_layer('Wx_p_b_1', bottleneck_input, [
                            bottleneck_tensor_size, 512], 0.01)
        logits2 = add_layer('Wx_p_b_2', logits1, [512, 128], 0.01)
        final_logits = add_layer('Wx_p_b_3', logits2, [128, class_count], 0.01)

    final_tensor = tf.nn.softmax(final_logits, name=final_tensor_name)

    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
            labels=ground_truth_input, logits=final_logits)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    # Decay the learning rate exponentially based on the number of steps.
    exp_learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   DECAY_STEPS,
                                                   DECAY_RATE)
    tf.summary.scalar('learning_rate', exp_learning_rate)
    with tf.name_scope('train'):
        optimizer = tf.train.RMSPropOptimizer(exp_learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor)
