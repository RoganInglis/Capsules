import tensorflow as tf


def squash(tensor, axis=0):
    with tf.name_scope('squash'):
        vector_norms = tf.norm(tensor, axis=axis, keep_dims=True)

        squash_coefficients = vector_norms/(1 + tf.square(vector_norms))

        squashed_tensor = tf.multiply(squash_coefficients, tensor)

    return squashed_tensor


def capsule_affine_transform(input_tensor, out_capsule_dim, scope=None, reuse=None):
    """
    Performs the affine transform for each capsule output -> next capsule input
    :param input_tensor: Tensor with shape [batch_size, n_in_capsules, in_capsule_dim]  ([-1, 6*6*32, 8] in paper)
    :param out_capsule_dim: dimension of the output (next capsule input) vectors
    :param scope:
    :param reuse:
    :return: Tensor with shape [batch_size, n_in_capsules, out_capsule_dim]  ([-1, 6*6*32, 16] in paper)
    """
    # TODO - implementing this in the straight forward but inefficient way for now; should re-implement using tf.scan
    input_shape = input_tensor.shape
    with tf.variable_scope(scope, reuse=reuse):
        # Create weight variables
        weights = tf.Variable(tf.random_normal([input_shape[1], input_shape[2], out_capsule_dim]), 'affine_weights')

        # Unstack input_tensor and weights
        input_tensor_unstacked = tf.unstack(input_tensor, axis=1)  # n_in_capsules*[batch_size, in_capsule_dim]
        weights_unstacked = tf.unstack(weights, axis=0)  # n_in_capsules*[in_capsule_dim, out_capsule_dim]

        # Iterate through unstacked tensors and perform affine transforms
        transformed_input = []
        for u, W in zip(input_tensor_unstacked, weights_unstacked):
            transformed_input.append(tf.matmul(u, W))

        output = tf.stack(transformed_input, axis=1)  # [batch_size, n_in_capsules, out_capsule_dim]

    return output


def dynamic_routing(u_hat, n_capsules, n_routing_iterations):
    """
    Sets up the TensorFlow graph for the dynamic routing between capsules section of the CapsNet architecture (although
    it should be usable for dynamic routing between arbitrary capsules) as presented in 'Dynamic routing between capsules'
    :param u_hat: Tensor with shape [batch_size, n_input_capsules, transformed_capsule_dimension] (-1, 6*6*32, 16)
    :param n_capsules: Int - Number of output capsules (10 in paper)
    :param n_routing_iterations: Int - Number of routing iterations to perform (3 in paper)
    :return: capsule output Tensor with shape [batch_size, n_capsules, out_capsule_dimension] ([-1, 10, 16] in paper)
    """
    with tf.name_scope('dynamic_routing'):
        # Create initial routing logits b
        input_shape = u_hat.shape
        b_init = tf.zeros([input_shape[0], input_shape[1], n_capsules])
        b = tf.Variable(b_init, trainable=False)

        # Initialise list of output vectors
        v_list = list(range(n_capsules))

        for routing_iteration in range(n_routing_iterations):
            for out_capsule in range(n_capsules):
                # Compute c as softmax of b
                c = tf.nn.softmax(b[:, :, out_capsule], dim=1)  # [batch_size, n_input_capsules, 1]

                # Compute s as sum of input capsule vectors scaled by c
                s = tf.reduce_sum(tf.multiply(u_hat, c), axis=1)  # [batch_size, output_capsule_dim]

                # Apply squash non-linearity to get output v
                v_list[out_capsule] = squash(s, axis=1)  # [batch_size, output_capsule_dim]

                # Update b
                if routing_iteration == n_routing_iterations - 1:
                    # Set b back to zero for next run
                    b[:, :, out_capsule] = tf.assign(b[:, :, out_capsule], b_init[:, :, out_capsule])  # TODO - test this is actually working
                else:
                    b_update = tf.reduce_sum(tf.multiply(u_hat, tf.expand_dims(v_list[out_capsule], axis=1)), axis=2) # u^ = [batch_size, n_input_capsules, output_capsule_dim]  v = [batch_size, output_capsule_dim]
                    b[:, :, out_capsule] = tf.assign_add(b[:, :, out_capsule], b_update)

        # Concatenate output vectors for each capsule
        v = tf.stack(v_list, axis=1)  # [batch_size, n_capsules, output_capsule_dim]

    return v


def primary_caps(input_tensor, n_channels, capsule_dim, kernel_shape, strides):
    """
    Sets up the TensorFlow graph for the PrimaryCaps layer as in 'Dynamic routing between capsules'
    :param input_tensor: Tensor with shape [batch_size, im_x, im_y, channels]
    :param n_channels: Int - Number of different channels of capsules to use (32 in paper)
    :param capsule_dim: Int - Number of dimensions for each capsule (8 in paper)
    :param kernel_shape: Int or List - Shape of the convolution kernel to use (passed directly to tf.layers.conv2d)
    :param strides: Int or List - strides to use for convolution (passed directly to tf.layers.conv2d)
    :return: Tensor with shape [batch_size, out_im_x, out_im_y, n_channels, capsule_dim] ([-1, 6, 6, 32, 8] in paper)
    """
    conv_out = tf.layers.conv2d(input_tensor, n_channels*capsule_dim, kernel_shape, strides)  # [batch_size, 6, 6, 32*8]

    # Split and concatenate to separate channels
    conv_out = tf.split(conv_out, n_channels, axis=3)  # 32*[batch_size, 6, 6, 8]
    conv_out = tf.stack(conv_out, axis=3)  # [batch_size, 6, 6, 32, 8]

    conv_out = squash(conv_out, axis=4)

    return conv_out


def digit_caps(input_tensor, n_capsules, capsule_dim, n_routing_iterations):
    # Reshape input for affine transform to [batch_size, -1, input_capsule_dim]
    input_shape = input_tensor.shape
    input_tensor = tf.reshape(input_tensor, [input_shape[0], -1, input_shape[4]])  # [batch_size, 6*6*32, 8]

    # Input affine transforms
    transformed_input = capsule_affine_transform(input_tensor, capsule_dim)  # [batch_size, 6*6*32, 16]

    # Dynamic routing
    routed_output = dynamic_routing(transformed_input, n_capsules, n_routing_iterations)

    return routed_output


def margin_loss(input_tensor, label_ph, margins):
    """
    Contructs the TensorFlow graph for the margin loss described in 'Dynamic routing between capsules'
    :param input_tensor: Tensor with shape [batch_size, n_classes, capsule_dim]  ([-1, 10, 16] in paper)
    :param label_ph: Placeholder for the true labels
    :param margins: List containing both the high and low margins in that order (i.e. [0.9, 0.1] for the paper margins)
    :return: The margin loss as a scalar tensor
    """
    input_shape = input_tensor.shape

    loss_list = []
    for k in range(input_shape[1]):
        L_k =

    loss = tf.add_n(loss_list)

    return loss


def reconstruction_net():



def reconstruction_loss():



def build_capsnet_graph(input_placeholders, primary_caps_args, digit_caps_args):
    # Create first convolutional layer
    conv1_out = tf.layers.conv2d(input_placeholders['inputs'], 256, 9, 1,
                                 activation=tf.nn.relu)  # [batch_size, 20, 20, 256] (if using mnist)

    # Create PrimaryCapsules
    with tf.name_scope('PrimaryCaps'):
        primary_caps_out = primary_caps(conv1_out, *primary_caps_args)

    # Create DigitCaps
    with tf.name_scope('DigitCaps'):
        digit_caps_out = digit_caps(primary_caps_out, *digit_caps_args)

    # Create Margin Loss
    with tf.name_scope('MarginLoss'):


    # Create Reconstruction Loss
    with tf.name_scope('ReconstructionLoss'):
