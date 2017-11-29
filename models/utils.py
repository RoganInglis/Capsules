import tensorflow as tf


def squash(tensor, axis=0):
    with tf.name_scope('squash'):
        vector_norms = tf.norm(tensor, axis=axis, keep_dims=True)

        squash_coefficients = vector_norms/(1 + tf.square(vector_norms))

        squashed_tensor = tf.multiply(squash_coefficients, tensor)

    return squashed_tensor


def capsule_affine_transform(input_tensor, n_out_capsules, out_capsule_dim, scope=None, reuse=None):
    """
    Performs the affine transform for each capsule output -> next capsule input
    :param input_tensor: Tensor with shape [batch_size, n_in_capsules, in_capsule_dim]  ([-1, 6*6*32, 8] in paper)
    :param n_out_capsules: number of capsules in the next layer
    :param out_capsule_dim: dimension of the output (next capsule input) vectors
    :param scope:
    :param reuse:
    :return: Tensor with shape [batch_size, n_in_capsules, n_out_capsules, out_capsule_dim]  ([-1, 6*6*32, 10, 16] in paper)
    """
    input_shape = input_tensor.get_shape().as_list()

    # Tile and expand input for matmul for each output capsule
    expanded_input = tf.expand_dims(input_tensor, -2)  # [batch_size, n_in_capsules, 1, in_capsule_dim]
    expanded_input = tf.expand_dims(expanded_input, -2)  # [batch_size, n_in_capsules, 1, 1, in_capsule_dim]
    tiled_input = tf.tile(expanded_input, [1, 1, n_out_capsules, 1, 1])  # TODO - Memory inefficient to have to tile. Is there a better way?

    # Create weight variables
    weight_shape = [1, input_shape[1], n_out_capsules, input_shape[2], out_capsule_dim]
    weights = tf.Variable(tf.random_normal(weight_shape), name='affine_weights')  # [1, n_in_caps, n_out_caps, in_cap_dim, out_cap_dim]

    # Tile weights for batch  TODO - Again memory inefficient to have to tile if there is a better way
    tiled_weights = tf.tile(weights, [tf.shape(input_tensor)[0], 1, 1, 1, 1])  # [batch_size, n_in_caps, n_out_caps, in_cap_dim, out_cap_dim]

    matmul_output = tf.matmul(tiled_input, tiled_weights)
    output = tf.squeeze(matmul_output, -2)  # [batch_size, n_in_caps, n_out_caps, out_capsule_dim]

    return output


def dynamic_routing(u_hat, n_routing_iterations):
    """
    Sets up the TensorFlow graph for the dynamic routing between capsules section of the CapsNet architecture (although
    it should be usable for dynamic routing between arbitrary capsules) as presented in 'Dynamic routing between capsules'
    :param u_hat: Tensor with shape [batch_size, n_input_capsules, n_output_capsules, transformed_capsule_dimension] (-1, 6*6*32, 10, 16)
    :param n_routing_iterations: Int - Number of routing iterations to perform (3 in paper)
    :return: capsule output Tensor with shape [batch_size, n_output_capsules, out_capsule_dimension] ([-1, 10, 16] in paper)
    """
    with tf.name_scope('dynamic_routing'):
        # Create initial routing logits b
        input_shape = tf.shape(u_hat)
        batch_size = input_shape[0]
        n_input_capsules = u_hat.get_shape().as_list()[1]
        n_out_capsules = u_hat.get_shape().as_list()[2]

        b = tf.zeros([batch_size, n_input_capsules, n_out_capsules, 1])  # TODO - double check this works as intended

        u_hat_stopped = tf.stop_gradient(u_hat)

        for routing_iteration in range(n_routing_iterations - 1):
            with tf.name_scope("routing_iteration_{}".format(routing_iteration)):
                # Compute c as softmax of b
                c = tf.nn.softmax(b, dim=1)

                # Compute s as sum of input capsule vectors scaled by c
                s = tf.reduce_sum(tf.multiply(u_hat_stopped, c), axis=1, keep_dims=True)  # [batch_size, 1, n_output_capsules, output_capsule_dim]

                # Apply squash non-linearity to get output v
                v = squash(s, axis=3)  # [batch_size, 1, n_output_capsules, output_capsule_dim]

                # Update b
                b_update = tf.reduce_sum(tf.multiply(u_hat, v), axis=3, keep_dims=True)
                b += b_update

        with tf.name_scope("routing_iteration_{}".format(n_routing_iterations - 1)):
            # Compute c as softmax of b
            c = tf.nn.softmax(b, dim=1)

            # Compute s as sum of input capsule vectors scaled by c
            s = tf.reduce_sum(tf.multiply(u_hat, c), axis=1,
                              keep_dims=True)  # [batch_size, 1, n_output_capsules, output_capsule_dim]

            # Apply squash non-linearity to get output v
            v = tf.squeeze(squash(s, axis=3), 1)  # [batch_size, n_output_capsules, output_capsule_dim]
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


def digit_caps(u, n_capsules, capsule_dim, n_routing_iterations):
    # Reshape input for affine transform to [batch_size, -1, input_capsule_dim]
    input_shape = u.get_shape().as_list()
    u = tf.reshape(u, [-1, input_shape[1] * input_shape[2] * input_shape[3], input_shape[4]])  # [batch_size, 6*6*32, 8]

    # Input affine transforms
    u_hat = capsule_affine_transform(u, n_capsules, capsule_dim)  # [batch_size, 6*6*32, 10, 16]

    # Dynamic routing
    v = dynamic_routing(u_hat, n_routing_iterations)  # [batch_size, 10, 16]

    return v


def margin_loss(preds, label_ph, margins, lambda_m):
    """
    Constructs the TensorFlow graph for the margin loss described in 'Dynamic routing between capsules'
    :param preds: Tensor with shape [batch_size, n_classes, capsule_dim]  ([-1, 10, 16] in paper)
    :param label_ph: Placeholder for the true labels in one-hot format
    :param margins: List containing both the high and low margins in that order (i.e. [0.9, 0.1] for the paper margins)
    :param lambda_m: Down weighting parameter for absent classes
    :return: The margin loss as a scalar tensor
    """
    loss = tf.cast(label_ph, tf.float32)*tf.square(tf.maximum(0., margins[0] - preds)) + lambda_m*tf.cast((1 - label_ph), tf.float32)*tf.square(tf.maximum(0., preds - margins[1]))

    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=0)

    return loss


def reconstruction_net(input_tensor, label_ph, image_dim=784):
    """
    Constructs the TensorFlow graph for the reconstruction net for the CapsNet architecture
    :param input_tensor: Tensor with shape [batch_size, n_classes, capsule_dim]  ([-1, 10, 16] in paper)
    :param label_ph: Placeholder for the true labels in one-hot format (shape [batch_size, n_classes])
    :param image_dim: dimension of the image to reconstruct (i.e. total number of pixels - 784 for paper/mnist)
    :return: Reconstructed images as a Tensor with shape [batch_size, image_dims] (i.e. the flattened image)
    """
    with tf.variable_scope("reconstruction_net"):
        # First mask input so only output vector of correct capsule is used for reconstruction
        masked_input = tf.multiply(input_tensor, tf.expand_dims(tf.cast(label_ph, tf.float32), axis=2))  # [batch_size, n_classes, capsule_dim]
        masked_input = tf.reduce_sum(masked_input, axis=1)  # [batch_size, capsule_dim]

        # Define fully connected layers for reconstruction
        fc_1_out = tf.layers.dense(masked_input, 512, activation=tf.nn.relu)

        fc_2_out = tf.layers.dense(fc_1_out, 1024, activation=tf.nn.relu)

        reconstruction = tf.layers.dense(fc_2_out, image_dim, activation=tf.nn.sigmoid)

        return reconstruction


def reconstruction_loss(reconstructed_image, true_image):
    """
    Constructs the reconstruction loss for the CapsNet architecture
    :param reconstructed_image: Tensor with shape [batch_size, image_dims] ([-1, 784] in paper for mnist)
    :param true_image: Tensor with shape [batch_size, image_dims] ([-1, 784] in paper for mnist)
    :return: Reconstruction loss as a scalar tensor
    """
    loss = tf.reduce_mean(tf.norm(reconstructed_image - true_image, axis=1), axis=0)

    return loss


def build_capsnet_graph(input_placeholders, primary_caps_args, digit_caps_args, margin_loss_args, image_dim=784,
                        lambda_reconstruction=0.0005):
    # Initalise summaries dict - using dict so that we can merge only select summaries; don't want image summaries all
    # the time
    summaries = {}
    summaries["general"] = []

    # Reshape flattened image tensor to 2D
    images = tf.reshape(input_placeholders['image'], [-1, 28, 28, 1])
    summaries['images'] = tf.summary.image('input_images', images)

    # Create first convolutional layer
    conv1_out = tf.layers.conv2d(images, 256, 9, 1,
                                 activation=tf.nn.relu)  # [batch_size, 20, 20, 256] (if using mnist)
    summaries['conv1_out_channels'] = [tf.summary.image('conv1_out_channels', x) for x in tf.unstack(conv1_out, axis=3)]

    # Create PrimaryCapsules
    with tf.variable_scope('PrimaryCaps'):
        primary_caps_out = primary_caps(conv1_out, *primary_caps_args)

    # Create DigitCaps
    with tf.variable_scope('DigitCaps'):
        digit_caps_out = digit_caps(primary_caps_out, *digit_caps_args)  # [batch_size, n_capsules, n_capsule_dims]

    probs = tf.norm(digit_caps_out, axis=2)  # [batch_size, n_classes]
    with tf.name_scope("accuracy"):
        predictions = tf.argmax(probs, axis=1)
        labels = tf.argmax(input_placeholders['label'], axis=1)
        correct = tf.cast(tf.equal(labels, predictions), tf.int32)
        accuracy = tf.reduce_sum(correct)/tf.shape(correct)[0]  # reduce_mean not working here for some reason
        summaries['accuracy'] = tf.summary.scalar('accuracy', accuracy)

    # Create Loss
    with tf.variable_scope('Loss'):
        # Create Margin Loss
        with tf.variable_scope('MarginLoss'):
            m_loss = margin_loss(probs, input_placeholders['label'], *margin_loss_args)

        # Create Reconstruction Loss
        with tf.variable_scope('ReconstructionLoss'):
            reconstructed_image = reconstruction_net(digit_caps_out, input_placeholders['label'], image_dim)
            summaries['reconstructed_images'] = tf.summary.image('reconstructed_images',
                                                                 tf.reshape(reconstructed_image, [-1, 28, 28, 1]))

            r_loss = reconstruction_loss(reconstructed_image, input_placeholders['image'])

        loss = m_loss + lambda_reconstruction*r_loss
        summaries['loss'] = tf.summary.scalar('loss', loss)
        summaries["general"].append(tf.summary.scalar("margin_loss", m_loss))
        summaries["general"].append(tf.summary.scalar("reconstruction_loss", r_loss))

    return loss, predictions, accuracy, summaries

