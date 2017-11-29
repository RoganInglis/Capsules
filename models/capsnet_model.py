from models import BaseModel
import tensorflow as tf
from models import utils
from tqdm import trange


class CapsNetModel(BaseModel):
    def set_model_props(self, config):
        self.n_primarycaps = config['n_primarycaps']
        self.d_primarycaps = config['d_primarycaps']
        self.n_digitcaps = config['n_digitcaps']
        self.d_digitcaps = config['d_digitcaps']
        self.lambda_reconstruction = config['lambda_reconstruction']

    def get_best_config(self):
        # This function is here to be overridden completely.
        # It returns a dictionary used to update the initial configuration (see __init__)
        return {}

    @staticmethod
    def get_random_config(fixed_params={}):
        # Why static? Because you want to be able to pass this function to other processes
        # so they can independently generate random configuration of the current model
        raise Exception('The get_random_config function must be overriden by the model')

    def build_graph(self, graph):
        with graph.as_default():
            # Create placeholders
            self.placeholders = {'image': tf.placeholder(tf.float32, [None, self.image_dim], name='image'),
                                 'label': tf.placeholder(tf.int32, [None, self.n_classes], name='label')}

            # Define main model graph
            primary_caps_args = [32, 8, 9, 2]
            digit_caps_args = [self.n_classes, 16, 3]
            margin_loss_args = [[0.9, 0.1], 0.5]
            self.loss, self.predictions, self.accuracy, self.summaries = utils.build_capsnet_graph(self.placeholders,
                                                                                                   primary_caps_args,
                                                                                                   digit_caps_args,
                                                                                                   margin_loss_args,
                                                                                                   image_dim=self.image_dim,
                                                                                                   lambda_reconstruction=self.lambda_reconstruction)

            # Define optimiser
            self.optim = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optim.minimize(self.loss, global_step=self.global_step)

            # Set up summaries
            self.train_summary = tf.summary.merge([self.summaries['accuracy'],
                                                   self.summaries['loss'],
                                                   *self.summaries['general']])

            self.validation_summary = tf.summary.merge([self.summaries['accuracy'],
                                                       self.summaries['loss']])

        return graph

    def infer(self, input):
        output = input  # TODO - implement this function
        return output

    def learn_from_epoch(self):
        for _ in range(self.data.train.num_examples//self.batch_size):
            # Get batch
            images, labels = self.data.train.next_batch(self.batch_size)
            feed_dict = {self.placeholders['image']: images,
                         self.placeholders['label']: labels}

            global_step = self.sess.run(self.global_step)  # TODO - add condition for max iteration or limit only by max epoch in main

            op_list = [self.train_op]

            train_summary_now = self.train_summary_every > 0 and global_step % self.train_summary_every == 0
            if train_summary_now:
                op_list.append(self.train_summary)

            train_out = self.sess.run([self.train_op, self.train_summary], feed_dict=feed_dict)

            # Add to tensorboard
            if train_summary_now:
                self.train_summary_writer.add_summary(train_out[1], global_step)

            validate_now = self.validation_summary_every > 0 and global_step % self.validation_summary_every == 0
            if validate_now:
                # Only run 1 batch of validation data while training for speed; can do full test later
                validation_images, validation_labels = self.data.validation.next_batch(self.batch_size)
                validation_feed_dict = {self.placeholders['image']: validation_images,
                                        self.placeholders['label']: validation_labels}

                validation_summary = self.sess.run(self.validation_summary, feed_dict=validation_feed_dict)

                self.validation_summary_writer.add_summary(validation_summary, global_step)
