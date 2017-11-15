from models import BaseModel
import tensorflow as tf


class CapsNetModel(BaseModel):
    def set_model_props(self, config):
        # This function is here to be overridden completely.
        # When you look at your model, you want to know exactly which custom options it needs.
        pass

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
            self.placeholders = {'image': tf.placeholder('int', []),
                                 'label': tf.placeholder('int', [None])}

            # Create first convolutional layer
            conv1_out = tf.layers.conv2d(self.placeholders['inputs'], 256, 9, 1)  # [batch_size, 20, 20, 256] (if using mnist)

            # Create PrimaryCapsules
            primarycaps = tf.layers.conv2d(conv1_out, 256, 9, 2)  # [batch_size, 6, 6, 256]
            primarycaps = tf.split(primarycaps, 8, axis=3)  # 32*[batch_size, 6, 6, 8]

            # Create DigitCaps


        return graph

    def infer(self, input):
        output = input  # TODO - implement this function
        return output

    def learn_from_epoch(self):

