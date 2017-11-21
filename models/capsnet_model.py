from models import BaseModel
import tensorflow as tf


class CapsNetModel(BaseModel):
    def set_model_props(self, config):
        self.n_primarycaps = config['n_primarycaps']
        self.d_primarycaps = config['d_primarycaps']
        self.n_digitcaps = config['n_digitcaps']
        self.d_digitcaps = config['d_digitcaps']

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




        return graph

    def infer(self, input):
        output = input  # TODO - implement this function
        return output

    def learn_from_epoch(self):
        # TODO - Implement this
        print("Not implemented yet")

