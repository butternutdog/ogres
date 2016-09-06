"""
Basics
======

Fundamental building blocks for ogres networks.

Must be imported before any other ogre components
are imported.
"""

import tensorflow as tf

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1) # Maybe look at Saxe paper for weight initialization later...
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    tf.histogram_summary(name, var)
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('stddev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))


class Net:
    """
    Class for easy definition of a neural net graph.
    Most methods add a layer to the graph, and also sets up name 
    scoping so that the resultant graph is easy to read, and
    adds a number of summary ops.
    """ 
    
    def __init__(self, inlayer):
        if(isinstance(inlayer, dict)):
            self.layers = inlayer
        else:
            self.layers = [ {
                "activations": inlayer,
                "type": "input"
            }]

    def dense(self, width=100, act=tf.nn.relu):
        """
        Fully connected layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        """
        input_tensor = self.layers[-1]["activations"]
        layer_name = "dense" + str(len([l for l in self.layers
            if l["type"]=="dense"]))
        input_dim = reduce(lambda p,f: p*f, input_tensor.get_shape()[1:].as_list(), 1)
        input_tensor = tf.reshape(input_tensor, (-1, input_dim))
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, width])
                variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                biases = bias_variable([width])
                variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                activations = act(preactivate, 'activation')
                tf.histogram_summary(layer_name + '/activations', activations)
        self.layers.append( {
            "activations": activations,
            "weights": weights,
            "biases": biases,
            "type": "dense"
            } )
        return self
    
    def reshape(self, shape=[]):
        if len(shape) == 0:
            return self
        input_tensor = self.layers[-1]["activations"]
        activations = tf.reshape(input_tensor, shape)
        self.layers.append( {
            "activations": activations,
            "type": "reshape"
        } )
        return self

    def dropout(self, keep_prob=1.0):
        input_tensor = self.layers[-1]["activations"]
        activations = tf.nn.dropout(input_tensor, keep_prob)
        self.layers.append( {
            "activations": activations,
            "type": "dropout"
        } )
        return self
    
    def bn(self, act=tf.nn.relu):
        """
        Batch normalization.
        See: http://arxiv.org/pdf/1502.03167v3.pdf
        Based on implementation found at: 
        http://www.r2rt.com/posts/implementations/2016-03-29-implementing-batch-normalization-tensorflow/
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.

        layer_name = "bn" + str(len([l for l in self.layers
            if l["type"]=="bn"]))

        input_tensor = self.layers[-1]["activations"]
        
        with tf.name_scope(layer_name):
            
            dim = input_tensor.get_shape()[1:] # 64, 1, 10, 100
            
            beta = tf.Variable(tf.zeros(dim))
            scale = tf.Variable(tf.ones(dim))
            variable_summaries(beta, layer_name + "/beta")
            variable_summaries(scale, layer_name + "/scale")
            z = input_tensor
            batch_mean, batch_var = tf.nn.moments(input_tensor,[0])
            epsilon = 1e-3
            z_hat = (z - batch_mean) / tf.sqrt(batch_var + epsilon)
            bn_z = scale * z_hat + beta
            activations = act(bn_z, 'activation')
            tf.histogram_summary(layer_name + '/activations', activations)
              
        self.layers.append({
            "activations": activations,
            "type": "bn"})
        return self

    def output(self):
        """Returns output from last layer"""
        return self.layers[-1]["activations"]

def layer(new_layer):
    """
    Functions with this decorator can be used as layers in the network. 
    """
    setattr(Net, new_layer.func_name, new_layer)