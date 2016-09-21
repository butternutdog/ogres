"""
Convolutional layers, and associated functions like
pooling layers etc.
"""

import functools
import operator

from .. import layer, tf, weight_variable, bias_variable, variable_summaries

@layer
def conv2d(self, filters=12, size=[3,3], act=tf.nn.relu, stride=1):
    """
    Convolutional layer in 2 dimensions.
    
    Args:
    `filters: int` -- Number of filter channels in the output.
    `size: (int, int)` -- The spatial extent of the filters.
    `stride: int | [int, int, int, int]` -- Step length when sliding the filter.
        Strides are ordered as `[batch, in_height, in_width, in_channels]`. If
        `stride` is just a single number, it is interpreted as `[1, stride, stride, 1]`
    `act: tf activation function` -- A Tensorflow activation function. Defaults
        to `tf.nn.relu`.
    """
    input_tensor = self.layers[-1]["activations"]
    layer_name = "conv" + str(len([l for l in self.layers
        if l["type"]=="conv"]))
    input_dim = functools.reduce(operator.mul, input_tensor.get_shape()[1:-1].as_list(), 1)
    input_filters = int(input_tensor.get_shape()[-1])
        
    if isinstance(stride, int):
        stride = [1, stride, stride, 1]

    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable((size[0], size[1], input_filters, filters))
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([filters])
            variable_summaries(biases, layer_name + '/biases')
        convs = tf.nn.conv2d(input_tensor, weights, stride, 'SAME',
                    use_cudnn_on_gpu=True, name=layer_name + "/conv")
        activations = act(convs + biases)
    self.layers.append( {
        "activations": activations,
        "weights": weights,
        "biases": biases,
        "type": "conv"
        } )
    return self


@layer
def pool2d(self, size=[2,2], stride=2):
    if isinstance(stride, int):
        stride = [1, stride, stride, 1]
    input_tensor = self.layers[-1]["activations"]
    activations = tf.nn.max_pool(
        input_tensor, [1, size[0], size[1], 1], stride, 'SAME')
    self.layers.append( {
        "activations": activations,
        "type": "pool"
        } )
    return self


@layer
def conv1d(self, filters=12, size=5, act=tf.nn.relu, stride=1):
    input_tensor = self.layers[-1]["activations"]
    assert len(input_tensor.get_shape()) == 3
    self.layers.append( {
        "activations": tf.expand_dims(input_tensor, 1),
        "type": "expand_dim"
        } )
    conv = self.conv2d(filters=filters,size=[1,size], act=act, stride=[1,1,stride,1])
    output_layer = self.layers[-1]["activations"]
    self.layers.append( {
        "activations": tf.squeeze(output_layer, [1]),
        "type": "squeeze_dim"
        } )
    return self


@layer
def pool1d(self, size=2, stride=2):
    input_tensor = self.layers[-1]["activations"]
    expand_tensor = tf.expand_dims(input_tensor, 1)
    pooled = tf.nn.max_pool(
        expand_tensor, [1, 1, size, 1],
        [1, 1, stride, 1], 'SAME')
    squeezed = tf.squeeze(pooled, [1])
    self.layers.append( {
        "activations": squeezed,
        "type": "pool"
        } )
    return self 


@layer
def rec_conv1d(self, filters=12, size=5, unrollings=3, input_act=tf.nn.relu, rec_act=tf.nn.relu):
    input_tensor = self.layers[-1]["activations"]
    layer_name = "rec_conv1d" + str(len([l for l in self.layers
        if l["type"]=="rec_conv1d"]))
    input_dim = functools.reduce(operator.mul, input_tensor.get_shape()[1:-1].as_list(), 1)
    input_filters = int(input_tensor.get_shape()[-1])
    STRIDES = [1, 1, 1, 1]
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('/input/weights'):
            weights = weight_variable((1, 1, input_filters, filters))
            variable_summaries(weights, layer_name + '/input/weights')
        with tf.name_scope('/rec/weights'):
            rec_weights = weight_variable((1, size, filters, filters))
            variable_summaries(rec_weights, layer_name + '/rec/weights')

        conv = tf.nn.conv2d(input_tensor, weights, STRIDES, 'SAME',
            use_cudnn_on_gpu=True, name=layer_name + "/input/conv")
        
        self.layers.append({
            "activations": conv,
            "type": "rec_conv1d_input"})
        
        self.bn(act=input_act)
        
        for i in range(unrollings):
            new_conv = tf.nn.conv2d(self.layers[-1]["activations"], rec_weights, STRIDES, 'SAME',
                use_cudnn_on_gpu=True, name=layer_name + "/rec/conv%s" % i)
            x = new_conv + conv
            
            self.layers.append({
                "activations": x,
                "type": "rec_conv1d_unrolling"})
            
            self.bn(act=rec_act)
            
    self.layers.append({
            "activations": self.layers[-1]["activations"],
            "type": "rec_conv1d"})
    return self
