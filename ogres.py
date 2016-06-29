import tensorflow as tf

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
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

    def __init__(self, inlayer):
        if(isinstance(inlayer, dict)):
            self.layers = inlayer
        else:
            self.layers = [ {
                "activations": inlayer,
                "type": "input"
            }]

    def dense(self, width = 100, act = tf.nn.relu):
        """
        Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read, and
        adds a number of summary ops.
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

    def dropout(self, keep_prob=1.0):
        input_tensor = self.layers[-1]["activations"]
        activations = tf.nn.dropout(input_tensor, keep_prob)
        self.layers.append( {
            "activations": activations,
            "type": "dropout"
        } )
        return self

    def conv2d(self, filters, extent, stride, act=tf.nn.relu):
        """
        Adds a 2d convolutional layer to the network.

        Args:
        `filters: int` -- Number of filter channels in the output.
        `extent: int | (int, int)` -- The spatial extent of the filters. If
            `extent` is just a single number, the filter is assumed to be square.
        `stride: int | [int, int, int, int]` -- Step length when sliding the filter.
            Strides are ordered as `[batch, in_height, in_width, in_channels]`. If
            `stride` is just a single number, it is interpreted as `[1, stride, stride, 1]`
        `act: tf activation function` -- A Tensorflow activation function. Defaults
            to `tf.nn.relu`.
        """
        input_tensor = self.layers[-1]["activations"]
        layer_name = "conv" + str(len([l for l in self.layers
            if l["type"]=="conv"]))
        input_dim = reduce(lambda p,f: p*f, input_tensor.get_shape()[1:-1].as_list(), 1)
        number_of_channels = int(input_tensor.get_shape()[-1])

        if isinstance(filters, int):
            extent_x = extent_y = extent
        elif len(extent) == 2:
            extent_x, extent_y = extent
        else:
            raise Exception("Wrong format for `extent`")

        if isinstance(stride, int):
            stride = [1, stride, stride, 1]
        elif len(extent) == 4:
            stride = stride
        else:
            raise Exception("Wrong format for `stride`")

        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable(
                    ( extent_x, extent_y,
                    number_of_channels, filters))
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

    def conv1d(self, filters = 12, size = 5, act=tf.nn.relu, stride = 1):
        """
        Reusable code for making a convolutional neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read, and
        adds a number of summary ops.
        """
        input_tensor = self.layers[-1]["activations"]
        layer_name = "conv" + str(len([l for l in self.layers
            if l["type"]=="conv"]))
        input_dim = reduce(lambda p,f: p*f, input_tensor.get_shape()[1:-1].as_list(), 1)
        input_filters = int(input_tensor.get_shape()[-1])
        STRIDES = [1, 1, stride, 1]
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable((1, size, input_filters, filters))
                variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                biases = bias_variable([filters])
                variable_summaries(biases, layer_name + '/biases')
            convs = tf.nn.conv2d(input_tensor, weights, STRIDES, 'SAME',
                        use_cudnn_on_gpu=True, name=layer_name + "/conv")
            activations = act(convs + biases)
        self.layers.append( {
            "activations": activations,
            "weights": weights,
            "biases": biases,
            "type": "conv"
            } )
        return self

    def pool1d(self, size=2, stride=2):
        input_tensor = self.layers[-1]["activations"]
        activations = tf.nn.max_pool(
            input_tensor, [1, 1, size, 1],
            [1, 1, stride, 1], 'SAME')
        self.layers.append( {
            "activations": activations,
            "type": "pool"
            } )
        return self

    def output(self):
        return self.layers[-1]["activations"]
