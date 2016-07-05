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
        input_dim = reduce(lambda p,f: p*f, input_tensor.get_shape()[1:-1].as_list(), 1)
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
    
    def conv1d(self, filters=12, size=5, act=tf.nn.relu, stride=1):
        
        return self.conv2d(filters=filters,size=[1,size], act=act, stride=[1,1,stride,1])

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

    def rec_conv1d(self, filters=12, size=5, unrollings=3, input_act=tf.nn.relu, rec_act=tf.nn.relu):
        """
        Reusable code for making a convolutional neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/2B_004.pdf
        """
        input_tensor = self.layers[-1]["activations"]
        layer_name = "rec_conv1d" + str(len([l for l in self.layers
            if l["type"]=="rec_conv1d"]))
        input_dim = reduce(lambda p,f: p*f, input_tensor.get_shape()[1:-1].as_list(), 1)
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
    
    def lstm(self, size=100, layers=1, keep_prob=1, forget_bias=1.0):
        """
        size: LSTM layer width
        layers: Number of layers
        forget_bias: initialize forget bias, defaults to 1
        """
        layer_name = "lstm" + str(len([l for l in self.layers
            if l["type"]=="lstm"]))
        
        with tf.name_scope(layer_name):
            input_tensor = self.layers[-1]["activations"]
            num_batches, _, num_steps, channels = input_tensor.get_shape()
            num_batches = int(num_batches)
            num_steps = int(num_steps)
            channels = int(channels)

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias, state_is_tuple=True)

            if keep_prob > 0 and keep_prob < 1:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, keep_prob)

            if layers > 1:
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * layers, state_is_tuple=True)

            #input_tensor = tf.reshape(input_tensor, [1, num_steps, channels])
            #input_tensor = tf.transpose(input_tensor, [1, 0, 2])  # permute num_steps and num_batches
            input_tensor = tf.squeeze(input_tensor, [1]) # Remove this when input is in better shape (e.g (250,64,32))
            inputs = [tf.squeeze(i, [1]) for i in tf.split(1, num_steps, input_tensor)]
            activations, state = tf.nn.rnn(lstm_cell, inputs, dtype=tf.float32)#initial_state=initial_state)

            self.layers.append({
                "activations": activations[-1],
                "type": "lstm"
                })
            return self
 
    def bn(self, act=tf.nn.relu):
        """Batch normalization.
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
