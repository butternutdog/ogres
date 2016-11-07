"""
Recurrent networks
==================

Thin wrappers around tensorflow's recurrent layers.
"""

from .. import layer, tf, weight_variable, bias_variable, variable_summaries

@layer
def lstm(self, size=100, layers=1, keep_prob=1, forget_bias=1.0):
    """
    size: LSTM layer width
    layers: Number of layers
    forget_bias: initialize forget bias, defaults to 1, see http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    For a great overview of LSTMs, see http://colah.github.io/posts/2015-08-Understanding-LSTMs/
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
