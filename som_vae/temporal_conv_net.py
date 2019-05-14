import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers

# Source: https://medium.com/the-artificial-impostor/notes-understanding-tensorflow-part-3-7f6633fcc7c7

class TemporalBlock(tfkl.Layer):
    def __init__(self, n_outputs, kernel_size, dilation_rate, dropout=0.2, trainable=True, name=None, dtype=None, activity_regularizer=None, **kwargs):
        """
        In the dilated convolution, the kernel only touches the signal at every lth entry
        See https://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/
        """
        super(TemporalBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs)
        self.dropout = dropout
        self.n_outputs = n_outputs
        causal_conv_args = {"padding": "causal",
                            "dilation_rate": dilation_rate,
                            "activation": tf.nn.relu}
        self.conv1 = tfkl.Conv1D(n_outputs, kernel_size, **causal_conv_args, name="conv1")
        self.conv2 = tfkl.Conv1D(n_outputs, kernel_size, **causal_conv_args, name="conv2")
        self.down_sample = None


    def build(self, input_shape):
        channel_dim = -1
        # from https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
        # noise_shape: 1D integer tensor representing the shape of the binary dropout mask that will be multiplied with the input.
        # For instance, if your inputs have shape (batch_size, timesteps, features) and you want the dropout mask to be the same
        # for all timesteps, you can use noise_shape=(batch_size, 1, features).
        # SpatialDropout1D proved to be much better. Original paper: https://arxiv.org/abs/1411.4280
        self.dropout1 = tfkl.SpatialDropout1D(self.dropout)
        self.dropout2 = tfkl.SpatialDropout1D(self.dropout)
        if input_shape[channel_dim] != self.n_outputs:
            # TODO why not a conv1d layer?
            # self.down_sample = tf.layers.Conv1D(
            #     self.n_outputs, kernel_size=1,
            #     activation=None, data_format="channels_last", padding="valid")
            self.down_sample = tf.layers.Dense(self.n_outputs, activation=None)
        self.built = True

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout2(x, training=training)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        return tf.nn.relu(x + inputs)

class TemporalConvNet(tfkl.Layer):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2,
                 trainable=True, name=None, dtype=None,
                 activity_regularizer=None, **kwargs):
        super(TemporalConvNet, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )

        self.layers = [TemporalBlock(n_out_channels,
                                     kernel_size,
                                     dilation_rate=2 ** i,
                                     dropout=dropout,
                                     name=f"temporal_block_{i}")
                       for i, n_out_channels in enumerate(num_channels)]

    def call(self, inputs, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)
        return outputs
