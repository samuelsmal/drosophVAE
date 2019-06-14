import tensorflow as tf
import tensorflow.keras.layers as tfkl


class PaddedConv1dTransposed(tfkl.Layer):
    """ The most inefficient transpose version. The focus is to get a roughly equal decompression speed.

    Build on https://arxiv.org/pdf/1603.07285.pdf relationship 8, page 22

    Note that this will almost certainly lead to artifacts as the receptive fields overlap.
    But... as we don't really care about it... (at least for now).

    See also https://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, n_filters, kernel_size=2, name=None, activation=tf.nn.leaky_relu, batch_norm=False, padding=None):
        """
        Do NOT set the padding by yourself, the input will be padded in a causal way. If you set padding, this padding will be applied.
        """

        if batch_norm:
            name += '_bn'
        super(PaddedConv1dTransposed, self).__init__(name=name)

        if padding is None:
            padding = 'valid'
            self._padding_overwrite_ = False
        else:
            self._padding_overwrite_ = True

        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.padding = [[0, 0], [1, 1], [0, 0]] # adds only a zero at the end of the time-dimension
        self.conv = tfkl.Conv1D(filters=self.n_filters, kernel_size=self.kernel_size, activation=activation, padding=padding)

        if batch_norm:
            self.batch_norm = tfkl.BatchNormalization()
        else:
            self.batch_norm = None

    def call(self, x):
        if not self._padding_overwrite_:
            x = tf.pad(x, self.padding)
        x = self.conv(x)

        if self.batch_norm:
            x = self.batch_norm(x)

        return x
