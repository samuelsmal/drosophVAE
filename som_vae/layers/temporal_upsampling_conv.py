import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

class TemporalUpsamplingConv(tfkl.Layer):
    """
    This layer requires good fine tuning of parameters. Use PaddedConv1dTransposed if you want an easier layer.

    For a general nice description of convolutions:
        https://arxiv.org/pdf/1603.07285.pdf

        A guide to convolution arithmetic for deeplearning
        Vincent Dumoulin and Francesco Visin

    For the artifacts:
        Conditional generative adversarial nets for convolutional face generation
        J. Gauthier.
        Class Project for Stanford CS231N: Convolutional Neural Networks for Visual Recognition, Winter semester, Vol 2014. 2014.
        http://www.foldl.me/uploads/papers/tr-cgans.pdf
    """
    def __init__(self, conv_n_filters, upsampling_size=3, conv_kernel_size=2, conv_padding='valid', conv_strides=2, name=None):
        super(TemporalUpsamplingConv, self).__init__(name=name)

        if conv_kernel_size % conv_strides != 0:
            warnings.warn(f"Using a kernel size not divisable by the stride will lead to artifacts.:"
                          f" Given kernel_size: {conv_kernel_size}"
                          f" stride: {conv_strides}")

        self.upsampling_size = upsampling_size
        self.conv_n_filters = conv_n_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = conv_padding
        self.conv_strides = conv_strides

        # upscale with 3 so that we can again apply `valid` padding and "reverse" the encoder
        self.upsampling = tfkl.UpSampling1D(size=self.upsampling_size,
                                            name=f"{name}_upsampling")
        # TODO maybe add some fancy flipping of the input, right now it cuts again from the "start", ideally it should append there...
        self.conv = tfkl.Conv1D(filters=self.conv_n_filters,
                                kernel_size=self.conv_kernel_size,
                                padding=self.conv_padding,
                                strides=self.conv_strides,
                                name=f"{name}_conv")

    def call(self, x):
        return self.conv(self.upsampling(x))
