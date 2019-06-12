import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from som_vae.layers.padded_conv1d_transposed import PaddedConv1dTransposed
from som_vae.models.drosoph_vae import DrosophVAE
from som_vae.models.utils import make_inference_net

class DrosophVAESkipConv(DrosophVAE):
    """
    About the Deconvolution: https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers

    Based on https://arxiv.org/pdf/1807.04863.pdf to avoid latent space collapse

    """
    def __init__(self, latent_dim, input_shape, batch_size,
                 n_conv_layers=None, n_start_filters=None, dropout_rate_temporal=0.2, loss_weight_reconstruction=1.0, loss_weight_kl=1.0):
        """
        Args:
        -----

        latent_dim              int, dimension of latent space
        input_shape             tuple, total input shape is: [batch_size, *input_shape]
        batch_size              int
        n_layers                int, number of dense layers.
                                output shape of the dense layers is linearly scaled.
        dropout_rate_temporal   float, in [0, 1). dropout rate for temporal blocks (conv layers).
        filters_conv_layer      list[int]. filter sizes for conv layers
        """
        # just a dummy init, we are only interested in defining the two nets.
        super(DrosophVAESkipConv, self).__init__(
            latent_dim=latent_dim,
            input_shape=input_shape,
            batch_size=batch_size,
            loss_weight_reconstruction=loss_weight_reconstruction,
            loss_weight_kl=loss_weight_kl,
            use_wavenet_temporal_layer=False,
            n_layers=1,
        )

        if n_start_filters is None:
            n_start_filters = input_shape[-1]

        if n_conv_layers is None:
            n_conv_layers = np.int(np.ceil((n_start_filters - latent_dim) / 2 + 1))

        self.latent_dim = latent_dim
        self._layer_sizes_inference  = np.linspace(n_start_filters, 2 * latent_dim, num=input_shape[-2] -1, dtype=np.int)
        # pseudo reverse as the inference network goes down to double the latent space, ask Semigh about this
        # the 2 * n_layers is to keep compression speed roughly the same

        # since the layers grow only one time-step per layer:
        self._layer_sizes_generative = np.linspace(latent_dim, n_start_filters, num=input_shape[-2] - 1, dtype=np.int)

        print(self._layer_sizes_inference)
        print(self._layer_sizes_generative)

        # TODO add MaxPooling
        self.inference_net = make_inference_net(tfk.Sequential([*[_convolutional_layer_(idx=i,
                                                                  filters=fs,
                                                                  kernel_size=2,
                                                                  padding='valid',
                                                                  name=f"inf_{i}",
                                                                  activation=tf.nn.leaky_relu)
                                            for i, fs in enumerate(self._layer_sizes_inference)],
                                          tfkl.Flatten(),
                                          tfkl.Dense(2 * self.latent_dim)],
                                         name='inference_net'), input_shape, batch_size)

        self.generative_net = _skip_connection_model_(input_shape=self.latent_dim,
                                                      layer_sizes=self._layer_sizes_generative,
                                                      output_dim=input_shape[-1],
                                                      name='generative_net')


def _convolutional_layer_(idx, **kwargs):
    return tfk.Sequential([tfkl.Conv1D(**{**kwargs, 'name': f"{kwargs['name']}_block_{idx}_conv_0"}),
                           tfkl.Conv1D(**{**kwargs, 'name': f"{kwargs['name']}_block_{idx}_conv_1", 'padding': 'same'}),
                            tfkl.BatchNormalization(name=f"conv_block_{idx}_batch_norm")],
                          name=f"conv_block_{idx}")

class SkipConnectionLayer(tfkl.Layer):
    """
    Taken from https://arxiv.org/pdf/1512.03385.pdf, Deep Residual Learning for Image Recognition

    The batch normalisation prior to a convolution and before activation follows:
    S. Ioffe and C. Szegedy. Batch normalization:  Accelerating deepnetwork training by reducing internal covariate shift. InICML, 2015.

    This class only exists because I can't read too much output. Tensorflow, yeah! (or not)

    Roughtly equivalent (as it was this some time ago) "functional"-style:
    x = tfkl.BatchNormalization()(x)
    x = PaddedConv1dTransposed(n_filters=fs, activation=None, batch_norm=False)(x)
    x_skip = tf.reshape(tfkl.Dense(fs, activation=None)(input_layer), [-1, 1, x.shape[-1]])
    x = x + x_skip
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Activation(tf.nn.leaky_relu)(x)
    """
    def __init__(self, n_filters_weight_layer, activation=tf.nn.leaky_relu, name=None):
        super(SkipConnectionLayer, self).__init__(name=name)
        self.bn_in = tfkl.BatchNormalization()
        self.weight_layer_0 = PaddedConv1dTransposed(n_filters=n_filters_weight_layer, activation=None, batch_norm=False)
        self.act_0 = tfkl.Activation(activation)
        self.weight_layer_1 = PaddedConv1dTransposed(n_filters=n_filters_weight_layer, activation=None, batch_norm=False, padding='same')

        self.bn_sk = tfkl.BatchNormalization()
        self.identity_layer = tfkl.Dense(n_filters_weight_layer, activation=None)
        self.reshape_layer = tfkl.Lambda(lambda x: tf.reshape(x, [-1, 1, n_filters_weight_layer]))
        self.act_1 = tfkl.Activation(activation)

    def call(self, x_and_skipped):
        x, x_skipped = x_and_skipped
        x = self.bn_in(x)
        x_skipped = self.bn_sk(x_skipped) # This is from another paper, which I forgot to bookmark, seemed sensible
        x_skipped = self.identity_layer(x_skipped)

        x = self.weight_layer_0(x)
        x = self.act_0(x)

        x = self.weight_layer_1(x)

        x = x + self.reshape_layer(x_skipped)

        x = self.act_1(x)

        return x


def _skip_connection_model_(input_shape, layer_sizes, output_dim, name):
    """
    """
    input_layer = tfkl.Input(shape=(input_shape,))
    x = tfkl.Lambda(lambda x: tf.reshape(x, [-1, 1, input_shape]))(input_layer)

    for i, fs in enumerate(layer_sizes):
        x = SkipConnectionLayer(fs)([x, input_layer])

    x = tfkl.TimeDistributed(tfkl.Dense(output_dim, activation=None))(x)

    return tfk.Model(inputs=[input_layer], outputs=[x])
