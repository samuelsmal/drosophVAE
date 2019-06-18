import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from drosoph_vae.layers.padded_conv1d_transposed import PaddedConv1dTransposed
from drosoph_vae.models.drosoph_vae import DrosophVAE
from drosoph_vae.models.utils import make_inference_net

class DrosophVAEConv(DrosophVAE):
    """
    About the Deconvolution: https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers
    """
    def __init__(self, latent_dim, input_shape, batch_size, n_start_filters=None, dropout_rate_temporal=0.2, loss_weight_reconstruction=1.0, loss_weight_kl=1.0, with_batch_norm=False):
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
        super(DrosophVAEConv, self).__init__(
            latent_dim=latent_dim,
            input_shape=input_shape,
            batch_size=batch_size,
            loss_weight_reconstruction=loss_weight_reconstruction,
            loss_weight_kl=loss_weight_kl,
            use_wavenet_temporal_layer=False,
            n_layers=1)

        if n_start_filters is None:
            n_start_filters = input_shape[-1]

        self.latent_dim = latent_dim
        self._layer_sizes_inference  = np.linspace(n_start_filters, 2 * latent_dim, num=input_shape[-2] -1, dtype=np.int)
        # pseudo reverse as the inference network goes down to double the latent space, ask Semigh about this
        # the 2 * n_layers is to keep compression speed roughly the same

        # since the layers grow only one time-step per layer:
        self._layer_sizes_generative = np.linspace(latent_dim, n_start_filters, num=input_shape[-2] - 1, dtype=np.int)

        # TODO add MaxPooling
        self.inference_net =make_inference_net(tfk.Sequential([self._convolutional_layer_(idx=i,
                                                                                          filters=fs,
                                                                                          kernel_size=2,
                                                                                          padding='valid',
                                                                                          name=f"inf_{i}",
                                                                                          activation=tf.nn.leaky_relu) 
                                                               for i, fs in enumerate(self._layer_sizes_inference)], name='inference_net'), 
                                               input_shape)
        # This does not work...
        #self.generative_net = tf.keras.Sequential([tfkl.InputLayer(input_shape=(latent_dim,)),
        #                                           tfkl.Lambda(lambda x: tf.reshape(x, [batch_size, 1, latent_dim]), name='gen_reshaping'),
        #                                           *[UpsamplingConv(n_filters=fs, name=f"gen_conv_{i}") for i, fs in enumerate(self._layer_sizes_generative)],
        #                                           tfkl.Dense(input_shape[-1])],
        #                                          name='generative_net')

        self.generative_net = tfk.Sequential([tfkl.InputLayer(input_shape=(self.latent_dim,), name='input_generative_net'),
                                              tfkl.Lambda(lambda x: tf.reshape(x, [-1, 1, self.latent_dim])),
                                              *[PaddedConv1dTransposed(n_filters=fs, batch_norm=with_batch_norm, name=f"gen_{i}") for i, fs
                                                in enumerate(self._layer_sizes_generative)],
                                              tfkl.TimeDistributed(tfkl.Dense(input_shape[-1], activation=tf.nn.leaky_relu, name=f"gen_dense_0")),
                                              tfkl.TimeDistributed(tfkl.Dense(input_shape[-1], activation=tf.nn.leaky_relu, name=f"gen_dense_1")),
                                              tfkl.TimeDistributed(tfkl.Dense(input_shape[-1], activation=None, name=f"gen_dense_2"))],
                                                  name='generative_net')

    def _convolutional_layer_(self, idx, **kwargs):
        return tfk.Sequential([tfkl.Conv1D(**{**kwargs, 'name': f"{kwargs['name']}_block_{idx}_conv_0"}),
                               tfkl.Conv1D(**{**kwargs, 'name': f"{kwargs['name']}_block_{idx}_conv_1", 'padding': 'same'}),
                                tfkl.BatchNormalization(name=f"conv_block_{idx}_batch_norm")],
                              name=f"conv_block_{idx}")
