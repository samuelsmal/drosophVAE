import warnings
import numpy as np

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from som_vae.helpers.misc import if_last
from som_vae.models.utils import make_inference_net
from som_vae.layers.temporal_block import TemporalBlock

# DrosophVAE base class
# build using:
#   - https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/generative_examples/cvae.ipynb
#   - https://www.kaggle.com/hone5com/fraud-detection-with-variational-autoencoder

def dense_layers(sizes, activation_fn=tf.nn.leaky_relu, name_prefix=None):
    # no activation in the last layer
    # both models inference and generative should be free (super important for the decoder)
    # the encoder could be fixed, but we want the "mean" to be represent any value,
    # a SoftPlus activation is applied to the "variance" in the `DrosophVAE.encode` method.
    return [tfkl.Dense(size, activation=None if is_last else activation_fn, name=f"{name_prefix}_dense_{idx}") for idx, is_last, size in if_last(sizes)]

def temporal_layers(filter_sizes, kernel_size=2, dropout=0.2):
    return [TemporalBlock(filter_size, kernel_size, dilation_rate=2 ** i, dropout=dropout, name=f"temporal_block_{i}") for i, filter_size in enumerate(filter_sizes)]

class DrosophVAE(tfk.Model):
    def __init__(self, latent_dim, input_shape, batch_size,
                 n_layers=3, dropout_rate_temporal=0.2,
                 loss_weight_reconstruction=1.0, loss_weight_kl=1.0,
                 filters_conv_layer=None, conv_layer_kernel_size=2,
                 use_wavenet_temporal_layer=True):
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
        super(DrosophVAE, self).__init__()
        self.latent_dim = latent_dim
        self._input_shape = input_shape
        self._batch_size = batch_size
        self._loss_weight_reconstruction = loss_weight_reconstruction
        self._loss_weight_kl = loss_weight_kl
        self._layer_sizes_inference  = np.linspace(input_shape[-1], 2 * latent_dim, n_layers).astype(np.int)
        # pseudo reverse as the inference network goes down to double the latent space, ask Semigh about this
        # the 2 * n_layers is to keep compression speed roughly the same
        self._layer_sizes_generative = np.linspace(latent_dim, input_shape[-1], 2 * n_layers).astype(np.int).tolist()
        self._conv_layer_kernel_size = conv_layer_kernel_size

        if use_wavenet_temporal_layer:
            # Remember that we do diluted convolutions -> The filter size can stay ~ constant. TODO discuss with Semigh
            if filters_conv_layer is None:
                # TODO this is probably not correct
                self.filters_conv_layer = [input_shape[-1]] * 3
            else:
                self.filters_conv_layer = filters_conv_layer

            self.temporal_conv_net = tfk.Sequential([tfkl.InputLayer(input_shape=input_shape, name='input_temporal_conv_net'),
                                                     *temporal_layers(kernel_size=self._conv_layer_kernel_size,
                                                                      filter_sizes=self.filters_conv_layer,
                                                                      dropout=dropout_rate_temporal)],
                                                     name='temporal_conv_net')

            inference_input_shape = input_shape
            generative_input_shape = list(input_shape[:1]) + [latent_dim]
        else:
            self.temporal_conv_net = None
            inference_input_shape = input_shape[-1]
            generative_input_shape = (latent_dim, )

        self.inference_net = make_inference_net(tfk.Sequential([tfkl.InputLayer(input_shape=inference_input_shape, name='input_inference_net'),
                                                 *dense_layers(self._layer_sizes_inference, name_prefix='inf')],
                                                 name='inference_net'), inference_input_shape, batch_size)

        self.generative_net = tfk.Sequential([tfkl.InputLayer(input_shape=generative_input_shape, name='input_generative_net'),
                                                  *dense_layers(self._layer_sizes_generative, name_prefix='gen')],
                                                  name='generative_net')

    def sample(self, eps=None):
        if eps is None:
            if self._loss_weight_kl == 0.0:
                warnings.warn('KL loss is 0.0. The latent space is not properly trained')
            # The KL-loss is calculated against a normal distribution,
            # thus it should resemble one and thus sampling should make sense.
            #eps = tf.random_normal(shape=(self._batch_size, self.latent_dim))
            eps = tf.random_normal(shape=[self._batch_size] + list(self.generative_net.input_shape[1:]))
        return self.decode(eps, apply_sigmoid=False)

    def encode(self, x, training=False):
        if self.temporal_conv_net:
            # TODO combine them into one? max pooling or something
            #x_tmp = tfkl.Lambda(lambda x: x[:, -1, :])(self.temporal_conv_net(x, training=training))
            mean, var = self.inference_net(self.temporal_conv_net(x, training=training))
            #mean, var = tf.split(self.inference_net(self.temporal_conv_net(x, training=training)),
            #                        num_or_size_splits=2,
            #                        axis=-1)
        else:
            mean, var = self.inference_net(x, training=training)
            #mean, var = tf.split(self.inference_net(x),
            #                        num_or_size_splits=2,
            #                        axis=1)

        # the variance should be in [0, inf)
        return mean, var

    def reparameterize(self, mean, var):
        # TODO check: the params should be correct? check original paper
        # I know of the error. the proposed function does not work...
        eps = tf.random_normal(shape=mean.shape)
        #return eps * tf.exp(logvar * .5) + mean
        # this is the truest form to the original paper https://arxiv.org/pdf/1312.6114v10.pdf
        return eps * var + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    def predict(self, x):
        # https://github.com/LynnHo/VAE-Tensorflow/blob/master/train.py
        # epsilon = tf.random_normal(tf.shape(z_mu))
        # if is_training:
        #     z = z_mu + tf.exp(0.5 * z_log_sigma_sq) * epsilon
        # else:
        #     z = z_mu
        mean, logvar = self.encode(x)
        z = model.reparameterize(mean, logvar)
        return model.decode(z, apply_sigmoid=True)

    def call(self, x, training=False, apply_sigmoid=False):
        if apply_sigmoid:
            raise ValueError('no longer supported')
        return self.decode(self.reparameterize(*self.encode(x, training=training)),
                           apply_sigmoid=apply_sigmoid)

    def _config_(self):
        return {
            "latent_dim": self.latent_dim,
            "input_shape": self._input_shape,
            "batch_size": self._batch_size,
            "layer_sizes_inference": self._layer_sizes_inference,
            "layer_sizes_generative": self._layer_sizes_generative,
            "loss_weight_reconstruction": self._loss_weight_reconstruction,
            "loss_weight_kl": self._loss_weight_kl,
            "model_impl": self._name
        }
