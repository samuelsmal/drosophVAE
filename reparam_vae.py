# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # VAE using the reparametrization trick

# <codecell>

_DEBUG_          = False # general flag for debug mode
_D_ZERO_DATA_    = True  # basically a debug mode in order to overfit the model on the most simple data
_USING_2D_DATA_  = False 
_USE_EARLY_STOPPING_ = False
_TIME_SERIES_LENGTH_ = 10
_BATCH_SIZE_ = 100

# <codecell>

# to generate gifs
#!pip install imageio

# <markdowncell>

# ## Import TensorFlow and enable Eager execution

# <codecell>

# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf
tfe = tf.contrib.eager
tf.enable_eager_execution()

tfk = tf.keras
tfkl = tf.keras.layers

from tcn import TCN

import warnings
import os
import time
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from IPython import display
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from som_vae.helpers.misc import extract_args, chunks, foldl
from som_vae.helpers.jupyter import fix_layout, display_video
from som_vae.settings import config, skeleton
from som_vae.helpers import video, plots, misc, jupyter
from som_vae import preprocessing
from som_vae.helpers.logging import enable_logging
from som_vae.helpers.tensorflow import _TF_DEFAULT_SESSION_CONFIG_

# <codecell>

jupyter.fix_layout()

# <codecell>

from som_vae.helpers.tensorflow import _TF_DEFAULT_SESSION_CONFIG_
sess = tf.InteractiveSession(config=_TF_DEFAULT_SESSION_CONFIG_)
tf.keras.backend.set_session(sess)

# <markdowncell>

# ## Load the MNIST dataset
# 
# Each MNIST image is originally a vector of 784 integers, each of which is between 0-255 and represents the intensity of a pixel. We model each pixel with a Bernoulli distribution in our model, and we statically binarize the dataset.
# 
# Remember they binarize the data previous.

# <codecell>

from som_vae import settings
from som_vae import preprocessing

joint_positions, normalisation_factors = preprocessing.get_data_and_normalization(settings.data.EXPERIMENTS)

frames_idx_with_labels = preprocessing.get_frames_with_idx_and_labels(settings.data.LABELLED_DATA)[:len(joint_positions)]

#frames_of_interest = frames_idx_with_labels.label.isin([settings.data._BehaviorLabel_.GROOM_ANT, settings.data._BehaviorLabel_.WALK_FORW, settings.data._BehaviorLabel_.REST])
frames_of_interest = ~frames_idx_with_labels.label.isin([settings.data._BehaviorLabel_.REST])

joint_positions = joint_positions[frames_of_interest]
frames_idx_with_labels = frames_idx_with_labels[frames_of_interest]

# <codecell>

def to_time_series(data, sequence_length):
    for i in range(len(data)):
        if i + sequence_length <= len(data):
            yield data[i:i+sequence_length]

# <codecell>

df_timed = list(to_time_series(reshaped_joint_position, 10))

# <codecell>

np.array(df_timed).shape

# <codecell>

scaler = StandardScaler()

# flatten the data
if _USING_2D_DATA_:
    reshaped_joint_position = joint_positions[:,:,:2].reshape(joint_positions.shape[0], -1).astype(np.float32)
else:
    warnings.warn('this is not proper, fix the standardisation here')
    reshaped_joint_position = scaler.fit_transform(joint_positions[:,:,:2].reshape(joint_positions.shape[0], -1).astype(np.float32))
    reshaped_joint_position = np.array(list(to_time_series(reshaped_joint_position, sequence_length=_TIME_SERIES_LENGTH_)))

if _DEBUG_ and _D_ZERO_DATA_:
    reshaped_joint_position = np.zeros_like(reshaped_joint_position)

# scaling the data to be in [0, 1]
# this is due to the sigmoid activation function in the reconstruction
#scaler = MinMaxScaler()

print(f"total number of input data:{reshaped_joint_position.shape}")

n_of_data_points = int(reshaped_joint_position.shape[0] * 0.7)

if _USING_2D_DATA_:
    data_train = scaler.fit_transform(reshaped_joint_position[:n_of_data_points])
    data_test = scaler.transform(reshaped_joint_position[n_of_data_points:])
    display.display(pd.DataFrame(data_train).describe())
else:
    data_train = reshaped_joint_position[:n_of_data_points]
    data_test = reshaped_joint_position[n_of_data_points:]
    display.display(pd.DataFrame(reshaped_joint_position[:, -1, :]).describe())

# <markdowncell>

# ## Use *tf.data* to create batches and shuffle the dataset

# <codecell>

_TRAIN_BUFFER_SIZE_ = len(data_train) 

_TEST_BUFFER_SIZE_ = len(data_test) 

train_dataset = tf.data.Dataset.from_tensor_slices(data_train).shuffle(_TRAIN_BUFFER_SIZE_).batch(_BATCH_SIZE_)
test_dataset = tf.data.Dataset.from_tensor_slices(data_test).shuffle(_TEST_BUFFER_SIZE_).batch(_BATCH_SIZE_)

# <codecell>

data_train.shape

# <markdowncell>

# ## Wire up the generative and inference network with *tf.keras.Sequential*
# 
# ### Generative Network
# This defines the generative model which takes a latent encoding as input, and outputs the parameters for a conditional distribution of the observation, i.e. $p(x|z)$. Additionally, we use a unit Gaussian prior $p(z)$ for the latent variable.
# 
# ### Inference Network
# This defines an approximate posterior distribution $q(z|x)$, which takes as input an observation and outputs a set of parameters for the conditional distribution of the latent representation. In this example, we simply model this distribution as a diagonal Gaussian. In this case, the inference network outputs the mean and log-variance parameters of a factorized Gaussian (log-variance instead of the variance directly is for numerical stability).
# 
# ### Reparameterization Trick
# During optimization, we can sample from $q(z|x)$ by first sampling from a unit Gaussian, and then multiplying by the standard deviation and adding the mean. This ensures the gradients could pass through the sample to the inference network parameters.
# 
# ### Network architecture
# For the inference network, we use two convolutional layers followed by a fully-connected layer. In the generative network, we mirror this architecture by using a fully-connected layer followed by three convolution transpose layers (a.k.a. deconvolutional layers in some contexts). Note, it's common practice to avoid using batch normalization when training VAEs, since the additional stochasticity due to using mini-batches may aggravate instability on top of the stochasticity from sampling.

# <codecell>

#class TemporalBlock(tf.keras.Model):
#    def __init__(self, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0.0): 
#        super(TemporalBlock, self).__init__()
#        # TODO check if this is correct
#        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
#        assert padding in ['causal', 'same']
#
#        # block1
#        self.conv1  = tfkl.Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
#        self.batch1 = tfkl.BatchNormalization(axis=-1)
#        self.ac1    = tfkl.Activation('relu')
#        self.drop1  = tfkl.Dropout(rate=dropout_rate)
#        
#        # block2
#        self.conv2  = tfkl.Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
#        self.batch2 = tfkl.BatchNormalization(axis=-1)        
#        self.ac2    = tfkl.Activation('relu')
#        self.drop2  = tfkl.Dropout(rate=dropout_rate)
#
#        self.downsample = tfkl.Conv1D(filters=nb_filters, kernel_size=1, padding='same', kernel_initializer=init)
#        self.ac3    = tfkl.Activation('relu')
#
#
#    def call(self, x, training):
#        prev_x = x
#        x = self.conv1(x)
#        x = self.batch1(x)
#        x = self.ac1(x)
#        x = self.drop1(x) if training else x
#
#        x = self.conv2(x)
#        x = self.batch2(x)
#        x = self.ac2(x)
#        x = self.drop2(x) if training else x
#
#        if prev_x.shape[-1] != x.shape[-1]:    # match the dimention
#            prev_x = self.downsample(prev_x)
#        assert prev_x.shape == x.shape
#
#        return self.ac3(prev_x + x)            # skip connection
#
#
#
#class TemporalConvNet(tf.keras.Model):
#    def __init__(self, num_channels, kernel_size=2, dropout=0.2):
#        # num_channels is a list contains hidden sizes of Conv1D
#        super(TemporalConvNet, self).__init__()
#        assert isinstance(num_channels, list)
#
#        model = tf.keras.Sequential()
#
#        # The model contains "num_levels" TemporalBlock
#        num_levels = len(num_channels)
#        for i in range(num_levels):
#            dilation_rate = 2 ** i                  # exponential growth
#            model.add(TemporalBlock(dilation_rate=dilation_rate, nb_filters=num_channels[i], kernel_size=kernel_size, padding='causal', dropout_rate=dropout))
#        self.network = model
#
#    def call(self, x, training):
#        return self.network(x, training=training)

# <codecell>

# V3
from typing import List, Tuple

import keras.backend as K
import keras.layers
from keras import optimizers
from keras.engine.topology import Layer
from keras.layers import Activation, Lambda
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Convolution1D, Dense
from keras.models import Input, Model


def residual_block(x, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0):
    # type: (Layer, int, int, int, str, float) -> Tuple[Layer, Layer]
    """Defines the residual block for the WaveNet TCN

    Args:
        x: The previous layer in the model
        dilation_rate: The dilation power of 2 we are using for this residual block
        nb_filters: The number of convolutional filters to use in this block
        kernel_size: The size of the convolutional kernel
        padding: The padding used in the convolutional layers, 'same' or 'causal'.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.

    Returns:
        A tuple where the first element is the residual model layer, and the second
        is the skip connection.
    """
    prev_x = x
    for k in range(2):
        x = Conv1D(filters=nb_filters,
                   kernel_size=kernel_size,
                   dilation_rate=dilation_rate,
                   padding=padding)(x)
        # x = BatchNormalization()(x)  # TODO should be WeightNorm here.
        x = Activation('relu')(x)
        x = SpatialDropout1D(rate=dropout_rate)(x)

    # 1x1 conv to match the shapes (channel dimension).
    prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
    res_x = keras.layers.add([prev_x, x])
    return res_x, x


def process_dilations(dilations):
    def is_power_of_two(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if all([is_power_of_two(i) for i in dilations]):
        return dilations

    else:
        new_dilations = [2 ** i for i in dilations]
        # print(f'Updated dilations from {dilations} to {new_dilations} because of backwards compatibility.')
        return new_dilations


class TCN(tf.layers.Layer):
    """Creates a TCN layer.

        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).

        Args:
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            name: Name of the model. Useful when having multiple TCN.

        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=[1, 2, 4, 8, 16, 32],
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 name='tcn'):
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")


    def __call__(self, inputs):
        x = inputs
        # 1D FCN.
        x = Convolution1D(self.nb_filters, 1, padding=self.padding)(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for d in self.dilations:
                x, skip_out = residual_block(x,
                                             dilation_rate=d,
                                             nb_filters=self.nb_filters,
                                             kernel_size=self.kernel_size,
                                             padding=self.padding,
                                             dropout_rate=self.dropout_rate)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = keras.layers.add(skip_connections)
        if not self.return_sequences:
            x = Lambda(lambda tt: tt[:, -1, :])(x)
        return x

# <codecell>

# Source: https://medium.com/the-artificial-impostor/notes-understanding-tensorflow-part-3-7f6633fcc7c7

# Redefining CausalConv1D to simplify its return values
class CausalConv1D(tf.layers.Conv1D):
    def __init__(self, filters,
               kernel_size,
               strides=1,
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
        )
        
       
    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding, mode="CONSTANT")
        return super(CausalConv1D, self).call(inputs)

class TemporalBlock(tf.layers.Layer):
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate, dropout=0.2, 
                 trainable=True, name=None, dtype=None, 
                 activity_regularizer=None, **kwargs):
        super(TemporalBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )        
        self.dropout = dropout
        self.n_outputs = n_outputs
        self.conv1 = CausalConv1D(n_outputs, kernel_size, strides=strides, dilation_rate=dilation_rate, activation=tf.nn.relu, name="conv1")
        self.conv2 = CausalConv1D(n_outputs, kernel_size, strides=strides, dilation_rate=dilation_rate, activation=tf.nn.relu, name="conv2")
        self.down_sample = None

    
    def build(self, input_shape):
        channel_dim = -1
        # from https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
        # noise_shape: 1D integer tensor representing the shape of the binary dropout mask that will be multiplied with the input. 
        # For instance, if your inputs have shape (batch_size, timesteps, features) and you want the dropout mask to be the same 
        # for all timesteps, you can use noise_shape=(batch_size, 1, features). 
        # TODO play around here, maybe also https://www.tensorflow.org/api_docs/python/tf/keras/layers/SpatialDropout1D
        #      see paper: https://arxiv.org/abs/1411.4280
        #self.dropout1 = tf.layers.Dropout(self.dropout, noise_shape=[tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        #self.dropout2 = tf.layers.Dropout(self.dropout, noise_shape=[tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
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

class TemporalConvNet(tf.layers.Layer):
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
                                     strides=1,
                                     dilation_rate=2 ** i,
                                     dropout=dropout,
                                     name=f"temporal_block_{i}") 
                       for i, n_out_channels in enumerate(num_channels)]
    
    def call(self, inputs, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)
        return outputs

# <codecell>

def _receptive_field_size_temporal_conv_net_(kernel_size, n_layers):
    return (1 + 2 * (kernel_size - 1) * (2 ** n_layers - 1))

for k in range(2, 5):
    plt.plot([_receptive_field_size_temporal_conv_net_(kernel_size=k, n_layers=n) for n in range(10)], label=f"kernel size: {k}")
plt.xlabel('number of layers')
plt.ylabel('receptive field size')
plt.legend()

# <codecell>

def dense_layers(sizes):
    return tfk.Sequential([tfkl.Dense(size, activation=tf.nn.leaky_relu if i + 1 != len(sizes) else None) for i, size in enumerate(sizes)])


class R_VAE(tf.keras.Model):
    def __init__(self, latent_dim, input_shape, batch_size):
        super(R_VAE, self).__init__()
        self.latent_dim = latent_dim
        self._input_shape = input_shape
        self._batch_size = batch_size
        self._layer_sizes = np.linspace(input_shape[0], 2 * latent_dim, 3)
        
        if len(input_shape) == 1:
            self.inference_net = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_shape),
                                                      dense_layers(self._layer_sizes)],
                                                     name='inference_net')

            self.generative_net = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                                                       tf.keras.layers.Dense(input_shape[0])],
                                                      name='generative_net')
            self.temporal_conv_net = None
        elif len(input_shape) == 2:
            self._layer_sizes = np.linspace(input_shape[-1], 2 * latent_dim, 3)
            self.inference_net = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_shape[-1]),
                                                      dense_layers(self._layer_sizes)],
                                                     name='inference_net')

            self.generative_net = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                                                       tf.keras.layers.Dense(input_shape[-1])],
                                                      name='generative_net')
            
            print("n channels", [input_shape[-1]] * 3)
            self.temporal_conv_net = TemporalConvNet(num_channels=[input_shape[-1]] * 3)
        else:
            raise ValueError(f"Input shape is not good, got: {input_shape}")
    
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random_normal(shape=(self._batch_size, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
  
    def encode(self, x, training=False):
        if self.temporal_conv_net:
            # taking only the last entry
            x_tmp = tfkl.Lambda(lambda x: x[:, -1, :])(self.temporal_conv_net(x, training=training))
            mean, logvar = tf.split(self.inference_net(x_tmp), num_or_size_splits=2, axis=1)
        else:
            mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar
  
    def reparameterize(self, mean, logvar):
        eps = tf.random_normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
  
    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
  
        return logits

    def predict(self, x):
        mean, logvar = self.encode(x)
        z = model.reparameterize(mean, logvar)
        return model.decode(z, apply_sigmoid=True)

# <markdowncell>

# ## Define the loss function and the optimizer
# 
# VAEs train by maximizing the evidence lower bound (ELBO) on the marginal log-likelihood:
# 
# $$\log p(x) \ge \text{ELBO} = \mathbb{E}_{q(z|x)}\left[\log \frac{p(x, z)}{q(z|x)}\right].$$
# 
# In practice, we optimize the single sample Monte Carlo estimate of this expectation:
# 
# $$\log p(x| z) + \log p(z) - \log q(z|x),$$
# where $z$ is sampled from $q(z|x)$.
# 
# **Note**: we could also analytically compute the KL term, but here we incorporate all three terms in the Monte Carlo estimator for simplicity.

# <codecell>

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
  
    if len(x.shape) == 1:
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    else:
        # Note, the model is trained to reconstruct only the last, most current time step (by taking the last entry in the timeseries)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x[:, -1, :])
    #logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1]) # down to [batch, loss]
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model, x): 
    with tf.GradientTape() as tape: 
        loss = compute_loss(model, x) 
        return tape.gradient(loss, model.trainable_variables), loss

optimizer = tf.train.AdamOptimizer(1e-4)
def apply_gradients(optimizer, gradients, variables, global_step=None):
    optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

# <codecell>

epochs = 400
latent_dim = 16 

num_examples_to_generate = 16

tf.reset_default_graph()

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random_normal(shape=[num_examples_to_generate, latent_dim])
model = R_VAE(latent_dim, input_shape=data_train.shape[1:], batch_size=BATCH_SIZE)

# <codecell>

losses = []
print('will train for {epochs} epochs')
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        gradients, loss = compute_gradients(model, train_x)
        apply_gradients(optimizer, gradients, model.trainable_variables)
    end_time = time.time()

    if epoch % 1 == 0:
        loss = tfe.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        losses += [elbo]
        #display.clear_output(wait=False)
        if epoch % 10 == 0:
            print('Epoch: {:0>3}, Test set ELBO: {:0.3f}, '
                  'time elapse for current epoch {:0.4f}'.format(epoch,
                                                            elbo,
                                                            end_time - start_time))
            #generate_and_save_images(model, epoch, random_vector_for_generation)

# <codecell>

plt.plot(losses)
plt.xlabel('epochs')
plt.ylabel('ELBO')

# <codecell>

def _reverse_to_original_shape_(pos_data, input_shape=None):
    if input_shape is None:
        input_shape = (-1, config.NB_DIMS)
        
    return scaler.inverse_transform(pos_data).reshape(pos_data.shape[0], *(input_shape))

# <codecell>

pred_train = model.predict(data_train)
pred_train_rev = _reverse_to_original_shape_(model.predict(data_train).numpy())

plots.plot_comparing_joint_position_with_reconstructed(_reverse_to_original_shape_(data_train), pred_train_rev, validation_cut_off=data_train.shape[0])

# <codecell>


