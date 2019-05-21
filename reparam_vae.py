# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # VAE using the reparametrization trick

# <codecell>

run_config = {
    'debug': False,   # general flag for debug mode
    'd_zero_data': True,    # basically a debug mode in order to overfit the model on the most simple data
    'd_no_compression': True,  # if true, the latent_space will be the same dimension as the input. basically the model needs to learn the identity function
    'use_all_experiments': False,
    'use_time_series': True,  # TODO make the time series also work on 2d data 
    'use_2d_data': True,
    'time_series_length': 10,
    'batch_size': 100
}

# <codecell>

# to generate gifs
#!pip install imageio

# <markdowncell>

# ## Import TensorFlow and enable Eager execution

# <codecell>

# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf
tfe = tf.contrib.eager
tfc = tf.contrib
tf.enable_eager_execution()

tfk = tf.keras
tfkl = tf.keras.layers

import json
from collections import namedtuple
import warnings
import os
import time
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
import imageio
from IPython import display
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE

from som_vae import settings
from som_vae import preprocessing
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

# ## Loading of 2d data

# <codecell>

!ls $config.__EXPERIMENT_ROOT__

# <codecell>

from functional import seq
from pathlib import Path
from functools import reduce

# <codecell>

from som_vae.settings import data as SD
from importlib import reload

# <codecell>

def experiments_from_root(root=config.__EXPERIMENT_ROOT__):
    return seq(Path(root).iterdir()).flat_map(lambda p: (c for c in p.iterdir() if c.is_dir()))\
                                    .flat_map(lambda p: (c for c in p.iterdir() if c.is_dir()))\
                                    .map(lambda p: reduce(lambda acc, el: ([*acc[0], acc[1].stem], acc[1].parent), range(3), ([], p)))\
                                    .map(lambda pair: list(reversed(pair[0])))\
                                    .map(lambda ls: SD.Experiment._make([*ls, SD._key_from_list_(ls)]))\
                                    .to_list()

# <codecell>

if run_config['use_all_experiments']:
    raise NotImplementedError
    all_experiments = experiments_from_root()
    print(len(all_experiments))

    pos_data = preprocessing.get_data_and_normalization(all_experiments, per_experiment=True)

    norm_pos_data, norm_pos_data_params = zip(*[preprocessing.normalize(p) for p in pos_data])
    experiment_lengths = [len(p) for p in norm_pos_data] # for applying the right normalization factors
    norm_pos_data = np.vstack(norm_pos_data)

    print(f"in total we have {len(all_experiments)} experiments, but only {len(experiment_lengths)} are usable right now")

    norm_pos_data_embedded = TSNE(n_components=2, random_state=42).fit_transform(norm_pos_data[:, :, :2].reshape(norm_pos_data.shape[0], -1))

    #_cs = sns.color_palette(n_colors=len(seen_labels))
    #
    #fig = plt.figure(figsize=(10, 10))
    #_all_frames_ = pd.concat((training_frames, testing_frames))
    #
    #behaviour_colours = dict(zip(seen_labels, _cs))
    #
    #for l, c in behaviour_colours.items():
    #    _d = X_embedded[_all_frames_['label'] == l]
    #    # c=[c] since matplotlib asks for it
    #    plt.scatter(_d[:, 0], _d[:,1], c=[c], label=l.name, marker='.')
    #    
    #plt.legend()
    #plt.title('simple t-SNE on latent space')
    #fig.savefig(f"../neural_clustering_data/figures/{som_vae_config['ex_name']}_tsne.png")

    _cs = sns.color_palette(n_colors=len(experiment_lengths))


    used_until = 0
    for i, l in enumerate(experiment_lengths):
        plt.scatter(norm_pos_data_embedded[used_until:used_until+l, 0], norm_pos_data_embedded[used_until:used_until+l, 1], c=[_cs[i]])
        used_until += l

else:
    joint_positions, normalisation_factors = preprocessing.get_data_and_normalization(settings.data.EXPERIMENTS)

    frames_idx_with_labels = preprocessing.get_frames_with_idx_and_labels(settings.data.LABELLED_DATA)[:len(joint_positions)]

    images_paths_for_experiments = settings.data.EXPERIMENTS.map(lambda x: (x, config.positional_data(x)))\
                                           .flat_map(lambda x: [(x[0], config.get_path_for_image(x[0], i)) for i in range(x[1].shape[1])])\
                                           .to_list()
    
    frames_of_interest = ~frames_idx_with_labels.label.isin([settings.data._BehaviorLabel_.REST])
    
    # TODO form a wrapper around them
    joint_positions = joint_positions[frames_of_interest]
    frames_idx_with_labels = frames_idx_with_labels[frames_of_interest]
    images_paths_for_experiments =  np.array(images_paths_for_experiments)[frames_of_interest].tolist()

# <markdowncell>

# # preprocessing

# <codecell>

def to_time_series(data, sequence_length):
    for i in range(len(data)):
        if i + sequence_length <= len(data):
            yield data[i:i+sequence_length]

# <codecell>

#scaler = StandardScaler()
scaler = MinMaxScaler()

# flatten the data
if run_config['use_time_series']:
    # TODO the scaling should be learned on the training data only
    warnings.warn('this is not proper, fix the standardisation here')
    reshaped_joint_position = scaler.fit_transform(joint_positions[:,:,:2].reshape(joint_positions.shape[0], -1).astype(np.float32))
    reshaped_joint_position = np.array(list(to_time_series(reshaped_joint_position, sequence_length=run_config['time_series_length'])))
else:
    reshaped_joint_position = joint_positions[:,:,:2].reshape(joint_positions.shape[0], -1).astype(np.float32)

if run_config['debug'] and run_config['d_zero_data']:
    reshaped_joint_position = scaler.fit_transform(np.zeros_like(reshaped_joint_position))

# scaling the data to be in [0, 1]
# this is due to the sigmoid activation function in the reconstruction
#scaler = MinMaxScaler()

print(f"total number of input data:{reshaped_joint_position.shape}")

n_of_data_points = int(reshaped_joint_position.shape[0] * 0.7)

if run_config['use_time_series']:
    data_train = reshaped_joint_position[:n_of_data_points]
    data_test = reshaped_joint_position[n_of_data_points:]
    display.display(pd.DataFrame(reshaped_joint_position[:, -1, :]).describe())
else:
    data_train = scaler.fit_transform(reshaped_joint_position[:n_of_data_points])
    data_test = scaler.transform(reshaped_joint_position[n_of_data_points:])
    display.display(pd.DataFrame(data_train).describe())

# <codecell>

data_train.shape

# <codecell>

if run_config['use_time_series']:
    plots.plot_2d_distribution(data_train[:,-1,:], data_test[:, -1, :])
else:
    plots.plot_2d_distribution(data_train, data_test);

# <markdowncell>

# ## Use *tf.data* to create batches and shuffle the dataset

# <codecell>

def to_tf_data(X):
    return tf.data.Dataset.from_tensor_slices(X).shuffle(len(X)).batch(run_config['batch_size'])

train_dataset = to_tf_data(data_train)
test_dataset = to_tf_data(data_test) 

# <markdowncell>

# # model def

# <markdowncell>

# ## doc

# <markdowncell>

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
# 
# The dilated convolution between signal $f$ and kernel $k$ and dilution factor $l$ is defined as:
# 
# $$\left(k \ast_{l} f\right)_t = \sum_{\tau=-\infty}^{\infty} k_\tau \cdot f_{t - l\tau}$$
# 
# ![](./figures/diluted_convolution.png)

# <codecell>

def _receptive_field_size_temporal_conv_net_(kernel_size, n_layers):
    return (1 + 2 * (kernel_size - 1) * (2 ** n_layers - 1))

for k in range(2, 5):
    plt.plot([_receptive_field_size_temporal_conv_net_(kernel_size=k, n_layers=n) for n in range(10)], label=f"kernel size: {k}")
plt.xlabel('number of layers')
plt.ylabel('receptive field size')
plt.legend()

# <markdowncell>

# ## code

# <codecell>

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
    def __init__(self, num_channels, kernel_size=3, dropout=0.2,
                 trainable=True, name=None, dtype=None, 
                 activity_regularizer=None, **kwargs):
        super(TemporalConvNet, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        
        # TODO why not use a sequential layer here?
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

# <codecell>

# build using:
#   - https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/generative_examples/cvae.ipynb 
#   - https://www.kaggle.com/hone5com/fraud-detection-with-variational-autoencoder

def if_last(ls):
    for i, x in enumerate(ls):
        yield i + 1 == len(ls), x

def dense_layers(sizes, activation_fn=tf.nn.leaky_relu):
    # no activation in the last layer, because 
    return [tfkl.Dense(size, activation=None if is_last else activation_fn) for is_last, size in if_last(sizes)]


class R_VAE(tf.keras.Model):
    def __init__(self, latent_dim, input_shape, batch_size, n_layers=3, dropout_rate_temporal=0.2, loss_weight_reconstruction=1.0, loss_weight_kl=1.0):
        """
        Args:
        -----
        
        latent_dim              int, dimension of latent space
        input_shape             tuple, total input shape is: [batch_size, *input_shape]
        batch_size              int
        n_layers                int, number of dense layers. 
                                output shape of the dense layers is linearly scaled.
        dropout_rate_temporal   float, in [0, 1). dropout rate for temporal blocks (conv layers).
        """
        super(R_VAE, self).__init__()
        self.latent_dim = latent_dim
        self._input_shape = input_shape
        self._batch_size = batch_size
        self._loss_weight_reconstruction = loss_weight_reconstruction
        self._loss_weight_kl = loss_weight_kl
        self._layer_sizes_inference  = np.linspace(input_shape[-1], 2 * latent_dim, n_layers).astype(np.int)
        # pseudo reverse as the inference network goes down to double the latent space, ask Semigh about this
        # the 2 * n_layers is to keep compression speed roughly the same
        self._layer_sizes_generative = np.linspace(latent_dim, input_shape[-1], 2 * n_layers).astype(np.int).tolist()
        
        self.inference_net = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_shape[-1]),
                                                 *dense_layers(self._layer_sizes_inference)],
                                                 name='inference_net')

        self.generative_net = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                                                  *dense_layers(self._layer_sizes_generative)],
                                                  name='generative_net')
        
        if len(input_shape) == 1:
            self.temporal_conv_net = None
        elif len(input_shape) == 2:
            self.temporal_conv_net = TemporalConvNet(num_channels=[input_shape[-1]] * 3,  
                                                     dropout=dropout_rate_temporal,
                                                     name='temporal_conv_net')
        else:
            raise ValueError(f"Input shape is not good, got: {input_shape}")
            
        print(self._config_())
    
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
    
    def call(self, x, training=False, apply_sigmoid=False):
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
        }

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

if run_config['use_time_series']:
    assert len(data_train.shape) == 3, 'run all the necessary code, shape does not align with config'
else:
    assert len(data_train.shape) == 2, 'run all the necessary code, shape does not align with config'

# <codecell>

data_test.shape

# <codecell>

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    
    if len(x.shape) == 2:
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    else:
        # Note, the model is trained to reconstruct only the last, most current time step (by taking the last entry in the timeseries)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x[:, -1, :])
    #logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    
    # reconstruction loss
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1]) # down to [batch, loss]
    
    # KL loss
    logpz = log_normal_pdf(z, 0., 0.) # shouldn't it be `logvar = 0.0001` or something small?
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(model._loss_weight_reconstruction*logpx_z + model._loss_weight_kl*(logpz - logqz_x))

def compute_gradients(model, x): 
    with tf.GradientTape() as tape: 
        loss = compute_loss(model, x) 
        return tape.gradient(loss, model.trainable_variables), loss

optimizer = tf.train.AdamOptimizer(1e-4)
def apply_gradients(optimizer, gradients, variables, global_step=None):
    optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

# <markdowncell>

# ## training

# <codecell>

from datetime import datetime

def get_config_hash(config, digest_length=5):
    return str(hash(json.dumps({**run_config, '_executed_at_': str(datetime.now())}, sort_keys=True)))[:digest_length]

# <codecell>

# This is the init cell. The model and all related objects are created here.
if run_config['debug'] and run_config['d_no_compression']:
    latent_dim=data_train.shape[-1]
else:
    latent_dim = 8

tf.reset_default_graph()
test_losses = []
train_losses = []

model = R_VAE(latent_dim, 
              input_shape=data_train.shape[1:], 
              batch_size=run_config['batch_size'], 
              n_layers=4, 
              dropout_rate_temporal=0.2,
              loss_weight_reconstruction=1.0,
              loss_weight_kl=0.0)

model.inference_net.summary()
model.generative_net.summary()

_CONFIG_HASH_ = get_config_hash({**model._config_(), **run_config})

_base_path_ = f"{settings.config.__DATA_ROOT__}/neural_clustering_data/tvae_logs/{_CONFIG_HASH_}"
train_log_dir = _base_path_ + '/train'
test_log_dir = _base_path_ + '/test'
train_summary_writer = tfc.summary.create_file_writer(train_log_dir)
test_summary_writer = tfc.summary.create_file_writer(test_log_dir)

global_step = tf.train.get_or_create_global_step()


# <codecell>

# This is the run cell. Designed to be able to train the model for an arbitrary amount of epochs.
def _compute_loss_for_data_(model, data):
    loss = tfe.metrics.Mean()
    for x in data:
        loss(compute_loss(model, x))
    elbo = -loss.result()
    
    return elbo

print(f"will train model {model._config_()}, with global params: {run_config}, hash: {_CONFIG_HASH_}")
print(f"will train for ever...")
epoch = len(train_losses)
while True:
    start_time = time.time()
    for train_x in train_dataset:
        gradients, loss = compute_gradients(model, train_x)
        apply_gradients(optimizer, gradients, model.trainable_variables)
    end_time = time.time()
        
    test_losses += [_compute_loss_for_data_(model, test_dataset)]
    train_losses += [_compute_loss_for_data_(model, train_dataset)]
    
    with train_summary_writer.as_default(), tfc.summary.always_record_summaries():
        tfc.summary.scalar('loss', train_losses[-1], step=epoch)
  
    with test_summary_writer.as_default(), tfc.summary.always_record_summaries():
        tfc.summary.scalar('loss', test_losses[-1], step=epoch)
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch:0>3}, train test loss: {test_losses[-1]:0.3f}, took {end_time - start_time:0.3f} sec")
        tfc.summary.flush()
        
    epoch += 1

# <codecell>

#print(f"will train model {model._config_()}, with global params: {run_config}")
## TODO add tensorboard stuff
#def _compute_loss_for_data_(model, data):
#    loss = tfe.metrics.Mean()
#    for x in data:
#        loss(compute_loss(model, x))
#    elbo = -loss.result()
#    
#    return elbo
#
##print(f"will train for {epochs} epochs")
##for epoch in range(1, epochs + 1):
#print(f"will train for ever...")
#epoch = len(train_losses)
#while True:
#    start_time = time.time()
#    for train_x in train_dataset:
#        gradients, loss = compute_gradients(model, train_x)
#        apply_gradients(optimizer, gradients, model.trainable_variables)
#    end_time = time.time()
#
#    test_losses += [_compute_loss_for_data_(model, test_dataset)]
#    train_losses += [_compute_loss_for_data_(model, train_dataset)]
#
#    if epoch % 10 == 0:
#        print(f"Epoch: {epoch:0>3}, train test loss: {test_losses[-1]:0.3f}, took {end_time - start_time:0.3f} sec")
#        
#    epoch += 1

# <codecell>

len(train_losses)

# <codecell>

plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.xlabel('epochs')
plt.ylabel('loss (ELBO)')
plt.legend()

# <codecell>

def _reverse_to_original_shape_(pos_data, input_shape=None):
    input_shape = (15, -1)
        
    return scaler.inverse_transform(pos_data).reshape(pos_data.shape[0], *(input_shape))

# <codecell>

gen_pos_data = _reverse_to_original_shape_(np.vstack([model.sample().numpy() for _ in range(11)]))

if run_config['use_time_series']: 
    plots.plot_comparing_joint_position_with_reconstructed(_reverse_to_original_shape_(data_train[:, -1, :]), 
                                                           gen_pos_data[:data_train.shape[0]],
                                                           validation_cut_off=data_train.shape[0])
else:
    plots.plot_comparing_joint_position_with_reconstructed(_reverse_to_original_shape_(data_train), 
                                                           gen_pos_data[:data_train.shape[0]],
                                                           validation_cut_off=data_train.shape[0])

# <codecell>

pred_train = model.predict(data_train).numpy()
pred_train_rev = _reverse_to_original_shape_(model.predict(data_train).numpy())

if run_config['use_time_series']: 
    plots.plot_comparing_joint_position_with_reconstructed(_reverse_to_original_shape_(data_train[:, -1, :]),
                                                           pred_train_rev,
                                                           validation_cut_off=data_train.shape[0])
else:
    plots.plot_comparing_joint_position_with_reconstructed(_reverse_to_original_shape_(data_train), 
                                                           pred_train_rev,
                                                           validation_cut_off=data_train.shape[0])

# <markdowncell>

# # Latent space

# <codecell>

_all_input_ = np.vstack((data_train, data_test))

# <codecell>

from hdbscan import HDBSCAN

# <codecell>

from collections import namedtuple
from sklearn.manifold import TSNE

LatentSpaceEncoding = namedtuple('LatentSpaceEncoding', 'mean var')

X_latent = LatentSpaceEncoding(*map(lambda x: x.numpy(), model.encode(_all_input_)))
X_latent_mean_tsne_proj = TSNE(n_components=2, random_state=42).fit_transform(np.hstack((X_latent.mean, X_latent.var)))

cluster_assignments = HDBSCAN(min_cluster_size=4).fit_predict(np.hstack((X_latent.mean, X_latent.var)))

plt.figure(figsize=(20, 12))
for cluster in np.unique(cluster_assignments):
    c_idx = cluster_assignments == cluster
    sns.scatterplot(X_latent_mean_tsne_proj[c_idx, 0], X_latent_mean_tsne_proj[c_idx, 1], label=cluster)
    
plt.title('T-SNE proejection of latent space (mean & var stacked)');

# <markdowncell>

# # videos

# <codecell>

def reverse_pos_pipeline(x, normalisation_term=normalisation_factors):
    """TODO This is again pretty shitty... ultra hidden global variable"""
    return x + normalisation_term[:x.shape[-1]]

def video_prep_raw_data(data):
    if run_config['use_time_series']:
        return reverse_pos_pipeline(scaler.inverse_transform(data[:, -1, :]).reshape(-1, 15, 2))
    else:
        return reverse_pos_pipeline(scaler.inverse_transform(data.reshape(-1, 30)).reshape(-1, 15, 2))
    
def video_prep_recon_data(input_data):
    return reverse_pos_pipeline(scaler.inverse_transform(model(input_data).numpy()).reshape(-1, 15, 2))

# <codecell>

p = video.comparision_video_of_reconstruction([video_prep_raw_data(_all_input_), video_prep_recon_data(_all_input_)],
                                              images_paths_for_experiments=images_paths_for_experiments, 
                                              n_train=len(data_train),
                                              cluster_assignments=cluster_assignments,
                                              as_frames=False)

display_video(p)

# <codecell>

from collections import OrderedDict
_N_CLUSTER_TO_VIZ_ = 10
_positional_data = [video_prep_raw_data(_all_input_), video_prep_recon_data(_all_input_)]
_t = [(misc.flatten(sequences), cluster_id) for cluster_id, sequences in video.group_by_cluster(cluster_assignments).items()]
_t = sorted(_t, key=lambda x: len(x[0]), reverse=True)

cluster_colors = sns.color_palette(n_colors=len(np.unique(cluster_assignments)))

cluster_vids = OrderedDict((p[1], video.comparision_video_of_reconstruction(_positional_data,
                                                                      cluster_assignments=cluster_assignments,
                                                                      images_paths_for_experiments=images_paths_for_experiments,
                                                                      n_train=data_train.shape[0],
                                                                      cluster_colors=cluster_colors,
                                                                      cluster_id_to_visualize=p[1]))
                    for p in _t[:_N_CLUSTER_TO_VIZ_])

print('cluster_vids: ', cluster_vids.keys())

# <codecell>

display_video(list(cluster_vids.values())[0])
