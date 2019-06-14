# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# sources: 
# 
# - https://github.com/tensorflow/probability/blob/v0.6.0/tensorflow_probability/examples/vae.py
# 
# 
# The VAE defines a generative model in which a latent code `Z` is sampled from a
# prior `p(Z)`, then used to generate an observation `X` by way of a decoder
# `p(X|Z)`. The full reconstruction follows
# 
# ```none
#    X ~ p(X)              # A random image from some dataset.
#    Z ~ q(Z | X)          # A random encoding of the original image ("encoder").
# Xhat ~ p(Xhat | Z)       # A random reconstruction of the original image
#                          #   ("decoder").
# ```
# 
# To fit the VAE, we assume an approximate representation of the posterior in the
# form of an encoder `q(Z|X)`. We minimize the KL divergence between `q(Z|X)` and
# the true posterior `p(Z|X)`: this is equivalent to maximizing the evidence lower
# bound (ELBO),
# 
# ```none
# -log p(x)
# = -log int dz p(x|z) p(z)
# = -log int dz q(z|x) p(x|z) p(z) / q(z|x)
# <= int dz q(z|x) (-log[ p(x|z) p(z) / q(z|x) ])   # Jensen's Inequality
# =: KL[q(Z|x) || p(x|Z)p(Z)]
# = -E_{Z~q(Z|x)}[log p(x|Z)] + KL[q(Z|x) || p(Z)]
# ```
# 
# -or-
# 
# ```none
# -log p(x)
# = KL[q(Z|x) || p(x|Z)p(Z)] - KL[q(Z|x) || p(Z|x)]
# <= KL[q(Z|x) || p(x|Z)p(Z)                        # Positivity of KL
# = -E_{Z~q(Z|x)}[log p(x|Z)] + KL[q(Z|x) || p(Z)]
# ```
# 
# The `-E_{Z~q(Z|x)}[log p(x|Z)]` term is an expected reconstruction loss and
# `KL[q(Z|x) || p(Z)]` is a kind of distributional regularizer. See
# [Kingma and Welling (2014)][1] for more details.
# 
# This script supports both a (learned) mixture of Gaussians prior as well as a
# fixed standard normal prior. You can enable the fixed standard normal prior by
# setting `mixture_components` to 1. Note that fixing the parameters of the prior
# (as opposed to fitting them with the rest of the model) incurs no loss in
# generality when using only a single Gaussian. The reasoning for this is
# two-fold:
# 
#   * On the generative side, the parameters from the prior can simply be absorbed
#     into the first linear layer of the generative net. If `z ~ N(mu, Sigma)` and
#     the first layer of the generative net is given by `x = Wz + b`, this can be
#     rewritten,
# 
#       s ~ N(0, I)
#       x = Wz + b
#         = W (As + mu) + b
#         = (WA) s + (W mu + b)
# 
#     where Sigma has been decomposed into A A^T = Sigma. In other words, the log
#     likelihood of the model (E_{Z~q(Z|x)}[log p(x|Z)]) is independent of whether
#     or not we learn mu and Sigma.
# 
#   * On the inference side, we can adjust any posterior approximation
#     q(z | x) ~ N(mu[q], Sigma[q]), with
# 
#     new_mu[p] := 0
#     new_Sigma[p] := eye(d)
#     new_mu[q] := inv(chol(Sigma[p])) @ (mu[p] - mu[q])
#     new_Sigma[q] := inv(Sigma[q]) @ Sigma[p]
# 
#     A bit of algebra on the KL divergence term `KL[q(Z|x) || p(Z)]` reveals that
#     it is also invariant to the prior parameters as long as Sigma[p] and
#     Sigma[q] are invertible.
# 
# This script also supports using the analytic KL (KL[q(Z|x) || p(Z)]) with the
# `analytic_kl` flag. Using the analytic KL is only supported when
# `mixture_components` is set to 1 since otherwise no analytic form is known.
# 
# Here we also compute tighter bounds, the IWAE [Burda et. al. (2015)][2].
# 
# These as well as image summaries can be seen in Tensorboard. For help using
# Tensorboard see
# https://www.tensorflow.org/guide/summaries_and_tensorboard
# which can be run with
#   `python -m tensorboard.main --logdir=MODEL_DIR`
# 
# #### References
# 
# [1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In
#      _International Conference on Learning Representations_, 2014.
#      https://arxiv.org/abs/1312.6114
# [2]: Yuri Burda, Roger Grosse, Ruslan Salakhutdinov. Importance Weighted
#      Autoencoders. In _International Conference on Learning Representations_,
#      2015.
#      https://arxiv.org/abs/1509.00519

# <codecell>

from datetime import date
from datetime import timedelta

#_NIGHTLY_VERSION_ = 20190312 # 20190430 # 20190502 # 20190312
#_NIGHTLY_VERSION_ = str((date(2019, 3, 19) - timedelta(days=diff))).replace('-', '')
#_NIGHTLY_VERSION_ = str((date.today() - timedelta(days=diff))).replace('-', '')
#!pip -q install --upgrade tf-nightly==1.14.1-dev{_NIGHTLY_VERSION_} \
#                          tf-nightly-gpu==1.14.1-dev{_NIGHTLY_VERSION_} \
#                          tfp-nightly==0.7.0.dev20190312


# <codecell>

import tensorflow_probability as tfp

# <codecell>

# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf
tfe = tf.contrib.eager
tf.enable_eager_execution()


#from tensorflow.python import tf2
#if not tf2.enabled():
#    import tensorflow.compat.v2 as tf
#    #import tensorflow.compat.v1 as tf
#    tf.enable_v2_behavior()
#    assert tf2.enabled()

import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
#import imageio
from IPython import display
from sklearn.preprocessing import StandardScaler, MinMaxScaler


import inspect

from drosoph_vae.helpers import tensorflow as _donotim
print(inspect.getsource(_donotim))

from tensorflow.python.client import device_lib

#device_lib.list_local_devices()

#import tensorflow as tf

_TF_DEFAULT_SESSION_CONFIG_ = tf.ConfigProto(device_count={'GPU': 1})
_TF_DEFAULT_SESSION_CONFIG_.gpu_options.allow_growth = True 
_TF_DEFAULT_SESSION_CONFIG_.gpu_options.polling_inactive_delay_msecs = 10
from tensorflow.keras.utils import plot_model

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import os

#%matplotlib inline
from drosoph_vae.helpers.misc import extract_args, chunks, foldl
from drosoph_vae.helpers.jupyter import fix_layout, display_video
from drosoph_vae.settings import config, skeleton
from drosoph_vae.helpers import video, plots, misc, jupyter
from drosoph_vae import preprocessing
from drosoph_vae.helpers.logging import enable_logging
#from drosoph_vae.helpers.tensorflow import _TF_DEFAULT_SESSION_CONFIG_

# <codecell>

jupyter.fix_layout()

# <codecell>

### Utility Functions
## Plots
# Plot Feature Projection [credit: https://www.kaggle.com/shivamb/semi-supervised-classification-using-autoencoders]
def tsne_plot(x1, y1, name=None):
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)
#     plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='Non Fraud', s=2)

    plt.legend(loc='best');
    #plt.savefig(name);
    plt.title('tsne')
    plt.show();
    
    
# Plot Keras training history
def plot_loss(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# <codecell>

from drosoph_vae import settings
from drosoph_vae import preprocessing

joint_positions, normalisation_factors = preprocessing.get_data_and_normalization(settings.data.EXPERIMENTS)

frames_idx_with_labels = preprocessing.get_frames_with_idx_and_labels(settings.data.LABELLED_DATA)[:len(joint_positions)]

#frames_of_interest = frames_idx_with_labels.label.isin([settings.data._BehaviorLabel_.GROOM_ANT, settings.data._BehaviorLabel_.WALK_FORW, settings.data._BehaviorLabel_.REST])
frames_of_interest = ~frames_idx_with_labels.label.isin([settings.data._BehaviorLabel_.REST])

joint_positions = joint_positions[frames_of_interest]
frames_idx_with_labels = frames_idx_with_labels[frames_of_interest]

# <codecell>

# flatten the data
reshaped_joint_position = joint_positions[:,:,: config.NB_DIMS].reshape(joint_positions.shape[0], -1).astype(np.float32)


# scaling the data to be in [0, 1]
# this is due to the sigmoid activation function in the reconstruction
scaler = MinMaxScaler()
#resh = scaler.fit_transform(resh)

print(f"total number of input data:{reshaped_joint_position.shape}")


#if drosoph_vae_config['time_series']:
#    _time_series_idx_ = list(to_time_series(range(len(joint_positions))))
#    _jp = np.concatenate([joint_positions[idx].reshape(1, -1, 30) for idx in _time_series_idx_], axis=0)
#else:
#    _jp = joint_positions
#    
#nb_of_data_points = (reshaped_joint_position.shape[0] // config['batch_size']) * config['batch_size']
# train - test split
nb_of_data_points = int(reshaped_joint_position.shape[0] * 0.7)
#
X_train = scaler.fit_transform(reshaped_joint_position[:nb_of_data_points])
X_test = scaler.transform(reshaped_joint_position[nb_of_data_points:])
# just generating some labels, no clue what they are for except validation?
#labels = frames_idx_with_labels['label'].apply(lambda x: x.value).values

#if drosoph_vae_config['time_series']:
#    labels = np.concatenate([labels[idx].reshape(1, -1, 1) for idx in _time_series_idx_], axis=0)

#data = {
#  "X_train": data_train,
#  "X_val": data_test,
#  "y_train": labels[:nb_of_data_points],
#  "y_val": labels[nb_of_data_points:]
#}


# <codecell>

#flags.DEFINE_float(
#    "learning_rate", default=0.001, help="Initial learning rate.")
#flags.DEFINE_integer(
#    "max_steps", default=5001, help="Number of training steps to run.")
#flags.DEFINE_integer(
#    "latent_size",
#    default=16,
#    help="Number of dimensions in the latent code (z).")
#flags.DEFINE_integer("base_depth", default=32, help="Base depth for layers.")
#flags.DEFINE_string(
#    "activation",
#    default="leaky_relu",
#    help="Activation function for all hidden layers.")
#flags.DEFINE_integer(
#    "batch_size",
#    default=32,
#    help="Batch size.")
#flags.DEFINE_integer(
#    "n_samples", default=16, help="Number of samples to use in encoding.")
#flags.DEFINE_integer(
#    "mixture_components",
#    default=100,
#    help="Number of mixture components to use in the prior. Each component is "
#         "a diagonal normal distribution. The parameters of the components are "
#         "intialized randomly, and then learned along with the rest of the "
#         "parameters. If `analytic_kl` is True, `mixture_components` must be "
#         "set to `1`.")
#flags.DEFINE_bool(
#    "analytic_kl",
#    default=False,
#    help="Whether or not to use the analytic version of the KL. When set to "
#         "False the E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)] form of the ELBO "
#         "will be used. Otherwise the -KL(q(Z|X) || p(Z)) + "
#         "E_{Z~q(Z|X)}[log p(X|Z)] form will be used. If analytic_kl is True, "
#         "then you must also specify `mixture_components=1`.")
#flags.DEFINE_string(
#    "data_dir",
#    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/data"),
#    help="Directory where data is stored (if using real data).")
#flags.DEFINE_string(
#    "model_dir",
#    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"),
#    help="Directory to put the model's fit.")
#flags.DEFINE_integer(
#    "viz_steps", default=500, help="Frequency at which to save visualizations.")
#flags.DEFINE_bool(
#    "fake_data",
#    default=False,
#    help="If true, uses fake data instead of MNIST.")
#flags.DEFINE_bool(
#    "delete_existing",
#    default=False,
#    help="If true, deletes existing `model_dir` directory.")
#
#FLAGS = flags.FLAGS


# <codecell>

params = {
    "learning_rate": 0.001,
    "max_steps": 5001,
    "latent_size":  2,
    "base_depth": 32,
    "activation": "leaky_relu",
    "batch_size": 32,
    "n_samples": 16,
    "mixture_components": 1,
    "analytic_kl": False,
    "model_dir": '../neural_clustering_data/models/',
    "viz_steps": 500
}

# <codecell>

def _softplus_inverse(x):
    """Helper which computes the function inverse of `tf.nn.softplus`."""
    return tf.log(tf.math.expm1(x))


def make_encoder(activation, latent_size, base_depth):
    """Creates the encoder function.
  
    Args:
      activation: Activation function in hidden layers.
      latent_size: The dimensionality of the encoding.
      base_depth: The lowest depth for a layer.
  
    Returns:
      encoder: A `callable` mapping a `Tensor` of images to a
        `tfd.Distribution` instance over encodings.
    """
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=activation)
  
    encoder_net = tf.keras.Sequential([
        conv(base_depth, 5, 1),
        conv(base_depth, 5, 2),
        conv(2 * base_depth, 5, 1),
        conv(2 * base_depth, 5, 2),
        conv(4 * latent_size, 7, padding="VALID"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2 * latent_size, activation=None),
    ])
  
    def encoder(images):
        images = 2 * tf.cast(images, dtype=tf.float32) - 1
        net = encoder_net(images)
        return tfd.MultivariateNormalDiag(
            loc=net[..., :latent_size],
            scale_diag=tf.nn.softplus(net[..., latent_size:] +
                                      _softplus_inverse(1.0)),
            name="code")
  
    return encoder


def make_decoder(activation, latent_size, output_shape, base_depth):
    """Creates the decoder function.

    Args:
      activation: Activation function in hidden layers.
      latent_size: Dimensionality of the encoding.
      output_shape: The output image shape.
      base_depth: Smallest depth for a layer.

    Returns:
      decoder: A `callable` mapping a `Tensor` of encodings to a
        `tfd.Distribution` instance over images.
    """
    deconv = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=activation)

    decoder_net = tf.keras.Sequential([
        deconv(2 * base_depth, 7, padding="VALID"),
        deconv(2 * base_depth, 5),
        deconv(2 * base_depth, 5, 2),
        deconv(base_depth, 5),
        deconv(base_depth, 5, 2),
        deconv(base_depth, 5),
        conv(output_shape[-1], 5, activation=None),
    ])

    def decoder(codes):
        original_shape = tf.shape(codes)
        # Collapse the sample and batch dimension and convert to rank-4 tensor for
        # use with a convolutional decoder network.
        codes = tf.reshape(codes, (-1, 1, 1, latent_size))
        logits = decoder_net(codes)
        logits = tf.reshape(logits, shape=tf.concat([original_shape[:-1], output_shape], axis=0))
        return tfd.Independent(tfd.Bernoulli(logits=logits),
                               reinterpreted_batch_ndims=len(output_shape),
                               name="image")

    return decoder


def make_mixture_prior(latent_size, mixture_components):
    """Creates the mixture of Gaussians prior distribution.
  
    Args:
      latent_size: The dimensionality of the latent representation.
      mixture_components: Number of elements of the mixture.
  
    Returns:
      random_prior: A `tfd.Distribution` instance representing the distribution
        over encodings in the absence of any evidence.
    """
    if mixture_components == 1:
       # See the module docstring for why we don't learn the parameters here.
       return tfd.MultivariateNormalDiag(
           loc=tf.zeros([latent_size]),
           scale_identity_multiplier=1.0)
  
    loc = tf.get_variable(name="loc", shape=[mixture_components, latent_size])
    raw_scale_diag = tf.get_variable(name="raw_scale_diag", shape=[mixture_components, latent_size])
    mixture_logits = tf.get_variable(name="mixture_logits", shape=[mixture_components])
  
    return tfd.MixtureSameFamily(
        components_distribution=tfd.MultivariateNormalDiag(
            loc=loc,
            scale_diag=tf.nn.softplus(raw_scale_diag)),
        mixture_distribution=tfd.Categorical(logits=mixture_logits),
        name="prior")

# <codecell>

def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images


def image_tile_summary(name, tensor, rows=8, cols=8):
    tf.summary.image(name, pack_images(tensor, rows, cols), max_outputs=1)

# <codecell>

def model_fn(features, labels, mode, params, config):
    """Builds the model function for use in an estimator.
  
    Arguments:
      features: The input features for the estimator.
      labels: The labels, unused here.
      mode: Signifies whether it is train or test or predict.
      params: Some hyperparameters as a dictionary.
      config: The RunConfig, unused here.
  
    Returns:
      EstimatorSpec: A tf.estimator.EstimatorSpec instance.
    """
    if params["analytic_kl"] and params["mixture_components"] != 1:
        raise NotImplementedError(
            "Using `analytic_kl` is only supported when `mixture_components = 1` "
            "since there's no closed form otherwise.")
  
    encoder = make_encoder(params["activation"],
                           params["latent_size"],
                           params["base_depth"])
    decoder = make_decoder(params["activation"],
                           params["latent_size"],
                           IMAGE_SHAPE,
                           params["base_depth"])
    latent_prior = make_mixture_prior(params["latent_size"],
                                      params["mixture_components"])
  
    image_tile_summary("input", tf.to_float(features), rows=1, cols=16)
  
    approx_posterior = encoder(features)
    approx_posterior_sample = approx_posterior.sample(params["n_samples"])
    decoder_likelihood = decoder(approx_posterior_sample)
    image_tile_summary(
        "recon/sample",
        tf.to_float(decoder_likelihood.sample()[:3, :16]),
        rows=3,
        cols=16)
    image_tile_summary(
        "recon/mean",
        decoder_likelihood.mean()[:3, :16],
        rows=3,
        cols=16)
  
    # `distortion` is just the negative log likelihood.
    distortion = -decoder_likelihood.log_prob(features)
    avg_distortion = tf.reduce_mean(distortion)
    tf.summary.scalar("distortion", avg_distortion)
  
    if params["analytic_kl"]:
        rate = tfd.kl_divergence(approx_posterior, latent_prior)
    else:
        rate = (approx_posterior.log_prob(approx_posterior_sample)
              - latent_prior.log_prob(approx_posterior_sample))
    avg_rate = tf.reduce_mean(rate)
    tf.summary.scalar("rate", avg_rate)
  
    elbo_local = -(rate + distortion)
  
    elbo = tf.reduce_mean(elbo_local)
    loss = -elbo
    tf.summary.scalar("elbo", elbo)
  
    importance_weighted_elbo = tf.reduce_mean(
        tf.reduce_logsumexp(elbo_local, axis=0) -
        tf.log(tf.to_float(params["n_samples"])))
    tf.summary.scalar("elbo/importance_weighted", importance_weighted_elbo)
  
    # Decode samples from the prior for visualization.
    random_image = decoder(latent_prior.sample(16))
    image_tile_summary(
        "random/sample", tf.to_float(random_image.sample()), rows=4, cols=4)
    image_tile_summary("random/mean", random_image.mean(), rows=4, cols=4)
  
    # Perform variational inference by minimizing the -ELBO.
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step,
                                          params["max_steps"])
    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
  
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
            "elbo": tf.metrics.mean(elbo),
            "elbo/importance_weighted": tf.metrics.mean(importance_weighted_elbo),
            "rate": tf.metrics.mean(avg_rate),
            "distortion": tf.metrics.mean(avg_distortion),
        },
    )

# <codecell>

def build_input_fns(data_dir, batch_size):
    """Builds an Iterator switching between train and heldout data."""
  
    # Build an iterator over training batches.
    def train_input_fn():
        dataset = static_mnist_dataset(data_dir, "train")
        dataset = dataset.shuffle(50000).repeat().batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()
  
    # Build an iterator over the heldout set.
    def eval_input_fn():
        eval_dataset = static_mnist_dataset(data_dir, "valid")
        eval_dataset = eval_dataset.batch(batch_size)
        return eval_dataset.make_one_shot_iterator().get_next()
  
    return train_input_fn, eval_input_fn


params['activation'] = getattr(tf.nn, params["activation"])

#train_input_fn, eval_input_fn = build_fake_input_fns(FLAGS.batch_size)

estimator = tf.estimator.Estimator(
  model_fn,
  params=params,
  config=tf.estimator.RunConfig(
      model_dir= "../neural_clustering_data/logs",
      save_checkpoints_steps=params['viz_steps'],
  ),
)

for _ in range(params['max_steps'] // params['viz_steps']):
    estimator.train(train_input_fn, steps=params['viz_steps'])
    eval_results = estimator.evaluate(eval_input_fn)
    print("Evaluation_results:\n\t%s\n" % eval_results)

# <codecell>

def dense_layers(sizes):
    return tfk.Sequential([tfkl.Dense(size, activation=tf.nn.leaky_relu) for size in sizes])

tf.reset_default_graph()

original_dim = data_train.shape[1]
input_shape = data_train[0].shape
dense_layer_dims = [20, 10, 8]
latent_dim = 2
batch_size = 128
max_epochs = 1000

# prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1),
#                         reinterpreted_batch_ndims=1)

prior = tfd.MultivariateNormalDiag(loc=tf.zeros([latent_dim]), 
                               scale_identity_multiplier=1.0)

encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=input_shape, name='encoder_input'),
    dense_layers(dense_layer_dims),
    tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim), activation=None),
    tfpl.MultivariateNormalTriL(latent_dim, activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
], name='encoder')

encoder.summary()
#plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[latent_dim]),
    dense_layers(reversed(dense_layer_dims)),
    tfkl.Dense(tfpl.IndependentNormal.params_size(original_dim), activation=None),
    tfpl.IndependentNormal(original_dim),
], name='decoder')

decoder.summary()
#plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

vae = tfk.Model(inputs=encoder.inputs,
                outputs=decoder(encoder.outputs[0]),
                name='vae_mlp')

negloglik = lambda x, rv_x: -rv_x.log_prob(x)

vae.compile(optimizer=tf.keras.optimizers.Nadam(), 
            loss=negloglik)

vae.summary()
plot_model(vae,
           to_file='vae_mlp.png',
           show_shapes=True)

# <codecell>

tf_train = tf.data.Dataset.from_tensor_slices((data_train, data_train)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(int(10e4))
tf_val = tf.data.Dataset.from_tensor_slices((data_test, data_test)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(int(10e4))

# <codecell>

checkpointer = ModelCheckpoint(filepath=f"{config.__DATA_ROOT__}/experimental/model_checkpoints/vae_v2-0.1.0-mdl.h5", verbose=0, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.005, patience=20, verbose=0, restore_best_weights=True)

hist = vae.fit(tf_train,
               epochs=max_epochs,
               shuffle=True,
               verbose=0,
               validation_data=tf_val,
               callbacks=[checkpointer, earlystopper])


plot_loss(hist)

# <markdowncell>

# # reconstruction

# <codecell>

reconstruct_samples_n = 100

def reconstruction_log_prob(eval_samples, reconstruct_samples_n):
    encoder_out = encoder(eval_samples)
    encoder_samples = encoder_out.sample(reconstruct_samples_n)
    return np.mean(decoder(encoder_samples).log_prob(eval_samples), axis=0)

# <codecell>

def _reverse_to_original_shape_(pos_data, input_shape=None):
    if input_shape is None:
        if config.NB_DIMS == 2:
            input_shape = (-1, config.NB_DIMS)
        else:
            input_shape = (-1,)
        
    return scaler.inverse_transform(pos_data).reshape(pos_data.shape[0], *(input_shape))

reconstructed_train =  _reverse_to_original_shape_()

# <codecell>

plots.plot_comparing_joint_position_with_reconstructed(joint_positions, 
                                                       np.vstack((reconstructed_from_encoding_train, reconstructed_from_encoding_val)), validation_cut_off=nb_of_data_points)

# <markdowncell>

# # latent space plot

# <codecell>

Y = frames_idx_with_labels.label.apply(lambda x: x.value)
X = np.vstack((data_train, data_test))
latent_x_mean = encoder(X).mean()
latent_x_stddev  = encoder(X).stddev()

plt.scatter(latent_x_mean[:, 0], latent_x_mean[:, 1], c=Y, cmap='RdYlGn_r', s=2)
plt.title('latent means')
plt.ylabel('mean[1]')
plt.xlabel('mean[0]')
plt.show()

# <codecell>

plt.scatter(latent_x_stddev[:, 0], latent_x_stddev[:, 1], c=Y, cmap='RdYlGn_r', s=2)
plt.title('latent standard deviations')
plt.ylabel('stddev[1]')
plt.xlabel('stddev[0]')
plt.show()

# <markdowncell>

# ## sampling

# <codecell>

latent_x = encoder(X).sample()
plt.scatter(latent_x[:, 0], latent_x[:, 1], c=Y, cmap='RdYlGn_r', s=2)
plt.title('latent vector samples')
plt.ylabel('z[1]')
plt.xlabel('z[0]')
plt.show()

# <codecell>

x_log_prob = reconstruction_log_prob(X, reconstruct_samples_n)
#ax = plt.hist(x_log_prob, 60)
plt.hist([x_log_prob[frames_idx_with_labels['label'] == l] for l in seen_labels], 60)
plt.title('reconstruction log probability')
plt.ylabel('frequency')
plt.xlabel("log p(x|x')")
plt.show()
