import time
import warnings
import numpy as np
from functools import partial

import tensorflow.contrib as tfc
import tensorflow.contrib.eager as tfe
import tensorflow as tf

from som_vae.helpers import tensorflow as tf_helpers
from som_vae.settings.config import ModelType, SetupConfig
from som_vae.losses.vae_loss import compute_loss
from som_vae.training.utils import train as _train_
from som_vae.models import DrosophVAE, DrosophVAEConv, DrosophVAESkipConv

def init(input_shape, run_config, reset_graph=True):
    # This is the init cell. The model and all related objects are created here.
    #if run_config['debug'] and run_config['d_no_compression']:
    #    run_config['latent_dim'] = data_train.shape[-1]

    if reset_graph:
        tf.reset_default_graph()

    # This is a bit shitty... but I like to run the training loop multiple times...

    print(f"Using model: {run_config['model_impl']}")
    #input_shape: data_train.shape[1:]
    model_config = {'latent_dim': run_config['latent_dim'],
                    'input_shape': input_shape,
                    'batch_size': run_config['batch_size'],
                    'loss_weight_reconstruction': run_config['loss_weight_reconstruction'],
                    'loss_weight_kl': run_config['loss_weight_kl']}

    if run_config['model_impl'] == ModelType.TEMP_CONV:
        model = DrosophVAE(run_config['latent_dim'],
                           input_shape=data_train.shape[1:],
                           batch_size=run_config['batch_size'],
                           n_layers=run_config['n_conv_layers'],
                           dropout_rate_temporal=run_config['dropout_rate'],
                           loss_weight_reconstruction=run_config['loss_weight_reconstruction'],
                           loss_weight_kl=run_config['loss_weight_kl'],
                           use_wavenet_temporal_layer=run_config['use_time_series'])

        if run_config['use_time_series']:
            model.temporal_conv_net.summary()
    elif run_config['model_impl'] == ModelType.PADD_CONV:
        model = DrosophVAEConv(**{**model_config, 'with_batch_norm': run_config['with_batch_norm']})
    elif run_config['model_impl'] == ModelType.SKIP_PADD_CONV:
        model = DrosophVAESkipConv(**model_config)
    else:
        raise ValueError('not such model')

    model.inference_net.summary(line_length=100)
    model.generative_net.summary(line_length=100)

    if run_config['optimizer'] == 'Adam':
        optimizer = tf.train.AdamOptimizer(1e-4)
    else:
        raise NotImplementedError

    cfg_description = run_config.description()
    base_path = f"{SetupConfig.value('data_root_path')}/tvae_logs/{cfg_description}"
    model_checkpoints_path = f"{SetupConfig.value('data_root_path')}/models/{cfg_description}/checkpoint"
    train_summary_writer = tfc.summary.create_file_writer(base_path + '/train')
    test_summary_writer = tfc.summary.create_file_writer(base_path + '/test')

    return {'model': model,
            'optimizer': optimizer,
            'train_summary_writer': train_summary_writer,
            'test_summary_writer': test_summary_writer,
            'model_checkpoints_path': model_checkpoints_path}

def compute_loss_for_data(model, data):
    loss = tfe.metrics.Mean()
    recon = tfe.metrics.Mean()
    kl = tfe.metrics.Mean()
    for batch in data:
        loss_b, recon_b, kl_b  = compute_loss(model, batch, detailed=True)
        loss(loss_b)
        recon(recon_b)
        kl(kl_b)

    total_loss = loss.result()
    total_recon = recon.result()
    total_kl = kl.result()

    return total_loss, total_recon, total_kl

def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
        return tape.gradient(loss, model.trainable_variables), loss


train = partial(_train_, gradient_fn=compute_gradients, loss_report_fn=compute_loss_for_data)
