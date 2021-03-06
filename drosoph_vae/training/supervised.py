"""Look at this as a functional interface. The others have the same functions.
"""
import time
import warnings
import numpy as np
from functools import partial

import tensorflow.contrib as tfc
import tensorflow.contrib.eager as tfe
import tensorflow as tf

from drosoph_vae.training.utils import train as _train_
from drosoph_vae.helpers import tensorflow as tf_helpers
from drosoph_vae.settings.config import ModelType, SetupConfig
from drosoph_vae.losses.triplet_loss import compute_loss_labels
from drosoph_vae.models import DrosophVAE, DrosophVAEConv, DrosophVAESkipConv

#labels_as_int = frames_idx_with_labels['label'].apply(lambda x: x.value).values
#train_dataset = to_tf_data(data_train, labels_as_int[run_config['time_series_length'] - 1:len(data_train)+run_config['time_series_length'] - 1])
#test_dataset = to_tf_data(data_test, labels_as_int[len(data_train) + run_config['time_series_length'] - 1:])

def init(model, run_config, reset_graph=False):

    if reset_graph:
        tf.reset_default_graph()

    if run_config['optimizer'] == 'Adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(run_config['supervised_learning_rate'])
    else:
        raise NotImplementedError

    cfg_description = f"{run_config.description()}_supervised"
    base_path = f"{SetupConfig.value('data_root_path')}/tvae_logs/{cfg_description}"
    model_checkpoints_path = f"{SetupConfig.value('data_root_path')}/models/{cfg_description}/checkpoint"
    train_summary_writer = tfc.summary.create_file_writer(base_path + '/train')
    test_summary_writer = tfc.summary.create_file_writer(base_path + '/test')

    return {'model': model,
            'optimizer': optimizer,
            'train_summary_writer': train_summary_writer,
            'test_summary_writer': test_summary_writer,
            'model_checkpoints_path': model_checkpoints_path}

def compute_gradients(model, x, y):
    with tf.GradientTape() as tape:
        mean, var = model(x)
        encoded = tf.nn.l2_normalize(tf.concat((mean, var), axis=-1))
        # super crappy way of finding out if the model goes down to one time-step or not
        if len(encoded.shape) > 2:
            loss = compute_loss_labels(encoded[:,-1,:], y[:, -1])
        else:
            loss = compute_loss_labels(encoded, y[:,-1])
        return tape.gradient(loss, model.trainable_variables), loss


def compute_loss_for_data(model, data):
    loss = tfe.metrics.Mean()
    for x, y in data:
        #mean, var = model(batch_x)
        #encoded = tf.nn.l2_normalize(((mean, var)))
        #loss_b = compute_loss_labels(mean, batch_y)
        mean, var = model(x)
        encoded = tf.nn.l2_normalize(tf.concat((mean, var), axis=-1))
        #loss_b = compute_loss_labels(model, batch_x, batch_y)

        if len(encoded.shape) > 2:
            loss_b = compute_loss_labels(encoded[:,-1,:], y[:, -1])
        else:
            loss_b = compute_loss_labels(encoded, y[:,-1])
        loss(loss_b)

    return (loss.result(), )

train = partial(_train_, gradient_fn=compute_gradients, loss_report_fn=compute_loss_for_data)
