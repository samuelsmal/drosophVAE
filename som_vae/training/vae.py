import time
import warnings
import numpy as np

import tensorflow.contrib as tfc
import tensorflow.contrib.eager as tfe
import tensorflow as tf

from som_vae.helpers import tensorflow as tf_helpers
from som_vae.settings.config import ModelType, SetupConfig
from som_vae.losses.vae_loss import compute_loss
from som_vae.models import DrosophVAE, DrosophVAEConv, DrosophVAESkipConv

def init(input_shape, run_config):
    # This is the init cell. The model and all related objects are created here.
    #if run_config['debug'] and run_config['d_no_compression']:
    #    run_config['latent_dim'] = data_train.shape[-1]

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
    base_path = f"{SetupConfig.value('data_root_path')}/tvae_logs/cfg_description"
    model_checkpoints_path = f"{SetupConfig.value('data_root_path')}/models/{cfg_description}/checkpoint"
    train_summary_writer = tfc.summary.create_file_writer(base_path + '/train')
    test_summary_writer = tfc.summary.create_file_writer(base_path + '/test')

    return model, model_checkpoints_path, train_summary_writer, test_summary_writer, optimizer

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

def _progress_str_(epoch, train_reports, test_reports, time=None, stopped=False):
    progress_str = f"Epoch: {epoch:0>4}, train/test loss: {train_reports[-1][0]:0.3f}\t {test_reports[-1][0]:0.3f}"
    if time:
        progress_str += f" took {time:0.3f} sec"

    if stopped:
        progress_str = "Stopped training during " + progress_str

    return progress_str


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
        return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables, global_step=None):
    # TODO try out gradient clipping
    #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

def train(model,
          optimizer,
          train_summary_writer,
          test_summary_writer,
          train_reports=None,
          test_reports=None,
          train_dataset=None,
          test_dataset=None,
          n_epochs=10,
          early_stopping=True):

    if train_reports is None:
        train_reports = []
        test_reports = []

    cur_min_val_idx = 0
    epoch = len(train_reports)

    with warnings.catch_warnings():
        # pesky tensorflow again
        warnings.simplefilter(action='ignore', category=FutureWarning)
        for _ in range(n_epochs):
            try:
                start_time = time.time()
                for train_x in train_dataset:
                    gradients, loss = compute_gradients(model, train_x)
                    apply_gradients(optimizer, gradients, model.trainable_variables)
                end_time = time.time()

                train_reports += [compute_loss_for_data(model, train_dataset)]
                test_reports += [compute_loss_for_data(model, test_dataset)]

                _recorded_scalars_ =  ['loss', 'recon', 'kl']
                tf_helpers.tf_write_scalars(train_summary_writer, zip(_recorded_scalars_, train_reports[-1]), step=epoch)
                tf_helpers.tf_write_scalars(test_summary_writer,  zip(_recorded_scalars_, test_reports[-1]),  step=epoch)

                with train_summary_writer.as_default(), tfc.summary.always_record_summaries():
                    for g, var_name in zip(gradients, [tf_helpers.tf_clean_variable_name(v.name) for v in model.trainable_variables]):
                        tfc.summary.histogram(f'gradient_{var_name}', g, step=epoch)

                if epoch % 10 == 0:
                    print(_progress_str_(epoch, train_reports, test_reports, time=end_time - start_time))
                    tfc.summary.flush()
                else:
                    # simple "loading bar"
                    print('=' * (epoch % 10) + '.' * (10 - (epoch % 10)), end='\r')

                if epoch > 10 and test_reports[-1][0] < test_reports[cur_min_val_idx][0]:
                    cur_min_val_idx = epoch
                    model.save_weights(_model_checkpoints_path_)

                epoch += 1

                if early_stopping and np.argmin(np.array(test_reports)[:, 1]) < (len(test_reports) - 10):
                    # if there was no improvement in the last 10 epochs, stop it
                    print('early stopping')
                    break
            except KeyboardInterrupt:
                tfc.summary.flush()
                print(_progress_str_(epoch, train_reports, test_reports, stopped=True))
                break


    tfc.summary.flush()

    return model, optimizer, train_reports, test_reports

