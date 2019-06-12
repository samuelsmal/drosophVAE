import time
import warnings
import numpy as np

import tensorflow.contrib as tfc
import tensorflow.contrib.eager as tfe
import tensorflow as tf

from som_vae.helpers import tensorflow as tf_helpers
from som_vae.settings.config import ModelType, SetupConfig

def _progress_str_(epoch, train_reports, test_reports, time=None, stopped=False):
    progress_str = f"Epoch: {epoch:0>4}, train/test loss: {train_reports[-1][0]:0.3f}\t {test_reports[-1][0]:0.3f}"
    if time:
        progress_str += f" took {time:0.3f} sec"

    if stopped:
        progress_str = "Stopped training during " + progress_str

    return progress_str


def apply_gradients(optimizer, gradients, variables, global_step=None):
    # TODO try out gradient clipping
    #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

def train(model,
          optimizer,
          train_summary_writer,
          test_summary_writer,
          model_checkpoints_path,
          gradient_fn=None,
          loss_report_fn=None,
          train_reports=None,
          test_reports=None,
          train_dataset=None,
          test_dataset=None,
          n_epochs=10,
          early_stopping=True):

    if train_reports is None:
        train_reports = []
        test_reports = []
    else:
        train_reports = train_reports.tolist()
        test_reports = test_reports.tolist()

    cur_min_val_idx = 0
    epoch = len(train_reports)

    with warnings.catch_warnings():
        # pesky tensorflow again
        warnings.simplefilter(action='ignore', category=FutureWarning)
        for _ in range(n_epochs):
            try:
                start_time = time.time()
                for train_x in train_dataset:
                    gradients, loss = gradient_fn(model, train_x)
                    apply_gradients(optimizer, gradients, model.trainable_variables)
                end_time = time.time()

                train_reports += [loss_report_fn(model, train_dataset)]
                test_reports += [loss_report_fn(model, test_dataset)]

                # this is a bit tricky here. zip takes the shortest of them (which is what we want
                # to support multiple training gradients and loss functions
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
                    model.save_weights(model_checkpoints_path)

                epoch += 1

                if early_stopping and epoch > 10 and np.argmin(np.array(test_reports)[:, 1]) < (len(test_reports) - 10):
                    # if there was no improvement in the last 10 epochs, stop it
                    print('early stopping')
                    break
            except KeyboardInterrupt:
                tfc.summary.flush()
                print(_progress_str_(epoch, train_reports, test_reports, stopped=True))
                break


    tfc.summary.flush()

    return {'model': model,
            'optimizer': optimizer,
            'train_reports':np.array(train_reports),
            'test_reports': np.array(test_reports)}

