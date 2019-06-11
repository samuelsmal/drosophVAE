import tensorflow as tf

tfc = tf.contrib

_TF_DEFAULT_SESSION_CONFIG_ = tf.ConfigProto()
_TF_DEFAULT_SESSION_CONFIG_.gpu_options.allow_growth = True
_TF_DEFAULT_SESSION_CONFIG_.gpu_options.polling_inactive_delay_msecs = 10


def tf_clean_variable_name(var_name):
    """
    Usage example (and and example of how to write a histogram in eager mode)

        with gradients_writer.as_default(), tfc.summary.always_record_summaries():
            for g, var_name in zip(gradients, [tf_clean_variable_name(v.name) for v in model.trainable_variables]):
                tfc.summary.histogram(f'gradient_{var_name}', g, step=epoch)
    """
    # example of a var_name: 'inf_dense_0_23/kernel:0'

    parts = var_name.split('/')

    parts[0] = '_'.join(parts[0].split('_')[:-1]) # drop the last bit. seems to be a counter (which leads to the confusion)
    parts[-1] = parts[-1][:-len(':x')] # and this is also not necessary

    return '/'.join(parts)


def tf_write_scalars(writer, scalars, step):
    """ Writes multiple scalars
    Usage example:

        _recorded_scalars_ =  ['loss', 'recon', 'kl']
        tf_write_scalars(train_summary_writer, zip(_recorded_scalars_, train_reports[-1]), step=epoch)
    """
    with writer.as_default(), tfc.summary.always_record_summaries():
        for n, v in scalars:
            tfc.summary.scalar(n, v, step=step)


def to_tf_data(X, y=None, batch_size=None):
    if y is None:
        return tf.data.Dataset.from_tensor_slices(X).shuffle(len(X)).batch(batch_size)
    else:
        return tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(X)).batch(batch_size)
