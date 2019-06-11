import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.eager.execution_callbacks import InfOrNanError
from tensorflow.python.eager.core import _NotOkStatusException

# LOSS FUNCTION
#
# https://github.com/pytorch/examples/issues/399
#   Argues that since we are using a normal distribution we should not use any activation function in the last layer
#   and the loss should be MSE.
# https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational-auto-encoder?rq=1
#   Some general discussion about KL vs recon-loss
# https://stats.stackexchange.com/questions/368001/is-the-output-of-a-variational-autoencoder-meant-to-be-a-distribution-that-can-b

def compute_loss(model, x, detailed=False, kl_nan_offset=1e-18):
    """
    Args:

        model          the model
        x              the data
        detailed       set to true if you want the separate losses to be returned as well, basically a debug mode
        kl_nan_offset  the kicker, can lead to NaN errors otherwise (don't ask me how long it took to find this)
                       value was found empirically
    """
    mean, var = model.encode(x)
    z = model.reparameterize(mean, var)
    x_logit = model.decode(z)

    #if run_config['use_time_series']:
    #    # Note, the model is trained to reconstruct only the last, most current time step (by taking the last entry in the timeseries)
    #    # this works on classification data (or binary data)
    #    #cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x[:, -1, :])
    #    recon_loss = tf.losses.mean_squared_error(predictions=x_logit, labels=x[:, -1, :])
    #else:
    #    recon_loss = tf.losses.mean_squared_error(predictions=x_logit, labels=x)

    # Putting more weight on the most current time epochs
    recon_loss = tf.losses.mean_squared_error(predictions=x_logit,
                                          labels=x,
                                          weights=np.exp(np.linspace(0, 1, num=x.shape[1]))\
                                                    .reshape((1, x.shape[1], 1)))

    # Checkout https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    # https://arxiv.org/pdf/1606.00704.pdf
    #  Adversarially Learned Inference
    #  Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Olivier Mastropietro1,Alex Lamb1,Martin Arjovsky3Aaron Courville1

    # This small constant offset prevents Nan errors
    p = tfp.distributions.Normal(loc=tf.zeros_like(mean) + tf.constant(kl_nan_offset), scale=tf.ones_like(var) + tf.constant(kl_nan_offset))
    q = tfp.distributions.Normal(loc=mean + tf.constant(kl_nan_offset), scale=var + tf.constant(kl_nan_offset))
    try:
        # the KL loss can explode easily, this is to prevent overflow errors
        kl = tf.reduce_mean(tf.clip_by_value(tfp.distributions.kl_divergence(p, q, allow_nan_stats=True), 0., 1e32))
    except (_NotOkStatusException, InfOrNanError) as e:
        print('Error with KL-loss: ', e, tf.reduce_mean(var))
        kl = 1.

    if not detailed:
        kl = tf.clip_by_value(kl, 0., 1.)

    if model._loss_weight_kl == 0.:
        loss = model._loss_weight_reconstruction*recon_loss
    else:
        # KL loss can be NaN for some data. This is inherit to KL-loss (but the data is probably more to blame)
        loss = model._loss_weight_reconstruction*recon_loss + model._loss_weight_kl*kl

    if detailed:
        return loss, recon_loss, kl
    else:
        return loss
