import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


def make_inference_net(model_to_wrap, input_shape, batch_size, activation=tf.nn.softplus):
    """This is basically a wrapper function. Define your model up to the split into μ and σ.
    Which is what this function will do.

    Depending on your interpretation of how a VAE should do it (depends mostly on your reparametrisation trick),
    you can provide an activation function.

    E.g. if you think σ should be the variance use `tf.nn.softplus` to force it to be positive.
    if you think it should be the deviation, then provide `None`.
    """

    x = tfk.Input(shape=input_shape, batch_size=batch_size)
    enc = model_to_wrap(x)
    mean, var = tf.split(enc, num_or_size_splits=2, axis=-1)

    if activation is not None:
        var = tfkl.Activation(activation)(var)

    return tfk.Model(x, [mean, var])
