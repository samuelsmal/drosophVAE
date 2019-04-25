import tensorflow as tf


_TF_DEFAULT_SESSION_CONFIG_ = tf.ConfigProto()
_TF_DEFAULT_SESSION_CONFIG_.gpu_options.allow_growth = True 
_TF_DEFAULT_SESSION_CONFIG_.gpu_options.polling_inactive_delay_msecs = 10
