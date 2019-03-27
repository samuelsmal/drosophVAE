# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # constants

# <codecell>

__NB_DIMS__ = 2

LEGS = [0, 1, 2, 5, 6, 7]
LEGS = [0, 1, 2] #, 5, 6, 7] # since we do not care about the other side
CAMERA_OF_INTEREST = 1
NB_OF_AXIS = 2
NB_TRACKED_POINTS = 5 # per leg, igoring the rest for now
NB_CAMERAS = 7
NB_RECORDED_DIMESIONS = 2

FRAMES_PER_SECOND = 100
NYQUIST_FREQUENCY_OF_MEASUREMENTS = FRAMES_PER_SECOND / 2

FLY = "181220_Rpr_R57C10_GC6s_tdTom"

POSE_DATA_PATH = "/ramdya-nas/SVB/{fly_id}/001_coronal/behData/images_renamed/pose_result__mnt_NAS_SVB_181220_Rpr_R57C10_GC6s_tdTom_001_coronal_behData_images_renamed.pkl".format(fly_id=FLY)
POSE_FRAME_PATH = "/ramdya-nas/SVB/{fly_id}/001_coronal/behData/images_renamed/camera_{{camera_id}}_img_{{frame_id:06d}}.jpg".format(fly_id=FLY)

SEQUENCE_GIF_PATH = "/home/samuel/Videos/{fly_id}/sequence_gif_{{begin_frame}}-{{end_frame}}.gif".format(fly_id=FLY)

# <markdowncell>

# # imports & general functions

# <markdowncell>

# ## imports

# <codecell>

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd

from sklearn.metrics import mean_squared_error
from tqdm import tqdm, trange
import pathlib
import logging
import datetime
from datetime import date
import os
import uuid
from glob import glob
import shutil
import pickle
import skimage
from functools import reduce
from skimage import io
import matplotlib.pyplot as plt
import seaborn as sns

# for creating the gifs
import PIL
import imageio
import cv2
from IPython import display

#%matplotlib inline

from importlib import reload
import inspect
import sys
sys.path.append('/home/samuel/')

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from drosophpose.GUI import skeleton

from som_vae import somvae_model
from som_vae.utils import *

# <markdowncell>

# ## general helpers

# <codecell>

def extract_args(config, function):
    return {k:config[k] for k in inspect.getfullargspec(function).args if k in config}

# <codecell>

def fix_layout(width:int=95):
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:" + str(width) + "% !important; }</style>"))
    
fix_layout()

# <markdowncell>

# ## data loading helpers

# <codecell>

def load_data(path=POSE_DATA_PATH):
    with open(POSE_DATA_PATH, 'rb') as f:
        pose_data_raw = pickle.load(f)
        return pose_data_raw['points2d']

# <codecell>

def _check_shape_(joint_positions):
    """ should be (7, <nb frames>, 38, 2)
    7 for the images, some should be 0 because it didn't record the images for these points
    1000 for the nb of frames
    38 for the features (some for the legs ...) check skeleton.py in semigh's code
    2 for the pose dimensions
    """
    s = joint_positions.shape
    
    if s[0] != NB_CAMERAS or s[2] != len(skeleton.tracked_points) or s[3] != NB_RECORDED_DIMESIONS:
        raise ValueError(f"shape of pose data is wrong, it's {joint_positions.shape}")
        
    return joint_positions 

def _crude_value_check_(joint_positions):
    if np.sum(joint_positions == 0) == np.product(joint_positions.shape):
        raise ValueError('not every value should be zero')
        
    return joint_positions

def _simple_checks_(data):
    return reduce(lambda acc, el: el(acc), [_check_shape_, _crude_value_check_], data)

# <codecell>

def _get_camera_of_interest_(joint_positions, camera_idx=CAMERA_OF_INTEREST):
    return joint_positions[CAMERA_OF_INTEREST]

def _get_visible_legs_(joint_positions, camera_idx=CAMERA_OF_INTEREST):
    idx_visible_joints = [skeleton.camera_see_joint(CAMERA_OF_INTEREST, j) for j in range(len(skeleton.tracked_points))]
    return joint_positions[:, idx_visible_joints, :]

# <codecell>

def get_data(path=POSE_DATA_PATH):
    return reduce(lambda acc, el: el(acc), [load_data, _simple_checks_, _get_camera_of_interest_, _get_visible_legs_], path)

# <codecell>

def add_third_dimension(joint_positions):
    # just add a z-axis
    # look up np.pad...
    # assumes that the positional data is in the last axis
    paddings = [[0, 0] for i in joint_positions.shape]
    paddings[-1][1] = 1

    return np.pad(joint_positions, paddings, mode='constant', constant_values=0)

# <markdowncell>

# ## plotting helpers

# <codecell>

def _get_feature_name_(tracking_id):
    return str(skeleton.tracked_points[tracking_id])[len('Tracked.'):]

def _get_feature_id_(leg_id, tracking_point_id):
    if leg_id < 3:
        return leg_id * 5 + tracking_point_id
    else:
        return (leg_id - 5) * 5 + tracking_point_id + 19
    
def _get_leg_name_(leg_id):
    __LEG_NAMES__ = ['foreleg', 'middle leg', 'hind leg']
    return __LEG_NAMES__[leg_id]

def ploting_frames(joint_positions):
    for leg in LEGS:
        fig, axs = plt.subplots(1, NB_OF_AXIS, sharex=True, figsize=(20, 10))
        for tracked_point in range(NB_TRACKED_POINTS):
            for axis in range(NB_OF_AXIS):
                cur_ax = axs[axis]
                cur_ax.plot(joint_positions[:, _get_feature_id_(leg, tracked_point),  axis], label = f"{_get_feature_name_(tracked_point)}_{('x' if axis == 0 else 'y')}")
                if axis == 0:
                    cur_ax.set_ylabel('x pos')
                else:
                    cur_ax.set_ylabel('y pos')
                cur_ax.legend(loc='upper right')
                cur_ax.set_xlabel('frame')

        #plt.xlabel('frame')
        #plt.legend(loc='lower right')
        plt.suptitle(_get_leg_name_(leg))

# <codecell>

def plot_comparing_joint_position_with_reconstructed(real_joint_positions, reconstructed_joint_positions, validation_cut_off=None):
    for leg in LEGS:
        fig, axs = plt.subplots(1, NB_OF_AXIS * 2, sharex=True, figsize=(25, 10))
        for axis in range(NB_OF_AXIS):
            cur_ax = axs[axis * 2]
            rec_ax = axs[axis * 2 + 1]
            
            if validation_cut_off is not None:
                for a in [cur_ax, rec_ax]:
                    a.axvline(validation_cut_off, label='validation cut off', linestyle='--')
                    
            for tracked_point in range(NB_TRACKED_POINTS):
                cur_ax.plot(joint_positions[:, _get_feature_id_(leg, tracked_point),  axis], label = f"{_get_feature_name_(tracked_point)}_{('x' if axis == 0 else 'y')}")
                rec_ax.plot(reconstructed_joint_positions[:, _get_feature_id_(leg, tracked_point),  axis], label = f"{_get_feature_name_(tracked_point)}_{('x' if axis == 0 else 'y')}")
                cur_ax.get_shared_y_axes().join(cur_ax, rec_ax)
                if axis == 0:
                    cur_ax.set_ylabel('x pos')
                    rec_ax.set_ylabel('x pos')
                else:
                    cur_ax.set_ylabel('y pos')
                    rec_ax.set_ylabel('y pos')
                cur_ax.legend(loc='upper right')
                cur_ax.set_xlabel('frame')
                rec_ax.legend(loc='upper right')
                rec_ax.set_xlabel('frame')
                cur_ax.set_title('original data')
                rec_ax.set_title('reconstructed data')
                

                
        #plt.xlabel('frame')
        #plt.legend(loc='lower right')
        plt.suptitle(_get_leg_name_(leg))

# <codecell>

def plot_losses(losses, legend=None):
    plt.figure(figsize=(15, 8))
    if legend is None:
        legend = ['train', 'test', 'test_recon'] 
    plt.figure()
    for l in losses:
        plt.plot(l)

    plt.legend(legend)
    plt.xlabel('epoch')
    plt.title('loss')

# <codecell>

def plot_latent_frame_distribution(latent_assignments, nb_bins):
    plt.figure()
    plt.hist(latent_assignments, bins=nb_bins)
    plt.title('distribution of latent-space-assignments')
    plt.xlabel('latent-space')
    plt.ylabel('nb of frames in latent-space')

# <codecell>

def plot_cluster_assignment_over_time(cluster_assignments):
    plt.figure()
    plt.plot(cluster_assignments)
    plt.title("cluster assignments over time")
    plt.ylabel("index of SOM-embeddings")
    plt.xlabel("frame")

# <markdowncell>

# ## gif helpers

# <codecell>

def group_by_cluster(data):
    """Returns the lengths of sequences.
    Example: AABAAAA -> [[0, 1], [2], [3, 4, 5], [6, 7]]
    
    """
    sequences = []
    cur_embedding_idx = 0
    cur_seq = [0]
    for i in range(len(data))[1:]:
        if data[i] == data[cur_embedding_idx]:
            cur_seq += [i]
        else:
            sequences += [cur_seq]
            cur_embedding_idx = i
            cur_seq = [i]
            
    sequences += [cur_seq]
            
    return sequences

def get_frame_path(frame_id, path=POSE_FRAME_PATH, camera_id=CAMERA_OF_INTEREST):
    return path.format(camera_id=camera_id, frame_id=frame_id)


def create_gif_of_sequence(sequence, file_name=None):
    if file_name is None:
        gif_file_path = SEQUENCE_GIF_PATH.format(begin_frame=sequence[0], end_frame=sequence[-1])
    else:
        gif_file_path = file_name

    pathlib.Path(gif_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with imageio.get_writer(gif_file_path, mode='I') as writer:
        filenames =  [(get_frame_path(i), i) for i in sequence]
        last = -1
        last_frame_id = filenames[0][1] - 1
        for i, (filename, frame_id) in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
                
            # adding fr nb to image
            image = cv2.imread(filename)  
            if last_frame_id + 1 != frame_id:
                color = (0, 255, 255)
            else:
                color = (255, 255, 255)
                
            last_frame_id = frame_id
                
            image = cv2.putText(img=np.copy(image), text=str(frame_id), org=(0, image.shape[0] // 2),fontFace=2, fontScale=3, color=color, thickness=2)
            #image = imageio.imread(filename)
            writer.append_data(image)

        image = imageio.imread(filename)
        writer.append_data(image)
    
    return gif_file_path

def video_with_embedding(embeddings, file_path=None):
    """Creates a video (saved as a gif) with the embedding overlay, displayed as an int.
    
    Args:
        embeddings: [<embeddings_id>]    
            assumed to be in sequence with `get_frame_path` function.
        file_path: <str>, default: SEQUENCE_GIF_PATH  
            file path used to get 
    Returns:
        <str>                            the file path under which the gif was saved
    """
    if file_path is None:
        gif_file_path = SEQUENCE_GIF_PATH.format(begin_frame="full-video", end_frame="with-embeddings")
    else:
        gif_file_path = file_path

    pathlib.Path(gif_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with imageio.get_writer(gif_file_path, mode='I') as writer:
        filenames =  [(get_frame_path(i), emb_id) for i, emb_id in enumerate(embeddings)]
        last = -1
        for i, (filename, emb_id) in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
                
            # adding fr nb to image
            image = cv2.imread(filename)  
            image = cv2.putText(img=np.copy(image), text=f"{emb_id:0>3}", org=(0, image.shape[0] // 2),fontFace=2, fontScale=3, color=(255, 255, 255), thickness=2)
            image = cv2.putText(img=np.copy(image), text=f"fr: {i:0>4}", org=(0, (image.shape[0] // 2) + 24),fontFace=1, fontScale=2, color=(255, 255, 255), thickness=2)
            writer.append_data(image)

        image = imageio.imread(filename)
        writer.append_data(image)
    
    return gif_file_path


def create_gifs_for_clusters(cluster_assignments, up_to_n_clusters=10):
    """
    Args:
        cluster_assignments: list of assignments for each frame
    Returns:
        file paths of the created gifs
    """
    sequences = sorted(group_by_cluster(cluster_assignments), key=len, reverse=True)

    return [create_gif_of_sequence(s) for s in sequences[:up_to_n_clusters]]

# <markdowncell>

# # data loading

# <codecell>

if __NB_DIMS__ == 3:
    joint_positions = add_third_dimension(get_data())
else:
    joint_positions = add_third_dimension(get_data())
    

NB_FRAMES = joint_positions.shape[1]

# <codecell>

for leg in LEGS:
    print("{0:.3}% of the data for leg {1} is 0".format((joint_positions[:, leg:leg+5, :2] == 0).mean(), leg))

# <codecell>

ploting_frames(joint_positions)

# <codecell>

def normalize(joint_positions, using_median=True, to_probability_distr=False):
    # alternatives could be to use only the median of the first joint -> data is then fixed to top (is that differnt to now?)
    if using_median:
        return joint_positions - np.median(joint_positions.reshape(-1, 3), axis=0)
    elif to_probability_distr:
        return
    else:
        raise NotImplementedError

# <codecell>

def normalize_ts(time_series, ax=0):
    # for shape (frame,feat)
    eps = 0.0001
    print("shapes:", np.shape(np.transpose(time_series)), np.shape(np.mean(np.transpose(time_series), axis=ax)))
#     n_time_series = (np.transpose(time_series) - np.mean(np.transpose(time_series), axis=ax))/(np.std(np.transpose(time_series), axis=ax) + eps)
    norm = np.sum(np.transpose(time_series), axis=ax); norm = np.transpose(norm) #shape = 1,frames
    n_time_series = np.transpose(time_series) / np.sum(np.transpose(time_series), axis=ax)
    n_time_series = np.transpose(n_time_series)
#     n_time_series = np.zeros(shape=np.shape(time_series))
#     for i in range(np.shape(time_series)[1]):
#         n_time_series[:,i] = (time_series[:,i] - np.mean(time_series[:,i])) / (np.std(time_series[:,i]) + eps)
    return n_time_series, norm


def normalize_pose(points3d, median3d=False):
    # normalize experiment
    if median3d:
        points3d -= np.median(points3d.reshape(-1, 3), axis=0)
    else:
        for i in range(np.shape(points3d)[1]): #frames
            for j in range(np.shape(points3d)[2]): #xyz
                points3d[:,i,j] = normalize_ts(points3d[:,i,j]) 
    return points3d

# <codecell>

joint_positions = normalize(joint_positions)

# <codecell>

ploting_frames(joint_positions)

# <markdowncell>

# # SOM-VAE model

# <markdowncell>

# ## functions

# <codecell>

def get_data_generator(data_train, labels_train, data_val, labels_val, time_series):
    """Creates a data generator for the training.
    
    Args:
        time_series (bool): Indicates whether or not we want interpolated MNIST time series or just
            normal MNIST batches.
    
    Returns:
        generator: Data generator for the batches."""

    def batch_generator(mode="train", batch_size=100):
        """Generator for the data batches.
        
        Args:
            mode (str): Mode in ['train', 'val'] that decides which data set the generator
                samples from (default: 'train').
            batch_size (int): The size of the batches (default: 100).
            
        Yields:
            np.array: Data batch.
        """
        assert mode in ["train", "val"], "The mode should be in {train, val}."
        if mode=="train":
            images = data_train.copy()
            labels = labels_train.copy()
        elif mode=="val":
            images = data_val.copy()
            labels = labels_val.copy()
        
        while True:
            indices = np.random.permutation(np.arange(len(images)))
            images = images[indices]
            labels = labels[indices]

            if time_series:
                for i, image in enumerate(images):
                    start_image = image
                    end_image = images[np.random.choice(np.where(labels == (labels[i] + 1) % 10)[0])]
                    interpolation = interpolate_arrays(start_image, end_image, batch_size)
                    yield interpolation + np.random.normal(scale=0.01, size=interpolation.shape)
            else:
                for i in range(len(images)//batch_size):
                    yield images[i*batch_size:(i+1)*batch_size]

    return batch_generator

# <codecell>

def train_model(model, x, lr_val, num_epochs, patience, batch_size, logdir,
        modelpath, learning_rate, interactive, generator):
    """Trains the SOM-VAE model.
    
    Args:
        model (SOM-VAE): SOM-VAE model to train.
        x (tf.Tensor): Input tensor or placeholder.
        lr_val (tf.Tensor): Placeholder for the learning rate value.
        num_epochs (int): Number of epochs to train.
        patience (int): Patience parameter for the early stopping.
        batch_size (int): Batch size for the training generator.
        logdir (path): Directory for saving the logs.
        modelpath (path): Path for saving the model checkpoints.
        learning_rate (float): Learning rate for the optimization.
        interactive (bool): Indicator if we want to have an interactive
            progress bar for training.
        generator (generator): Generator for the data batches.
    """
    train_gen = generator("train", batch_size)
    val_gen = generator("val", batch_size)

    num_batches = len(data_train)//batch_size

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.)
    summaries = tf.summary.merge_all()
    
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True  
    session_config.gpu_options.polling_inactive_delay_msecs = 10
    
    with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        patience_count = 0
        train_losses = []
        test_losses = []
        test_losses_reconstrution = []
        train_writer = tf.summary.FileWriter(logdir+"/train", sess.graph)
        test_writer = tf.summary.FileWriter(logdir+"/test", sess.graph)
        print("Training...")
        train_step_SOMVAE, train_step_prob = model.optimize
        try:
            if interactive:
                pbar = tqdm(total=num_epochs*(num_batches)) 
            for epoch in range(num_epochs):
                batch_val = next(val_gen)
                test_loss, summary, test_loss_reconstruction = sess.run([model.loss, summaries, model.loss_reconstruction], feed_dict={x: batch_val})
                test_losses.append(test_loss)
                test_losses_reconstrution.append(test_loss_reconstruction)
                test_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                if test_losses[-1] == min(test_losses):
                    saver.save(sess, modelpath, global_step=epoch)
                    patience_count = 0
                else:
                    patience_count += 1
                if patience_count >= patience:
                    break
                for i in range(num_batches):
                    batch_data = next(train_gen)
                    if i%100 == 0:
                        train_loss, summary = sess.run([model.loss, summaries], feed_dict={x: batch_data})
                        train_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                        train_losses += [train_loss]
                    train_step_SOMVAE.run(feed_dict={x: batch_data, lr_val:learning_rate})
                    train_step_prob.run(feed_dict={x: batch_data, lr_val:learning_rate*100})
                    if interactive:
                        pbar.set_postfix(epoch=epoch, train_loss=train_loss, test_loss=test_loss, refresh=False)
                        pbar.update(1)

        except KeyboardInterrupt:
            pass
        finally:
            saver.save(sess, modelpath)
            if interactive:
                pbar.close()
                
    return test_losses, train_losses, test_losses_reconstrution

# <codecell>

def evaluate_model(model, x, modelpath, batch_size, data, labels):
    """Evaluates the performance of the trained model in terms of normalized
    mutual information, purity and mean squared error.
    
    Args:
        model (SOM-VAE): Trained SOM-VAE model to evaluate.
        x (tf.Tensor): Input tensor or placeholder.
        modelpath (path): Path from which to restore the model.
        batch_size (int): Batch size for the evaluation.
        
    Returns:
        dict: Dictionary of evaluation results (NMI, Purity, MSE).
        x hat, reconstructed data
        cluster assignments for each row
        encoding of x
    """
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.)
    
    
    num_batches = len(data)//batch_size
    
    # TODO this belongs into the global config handling
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True 
    session_config.gpu_options.polling_inactive_delay_msecs = 10
    
    with tf.Session(config=session_config) as sess:
        
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, modelpath)

        k_all = []
        x_hat_embedding = []
        x_hat_encoding = []
        test_mse_all = []
        print("Evaluation...")
        for i in range(num_batches):
            batch_data = data[i*batch_size:(i+1)*batch_size]
            _k, x_embedding, x_encoding = sess.run([model.k, model.x_hat_embedding, model.x_hat_encoding], feed_dict={x: batch_data})
            k_all.extend(_k)
            x_hat_embedding.extend(x_embedding)
            x_hat_encoding.extend(x_encoding)
            
            
            # is it encoding or embedding?
            test_mse_all.append(mean_squared_error(x_encoding.flatten(), batch_data.flatten()))

        test_nmi = compute_NMI(k_all, labels[:len(k_all)])
        test_purity = compute_purity(k_all, labels[:len(k_all)])
        test_mse = np.mean(test_mse_all)

    results = {}
    results["NMI"] = test_nmi
    results["Purity"] = test_purity
    results["MSE"] = test_mse
#    results["optimization_target"] = 1 - test_nmi

    return results, x_hat_embedding, k_all, x_hat_encoding

# <codecell>

def main(X_train, X_val, y_train, y_val, latent_dim, som_dim, learning_rate, decay_factor, alpha, beta, gamma, tau, modelpath, save_model, mnist, time_series, config):
    """Main method to build a model, train it and evaluate it.
    
    Args:
        latent_dim (int): Dimensionality of the SOM-VAE's latent space.
        som_dim (list): Dimensionality of the SOM.
        learning_rate (float): Learning rate for the training.
        decay_factor (float): Factor for the learning rate decay.
        alpha (float): Weight for the commitment loss.
        beta (float): Weight for the SOM loss.
        gamma (float): Weight for the transition probability loss.
        tau (float): Weight for the smoothness loss.
        modelpath (path): Path for the model checkpoints.
        save_model (bool): Indicates if the model should be saved after training and evaluation.
        
    Returns:
        dict: Results of the evaluation (NMI, Purity, MSE).
    """
    print(f"running with config: {config}")
    # Dimensions for MNIST-like data
    #input_length = 28
    #input_channels = 28
    #x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    input_length = 1
    input_channels = 19 * __NB_DIMS__
    x = tf.placeholder(tf.float32, shape=[None, input_channels])
    #x = tf.placeholder(tf.float32, shape=[None, input_length, input_channels, 1])
    #x = tf.placeholder(tf.float32, shape=[None, input_length, 1, input_channels])
    data_generator = get_data_generator(data_train=X_train, data_val=X_val, labels_train=y_train, labels_val=y_val,time_series=time_series)

    lr_val = tf.placeholder_with_default(learning_rate, [])

    model = somvae_model.SOMVAE(inputs=x, latent_dim=latent_dim, som_dim=som_dim, learning_rate=lr_val, decay_factor=decay_factor,
            input_length=input_length, input_channels=input_channels, alpha=alpha, beta=beta, gamma=gamma,
            tau=tau, mnist=mnist)

    test_losses, train_losses, test_losses_reconstruction = train_model(model, x, lr_val, generator=data_generator, **extract_args(config, train_model))

    result = evaluate_model(model, x, data=X_train, labels=y_train, **extract_args(config, evaluate_model))
    result_val = evaluate_model(model, x, data=X_val, labels=y_val, **extract_args(config, evaluate_model))
    

    if not save_model:
        shutil.rmtree(os.path.dirname(modelpath))
        
    print(f"got: {result[0]}")

    return result, model, (train_losses, test_losses, test_losses_reconstruction), result_val

# <markdowncell>

# ## model config

# <codecell>

## config
"""
Params:
    num_epochs (int): Number of training epochs.
    patience (int): Patience for the early stopping.
    batch_size (int): Batch size for the training.
    latent_dim (int): Dimensionality of the SOM-VAE's latent space.
    som_dim (list): Dimensionality of the self-organizing map.
    learning_rate (float): Learning rate for the optimization.
    alpha (float): Weight for the commitment loss.
    beta (float): Weight for the SOM loss.
    gamma (float): Weight for the transition probability loss.
    tau (float): Weight for the smoothness loss.
    decay_factor (float): Factor for the learning rate decay.
    name (string): Name of the experiment.
    ex_name (string): Unique name of this particular run.
    logdir (path): Directory for the experiment logs.
    modelpath (path): Path for the model checkpoints.
    interactive (bool): Indicator if there should be an interactive progress bar for the training.
    data_set (string): Data set for the training.
    save_model (bool): Indicator if the model checkpoints should be kept after training and evaluation.
    time_series (bool): Indicator if the model should be trained on linearly interpolated
        MNIST time series.
    mnist (bool): Indicator if the model is trained on MNIST-like data.
"""
__name__ = "tryouts"
__latent_dim__ = 64
__som_dim__ = [8,8]
__ex_name__ = "{}_{}_{}-{}_{}_{}".format(__name__, __latent_dim__, __som_dim__[0], __som_dim__[1], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), uuid.uuid4().hex[:5])

config = {
    "num_epochs": 200,
    "patience": 100,
    "batch_size": 50, # len(joint_positions), # if time_series then each batch should be a time series
    "latent_dim": __latent_dim__,
    "som_dim": __som_dim__,
    "learning_rate": 0.0005,
    "alpha": 0.0, #1.0,
    "beta": 0.0, #0.9,
    "gamma": 0.0, #1.8,
    "tau": 0.0, # 1.4,
    "decay_factor": 0.9,
    "name": __name__,
    "ex_name": __ex_name__,
    "logdir": "../logs/{}".format(__ex_name__),
    "modelpath": "../models/{0}/{0}.ckpt".format(__ex_name__),
    "interactive": True, # this is just for the progress bar
    "data_set": "MNIST_data",
    "save_model": False,
    "time_series": False,
    "mnist": False,
}

# <codecell>

# creating path to store model
pathlib.Path(config['modelpath']).parent.mkdir(parents=True, exist_ok=True)

# <markdowncell>

# ## data processing

# <codecell>

# reshaping the data, the selection is there to be sure
reshaped_joint_position = joint_positions[:,:,:__NB_DIMS__].reshape(-1, 19 * __NB_DIMS__)

# scaling the data to be in [0, 1]
# this is due to the sigmoid activation function in the reconstruction
scaler = MinMaxScaler()
#resh = scaler.fit_transform(resh)

# <codecell>

joint_positions.shape

# <codecell>

data_test.shape

# <codecell>

#nb_of_data_points = (reshaped_joint_position.shape[0] // config['batch_size']) * config['batch_size']
nb_of_data_points = int(joint_positions.shape[0] * 0.7)

data_train = scaler.fit_transform(reshaped_joint_position[:nb_of_data_points])
data_test = scaler.transform(reshaped_joint_position[nb_of_data_points:])
# just generating some labels, no clue what they are for except validation?
labels = np.array(list(range(reshaped_joint_position.shape[0])))

data = {
  "X_train": data_train,
  "X_val": data_test,
  "y_train": labels[:nb_of_data_points],
  "y_val": labels[nb_of_data_points:]
}

#data = {
#  "X_val": data_train,
#  "y_val": labels,
#  "X_train": data_train,
#  "y_train": labels
#}

# <markdowncell>

# ## running fit & test

# <codecell>

reload(somvae_model)

tf.reset_default_graph()

main_args = inspect.getfullargspec(main).args
res, mdl, losses, res_val = main(**{**{k:config[k] for k in main_args if k in config}, **data, **{"config": config}})

reconstructed_from_encoding =  scaler.inverse_transform(res[3]).reshape(-1, 19, 2)
reconstructed_from_encoding_val = scaler.inverse_transform(res_val[3]).reshape(-1, 19, 2)

# <codecell>

plot_losses(losses)
plot_latent_frame_distribution(res[2], nb_bins=__latent_dim__)
plot_cluster_assignment_over_time(res[2])

# <codecell>

plot_comparing_joint_position_with_reconstructed(joint_positions, 
                                                 np.vstack((reconstructed_from_encoding, reconstructed_from_encoding_val)), validation_cut_off=nb_of_data_points)

# <codecell>

((joint_positions[:len(res[3])] - reconstructed_from_encoding) ** 2).mean()

# <codecell>



# <codecell>



# <markdowncell>

# ## cool gifs

# <markdowncell>

# ```
# # use this to display a gif inside the notebook, no idea why a function doesn't work
# # this is a hack to display the gif inside the notebook
# os.system('cp {0} {0}.png'.format(path))
# display.Image(filename=f"{path}.png")
# ``` 

# <codecell>

def gif_with_x(x, embeddings, file_path=None):
    """Creates a video (saved as a gif) with the embedding overlay, displayed as an int.
    
    Args:
        embeddings: [<embeddings_id>]    
            assumed to be in sequence with `get_frame_path` function.
        file_path: <str>, default: SEQUENCE_GIF_PATH  
            file path used to get 
    Returns:
        <str>                            the file path under which the gif was saved
    """
    if file_path is None:
        gif_file_path = SEQUENCE_GIF_PATH.format(begin_frame="full-video", end_frame="with-embeddings")
    else:
        gif_file_path = file_path

    pathlib.Path(gif_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with imageio.get_writer(gif_file_path, mode='I') as writer:
        filenames =  [(get_frame_path(i), emb_id) for i, emb_id in enumerate(embeddings)]
        last = -1
        for i, (filename, emb_id) in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
                
            image = cv2.imread(filename)  
            
            for p in x[i]:
                cv2.circle(image, tuple(p.astype('int').tolist()), radius=3, thickness=-1, color=(0,255,0)) 
                
            cv2.line(image, (0, emb_id), (10, emb_id), (0, 255, 0), 2)
            image = cv2.putText(img=np.copy(image), text=f"{emb_id:0>3}", org=(0, image.shape[0] // 2),fontFace=2, fontScale=3, color=(255, 255, 255), thickness=2)
            image = cv2.putText(img=np.copy(image), text=f"fr: {i:0>4}", org=(0, (image.shape[0] // 2) + 24),fontFace=1, fontScale=2, color=(255, 255, 255), thickness=2)
            writer.append_data(image)

        image = imageio.imread(filename)
        writer.append_data(image)
    
    return gif_file_path

# <codecell>



# <codecell>

x = reconstructed_from_encoding
i = 10
image = cv2.imread(get_frame_path(i))  

for p in x[i]:
    cv2.circle(image, tuple((p + np.array()).astype('int').tolist()), radius=3, thickness=-1, color=(0,255,0)) 

# <codecell>

plt.imshow(image)

# <codecell>

# change to 
_p = gif_with_x(x=reconstructed_from_encoding, embeddings=res[2])

os.system('cp {0} {0}.png'.format(_p))
display.Image(filename="{}.png".format(_p))

# <codecell>

cluster_clips = create_gifs_for_clusters(res[2])
full_clip = video_with_embedding(res[2])

# <codecell>

os.system('cp {0} {0}.png'.format(full_clip))
display.Image(filename="{}.png".format(full_clip))

# <codecell>

idx = 0
os.system('cp {0} {0}.png'.format(cluster_clips[idx]))
display.Image(filename="{}.png".format(cluster_clips[idx]))


# <codecell>

idx = 1
os.system('cp {0} {0}.png'.format(gif_file_paths[idx]))
display.Image(filename="{}.png".format(gif_file_paths[idx]))

# <codecell>

idx = 2
os.system('cp {0} {0}.png'.format(gif_file_paths[idx]))
display.Image(filename="{}.png".format(gif_file_paths[idx]))

# <codecell>

idx = 3
os.system('cp {0} {0}.png'.format(gif_file_paths[idx]))
display.Image(filename="{}.png".format(gif_file_paths[idx]));
