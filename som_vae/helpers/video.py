import warnings
from itertools import groupby
import pathlib
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import imageio
import colorsys
import seaborn as sns

from PIL import Image
from som_vae.helpers.misc import flatten, is_file
from som_vae.settings import config, skeleton, data
from som_vae.settings.config import SetupConfig


__FRAME_ACTIVE_COLOUR__ = (255, 0, 0)
__FRAME_BAR_MARKER__ = (0, 255, 255)



def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Taken from here: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib#answer-49601444

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def lighten_int_colors(cs, amount=0.5):
    return [(np.array(lighten_color(np.array(c) / 255, amount=amount)) * 255).astype(np.int).tolist() for c in cs]


def plot_drosophila_2d(pts=None, draw_joints=None, img=None, colors=None, thickness=None,
                       draw_limbs=None, circle_color=None):
    """
    taken from https://github.com/NeLy-EPFL/drosoph3D/blob/master/GUI/plot_util.py
    """
    if colors is None:
        colors = skeleton.colors
    if thickness is None:
        thickness = [2] * 10
    if draw_joints is None:
        draw_joints = np.arange(skeleton.num_joints)
    if draw_limbs is None:
        draw_limbs = np.arange(skeleton.num_limbs)
    for joint_id in range(pts.shape[0]):
        limb_id = skeleton.get_limb_id(joint_id)
        if (pts[joint_id, 0] == 0 and pts[joint_id, 1] == 0) or limb_id not in draw_limbs or joint_id not in draw_joints:
            continue

        color = colors[limb_id]
        r = 5 if joint_id != skeleton.num_joints - 1 and joint_id != ((skeleton.num_joints // 2) - 1) else 8
        cv2.circle(img, (pts[joint_id, 0], pts[joint_id, 1]), r, color, -1)

        # TODO replace this with skeleton.bones
        if (not skeleton.is_tarsus_tip(joint_id)) and (not skeleton.is_antenna(
                joint_id)) and (joint_id != skeleton.num_joints - 1) and (
                joint_id != (skeleton.num_joints // 2 - 1)) and (not (
                pts[joint_id + 1, 0] == 0 and pts[joint_id + 1, 1] == 0)):
            cv2.line(img, (pts[joint_id][0], pts[joint_id][1]), (pts[joint_id + 1][0], pts[joint_id + 1][1]),
                     color=color,
                     thickness=thickness[limb_id])

    if circle_color is not None:
        img = cv2.circle(img=img, center=(img.shape[1]-20, 20), radius=10, color=circle_color, thickness=-1)

    return img


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
            sequences += [(data[cur_embedding_idx], cur_seq)]
            cur_embedding_idx = i
            cur_seq = [i]

    sequences += [(data[cur_embedding_idx], cur_seq)]

    return {embedding_id: [el[1] for el in emb_frames] for embedding_id, emb_frames in groupby(sorted(sequences, key=lambda x: x[0]), key=lambda x: x[0])}


def get_frame_path(frame_id, path, camera_id):
    logging.warn('this is needs to be adapted!')
    return path.format(camera_id=camera_id, frame_id=frame_id)


#def _get_and_check_file_path_(args, template=SetupConfig.value('video_root_path')):
#    gif_file_path = template.format(begin_frame=args[0], end_frame=args[-1])
#    pathlib.Path(gif_file_path).parent.mkdir(parents=True, exist_ok=True)
#
#    return gif_file_path


def _save_frames_(file_path, frames, format='mp4', **kwargs):
    """
    If format==GIF then fps has to be None, duration should be ~10/60
    If format==mp4 then duration has to be None, fps should be TODO
    """
    if format.lower() == 'gif':
        _kwargs = {'duration': 10/60}
    elif format.lower() == 'mp4':
        _kwargs = {'fps': 24}

    pathlib.Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16')
        imageio.mimsave(file_path, frames, format=format, **{**_kwargs, **kwargs})


def _add_frame_and_embedding_id_(frame, emb_id=None, frame_id=None):
    params = {"org": (0, frame.shape[0] // 2),
              "fontFace": 1,
              "fontScale": 2,
              "color": (255, 255, 255),
              "thickness": 2}

    if emb_id is not None:
       frame = cv2.putText(img=np.copy(frame), text=f"cluster_id: {emb_id:0>3}", **params)

    if frame_id is not None:
       frame = cv2.putText(img=np.copy(frame), text=f"frame_id: {frame_id:0>4}", **{**params, 'org': (params['org'][0], params['org'][1] + 24)})

    return frame


def _float_to_int_color_(colors):
    return (np.array(colors) * 255).astype(np.int).tolist()

def comparision_video_of_reconstruction(positional_data, cluster_assignments, image_id_with_exp, labels,
                                        n_train_data_points, images_paths, cluster_colors=None,
                                        run_desc=None, epochs=None):
    """Creates a video (saved as a gif) with the embedding overlay, displayed as an int.

    Args:
        xs: [<pos data>] list of pos data, of shape: [frames, limb, dimensions] (can be just one, but in an array)
            will plot all of them, the colors get lighter
        embeddings: [<embeddings_id>]
            assumed to be in sequence with `get_frame_path` function.
            length of embeddings -> number of frames
        file_path: <str>, default: SEQUENCE_GIF_PATH
            file path used to get
    Returns:
        <str>                            the file path under which the gif was saved
    """
    text_default_args = {
        "fontFace": 1,
        "fontScale": 1,
        "thickness": 1,
    }

    cluster_ids = np.unique(cluster_assignments)
    if cluster_colors is None:
        cluster_colors = dict(zip(cluster_ids, _float_to_int_color_(sns.color_palette(palette='bright', n_colors=len(cluster_ids)))))

    n_frames = len(images_paths)
    image_height, image_width, _ = cv2.imread(images_paths[0]).shape
    lines_pos = ((np.array(range(n_frames)) / n_frames) * image_width).astype(np.int).tolist()

    _train_test_split_marker = np.int(n_train_data_points / n_frames * image_width)
    _train_test_split_marker_colours = [(255, 0, 0), (0, 255, 0)]

    _colors_for_pos_data = [lighten_int_colors(skeleton.colors, amount=v) for v in np.linspace(1, 0.3, len(positional_data))]

    def pipeline(frame_id, frame):
        f = _add_frame_and_embedding_id_(frame, cluster_assignments[frame_id], frame_id)

        # xs are the multiple positional data to plot
        for x_i, x in enumerate(positional_data):
            f = plot_drosophila_2d(x[frame_id].astype(np.int), img=f, colors=_colors_for_pos_data[x_i])


        # train test split marker
        if n_train_data_points == frame_id:
            cv2.line(f, (_train_test_split_marker, image_height - 20), (_train_test_split_marker, image_height - 40), (255, 255, 255), 1)
        else:
            cv2.line(f, (_train_test_split_marker, image_height - 10), (_train_test_split_marker, image_height - 40), (255, 255, 255), 1)



        # train / test text
        f = cv2.putText(**text_default_args,
                        img=f,
                        text='train' if frame_id < n_train_data_points else 'test',
                        org=(_train_test_split_marker, image_height - 40),
                        color=_train_test_split_marker_colours[0 if frame_id < n_train_data_points else 1])

        # experiment id
        f = cv2.putText(**text_default_args,
                        img=f,
                        text=data.experiment_key(obj=image_id_with_exp[frame_id][1]),
                        org=(0, 20),
                        color=(255, 255, 255))

        # image id
        #_text_size, _ = cv2.getTextSize(**text_default_args, text=experiment_key(obj=image_id_with_exp[frame_id][1]))
        #f = cv2.putText(**text_default_args,
        #                img=f,
        #                text=image_id_with_exp[frame_id][0],
        #                org=(_text_size[0], 20),
        #                color=(255, 255, 255))

        # model experiment description
        f = cv2.putText(**text_default_args,
                        img=f,
                        text=labels[frame_id],
                        org=(0, 40),
                        color=(255, 255, 255))

        # cluster assignment bar
        for line_idx, l in enumerate(lines_pos):
            if line_idx == frame_id:
                cv2.line(f, (l, image_height), (l, image_height - 20), cluster_colors[cluster_assignments[line_idx]], 2)
            else:
                cv2.line(f, (l, image_height), (l, image_height - 10), cluster_colors[cluster_assignments[line_idx]], 1)


        return f

    frames = (pipeline(frame_id, cv2.imread(path)) for frame_id, path in enumerate(images_paths) if is_file(path))

    output_path = f"{SetupConfig.value('video_root_path')}/{run_desc}_e-{epochs}_hubert_full.mp4"
    _save_frames_(output_path, frames, format='mp4')

    return output_path



def plot_embedding_assignment(x_id_of_interest, X_embedded, label_assignments):
    seen_labels = label_assignments['label'].unique()
    _cs = sns.color_palette(n_colors=len(seen_labels))

    fig = plt.figure(figsize=(10, 10))
    behaviour_colours = dict(zip(seen_labels, _cs))

    for l, c in behaviour_colours.items():
        _d = X_embedded[label_assignments['label'] == l]
        # c=[c] since matplotlib asks for it
        plt.scatter(_d[:, 0], _d[:,1], c=[c], label=l.name, marker='.')

    #print(x_id_of_interest)
    _t = label_assignments.iloc[x_id_of_interest]['label']
    #print(_t)
    cur_color = behaviour_colours[_t]
    plt.scatter(X_embedded[x_id_of_interest, 0], X_embedded[x_id_of_interest, 1], c=[cur_color], linewidth=10, edgecolors=[[0, 0, 1]])
    plt.legend()
    plt.title('simple t-SNE on latent space')

    # TODO I would like to move the lower part to a different function, not tested if that works
    # though
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()
    #fig.canvas.draw_idle()
#
    ## Now we can save it to a numpy array.
    #plot_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)\
    #              .reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #
    #return plot_data

    fig_val = np.array(fig.canvas.renderer._renderer)[:, :, :3]
    plt.close()
    return fig_val

def combine_images_h(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2, img1.shape[2]), np.uint8)
    vis[:h1, :w1, :] = img1
    vis[:h2, w1:w1+w2, :] = img2
    #vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    return vis
    #cv2.imshow("test", vis)



def video_angle(cluster_assignments, images_paths_for_experiments, cluster_id_to_visualize=None, cluster_colors=None, exp_desc=None, as_frames=False):
    """
    exp_desc refers to the model experimnt id, not fly-experiment


    ... in general stuff in here sucks... big time...
    """
    if cluster_id_to_visualize is None:
        cluster_assignment_idx = list(range(len(cluster_assignments)))
    else:
        cluster_assignment_idx = np.where(cluster_assignments == cluster_id_to_visualize)[0]

    text_default_args = {
        "fontFace": 1,
        "fontScale": 1,
        "thickness": 1,
    }

    cluster_ids = np.unique(cluster_assignments)
    if cluster_colors is None:
        cluster_colors = dict(zip(cluster_ids, _float_to_int_color_(sns.color_palette(palette='bright', n_colors=len(cluster_ids)))))

    image_height, image_width, _ = cv2.imread(images_paths_for_experiments[0][1]).shape
    lines_pos = ((np.array(range(len(cluster_assignments))) / len(cluster_assignments)) * image_width).astype(np.int)[cluster_assignment_idx].tolist()

    def pipeline(frame_nb, frame, frame_id, embedding_id, experiment, experiment_path=None):
        # frame_nb is the number of the frame shown, continuous
        # frame_id is the id of the order of the frame,
        # e.g. frame_nb: [0, 1, 2, 3], frame_id: [123, 222, 333, 401]
        # kinda ugly... note that some variables are from the upper "frame"
        #f = _add_frame_and_embedding_id_(frame, embedding_id, frame_id)
        f = frame

        # experiment id
        f = cv2.putText(**text_default_args,
                        img=f,
                        text=data._key_(experiment),
                        org=(0, 20),
                        color=(255, 255, 255))

        # image id
        _text_size, _ = cv2.getTextSize(**text_default_args, text=data._key_(experiment))
        f = cv2.putText(**text_default_args,
                        img=f,
                        text=pathlib.Path(experiment_path).stem,
                        org=(_text_size[0], 20),
                        color=(255, 255, 255))

        # model experiment description
        f = cv2.putText(**text_default_args,
                        img=f,
                        text=exp_desc,
                        org=(0, 40),
                        color=(255, 255, 255))

        # cluster assignment bar
        for line_idx, l in enumerate(lines_pos):
            if line_idx == frame_nb:
                cv2.line(f, (l, image_height), (l, image_height - 20), cluster_colors[cluster_assignments[cluster_assignment_idx[line_idx]]], 2)
            else:
                cv2.line(f, (l, image_height), (l, image_height - 10), cluster_colors[cluster_assignments[cluster_assignment_idx[line_idx]]], 1)


        return f

    frames = (pipeline(frame_nb, cv2.imread(experiment[1]), frame_id, cluster_assignment,
                       experiment[0], experiment_path=experiment[1])
              for frame_nb, (frame_id, cluster_assignment, experiment) in enumerate(zip(
                  cluster_assignment_idx,
                  cluster_assignments[cluster_assignment_idx],
                  np.array(images_paths_for_experiments)[cluster_assignment_idx]))
              if pathlib.Path(experiment[1]).is_file())

    if as_frames:
        return frames
    else:
        output_path = config.EXPERIMENT_VIDEO_PATH.format(experiment_id=exp_desc, vid_id=cluster_id_to_visualize or 'all')
        _save_frames_(output_path, frames, format='mp4')

        return output_path

# new video helpers

def _path_for_image_(image_id, label):
    base_path = SetupConfig.value('experiment_root_path')
    exp_path = SetupConfig.value('experiment_path_template').format(base_path=base_path,
                                                         study_id=label.study_id,
                                                         fly_id=label.fly_id,
                                                         experiment_id=label.experiment_id)
    return SetupConfig.value('fly_image_template').format(base_experiment_path=exp_path, image_id=image_id)

def resize_image(img, new_width=304):
    wpercent = (new_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    return img.resize((new_width, hsize), Image.ANTIALIAS)

def pad_with_last(list_of_lists):
    max_len = max([len(i) for i in list_of_lists])

    def _pad_with_last_(ls, to_len):
        diff_len = to_len - len(ls)
        return ls + [ls[-1]] * diff_len

    return [_pad_with_last_(ls, max_len) for ls in list_of_lists]

def group_video_of_cluster(cluster_id, paths, run_desc, epochs, n_sequences_to_draw=9,
                           pad_videos=False):
    images = [[resize_image(Image.open(p)) for p in ax1] for ax1 in paths[:n_sequences_to_draw]]

    if pad_videos:
        images = pad_with_last(images)

    img = images[0][0]

    element_width, element_height = img.size
    n_elements_x_dim = np.int(np.sqrt(n_sequences_to_draw))
    n_elements_y_dim = np.int(np.sqrt(n_sequences_to_draw))

    combined_images = [Image.new('RGB', (n_elements_x_dim * element_width, n_elements_y_dim * element_height)) for _ in range(len(images[0]))]

    for sequence_id, sequence in enumerate(images):
        x_offset = (sequence_id % n_elements_x_dim) * element_width
        y_offset = (sequence_id // n_elements_x_dim) * element_height

        for frame_number, image in enumerate(sequence):
            combined_images[frame_number].paste(image, (x_offset, y_offset))

    #return combined_images, images

    file_path = (f"{SetupConfig.value('video_root_path')}"
                 f"/group_of_cluster-{cluster_id}-{run_desc}-e-{epochs}.mp4")
    _save_frames_(file_path, combined_images)
    return file_path

def group_video_of_clusters(cluster_assignments, frames_with_labels, run_desc, epochs,
                            n_sequences_to_draw=9, n_clusters_to_draw=10):
    grouped = group_by_cluster(cluster_assignments)

    sorted_groups = sorted([(g, sorted(vals, key=len, reverse=True)) for g, vals in grouped.items()],
                           key=lambda x: max(map(len, x[1])),
                           reverse=True)

    for cluster_id, sequences in sorted_groups[:n_clusters_to_draw]:
        sequences[:n_sequences_to_draw]
        paths = [[_path_for_image_(image_id, label) for image_id, label in frames_with_labels[seq]] for seq in sequences]
        #return paths
        yield cluster_id, group_video_of_cluster(cluster_id, paths, run_desc, epochs=epochs, n_sequences_to_draw=n_sequences_to_draw)

