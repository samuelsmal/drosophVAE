from itertools import groupby
import pathlib
import logging
import numpy as np
import cv2
import matplotlib.colors as mc
import imageio
import colorsys
import seaborn as sns

from som_vae.helpers.misc import flatten
from som_vae.settings import config, skeleton, data


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


def _get_and_check_file_path_(args, template=config.EXPERIMENT_VIDEO_PATH):
    gif_file_path = template.format(begin_frame=args[0], end_frame=args[-1])
    pathlib.Path(gif_file_path).parent.mkdir(parents=True, exist_ok=True)

    return gif_file_path


def _save_frames_(file_path, frames, format='GIF', **kwargs):
    """
    If format==GIF then fps has to be None, duration should be ~10/60
    If format==mp4 then duration has to be None, fps should be TODO
    """
    if format == 'GIF':
        _kwargs = {'duration': 10/60}
    elif format == 'mp4':
        _kwargs = {'fps': 24}

    pathlib.Path(file_path).parent.mkdir(parents=True, exist_ok=True)
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


def comparision_video_of_reconstruction(xs, embeddings, n_train, experiments, file_path=None, cluster_colors=None, cluster_assignment_idx=None, xs_labels=None, n_frames=1000):
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
    gif_file_path = _get_and_check_file_path_(('video-x_x-hat', 'with-embeddings_cluster_' + str(embeddings[cluster_assignment_idx][0])))

    if cluster_colors is None:
        cluster_ids = np.unique(embeddings)
        cluster_colors = dict(zip(cluster_ids,
                                  _float_to_int_color_(sns.color_palette(palette='bright', n_colors=len(cluster_ids)))))

    if cluster_assignment_idx is None:
        cluster_assignment_idx = list(range(embeddings.shape[0]))
        gif_file_path = _get_and_check_file_path_(('full-video-x_x-hat', 'with-embeddings_cluster_all'))

    image_height, image_width, _ = cv2.imread(get_frame_path(0)).shape
    lines_pos = ((np.array(range(n_frames)) / n_frames) * image_width)\
                    .astype(np.int)[cluster_assignment_idx].tolist()


    _train_test_split_marker = np.int(n_train / n_frames * image_width)
    _train_test_split_marker_colours = [(255, 0, 0), (0, 255, 0)]

    _colors_for_pos_data = [lighten_int_colors(skeleton.colors, amount=v) for v in np.linspace(1, 0.3, len(xs))]

    def pipeline(frame, frame_id, embedding_id):
        # kinda ugly... note that some variables are from the upper "frame"
        f = _add_frame_and_embedding_id_(frame, embedding_id, frame_id)

        # xs are the multiple positional data to plot
        for x_i, x in enumerate(xs):
            f = plot_drosophila_2d(x[frame_id].astype(np.int), img=f, colors=_colors_for_pos_data[x_i])


        # train test split marker
        if n_train == frame_id:
            cv2.line(f, (_train_test_split_marker, image_height - 20), (_train_test_split_marker, image_height - 40), (255, 255, 255), 1)
        else:
            cv2.line(f, (_train_test_split_marker, image_height - 10), (_train_test_split_marker, image_height - 40), (255, 255, 255), 1)

        # train / test text
        f = cv2.putText(img=np.copy(f),
                        text='train' if frame_id < n_train else 'test',
                        org=(_train_test_split_marker, image_height - 40),
                        fontFace=1,
                        fontScale=1,
                        color=_train_test_split_marker_colours[0 if frame_id < n_train else 1],
                        thickness=1)


        # cluster assignment bar
        for line_idx, l in enumerate(lines_pos):
            if line_idx == frame_id:
                cv2.line(f, (l, image_height), (l, image_height - 20), cluster_colors[embeddings[cluster_assignment_idx[line_idx]]], 2)
            else:
                cv2.line(f, (l, image_height), (l, image_height - 10), cluster_colors[embeddings[cluster_assignment_idx[line_idx]]], 1)

        return f

    frames = [pipeline(cv2.imread(get_frame_path(i)), i, emb_id)
              for i, emb_id in enumerate(embeddings[cluster_assignment_idx])]
    _save_frames_(gif_file_path, frames, format='mp4')

    return gif_file_path


_BEHAVIOR_COLORS_ = dict(zip(list(data._BehaviorLabel_),
                             _float_to_int_color_(sns.color_palette(n_colors=len(data._BehaviorLabel_)))))
