import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from som_vae.settings import config, skeleton


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
    # TODO move this into one single plot
    # TODO provide decorator which saves the figure
    for leg in config.LEGS:
        fig, axs = plt.subplots(1, config.NB_OF_AXIS, sharex=True, figsize=(20, 10))
        for tracked_point in range(config.NB_TRACKED_POINTS):
            for axis in range(config.NB_OF_AXIS):
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


def plot_comparing_joint_position_with_reconstructed(real_joint_positions, reconstructed_joint_positions, validation_cut_off=None):
    for leg in config.LEGS:
        fig, axs = plt.subplots(1, config.NB_OF_AXIS * 2, sharex=True, figsize=(25, 10))
        for axis in range(config.NB_OF_AXIS):
            cur_ax = axs[axis * 2]
            rec_ax = axs[axis * 2 + 1]

            if validation_cut_off is not None:
                for a in [cur_ax, rec_ax]:
                    a.axvline(validation_cut_off, label='validation cut off', linestyle='--')

            for tracked_point in range(config.NB_TRACKED_POINTS):
                cur_ax.plot(real_joint_positions[:, _get_feature_id_(leg, tracked_point),  axis], label = f"{_get_feature_name_(tracked_point)}_{('x' if axis == 0 else 'y')}")
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

def plot_losses(losses, legend=None, title=None):
    plt.figure(figsize=(15, 8))
    if legend is None:
        legend = ['train', 'test', 'test_recon']
    fig, ax1 = plt.subplots()

    for i, l in enumerate(losses[:-1]):
        ax1.plot(l, label=legend[i])
        ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(losses[-1], label=legend[-1], color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax2.set_xlabel('epoch')

    fig.legend()
    fig.tight_layout()
    plt.title('loss')
    if title is not None:
        plt.title(f"loss with {title}")
    return fig


def plot_latent_frame_distribution(latent_assignments, nb_bins):
    plt.figure()
    plt.hist(latent_assignments, bins=nb_bins)
    plt.title('distribution of latent-space-assignments')
    plt.xlabel('latent-space')
    plt.ylabel('nb of frames in latent-space')


def plot_cluster_assignment_over_time(cluster_assignments):
    plt.figure()
    plt.plot(cluster_assignments)
    plt.title("cluster assignments over time")
    plt.ylabel("index of SOM-embeddings")
    plt.xlabel("frame")


def plot_reconstructed_angle_data(real_data, reconstructed_data, columns, fix_ylim=False):
    _colors = sns.color_palette(n_colors=2)

    fig, axs = plt.subplots(nrows=real_data.shape[1], ncols=1, figsize=(5, 30))
    for a in range(real_data.shape[1]):
        axs[a].plot(real_data[:,a], c=_colors[0], label='real')
        axs[a].plot(reconstructed_data[:,a], c=_colors[1], label='reconstructed')
        axs[a].set_title(f"col: {columns[a]}")
        if fix_ylim:
            axs[a].set_ylim(-np.pi, np.pi)

    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
    fig.suptitle('real vs reconstructed angle data')

    plt.tight_layout()
    plt.subplots_adjust(top=0.96)


    return fig


def plot_angle_columns(data, columns):
    fig, axs = plt.subplots(ncols=1, nrows=len(columns), figsize=(5, 3 * len(columns)))
    for i, c in enumerate(columns):
        axs[i].set_title(c)
        axs[i].plot(data[:, i])
        axs[i].set_xlabel('time')
        axs[i].set_ylabel('[radians]')

    fig.suptitle('Angle data')

    plt.tight_layout()
    plt.subplots_adjust(top=0.97) # necessary for the title not to be in the first plot

    return fig


def plot_tnse(X, y, title='t-SNE'):
    """X is really the data

    y is a pandas dataframe with a column called `label`, which are of type _BehaviorLabel_
    """
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)

    seen_labels = y.label.unique()

    _cs = sns.color_palette(n_colors=len(seen_labels))

    fig = plt.figure(figsize=(10, 10))

    behaviour_colours = dict(zip(seen_labels, _cs))

    for l, c in behaviour_colours.items():
        _d = X_embedded[y['label'] == l]
        # c=[c] since matplotlib asks for it
        plt.scatter(_d[:, 0], _d[:,1], c=[c], label=l.name, marker='.')

    plt.legend()
    plt.title(title)

    return fig


def plot_2d_distribution(X_train, X_test, n_legs=3):
    fig, ax = plt.subplots(nrows=n_legs, ncols=2, figsize=(10, 8))

    for leg_idx in range(n_legs):
        for j in range(5 * 2):
            cur_col = leg_idx * 10 + j
            sns.distplot(X_train[:, cur_col],
                         ax=ax[leg_idx][0],
                         bins=50)
            sns.distplot(X_test[:, cur_col],
                         ax=ax[leg_idx][1],
                         bins=50)

    ax[0][0].set_title('training data')
    ax[0][1].set_title('testing data')

    fig.suptitle('distribution of input')
    plt.tight_layout()
    plt.subplots_adjust(top=0.97) # necessary for the title not to be in the first plot

    return fig


