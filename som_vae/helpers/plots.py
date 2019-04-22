import seaborn
import matplotlib.pyplot as plt

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

def plot_losses(losses, legend=None):
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
    #fig.title('loss')

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
