import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from som_vae.settings import config, skeleton


def save_figure(func):
    """Decorator for saving figures. Suptitle must be set."""
    def clean_string(s):
        _replacements_ = [("\'", ""), (" ", "-"), (",", "-"), ("\n", ""), ("(", "_"), (")", "")]
        for m, r in _replacements_:
            s = s.replace(m, r)

        return s.lower()
    def wrapper(*args, **kwargs):
        fig = func(*args, **kwargs)
        if fig is None:
            return fig
        s = clean_string(fig._suptitle.get_text())
        fig.savefig(f"{config.PATH_TO_FIGURES}/{s}.png")
        return fig
    return wrapper


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


@save_figure
def plot_comparing_joint_position_with_reconstructed(real_joint_positions, reconstructed_joint_positions, generated_positions, validation_cut_off=None, run_config=None, epochs=None):
    fig, axs = plt.subplots(len(config.LEGS), config.NB_OF_AXIS * 3, sharex=True, figsize=(25, 10))

    for idx_leg, leg in enumerate(config.LEGS):
        for axis in range(config.NB_OF_AXIS):
            cur_ax = axs[idx_leg][axis * 3]
            rec_ax = axs[idx_leg][axis * 3 + 1]
            gen_ax = axs[idx_leg][axis * 3 + 2]


            if validation_cut_off is not None:
                for a in [cur_ax, rec_ax]:
                    a.axvline(validation_cut_off, label='validation cut off', linestyle='--')

            for tracked_point in range(config.NB_TRACKED_POINTS):
                _label_ = f"{_get_feature_name_(tracked_point)}_{('x' if axis == 0 else 'y')}"
                cur_ax.plot(real_joint_positions[:, _get_feature_id_(leg, tracked_point),  axis], label=_label_)
                rec_ax.plot(reconstructed_joint_positions[:, _get_feature_id_(leg, tracked_point),  axis], label=_label_)
                gen_ax.plot(generated_positions[:, _get_feature_id_(leg, tracked_point),  axis], label=_label_)

                cur_ax.get_shared_y_axes().join(cur_ax, rec_ax)
                cur_ax.get_shared_y_axes().join(cur_ax, gen_ax)
                rec_ax.set_yticks([])
                gen_ax.set_yticks([])

    for i in range(config.NB_OF_AXIS):
        axs[0][i * 3].set_title('input data')
        axs[0][i * 3 + 1].set_title('reconstructed data')
        axs[0][i * 3 + 2].set_title('generated data')

        axs[-1][i * 3].set_xlabel('frames')
        axs[-1][i * 3 + 1].set_xlabel('frames')
        axs[-1][i * 3 + 2].set_xlabel('frames')

    for i in range(len(config.LEGS)):
        axs[i][0].set_ylabel(f"{_get_leg_name_(leg)}: x pos")
        axs[i][2].set_ylabel(f"{_get_leg_name_(leg)}: y pos")

    _, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(labels, loc='upper right')
    fig.suptitle(f"Comparing input and reconstruction\n({config.config_description(run_config)}_epochs-{epochs})")
    fig.align_ylabels(axs)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig


@save_figure
def plot_losses(train_loss, test_loss, run_config):
    fig = plt.figure(figsize=(15, 8))
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.xlabel('epochs')
    plt.ylabel('loss (ELBO)')
    plt.legend()

    fig.suptitle(f"Loss (ELBO)\n({config.config_description(run_config)})")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return fig


def plot_losses_v0(losses, legend=None, title=None):
    """the version for the SOM-VAE model"""
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


@save_figure
def plot_2d_distribution(X_train, X_test, n_legs=3, run_config=None):
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

    plt.suptitle(f"distribution of input\n({config.config_description(run_config)})")
    plt.tight_layout()
    plt.subplots_adjust(top=0.89) # necessary for the title not to be in the first plot

    return fig


@save_figure
def plot_distribution_of_angle_data(data, run_config):
    from som_vae.settings.data import get_3d_columns_names
    """
    data is should be a list of lists (on list for each experiment)
    """
    # This will take some time... you can set `sharey=False` to speed it up.
    fig, axs = plt.subplots(nrows=len(data), ncols=3, figsize=(20, len(data) // 2))

    for i, data_set in enumerate(data):
        selected_cols = np.where(np.var(data_set, axis=0) > 0.0)[0]
        column_names = get_3d_columns_names(selected_cols)

        for s, cn in zip(selected_cols, column_names):
            sns.distplot(data_set[:, s], label=cn, ax=axs[i][int(cn[len('limb: '):len('limb: 0')])])


    plt.suptitle(f"distribution of angled data\n({config.config_description(run_config)})")
    plt.tight_layout()

    plt.subplots_adjust(top=0.96)
    for i, ax in enumerate(axs[0]):
        ax.set_title(f"limb {i}")

    return fig


@save_figure
def plot_3d_angle_data_distribution(X_train, X_test, selected_columns, run_config):
    fig, axs = plt.subplots(nrows=X_train.shape[-1] // 3, ncols=2, figsize=(10, 6))
    col_names = SD.get_3d_columns_names(selected_columns)

    for c in range(X_train.shape[-1]):
        if run_config['use_time_series']:
            sns.distplot(X_train[:, -1, c],ax=axs[c // 3][0])
            sns.distplot(X_test[:, -1, c], ax=axs[c // 3][1])
        else:
            sns.distplot(X_train[:, c],ax=axs[c // 3][0])
            sns.distplot(X_test[:, c], ax=axs[c // 3][1])


    for i, a in enumerate(axs):
        a[0].set_xlabel(col_names[i * 3][:len('limb: 0')])

    plt.suptitle(f"distribution of train and test data\n({config.config_description(run_config)})")

    axs[0][0].set_title('train')
    axs[0][1].set_title('test')

    # order of these two calls is important, sadly
    plt.tight_layout()
    plt.subplots_adjust(top=0.84)

    return fig
