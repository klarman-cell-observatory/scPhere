from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


# ==============================================================================
def plot_trace(x, y, y_label=None, x_label="Iteration"):
    num_plot = len(x)
    fig, ax = plt.subplots(num_plot, figsize=(16, 12.5))
    if num_plot == 1:
        ax = [ax]

    # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y'])))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.08, left=0.1)

    for i, x_i in enumerate(x):
        ax[i].plot(x_i, y[i])
        ax[i].set_ylabel(y_label[i])
        ax[i].tick_params(axis='x', which='both',
                          bottom='off', top='off', right='off',
                          labelbottom='off')
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[i].grid()

    plt.subplots_adjust(wspace=0.1, hspace=0.08, left=0.1, bottom=0.1)

    ax[-1].tick_params(axis='x', which='both',
                       bottom='on', top='off', right='off',
                       labelbottom='on')
    ax[-1].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[-1].set_xlabel("{}".format(x_label))
