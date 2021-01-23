import matplotlib.pyplot as plt
import numpy as np
import copy
from matplotlib import rc
from utils.utils_io import load_run

rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica'],
              'size': 11})
rc('text', usetex=True)


def smooth(y, box_pts):
    if box_pts == 1:
        return y
    else:
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


def plot_training_time(files,
                       labels,
                       dir='res',
                       file_name='plotfile'):
    """
    input: files: a list of lists of plotted files
           labels: the corresponding legend in each plot
           smoothing: moving average window to smooth the plots
           dir: name of directory
           file_name: saved figure name
    """
    ress_return = copy.deepcopy(files)

    ress_90pcnt_return = []
    ress_10pcnt_return = []
    ress_50pcnt_return = []

    for ii in range(len(files)):
        for jj in range(len(files[ii])):
            config, res = load_run({}, dir_name=dir, file_name=files[ii][jj])
            ress_return[ii][jj] = res['highest_return']
        avg_return = np.array(ress_return[ii])
        ress_90pcnt_return.append(np.percentile(avg_return, q=90, axis=0))
        ress_10pcnt_return.append(np.percentile(avg_return, q=10, axis=0))
        ress_50pcnt_return.append(np.percentile(avg_return, q=50, axis=0))

    recording_interval = config['recording_interval']

    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    for ii in range(len(files)):
        median_rcv = ress_50pcnt_return[ii]
        upper_rcv = ress_90pcnt_return[ii]
        lower_rcv = ress_10pcnt_return[ii]
        xlen = len(median_rcv)
        ax1.plot(np.arange(xlen)*recording_interval, median_rcv, label=labels[ii], linewidth=1.0)
        ax1.fill_between(x=np.arange(xlen)*recording_interval, y1=upper_rcv, y2=lower_rcv, alpha=0.1)
    ax1.legend(prop={'size': 10})
    ax1.set_ylabel("Highest return so far")
    ax1.set_xlabel("time (s)")
    ax1.grid(True, alpha=0.1)

    plt.savefig('./figs/' + file_name + '_res.pdf', bbox_inches='tight')

    plt.show()
