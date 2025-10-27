import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py
import argparse

# Setting up the matplotlib environment. These sizes work well for plt.figure(figsize=(10,x)).
def setFonts(fontsize=18, axisLW=1, ticksize=5, tick_direction='out', padding=5, top_ticks=False, right_ticks=False):
	plt.rc('font', size=fontsize)													                # controls default font family and text sizes
	plt.rc('axes', titlesize=fontsize, linewidth=axisLW, labelsize=fontsize, labelpad=padding)		# fontsize of the axes title and the x and y labels
	plt.rc('xtick', labelsize=fontsize, direction=tick_direction, top=top_ticks)					# fontsize of the xtick labels
	plt.rc('ytick', labelsize=fontsize, direction=tick_direction, right=right_ticks)				# fontsize of the ytick labels
	plt.rc('xtick.major', pad=padding, width=axisLW, size=ticksize)									# size of x major ticks
	plt.rc('ytick.major', pad=padding, width=axisLW, size=ticksize)									# size of y major ticks
	plt.rc('xtick.minor', width=axisLW, size=ticksize/2)											# size of x minor ticks
	plt.rc('ytick.minor', width=axisLW, size=ticksize/2)											# size of y minor ticks
	plt.rc('legend', fontsize=fontsize)    															# legend fontsize
	plt.rc('figure', titlesize=fontsize)															# fontsize of the figure title
	plt.rc('mathtext', fontset='custom', rm='serif')												# raw text in math environment will be set to serif

setFonts()

# parse command line arguments
parser = argparse.ArgumentParser(description="Generate trigger summary plots.")
parser.add_argument("--csv-file", "-c", default="",
                    help="Path to classification_results.csv")
parser.add_argument("--data-dir", "-d", default="",
                    help="Root FRBPipeData directory (contains BMxx subdirs)")
parser.add_argument("--zoom", "-z", action="store_true",
                    help="Apply central zoom (slice out edges) when True")
args = parser.parse_args()

csv_file = args.csv_file
data_root = args.data_dir
zoom = args.zoom
name = "zoomed-in" if zoom else "full"
os.makedirs(f"{data_root}/Triggered_candidates", exist_ok=True)

triggers = pd.read_csv(csv_file, header=0, skiprows=[1,2])
triggers=triggers.to_records(index=False)

for i in range(len(triggers)):
    fname = triggers['id'][i]+".h5"
    beam = triggers['beam'][i]
    beam_data_dir = f"{data_root}/BM{beam}/"

    file_path = None
    for root, dirs, files in os.walk(beam_data_dir):
        if fname in files:
            file_path = os.path.join(root, fname)
            break

    if file_path is None:
        print(f"File {fname} not found in {beam_data_dir}")
        continue

    with h5py.File(file_path, 'r') as f:
        # load freq-time and dm-time arrays
        dt = f['data_freq_time'][:].T
        if zoom:
            dt = dt[:, 65:-65]  # Focus on central time bins

        mean_profile = dt.mean(axis=0) 
        mean_profile /= mean_profile.max()

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"#{i+1} {triggers['id'][i]}, BM {triggers['beam'][i]}")
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 3])
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        # top: 1D mean profile
        ax1.plot(mean_profile)
        ax1.set_xlabel(r'$\Delta t$')
        ax1.set_yticks([]) # Sets an empty list for y-ticks
        ax1.set_ylabel('')  # Sets an empty string for the y-label

        # middle: freq-time image (shares x-axis with top)
        im1 = ax2.imshow(dt, origin='lower', aspect='equal')
        ax2.set_ylabel('Frequency bins')
        ax2.set_xlabel(r'$\Delta t$')

        # bottom: dm-time image (separate x-axis)
        data_dm_time = f['data_dm_time'][:]
        if zoom:
            data_dm_time = data_dm_time[65:-65, 65:-65]  # Focus on central DM and time bins
        im2 = ax3.imshow(data_dm_time, aspect='equal')
        ax3.set_ylabel('DM bins')
        ax3.set_xlabel(r'$\Delta t$')

        fig.tight_layout()
        fig.savefig(f"{data_root}/Triggered_candidates/{i+1}_{triggers['id'][i]}_{name}.png", bbox_inches='tight')
        plt.close(fig)