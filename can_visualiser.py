import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from cyclopts import App, Parameter
from typing import Annotated
from pathlib import Path

app = App(version="0.1.0")


# Setting up the matplotlib environment. These sizes work well for plt.figure(figsize=(10,x)).
def setFonts(fontsize=18, axisLW=1, ticksize=5, tick_direction='out', padding=5, top_ticks=False, right_ticks=False):
    plt.rc('font', family='serif', size=fontsize)                                                   # controls default font family and text sizes
    plt.rcParams['font.cursive'] = ['DejaVu Sans', 'Liberation Sans', 'Nimbus Sans L', 'Arial', 'Helvetica', 'Comic Sans MS', 'cursive']
    plt.rc('axes', titlesize=fontsize, linewidth=axisLW, labelsize=fontsize, labelpad=padding)      # fontsize of the axes title and the x and y labels
    plt.rc('xtick', labelsize=fontsize, direction=tick_direction, top=top_ticks)                    # fontsize of the xtick labels
    plt.rc('ytick', labelsize=fontsize, direction=tick_direction, right=right_ticks)                # fontsize of the ytick labels
    plt.rc('xtick.major', pad=padding, width=axisLW, size=ticksize)                                 # size of x major ticks
    plt.rc('ytick.major', pad=padding, width=axisLW, size=ticksize)                                 # size of y major ticks
    plt.rc('xtick.minor', width=axisLW, size=ticksize/2)                                            # size of x minor ticks
    plt.rc('ytick.minor', width=axisLW, size=ticksize/2)                                            # size of y minor ticks
    plt.rc('legend', fontsize=fontsize)                                                             # legend fontsize
    plt.rc('figure', titlesize=fontsize)                                                            # fontsize of the figure title
    plt.rc('mathtext', fontset='custom', rm='serif')                                                # raw text in math environment will be set to serif

@app.default
def find(
    obs_dir: Annotated[Path, Parameter(name=["-o", "--obs-dir"], help="Name of the observation directory. If provided, --csv-file and --data-dir are not needed")] = None,
    csv_file: Annotated[Path, Parameter(name=["-c", "--csv-file"], help="Path to classification_results.csv. Required if -o is not provided.")] = None,
    data_dir: Annotated[Path, Parameter(name=["-d", "--data-dir"], help="Path to the root `FRBPipeData` directory, which contains the `BMxx` subdirectories. Required if `-o` is not provided.")] = None,
    zoom: Annotated[bool, Parameter(name=["-z", "--zoom"], help="(Optional) If specified, the script will zoom in on the central part of the plots.")] = False,
):
    """Generate feature plots of all the triggered candidates. Either --obs-dir or both --csv-file and --data-dir must be provided."""
    setFonts()

    if obs_dir:
        if csv_file is None:
            csv_file = Path(f"/lustre_archive/spotlight/data/{obs_dir}/DetClassCsv/classification_results.csv")
        if data_dir is None:
            data_dir = Path(f"/lustre_data/spotlight/data/{obs_dir}/FRBPipeData/")
    elif csv_file is None or data_dir is None:
        raise ValueError("Either --obs-dir or both --csv-file and --data-dir must be provided.")

    name = "zoomed-in" if zoom else "full"
    output_dir = data_dir / "Triggered_candidates"
    os.makedirs(output_dir, exist_ok=True)

    triggers = pd.read_csv(csv_file, header=0, skiprows=[1,2])
    triggers=triggers.to_records(index=False)

    console = Console()
    with Progress(
        TextColumn("[bold cyan]Processing triggers"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("process", total=len(triggers))
        for i in range(len(triggers)):
            fname = triggers['id'][i]+".h5"
            beam = triggers['beam'][i]
            beam_data_dir = data_dir / f"BM{beam}/"

            try:
                file_path = next(
                    os.path.join(root, fname)
                    for root, _, files in os.walk(beam_data_dir)
                    if fname in files
                )
            except StopIteration:
                console.log(f"File {fname} not found in {beam_data_dir}.")
                progress.advance(task)
                continue

            with h5py.File(file_path, 'r') as f:
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
                ax1.set_yticks([])  # Sets an empty list for y-ticks
                ax1.set_ylabel('')  # Sets an empty string for the y-label

                # middle: freq-time image (shares x-axis with top)
                im1 = ax2.imshow(dt, origin='lower', aspect='equal')
                ax2.set_ylabel('Frequency bins')
                ax2.set_xlabel(r'$\Delta t$')

                # bottom: dm-time image (separate x-axis)
                data_dm_time = f['data_dm_time'][:]
                if zoom:  # Focus on central DM and time bins
                    data_dm_time = data_dm_time[65:-65, 65:-65]
                im2 = ax3.imshow(data_dm_time, aspect='equal')
                ax3.set_ylabel('DM bins')
                ax3.set_xlabel(r'$\Delta t$')

                fig.tight_layout()
                output_path = output_dir / f"{i+1}_{triggers['id'][i]}_{name}.png"
                fig.savefig(output_path, bbox_inches='tight')
                plt.close(fig)

            progress.advance(task)

@app.command()
def see(
    cand_file: Annotated[Path, Parameter(name=["-c", "--cand-file"], help="Path to the candidate (.h5) file. The candidate name, beam, and data directory are inferred from this path.")],
    zoom: Annotated[bool, Parameter(name=["-z", "--zoom"], help="(Optional) If specified, the script will zoom in on the central part of the plots.")] = False,
):
    """Generate feature plots of a single candidate from if the path to the .h5 file is known."""
    setFonts()
    console = Console()

    if not cand_file.exists():
        console.log(f"[bold red]Error: File not found at {cand_file}[/bold red]")
        return
    
    cand_name = cand_file.stem
    beam = cand_file.parent.parent.name.replace("BM", "")
    data_dir = cand_file.parent.parent.parent.parent

    name = "zoomed-in" if zoom else "full"
    output_dir = data_dir / "Triggered_candidates"
    os.makedirs(output_dir, exist_ok=True)

    console.log(f"Processing candidate [cyan]{cand_name}[/cyan] from beam [cyan]{beam}[/cyan]...")

    with h5py.File(cand_file, 'r') as f:
        dt = f['data_freq_time'][:].T
        if zoom:
            dt = dt[:, 65:-65]  # Focus on central time bins

        mean_profile = dt.mean(axis=0)
        mean_profile /= mean_profile.max()

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"{cand_name}, BM {beam}")
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 3])
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        # top: 1D mean profile
        ax1.plot(mean_profile)
        ax1.set_xlabel(r'$\Delta t$')
        ax1.set_yticks([])  # Sets an empty list for y-ticks
        ax1.set_ylabel('')  # Sets an empty string for the y-label

        # middle: freq-time image (shares x-axis with top)
        ax2.imshow(dt, origin='lower', aspect='equal')
        ax2.set_ylabel('Frequency bins')
        ax2.set_xlabel(r'$\Delta t$')

        # bottom: dm-time image (separate x-axis)
        data_dm_time = f['data_dm_time'][:]
        if zoom:  # Focus on central DM and time bins
            data_dm_time = data_dm_time[65:-65, 65:-65]
        ax3.imshow(data_dm_time, aspect='equal')
        ax3.set_ylabel('DM bins')
        ax3.set_xlabel(r'$\Delta t$')

        fig.tight_layout()
        output_path = output_dir / f"{cand_name}_{name}.png"
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    console.log(f"Successfully generated plot: [green]{output_path}[/green]")

if __name__ == '__main__':
    app()