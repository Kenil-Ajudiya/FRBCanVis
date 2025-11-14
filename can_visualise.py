from pandas import read_csv
from numpy import linspace, pi
from matplotlib.pyplot import rc, rcParams, figure, close
from os import makedirs, walk
from h5py import File
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from cyclopts import App, Parameter
from typing import Annotated
from pathlib import Path

app = App(
    version="0.2.0",
    help="A CLI tool for visualising FRB candidates from the SPOTLIGHT survey.",
)

# Setting up the matplotlib environment. These sizes work well for figure(figsize=(10,x)).
def setFonts(fontsize=18, axisLW=1, ticksize=5, tick_direction='out', padding=5, top_ticks=False, right_ticks=False):
    rc('font', family='serif', size=fontsize)                                                   # controls default font family and text sizes
    rcParams['font.cursive'] = ['DejaVu Sans', 'Liberation Sans', 'Nimbus Sans L', 'Arial', 'Helvetica', 'Comic Sans MS', 'cursive']
    rc('axes', titlesize=fontsize, linewidth=axisLW, labelsize=fontsize, labelpad=padding)      # fontsize of the axes title and the x and y labels
    rc('xtick', labelsize=fontsize, direction=tick_direction, top=top_ticks)                    # fontsize of the xtick labels
    rc('ytick', labelsize=fontsize, direction=tick_direction, right=right_ticks)                # fontsize of the ytick labels
    rc('xtick.major', pad=padding, width=axisLW, size=ticksize)                                 # size of x major ticks
    rc('ytick.major', pad=padding, width=axisLW, size=ticksize)                                 # size of y major ticks
    rc('xtick.minor', width=axisLW, size=ticksize/2)                                            # size of x minor ticks
    rc('ytick.minor', width=axisLW, size=ticksize/2)                                            # size of y minor ticks
    rc('legend', fontsize=fontsize)                                                             # legend fontsize
    rc('figure', titlesize=fontsize)                                                            # fontsize of the figure title
    rc('mathtext', fontset='custom', rm='serif')                                                # raw text in math environment will be set to serif

def write_txt(obs_metadata: dict, cand_metadata: dict, file_path: Path):
    """Writes candidate and observation metadata to a text file."""
    ra = obs_metadata['ra'] * 12 / pi
    ra_h = int(ra)
    ra = (ra - ra_h) * 60
    ra_m = int(ra)
    ra_s = (ra - ra_m) * 60
    ra_str = f"{ra_h}h{ra_m}m{ra_s:.2f}s"

    dec = obs_metadata['dec'] * 180 / pi
    dec_h = int(dec)
    dec = (dec - dec_h) * 60
    dec_m = int(dec)
    dec_s = (dec - dec_m) * 60
    dec_str = f"{dec_h}d{abs(dec_m)}m{abs(dec_s):.2f}s"

    beam = obs_metadata['beamid'] * obs_metadata['nbeamspernode'] + obs_metadata['beamid']

    beam_ra = cand_metadata['ra'] * 12 / pi
    beam_ra_h = int(beam_ra)
    beam_ra = (beam_ra - beam_ra_h) * 60
    beam_ra_m = int(beam_ra)
    beam_ra_s = (beam_ra - beam_ra_m) * 60
    beam_ra_str = f"{beam_ra_h}h{beam_ra_m}m{beam_ra_s:.2f}s"

    beam_dec = cand_metadata['dec'] * 180 / pi
    beam_dec_h = int(beam_dec)
    beam_dec = (beam_dec - beam_dec_h) * 60
    beam_dec_m = int(beam_dec)
    beam_dec_s = (beam_dec - beam_dec_m) * 60
    beam_dec_str = f"{beam_dec_h}d{abs(beam_dec_m)}m{abs(beam_dec_s):.2f}s"

    labels = ['source','ra_j2000','dec_j2000','beam_no','burst_ra','burst_dec','mjd','toa','dm','snr','width','t_samp','freq_low','freq_high','band_width','probability']

    fields = [str(obs_metadata['source']),
                ra_str,
                dec_str,
                str(beam),
                beam_ra_str,
                beam_dec_str,
                str(cand_metadata['mjd']),
                f"{cand_metadata['tbegist']}",
                f"{cand_metadata['dm']}",
                f"{cand_metadata['snr']}",
                f"{cand_metadata['width']}",
                f"{obs_metadata['dt']}",
                str(round(obs_metadata['fl'])),
                str(round(obs_metadata['fh'])),
                str(obs_metadata['bw']),
                f"{cand_metadata['probability']:.2f}"]

    out_lines = []
    for k, v in zip(labels, fields):
        out_lines.append(f"{k}={v}")

    with open(file_path, 'w') as fo:
        fo.write('\n'.join(out_lines))

def visualise(file_path: Path, output_path: Path, zoom: bool):
    """
    Generates and saves the feature plots for a single candidate.

    Parameters
    ----------
    file_path : Path
        Path to the candidate's .h5 file.
    output_path : Path
        Path to save the output PNG image.
    zoom : bool
        Whether to apply a central zoom to the data.
    """
    labels_L = [r'Source',
                r'RA (J2000)',
                r'DEC (J2000)',
                r'Beam No.',
                r'Beam RA',
                r'Beam DEC',
                r'MJD']
    labels_R = [r'DM',
                r'SNR',
                r'Width',
                r't$_{\rm samp}$',
                r'$\nu_{\rm low}$',
                r'$\nu_{\rm high}$',
                r'$\Delta \nu$']

    with File(file_path, 'r') as f:
        cand_metadata = dict(f.attrs)
        obs_metadata = dict(f['extras'].attrs)
        write_txt(obs_metadata, cand_metadata, file_path.parent / (file_path.stem + ".txt"))
        ra = obs_metadata['ra'] * 12 / pi
        ra_h = int(ra)
        ra = (ra - ra_h) * 60
        ra_m = int(ra)
        ra_s = (ra - ra_m) * 60
        ra_str = f"{ra_h}\ h\ {ra_m}\ m\ {ra_s:.2f}\ s"
        dec = obs_metadata['dec'] * 180 / pi
        dec_h = int(dec)
        dec = (dec - dec_h) * 60
        dec_m = int(dec)
        dec_s = (dec - dec_m) * 60
        dec_str = f"{dec_h}^{{\circ}}\ {abs(dec_m)}\ m\ {abs(dec_s):.2f}\ s"

        beam = obs_metadata['beamid'] * obs_metadata['nbeamspernode'] + obs_metadata['beamid']
        beam_ra = cand_metadata['ra'] * 12 / pi
        beam_ra_h = int(beam_ra)
        beam_ra = (beam_ra - beam_ra_h) * 60
        beam_ra_m = int(beam_ra)
        beam_ra_s = (beam_ra - beam_ra_m) * 60
        beam_ra_str = f"{beam_ra_h}\ h\ {beam_ra_m}\ m\ {beam_ra_s:.2f}\ s"
        beam_dec = cand_metadata['dec'] * 180 / pi
        beam_dec_h = int(beam_dec)
        beam_dec = (beam_dec - beam_dec_h) * 60
        beam_dec_m = int(beam_dec)
        beam_dec_s = (beam_dec - beam_dec_m) * 60
        beam_dec_str = f"{beam_dec_h}^{{\circ}}\ {abs(beam_dec_m)}\ m\ {abs(beam_dec_s):.2f}\ s"

        fields_L = [[rf"$\rm {obs_metadata['source']}$"],
                    [rf"$\rm {ra_str}$"],
                    [rf"$\rm {dec_str}$"],
                    [rf"$\rm {beam}$"],
                    [rf"$\rm {beam_ra_str}$"],
                    [rf"$\rm {beam_dec_str}$"],
                    [rf"$\rm {cand_metadata['mjd']}$"]]
        fields_R = [[rf"$\rm {cand_metadata['dm']:.3f}\ pc \cdot cm^{{-3}}$"],
                    [rf"$\rm {cand_metadata['snr']:.3f}$"],
                    [rf"$\rm {cand_metadata['width'] * 1e6}\ \mu s$"],
                    [rf"$\rm {obs_metadata['dt'] * 1e6}\ \mu s$"],
                    [rf"$\rm {round(obs_metadata['fl'])}\ MHz$"],
                    [rf"$\rm {round(obs_metadata['fh'])}\ MHz$"],
                    [rf"$\rm {obs_metadata['bw']}\ MHz$"]]

        if zoom:
            dyn_spec = f['data_freq_time'][:].T[:, 65:-65]
            data_dm_time = f['data_dm_time'][:][65:-65, 65:-65]
        else:
            dyn_spec = f['data_freq_time'][:].T
            data_dm_time = f['data_dm_time'][:]
        mean_profile = dyn_spec.mean(axis=0)

        # Calculate axis ranges from metadata
        delta_t_ms = obs_metadata['dt'] * cand_metadata['tbin'] * 128 * 1e3
        time_extent = [-delta_t_ms, delta_t_ms]
        freq_extent = [round(obs_metadata['fl']), round(obs_metadata['fh'])]
        dm_extent = [cand_metadata['lodm'], cand_metadata['hidm']]

        fig = figure(figsize=(16, 12))
        fig.suptitle(f"Time of arrival: {cand_metadata['tbegist']}; Probability: {cand_metadata['probability']:.2f}")
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 3])
        gs1 = gs[1, 0].subgridspec(2, 1, hspace=0, height_ratios=[1, 2])
        axtab_L = fig.add_subplot(gs[0, 0])
        axtab_R = fig.add_subplot(gs[0, 1])
        axbtm = fig.add_subplot(gs1[1, 0])
        btm = axbtm.get_position()
        axtop = fig.add_subplot(gs1[0, 0], sharex=axbtm, box_aspect=btm.height / 1.99 / btm.width) # Share x-axis with axbtm
        top = axtop.get_position()
        axright = fig.add_subplot(gs[1, 1])

        # top: observation and candidate metadata
        axtab_L.axis("off")
        table_L = axtab_L.table(cellText=fields_L, rowLabels=labels_L, cellLoc='center', rowLoc='left', colWidths=[1/2], bbox=[0.35, 0, 0.55, 1])
        table_L.auto_set_font_size(False)
        axtab_R.axis("off")
        table_R = axtab_R.table(cellText=fields_R, rowLabels=labels_R, cellLoc='center', rowLoc='left', colWidths=[1/2], bbox=[0.25, 0, 0.55, 1])
        table_R.auto_set_font_size(False)

        # bottom left: pulse profile and freq-time
        axbtm.minorticks_on()
        im1 = axbtm.imshow(dyn_spec, aspect=(2 * delta_t_ms) / (round(obs_metadata['fh']) - round(obs_metadata['fl'])), extent=[*time_extent, *freq_extent])
        axtop.plot(linspace(*time_extent, len(mean_profile)), mean_profile)
        axtop.minorticks_on()
        axtop.tick_params(which='both', top=True, labeltop=False, bottom=False, labelbottom=False, left=False, labelleft=False)
        axtop.grid(True, alpha=0.25)
        axbtm.grid(True, alpha=0.25)
        axbtm.set_ylabel('Frequency (MHz)')
        axbtm.set_xlabel(r'$\Delta$t (ms)')

        # bottom right: dm-time image
        im2 = axright.imshow(data_dm_time, aspect=(2 * delta_t_ms) / (cand_metadata['hidm'] - cand_metadata['lodm']), extent=[*time_extent, *dm_extent])
        axright.minorticks_on()
        axright.grid(True, alpha=0.25)
        axright.set_ylabel(r'DM ($\rm pc \cdot cm^{-3}$)')
        axright.set_xlabel(r'$\Delta$t (ms)')

        fig.tight_layout()
        fig.savefig(output_path, bbox_inches='tight')
        close(fig)

@app.default
def quickly_plot(
    obs_dir: Annotated[Path, Parameter(name=["-o", "--obs-dir"], help="Name of the observation directory. If provided, --csv-file and --data-dir are not needed")] = None,
    csv_file: Annotated[Path, Parameter(name=["-c", "--csv-file"], help="Path to classification_results.csv. Required if -o is not provided.")] = None,
    data_dir: Annotated[Path, Parameter(name=["-d", "--data-dir"], help="Path to the root `FRBPipeData` directory, which contains the `BMxx` subdirectories. Required if `-o` is not provided.")] = None,
    zoom: Annotated[bool, Parameter(name=["-z", "--zoom"], help="(Optional) If specified, the script will zoom in on the central part of the plots.")] = False,
):
    """Quickly generate feature plots of all the triggered candidates if their buffer count directories are known. Either --obs-dir or both --csv-file and --data-dir must be provided."""
    if obs_dir is None and (csv_file is None or data_dir is None):
        raise ValueError("Either --obs-dir or both --csv-file and --data-dir must be provided.")
    else:
        if csv_file is None:
            csv_file = Path(f"/lustre_archive/spotlight/data/{obs_dir}/DetClassCsv/classification_results.csv")
        if data_dir is None:
            data_dir = Path(f"/lustre_data/spotlight/data/{obs_dir}/FRBPipeData/")

    name = "zoomed-in" if zoom else "full"
    output_dir = data_dir / "Triggered_candidates"
    makedirs(output_dir, exist_ok=True)

    triggers = read_csv(csv_file, header=0, skiprows=[1,2])
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
            cand_id = triggers['id'][i]
            file_path = data_dir / f"BM{triggers['beam'][i]}/{triggers['bufcount'][i]}/{triggers['id'][i]}.h5"

            output_path = output_dir / f"{i+1}_{cand_id}_{name}.png"
            visualise(file_path, output_path, zoom)

            progress.advance(task)

@app.command
def find(
    obs_dir: Annotated[Path, Parameter(name=["-o", "--obs-dir"], help="Name of the observation directory. If provided, --csv-file and --data-dir are not needed")] = None,
    csv_file: Annotated[Path, Parameter(name=["-c", "--csv-file"], help="Path to classification_results.csv. Required if -o is not provided.")] = None,
    data_dir: Annotated[Path, Parameter(name=["-d", "--data-dir"], help="Path to the root `FRBPipeData` directory, which contains the `BMxx` subdirectories. Required if `-o` is not provided.")] = None,
    zoom: Annotated[bool, Parameter(name=["-z", "--zoom"], help="(Optional) If specified, the script will zoom in on the central part of the plots.")] = False,
):
    """Generate feature plots of all the triggered candidates. Either --obs-dir or both --csv-file and --data-dir must be provided."""
    if obs_dir:
        if csv_file is None:
            csv_file = Path(f"/lustre_archive/spotlight/data/{obs_dir}/DetClassCsv/classification_results.csv")
        if data_dir is None:
            data_dir = Path(f"/lustre_data/spotlight/data/{obs_dir}/FRBPipeData/")
    elif csv_file is None or data_dir is None:
        raise ValueError("Either --obs-dir or both --csv-file and --data-dir must be provided.")

    name = "zoomed-in" if zoom else "full"
    output_dir = data_dir / "Triggered_candidates"
    makedirs(output_dir, exist_ok=True)

    triggers = read_csv(csv_file, header=0, skiprows=[1,2])
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
            cand_id = triggers['id'][i]
            beam = triggers['beam'][i]
            fname = cand_id + ".h5"
            beam_data_dir = data_dir / f"BM{beam}/"

            try:
                file_path = next(
                    Path(root) / fname
                    for root, _, files in walk(beam_data_dir)
                    if fname in files
                )
            except StopIteration:
                console.log(f"File {fname} not found in {beam_data_dir}.")
                progress.advance(task)
                continue

            output_path = output_dir / f"{i+1}_{cand_id}_{name}.png"
            visualise(file_path, output_path, zoom)

            progress.advance(task)

@app.command
def see(
    cand_file: Annotated[Path, Parameter(name=["-c", "--cand-file"], help="Path to the candidate (.h5) file. The candidate name, beam, and data directory are inferred from this path.")],
    zoom: Annotated[bool, Parameter(name=["-z", "--zoom"], help="(Optional) If specified, the script will zoom in on the central part of the plots.")] = False,
):
    """Generate feature plots of a single candidate from if the path to the .h5 file is known."""
    console = Console()

    if not cand_file.exists():
        console.log(f"[bold red]Error: File not found at {cand_file}[/bold red]")
        return
    
    cand_name = cand_file.stem
    beam = cand_file.parent.parent.name[2:]
    name = "zoomed-in" if zoom else "full"
    console.log(f"Processing candidate [cyan]{cand_name}[/cyan] from beam [cyan]{beam}[/cyan]...")

    output_path = cand_file.parent / f"{cand_name}_{name}.png"
    visualise(cand_file, output_path, zoom)

    console.log(f"Successfully generated plot: [green]{output_path}[/green]")

if __name__ == '__main__':
    setFonts()
    app()