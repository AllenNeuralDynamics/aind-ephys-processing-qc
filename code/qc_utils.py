from __future__ import annotations

from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram


import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw


from aind_data_schema.core.processing import Processing
from aind_data_schema.core.quality_control import QCMetric, QCStatus, QCEvaluation, Stage, Status


def _get_fig_axs(nrows, ncols, subplot_figsize=(3, 3)):
    figsize = (subplot_figsize[0] * ncols, subplot_figsize[1] * nrows)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(subplot_figsize[0] * ncols, subplot_figsize[1] * nrows))
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])[:, None]
    elif nrows == 1:
        axs = axs[None, :]
    elif ncols == 1:
        axs = axs[:, None]
    return fig, axs


def plot_raw_data(
    recording: si.BaseRecording,
    recording_lfp: si.BaseRecording | None = None,
    num_snippets_per_segment: int = 3,
    duration_s: float = 0.1,
    freq_ap: float = 300,
    freq_lfp: float = 500
):
    """
    Plot snippets of raw data as an image

    Parameters
    ----------
    recording : BaseRecording
        The recording object.
    recording_lfp : BaseRecording | None
        The LFP recording object.
    num_snippets_per_segment: int, default: 3
        Number of snippets to plot for each segment.
    duration_s : float, default: 0.1
        The duration of each snippet.
    freq_ap : float, default 300
        The highpass cutoff frequency for the ap band
    freq_lfp : float, default: 500
        The lowpass cutoff frequency in case recording_lfp is None

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    num_segments = recording.get_num_segments()
    fig, axs = _get_fig_axs(num_segments * 2, num_snippets_per_segment)

    recording_hp = spre.highpass_filter(recording, freq_min=freq_ap)
    if recording_lfp is None:
        recording_lfp = spre.bandpass_filter(recording, freq_min=0.1, freq_max=freq_lfp)
    for segment_index in range(num_segments):
        # evenly distribute t_starts across segments
        times = recording.get_times(segment_index=segment_index)
        t_starts = np.round(np.linspace(times[0], times[-1], num_snippets_per_segment + 2)[1:-1], 1)
        for snippet_index, t_start in enumerate(t_starts):
            ax_ap = axs[segment_index * 2, snippet_index]
            ax_lfp = axs[segment_index * 2 + 1, snippet_index]
            sw.plot_traces(
                recording_hp,
                time_range=[t_start, t_start + duration_s],
                mode="map",
                return_scaled=True,
                with_colorbar=True,
                ax=ax_ap,
                clim=(-50, 50)
            )
            sw.plot_traces(
                recording_lfp,
                time_range=[t_start, t_start + duration_s],
                mode="map",
                return_scaled=True,
                with_colorbar=True,
                ax=ax_lfp,
                clim=(-300, 300)
            )
            if np.mod(num_snippets_per_segment, 2) == 1:
                if snippet_index == num_snippets_per_segment // 2:
                    ax_ap.set_title(f"seg{segment_index} @ {t_start}s\nAP")
                    ax_lfp.set_title(f"LFP")
                else:
                    ax_ap.set_title(f"seg{segment_index} @ {t_start}s\n ")
            else:
                ax_ap.set_title(f"seg{segment_index} @ {t_start}s\nAP")
                ax_lfp.set_title(f"LFP")
            if snippet_index == 0:
                ax_ap.set_ylabel("Depth ($\mu m$)")
                ax_lfp.set_ylabel("Depth ($\mu m$)")
            else:
                ax_ap.set_yticklabels([])
                ax_lfp.set_yticklabels([])
            if segment_index == num_segments - 1:
                ax_lfp.set_xlabel(f"Time (tot. {duration_s} s)")
            ax_ap.set_xticklabels([])
            ax_lfp.set_xticklabels([])

    fig.subplots_adjust(wspace=0.3, hspace=0.2, top=0.9)
    return fig


def plot_psd(
    recording: si.BaseRecording,
    recording_lfp: si.BaseRecording | None = None,
    num_snippets_per_segment: int = 3,
    duration_s: float = 1,
    freq_lf_filt: float = 500.0,
    freq_lf_viz: float = 100.0,
    freq_hf_filt: float = 3000.0,
    freq_hf_viz: float = 5000.0,
):
    """
    Plot spectra for wide/band, low frequency, and high frequency.

    Parameters
    ----------
    recording : BaseRecording
        The recording object.
    num_snippets_per_segment: int, default: 3
        Number of snippets to plot for each segment.
    duration_s : float, default: 1
        The duration of each snippet.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the wide-band frequency spectrum.
    matplotlib.figure.Figure
        Figure object containing the high frequency spectrum.
    matplotlib.figure.Figure
        Figure object containing the low frequency spectrum.
    """
    num_segments = recording.get_num_segments()
    fig_psd, axs_psd = _get_fig_axs(num_segments * 2, num_snippets_per_segment)
    fig_psd_hf, axs_psd_hf = _get_fig_axs(num_segments * 2, num_snippets_per_segment)
    fig_psd_lf, axs_psd_lf = _get_fig_axs(num_segments * 2, num_snippets_per_segment)

    if recording_lfp is None:
        recording_lfp = recording
    recording_lfp = spre.bandpass_filter(recording, freq_min=0.1, freq_max=freq_lf_filt)
    recording_lfp = spre.resample(recording_lfp, int(1.5 * freq_lf_filt))
    depths = recording.get_channel_locations()[:, 1]

    for segment_index in range(num_segments):
        # evenly distribute t_starts across segments
        times = recording.get_times(segment_index=segment_index)
        t_starts = np.round(np.linspace(times[0], times[-1], num_snippets_per_segment + 2)[1:-1], 1)
        for snippet_index, t_start in enumerate(t_starts):
            ax_psd = axs_psd[segment_index * 2, snippet_index]
            ax_psd_channels = axs_psd[segment_index * 2 + 1, snippet_index]
            ax_psd_hf = axs_psd_hf[segment_index * 2, snippet_index]
            ax_psd_hf_channels = axs_psd_hf[segment_index * 2 + 1, snippet_index]
            ax_psd_lf = axs_psd_lf[segment_index * 2, snippet_index]
            ax_psd_lf_channels = axs_psd_lf[segment_index * 2 + 1, snippet_index]

            start_frame_hf = recording.time_to_sample_index(t_start, segment_index=segment_index)
            end_frame_hf = recording.time_to_sample_index(t_start + duration_s, segment_index=segment_index)
            traces_wide = recording.get_traces(
                start_frame=start_frame_hf, end_frame=end_frame_hf, segment_index=segment_index, return_scaled=True
            )

            start_frame_lf = recording_lfp.time_to_sample_index(t_start, segment_index=segment_index)
            end_frame_lf = recording_lfp.time_to_sample_index(t_start + duration_s, segment_index=segment_index)
            traces_lf = recording_lfp.get_traces(
                start_frame=start_frame_lf, end_frame=end_frame_lf, segment_index=segment_index, return_scaled=True
            )
            power_channels_wide = []
            power_channels_lf = []

            for i in range(traces_wide.shape[1]):
                f_wide, p_wide = welch(traces_wide[:, i], fs=recording.sampling_frequency)
                power_channels_wide.append(p_wide)
                ax_psd.plot(f_wide, p_wide, color="gray", alpha=0.5)

                f_lfp, p_lfp = welch(traces_lf[:, i], fs=recording_lfp.sampling_frequency)
                power_channels_lf.append(p_lfp)

                hf_mask = f_wide > freq_hf_viz
                f_hf = f_wide[hf_mask]
                ax_psd_hf.plot(f_hf, p_wide[hf_mask], color="gray", alpha=0.5)

                lf_mask = f_lfp < freq_lf_viz
                f_lf = f_lfp[lf_mask]
                ax_psd_lf.plot(f_lf, p_lfp[lf_mask], color="gray", alpha=0.5)

            power_channels_wide = np.array(power_channels_wide)
            power_channels_lf = np.array(power_channels_lf)

            p_wide_mean = np.mean(power_channels_wide, axis=0)
            ax_psd.plot(f_wide, p_wide_mean, color="k", lw=1)
            ax_psd_hf.plot(f_hf, p_wide_mean[hf_mask], color="k", lw=1)
            p_lfp_mean = np.mean(power_channels_lf, axis=0)
            ax_psd_lf.plot(f_lf, p_lfp_mean[lf_mask], color="k", lw=1)

            extent_wide = [f_wide.min(), f_wide.max(), np.min(depths), np.max(depths)]
            ax_psd_channels.imshow(
                power_channels_wide, extent=extent_wide, aspect="auto", cmap="inferno", origin="lower", norm="log"
            )
            extent_hf = [f_hf.min(), f_hf.max(), np.min(depths), np.max(depths)]
            ax_psd_hf_channels.imshow(
                power_channels_wide[:, hf_mask],
                extent=extent_hf,
                aspect="auto",
                cmap="inferno",
                origin="lower",
                norm="log",
            )
            extent_lf = [f_lf.min(), f_lf.max(), np.min(depths), np.max(depths)]
            ax_psd_lf_channels.imshow(
                power_channels_lf[:, lf_mask],
                extent=extent_lf,
                aspect="auto",
                cmap="inferno",
                origin="lower",
                norm="log",
            )
            ax_psd.set_title(f"seg{segment_index} @ {t_start}s")
            ax_psd_hf.set_title(f"seg{segment_index} @ {t_start}s")
            ax_psd_lf.set_title(f"seg{segment_index} @ {t_start}s")
            if snippet_index == 0:
                ax_psd.set_ylabel("Power ($\mu V^2/Hz$)")
                ax_psd_hf.set_ylabel("Power ($\mu V^2/Hz$)")
                ax_psd_lf.set_ylabel("Power ($\mu V^2/Hz$)")

                ax_psd_channels.set_ylabel("Depth ($\mu$ m)")
                ax_psd_hf_channels.set_ylabel("Depth ($\mu$ m)")
                ax_psd_lf_channels.set_ylabel("Depth ($\mu$ m)")
            if segment_index == num_segments - 1:
                ax_psd_channels.set_xlabel("Frequency (Hz)")
                ax_psd_hf_channels.set_xlabel("Frequency (Hz)")
                ax_psd_lf_channels.set_xlabel("Frequency (Hz)")

            ax_psd.set_yscale("log")
            ax_psd_hf.set_yscale("log")
            ax_psd_lf.set_yscale("log")

    fig_psd.subplots_adjust(wspace=0.3, hspace=0.3, top=0.8)
    fig_psd_hf.subplots_adjust(wspace=0.3, hspace=0.3, top=0.8)
    fig_psd_lf.subplots_adjust(wspace=0.3, hspace=0.3, top=0.8)
    fig_psd.suptitle("Wideband")
    fig_psd_hf.suptitle("High-frequency")
    fig_psd_lf.suptitle("Low-frequency")


    return fig_psd, fig_psd_hf, fig_psd_lf


def plot_rms_by_depth(recording, recording_preprocessed=None, recording_lfp=None):
    """ """
    num_segments = recording.get_num_segments()
    fig_rms, ax_rms = plt.subplots(figsize=(5, 12))

    if recording_lfp is None:
        # this means the recording is wide-band, so we apply an additional hp filter
        recording = spre.highpass_filter(recording)

    recording = spre.average_across_direction(recording, direction="y")

    data_raw = si.get_random_data_chunks(recording, return_scaled=True)
    depths_raw = recording.get_channel_locations()[:, 1]
    rms_raw = np.sqrt(np.sum(data_raw**2, axis=0) / data_raw.shape[0])

    ax_rms.plot(rms_raw, depths_raw, color="gray", label="raw")

    if recording_preprocessed is not None:
        recording_preprocessed = spre.average_across_direction(recording_preprocessed, direction="y")
        data_pre = si.get_random_data_chunks(recording_preprocessed, return_scaled=True)

        depths_pre = recording_preprocessed.get_channel_locations()[:, 1]
        rms_pre = np.sqrt(np.sum(data_pre**2, axis=0) / data_pre.shape[0])
        ax_rms.plot(rms_pre, depths_pre, color="r", label="preprocessed")
        ax_rms.legend()

    ax_rms.set_xlabel("RMS ($\mu V$)")
    ax_rms.set_ylabel("Depth ($\mu m$)")
    ax_rms.spines[["right", "top"]].set_visible(False)

    return fig_rms


def probe_raw_qc(
    recording: si.BaseRecording,
    recording_name: str,
    output_qc_path: Path,
    relative_to: Path | None = None,
    recording_lfp: si.BaseRecording | None = None,
    recording_preprocessed: si.BaseRecording | None = None,
    processing: Processing | None = None,
    visualization_output: dict | None = None,
) -> dict[str : list[QCMetric]]:
    metrics = {}
    recording_fig_folder = output_qc_path / recording_name
    recording_fig_folder.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    status_pending = QCStatus(evaluator="", status=Status.PENDING, timestamp=now)
    status_pass = QCStatus(evaluator="", status=Status.PASS, timestamp=now)

    raw_traces_value = {}

    print("Generating RAW DATA metrics")
    fig_raw = plot_raw_data(recording, recording_lfp)
    raw_traces_path = recording_fig_folder / "traces_raw.png"
    fig_raw.savefig(raw_traces_path, dpi=300)
    if relative_to is not None:
        raw_traces_path = raw_traces_path.relative_to(relative_to)

    raw_data_value_with_flags = {
        "value": "",
        "options": ["Normal", "No spikes", "Noisy"],
        "status": ["Pass", "Fail", "Fail"],
        "type": "dropdown",
    }
    if visualization_output is not None:
        visualization_rec = visualization_output.get(recording_name)
        if visualization_rec:
            raw_data_value_with_flags["timeseries_link"] = visualization_rec.get("timeseries")

    raw_data_metric = QCMetric(
        name=f"Raw data {recording_name} ",
        description=f"Evaluation of {recording_name} raw data",
        value=raw_data_value_with_flags,
        reference=str(raw_traces_path),
        status_history=[status_pending],
    )
    metrics["Raw Data"] = [raw_data_metric]

    print("Generating PSD metrics")
    fig_psd_wide, fig_psd_hf, fig_psd_lf = plot_psd(recording, recording_lfp=recording_lfp)
    psd_wide_path = recording_fig_folder / "psd_wide.png"
    psd_hf_path = recording_fig_folder / "psd_hf.png"
    psd_lf_path = recording_fig_folder / "psd_lf.png"

    fig_psd_wide.savefig(psd_wide_path, dpi=300)
    fig_psd_hf.savefig(psd_hf_path, dpi=300)
    fig_psd_lf.savefig(psd_lf_path, dpi=300)

    if relative_to is not None:
        psd_wide_path = psd_wide_path.relative_to(relative_to)
        psd_hf_path = psd_hf_path.relative_to(relative_to)
        psd_lf_path = psd_lf_path.relative_to(relative_to)

    psd_wide_metric = QCMetric(
        name=f"PSD (Wide Band) {recording_name} ",
        description=f"Evaluation of {recording_name} wide-band power spectrum density",
        reference=str(psd_wide_path),
        value=None,
        status_history=[status_pass],
    )

    hf_value_with_flags = {
        "value": "",
        "options": ["No contamination", "High frequency contamination"],
        "status": ["Pass", "Fail"],
        "type": "dropdown",
    }

    psd_hf_metric = QCMetric(
        name=f"PSD (High Frequency) {recording_name} ",
        description=f"Evaluation of {recording_name} high-frequency power spectrum density",
        reference=str(psd_hf_path),
        value=hf_value_with_flags,
        status_history=[status_pending],
    )
    lf_value_with_flags = {
        "value": "",
        "options": ["No contamination", "Line (60 Hz) contamination"],
        "status": ["Pass", "Fail"],
        "type": "dropdown",
    }

    psd_lf_metric = QCMetric(
        name=f"PSD (Low Frequency) {recording_name}",
        description=f"Evaluation of {recording_name} low-frequency power spectrum density",
        reference=str(psd_hf_path),
        value=lf_value_with_flags,
        status_history=[status_pending],
    )
    metrics["PSD"] = [psd_wide_metric, psd_hf_metric, psd_lf_metric]

    print("Generating NOISE metrics")
    fig_rms = plot_rms_by_depth(recording, recording_preprocessed)
    rms_path = recording_fig_folder / "rms.png"
    fig_rms.savefig(rms_path, dpi=300)
    if relative_to is not None:
        rms_path = rms_path.relative_to(relative_to)
    rms_metric = QCMetric(
        name=f"RMS {recording_name}",
        description=f"Evaluation of {recording_name} RMS",
        reference=str(rms_path),
        value=None,
        status_history=[status_pass],
    )
    metrics["Noise"] = [rms_metric]

    # Bad channel detection out of brain, noisy, silent
    if processing is not None:
        try:
            data_processes = processing.processing_pipeline.data_processes
            for data_process in data_processes:
                params = data_process.parameters.model_dump()
                outputs = data_process.outputs.model_dump()
                if (
                    data_process.name == "Ephys preprocessing"
                    and params.get("recording_name") is not None
                    and params.get("recording_name") == recording_name
                ):
                    channel_labels = np.array(outputs.get("channel_labels"))
                    if channel_labels is not None:
                        channel_labels_value = {
                            "good": int(np.sum(channel_labels == "good")),
                            "noise": int(np.sum(channel_labels == "noise")),
                            "dead": int(np.sum(channel_labels == "dead")),
                            "out": int(np.sum(channel_labels == "out")),
                        }
                        channel_labels_metric = QCMetric(
                            name=f"Bad channel detection {recording_name}",
                            description=f"Evaluation of {recording_name} bad channel detection",
                            value=channel_labels_value,
                            status_history=[status_pass],
                        )
                        metrics["Noise"].append(channel_labels_metric)
        except:
            print(f"Failed to load bad channel labels for {recording_name}")

    return metrics

def get_probe_unit_yield_qc(probe_analyzer: si.SortingAnalyzer, probe_experiment_name: str, output_path: pathlib.Path) -> QCMetric:
    number_of_units = probe_analyzer.get_num_units()

    decoder_label = probe_analyzer.sorting.get_property('decoder_label')
    number_of_sua_units = decoder_label[decoder_label == 'sua'].shape[0]
    number_of_mua_units = decoder_label[decoder_label == 'mua'].shape[0]
    number_of_noise_units = decoder_label[decoder_label == 'noise'].shape[0]
    assert (number_of_mua_units + number_of_noise_units + number_of_sua_units) == number_of_units, f'Total number of units {number_of_units} does not equal {number_of_sua_units} sua units + {number_of_mua_units} mua units + {number_of_noise_units} noise units'

    default_qc = probe_analyzer.sorting.get_property('default_qc')
    number_of_good_units = default_qc[default_qc == True].shape[0]

    quality_metrics = probe_analyzer.get_extension('quality_metrics').get_data()

    fig, ax = plt.subplots(1, 5, figsize=(15,6))

    ax[0].hist(quality_metrics['isi_violations_ratio'], bins=20)
    ax[0].set_xscale('log')
    ax[0].set_title(f'ISI Violations')
    ax[0].set_xlabel('violation rate')

    ax[1].hist(quality_metrics['amplitude_cutoff'], bins=20, density=True)
    ax[1].set_title(f'Amplitude Cutoff')
    ax[1].set_xlabel('frequency')

    ax[2].hist(quality_metrics['presence_ratio'], bins=20, density=True)
    ax[2].set_title(f'Presence Ratio')
    ax[2].set_xlabel('fraction of session')

    channel_indices = np.array(list(si.get_template_extremum_channel(probe_analyzer, mode='peak_to_peak', outputs='index').values()))
    amplitudes = probe_analyzer.sorting.get_property('Amplitude')
    amplitudes[amplitudes > 800] = 800
    df_amplitudes_channel_indices = pd.DataFrame({'amplitude': amplitudes, 'peak_channel': channel_indices})
    mean_amplitude_by_peak_channel = df_amplitudes_channel_indices.groupby('peak_channel').mean()

    colors = {'sua': 'blue', 'mua': 'green', 'noise': 'orange'}
    for label in np.unique(decoder_label):
        mask = decoder_label == label
        ax[3].scatter(amplitudes[mask], channel_indices[mask], c=colors[label], label=label)

    xnew = np.linspace(mean_amplitude_by_peak_channel.index.min(), mean_amplitude_by_peak_channel.index.max(), 100) 
    spl = make_interp_spline(mean_amplitude_by_peak_channel.index.tolist(), mean_amplitude_by_peak_channel['amplitude'], k=3)  # type: BSpline
    smoothed = spl(xnew)

    #ax[3].plot(smoothed, xnew, c='r', alpha=0.7)
    ax[3].set_title('Unit Amplitude By Depth')
    ax[3].set_xlabel('uV')
    ax[3].set_ylabel('Channel Index')
    ax[3].legend()

    firing_rate = np.array(quality_metrics['firing_rate'].tolist())
    firing_rate[firing_rate > 50] = 50
    df_firing_rate_channel_indices = pd.DataFrame({'firing_rate': firing_rate, 'peak_channel': channel_indices})
    mean_firing_rate_by_peak_channel = df_firing_rate_channel_indices.groupby('peak_channel').mean()

    for label in np.unique(decoder_label):
        mask = decoder_label == label
        ax[4].scatter(firing_rate[mask], channel_indices[mask], c=colors[label], label=label)

    ax[4].plot(mean_firing_rate_by_peak_channel['firing_rate'], mean_amplitude_by_peak_channel.index.tolist(), c='r', alpha=0.7)
    ax[4].set_title('Unit Firing Rate By Depth')
    ax[4].set_xlabel('spikes/s')
    ax[4].set_ylabel('Channel Index')

    fig.suptitle(f'{probe_experiment_name} Unit Metrics Yield')
    plt.tight_layout()
    fig.savefig(output_path / 'unit_yield.png')

    visualization_json_path = tuple(DATA_PATH.glob('*/visualization_output.json'))
    if not visualization_json_path:
        raise FileNotFoundError('No visualization json found in sorting result')

    with open(visualization_json_path[0]) as f:
        json_visualization = json.load(f)
    
    if probe_experiment_name in json_visualization:
        sorting_summary_link = json_visualization[probe_experiment_name]['sorting_summary']
    else:
        sorting_summary_link = f'No sorting summary link found for {probe_experiment_name}. Pipeline possibly errored'

    metric_values = {'number_of_total_units': number_of_units, 
                    'number_of_sua_units': number_of_sua_units, 'number_of_mua_units': number_of_mua_units, 'number_of_noise_units': number_of_noise_units,
                    'number_of_units_passing_quality_metrics': number_of_good_units,
                    'sorting_summary_visualization_link': sorting_summary_link}

    value_with_options = {
        "unit quality metrics": metric_values,
        "options": ["Good", "Low Yield", "High Noise"],
        "status": [
            "Pass",
            "Fail",
            "Fail"
        ],
        "type": "checkbox"
    }
    # TODO: Update reference path
    yield_metric = QCMetric(name=f'{probe_experiment_name} unit metrics yield', description=f'Evaluation of {probe_experiment_name} unit metrics yield',
                            reference=f'/{probe_experiment_name}/unit_yield.png', value=value_with_options,
                            status_history=[QCStatus(evaluator='aind-ephys-qc', timestamp=datetime.now(), status=Status.PENDING)])
    plt.close(fig)

    return yield_metric

def get_probe_firing_rate_qc(probe_analyzer: si.SortingAnalyzer, probe_experiment_name: str, output_path: pathlib.Path, bin_interval: int=1) -> QCMetric:
    spike_vector = probe_analyzer.sorting.to_spike_vector()
    spike_times = np.array([spike[0] for spike in spike_vector]) / probe_analyzer.sampling_frequency
    spike_times_hist = np.histogram(spike_times, bins=np.arange(np.floor(np.min(spike_times)), np.ceil(np.max(spike_times)), bin_interval))

    plt.figure(figsize=(10, 6))
    plt.plot(spike_times_hist[0])
    plt.title(f'Firing rate for {probe_experiment_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Firing rate')

    plt.savefig(output_path / 'firing_rate.png')
    plt.close()

    # TODO: Update reference path
    firing_rate_metric = QCMetric(name=f'{probe_experiment_name} firing rate image', description=f'Evaluation of {probe_experiment_name} firing rate',
                        reference=f'/{probe_experiment_name}/firing_rate.png', 
                        value=CheckboxMetric(value='Firing Rate Evaluation', options=['No problems detected', 'Seizure', 'Firing Rate Gap'],
                        status=[Status.PASS, Status.FAIL, Status.FAIL]),
                        status_history=[QCStatus(evaluator='aind-ephys-qc', timestamp=datetime.now(), status=Status.PENDING)])
    
    return firing_rate_metric

