from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter


import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw

from aind_data_schema.core.processing import Processing
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status


def load_preprocessed_recording(preprocessed_json_file, session_name, ecephys_folder, data_folder):
    recording_preprocessed = None
    try:
        recording_preprocessed = si.load_extractor(preprocessed_json_file, base_folder=data_folder)
    except Exception as e:
        pass
    if recording_preprocessed is None:
        try:
            json_str = json.dumps(json.load(open(preprocessed_json_file)))
            if "ecephys_session" in json_str:
                json_str_remapped = json_str.replace("ecephys_session", ecephys_folder.name)
            elif ecephys_folder.name != session_name and session_name in json_str:
                json_str_remapped = json_str.replace(session_name, ecephys_folder.name)
            recording_dict = json.loads(json_str_remapped)
            recording_preprocessed = si.load_extractor(recording_dict, base_folder=data_folder)
        except:
            pass
    if recording_preprocessed is None:
        print(f"Error loading preprocessed data...")
    return recording_preprocessed


def load_processing_metadata(processing_json):
    with open(processing_json) as f:
        processing_dict = json.load(f)
    processing_dict.pop("schema_version")
    data_processes = processing_dict["processing_pipeline"]["data_processes"]
    new_data_processes = []
    for dp in data_processes:
        if isinstance(dp, list):
            for dpp in dp:
                new_data_processes.append(dpp)
        else:
            new_data_processes.append(dp)
    processing_dict["processing_pipeline"]["data_processes"] = new_data_processes
    return Processing(**processing_dict)


def _get_fig_axs(nrows, ncols, subplot_figsize=(3, 3)):
    figsize = (subplot_figsize[0] * ncols, subplot_figsize[1] * nrows)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
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
    freq_lfp: float = 500,
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
                clim=(-50, 50),
            )
            sw.plot_traces(
                recording_lfp,
                time_range=[t_start, t_start + duration_s],
                mode="map",
                return_scaled=True,
                with_colorbar=True,
                ax=ax_lfp,
                clim=(-300, 300),
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
        target_lfp_sampling = int(1.5 * freq_lf_filt)
        decimate_factor = int(recording.sampling_frequency / target_lfp_sampling)
        recording_lfp = spre.decimate(recording_lfp, decimate_factor)
    else:
        recording_lfp = spre.bandpass_filter(recording_lfp, freq_min=0.1, freq_max=freq_lf_filt)
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

    fig_rms.subplots_adjust(top=0.8)
    return fig_rms, ax_rms


def generate_raw_qc(
    recording: si.BaseRecording,
    recording_name: str,
    output_qc_path: Path,
    relative_to: Path | None = None,
    recording_lfp: si.BaseRecording | None = None,
    recording_preprocessed: si.BaseRecording | None = None,
    processing: Processing | None = None,
    visualization_output: dict | None = None,
) -> dict[str : list[QCMetric]]:
    """
    Generate raw data quality control metrics for a given recording.

    Parameters
    ----------
    recording : BaseRecording
        The recording object.
    recording_name : str
        The name of the recording.
    output_qc_path : Path
        The output path for the quality control.
    relative_to : Path | None, default: None
        The relative path to the output path.
    recording_lfp : BaseRecording | None, default: None
        The LFP recording object.
    recording_preprocessed : BaseRecording | None, default: None
        The preprocessed recording object.
    processing : Processing | None, default: None
        The processing metadata object.
    visualization_output : dict | None, default: None
        The visualization output dict.

    Returns
    -------
    dict[str : list[QCMetric]]:
        The quality control metrics.
    """
    metrics = {}
    recording_fig_folder = output_qc_path / recording_name
    recording_fig_folder.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    status_pending = QCStatus(evaluator="", status=Status.PENDING, timestamp=now)
    status_pass = QCStatus(evaluator="", status=Status.PASS, timestamp=now)

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
    fig_rms, ax_rms = plot_rms_by_depth(recording, recording_preprocessed)
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
                        metric_values = {
                            "good": int(np.sum(channel_labels == "good")),
                            "noise": int(np.sum(channel_labels == "noise")),
                            "dead": int(np.sum(channel_labels == "dead")),
                            "out": int(np.sum(channel_labels == "out")),
                        }
                        metric_values_str = None
                        for metric_name, metric_value in metric_values.items():
                            if metric_values_str is None:
                                metric_values_str = f"{metric_name}: {metric_value}"
                            else:
                                metric_values_str += f"\n{metric_name}: {metric_value}"
                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                        ax_rms.text(0.2, 0.9, metric_values_str, transform=fig_rms.transFigure, fontsize=14,
                                    verticalalignment='top', bbox=props)
                        
        except:
            print(f"Failed to load bad channel labels for {recording_name}")

    rms_path = recording_fig_folder / "rms.png"
    fig_rms.savefig(rms_path, dpi=300)
    if relative_to is not None:
        rms_path = rms_path.relative_to(relative_to)

    # make metric for qc json
    value_with_options = {
        "value": "",
        "options": ["Good", "No out channels detected"],
        "status": ["Pass", "Fail"],
        "type": "dropdown",
    }
    rms_metric = QCMetric(
        name=f"RMS {recording_name}",
        description=f"Evaluation of {recording_name} RMS",
        reference=str(rms_path),
        value=value_with_options,
        status_history=[status_pass],
    )
    metrics["Noise"] = [rms_metric]

    return metrics


def generate_drift_qc(
    recording: si.BaseRecording,
    recording_name: str,
    motion_path: Path,
    output_qc_path: Path,
    relative_to: Path | None = None,
) -> QCMetric:
    """
    Generate drift metrics for a given sorting result.

    Parameters
    ----------
    recording : BaseRecording
        The sorting analyzer.
    recording_name : str
        The name of the recording.
    motion_path: Path,
        The path of the recording's motion folder.
    output_qc_path : Path
        The output path for the quality control.

    Returns
    -------
    QCMetric:
        The quality control metric for drift.
    """

    print("Generating DRIFT metric")

    recording_fig_folder = output_qc_path / recording_name
    recording_fig_folder.mkdir(parents=True, exist_ok=True)

    # open displacement arrays
    if not motion_path.is_dir():
        print(f"\tMotion not found for {recording_name}")
        return {}

    motion_info = spre.load_motion_info(motion_path)
    all_peaks = motion_info["peaks"]
    all_peak_locations = motion_info["peak_locations"]
    motion = motion_info["motion"]
    spatial_bins = motion.spatial_bins_um

    fig_drift, axs_drift = plt.subplots(
        ncols=recording.get_num_segments(), figsize=(10,10)
    )
    y_locs = recording.get_channel_locations()[:, 1]
    sampling_frequency = recording.sampling_frequency
    depth_lim = [np.min(y_locs), np.max(y_locs)]

    for segment_index in range(recording.get_num_segments()):
        if recording.get_num_segments() == 1:
            ax_drift = axs_drift
        else:
            ax_drift = axs_drift[segment_index]

        segment_mask = all_peaks["segment_index"] == segment_index
        peaks_to_plot = all_peaks[segment_mask]
        peak_locations_to_plot = all_peak_locations[segment_mask]

        _ = sw.plot_drift_raster_map(
            sorting_analyzer=None,
            peaks=peaks_to_plot,
            peak_locations=peak_locations_to_plot,
            recording=recording,
            sampling_frequency=sampling_frequency,
            segment_index=segment_index,
            depth_lim=depth_lim,
            clim=(-200, 0),
            cmap="Greys_r",
            scatter_decimate=10,
            alpha=0.3,
            ax=ax_drift,
        )
        ax_drift.spines["top"].set_visible(False)
        ax_drift.spines["right"].set_visible(False)

        displacement_arr = motion.displacement[segment_index]
        temporal_bins = motion.temporal_bins_s[segment_index]
        
        # calculate cumulative_drift and max displacement
        drift_ptps = np.ptp(displacement_arr, axis=0)
        cumulative_drifts = np.sum(displacement_arr, axis=0)
        max_displacement_index = np.argmax(drift_ptps)
        max_displacement = np.round(drift_ptps[max_displacement_index], 2)
        depth_at_max_displacement = int(spatial_bins[max_displacement_index])

        max_cumulative_drift_index = np.argmax(cumulative_drifts)
        max_cumulative_drift = np.round(cumulative_drifts[max_cumulative_drift_index], 2)
        depth_at_max_cumulative_drift = int(spatial_bins[max_cumulative_drift_index])

        ax_drift.plot(temporal_bins, displacement_arr + spatial_bins, color='red', alpha=0.5)
        ax_drift.set_title(
            f"Max displacement: {max_displacement} $\mu m$ (depth: {depth_at_max_displacement} ) $\\mu m$\n"
            f"Max cumulative drift: {max_cumulative_drift} $\mu m$ (depth: {depth_at_max_cumulative_drift} ) $\\mu m$\n"
        )

    drift_map_path = recording_fig_folder / "drift_map.png"
    fig_drift.savefig(drift_map_path, dpi=300)
    if relative_to is not None:
        drift_map_path = drift_map_path.relative_to(relative_to)

    # make metric for qc json
    value_with_options = {
        "value": "",
        "options": ["Good", "High Drift"],
        "status": ["Pass", "Fail"],
        "type": "checkbox",
    }
    drift_metric = QCMetric(
        name=f"Probe Drift - {recording_name}",
        description=f"Evaluation of {recording_name} probe drift",
        reference=str(drift_map_path),
        value=value_with_options,
        status_history=[QCStatus(evaluator="", status=Status.PENDING, timestamp=datetime.now())],
    )
    drift_metrics = {"Drift": [drift_metric]}

    return drift_metrics


def generate_units_qc(
    sorting_analyzer: si.SortingAnalyzer | None,
    recording_name: str,
    output_qc_path: Path,
    relative_to: Path | None = None,
    visualization_output: dict | None = None,
    max_amplitude_for_visualization: float = 5000,
    max_firing_rate_for_visualization: float = 50,
    bin_duration_hist_s: float = 1,
) -> dict[str : list[QCMetric]]:
    """
    Generate unit yield metrics for a given sorting result.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer.
    recording_name : str
        The name of the recording.
    output_qc_path : Path
        The output path for the quality control.
    relative_to : Path | None, default: None
        The relative path to the output path.
    visualization_output : dict | None, default: None
        The visualization output dict.
    max_amplitude_for_visualization : float, default: 5000
        The maximum amplitude used for plotting.
    max_firing_rate_for_visualization : float, default: 50
        The maximum firing rate used for plotting.
    bin_duration_hist_s : float, default: 1
        The duration of the histogram bins.

    Returns
    -------
    dict[str : list[QCMetric]]:
        The quality control metrics.
    """
    metrics = {}
    recording_fig_folder = output_qc_path / recording_name
    recording_fig_folder.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    status_pending = QCStatus(evaluator="", status=Status.PENDING, timestamp=now)

    print("Generating UNIT YIELD metric")
    if sorting_analyzer is None:
        print(f"\tNo sorting analyzer found for {recording_name}")
        return metrics

    number_of_units = sorting_analyzer.get_num_units()

    decoder_label = sorting_analyzer.sorting.get_property("decoder_label")
    number_of_sua_units = int(np.sum(decoder_label == "sua"))
    number_of_mua_units = int(np.sum(decoder_label == "mua"))
    number_of_noise_units = int(np.sum(decoder_label == "noise"))

    default_qc = sorting_analyzer.sorting.get_property("default_qc")
    number_of_good_units = int(np.sum(default_qc == True))

    quality_metrics = sorting_analyzer.get_extension("quality_metrics").get_data()

    fig_yield, axs_yield = plt.subplots(2, 3, figsize=(15, 10))

    bins = np.linspace(0, 2, 20)
    ax_isi = axs_yield[0, 0]
    ax_isi.hist(quality_metrics["isi_violations_ratio"], bins=bins, density=True)
    ax_isi.set_xscale("log")
    ax_isi.set_title(f"ISI Violations Ratio")
    ax_isi.spines[["top", "right"]].set_visible(False) 

    ax_amp_cutoff = axs_yield[0, 1]
    ax_amp_cutoff.hist(quality_metrics["amplitude_cutoff"], bins=20, density=True)
    ax_amp_cutoff.set_title(f"Amplitude Cutoff")
    ax_amp_cutoff.spines[["top", "right"]].set_visible(False) 

    ax_presence_ratio = axs_yield[0, 2]
    ax_presence_ratio.hist(quality_metrics["presence_ratio"], bins=20, density=True)
    ax_presence_ratio.set_title(f"Presence Ratio")
    ax_presence_ratio.spines[["top", "right"]].set_visible(False) 

    channel_indices = np.array(
        list(si.get_template_extremum_channel(sorting_analyzer, mode="peak_to_peak", outputs="index").values())
    )
    channel_depths = sorting_analyzer.get_channel_locations()[channel_indices, 1]
    amplitudes = np.array(list(si.get_template_extremum_amplitude(sorting_analyzer, mode="peak_to_peak").values()))
    amplitudes[amplitudes > max_amplitude_for_visualization] = max_amplitude_for_visualization
    df_amplitudes_depths = pd.DataFrame({"amplitude": amplitudes, "channel_depth": channel_depths})
    mean_amplitude_by_depth = df_amplitudes_depths.groupby("channel_depth").mean()

    colors = {"sua": "green", "mua": "orange", "noise": "red"}
    ax_amplitudes = axs_yield[1, 0]
    for label in np.unique(decoder_label):
        mask = decoder_label == label
        ax_amplitudes.scatter(amplitudes[mask], channel_depths[mask], c=colors[label], label=label, alpha=0.4)

    smoothed_amplitude = savgol_filter(mean_amplitude_by_depth["amplitude"], 10, 2)
    ax_amplitudes.plot(smoothed_amplitude, mean_amplitude_by_depth.index.tolist(), c="r")
    ax_amplitudes.set_title("Unit Amplitude By Depth")
    ax_amplitudes.set_xlabel("Amplitude ($\\mu V$)")
    ax_amplitudes.set_ylabel("Depth ($\\mu m$)")
    ax_amplitudes.legend()
    ax_amplitudes.spines[["top", "right"]].set_visible(False) 

    ax_fr = axs_yield[1, 1]
    firing_rate = np.array(quality_metrics["firing_rate"].tolist())
    firing_rate[firing_rate > max_firing_rate_for_visualization] = max_firing_rate_for_visualization
    df_firing_rate_depths = pd.DataFrame({"firing_rate": firing_rate, "channel_depth": channel_depths})
    mean_firing_rate_by_depth = df_firing_rate_depths.groupby("channel_depth").mean()
    
    for label in np.unique(decoder_label):
        mask = decoder_label == label
        ax_fr.scatter(firing_rate[mask], channel_depths[mask], c=colors[label], label=label, alpha=0.4)

    smoothed_firing_rate = savgol_filter(mean_firing_rate_by_depth["firing_rate"], 10, 2)
    ax_fr.plot(smoothed_firing_rate, mean_firing_rate_by_depth.index.tolist(), c="r")
    ax_fr.set_title("Unit Firing Rate By Depth")
    ax_fr.set_xlabel("Firing rate (Hz)")
    ax_fr.set_ylabel("Depth ($\\mu m$)")
    ax_fr.legend()
    ax_fr.spines[["top", "right"]].set_visible(False) 

    ax_text = axs_yield[1, 2]
    ax_text.axis("off")

    metric_values = {
        "# units": number_of_units,
        "# SUA": number_of_sua_units,
        "# MUA": number_of_mua_units,
        "# noise": number_of_noise_units,
        "# passing default QC": number_of_good_units,
    }

    metric_values_str = None
    for metric_name, metric_value in metric_values.items():
        if metric_values_str is None:
            metric_values_str = f"{metric_name}: {metric_value}"
        else:
            metric_values_str += f"\n{metric_name}: {metric_value}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax_text.text(0, 1, metric_values_str, transform=ax_text.transAxes, fontsize=14,
                 verticalalignment='top', bbox=props)

    fig_yield.suptitle("Unit Metrics Yield", fontsize=15)
    fig_yield.tight_layout()
    unit_yield_path = recording_fig_folder / "unit_yield.png"
    fig_yield.savefig(unit_yield_path, dpi=300)
    
    if relative_to is not None:
        unit_yield_path = unit_yield_path.relative_to(relative_to)

    if visualization_output is not None:
        if recording_name in visualization_output:
            sorting_summary_link = visualization_output[recording_name].get("sorting_summary")
        else:
            sorting_summary_link = f"No sorting summary link found for {recording_name}."

    value_with_options = {
        "value": "",
        "options": ["Good", "Low Yield", "High Noise"],
        "status": ["Pass", "Fail", "Fail"],
        "type": "checkbox",
    }
    yield_metric = QCMetric(
        name=f"Unit Metrics Yield - {recording_name}",
        description=f"Evaluation of {recording_name} unit metrics yield. Sortingview link: {sorting_summary_link}",
        reference=str(unit_yield_path),
        value=value_with_options,
        status_history=[status_pending],
    )
    metrics["Unit Yield"] = [yield_metric]

    print("Generating FIRING RATE metric")
    num_segments = sorting_analyzer.get_num_segments()
    fig_fr, axs_fr = _get_fig_axs(num_segments, 1, subplot_figsize=(5, 3))
    spike_vector = sorting_analyzer.sorting.to_spike_vector()
    if sorting_analyzer.has_recording():
        recording = sorting_analyzer.recording
    else:
        recording = None

    for segment_index in range(num_segments):
        spike_vector_segment = spike_vector[spike_vector["segment_index"] == segment_index]

        if recording is not None:
            times = recording.get_times(segment_index=segment_index)
            spike_times = times[spike_vector_segment["sample_index"]]
        else:
            spike_times = spike_vector_segment["sample_index"] / sorting_analyzer.sampling_frequency

        duration = sorting_analyzer.get_num_samples(segment_index=segment_index) / sorting_analyzer.sampling_frequency
        num_bins = int(np.ceil(duration / bin_duration_hist_s))

        spike_times_hist, bin_edges = np.histogram(spike_times, bins=num_bins)
        bin_centers = bin_edges[:-1] + bin_duration_hist_s / 2
        axs_fr[segment_index, 0].plot(bin_centers, spike_times_hist)

        axs_fr[segment_index, 0].set_title(f"Population firing rate")
        axs_fr[segment_index, 0].set_xlabel("Time (s)")
        axs_fr[segment_index, 0].set_ylabel("Firing rate (Hz)")
        axs_fr[segment_index, 0].spines[["top", "right"]].set_visible(False)

    fig_fr.tight_layout()
    firing_rate_path = recording_fig_folder / "firing_rate.png"
    fig_fr.savefig(firing_rate_path, dpi=300)
    if relative_to is not None:
        firing_rate_path = firing_rate_path.relative_to(relative_to)

    value_with_options = {
        "value": "",
        "options": ["No problems detected", "Seizure", "Firing Rate Gap"],
        "status": ["Pass", "Fail", "Fail"],
        "type": "checkbox",
    }
    firing_rate_metric = QCMetric(
        name=f"Firing rate - {recording_name}",
        description=f"Evaluation of {recording_name} firing rate",
        reference=str(firing_rate_path),
        value=value_with_options,
        status_history=[status_pending],
    )
    metrics["Firing Rate"] = [firing_rate_metric]

    return metrics
