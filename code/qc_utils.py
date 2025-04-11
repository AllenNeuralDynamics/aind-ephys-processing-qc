from __future__ import annotations

from pathlib import Path
import logging
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import welch, savgol_filter

import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw

from aind_data_schema.core.processing import Processing
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status
from aind_qcportal_schema.metric_value import CurationMetric

ISI_VISUAL_AREAS = ["VISp", "VISa", "VISal", "VISam", "VISl", "VISli", "VISpl", "VISpm", "VISpor", "VISrl", "VIS"]

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


def load_preprocessed_recording(preprocessed_json_file, session_name, ecephys_folder, data_folder):
    recording_preprocessed = None
    if preprocessed_json_file.is_file():
        try:
            recording_preprocessed = si.load(preprocessed_json_file, base_folder=data_folder)
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
                recording_preprocessed = si.load(recording_dict, base_folder=data_folder)
            except:
                pass
        if recording_preprocessed is None:
            logging.info(f"Error loading preprocessed data...")
    else:
        logging.info(f"Preprocessed recording not found...")
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
    fig_rms, ax_rms = plt.subplots(figsize=(5, 8))

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
    recording_fig_folder = output_qc_path
    recording_fig_folder.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    status_pending = QCStatus(evaluator="", status=Status.PENDING, timestamp=now)
    status_pass = QCStatus(evaluator="", status=Status.PASS, timestamp=now)

    logging.info("Generating RAW DATA metrics")
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
        if recording_name in visualization_output:
            timeseries_link = visualization_output[recording_name].get("timeseries")
            timeseries_str = f"[Sortingview link]({timeseries_link})"
        else:
            timeseries_str = f"No timeseries link found for {recording_name}."
    else:
        timeseries_str = "No visualization output found in results."

    raw_data_metric = QCMetric(
        name=f"Raw data {recording_name} ",
        description=f"Evaluation of {recording_name} raw data. {timeseries_str}",
        value=raw_data_value_with_flags,
        reference=str(raw_traces_path),
        status_history=[status_pending],
    )
    metrics["Raw Data"] = [raw_data_metric]

    logging.info("Generating PSD metrics")
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
        reference=str(psd_lf_path),
        value=lf_value_with_flags,
        status_history=[status_pending],
    )
    metrics["PSD"] = [psd_wide_metric, psd_hf_metric, psd_lf_metric]

    logging.info("Generating NOISE metrics")
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
                        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
                        ax_rms.text(
                            0.2,
                            0.9,
                            metric_values_str,
                            transform=fig_rms.transFigure,
                            fontsize=14,
                            verticalalignment="top",
                            bbox=props,
                        )

        except:
            logging.info(f"Failed to load bad channel labels for {recording_name}")

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
    processing: Processing | None = None
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
    processing : Processing | None, default: None
        The processing metadata object.

    Returns
    -------
    QCMetric:
        The quality control metric for drift.
    """

    logging.info("Generating DRIFT metric")

    recording_fig_folder = output_qc_path
    recording_fig_folder.mkdir(parents=True, exist_ok=True)

    # open displacement arrays
    if not motion_path.is_dir():
        logging.info(f"\tMotion not found for {recording_name}")
        return {}

    motion_info = spre.load_motion_info(motion_path)
    all_peaks = motion_info["peaks"]
    all_peak_locations = motion_info["peak_locations"]
    motion = motion_info["motion"]
    spatial_bins = motion.spatial_bins_um

    fig_drift, axs_drift = plt.subplots(ncols=recording.get_num_segments(), figsize=(10, 10))
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
        displacements_diff_arr = np.diff(displacement_arr, axis=0)
        cumulative_drifts = np.sum(displacements_diff_arr, axis=0)
        max_displacement_index = np.argmax(drift_ptps)
        max_displacement = np.round(drift_ptps[max_displacement_index], 2)
        depth_at_max_displacement = int(spatial_bins[max_displacement_index])

        max_cumulative_drift_index = np.argmax(cumulative_drifts)
        max_cumulative_drift = np.round(cumulative_drifts[max_cumulative_drift_index], 2)
        depth_at_max_cumulative_drift = int(spatial_bins[max_cumulative_drift_index])

        ax_drift.plot(temporal_bins, displacement_arr + spatial_bins, color="red", alpha=0.5)

    # Plot surface channel line
    surface_channel_y_position = None
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
                        channel_ids_out = recording.channel_ids[channel_labels == 'out']

                        if not channel_ids_out:
                            logging.info(f"No out channels detected for {recording_name}")
                        else:
                            surface_channel_id = channel_ids_out[0]
                            surface_channel_index = recording.channel_ids.tolist().index(surface_channel_id)
                            surface_channel_y_position = y_locs[surface_channel_index]
                            ax_drift.axhline(y=surface_channel_y_position, c='g')
        except:
            logging.info(f"Failed to load bad channel labels for {recording_name}")

    if surface_channel_y_position is not None:
        ax_drift.set_title(
            f"Max displacement: {max_displacement} $\mu m$ (depth: {depth_at_max_displacement} ) $\\mu m$\n"
            f"Max cumulative drift: {max_cumulative_drift} $\mu m$ (depth: {depth_at_max_cumulative_drift} ) $\\mu m$\n"
            f"Surface channel location (green line) at {y_position} $\\mu m$\n"
        )
    else:
        ax_drift.set_title(
            f"Max displacement: {max_displacement} $\mu m$ (depth: {depth_at_max_displacement} ) $\\mu m$\n"
            f"Max cumulative drift: {max_cumulative_drift} $\mu m$ (depth: {depth_at_max_cumulative_drift} ) $\\mu m$\n"
            f"No surface channel detected"
        )

    drift_map_path = recording_fig_folder / "drift_map.png"
    fig_drift.savefig(drift_map_path, dpi=300)
    if relative_to is not None:
        drift_map_path = drift_map_path.relative_to(relative_to)

    # make metric for qc json
    value_with_options = {
        "value": "",
        "options": ["Good", "High drift", "Bad drift estimation"],
        "status": ["Pass", "Fail", "Fail"],
        "type": "dropdown",
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


def generate_event_qc(
    recording: si.BaseRecording,
    recording_name: str,
    output_qc_path: Path,
    relative_to: Path | None = None,
    event_dict: dict | None = None,
    event_keys: list[str] = ["licktime", "optogeneticstime"],
    t_cutout_saturation_s: float = 0.002,
    num_saturation_events_to_plot: int = 3,
    **job_kwargs,
):
    """
    Generate event metrics for a given recording.
    The event metrics include saturation and responses to certain events.
    """
    recording_fig_folder = output_qc_path
    recording_fig_folder.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    status_pending = QCStatus(evaluator="", status=Status.PENDING, timestamp=now)
    status_pass = QCStatus(evaluator="", status=Status.PASS, timestamp=now)

    logging.info("Generating SATURATION metric")

    part_number = recording.get_annotation("probes_info")[0].get("part_number")

    metrics = {}
    if part_number is None:
        logging.info(f"\tNo part number found for {recording_name}. Cannot generate saturation metrics.")
    else:
        saturation_threshold_uv = saturation_thresholds_uv[part_number]
        pos_evts, neg_evts = find_saturation_events(recording, saturation_threshold_uv, **job_kwargs)

        clim = saturation_threshold_uv / 2
        recording_ps = spre.phase_shift(recording)

        # make figure for saturation events
        saturation_status = status_pending
        if len(pos_evts) > 0 and len(neg_evts) > 0:
            nrows = min(num_saturation_events_to_plot, max(len(pos_evts), len(neg_evts)))
            fig_sat_samples, axs_sat = _get_fig_axs(ncols=2, nrows=nrows)
            pos_ax_col = 0
            neg_ax_col = 1
        elif len(pos_evts) > 0:
            nrows = min(num_saturation_events_to_plot, len(pos_evts))
            fig_sat_samples, axs_sat = _get_fig_axs(ncols=1, nrows=nrows)
            pos_ax_col = 0
        elif len(neg_evts) > 0:
            nrows = min(num_saturation_events_to_plot, len(neg_evts))
            fig_sat_samples, axs_sat = _get_fig_axs(ncols=1, nrows=nrows)
            neg_ax_col = 0
        else:
            nrows = 1
            fig_sat_samples, axs_sat = _get_fig_axs(ncols=1, nrows=1)
            axs_sat[0, 0].axis("off")
            saturation_status = status_pass

        if len(pos_evts) > 0:
            logging.info(f"\tFound {len(pos_evts)} positive saturation events!")
            if len(pos_evts) > num_saturation_events_to_plot:
                random_saturation_events = pos_evts[
                    np.random.choice(np.arange(len(pos_evts)), size=num_saturation_events_to_plot, replace=False)
                ]
                random_saturation_events = random_saturation_events[np.argsort(random_saturation_events["sample_index"])]
            else:
                random_saturation_events = pos_evts
            for i_r, r in enumerate(random_saturation_events):
                ax = axs_sat[i_r, pos_ax_col]
                t0 = recording.sample_index_to_time(r["sample_index"])
                w = sw.plot_traces(
                    recording_ps,
                    time_range=[t0 - t_cutout_saturation_s, t0 + t_cutout_saturation_s],
                    clim=(-clim, clim),
                    ax=ax,
                )
                ax.set_xticks([t0 - t_cutout_saturation_s, t0, t0 + t_cutout_saturation_s])
                ax.set_xticklabels([- t_cutout_saturation_s * 1000, 0, t_cutout_saturation_s  * 1000])
                if i_r == 0:
                    ax.set_title(f"Positive\n@{np.round(t0, 2)}")
                else:
                    ax.set_title(f"@{np.round(t0, 2)}")
                ax.set_ylabel("Depth ($\\mu m$)")
            ax.set_xlabel("Time (ms)")
            for missing_ax in np.arange(i_r + 1, nrows):
                axs_sat[missing_ax, pos_ax_col].axis("off")
        else:
            logging.info("\tNo positive saturation events found")
        if len(neg_evts) > 0:
            logging.info(f"\tFound {len(neg_evts)} negative saturation events!")
            if len(neg_evts) > num_saturation_events_to_plot:
                random_saturation_events = neg_evts[
                    np.random.choice(np.arange(len(neg_evts)), size=num_saturation_events_to_plot, replace=False)
                ]
                random_saturation_events = random_saturation_events[np.argsort(random_saturation_events["sample_index"])]
            else:
                random_saturation_events = neg_evts
            for i_r, r in enumerate(random_saturation_events):
                ax = axs_sat[i_r, neg_ax_col]
                t0 = recording.sample_index_to_time(r["sample_index"])
                w = sw.plot_traces(
                    recording_ps,
                    time_range=[t0 - t_cutout_saturation_s, t0 + t_cutout_saturation_s],
                    clim=(-clim, clim),
                    ax=ax,
                )
                ax.set_xticks([t0 - t_cutout_saturation_s, t0, t0 + t_cutout_saturation_s])
                ax.set_xticklabels([- t_cutout_saturation_s * 1000, 0, t_cutout_saturation_s * 1000])
                if i_r == 0:
                    ax.set_title(f"Negative\n@{np.round(t0, 2)}")
                else:
                    ax.set_title(f"{np.round(t0, 2)}")
                ax.set_ylabel("Depth ($\\mu m$)")

            ax.set_xlabel("Time (ms)")
            for missing_ax in np.arange(i_r + 1, nrows):
                axs_sat[missing_ax, neg_ax_col].axis("off")
        else:
            logging.info("\tNo negative saturation events found")
        fig_sat_samples.suptitle(
            f"Saturation events:\nPositive: {len(pos_evts)} -- Negative: {len(neg_evts)}"
        )
        if axs_sat.shape[1] == 1:
            fig_sat_samples.subplots_adjust(left=0.3, right=0.85, wspace=0.5, hspace=0.3)
        else:
            fig_sat_samples.subplots_adjust(left=0.3, wspace=0.5, hspace=0.3)

        fig_sat_samples_path = recording_fig_folder / "saturation_samples.png"
        fig_sat_samples.savefig(fig_sat_samples_path, dpi=300)
        if relative_to is not None:
            fig_sat_samples_path = fig_sat_samples_path.relative_to(relative_to)

        saturation_samples_metric = QCMetric(
            name=f"Saturation events samples {recording_name}",
            description=f"Evaluation of {recording_name} saturation samples",
            reference=str(fig_sat_samples_path),
            value={"value": "Pass"},
            status_history=[status_pass],
        )

        # saturation events timeline
        fig_sat_time, ax_sat_time = plt.subplots(figsize=(15, 10))
        ax_sat_time.ticklabel_format(useOffset=False, style='plain', axis='x')

        ax_sat_time.set_title(
            f"Saturation events:\nPositive: {len(pos_evts)} -- Negative: {len(neg_evts)}"
        )
        if len(pos_evts) > 0:
            pos_evt_times = recording.get_times()[pos_evts["sample_index"]]
            ax_sat_time.plot(pos_evt_times, np.ones_like(pos_evt_times), ls="", marker="|", markersize=20, color="r", label="positive")
        if len(neg_evts) > 0:
            neg_evt_times = recording.get_times()[neg_evts["sample_index"]]
            ax_sat_time.plot(neg_evt_times, -np.ones_like(neg_evt_times), ls="", marker="|", markersize=20, color="b", label="negative")
        ax_sat_time.legend()
        ax_sat_time.set_xlabel("Time (s)")
        ax_sat_time.set_ylabel("Sign")
        ax_sat_time.set_yticks([-1, +1])
        ax_sat_time.set_yticklabels(["-", "+"])
        ax_sat_time.set_ylim(-2, 2)
        ax_sat_time.spines[["top", "right"]].set_visible(False)
        ax_sat_time.get_xaxis().get_major_formatter().set_scientific(False)

        fig_sat_time_path = recording_fig_folder / "saturation_timeline.png"
        fig_sat_time.savefig(fig_sat_time_path, dpi=300)
        if relative_to is not None:
            fig_sat_time_path = fig_sat_time_path.relative_to(relative_to)

        if len(pos_evts) > 0 or len(neg_evts) > 0:
            value_with_options = {
                "value": "",
                "options": ["Good", "Too many saturation events"],
                "status": ["Pass", "Fail"],
                "type": "dropdown",
            }
            saturation_status = status_pending
        else:
            value_with_options = {"value": "Pass"}
            saturation_status = status_pass

        saturation_timeline_metric = QCMetric(
            name=f"Saturation events timeline {recording_name}",
            description=f"Evaluation of {recording_name} saturation timeline",
            reference=str(fig_sat_time_path),
            value=value_with_options,
            status_history=[saturation_status],
        )

        metrics["Saturation"] = [saturation_timeline_metric, saturation_samples_metric]

    if event_dict is not None:
        logging.info("Generating TRIGGER EVENT metrics")

        # make a sorting object with the events
        unit_dict = {}
        for k in event_dict.keys():
            if any(keyword in k.lower() for keyword in event_keys):
                events = np.array(event_dict[k])
                events_in_range = events[events >= recording.get_times()[0]]
                events_in_range = events_in_range[events_in_range < recording.get_times()[-1]]
                if len(events_in_range) > 0:
                    sample_indices = recording.time_to_sample_index(events_in_range)
                    unit_dict[k] = sample_indices
        if len(unit_dict) > 0:
            logging.info(f"\tFound {len(unit_dict)} trigger event sources for {event_keys} keywords.")
            for event_key, event_values in unit_dict.items():
                logging.info(f"\t{event_key}: {len(event_values)} events")
            sorting_events = si.NumpySorting.from_unit_dict(
                [unit_dict], sampling_frequency=recording.sampling_frequency
            )
            analyzer = si.create_sorting_analyzer(sorting_events, recording, sparse=False)

            extensions = {
                "random_spikes": {"method": "all"},
                "templates": {"ms_before": 10, "ms_after": 10},
            }
            analyzer.compute(extensions, **job_kwargs)
            template_ext = analyzer.get_extension("templates")
            templates = template_ext.get_data()

            # create plots for each event
            fig_events, axs_events = _get_fig_axs(ncols=1, nrows=len(analyzer.unit_ids))

            num_spikes = analyzer.sorting.count_num_spikes_per_unit()

            for unit_index, unit_id in enumerate(analyzer.unit_ids):
                ax = axs_events[unit_index, 0]
                # color by depth
                cmap = plt.get_cmap("viridis", analyzer.get_num_channels())
                depths = analyzer.get_channel_locations()[:, 1]
                norm = mpl.colors.Normalize(vmin=np.min(depths), vmax=np.max(depths))

                colors = cmap(np.arange(analyzer.get_num_channels()))
                times = np.linspace(
                    -template_ext.params["ms_before"], template_ext.params["ms_after"], templates.shape[1]
                )

                for template, color in zip(templates[unit_index].T, colors):
                    ax.plot(times, template, color=color, alpha=0.3)

                ax.set_title(f"{unit_id} (#{num_spikes[unit_id]})")
                ax.set_ylabel("Voltage ($\\mu V$)")
                if unit_index == analyzer.get_num_units() - 1:
                    ax.set_xlabel("Time ($ms$)")
                ax.spines[["top", "right"]].set_visible(False)
                fig_events.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="vertical", label="Depth ($\\mu m$)"
                )
            fig_events.subplots_adjust(hspace=0.3, left=0.3, right=0.85)

            value_with_options = {
                "value": "",
                "options": ["Good", "Trigger artifacts"],
                "status": ["Pass", "Fail"],
                "type": "dropdown",
            }
            events_status = status_pending
        else:
            logging.info(f"\tNo events found for {recording_name}")

            value_with_options = {
                "value": "Pass",
            }
            events_status = status_pass

            fig_events, ax_events = _get_fig_axs(1, 1)
            ax_events[0, 0].axis("off")
            fig_events.suptitle(
                f"No trigger events found."
            )
            events_status = status_pass

        fig_events_path = recording_fig_folder / "trigger_events.png"
        fig_events.savefig(fig_events_path, dpi=300)
        if relative_to is not None:
            fig_events_path = fig_events_path.relative_to(relative_to)

        trigger_event_metric = QCMetric(
            name=f"Trigger events {recording_name}",
            description=f"Evaluation of {recording_name} trigger events",
            reference=str(fig_events_path),
            value=value_with_options,
            status_history=[events_status],
        )
        metrics["Trigger Events"] = [trigger_event_metric]

    return metrics


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
    recording_fig_folder = output_qc_path
    recording_fig_folder.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    status_pending = QCStatus(evaluator="", status=Status.PENDING, timestamp=now)

    logging.info("Generating UNIT YIELD metric")
    if sorting_analyzer is None:
        logging.info(f"\tNo sorting analyzer found for {recording_name}")
        return metrics

    number_of_units = sorting_analyzer.get_num_units()

    decoder_label = sorting_analyzer.sorting.get_property("decoder_label")
    number_of_sua_units = int(np.sum(decoder_label == "sua"))
    number_of_mua_units = int(np.sum(decoder_label == "mua"))
    number_of_noise_units = int(np.sum(decoder_label == "noise"))

    default_qc = sorting_analyzer.sorting.get_property("default_qc")
    number_of_good_units = int(np.sum(default_qc == True))

    quality_metrics = sorting_analyzer.get_extension("quality_metrics").get_data()
    template_metrics = sorting_analyzer.get_extension("template_metrics").get_data()

    fig_yield, axs_yield = plt.subplots(3, 3, figsize=(15, 10))

    # protect against all NaNs
    bins = np.linspace(0, 2, 20)
    ax_isi = axs_yield[0, 0]
    if not np.isnan(quality_metrics["isi_violations_ratio"]).all():
        ax_isi.hist(quality_metrics["isi_violations_ratio"], bins=bins, density=True)
    ax_isi.set_xscale("log")
    ax_isi.set_title(f"ISI Violations Ratio")
    ax_isi.spines[["top", "right"]].set_visible(False)

    ax_amp_cutoff = axs_yield[0, 1]
    if not np.isnan(quality_metrics["amplitude_cutoff"]).all():
        ax_amp_cutoff.hist(quality_metrics["amplitude_cutoff"], bins=20, density=True)
    ax_amp_cutoff.set_title(f"Amplitude Cutoff")
    ax_amp_cutoff.spines[["top", "right"]].set_visible(False)

    ax_presence_ratio = axs_yield[0, 2]
    if not np.isnan(quality_metrics["presence_ratio"]).all():
        ax_presence_ratio.hist(quality_metrics["presence_ratio"], bins=20, density=True)
    ax_presence_ratio.set_title(f"Presence Ratio")
    ax_presence_ratio.spines[["top", "right"]].set_visible(False)

    ax_drift = axs_yield[1, 0]
    if not np.isnan(quality_metrics['drift_ptp']).all():
        ax_drift.hist(quality_metrics['drift_ptp'], bins=20, density=True)
    ax_drift.set_title(f"Drift Peak to Peak")
    ax_drift.spines[["top", "right"]].set_visible(False)

    ax_snr = axs_yield[1, 1]
    if not np.isnan(quality_metrics['snr']).all():
        ax_snr.hist(quality_metrics['snr'], bins=20, density=True)
    ax_snr.set_title(f"SNR")
    ax_snr.spines[["top", "right"]].set_visible(False)

    ax_halfwidth = axs_yield[1, 2]
    if not np.isnan(template_metrics['half_width']).all():
        ax_halfwidth.hist(template_metrics['half_width'], bins=20, density=True)
    ax_halfwidth.set_title(f"Half Width")
    ax_halfwidth.spines[["top", "right"]].set_visible(False)

    channel_indices = np.array(
        list(si.get_template_extremum_channel(sorting_analyzer, mode="peak_to_peak", outputs="index").values())
    )
    channel_depths = sorting_analyzer.get_channel_locations()[channel_indices, 1]
    amplitudes = np.array(list(si.get_template_extremum_amplitude(sorting_analyzer, mode="peak_to_peak").values()))
    amplitudes[amplitudes > max_amplitude_for_visualization] = max_amplitude_for_visualization
    df_amplitudes_depths = pd.DataFrame({"amplitude": amplitudes, "channel_depth": channel_depths})
    mean_amplitude_by_depth = df_amplitudes_depths.groupby("channel_depth").mean()

    colors = {"sua": "green", "mua": "orange", "noise": "red"}
    ax_amplitudes = axs_yield[2, 0]
    if decoder_label is not None:
        for label in np.unique(decoder_label):
            mask = decoder_label == label
            ax_amplitudes.scatter(amplitudes[mask], channel_depths[mask], c=colors[label], label=label, alpha=0.4)
    else:
        ax_amplitudes.scatter(amplitudes, channel_depths, alpha=0.4)

    try:
        smoothed_amplitude = savgol_filter(mean_amplitude_by_depth["amplitude"], 10, 2)
        ax_amplitudes.plot(smoothed_amplitude, mean_amplitude_by_depth.index.tolist(), c="r")
    except Exception:
        logging.info("Smooting amplitudes failed.")
    ax_amplitudes.set_title("Unit Amplitude By Depth")
    ax_amplitudes.set_xlabel("Amplitude ($\\mu V$)")
    ax_amplitudes.set_ylabel("Depth ($\\mu m$)")
    ax_amplitudes.legend()
    ax_amplitudes.spines[["top", "right"]].set_visible(False)

    ax_fr = axs_yield[2, 1]
    firing_rate = np.array(quality_metrics["firing_rate"].tolist())
    firing_rate[firing_rate > max_firing_rate_for_visualization] = max_firing_rate_for_visualization
    df_firing_rate_depths = pd.DataFrame({"firing_rate": firing_rate, "channel_depth": channel_depths})
    mean_firing_rate_by_depth = df_firing_rate_depths.groupby("channel_depth").mean()

    if decoder_label is not None:
        for label in np.unique(decoder_label):
            mask = decoder_label == label
            ax_fr.scatter(firing_rate[mask], channel_depths[mask], c=colors[label], label=label, alpha=0.4)
    else:
        ax_fr.scatter(firing_rate, channel_depths, alpha=0.4)

    try:
        smoothed_firing_rate = savgol_filter(mean_firing_rate_by_depth["firing_rate"], 10, 2)
        ax_fr.plot(smoothed_firing_rate, mean_firing_rate_by_depth.index.tolist(), c="r")
    except Exception:
        logging.info("Smooting firing rates failed.")
    ax_fr.set_title("Unit Firing Rate By Depth")
    ax_fr.set_xlabel("Firing rate (Hz)")
    ax_fr.set_ylabel("Depth ($\\mu m$)")
    ax_fr.legend()
    ax_fr.spines[["top", "right"]].set_visible(False)

    ax_text = axs_yield[2, 2]
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
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax_text.text(0, 1, metric_values_str, transform=ax_text.transAxes, fontsize=14, verticalalignment="top", bbox=props)

    fig_yield.suptitle("Unit Metrics Yield", fontsize=15)
    fig_yield.tight_layout()
    unit_yield_path = recording_fig_folder / "unit_yield.png"
    fig_yield.savefig(unit_yield_path, dpi=300)

    if relative_to is not None:
        unit_yield_path = unit_yield_path.relative_to(relative_to)

    sorting_summary_link = None
    if visualization_output is not None:
        if recording_name in visualization_output:
            sorting_summary_link = visualization_output[recording_name].get("sorting_summary")
            sorting_summary_str = f"[Sortingview link]({sorting_summary_link})"
        else:
            sorting_summary_str = f"No sorting summary link found for {recording_name}."
    else:
        sorting_summary_str = "No visualization output found in results."

    value_with_options = {
        "value": "",
        "options": ["Good", "Low Yield", "High Noise"],
        "status": ["Pass", "Fail", "Fail"],
        "type": "dropdown",
    }
    yield_metric = QCMetric(
        name=f"Unit Metrics Yield - {recording_name}",
        description=f"Evaluation of {recording_name} unit metrics yield. {sorting_summary_str}",
        reference=str(unit_yield_path),
        value=value_with_options,
        status_history=[status_pending],
    )
    metrics["Unit Yield"] = [yield_metric]

    logging.info("Generating SORTING CURATION metric")
    if sorting_summary_link is not None:
        gh_uri_str = sorting_summary_link[sorting_summary_link.find("&s="):sorting_summary_link.find("}&")+1]
        sorting_summary_link_no_gh = sorting_summary_link.replace(gh_uri_str, "")
        sorting_curation_metric = QCMetric(
            name=f"Sorting Curation - {recording_name}",
            description=f"Sorting Curation for {recording_name}",
            reference=sorting_summary_link_no_gh,
            value=CurationMetric(),
            status_history=[QCStatus(evaluator="", status=Status.PASS, timestamp=now)]
        )
        metrics["Sorting Curation"] = [sorting_curation_metric]
    
    logging.info("Generating ISI VISUAL AREA LABEL metric")
    value_with_options = {
        "value": "",
        "options": ISI_VISUAL_AREAS,
        "status": ["Pass" for area in ISI_VISUAL_AREAS],
        "type": "dropdown",
    }
    isi_visual_area_metric = QCMetric(
        name=f"Manual annotation of ISI Visual Area Label - {recording_name}",
        description=f"Manual annotation of visual area label based on ISI imaging for {recording_name}",
        value=value_with_options,
        status_history=[QCStatus(evaluator="", status=Status.PASS, timestamp=now)]
    )
    metrics["ISI Visual Area Label"] = [isi_visual_area_metric]

    logging.info("Generating FIRING RATE metric")
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
        axs_fr[segment_index, 0].ticklabel_format(useOffset=False, style='plain', axis='x')

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


### EVENTS UTILS

# saturation values for different NP probes
saturation_thresholds_uv = {
    "PRB_1_4_0480_1": 0.6 / 500 * 1e6,
    "PRB_1_4_0480_1_C": 0.6 / 500 * 1e6,
    "PRB_1_2_0480_2": 0.6 / 500 * 1e6,
    "NP1010": 0.6 / 500 * 1e6,
    # NHP probes
    "NP1015": 0.6 / 500 * 1e6,
    "NP1016": 0.6 / 500 * 1e6,
    "NP1022": 0.6 / 500 * 1e6,
    "NP1030": 0.6 / 500 * 1e6,
    "NP1031": 0.6 / 500 * 1e6,
    "NP1032": 0.6 / 500 * 1e6,
    # NP2.0
    "NP2000": 0.5 / 80 * 1e6,
    "NP2010": 0.5 / 80 * 1e6,
    "NP2013": 0.62 / 100 * 1e6,
    "NP2014": 0.62 / 100 * 1e6,
    "NP2003": 0.62 / 100 * 1e6,
    "NP2004": 0.62 / 100 * 1e6,
    "PRB2_1_2_0640_0": 0.5 / 80 * 1e6,
    "PRB2_4_2_0640_0": 0.5 / 80 * 1e6,
    # Other probes
    "NP1100": 0.6 / 500 * 1e6,  # Ultra probe - 1 bank
    "NP1110": 0.6 / 500 * 1e6,  # Ultra probe - 16 banks
    "NP1121": 0.6 / 500 * 1e6,  # Ultra probe - beta configuration
    "NP1300": 0.6 / 500 * 1e6,  # Opto probe
}


def find_saturation_events(
    recording: si.BaseRecording,
    saturation_threshold_uv: float,
    exclude_sweep_ms: float = 1,
    radius_um: float = 200,
    **job_kwargs,
):
    """
    Find saturation events in a recording.

    Parameters
    ----------
    recording : BaseRecording
        The recording object.
    saturation_threshold_uv : float
        The saturation threshold in microvolts.
    exclude_sweep_ms : float, default: 1
        The time in milliseconds to exclude around the saturation event.
    radius_um : float, default: 200
        The radius in micrometers to consider for saturation events.
    job_kwargs : dict
        The job kwargs.

    Returns
    -------
    positive_saturation_events : np.ndarray
        The positive saturation events.
    negative_saturation_events : np.ndarray
        The negative saturation events.
    """
    from spikeinterface.core.node_pipeline import run_node_pipeline
    from spikeinterface.sortingcomponents.peak_detection import DetectPeakLocallyExclusive

    channel_distance = si.get_channel_distances(recording)
    neighbours_mask = channel_distance <= radius_um
    num_channels = recording.get_num_channels()

    # here we set absolute thresholds externally, since we know the saturation thresholds
    saturation_both = DetectPeakLocallyExclusive(recording, noise_levels=np.ones(num_channels))
    abs_thresholds = np.array([saturation_threshold_uv / recording.get_channel_gains()[0]] * num_channels)
    saturation_both.args = ("both", abs_thresholds, exclude_sweep_ms, neighbours_mask)

    job_name = f"finding saturation events"
    squeeze_output = True
    nodes = [saturation_both]
    outs = run_node_pipeline(
        recording,
        nodes,
        job_kwargs,
        job_name=job_name,
        squeeze_output=squeeze_output,
    )

    # only keep unique saturation events
    outs = outs[np.unique(outs["sample_index"], return_index=True)[1]]

    positive_saturation_events = outs[outs["amplitude"] > 0]
    negative_saturation_events = outs[outs["amplitude"] < 0]

    return positive_saturation_events, negative_saturation_events

if __name__ == '__main__':
    motion_path = Path('/root/capsule/data/ecephys_764790_2024-12-19_16-11-34_sorted_2025-02-21_14-46-33/preprocessed/motion/experiment2_Record Node 103#Neuropix-PXI-100.ProbeA_recording1')
    recording_name = 'experiment2_Record Node 103#Neuropix-PXI-100.ProbeA_recording1'
    recording = si.load('/root/capsule/data/ecephys_764790_2024-12-19_16-11-34/ecephys/ecephys_compressed/experiment2_Record Node 103#Neuropix-PXI-100.ProbeA.zarr')
    quality_control_fig_folder = Path(f"/results/quality_control_{recording_name}")
    relative_to = Path("/results")
    generate_drift_qc(recording, recording_name, motion_path, quality_control_fig_folder, relative_to=relative_to)
