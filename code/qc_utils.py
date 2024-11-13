from __future__ import annotations

from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import welch


import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw


from aind_data_schema_models.modalities import Modality
from aind_data_schema.core.processing import Processing
from aind_data_schema.core.quality_control import QCMetric, QCStatus, QCEvaluation, Stage, Status
from aind_qcportal_schema.metric_value import CheckboxMetric


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
    num_snippets_per_segment: int=3,
    duration_s: float=0.1,
):
    """
    Plot snippets of raw data as an image

    Parameters
    ----------
    recording : BaseRecording
        The recording object.
    num_snippets_per_segment: ind, default: 3
        Number of snippets to plot for each segment.
    duration_s : float, default: 0.1
        The duration of each snippet.


    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    num_segments = recording.get_num_segments()
    fig, axs = _get_fig_axs(num_segments * 2, num_snippets_per_segment)

    recording_hp = spre.highpass_filter(recording)
    for segment_index in range(num_segments):
        # evenly distribute t_starts across segments
        times = recording.get_times(segment_index=segment_index)
        t_starts = np.round(np.linspace(times[0], times[-1], num_snippets_per_segment + 2)[1:-1], 1)
        for snippet_index, t_start in enumerate(t_starts):
            ax_raw = axs[segment_index * 2, snippet_index]
            ax_hp = axs[segment_index * 2 + 1, snippet_index]
            sw.plot_traces(
                recording,
                time_range=[t_start, t_start + duration_s],
                mode="map",
                return_scaled=True,
                with_colorbar=True,
                ax=ax_raw
            )
            sw.plot_traces(
                recording_hp,
                time_range=[t_start, t_start + duration_s],
                mode="map",
                return_scaled=True,
                with_colorbar=True,
                ax=ax_hp
            )
            if segment_index == 0:
                ax_raw.set_title(f"seg{segment_index} @ {t_start}s\nRaw")
                ax_hp.set_title(f"Highpass")
            if snippet_index == 0:
                ax_raw.set_ylabel("Depth ($\mu m$)")
                ax_hp.set_ylabel("Depth ($\mu m$)")
            if segment_index == num_segments - 1:
                ax_hp.set_xlabel(f"Time (tot. {duration_s} s)")
            ax_raw.set_xticklabels([])
            ax_hp.set_xticklabels([])
            

    fig.subplots_adjust(wspace=0.3, hspace=0.2, top=0.9)
    return fig


def plot_psd(
    recording: si.BaseRecording,
    recording_lfp: si.BaseRecording | None = None,
    num_snippets_per_segment: int=3,
    num_depths_for_spectrograms: int=3,
    duration_s: float=1,
    freq_lf_filt: float=500.,
    freq_lf_viz: float=100.,
    freq_hf_filt: float=3000.,
    freq_hf_viz: float=5000.
):
    """
    Plot spectra and spectrograms.

    Parameters
    ----------
    recording : BaseRecording
        The recording object.
    num_snippets_per_segment: ind, default: 3
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
    recording_lfp = spre.resample(recording_lfp, 1.5 * freq_lf_filt)
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
                start_frame=start_frame_hf,
                end_frame=end_frame_hf,
                segment_index=segment_index,
                return_scaled=True
            )

            start_frame_lf = recording_lfp.time_to_sample_index(t_start, segment_index=segment_index)
            end_frame_lf = recording_lfp.time_to_sample_index(t_start + duration_s, segment_index=segment_index)
            traces_lf = recording_lfp.get_traces(
                start_frame=start_frame_lf,
                end_frame=end_frame_lf,
                segment_index=segment_index,
                return_scaled=True
            )
            power_channels_wide = []
            power_channels_lf = []
            
            for i in range(traces_hf.shape[1]):
                f_wide, p_wide = welch(traces_hf[:, i], fs=recording.sampling_frequency)
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

            extent_wide = [f.min(), f.max(), np.min(depths), np.max(depths)]
            ax_psd_channels.imshow(
                power_channels_wide,
                extent=extent_wide,
                aspect="auto",
                cmap="inferno",
                origin="lower",
            )
            extent_hf = [f_hf.min(), f_hf.max(), np.min(depths), np.max(depths)]
            ax_psd_hf_channels.imshow(
                power_channels_wide[:, hf_mask],
                extent=extent_hf,
                aspect="auto",
                cmap="inferno",
                origin="lower",
            )
            extent_lf = [f_lf.min(), f_lf.max(), np.min(depths), np.max(depths)]
            ax_psd_lf_channels.imshow(
                power_channels_lf[:, lf_mask],
                extent=extent_lf,
                aspect="auto",
                cmap="inferno",
                origin="lower",
            )
            if segment_index == 0:
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
                ax_psd.set_xlabel("Frequency (Hz)")
                ax_psd_hf.set_xlabel("Frequency (Hz)")
                ax_psd_lf.set_xlabel("Frequency (Hz)")

            ax_psd.set_yscale("log")
            ax_psd_hf.set_yscale("log")
            ax_psd_lf.set_yscale("log")

    fig_psd.subplots_adjust(wspace=0.3, hspace=0.3)
    fig_psd_hf.subplots_adjust(wspace=0.3, hspace=0.3)
    fig_psd_lf.subplots_adjust(wspace=0.3, hspace=0.3)

    return fig_psd, fig_psd_hf, fig_psd_lf


def plot_rms_by_depth(recording, recording_preprocessed=None, recording_lfp=None):
    """
    """
    num_segments = recording.get_num_segments()
    fig_rms, ax_rms = plt.subplots(figsize=(10, 7))

    if recording_lfp is None:
        # this means the recording is wide-band, so we apply an additional hp filter
        recording = spre.highpass_filter(recording)
    
    recording = spre.average_across_direction(recording, direction="y")

    data_raw = si.get_random_data_chunks(
        recording,
        return_scaled=True
    )
    depths_raw = recording.get_channel_locations()[:, 1]
    rms_raw = np.sqrt(np.sum(data_raw ** 2, axis=0) / data_raw.shape[0])

    ax_rms.plot(rms_raw, depths_raw, color="gray", label="raw")
    
    if recording_preprocessed is not None:
        recording_preprocessed = spre.average_across_direction(recording_preprocessed, direction="y")
        data_pre = si.get_random_data_chunks(
            recording_preprocessed,
            return_scaled=True
        )

        depths_pre = recording_preprocessed.get_channel_locations()[:, 1]    
        rms_pre = np.sqrt(np.sum(data_pre ** 2, axis=0) / data_pre.shape[0])
        ax_rms.plot(rms_pre, depths_pre, color="r", label="preprocessed")
        ax_rms.legend()

    ax_rms.set_xlabel("RMS ( $\mu V$)")
    ax_rms.set_ylabel("Depth ( $\mu m$)")
    ax_rms.spines[['right', 'top']].set_visible(False)

    return fig_rms
        

def probe_noise_levels_qc(
    recording: si.BaseRecording,
    recording_name: str,
    output_qc_path: Path,
    relative_to: Path | None = None,
    recording_lfp: si.BaseRecording | None = None,
    recording_preprocessed: si.BaseRecording | None = None,
    processing: Processing | None = None,
    visualization_output: dict | None = None,
) -> dict[str: list[QCMetric]]:
    metrics = {}
    recording_fig_folder = output_qc_path / recording_name
    recording_fig_folder.mkdir(exist_ok=True, parents=True)
    sp = QCStatus(evaluator="", status=Status.PENDING, timestamp=t)

    raw_traces_value = {}
    if visualization_output is not None:
        visualization_rec = visualization_output.get(recording_name)
        if visualization_rec:
            raw_traces_value = {"value": visualization_rec.get("timeseries")}

    print("Plotting raw data")
    fig_raw = plot_raw_data(recording)
    raw_traces_path = recording_fig_folder / "traces_raw.png"
    fig_raw.savefig(raw_traces_path, dpi=300)
    if relative_to is not None:
        raw_traces_path = raw_traces_path.relative_to(relative_to)
    raw_data_metric = QCMetric(
        name=f'{recording_name} raw data',
        value={},
        description=f'Evaluation of {recording_name} raw data',
        reference=str(raw_traces_path),
        value=raw_traces_value,
        status_history=sp
    )
    metrics["Raw Data"] = [raw_data_metric]

    print("Plotting PSD")
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
        name=f'{recording_name} PSD - Wide Band',
        description=f'Evaluation of {recording_name} wide-band power spectrum density',
        reference=str(psd_wide_path),
        value=None,
        status_history=sp
    )

    hf_value_with_flags = {
        "value": "",
        "options": ["No contamination", "High frequency contamination"],
        "status": ["Pass", "Fail"],
        "type": "checkbox",
    }

    psd_hf_metric = QCMetric(
        name=f'{recording_name} PSD - High Frequency',
        description=f'Evaluation of {recording_name} high-frequency power spectrum density',
        reference=str(psd_hf_path),
        value=hf_value_with_flags,
        status_history=sp

    )
    lf_value_with_flags = {
        "value": "",
        "options": ["No contamination", "Line (60 Hz) contamination"],
        "status": ["Pass", "Fail"],
        "type": "checkbox",
    }

    psd_lf_metric = QCMetric(
        name=f'{recording_name} PSD - Low Frequency',
        description=f'Evaluation of {recording_name} low-frequency power spectrum density',
        reference=str(psd_hf_path),
        value=lf_value_with_flags,
        status_history=sp
    )
    metrics["PSD"] = [psd_wide_metric, psd_hf_metric, psd_lf_metric]

    # TODO: Spectrograms

    print("Plotting RMS")
    fig_rms = plot_rms_by_depth(recording, recording_preprocessed)
    rms_path = recording_fig_folder / "rms.png"
    fig_rms.savefig(rms_path, dpi=300)
    if relative_to is not None:
        rms_path = rms_path.relative_to(relative_to)
    rms_metric = QCMetric(
        name=f'{recording_name} RMS',
        description=f'Evaluation of {recording_name} RMS',
        reference=str(rms_path),
        value=None,
        status_history=sp
    )
    metrics["Noise"] = [rms_metric]

    # Bad channel detection out of brain, noisy, silent
    if processing is not None:
        try:
            data_processes = processing.processing_pipeline.data_processes
            for data_process in data_processes:
                if data_process.name == "Ephys preprocessing" and \
                   data_process.parameters.get("recording_name") == recording_name:
                    channel_labels = data_process.output.get("channel_labels")
                    channel_labels_value = {
                        "good": np.sum(channel_labels == "good"),
                        "noise": np.sum(channel_labels == "noise"),
                        "dead": np.sum(channel_labels == "dead"),
                        "out": np.sum(channel_labels == "out"),
                    }
                    channel_labels_metric = QCMetric(
                        name=f'{recording_name} RMS',
                        description=f'Evaluation of {recording_name} bad_channel detection',
                        value=channel_labels_value,
                        status_history=sp
                    )
                    metrics["Noise"].append(channel_labels_metric)
        except:
            print(f"Failed to load bad channel labels for {recording_name}")

    return metrics
