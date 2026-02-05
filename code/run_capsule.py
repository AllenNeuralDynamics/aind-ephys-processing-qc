""" Quality control for ecephys pipeline """

import os
import sys
import argparse
import shutil
import json
import numpy as np
import time
import logging
from pathlib import Path


import spikeinterface as si

# AIND
from aind_data_schema_models.modalities import Modality
from aind_data_schema.core.quality_control import QualityControl, Stage

try:
    from aind_log_utils import log
    HAVE_AIND_LOG_UTILS = True
except ImportError:
    HAVE_AIND_LOG_UTILS = False

from qc_utils import (
    load_preprocessed_recording,
    load_processing_metadata,
    recording_abbrv_name,
    generate_raw_qc,
    generate_units_qc,
    generate_drift_qc,
    generate_event_qc,
)

data_folder = Path("../data")
results_folder = Path("../results")

# Define argument parser
parser = argparse.ArgumentParser(description="Compute Quality Control for Ephys pipeline")

skip_event_group = parser.add_mutually_exclusive_group()
skip_event_group_help = "Whether to compute event metrics (saturation+trigger). Default: True"
skip_event_group.add_argument("--no-event-metrics", action="store_true", help=skip_event_group_help)
skip_event_group.add_argument("static_compute_event", nargs="?", default="true", help=skip_event_group_help)


min_duration_allow_failed_group = parser.add_mutually_exclusive_group()
min_duration_allow_failed_help = (
    "Minimum recording duration below which metrics will be allowed to fail. Default: 300"
)
min_duration_allow_failed_group.add_argument("static_min_duration_allow_failed", nargs="?", default=None, help=min_duration_allow_failed_help)
min_duration_allow_failed_group.add_argument("--min-duration-allow-failed", default=None, help=min_duration_allow_failed_help)


if __name__ == "__main__":
    t_qc_start_all = time.perf_counter()

    args = parser.parse_args()
    COMPUTE_EVENT_METRIC = (
        args.static_compute_event.lower() == "true" if args.static_compute_event
        else not args.no_event_metrics
    )
    MIN_DURATION_ALLOW_FAILED = args.static_min_duration_allow_failed or args.min_duration_allow_failed
    if MIN_DURATION_ALLOW_FAILED is None:
        MIN_DURATION_ALLOW_FAILED = 0
    MIN_DURATION_ALLOW_FAILED = float(MIN_DURATION_ALLOW_FAILED)

    # pipeline mode VS capsule mode
    ecephys_folders = [
        p
        for p in data_folder.iterdir()
        if p.is_dir() and ("ecephys" in p.name or "behavior" in p.name) and "sorted" not in p.name
    ]

    # capsule mode
    ecephys_folder = None
    if len(ecephys_folders) == 1:
        ecephys_folder = ecephys_folders[0]
        if HAVE_AIND_LOG_UTILS:
            # look for subject.json and data_description.json files
            subject_json = ecephys_folder / "subject.json"
            subject_id = "undefined"
            if subject_json.is_file():
                subject_data = json.load(open(subject_json, "r"))
                subject_id = subject_data["subject_id"]

            data_description_json = ecephys_folder / "data_description.json"
            session_name = "undefined"
            if data_description_json.is_file():
                data_description = json.load(open(data_description_json, "r"))
                session_name = data_description["name"]

            log.setup_logging(
                "Quality Control Ecephys",
                subject_id=subject_id,
                asset_name=session_name,
            )
        else:
            logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    logging.info(f"Running Ephys QC with the following parameters:")
    logging.info(f"\tCOMPUTE EVENT METRICS: {COMPUTE_EVENT_METRIC}")
    logging.info(f"\tMIN DURATION ALLOW FAILED: {MIN_DURATION_ALLOW_FAILED}")

    # Use CO_CPUS/SLURM_CPUS_ON_NODE env variable if available
    N_JOBS_EXT = os.getenv("CO_CPUS") or os.getenv("SLURM_CPUS_ON_NODE") or os.getenv("SLURM_CPUS_PER_TASK")
    N_JOBS = int(N_JOBS_EXT) if N_JOBS_EXT is not None else -1
    job_kwargs = dict(n_jobs=N_JOBS, progress_bar=False, mp_context="spawn")
    si.set_global_job_kwargs(**job_kwargs)

    ecephys_sorted_folders = [
        p
        for p in data_folder.iterdir()
        if p.is_dir() and ("ecephys" in p.name or "behavior" in p.name) and "sorted" in p.name
    ]
    if len(ecephys_sorted_folders) == 1:
        ecephys_sorted_folder = ecephys_sorted_folders[0]
    elif (data_folder / "preprocessed").is_dir():
        ecephys_sorted_folder = data_folder
    else:
        logging.info(
            "Sorted folder not found and required for Processed Evaluations. "
            "Only Raw evaluations will be computed"
        )
        ecephys_sorted_folder = None

    job_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
    job_dicts = []
    for job_json_file in job_json_files:
        with open(job_json_file) as f:
            job_dict = json.load(f)
        job_dicts.append(job_dict)
    logging.info(f"Found {len(job_dicts)} JSON job files")

    if len(job_dicts) == 0:
        logging.info("Parsing AIND-specific input data")
        # here we load the compressed recordings
        if (ecephys_folder / "ecephys").is_dir():
            ecephys_compressed_folder = ecephys_folder / "ecephys" / "ecephys_compressed"
        else:
            ecephys_compressed_folder = ecephys_folder / "ecephys_compressed"
        for stream_folder in ecephys_compressed_folder.iterdir():
            stream_name = stream_folder.name
            if "LFP" in stream_name or "NI-DAQ" in stream_name:
                continue
            job_dict = {}
            job_dict["session_name"] = ecephys_folder.name
            recording_base_name = stream_name[: stream_name.find(".zarr")]
            recording = si.read_zarr(stream_folder)
            recording_lfp = None

            if "AP" in stream_name:
                lfp_stream_path = Path(str(stream_folder).replace("AP", "LFP"))
                if lfp_stream_path.is_dir():
                    recording_lfp = si.read_zarr(lfp_stream_path)
            for segment_index in range(recording.get_num_segments()):
                recording_one = si.split_recording(recording)[segment_index]
                if recording_lfp is not None:
                    recording_lfp_one = si.split_recording(recording_lfp)[segment_index]
                recording_name = f"{recording_base_name}_recording{segment_index+1}"
                # timestamps should be monotonically increasing, but we allow for small glitches
                skip_times = False
                for segment_index in range(recording.get_num_segments()):
                    times = recording.get_times(segment_index=segment_index)
                    times_diff = np.diff(times)
                    num_negative_times = np.sum(times_diff < 0)

                    if num_negative_times > 0:
                        logging.info(f"\t{recording_name} - Times not monotonically increasing.")
                        skip_times = True
                job_dict["skip_times"] = skip_times

                if len(np.unique(recording_one.get_channel_groups())) > 1:
                    for group, recording_group in recording_one.split_by("group").items():
                        job_dict["recording_name"] = f"{recording_base_name}_recording{segment_index+1}_group{group}"
                        job_dict["recording_dict"] = recording_group.to_dict(recursive=True, relative_to=data_folder)
                        if recording_lfp is not None:
                            recording_lfp_group = recording_lfp_one.split_by("group")[group]
                            job_dict["recording_lfp_dict"] = recording_lfp_group.to_dict(
                                recursive=True, relative_to=data_folder
                            )
                        job_dicts.append(job_dict)
                else:
                    job_dict["recording_name"] = f"{recording_base_name}_recording{segment_index+1}"
                    job_dict["recording_dict"] = recording_one.to_dict(recursive=True, relative_to=data_folder)
                    if recording_lfp is not None:
                        job_dict["recording_lfp_dict"] = recording_lfp_one.to_dict(
                            recursive=True, relative_to=data_folder
                        )
                    job_dicts.append(job_dict)
        logging.info(f"Found {len(job_dicts)} recordings")

    processing = None
    visualization_output = None

    if ecephys_sorted_folder is not None:
        processing_json_file = ecephys_sorted_folder / "processing.json"
        if processing_json_file.is_file():
            try:
                processing = load_processing_metadata(processing_json_file)
            except:
                logging.info(f"Failed to load processing.json")

        visualization_json_file = ecephys_sorted_folder / "visualization_output.json"
        if visualization_json_file.is_file():
            with open(visualization_json_file) as f:
                visualization_output = json.load(f)

    event_dict = None

    if ecephys_folder is not None:
        harp_folder = [p for p in (ecephys_folder / "behavior").glob("**/raw.harp")]
        if len(harp_folder) == 1:
            harp_folder = harp_folder[0]
            logging.info("Harp folder found")
            event_json_files = [p for p in harp_folder.parent.iterdir() if p.suffix == ".json"]
            event_json_file = None
            if len(event_json_files) == 1:
                event_json_file = event_json_files[0]
            elif len(event_json_files) > 1:
                logging.info(f"Found {len(event_json_files)} JSON files in behavior folder. Determining behavior file by name")
                # the JSON file should start with {subject_id}_{date}
                if session_name != "undefined":
                    subject_date_str = "_".join(session_name.split("_")[1:-1])
                    for json_file in event_json_files:
                        if json_file.name.startswith(subject_date_str):
                            event_json_file = json_file
                            break
            if event_json_file is not None:
                with open(event_json_file) as f:
                    event_dict = json.load(f)

    if event_dict is None:
        logging.info("Events from HARP not found. Trigger event metrics will not be generated.")

    # look for JSON files or loop through preprocessed
    recording_names = [jd["recording_name"] for jd in job_dicts]
    for job_dict in job_dicts:
        all_metrics = []
        recording_name = job_dict["recording_name"]
        recording = si.load(job_dict["recording_dict"], base_folder=data_folder)
        skip_times = job_dict.get("skip_times", False)
        if skip_times:
            logging.info(f"Resetting times for {recording_name}")
            recording.reset_times()
        recording_lfp_dict = job_dict.get("recording_lfp_dict")
        if recording_lfp_dict is not None:
            recording_lfp = si.load(recording_lfp_dict, base_folder=data_folder)
            if skip_times:
                recording_lfp.reset_times()
        else:
            recording_lfp = None
        session_name = job_dict["session_name"]
        logging.info(f"Recording {recording_name}")
        recording_preprocessed = None
        if ecephys_sorted_folder is not None:
            sorting_analyzer = None
            preprocessed_json_file = ecephys_sorted_folder / "preprocessed" / f"{recording_name}.json"
            recording_preprocessed = load_preprocessed_recording(
                preprocessed_json_file, session_name, ecephys_folder, data_folder
            )
            if recording_preprocessed is not None and skip_times:
                recording_preprocessed.reset_times()

            postprocessed_folder_zarr = ecephys_sorted_folder / "postprocessed" / f"{recording_name}.zarr"
            postprocessed_folder = ecephys_sorted_folder / "postprocessed" / recording_name
            if postprocessed_folder_zarr.is_dir():
                sorting_analyzer = si.load(postprocessed_folder_zarr, load_extensions=False)
            elif postprocessed_folder.is_dir():
                # this is for legacy waveform extractor folders
                sorting_analyzer = si.load_waveforms(postprocessed_folder, output="SortingAnalyzer")

            if recording_preprocessed is not None and sorting_analyzer is not None:
                sorting_analyzer.set_temporary_recording(recording_preprocessed)

        quality_control_fig_folder = results_folder / f"quality_control_{recording_name}"
        
        metrics_raw, raw_names = generate_raw_qc(
            recording,
            recording_name,
            quality_control_fig_folder,
            relative_to=results_folder,
            recording_lfp=recording_lfp,
            recording_preprocessed=recording_preprocessed,
            processing=processing,
            visualization_output=visualization_output,
        )
        all_metrics.extend(metrics_raw)

        if COMPUTE_EVENT_METRIC:
            metrics_event, event_names = generate_event_qc(
                recording,
                recording_name,
                quality_control_fig_folder,
                relative_to=results_folder,
                event_dict=event_dict,
                event_keys=["licktime", "optogeneticstime"],
            )
            all_metrics.extend(metrics_event)
        else:
            logging.info("Skipping computation of event metrics.")
        
        if ecephys_sorted_folder is not None:
            motion_path = ecephys_sorted_folder / "preprocessed" / "motion" / recording_name

            metrics_drift, drift_names = generate_drift_qc(
                recording,
                recording_name,
                motion_path,
                quality_control_fig_folder,
                relative_to=results_folder,
            )
            all_metrics.extend(metrics_drift)
        
        if ecephys_sorted_folder is not None:
            metrics_units, units_names = generate_units_qc(
                sorting_analyzer,
                recording_name,
                quality_control_fig_folder,
                relative_to=results_folder,
                visualization_output=visualization_output,
                raw_recording=recording,
            )
            all_metrics.extend(metrics_units)

        # If recording is too short, allow tagged metrics to fail
        if recording.get_total_duration() < MIN_DURATION_ALLOW_FAILED:
            logging.info(
                f"Recording {recording_name} duration below {MIN_DURATION_ALLOW_FAILED}. "
                f"Adding it to allow_tag_failures."
            )
            allow_tag_failures = [recording_abbrv_name(recording_name)]
        else:
            allow_tag_failures = []

        # make quality control with metric types as groups
        # probe/streams are added at aggregation
        quality_control = QualityControl(
            metrics=all_metrics,
            default_grouping=("stage", "probe"),
            allow_tag_failures=allow_tag_failures
        )
        quality_control.write_standard_file(output_directory=results_folder, suffix=f"_{recording_name}.json")

    t_qc_end_all = time.perf_counter()
    elapsed_time_qc_all = np.round(t_qc_end_all - t_qc_start_all, 2)

    logging.info(f"EPHYS QC time: {elapsed_time_qc_all}s")
