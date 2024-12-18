""" Quality control for ecephys pipeline """

from pathlib import Path
import shutil
import json
import numpy as np
import argparse
import time
import logging

import spikeinterface as si

from aind_data_schema.core.processing import Processing
from aind_data_schema_models.modalities import Modality
from aind_data_schema.core.quality_control import QualityControl, QCEvaluation, Stage

from qc_utils import (
    generate_raw_qc,
    generate_units_qc,
    load_preprocessed_recording,
    load_processing_metadata,
    generate_drift_qc,
)

data_folder = Path("../data")
results_folder = Path("../results")


# define argument parser
parser = argparse.ArgumentParser(description="Generate Ephys QC data")

copy_sorted_to_results_group = parser.add_mutually_exclusive_group()
copy_sorted_to_results_help = "Whether to copy the sorted data to the results folder"
copy_sorted_to_results_group.add_argument(
    "--copy-sorted-to-results", action="store_true", help=copy_sorted_to_results_help
)
copy_sorted_to_results_group.add_argument(
    "static_copy_sorted_to_results", nargs="?", default="false", help=copy_sorted_to_results_help
)


if __name__ == "__main__":
    logging.info("\nEPHYS QC")
    t_qc_start_all = time.perf_counter()
    args = parser.parse_args()

    COPY_SORTED_TO_RESULTS = True if args.copy_sorted_to_results else args.static_copy_sorted_to_results == "true"

    # pipeline mode VS capsule mode
    ecephys_folders = [
        p
        for p in data_folder.iterdir()
        if p.is_dir() and ("ecephys" in p.name or "behavior" in p.name) and "sorted" not in p.name
    ]

    # capsule mode
    assert len(ecephys_folders) == 1, "Attach one raw asset at a time"
    ecephys_folder = ecephys_folders[0]

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

    processing_json_file = ecephys_sorted_folder / "processing.json"
    processing = None
    if processing_json_file.is_file():
        processing = load_processing_metadata(processing_json_file)

    visualization_json_file = ecephys_sorted_folder / "visualization_output.json"
    visualization_output = None
    if visualization_json_file.is_file():
        with open(visualization_json_file) as f:
            visualization_output = json.load(f)

    # look for JSON files or loop through preprocessed
    all_metrics_raw = {}
    all_metrics_processed = {}
    for job_dict in job_dicts:
        recording = si.load_extractor(job_dict["recording_dict"], base_folder=data_folder)
        if job_dict["skip_times"]:
            recording.reset_times()
        recording_lfp_dict = job_dict.get("recording_lfp_dict")
        if recording_lfp_dict is not None:
            recording_lfp = si.load_extractor(recording_lfp_dict, base_folder=data_folder)
            if job_dict["skip_times"]:
                recording_lfp.reset_times()
        else:
            recording_lfp = None
        recording_name = job_dict["recording_name"]
        session_name = job_dict["session_name"]
        logging.info(f"Recording {recording_name}")
        recording_preprocessed = None
        if ecephys_sorted_folder is not None:
            preprocessed_json_file = ecephys_sorted_folder / "preprocessed" / f"{recording_name}.json"
            recording_preprocessed = load_preprocessed_recording(
                preprocessed_json_file, session_name, ecephys_folder, data_folder
            )
            if recording_preprocessed is not None and job_dict["skip_times"]:
                recording_preprocessed.reset_times()

            postprocessed_folder_zarr = ecephys_sorted_folder / "postprocessed" / f"{recording_name}.zarr"
            postprocessed_folder = ecephys_sorted_folder / "postprocessed" / recording_name
            if postprocessed_folder_zarr.is_dir():
                sorting_analyzer = si.load_sorting_analyzer(postprocessed_folder_zarr)
            elif postprocessed_folder.is_dir():
                # this is for legacy waveform extractor folders
                sorting_analyzer = si.load_waveforms(postprocessed_folder, output="SortingAnalyzer")
            else:
                sorting_analyzer = None

            if recording_preprocessed is not None and sorting_analyzer is not None:
                sorting_analyzer.set_temporary_recording(recording_preprocessed)

        quality_control_fig_folder = results_folder / "quality_control"

        metrics_raw = generate_raw_qc(
            recording,
            recording_name,
            quality_control_fig_folder,
            relative_to=results_folder,
            recording_lfp=recording_lfp,
            recording_preprocessed=recording_preprocessed,
            processing=processing,
            visualization_output=visualization_output,
        )

        motion_path = ecephys_sorted_folder / "preprocessed" / "motion" / recording_name

        metrics_drift = generate_drift_qc(
            recording, recording_name, motion_path, quality_control_fig_folder, relative_to=results_folder
        )

        metrics_raw.update(metrics_drift)
        for evaluation_name, metric_list in metrics_raw.items():
            if evaluation_name in all_metrics_raw:
                all_metrics_raw[evaluation_name].extend(metric_list)
            else:
                all_metrics_raw[evaluation_name] = metric_list

        metrics_processed = generate_units_qc(
            sorting_analyzer,
            recording_name,
            quality_control_fig_folder,
            relative_to=results_folder,
            visualization_output=visualization_output,
        )

        for evaluation_name, metric_list in metrics_processed.items():
            if evaluation_name in all_metrics_processed:
                all_metrics_processed[evaluation_name].extend(metric_list)
            else:
                all_metrics_processed[evaluation_name] = metric_list

    # generate evaluations
    evaluations = []

    for evaluation_name, metrics in all_metrics_raw.items():
        evaluation = QCEvaluation(
            modality=Modality.ECEPHYS,
            stage=Stage.RAW,
            name=evaluation_name,
            description=evaluation_name,
            metrics=metrics,
        )
        evaluations.append(evaluation)

    for evaluation_name, metrics in all_metrics_processed.items():
        evaluation = QCEvaluation(
            modality=Modality.ECEPHYS,
            stage=Stage.PROCESSING,
            name=evaluation_name,
            description=evaluation_name,
            metrics=metrics,
        )
        evaluations.append(evaluation)

    # make quality control
    quality_control = QualityControl(evaluations=evaluations)

    with (results_folder / "quality_control.json").open("w") as f:
        f.write(quality_control.model_dump_json(indent=3))

    if COPY_SORTED_TO_RESULTS:
        shutil.copytree(ecephys_sorted_folder, results_folder)
        logging.info(f"Sorted data copied to results folder")

    t_qc_end_all = time.perf_counter()
    elapsed_time_preprocessing_all = np.round(t_qc_end_all - t_qc_start_all, 2)

    logging.info(f"EPHYS QC time: {elapsed_time_preprocessing_all}s")
