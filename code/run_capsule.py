""" Quality control for ecephys pipeline """

from pathlib import Path
import json
import numpy as np

import spikeinterface as si

from aind_data_schema.core.processing import Processing
from aind_data_schema_models.modalities import Modality
from aind_data_schema.core.quality_control import QualityControl, QCEvaluation, Stage

from qc_utils import probe_raw_qc, get_probe_firing_rate_qc, get_probe_unit_yield_qc

data_folder = Path("../data")
results_folder = Path("../results")

def get_unit_yield_evaluation(probe_analyzers: dict[str, si.SortingAnalyzer], output_path: Path) -> QCEvaluation:
    probe_unit_yield_qc_metrics = []
    for probe_name in probe_analyzers:
        probe_unit_yield_qc_metrics.append(get_probe_unit_yield_qc(probe_analyzers[probe_name], probe_name, output_path / probe_name))

    return QCEvaluation(name='Unit Yield', description='Quality of unit yield', stage=Stage.PROCESSING, modality=Modality.from_abbreviation('ecephys'), notes='',
                        allow_failed_metrics=True, metrics=probe_unit_yield_qc_metrics)

def get_firing_rate_evaluation(probe_analyzers: dict[str, si.SortingAnalyzer], output_path: Path) -> QCEvaluation:
    probe_firing_rate_qc_metrics = []

    for probe_name in probe_analyzers:
        probe_firing_rate_qc_metrics.append(get_probe_firing_rate_qc(probe_analyzers[probe_name], probe_name, output_path / probe_name))
    
    return QCEvaluation(name='Interictal Events', description='Quality of firing rate across session', stage=Stage.PROCESSING,
                                modality=Modality.from_abbreviation('ecephys'), notes='', allow_failed_metrics=True, metrics=probe_firing_rate_qc_metrics)

def load_preprocessed(preprocessed_json_file, session_name, ecephys_folder, data_folder):
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


if __name__ == "__main__":
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

    quality_control_fig_folder = results_folder / "quality_control"

    job_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
    job_dicts = []
    for job_json_file in job_json_files:
        with open(job_json_file) as f:
            job_dict = json.load(f)
        job_dicts.append(job_dict)
    print(f"Found {len(job_dicts)} JSON job files")

    if len(job_dicts) == 0:
        print("Parsing AIND-specific input data")
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
                        print(f"\t{recording_name} - Times not monotonically increasing.")
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
        print(f"Found {len(job_dicts)} recordings")

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
    all_metrics = {}
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
        print(f"Recording {recording_name}")
        recording_preprocessed = None
        if ecephys_sorted_folder is not None:
            preprocessed_json_file = ecephys_sorted_folder / "preprocessed" / f"{recording_name}.json"
            recording_preprocessed = load_preprocessed(
                preprocessed_json_file, session_name, ecephys_folder, data_folder
            )
            if job_dict["skip_times"]:
                recording_preprocessed.reset_times()

        metrics = probe_raw_qc(
            recording,
            recording_name,
            quality_control_fig_folder,
            relative_to=results_folder,
            recording_lfp=recording_lfp,
            recording_preprocessed=recording_preprocessed,
            processing=processing,
            visualization_output=visualization_output,
        )
        for evaluation_name, metric_list in metrics.items():
            if evaluation_name in all_metrics:
                all_metrics[evaluation_name].extend(metric_list)
            else:
                all_metrics[evaluation_name] = metric_list

    # generate evaluations
    evaluations = []

    for evaluation_name, metrics in all_metrics.items():
        evaluation = QCEvaluation(
            modality=Modality.ECEPHYS,
            stage=Stage.RAW,
            name=evaluation_name,
            description=evaluation_name,
            metrics=metrics,
        )
        evaluations.append(evaluation)
    
    probe_postprocessed_zarr_files = sorted(tuple(ecephys_sorted_folder.glob('postprocessed/*')))
    probe_analyzers = {}

    if not quality_control_fig_folder.exists():
        quality_control_fig_folder.mkdir()

    for zarr_file in probe_postprocessed_zarr_files:
        if not (quality_control_fig_folder / zarr_file.stem).exists():
            (quality_control_fig_folder / zarr_file.stem).mkdir()

        probe_analyzers[zarr_file.stem] = si.load_sorting_analyzer(zarr_file)

    evaluations.append(get_firing_rate_evaluation(probe_analyzers, quality_control_fig_folder))
    evaluations.append(get_unit_yield_evaluation(probe_analyzers, quality_control_fig_folder))

    # make quality control
    quality_control = QualityControl(evaluations=evaluations)

    with (results_folder / "quality_control.json").open("w") as f:
        f.write(quality_control.model_dump_json(indent=3))