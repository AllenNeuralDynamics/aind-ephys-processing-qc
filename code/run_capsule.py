import pathlib
import utils
import spikeinterface as si
import numpy as np

from aind_data_schema.core.quality_control import QualityControl, QCEvaluation, Stage
from aind_data_schema_models.modalities import Modality

def get_unit_yield_evaluation(probe_analyzers: dict[str, si.SortingAnalyzer], output_path: pathlib.Path) -> QCEvaluation:
    probe_unit_yield_qc_metrics = []
    for probe_name in probe_analyzers:
        probe_unit_yield_qc_metrics.append(utils.probe_unit_yield_qc(probe_analyzers[probe_name], probe_name, output_path / probe_name))

    return QCEvaluation(name='Unit Yield', description='Quality of unit yield', stage=Stage.PROCESSING, modality=Modality.from_abbreviation('ecephys'), notes='',
                        allow_failed_metrics=True, metrics=probe_unit_yield_qc_metrics)

def get_firing_rate_evaluation(probe_analyzers: dict[str, si.SortingAnalyzer], output_path: pathlib.Path) -> QCEvaluation:
    probe_firing_rate_qc_metrics = []

    for probe_name in probe_analyzers:
        probe_firing_rate_qc_metrics.append(utils.probe_firing_rate_qc(probe_analyzers[probe_name], probe_name, output_path / probe_name))
    
    return QCEvaluation(name='Interictal Events', description='Quality of firing rate across session', stage=Stage.PROCESSING,
                                modality=Modality.from_abbreviation('ecephys'), notes='', allow_failed_metrics=True, metrics=probe_firing_rate_qc_metrics)

if __name__ == '__main__':
    probe_postprocessed_zarr_files = sorted(tuple(utils.DATA_PATH.glob('*/postprocessed/*')))
    probe_analyzers = {}
    output_path = utils.RESULTS_PATH / 'quality_control_visualizations'

    if not output_path.exists():
        output_path.mkdir()

    for zarr_file in probe_postprocessed_zarr_files:
        if not (output_path / zarr_file.stem).exists(): # TODO: update path if needed
            (output_path / zarr_file.stem).mkdir()

        probe_analyzers[zarr_file.stem] = si.load_sorting_analyzer(zarr_file)
        pass

    qc_evaluations = []
    qc_evaluations.append(get_firing_rate_evaluation(probe_analyzers, output_path))
    qc_evaluations.append(get_unit_yield_evaluation(probe_analyzers, output_path))

    qc = QualityControl(notes='Quality control of ephys processing pipeline', evaluations=qc_evaluations)
    qc.write_standard_file(utils.RESULTS_PATH)