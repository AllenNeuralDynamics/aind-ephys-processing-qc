import pathlib
import spikeinterface as si
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status
from aind_qcportal_schema.metric_value import CheckboxMetric

DATA_PATH = pathlib.Path('/data')
RESULTS_PATH = pathlib.Path('/results')

def probe_unit_yield_qc(probe_analyzer: si.SortingAnalyzer, probe_experiment_name: str, output_path: pathlib.Path) -> QCMetric:
    number_of_units = probe_analyzer.get_num_units()

    decoder_label = probe_analyzer.sorting.get_property('decoder_label')
    number_of_sua_units = decoder_label[decoder_label == 'sua'].shape[0]
    number_of_mua_units = decoder_label[decoder_label == 'mua'].shape[0]
    number_of_noise_units = decoder_label[decoder_label == 'noise'].shape[0]
    assert (number_of_mua_units + number_of_noise_units + number_of_sua_units) == number_of_units, f'Total number of units {number_of_units} does not equal {number_of_sua_units} sua units + {number_of_mua_units} mua units + {number_of_noise_units} noise units'

    default_qc = probe_analyzer.sorting.get_property('default_qc')
    fraction_of_good_units = default_qc[default_qc == True].shape[0] / number_of_units

    quality_metrics = probe_analyzer.get_extension('quality_metrics').get_data()

    fig, ax = plt.subplots(1, 5, figsize=(15,6))

    ax[0].hist(quality_metrics['isi_violations_ratio'], bins=20, density=True)
    ax[0].set_title(f'ISI Violations')
    ax[0].set_xlabel('violation rate')

    ax[1].hist(quality_metrics['amplitude_cutoff'], bins=20, density=True)
    ax[1].set_title(f'Amplitude Cutoff')
    ax[1].set_xlabel('frequency')

    ax[2].hist(quality_metrics['presence_ratio'], bins=20, density=True)
    ax[2].set_title(f'Presence Ratio')
    ax[2].set_xlabel('fraction of session')

    channel_indices = list(si.get_template_extremum_channel(probe_analyzer, mode='peak_to_peak', outputs='index').values())
    amplitudes = probe_analyzer.sorting.get_property('Amplitude')
    ax[3].scatter(amplitudes, channel_indices)
    ax[3].set_title('Unit Amplitude By Depth')
    ax[3].set_xlabel('uV')
    ax[3].set_ylabel('Channel Index')

    ax[4].scatter(quality_metrics['firing_rate'], channel_indices)
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
        timeseries_link = json_visualization[probe_experiment_name]['timeseries']
        sorting_summary_link = json_visualization[probe_experiment_name]['sorting_summary']
    else:
        timeseries_link = f'No timeseries link found for {probe_experiment_name}. Pipeline possibly errored'
        sorting_summary_link = f'No sorting summary link found for {probe_experiment_name}. Pipeline possibly errored'

    metric_values = {'number_of_sua_units': number_of_sua_units, 'number_of_mua_units': number_of_mua_units, 'number_of_noise_units': number_of_noise_units,
                    'fraction_of_units_passing_quality_metrics': fraction_of_good_units,
                    'timeseries_visualization_link': timeseries_link,
                    'sorting_summary_visualization_link': sorting_summary_link}

    value_with_options = {
        "unit quality metrics": metric_values,
        "options": ["Good", "Suspicous"],
        "status": [
            "Pass",
            "Fail",
        ],
        "type": "dropdown"
    }
    # TODO: Update reference path
    yield_metric = QCMetric(name=f'{probe_experiment_name} unit metrics yield', description=f'Evaluation of {probe_experiment_name} unit metrics yield',
                            reference=f'/{probe_experiment_name}/unit_yield.png', value=value_with_options,
                            status_history=[QCStatus(evaluator='aind-ephys-qc', timestamp=datetime.now(), status=Status.PENDING)])
    plt.close(fig)
    return yield_metric

def probe_firing_rate_qc(probe_analyzer: si.SortingAnalyzer, probe_experiment_name: str, output_path: pathlib.Path, bin_interval: int=1) -> QCMetric:
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
                        value=CheckboxMetric(value='Firing Rate Evaluation', options=['No problems detected', 'Seizure', 'Suspicious looking', 'Other issues'],
                        status=[Status.PASS, Status.FAIL, Status.FAIL, Status.PENDING]),
                        status_history=[QCStatus(evaluator='aind-ephys-qc', timestamp=datetime.now(), status=Status.PENDING)])
    
    return firing_rate_metric