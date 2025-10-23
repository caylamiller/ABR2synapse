import numpy as np
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from utils.utils import load_peak_finding_model, preprocess_waveform, peak_finding, extract_metadata, read_custom_tsv

peak_finding_model = load_peak_finding_model()

def peaks_troughs_amp_final(df, freq, db, time_scale=10):
    khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]
    if not khz.empty:
        index = khz.index.values[0]
        orig_y = df.loc[index, '0':].dropna()
        orig_y = pd.to_numeric(orig_y, errors='coerce').dropna()

        tenms = int((10/time_scale)*len(orig_y)) if time_scale>10 else len(orig_y)
        y_values_fpf = preprocess_waveform(orig_y, time_scale)

        highest_peaks, relevant_troughs = peak_finding(y_values_fpf, peak_finding_model)

        # convert back to original x-coordinates:
        if len(highest_peaks) > 0:
            highest_peaks = np.array([int((peak / 244) * tenms) for peak in highest_peaks])
            relevant_troughs = np.array([int((trough / 244) * tenms) for trough in relevant_troughs])
            # check for +/- interpolation errors:
            highest_peaks = np.array([np.argmax(orig_y[peak-1:peak+2]) + peak - 1 for peak in highest_peaks])
            relevant_troughs = np.array([np.argmin(orig_y[trough-1:trough+2]) + trough - 1 for trough in relevant_troughs])
        
            first_peak_amplitude = max(orig_y[highest_peaks[0]] - orig_y[relevant_troughs[0]], 0)

        return orig_y, highest_peaks, relevant_troughs, first_peak_amplitude
    
    return None, None, None, None

def load_data(verbose=False, db_level=70):
    time_scale=20
    amp_df_full = pd.DataFrame({'Subject': [], 'Frequency(kHz)': [], 'Level(dB)': [], 
                                'Amplitude':[], 'Peaks': [], 'Troughs': [], 
                                'Waveform': [], 'WaveI': []})
    for col in ['Waveform', 'Peaks', 'Troughs', 'WaveI']:
        amp_df_full[col] = amp_df_full[col].astype(object)
    idx = 0

    start_path = '../liberman_wpz/WPZ Electrophysiology'
    for subject in os.listdir(start_path):
        if not os.path.isdir(os.path.join(start_path, subject)):
            continue
        if verbose:
            print("Subject:",subject)
        for fq in os.listdir(os.path.join(start_path,subject)):
            if fq.startswith('ABR') and fq.endswith('.tsv'):
                path = os.path.join(start_path,subject,fq)
                data_df = read_custom_tsv(path)
                freqs = data_df['Freq(Hz)'].unique().tolist()
                levels = data_df['Level(dB)'].unique().tolist()
                for freq in freqs:
                    for lvl in levels:
                        if lvl < db_level:
                            continue
                        waveform, pks, trs, amp = peaks_troughs_amp_final(df=data_df, freq=freq, db=lvl, time_scale=time_scale)
                        amp_df_full.at[idx, 'Subject'] = subject
                        amp_df_full.at[idx, 'Frequency(kHz)'] = freq/1000
                        amp_df_full.at[idx, 'Level(dB)'] = int(lvl)
                        amp_df_full.at[idx, 'Amplitude'] = amp
                        amp_df_full.at[idx, 'Waveform'] = waveform
                        amp_df_full.at[idx, 'WaveI'] = waveform[pks[0]-10:pks[0]+20]
                        amp_df_full.at[idx, 'Peaks'] = pks
                        amp_df_full.at[idx, 'Troughs'] = trs
                        idx += 1

    raw_synapse_counts = pd.read_excel('../liberman_wpz/WPZ Ribbon and Synapse Counts.xlsx')
    raw_synapse_counts = raw_synapse_counts.mask(lambda x: x.isnull()).dropna()

    raw_synapse_counts.rename(columns={
        'Freq':'Frequency(kHz)',
        'Case':'Subject', 
        'IHCs' : 'IHCs', 
        'Synapses / IHC ' : 'SynapsesPerIHC', 
        'Orphans / IHC' : 'OrphansPerIHC'
        }, inplace=True)

    paired = amp_df_full.join(raw_synapse_counts.set_index(['Subject', 'Frequency(kHz)']), on=['Subject', 'Frequency(kHz)'])
    final = paired[['Subject', 'Frequency(kHz)', 'Level(dB)', 'Amplitude', 'vx',
                    'SynapsesPerIHC', 'IHCs', 'OrphansPerIHC', 
                    'Waveform', 'WaveI', 'Peaks', 'Troughs']]

    final_clean = final.dropna()

    # adding in the strain feature
    strains = pd.read_excel('../liberman_wpz/WPZ Mouse groups.xlsx')
    final_clean_strained = final_clean.join(strains.set_index('ID#'), on='Subject')
    final_clean_strained['Strain'] = final_clean_strained['Strain'].str.strip()

    final_clean_strained['Noise'] = final_clean_strained['Group'].apply(lambda x: 0 if 'ctrl' in x else float(x.split('dB')[0])/100)
    final_clean_strained['Time (str)'] = final_clean_strained['Group'].apply(lambda x: x.split(' ')[0] if 'ctrl' in x else x.split(' ')[1])
    final_clean_strained['Time (hrs)'] = final_clean_strained['Time (str)'].apply(lambda x: int(x.split('w')[0])*7 if 'w' in x else int(x.split('h')[0])/24)
    final_clean_strained['Strain (binary)'] = final_clean_strained['Strain'].apply(lambda x: 0 if x == 'CBA/CaJ' else 1)

    return final_clean_strained


def load_data_2(verbose=False):
    time_scale=20
    amp_df_full = pd.DataFrame({'Subject': [], 'Freq(Hz)': [], 'Levels(dB)': [], 
                                'Amplitudes':[], 'Peaks': [], 'Troughs': [], 
                                'Waveforms': [], 'WaveIs': []})
    for col in ['Levels(dB)', 'Amplitudes', 'Waveforms', 'Peaks', 'Troughs', 'WaveIs']:
        amp_df_full[col] = amp_df_full[col].astype(object)
    idx = 0

    all_levels = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

    start_path = '../liberman_wpz/WPZ Electrophysiology'
    for subject in os.listdir(start_path):
        if not os.path.isdir(os.path.join(start_path, subject)):
            continue
        if verbose:
            print("Subject:",subject)
        for fq in os.listdir(os.path.join(start_path,subject)):
            if fq.startswith('ABR') and fq.endswith('.tsv'):
                path = os.path.join(start_path,subject,fq)
                data_df = read_custom_tsv(path)
                freqs = data_df['Freq(Hz)'].unique().tolist()
                levels = data_df['Level(dB)'].unique().tolist()
                waveform1, pks, trs, amp = peaks_troughs_amp_final(df=data_df, freq=freqs[0], db=levels[0], time_scale=time_scale)
                for freq in freqs:
                    waveforms = np.zeros((len(all_levels), len(waveform1)), dtype=float)
                    peaks = np.zeros((len(all_levels), 5))
                    troughs = np.zeros((len(all_levels), 5))
                    amps = np.zeros((len(all_levels), 1))
                    waveis = np.zeros((len(all_levels), 30))
                    for lvl in levels:
                        lvl_idx = all_levels.index(lvl)
                        waveform, pks, trs, amp = peaks_troughs_amp_final(df=data_df, freq=freq, db=lvl, time_scale=time_scale)
                        waveforms[lvl_idx] = waveform
                        peaks[lvl_idx] = np.pad(pks, (0, 5 - pks.shape[0]), mode='constant', constant_values=0)
                        troughs[lvl_idx] = np.pad(trs, (0, 5 - trs.shape[0]), mode='constant', constant_values=0)
                        amps[lvl_idx] = amp
                        waveis[lvl_idx] = waveform[pks[0]-10:pks[0]+20]
                    amp_df_full.at[idx, 'Subject'] = subject
                    amp_df_full.at[idx, 'Freq(Hz)'] = freq
                    amp_df_full.at[idx, 'Levels(dB)'] = levels
                    amp_df_full.at[idx, 'Amplitudes'] = amps
                    amp_df_full.at[idx, 'Waveforms'] = waveforms
                    amp_df_full.at[idx, 'WaveIs'] = waveis
                    amp_df_full.at[idx, 'Peaks'] = peaks
                    amp_df_full.at[idx, 'Troughs'] = troughs
                    idx += 1

    raw_synapse_counts = pd.read_excel('../liberman_wpz/WPZ Ribbon and Synapse Counts.xlsx')
    raw_synapse_counts = raw_synapse_counts.mask(lambda x: x.isnull()).dropna()
    raw_synapse_counts['Freq'] = raw_synapse_counts['Freq']*1000

    raw_synapse_counts.rename(columns={
        'Freq':'Freq(Hz)',
        'Case':'Subject', 
        'IHCs' : 'IHCs', 
        'Synapses / IHC ' : 'Synapses to IHC', 
        'Orphans / IHC' : 'Orphans to IHC'
        }, inplace=True)
    
    index_cols = ['Subject', 'Freq(Hz)']
    all_cols = index_cols + ['Ribbons ', 'Synapses', 'IHCs', 'Synapses to IHC', 'Orphans to IHC']
    v1s = raw_synapse_counts[raw_synapse_counts['vx'] == 'v1'][all_cols]
    v2s = raw_synapse_counts[raw_synapse_counts['vx'] == 'v2'][all_cols]
    synapses = pd.merge(v1s, v2s, on=index_cols, suffixes=('_v1', '_v2'))


    paired = amp_df_full.join(synapses.set_index(['Subject', 'Freq(Hz)']), on=['Subject', 'Freq(Hz)'])
    final = paired[['Subject', 'Freq(Hz)', 'Levels(dB)', 'Amplitudes', 
                    'Synapses to IHC_v1', 'Synapses to IHC_v2',
                    'Orphans to IHC_v1', 'Orphans to IHC_v2',
                    'Waveforms', 'WaveIs', 'Peaks', 'Troughs']]

    final_clean = final.dropna()

    # adding in the strain feature
    strains = pd.read_excel('../liberman_wpz/WPZ Mouse groups.xlsx')
    final_clean_strained = final_clean.join(strains.set_index('ID#'), on='Subject')
    final_clean_strained['Strain'] = final_clean_strained['Strain'].str.strip()
    final_clean_strained = final_clean_strained.rename(columns={'Strain': 'Strain (x5)'})

    final_clean_strained['Noise'] = final_clean_strained['Group'].apply(lambda x: 0 if 'ctrl' in x else float(x.split('dB')[0])/100)
    final_clean_strained['Time (str)'] = final_clean_strained['Group'].apply(lambda x: x.split(' ')[0] if 'ctrl' in x else x.split(' ')[1])
    final_clean_strained['Time (hrs)'] = final_clean_strained['Time (str)'].apply(lambda x: int(x.split('w')[0])*7 if 'w' in x else int(x.split('h')[0])/24)
    final_clean_strained['Strain (binary)'] = final_clean_strained['Strain (x5)'].apply(lambda x: 0 if x == 'CBA/CaJ' else 1)

    return final_clean_strained