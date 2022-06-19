import numpy as np
import pandas as pd
import pickle as pkl 
import xarray as xr
import copy
import os
import sys 
import metrics
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import kstest
import pylab 
import scipy.stats as stats
from numpy.linalg import LinAlgError
from IPython.display import clear_output
print("XArray version: ", xr.__version__)
colz = ['#1f77b4', '#ff7f0e', '#2ca02c', '#bdbdbd']




##########################################################################################################
# LOAD IN THE DATA
##########################################################################################################
with open('./model_output_for_analysis/nwm_chrt_v2_1d_local.p', 'rb') as fb: 
    nwm_results = pkl.load(fb)

lstm_results_time_split1={}
mclstm_results_time_split1={}
sacsma_results_time_split1={}
lstm_results_time_split2={}
mclstm_results_time_split2={}
sacsma_results_time_split2={}

for forcing_type in ['nldas', 'daymet']:
    
    with open('./model_output_for_analysis/lstm_time_split1_{}_ens.p'.format(forcing_type), 'rb') as fb: 
        lstm_results_time_split1[forcing_type] = pkl.load(fb)
    with open('./model_output_for_analysis/mclstm_time_split1_{}_ens.p'.format(forcing_type), 'rb') as fb: 
        mclstm_results_time_split1[forcing_type] = pkl.load(fb)
    with open('./model_output_for_analysis/sacsma_time_split1_{}_ens.p'.format(forcing_type), 'rb') as fb: 
        sacsma_results_time_split1[forcing_type] = pkl.load(fb)

    with open('./model_output_for_analysis/lstm_time_split2_{}.p'.format(forcing_type), 'rb') as fb: 
        lstm_results_time_split2[forcing_type] = pkl.load(fb)
    with open('./model_output_for_analysis/mclstm_time_split2_{}.p'.format(forcing_type), 'rb') as fb: 
        mclstm_results_time_split2[forcing_type] = pkl.load(fb)
    with open('./model_output_for_analysis/sacsma_time_split2_{}.p'.format(forcing_type), 'rb') as fb: 
        sacsma_results_time_split2[forcing_type] = pkl.load(fb)

train_split_type_model_set = {'time_split1':{'nwm':nwm_results, 
                                           'lstm':lstm_results_time_split1,
                                            'mc':mclstm_results_time_split1,
                                            'sac':sacsma_results_time_split1},
                              'time_split2':{'nwm':nwm_results,
                                           'lstm':lstm_results_time_split2,
                                            'mc':mclstm_results_time_split2,
                                            'sac':sacsma_results_time_split2}}




##########################################################################################################
# USE A CONVERSION BETWEEN MODELS AND DATA
##########################################################################################################
# Convert flow to   CFS mm -> ft     km^2 -> ft^2    hr->s
conversion_factor = 0.00328084 * 10763910.41671 / 3600 / 24




##########################################################################################################
# Get all the CAMELS attributes.  
##########################################################################################################

# Camels attributes with RI information
dataName = '../data/camels_attributes.csv'
# load the data with pandas
pd_attributes = pd.read_csv(dataName, sep=',', index_col='gauge_id')

# Add the basin ID as a 8 element string with a leading zero if neccessary
basin_id_str = []
for a in pd_attributes.index.values:
    basin_id_str.append(str(a).zfill(8))
pd_attributes['basin_id_str'] = basin_id_str



# Get the hydrologic units for each basin.
with open('../data/usgs_site_info.csv', 'r') as f:
    usgs_sites = pd.read_csv(f, skiprows=24, index_col='site_no')
usgs_idx_int = []
for idx in usgs_sites.index.values:
    usgs_idx_int.append(int(idx))
usgs_sites.reindex(usgs_idx_int)
usgs_sites = usgs_sites.reindex(usgs_idx_int)
basin_hydro_unit = []
for b in pd_attributes.basin_id_str.values:
    huc_cd = usgs_sites.loc[int(b),'huc_cd']
    hu = '{:08d}'.format(huc_cd)
    basin_hydro_unit.append(hu[0:2])
pd_attributes['basin_hydro_unit'] = basin_hydro_unit
huc_regions = set(list(pd_attributes['basin_hydro_unit']))




huc_regions = pd_attributes.basin_hydro_unit.unique()



def get_basin_region(pd_attributes, basin_0str):
    return pd_attributes.loc[int(basin_0str), "basin_hydro_unit"]




##########################################################################################################
# Loop through all the SACSMA runs and check that the results are good. 
# Get a list of basins that has good calibration results.
##########################################################################################################
basin_list_all_camels = list(pd_attributes['basin_id_str'].values)
basin_list_sacsma_good = {ts:copy.deepcopy(basin_list_all_camels) for ts in ['time_split1', 'time_split2']}

for ib, basin_0str in enumerate(basin_list_all_camels): 
    remove_basin_id_from_list = False
    for train_split_type in ['time_split1', 'time_split2']:
        for forcing_type in ['nldas', 'daymet']:

            if basin_0str not in list(train_split_type_model_set[train_split_type]['sac'][forcing_type].columns):
                remove_basin_id_from_list = True
            elif train_split_type_model_set[train_split_type]['sac'][forcing_type][basin_0str].sum() <=0:
                remove_basin_id_from_list = True

            if train_split_type == 'time_split2' and forcing_type == 'nldas':
                if basin_0str not in list(train_split_type_model_set[train_split_type]['nwm'].keys()):
                    remove_basin_id_from_list = True

    if remove_basin_id_from_list:
        basin_list_sacsma_good[train_split_type].remove(basin_0str)




##########################################################################################################
# REVERT TO THESE AS THE FLOWS
##########################################################################################################
flows = ['lstm', 'mc', 'sac', 'obs']




def get_specifications(tsplt, forcing_type):
    """
    This function is designed to return specific details of the simulation period
    Inputs:
        tsplit (str): Either time_split2 or time_split1
        forcing_type (str): Either nldas or daymet
    Returns
        start_date (pd.Timestamp): The date the simulation period started
        end_date (pd.Timestamp): The date the simulation period ended
        labelz (dictionary): A mapping between short model name and long model name
        models (list): the short model names
        flows (list): the short model names plus "obs" for observed flow
        basin_list (list): the list of basins that meet the criteria for analysis
        tsplit (str): Either time_split2 or time_split1
        forcing_type (str): Either nldas or daymet
    """
    if tsplt == 'time_split2' and forcing_type == 'nldas':
        start_date = pd.Timestamp('1996-10-01')
        end_date = pd.Timestamp('2014-01-01')
        labelz={'nwm':'NWM*', 'lstm':'LSTM', 'mc':'MC-LSTM','sac':'SAC-SMA', 'obs':'Observed'}
        models = ['nwm', 'lstm', 'mc', 'sac']
        flows = ['nwm', 'lstm', 'mc', 'sac', 'obs']
        basin_list = list(lstm_results_time_split2[forcing_type].keys())[:-1]
    elif tsplt == 'time_split2':
        start_date = pd.Timestamp('1996-10-01')
        end_date = pd.Timestamp('2014-01-01')
        labelz={'lstm':'LSTM', 'mc':'MC-LSTM','sac':'SAC-SMA', 'obs':'Observed'}
        models = ['lstm', 'mc', 'sac']
        flows = ['lstm', 'mc', 'sac', 'obs']
        basin_list = list(lstm_results_time_split2[forcing_type].keys())[:-1]
    else:
        start_date = pd.Timestamp('1989-10-01')
        end_date = pd.Timestamp('1999-09-30')
        labelz={'lstm':'LSTM', 'mc':'MC-LSTM','sac':'SAC-SMA', 'obs':'Observed'}
        models = ['lstm', 'mc', 'sac']
        flows = ['lstm', 'mc', 'sac', 'obs']
        basin_list = list(lstm_results_time_split1[forcing_type].keys())[:-1]

    spex = {"start_date":start_date,
            "end_date":end_date,
            "labelz":labelz,
            "models":models,
            "flows":flows, 
            "basin_list":basin_list,
            "tsplt":tsplt,
            "forcing_type":forcing_type}
    return spex #(start_date, end_date, labelz, models, flows, basin_list)




def get_precip_and_flows(tsplt, forcing_type, basin_0str, start_date, end_date):
    #-------------------------------------------------------------------------------------------------
    # Make dictionary with all the flows
    flow_mm = {}
    #-------------------------------------------------------------------------------------------------
    if tsplt == 'time_split2' and forcing_type == 'nldas':
        #-------------------------------------------------------------------------------------------------
        # We need the basin area to convert to CFS
        basin_area = pd_attributes.loc[int(basin_0str), 'area_geospa_fabric']\
        #-------------------------------------------------------------------------------------------------

        # Get the NWM data for this basin in an xarray dataset.
        xr_nwm = xr.DataArray(nwm_results[basin_0str]['streamflow'].values,
                 coords=[nwm_results[basin_0str]['streamflow'].index],
                 dims=['datetime'])
        # convert from CFS to mm/day
        # fm3/s * 3600 sec/hour * 24 hour/day / (m2 * mm/m)
        xrr = xr_nwm.loc[start_date:end_date]*3600*24/(basin_area*1000)
        flow_mm['nwm'] = pd.DataFrame(data=xrr.values)
    #-------------------------------------------------------------------------------------------------
    # Standard LSTM 
    if tsplt == 'time_split1':
        xrr = lstm_results_time_split1[forcing_type][basin_0str]['1D']['xr']['QObs(mm/d)_sim'].loc[start_date:end_date]
        flow_mm['lstm'] = pd.DataFrame(data=xrr.values,index=xrr.datetime.values)
    if tsplt == 'time_split2':
        xrr = lstm_results_time_split2[forcing_type][basin_0str]['1D']['xr']['QObs(mm/d)_sim'].loc[start_date:end_date]
        flow_mm['lstm'] = pd.DataFrame(data=xrr.values,index=xrr.date.values)
    #-------------------------------------------------------------------------------------------------
    # Mass-conserving LSTM data trained on all years
    if tsplt == 'time_split1':
        xrr = mclstm_results_time_split1[forcing_type][basin_0str]['1D']['xr']['QObs(mm/d)_sim'].loc[start_date:end_date]
        flow_mm['mc'] = pd.DataFrame(data=xrr.values,index=xrr.datetime.values)
    if tsplt == 'time_split2':
        xrr = mclstm_results_time_split2[forcing_type][basin_0str]['1D']['xr']['QObs(mm/d)_sim'].loc[start_date:end_date]
        flow_mm['mc'] = pd.DataFrame(data=xrr.values,index=xrr.date.values)
    #-------------------------------------------------------------------------------------------------
    # SACSMA Mean
    if tsplt == 'time_split1':
        df = sacsma_results_time_split1[forcing_type][basin_0str].loc[start_date:end_date]
    if tsplt == 'time_split2':
        df = sacsma_results_time_split2[forcing_type][basin_0str].loc[start_date:end_date]
    flow_mm['sac'] = df
    #-------------------------------------------------------------------------------------------------
    # OBSERVATIONS
    if tsplt == 'time_split1':
        xrr = mclstm_results_time_split1[forcing_type][basin_0str]['1D']['xr']['QObs(mm/d)_obs'].loc[start_date:end_date]
        flow_mm['obs'] = pd.DataFrame(data=xrr.values,index=xrr.datetime.values)
    if tsplt == 'time_split2':
        xrr = mclstm_results_time_split2[forcing_type][basin_0str]['1D']['xr']['QObs(mm/d)_obs'].loc[start_date:end_date]
        flow_mm['obs'] = pd.DataFrame(data=xrr.values,index=xrr.date.values)

    #-------------------------------------------------------------------------------------------------
    # Make sure we are in a time period that all the flow members have values
    # If there is missin observations than we can't compare the mass of the observed with simulaitons
    skip_basin_because_missing_obs = False
    if tsplt == 'time_split1':
        obs_temp = mclstm_results_time_split1[forcing_type][basin_0str]['1D']['xr']['QObs(mm/d)_obs'].datetime
    if tsplt == 'time_split2':
        obs_temp = mclstm_results_time_split2[forcing_type][basin_0str]['1D']['xr']['QObs(mm/d)_obs'].date
        
    return flow_mm, obs_temp







forcing_products = ['nldas','daymet']
file_name_map = {'nldas':'nldas', 'daymet':'cida'}
precip_column_map = {'nldas':'PRCP(mm/day)', 'daymet':'prcp(mm/day)'}








def precipitation_event_thresholds(percent_threshold, forcing):
    precip_threshold = 0
    # Event definition threshold
    any_precip = forcing[forcing>0].values
    any_precip.sort()
    onethrough = np.array([i for i in range(any_precip.shape[0])])/any_precip.shape[0]
    for i in range(any_precip.shape[0]):
        if onethrough[i] > percent_threshold:
            precip_threshold = any_precip[i]
            break
    return precip_threshold

def load_forcing_and_identify_events(tsplt, basin_0str, file_name_map, forcing_type):
    """
    This function loads in the forcing, and also identifies the indices of precipitation "events"
    Events are arbitrarily defined as any time the precipitation is greater than the median (non zero) precip
    
    Inputs:
        tsplit (str): Either time_split2 or time_split1
        basin_0str (str): The basin ID as a string with a leading zero
        forcing_dir (str): The directory where to find the forcing file
        file_name_map (dictionary): 
        forcing_type (str): either nldas or daymet
        precip_threshold (float): The decimal number representing the event threshold percentage
    Return:
        forcing (pd.DataFrame): The forcing data for a particular basin
        precip_events (list): Indices of official precipitation "events"
    """
    
    forcing_dir = '/home/NearingLab/data/camels_data/basin_dataset_public_v1p2'+\
        f'/basin_mean_forcing/{forcing_type}_all_basins_in_one_directory/'
    
    basin_int = int(basin_0str)
    #-------------------------------------------------------------------------------------------------
    # FORCING
    forcing = pd.read_csv(f'{forcing_dir}{basin_0str}_lump_{file_name_map[forcing_type]}_forcing_leap.txt',
                          delim_whitespace=True, header=3)
    if tsplt == 'time_split1':
        forcing = forcing.iloc[3560:7214]
    if tsplt == 'time_split2':
        forcing = forcing.iloc[6118:]
    forcing.index=pd.to_datetime((forcing.Year*10000+forcing.Mnth*100+forcing.Day).apply(str),format='%Y%m%d')
    #-------------------------------------------------------------------------------------------------
    
    f_label = precip_column_map[forcing_type]
    
    
    precip_threshold_peak = precipitation_event_thresholds(0.25, forcing[f_label])
    precip_threshold_low = precipitation_event_thresholds(0.05, forcing[f_label])   
    
    # Identify events (criteria: precip days with two precip free days before and after)
    max_precip_event = 10
    event_window = 2
    precip_event_start_end_index = []
    for i in range(3, forcing[f_label].shape[0]-max_precip_event-event_window):
        if i < event_window or i > forcing[f_label].shape[0]-event_window:
            continue
        if forcing[f_label][i] > precip_threshold_peak:
            if np.sum(forcing[f_label][i-event_window:i]) < precip_threshold_low:
                for j in range(i+1,i+max_precip_event):
                    if forcing[f_label][j] == 0:
                        if np.sum(forcing[f_label][j:j+event_window]) < precip_threshold_low:
                            precip_event_start_end_index.append([i-event_window,j+event_window+1])
                        break
    print(f"Number of precipitation events {len(precip_event_start_end_index)} above threshold {precip_threshold_peak}mm and a window below {precip_threshold_low}mm")
    
    return forcing, precip_event_start_end_index





def calculate_mass_balance_over_events(basin_0str, spex, forcing, precip_events, total_mass, r):
    
    basin_int = int(basin_0str)
    start_date = spex["start_date"]
    end_date = spex["end_date"]
    tsplt = spex["tsplt"]
    models = spex['models']
    flows = spex['flows']
    forcing_type = spex['forcing_type']
    
    mass_balance_over_events = pd.DataFrame(columns=["event",
                                                     "event_date",
                                                     "event_days",
                                                     "total_precip", 
                                                     "total_obs", 
                                                     "IN_obs",
                                                     "total_lstm",
                                                     "total_mc",
                                                     "total_sac",
                                                     "RR_obs",
                                                     "AME_lstm",
                                                     "PME_lstm",
                                                     "NME_lstm",
                                                     "RR_lstm",
                                                     "IN_lstm"
                                                     "AME_mc",
                                                     "PME_mc",
                                                     "NME_mc",
                                                     "RR_mc",
                                                     "IN_mc",
                                                     "AME_sac",
                                                     "PME_sac",
                                                     "NME_sac",
                                                     "RR_sac",
                                                     "IN_sac"])
    if "nwm" in models:
        mass_balance_over_events["total_nwm"] = np.nan
        mass_balance_over_events["AME_nwm"] = np.nan
        mass_balance_over_events["PME_nwm"] = np.nan
        mass_balance_over_events["NME_nwm"] = np.nan
        mass_balance_over_events["RR_nwm"] = np.nan
        mass_balance_over_events["IN_nwm"] = np.nan        
    
    #-------------------------------------------------------------------------------------------------    
    flow_mm, obs_temp = get_precip_and_flows(tsplt, forcing_type, basin_0str, start_date, end_date)
    #-------------------------------------------------------------------------------------------------

                
        
    #-------------------------------------------------------------------------------------------------
    #################    DO MASS PER EVENT
    #-------------------------------------------------------------------------------------------------

    total_mass[r][forcing_type][tsplt][basin_0str] = {}
    
    
    #-------------------------------------------------------------------------------------------------
    # Define the event as the index of the maximum precipitation within the window.    
    for ievent, (sevd, eevd) in enumerate(precip_events):
        
        # The NWM ends before the other models. So stop time split 2 here.
        if tsplt == "time_split2" and eevd > 6301:
            continue
        
        max_precip=0
        for event_day in range(sevd, eevd+1):
            if forcing[precip_column_map[forcing_type]].values[event_day] > max_precip:
                max_precip = forcing[precip_column_map[forcing_type]].values[event_day]
                event = event_day
    
        #-------------------------------------------------------------------------------------------------
        # Set the total mass to zero for this basin    
        total_mass[r][forcing_type][tsplt][basin_0str][event] = {flow:0 for flow in flows}
        
        
        #-------------------------------------------------------------------------------------------------
        # Get the initial and total mass of each flow
        initial_event_flow = {flow:0 for flow in flows}
        for flow in flows:
            
            _flow = np.array(flow_mm[flow].iloc[sevd:eevd+1]).flatten()
            initial_event_flow[flow] = _flow[0]
            total_mass[r][forcing_type][tsplt][basin_0str][event][flow] = np.sum(_flow)
        if total_mass[r][forcing_type][tsplt][basin_0str][event]['obs'] == 0:
            continue
        else:
            mass_balance_over_events.loc[event,'event'] = event

        #-------------------------------------------------------------------------------------------------
        # Start filling in the event data
        ts = pd.to_datetime(str(forcing[precip_column_map[forcing_type]].index.values[event])) 
        d = ts.strftime('%Y.%m.%d')
        mass_balance_over_events.loc[event,'event_date'] = d
        mass_balance_over_events.loc[event,'event_days'] = eevd - sevd

        _precip = np.sum(forcing[precip_column_map[forcing_type]].values[sevd:eevd])
        mass_balance_over_events.loc[event,'total_precip'] = _precip
        
        mass_balance_over_events.loc[event,'total_obs'] = \
            total_mass[r][forcing_type][tsplt][basin_0str][event]['obs']

        mass_balance_over_events.loc[event,'RR_obs'] = \
            mass_balance_over_events.loc[event,'total_obs'] / \
            mass_balance_over_events.loc[event,'total_precip']

        for model in models:
            mass_balance_over_events.loc[event,f'total_{model}'] = \
                total_mass[r][forcing_type][tsplt][basin_0str][event][model]
        for flow in flows:
            mass_balance_over_events.loc[event,f'IN_{flow}'] = initial_event_flow[flow]
            
            
        #-------------------------------------------------------------------------------------------------
        # Calculate the model metrics
        for model in models:
            
            _obs = total_mass[r][forcing_type][tsplt][basin_0str][event]['obs']
            if _obs == 0:
                break
            _sim = total_mass[r][forcing_type][tsplt][basin_0str][event][model]
            
            mass_balance_over_events.loc[event,f'AME_{model}'] = np.abs(_sim - _obs) / _obs
            if (_sim - _obs) > 0:
                mass_balance_over_events.loc[event,f'PME_{model}'] = (_sim - _obs) / _obs
                mass_balance_over_events.loc[event,f'NME_{model}'] = 0
            else:
                mass_balance_over_events.loc[event,f'NME_{model}'] = (_sim - _obs) / _obs
                mass_balance_over_events.loc[event,f'PME_{model}'] = 0
                
            mass_balance_over_events.loc[event,f'RR_{model}'] = _sim / _precip

    return mass_balance_over_events




event_results_region_file = "event_results_region.pkl"
if True:
    with open(event_results_region_file, "rb") as fb:
        events_results = pkl.load(fb)
else:
    ##########################################################################################################
    # IDENTIFY EVENTS WITH PRECIP OVER 10mm
    # THEN DO THE MASS BALANCE CALC OVER SOME WINDOW
    ##########################################################################################################

    total_mass = {r:{forcing_type:{time_split:{} for time_split in ['time_split1', 'time_split2']} for forcing_type in forcing_products} for r in huc_regions}

    events_results = {r:{tsplt:{forcing_type:{} for forcing_type in forcing_products} for tsplt in ['time_split1', 'time_split2']} for r in huc_regions}

    for r in huc_regions:
        
        for tsplt in ['time_split2', 'time_split1']:

            for forcing_type in forcing_products:

                forcing_dir = '/home/NearingLab/data/camels_data/basin_dataset_public_v1p2'+\
                    '/basin_mean_forcing/{}_all_basins_in_one_directory/'.format(forcing_type)

                spex = get_specifications(tsplt, forcing_type)

                clear_every_5 = 0
                for basin_0str in spex["basin_list"]:

                    ########################################################
                    ######     THIS IS A TEMPORARY HACK     ################
                    if get_basin_region(pd_attributes, basin_0str) != r:
                        continue

                    print(basin_0str)

                    forcing, precip_events = load_forcing_and_identify_events(tsplt, 
                                                                              basin_0str, 
                                                                              file_name_map, 
                                                                              forcing_type)

                    events_results[r][tsplt][forcing_type][basin_0str] = calculate_mass_balance_over_events(basin_0str, 
                                                                                  spex, 
                                                                                  forcing, 
                                                                                  precip_events,
                                                                                  total_mass, r)

    

                
    with open(event_results_region_file, "wb") as fb:
        pkl.dump(events_results, fb)





# Get rid of anything with NaNs
def remove_nans_from_results(events_results):
    """
        We can't do the analysis with NaNs, 
        so at least a couple of times we'll have to drop them from the results.
    
        Args:
            events_results (dict): A dictionary that has the event results for each:
                time split, forcing type and basin
        
    """
    
    for r in huc_regions:
    
        for tsplt in ['time_split2', 'time_split1']:
            for forcing_type in forcing_products:
                spex = get_specifications(tsplt, forcing_type)
                for basin_0str in spex["basin_list"]:

                    if get_basin_region(pd_attributes, basin_0str) != r:
                        continue

                    df_temporary = events_results[r][tsplt][forcing_type][basin_0str]
                    for i in df_temporary.index:
                        if np.isnan(df_temporary['total_obs'][i]):
                            df_temporary = df_temporary.drop([i])
                        elif tsplt == "time_split2" and forcing_type == "nldas":
                            if np.isnan(df_temporary['RR_nwm'][i]):
                                df_temporary = df_temporary.drop([i])
                            
    return events_results




events_results = remove_nans_from_results(events_results)




def mahalanobis_distances(df, axis=0):
    '''
    Returns a pandas Series with Mahalanobis distances for each sample on the
    axis.

    Note: does not work well when # of observations < # of dimensions
    Will either return NaN in answer
    or (in the extreme case) fail with a Singular Matrix LinAlgError

    Args:
        df: pandas DataFrame with columns to run diagnostics on
        axis: 0 to find outlier rows, 1 to find outlier columns
    '''
    df = df.transpose() if axis == 1 else df
    means = df.mean()
    try:
        inv_cov = np.linalg.inv(df.cov())
    except LinAlgError:
        return pd.Series([np.NAN] * len(df.index), df.index,
                         name='Mahalanobis')
    dists = []
    for i, sample in df.iterrows():
        dists.append(distance.mahalanobis(sample, means, inv_cov))

    return pd.Series(dists, df.index, name='Mahalanobis') 



# Calculate the location of each runoff event within the "local" distribution
print("Calculate the location of each runoff event within the local distribution")
for r in huc_regions:
    
    print(f"Working on HUC REGION: {r}")

    for tsplt in ['time_split1', 'time_split2']:
        for forcing_type in forcing_products:
            spex = get_specifications(tsplt, forcing_type)
            for basin_0str in spex["basin_list"]:

                if get_basin_region(pd_attributes, basin_0str) != r:
                    continue

                for flow in spex['flows']:
                    u = events_results[r][tsplt][forcing_type][basin_0str][f'IN_{flow}']
                    v = events_results[r][tsplt][forcing_type][basin_0str][f'total_precip']
                    df = pd.DataFrame(data=[u,v], columns=u.index.values).transpose()
                    events_results[r][tsplt][forcing_type][basin_0str][f'MD_{flow}'] = mahalanobis_distances(df)




# Count basins with missing SAC predictions, these have NaNs and cannot be used.
print("Calculate the location of each runoff event within the local distribution")
count_nans=0
count_nonans=0
for r in huc_regions:
    print(f"Working on HUC REGION: {r}")
    for tsplt in ['time_split1', 'time_split2']:
        for forcing_type in forcing_products:
            spex = get_specifications(tsplt, forcing_type)
            for basin_0str in spex["basin_list"]:

                if get_basin_region(pd_attributes, basin_0str) != r:
                    continue


                if np.sum(np.isnan(events_results[r][tsplt][forcing_type][basin_0str].MD_sac)) > 0:
                    count_nans+=1
                else:
                    count_nonans+=1

print("Number of NaNs", count_nans)
print("Number of Not NaNs", count_nonans)




############################################
def get_cum_dist_x_y(series):
    x = list(series)
    y = list(range(len(x)))
    for i, _ in enumerate(y):
        y[i] = y[i]/len(y)
    x.sort()
    return x, y




#####################################################
def calculate_rr_location_within_nn_distribution(rr, rr_dist):
    """
        Args:
            rr (float): A runoff ratio for a particular event. 
            rr_dist (series): a sorted series of runoff ratios

    """

    x,y = get_cum_dist_x_y(rr)

    percentile = None
    
    for i, j in zip(x,y):
        if i > rr:
            percentile = y

    return percentile




from sklearn.metrics import r2_score
def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi







##############################################################
##############################################################
##############################################################
print("Event based analysis (Note: an event is defined by the magnitude of precipitation, not runoff)")



for r in huc_regions:

    print(" \n")
    print(" \n")
    print(f"HUC REGION {r}")

    for tsplt in ['time_split1', 'time_split2']:

        spex = get_specifications(tsplt, 'nldas')

        rr_nldas_obs = []
        rr_nldas_lstm = []
        rr_nldas_mc = []
        rr_nldas_sac = []
        rr_daymet_obs = []
        rr_daymet_lstm = []
        rr_daymet_mc = []
        rr_daymet_sac = []

        if tsplt == 'time_split2':
            rr_nldas_nwm = []

        for i, basin_0str in enumerate(spex["basin_list"]):


            ########################################################
            ########################################################
            if get_basin_region(pd_attributes, basin_0str) != r:
                continue


            _nldas = events_results[r][tsplt]['nldas'][basin_0str]
            nn_nldas = np.min([100, _nldas.shape[0]])
            _daymet = events_results[r][tsplt]['daymet'][basin_0str]
            nn_daymet = np.min([100, _daymet.shape[0]])

            if tsplt == 'time_split2':
                if np.isnan(list(_nldas.RR_nwm.values)).sum() == 0:
                    if np.isnan(list(_nldas.RR_obs.values)).sum() == 0:
                        rr_nldas_nwm.extend(list(_nldas.sort_values('MD_obs')['RR_nwm'][:nn_nldas]))                
                        rr_nldas_obs.extend(list(_nldas.sort_values('MD_obs')['RR_obs'][:nn_nldas]))
                        rr_nldas_lstm.extend(list(_nldas.sort_values('MD_obs')['RR_lstm'][:nn_nldas])) 
                        rr_nldas_mc.extend(list(_nldas.sort_values('MD_obs')['RR_mc'][:nn_nldas])) 
                        rr_nldas_sac.extend(list(_nldas.sort_values('MD_obs')['RR_sac'][:nn_nldas])) 
            elif np.isnan(list(_nldas.RR_obs.values)).sum() == 0:
                rr_nldas_obs.extend(list(_nldas.sort_values('MD_obs')['RR_obs'][:nn_nldas]))
                rr_nldas_lstm.extend(list(_nldas.sort_values('MD_obs')['RR_lstm'][:nn_nldas])) 
                rr_nldas_mc.extend(list(_nldas.sort_values('MD_obs')['RR_mc'][:nn_nldas])) 
                rr_nldas_sac.extend(list(_nldas.sort_values('MD_obs')['RR_sac'][:nn_nldas]))

            if np.isnan(list(_daymet.RR_obs.values)).sum() ==0:
                rr_daymet_obs.extend(list(_daymet.sort_values('MD_obs')['RR_obs'][:nn_daymet]))
                rr_daymet_lstm.extend(list(_daymet.sort_values('MD_obs')['RR_lstm'][:nn_daymet])) 
                rr_daymet_mc.extend(list(_daymet.sort_values('MD_obs')['RR_mc'][:nn_daymet])) 
                rr_daymet_sac.extend(list(_daymet.sort_values('MD_obs')['RR_sac'][:nn_daymet]))

        df = pd.DataFrame(columns=["forcing", "model", "MI", "R2", "n"])
        df.loc[len(df)] = ["NLDAS", "LSTM", 
                 np.round(calc_MI(rr_nldas_lstm, rr_nldas_obs, 100),3),
                 np.round(r2_score(rr_nldas_lstm, rr_nldas_obs),3),
                 len(rr_nldas_obs)]
        df.loc[len(df)] = ["NLDAS", "MC-LSTM", 
                 np.round(calc_MI(rr_nldas_mc, rr_nldas_obs, 100),3),
                 np.round(r2_score(rr_nldas_mc, rr_nldas_obs),3),
                 len(rr_nldas_obs)]
        df.loc[len(df)] = ["NLDAS", "SAC-SMA", 
                 np.round(calc_MI(rr_nldas_sac, rr_nldas_obs, 100),3),
                 np.round(r2_score(rr_nldas_sac, rr_nldas_obs),3),
                 len(rr_nldas_obs)]
        if tsplt == 'time_split2':
            df.loc[len(df)] = ["NLDAS", "NWM", 
                 np.round(calc_MI(rr_nldas_nwm, rr_nldas_obs, 100),3),
                 np.round(r2_score(rr_nldas_nwm, rr_nldas_obs),3),
                 len(rr_nldas_obs)]
        df.loc[len(df)] = ["Daymet", "LSTM", 
                 np.round(calc_MI(rr_daymet_lstm, rr_daymet_obs, 100),3),
                 np.round(r2_score(rr_daymet_lstm, rr_daymet_obs),3),
                 len(rr_daymet_obs)]
        df.loc[len(df)] = ["Daymet", "MC-LSTM", 
                 np.round(calc_MI(rr_daymet_mc, rr_daymet_obs, 100),3),
                 np.round(r2_score(rr_daymet_mc, rr_daymet_obs),3),
                 len(rr_daymet_obs)]
        df.loc[len(df)] = ["Daymet", "SAC-SMA", 
                 np.round(calc_MI(rr_daymet_sac, rr_daymet_obs, 100),3),
                 np.round(r2_score(rr_daymet_sac, rr_daymet_obs),3),
                 len(rr_daymet_obs)]
        print(f"For {tsplt}")
        print("Mutual information between observed and predicted Runoff Ratios")
        print(f"{len(rr_nldas_obs)} total events with NLDAS")
        print("NLDAS LSTM", np.round(calc_MI(rr_nldas_lstm, rr_nldas_obs, 100),3))
        print("NLDAS MC-LSTM", np.round(calc_MI(rr_nldas_mc, rr_nldas_obs, 100),3))
        print("NLDAS Sac-SMA", np.round(calc_MI(rr_nldas_sac, rr_nldas_obs, 100),3))
        if tsplt == 'time_split2':
            print("NLDAS NWM", np.round(calc_MI(rr_nldas_nwm, rr_nldas_obs, 100),3))
        print(f"{len(rr_daymet_obs)} total events with Daymet")
        print("Daymet LSTM", np.round(calc_MI(rr_daymet_lstm, rr_daymet_obs, 100),3))
        print("Daymet MC-LSTM", np.round(calc_MI(rr_daymet_mc, rr_daymet_obs, 100),3))
        print("Daymet Sac-SMA", np.round(calc_MI(rr_daymet_sac, rr_daymet_obs, 100),3))
        print(" ")
        print(df)



import statsmodels.api as sm


event_stats_100_nearest_neibors_region_file = "event_stats_100_nearest_neibors_region.pkl"
if False:
    with open(event_stats_100_nearest_neibors_region_file, "rb") as fb:
        event_100 = pkl.load(fb)
else:
    event_100 = {r:{'time_split1':{'nldas':{},
                                'daymet':{}},
                 'time_split2':{'nldas':{},
                                'daymet':{}}} for r in huc_regions}

    for r in huc_regions:
        
        print(" \n")
        print(" \n")
        print(f"HUC REGION {r}")
        
        for tsplt in ['time_split2','time_split1']:
            print(tsplt)

            for forcing_type in forcing_products:
                print(forcing_type)

                spex = get_specifications(tsplt, forcing_type)

                for model in spex['models']:
                    print(model)

                    event_100[r][tsplt][forcing_type][model] = {"md_ks":[], "md_mi":[], "md_r2":[], "md_perc":[], 
                                                             "rr_ks":[], "rr_mi":[], "rr_r2":[], "rr_perc":[]}

                    for basin_0str in spex["basin_list"]:


                        ########################################################
                        ########################################################
                        if get_basin_region(pd_attributes, basin_0str) != r:
                            continue


                        # Don't include any models if SAC-SMA is NaN
                        if np.isnan(events_results[r][tsplt][forcing_type][basin_0str][f'MD_sac']).sum() > 0:
                            continue
                        # Don't include any models if Observation is NaN
                        if np.isnan(events_results[r][tsplt][forcing_type][basin_0str][f'MD_obs']).sum() > 0:
                            continue

                        df = events_results[r][tsplt][forcing_type][basin_0str]

                        md_obs = df[f'MD_obs']
                        rr_obs = df[f'RR_obs']

                        md_model = df[f'MD_{model}']
                        rr_model = df[f'RR_{model}']

                        for event in df.index.values:

                            this_event_obs_md = md_model[event]
                            this_event_model_md = md_model[event]

                            this_event_obs_rr = rr_obs[event]
                            this_event_model_rr = rr_model[event]

                            # Now find 100 closest events!

                            md_model_100 = md_model.iloc[(df[f'MD_{model}']-this_event_model_md).abs().argsort()[:100]]
                            model_indx = md_model_100.index.values
                            md_obs_100 = md_model.iloc[(df[f'MD_obs']-this_event_model_md).abs().argsort()[:100]]
                            obs_indx = md_model_100.index.values

                            rr_model_100 = df.loc[model_indx, f'RR_{model}']
                            rr_obs_100 = df.loc[obs_indx, f'RR_obs']

                            # For the R2 metric, there needs to be a 1 to 1 correspondence between events
                            # it is not 'simple' the distributions we are interested in,
                            #     but the ability to explain the variance of observation with the model.
                            rr_obs_100_mod = df.loc[model_indx, f'RR_obs']
                            md_obs_100_mod = df.loc[model_indx, f'MD_obs']

                            event_100[r][tsplt][forcing_type][model]["md_ks"].append(kstest(md_model_100, md_obs_100)[0])
                            event_100[r][tsplt][forcing_type][model]["md_mi"].append(calc_MI(md_model_100, md_obs_100, 100))

                            event_100[r][tsplt][forcing_type][model]["rr_ks"].append(kstest(rr_model_100, rr_obs_100)[0])
                            event_100[r][tsplt][forcing_type][model]["rr_mi"].append(calc_MI(rr_model_100, rr_obs_100, 100))

                            # NOTE: The R2 score needs to have the same events to make a 1 to 1 correspondence.
                            # NOTE: Could do both for the MI and KS as well...
                            event_100[r][tsplt][forcing_type][model]["md_r2"].append(r2_score(md_model_100, md_obs_100_mod))
                            event_100[r][tsplt][forcing_type][model]["rr_r2"].append(r2_score(rr_model_100, rr_obs_100_mod))

                            event_100[r][tsplt][forcing_type][model]["md_perc"].append(stats.percentileofscore(md_model_100,this_event_model_md))
                            event_100[r][tsplt][forcing_type][model]["rr_perc"].append(stats.percentileofscore(rr_model_100,this_event_model_rr))

    with open(event_stats_100_nearest_neibors_region_file, "wb") as fb:
        pkl.dump(event_100, fb)
       













event_stats_100_rr1_nearest_neibors_region_file = "event_stats_100_rr1_nearest_neibors_region.pkl"
if False:
    with open(event_stats_100_rr1_nearest_neibors_region_file, "rb") as fb:
        event_100_rr1 = pkl.load(fb)
else:
    event_100_rr1 = {r:{'time_split1':{'nldas':{},
                                'daymet':{}},
                 'time_split2':{'nldas':{},
                                'daymet':{}}} for r in huc_regions}

    for r in huc_regions:

        print(" \n")
        print(" \n")
        print(f"HUC REGION {r}")

        for tsplt in ['time_split2','time_split1']:
            print(tsplt)

            for forcing_type in forcing_products:
                print(forcing_type)

                spex = get_specifications(tsplt, forcing_type)

                for model in spex['models']:
                    print(model)

                    event_100_rr1[r][tsplt][forcing_type][model] = {"md_ks":[], "md_mi":[], "md_r2":[], 
                                                             "rr_ks":[], "rr_mi":[], "rr_r2":[]}

                    for basin_0str in spex["basin_list"]:


                        ########################################################
                        ########################################################
                        if get_basin_region(pd_attributes, basin_0str) != r:
                            continue



                        # Don't include any models if SAC-SMA is NaN
                        if np.isnan(events_results[r][tsplt][forcing_type][basin_0str][f'MD_sac']).sum() > 0:
                            continue
                        # Don't include any models if Observation is NaN
                        if np.isnan(events_results[r][tsplt][forcing_type][basin_0str][f'MD_obs']).sum() > 0:
                            continue

                        df = events_results[r][tsplt][forcing_type][basin_0str]

                        md_obs = df[f'MD_{model}']

                        md_model = df[f'MD_{model}']

                        for event in df.index.values:

                            this_event_obs_md = md_model[event]
                            this_event_model_md = md_model[event]

                            ###################################################
                            ###################################################
                            # Now lets consider events with runoff ratios of 1 or less!
                            if df.loc[event, f'RR_obs'] > 1:
                                continue

                            # Now find 100 closest events!

                            md_model_100 = md_model.iloc[(df[f'MD_{model}']-this_event_model_md).abs().argsort()[:100]]
                            model_indx = md_model_100.index.values
                            md_obs_100 = md_model.iloc[(df[f'MD_obs']-this_event_model_md).abs().argsort()[:100]]
                            obs_indx = md_model_100.index.values

                            rr_model_100 = df.loc[model_indx, f'RR_{model}']
                            rr_obs_100 = df.loc[obs_indx, f'RR_obs']

                            # For the R2 metric, there needs to be a 1 to 1 correspondence between events
                            # it is not 'simple' the distributions we are interested in,
                            #     but the ability to explain the variance of observation with the model.
                            rr_obs_100_mod = df.loc[model_indx, f'RR_obs']
                            md_obs_100_mod = df.loc[model_indx, f'MD_obs']

                            event_100_rr1[r][tsplt][forcing_type][model]["md_ks"].append(kstest(md_model_100, md_obs_100)[0])
                            event_100_rr1[r][tsplt][forcing_type][model]["md_mi"].append(calc_MI(md_model_100, md_obs_100, 100))

                            event_100_rr1[r][tsplt][forcing_type][model]["rr_ks"].append(kstest(rr_model_100, rr_obs_100)[0])
                            event_100_rr1[r][tsplt][forcing_type][model]["rr_mi"].append(calc_MI(rr_model_100, rr_obs_100, 100))

                            # NOTE: The R2 score needs to have the same events to make a 1 to 1 correspondence.
                            # NOTE: Could do both for the MI and KS as well...
                            event_100_rr1[r][tsplt][forcing_type][model]["md_r2"].append(r2_score(md_model_100, md_obs_100_mod))
                            event_100_rr1[r][tsplt][forcing_type][model]["rr_r2"].append(r2_score(rr_model_100, rr_obs_100_mod))

    with open(event_stats_100_rr1_nearest_neibors_region_file, "wb") as fb:
        pkl.dump(event_100_rr1, fb)




 
