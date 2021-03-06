import numpy as np
import pandas as pd
import pickle as pkl
import xarray as xr
import copy
import os
import sys
import metrics
from sklearn.metrics import mutual_info_score
print("XArray version: ", xr.__version__)



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

range_for_analysis = {'time_split1': [1989,1999],'time_split2': [1996, 2014]}



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
#-------------------------------------------------------------------------------------------------
# Solve this problem. I think it is the xarray structures...
# isibleDeprecationWarning: Creating an ndarray from ragged nested sequences 
# (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. 
# If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
##########################################################################################################
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)




##########################################################################################################
# Most of the metricts we want to calculate are available in NeuralHydrology.
# But we also want to calculate the mutual information. So we have to add that.
loop_these_metrics = metrics.get_available_metrics()
loop_these_metrics.append("mi")
##########################################################################################################
def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi
##########################################################################################################
def calculate_all_metrics_for_frequency_analysis(analysis_dict, flows, recurrance_interval):

    sims = list(flows.keys())[:-1]

    for metric in loop_these_metrics:

        score = {sim:0 for sim in sims}
    
        analysis_dict[metric]['ri'].append(recurrance_interval)
    
        if metric == 'NSE':
            for sim in sims:
                score[sim] = metrics.nse(flows['obs'],flows[sim])
        if metric == 'MSE':
            for sim in sims:
                score[sim] = metrics.mse(flows['obs'],flows[sim])
        if metric == 'RMSE':
            for sim in sims:
                 score[sim] = metrics.rmse(flows['obs'],flows[sim])
        if metric == 'KGE':
            for sim in sims:
                score[sim] = metrics.kge(flows['obs'],flows[sim])
        if metric == 'Alpha-NSE':
            for sim in sims:
                score[sim] = metrics.alpha_nse(flows['obs'],flows[sim])
        if metric == 'Beta-NSE':
            for sim in sims:
                score[sim] = metrics.beta_nse(flows['obs'],flows[sim])
        if metric == 'Pearson-r':
            for sim in sims:
                score[sim] = metrics.pearsonr(flows['obs'],flows[sim])
        if metric == 'Peak-Timing':
            for sim in sims:
                score[sim] = np.abs(metrics.mean_peak_timing(flows['obs'],flows[sim]))
        if metric == 'FHV':
            for sim in sims:
                score[sim] = metrics.fdc_fhv(flows['obs'],flows[sim])
        if metric == 'FLV':
            for sim in sims:
                score[sim] = metrics.fdc_flv(flows['obs'],flows[sim])
        if metric == 'FMS':
            for sim in sims:
                score[sim] = metrics.fdc_fms(flows['obs'],flows[sim])
        if metric == "mi":
                score[sim] = calc_MI(flows['obs'],flows[sim], 100)
        for sim in sims:
            analysis_dict[metric][sim].append(score[sim])

    return





##########################################################################################################
# REVERT TO THESE AS THE FLOWS
##########################################################################################################
flows = ['lstm', 'mc', 'sac', 'obs']


##########################################################################################################
# DO THE MASS BALANCE ANALYSIS
##########################################################################################################
forcing_products = ['nldas','daymet']

file_name_map = {'nldas':'nldas', 'daymet':'cida'}
precip_column_map = {'nldas':'PRCP(mm/day)', 'daymet':'prcp(mm/day)'}

total_mass_error = {forcing_type:{time_split:{'absolute':{flow:[] for flow in flows}, 
              'positive':{flow:[] for flow in flows}, 
              'negative':{flow:[] for flow in flows}} for time_split in ['time_split1', 'time_split2']} for \
               forcing_type in forcing_products}
for err_type in ['absolute','positive', 'negative']:
    total_mass_error['nldas']['time_split2'][err_type]['nwm']=[]
        
cumulative_mass_all = {forcing_type:{time_split:{} for time_split in ['time_split1', 'time_split2']} for \
                       forcing_type in forcing_products}
total_mass = {forcing_type:{time_split:{} for time_split in ['time_split1', 'time_split2']} for \
                       forcing_type in forcing_products}
    
mass_basin_list={}
    
for tsplt in ['time_split1', 'time_split2']:
    print('tsplt', tsplt)
    for forcing_type in forcing_products:

        print('forcing_type ',forcing_type)

        mass_basin_list[tsplt] = []

        forcing_dir = '/home/NearingLab/data/camels_data/basin_dataset_public_v1p2'+\
            '/basin_mean_forcing/{}_all_basins_in_one_directory/'.format(forcing_type)

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
            print('flows',flows)
        else:
            start_date = pd.Timestamp('1989-10-01')
            end_date = pd.Timestamp('1999-09-30')
            labelz={'lstm':'LSTM', 'mc':'MC-LSTM','sac':'SAC-SMA', 'obs':'Observed'}
            models = ['lstm', 'mc', 'sac']
            flows = ['lstm', 'mc', 'sac', 'obs']
            print('flows',flows)
            basin_list = list(lstm_results_time_split1[forcing_type].keys())[:-1]

        first_basin = True

        for basin_0str in basin_list:
            basin_int = int(basin_0str)
#            print(basin_0str)

            #-------------------------------------------------------------------------------------------------
            # Reset the total mass to zero for this basin    
            cumulative_mass = {flow:[0] for flow in flows}
            cumulative_mass['precip'] = [0]
            total_mass[forcing_type][tsplt][basin_0str] = {flow:0 for flow in flows}
            imass=1
            #-------------------------------------------------------------------------------------------------


            #-------------------------------------------------------------------------------------------------
            # We need the basin area to convert to CFS, to interpolate the RI from LPIII
            basin_area = pd_attributes.loc[basin_int, 'area_geospa_fabric']
            basin_str = str(basin_int).zfill(8)
            #-------------------------------------------------------------------------------------------------

            #-------------------------------------------------------------------------------------------------
            # Make dictionary with all the flows
            flow_mm = {}    
            #-------------------------------------------------------------------------------------------------
            if tsplt == 'time_split2' and forcing_type == 'nldas':
                # Get the NWM data for this basin in an xarray dataset.
                xr_nwm = xr.DataArray(nwm_results[basin_0str]['streamflow'].values, 
                         coords=[nwm_results[basin_0str]['streamflow'].index], 
                         dims=['datetime'])
                # convert from CFS to mm/day
                # fm3/s * 3600 sec/hour * 24 hour/day / (m2 * mm/m)
                flow_mm['nwm'] = xr_nwm.loc[start_date:end_date]*3600*24/(basin_area*1000)
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
            # FORCING
            forcing = pd.read_csv(forcing_dir+basin_0str+'_lump_{}_forcing_leap.txt'.format(file_name_map[forcing_type]), 
                                  delim_whitespace=True, header=3)
            if tsplt == 'time_split1':
                forcing = forcing.iloc[3560:7214]
            if tsplt == 'time_split2':
                forcing = forcing.iloc[6118:]
            forcing.index=pd.to_datetime((forcing.Year*10000+forcing.Mnth*100+forcing.Day).apply(str),format='%Y%m%d')
            #-------------------------------------------------------------------------------------------------

            #-------------------------------------------------------------------------------------------------
            # Make sure we are in a time period that all the flow members have values
            # If there is missin observations than we can't compare the mass of the observed with simulaitons
            skip_basin_because_missing_obs = False
            if tsplt == 'time_split1':
                obs_temp = mclstm_results_time_split1[forcing_type][basin_0str]['1D']['xr']['QObs(mm/d)_obs'].datetime
            if tsplt == 'time_split2':
                obs_temp = mclstm_results_time_split2[forcing_type][basin_0str]['1D']['xr']['QObs(mm/d)_obs'].date
                
            for d in obs_temp:
                if d.values < start_date:
                    continue
                if d.values > end_date:
                    break
                if np.isnan(flow_mm['obs'].loc[d.values].values[0]):
                    skip_basin_because_missing_obs = True
                    break
                else:
                    #-------------------------------------------------------------------------------------------------
                    # Keep track of the cumulative mass and add it to the list
                    cumulative_mass['precip'].append(forcing[precip_column_map[forcing_type]].loc[d.values] + \
                                                     cumulative_mass['precip'][imass-1])

                    cumulative_mass['obs'].append(flow_mm['obs'].loc[d.values].values[0] + \
                                                  cumulative_mass['obs'][imass-1])

                    if tsplt == 'time_split2' and forcing_type == 'nldas':
                        cumulative_mass['nwm'].append(flow_mm['nwm'].loc[d.values].values + \
                                                      cumulative_mass['nwm'][imass-1])

                    cumulative_mass['lstm'].append(flow_mm['lstm'].loc[d.values].values[0] + \
                                                   cumulative_mass['lstm'][imass-1])

                    cumulative_mass['mc'].append(flow_mm['mc'].loc[d.values].values[0] + \
                                                 cumulative_mass['mc'][imass-1])

                    cumulative_mass['sac'].append(flow_mm['sac'].loc[d.values] + \
                                                  cumulative_mass['sac'][imass-1])
                    imass+=1
                    #-------------------------------------------------------------------------------------------------

            #-------------------------------------------------------------------------------------------------
            # If there is missin observations than we can't compare the mass of the observed with simulaitons            
            if skip_basin_because_missing_obs:
    #            print("skipping basin {} because of missing observations".format(basin_0str))
                continue
            else:
                mass_basin_list[tsplt].append(basin_0str)

            for flow in flows:
                total_mass[forcing_type][tsplt][basin_0str][flow] = np.nansum(flow_mm[flow].loc[start_date:end_date])

            for flow in flows:
                total_mass_error[forcing_type][tsplt]['absolute'][flow].append( \
                                        np.abs(total_mass[forcing_type][tsplt][basin_0str][flow] - \
                                        total_mass[forcing_type][tsplt][basin_0str]['obs'])/ \
                                        total_mass[forcing_type][tsplt][basin_0str]['obs'])
                if (total_mass[forcing_type][tsplt][basin_0str][flow] - total_mass[forcing_type][tsplt][basin_0str]['obs']) > 0:
                    total_mass_error[forcing_type][tsplt]['positive'][flow].append((\
                                        total_mass[forcing_type][tsplt][basin_0str][flow] - \
                                        total_mass[forcing_type][tsplt][basin_0str]['obs'])/ \
                                        total_mass[forcing_type][tsplt][basin_0str]['obs'])
                    total_mass_error[forcing_type][tsplt]['negative'][flow].append(0)
                else:
                    total_mass_error[forcing_type][tsplt]['negative'][flow].append(( \
                                        total_mass[forcing_type][tsplt][basin_0str][flow] - \
                                        total_mass[forcing_type][tsplt][basin_0str]['obs']) / \
                                        total_mass[forcing_type][tsplt][basin_0str]['obs'])
                    total_mass_error[forcing_type][tsplt]['positive'][flow].append(0)

            # _______________________________________________________________________
            # Keep track of all the cumulative mass through time for each basin
            if first_basin and not skip_basin_because_missing_obs:
                for flow in flows:
                    cumulative_mass_all[forcing_type][tsplt][flow] = np.array(cumulative_mass[flow])
                cumulative_mass_all[forcing_type][tsplt]['precip'] = np.array(cumulative_mass['precip'])
                first_basin = False
            if  not skip_basin_because_missing_obs and not first_basin:
                for flow in flows:
                    cumulative_mass_all[forcing_type][tsplt][flow] += np.array(cumulative_mass[flow])
                cumulative_mass_all[forcing_type][tsplt]['precip'] +=np.array(cumulative_mass['precip'])

# _______________________________________________________________________
# Save the mass balance results.
with open('total_mass_error_ens_slurm.pkl', 'wb') as fb:
    pkl.dump(total_mass_error, fb)
with open('total_mass_ens_slurm.pkl', 'wb') as fb:
    pkl.dump(total_mass, fb)
with open('cumulative_mass_all_ens_slurm.pkl', 'wb') as fb:
    pkl.dump(cumulative_mass_all, fb)












##########################################################################################################
##########################################################################################################






##########################################################################################################
##########################################################################################################






