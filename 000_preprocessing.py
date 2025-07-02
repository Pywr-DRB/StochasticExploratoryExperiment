#%%
import pandas as pd
import pywrdrb

## Load WRF-Hydro flow scenarios

pywrdrb_inflow_scenarios = [
    'wrfaorc_withObsScaled',
    'wrfaorc_calib_nlcd2016',
    'wrf1960s_calib_nlcd2016',
    ]

data = pywrdrb.Data(results_sets=['major_flow', 'inflow'])
data.load_hydrologic_model_flow(flowtypes=pywrdrb_inflow_scenarios)

#%% WRF-Hydro combined 1960s + AORC data
## Create a 'combined' WRF dataset with the 1959-1969 plus 1979-2021 data

# This data is from 1959-10-01 till 1969-12-31
wrf_1960s_inflow = data.inflow['wrf1960s_calib_nlcd2016'][0].copy()
wrf_1960s_gage_flow = data.major_flow['wrf1960s_calib_nlcd2016'][0].copy()

# Reindex the 1960s data to be redated as 1969-1979
wrf_1960s_inflow.index = wrf_1960s_inflow.index + pd.DateOffset(years=10)
wrf_1960s_gage_flow.index = wrf_1960s_gage_flow.index + pd.DateOffset(years=10)

# This data is from 1979-10-01 till 2021-12-31
wrf_aorc_inflow = data.inflow['wrfaorc_withObsScaled'][0].copy()
wrf_aorc_gage_flow = data.major_flow['wrfaorc_withObsScaled'][0].copy()

# Combine data used:
# - wrf_1960s_inflow: 1969-10-01 to 1979-09-30
# - wrf_aorc_inflow: 1979-10-01 to 2021-12-31
wrf_combined_inflow = pd.concat([wrf_1960s_inflow.loc['1969-10-01':'1979-09-30', :], 
                                 wrf_aorc_inflow.loc['1979-10-01':, :]])

wrf_combined_gage_flow = pd.concat([wrf_1960s_gage_flow.loc['1969-10-01':'1979-09-30', :],
                                    wrf_aorc_gage_flow.loc['1979-10-01':, :]])

# Save to CSV
