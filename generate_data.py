
#%%
# IMPORT FUNCTIONS
import pandas as pd
from important_functions import collect_vcp_data

#%%
# Get Paci Data
vcp_data = []
ind = {'multipliers.i_cal_pca_multiplier':1, 'multipliers.i_ks_multiplier':1, 'multipliers.i_kr_multiplier':1, 'multipliers.i_na_multiplier':1, 'multipliers.i_to_multiplier':1, 'multipliers.i_k1_multiplier':1, 'multipliers.i_f_multiplier':1}
ik1_val = 0.5
vcp_data.append(collect_vcp_data(ind, ik1_val))
paci_vcp_data = pd.DataFrame(vcp_data)
paci_vcp_data['Type'] = ['Paci']+['Control']*(len(paci_vcp_data)-1)
paci_vcp_data.to_csv('./data/paci_vcp_data.csv.bz2')

# Get Kernik Data
vcp_data = []
ind = {'multipliers.i_cal_pca_multiplier':1, 'multipliers.i_ks_multiplier':1, 'multipliers.i_kr_multiplier':1, 'multipliers.i_na_multiplier':1, 'multipliers.i_to_multiplier':1, 'multipliers.i_k1_multiplier':1, 'multipliers.i_f_multiplier':1}
ik1_val = 0.5
leak_params = {'voltageclamp.gLeak':0.5, 'geom.Cm':50}
vcp_data.append(collect_vcp_data(ind, ik1_val, mod_name='Kernik',leak_params = leak_params))
kernik_vcp_data = pd.DataFrame(vcp_data)
kernik_vcp_data['Type'] = ['Kernik']+['Control']*(len(kernik_vcp_data)-1)
kernik_vcp_data.to_csv('./data/kernik_vcp_data.csv.bz2')

#%%
# Get Adjusted Model Data
best_data = pd.read_csv('./data/adjusted_model.csv')
vcp_data = []
ind = best_data.filter(like = 'multiplier').iloc[0].to_dict()
ik1_val = best_data['ik1_dc'][0]
vcp_data.append(collect_vcp_data(ind, ik1_val)) #drug_labels = ['Control']
best_drug_data = pd.DataFrame(vcp_data)
best_drug_data.to_csv('./data/best_drug_data.csv')
# %%
