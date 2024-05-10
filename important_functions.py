# IMPORT FUNCTIONS
import myokit
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.pyplot as plt 
import myokit.formats
from os import listdir
import matplotlib
import os
from multiprocessing import Pool



# DEFINE CLASSES 
class VCProtocol():
    def __init__(self, segments):
        self.segments = segments #list of VCSegments

    def get_protocol_length(self):
        proto_length = 0
        for s in self.segments:
            proto_length += s.duration
        
        return proto_length

    def get_myokit_protocol(self, scale=1):
        segment_dict = {'v0': f'{-82*scale}'}
        piecewise_txt = f'piecewise((engine.time >= 0 and engine.time < {99.9*scale}), v0, '
        current_time = 99.9*scale

        #piecewise_txt = 'piecewise( '
        #current_time = 0
        #segment_dict = {}

        for i, segment in enumerate(self.segments):
            start = current_time
            end = current_time + segment.duration
            curr_step = f'v{i+1}'
            time_window = f'(engine.time >= {start} and engine.time < {end})'
            piecewise_txt += f'{time_window}, {curr_step}, '

            if segment.end_voltage is None:
                segment_dict[curr_step] = f'{segment.start_voltage}'
            else:
                slope = ((segment.end_voltage - segment.start_voltage) /
                                                                segment.duration)
                intercept = segment.start_voltage - slope * start

                segment_dict[curr_step] = f'{slope} * engine.time + {intercept}'
            
            current_time = end
        
        piecewise_txt += 'vp)'

        return piecewise_txt, segment_dict, current_time
        
    def plot_protocol(self, is_shown=False):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        pts_v = []
        pts_t = []
        current_t = 0
        for seg in self.segments:
            pts_v.append(seg.start_voltage)
            if seg.end_voltage is None:
                pts_v.append(seg.start_voltage)
            else:
                pts_v.append(seg.end_voltage)
            pts_t.append(current_t)
            pts_t.append(current_t + seg.duration)

            current_t += seg.duration

        plt.plot(pts_t, pts_v)

        if is_shown:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('Time (ms)', fontsize=16)
            ax.set_xlabel('Voltage (mV)', fontsize=16)

            plt.show()

    def plot_with_curr(self, curr, cm=60):
        mod = myokit.load_model('mmt-files/kernik_2019_NaL_art.mmt')

        p = mod.get('engine.pace')
        p.set_binding(None)

        c_m = mod.get('artifact.c_m')
        c_m.set_rhs(cm)

        v_cmd = mod.get('artifact.v_cmd')
        v_cmd.set_rhs(0)
        v_cmd.set_binding('pace') # Bind to the pacing mechanism

        # Run for 20 s before running the VC protocol
        holding_proto = myokit.Protocol()
        holding_proto.add_step(-81, 30000)
        t = holding_proto.characteristic_time()
        sim = myokit.Simulation(mod, holding_proto)
        dat = sim.run(t)
        mod.set_state(sim.state())

        # Get protocol to run
        piecewise_function, segment_dict, t_max = self.get_myokit_protocol()
        mem = mod.get('artifact')

        for v_name, st in segment_dict.items():
            v_new = mem.add_variable(v_name)
            v_new.set_rhs(st)

        vp = mem.add_variable('vp')
        vp.set_rhs(0)

        v_cmd = mod.get('artifact.v_cmd')
        v_cmd.set_binding(None)
        vp.set_binding('pace')

        v_cmd.set_rhs(piecewise_function)
        times = np.arange(0, t_max, 0.1)
        ## CHANGE THIS FROM holding_proto TO SOMETHING ELSE
        sim = myokit.Simulation(mod, holding_proto)
        dat = sim.run(t_max, log_times=times)

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
        axs[0].plot(times, dat['membrane.V'])
        axs[0].plot(times, dat['artifact.v_cmd'], 'k--')
        axs[1].plot(times, np.array(dat['artifact.i_out']) / cm)
        axs[2].plot(times, dat[curr])

        axs[0].set_ylabel('Voltage (mV)')
        axs[1].set_ylabel('I_out (A/F)')
        axs[2].set_ylabel(curr)
        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        plt.show()

class VCSegment():
    def __init__(self, duration, start_voltage, end_voltage=None):
        self.duration = duration
        self.start_voltage = start_voltage
        self.end_voltage = end_voltage

# DEFINE FUNCTIONS
def get_ind(vals = [1,1,1,1,1,1,1,1,1,1], celltype = 'adult'):
    if celltype == 'ipsc':
        tunable_parameters=['i_cal_pca_multiplier','i_ks_multiplier','i_kr_multiplier','i_na_multiplier','i_to_multiplier','i_k1_multiplier','i_f_multiplier']
        ind = dict(zip(tunable_parameters, vals))
    else:
        tunable_parameters=['i_cal_pca_multiplier','i_ks_multiplier','i_kr_multiplier','i_nal_multiplier','i_na_multiplier','i_to_multiplier','i_k1_multiplier','i_NCX_multiplier','i_nak_multiplier','i_kb_multiplier']
        ind = dict(zip(tunable_parameters, vals))
    return(ind)

def run_model(mod_name, all_params = None, beats = 1, stim = 0, stim_1 = 0, start = 0.1, start_1 = 0, length = 1, length_1 = 0, cl = 1000, prepace = 600, I0 = 0, artifact = False): 
    mod, proto = get_ind_data(mod_name, all_params = all_params, artifact = artifact)  
    
    proto.schedule(stim, start, length, cl, 0) 
    if stim_1 != 0:
        proto.schedule(stim_1, start_1, length_1, cl, 1)
    sim = myokit.Simulation(mod,proto)

    if I0 != 0:
        sim.set_state(I0)

    sim.pre(cl * prepace) #pre-pace for 100 beats
    dat = sim.run(beats*cl) 
    IC = sim.state()

    return(dat, IC) 

def rrc_search(mod_name, IC, all_params = None, artifact = False, max_rrc = 10, stim = 0, start = 0.1, length = 5, cl = 1000):
    all_data = []

    mod, proto, v_label, t_label, c_label, ion_label, stim_label = get_ind_data(mod_name, all_params = all_params, artifact = artifact, get_labels = True)

    proto.schedule(stim, start, length, cl, 0)
    proto.schedule(max_rrc, (5*cl)+int(start+length+4), cl-int(start+length+4), cl, 1)
    sim = myokit.Simulation(mod, proto)
    sim.set_state(IC)
    dat = sim.run(7*cl)

    d0 = get_last_ap(dat, 4, cl=cl, v_label = v_label, t_label = t_label, c_label=c_label, ion_label=ion_label, stim_label=stim_label)
    result_abnormal0 = detect_abnormal_ap(d0['t'], d0['v']) 
    all_data.append({**{'t_rrc': d0['t'], 'v_rrc': d0['v'], 'stim': 0}, **result_abnormal0})

    d3 = get_last_ap(dat, 5, cl=cl, v_label = v_label, t_label = t_label, c_label=c_label, ion_label=ion_label, stim_label=stim_label)
    result_abnormal3 = detect_abnormal_ap(d3['t'], d3['v'])
    all_data.append({**{'t_rrc': d3['t'], 'v_rrc': d3['v'], 'stim': max_rrc}, **result_abnormal3})

    #if result_EAD0 == 1 or result_RF0 == 1:
    if result_abnormal0['result'] == 1:
        RRC = 0

    #elif result_EAD3 == 0 and result_RF3 == 0:
    elif result_abnormal3['result'] == 0:
        # no abnormality at 0.3 stim, return RRC
        RRC = max_rrc

    else:
        low = 0
        high = max_rrc
       
        while (high-low)>0.01:
            mid = low + (high-low)/2 

            mod, proto = get_ind_data(mod_name, all_params = all_params, artifact = artifact)  

            proto.schedule(stim, start, length, cl, 0)
            proto.schedule(mid, (5*cl)+int(start+length+4), cl-int(start+length+4), cl, 1)
            sim = myokit.Simulation(mod, proto)
            sim.set_state(IC) #added
            dat = sim.run(7*cl) #added

            data = get_last_ap(dat, 5, cl=cl, v_label = v_label, t_label = t_label, c_label=c_label, ion_label=ion_label, stim_label=stim_label)
            result_abnormal = detect_abnormal_ap(data['t'], data['v'])
            all_data.append({**{'t_rrc': data['t'], 'v_rrc': data['v'], 'stim': mid}, **result_abnormal})
            
            
            if result_abnormal['result'] == 0:
                # no RA so go from mid to high
                low = mid

            else:
                #repolarization failure so go from mid to low 
                high = mid
        
        for i in list(range(1, len(all_data))):
            if all_data[-i]['result'] == 0:
                RRC = all_data[-i]['stim']
                break
            else:
                RRC = 0 #in this case there would be no stim without an RA

    result = {'RRC':RRC, 'data':all_data}

    return(result)

def get_last_ap(dat, AP, cl = 1000, type = 'full', t_label = 'engine.time', v_label = 'membrane.V', c_label = 'intracellular_ions.cai', ion_label = 'membrane.i_ion', stim_label = 'stimulus.i_stim'):

    if type == 'full':
        start_ap = list(dat[t_label ]).index(closest(list(dat[t_label ]), AP*cl))
        end_ap = list(dat[t_label ]).index(closest(list(dat[t_label ]), (AP+1)*cl))

        t = np.array(dat[t_label][start_ap:end_ap])
        t = t-t[0]

        v = np.array(dat[v_label][start_ap:end_ap])
        cai = np.array(dat[c_label][start_ap:end_ap])
        i_ion = np.array(dat[ion_label][start_ap:end_ap])
        i_stim = np.array(dat[stim_label][start_ap:end_ap])

        data = {}
        data['t'] = t
        data['v'] = v
        data['cai'] = cai
        data['i_ion'] = i_ion
        data['i_stim'] = i_stim
    
    else:
        # Get t, v, and cai for second to last AP#######################
        ti, vol = dat

        start_ap = list(ti).index(closest(ti, AP*cl))
        end_ap = list(ti).index(closest(ti, (AP+1)*cl))

        t = np.array(ti[start_ap:end_ap])
        t = t-t[0]
        v = np.array(vol[start_ap:end_ap])

        data = {}
        data['t'] = t
        data['v'] = v

    return (data)

def count_aps(t, v):
    # Calcualte dv/dt max
    dvdt_max = np.max(np.diff(v)/np.diff(t))

    # Indexes where dvdt is above the value of the maximum dvdt-20 to allow for more values to be found. 
    #dvdt_indexes = np.where((np.diff(v)/np.diff(t))>(round(dvdt_max)-20))

    # Lower bound taken from literature
    dvdt_indexes = np.where((np.diff(v)/np.diff(t))>3)

    # Take the difference of each consecutive number 
    assess_consecutive = np.diff(dvdt_indexes)[0]

    # Consecutive numbers will have a value of 1 so values greater than 1 indicate the start of a new action potnetial
    break_points = np.where(assess_consecutive>50)

    # Formating the break_points to make it a list instead of an array
    break_points = break_points[0].tolist()

    # Formatting break_points again so that all values above 0 are increased by 1. This is completed to effectively separate the index groups.
    break_points = [i+1 for i in break_points]

    # Format dvdt_indexes into a list instead of an array
    dvdt_indexes = dvdt_indexes[0].tolist()

    break_points = [0]+break_points

    # Lets see how we did!
    # print('There should be', len(break_points)-1, 'action potentials.')

    info = {'break_points':break_points, 'dvdt_indexes':dvdt_indexes, 'aps':len(break_points)-1}

    return(info)

def isolate_ap(t, v, cai = None, ap = 1, cl = 1):
    info = count_aps(t, v)
    break_points = info['break_points']
    dvdt_indexes = info['dvdt_indexes']

    # IF there is no dvdtmax greater than 3 (usually abnormal ap) then return original t and v traces 
    if len(break_points)>1:
        # Groups of start indexes for each action potential
        dvdt_breakpoints = [dvdt_indexes[i] for i in break_points]

        if len(dvdt_breakpoints) <= 1:
            t_ap = t[dvdt_breakpoints[0]:len(t)]
            v_ap = v[dvdt_breakpoints[0]:len(v)]
        elif (ap+1) >= len(dvdt_breakpoints):
            t_ap = t[dvdt_breakpoints[ap]:len(t)]
            v_ap = v[dvdt_breakpoints[ap]:len(v)]
        else:
            t_ap = t[dvdt_breakpoints[ap]:dvdt_breakpoints[ap+cl]]
            v_ap = v[dvdt_breakpoints[ap]:dvdt_breakpoints[ap+cl]]

        if cai is not None:
            if ap > len(dvdt_breakpoints):
                cai_ap = cai[dvdt_breakpoints[ap]:len(cai)]
            else:
                cai_ap = cai[dvdt_breakpoints[ap]:dvdt_breakpoints[ap+cl]]
        
        if cai is None:
            return(t_ap, v_ap)
        else:
            return(t_ap, v_ap, cai_ap)
    else:
        if cai is None:
            return(t, v)
        else:
            return(t, v, cai)

def get_cl(t, v):
    # Calcualte dv/dt max
    dvdt_max = np.max(np.diff(v)/np.diff(t))

    # Indexes where dvdt is above 28. 28 was chosen because it is the value of the maximum dvdt-5 to allow for more values to be found. 
    dvdt_indexes = np.where((np.diff(v)/np.diff(t))>(round(dvdt_max)-5))

    # Take the difference of each consecutive number 
    assess_consecutive = np.diff(dvdt_indexes)[0]

    # Consecutive numbers will have a value of 1 so values greater than 1 indicate the start of a new action potnetial
    break_points = np.where(assess_consecutive!=1)

    # Formating the break_points to make it a list instead of an array
    break_points = break_points[0].tolist()

    # Formatting break_points again so that all values above 0 are increased by 1. This is completed to effectively separate the index groups.
    break_points = [i+1 for i in break_points]

    # Format dvdt_indexes into a list instead of an array
    dvdt_indexes = dvdt_indexes[0].tolist()

    break_points = [0]+break_points

    # Groups of start indexes for each action potential
    dvdt_breakpoints = [dvdt_indexes[i] for i in break_points]

    return((t[-1]-t[0])/len(dvdt_breakpoints))

def shift_ap(t, v):

    dvdt_max = np.max(np.diff(v)/np.diff(t))
    #dvdt_max_index = np.argmax(np.diff(v)/np.diff(t))
    dvdt_indexes = np.where(np.abs(((np.diff(v)/np.diff(t))-dvdt_max))<1)[0]
    dvdt_indexes = [dvdt_indexes[i] for i in list(np.where(np.diff(dvdt_indexes)>50)[0])]
    if dvdt_indexes == []:
        first_ap_idx = np.where(np.abs(((np.diff(v)/np.diff(t))-dvdt_max))<1)[0][0]
    else:
        first_ap_idx = dvdt_indexes[0]

    #new_t = np.array(t) - t[dvdt_max_index]
    new_t = [i-t[first_ap_idx] for i in t]

    return(new_t)

def check_physio(ap_features, normalize = True, feature_targets = {'RMP': [-74.21, 2.78], 'apa': [108.15, 6.97], 'dvdt_max': [113.58, 51.01], 'apd10':[36.78, 14.09], 'apd20':[58.7, 22.03], 'apd30':[75.5, 28.58], 'apd60':[107.6, 39.35], 'apd90':[140.68, 47.4], 'triangulation':[2.08, 1.37]}):

    error = 0
    for k, v in feature_targets.items():
        if normalize == True:
            normalized_bio = (ap_features[k]-v[0])/v[1] #(ind_biomarker-mean)/std
            normalized_mean = (v[0]-v[0])/v[1] #mean should be zero when normalized by standard deviation
            error+=(normalized_mean-normalized_bio)**2
        else:
            error+=(v[0]-ap_features[k])**2

    return(error)

def get_rrc_error(RRC):

    #################### RRC DETECTION & ERROR CALCULATION ##########################
    error = (10 - (np.abs(RRC)))  #weight = 20000
    #error = (1/np.abs(RRC))*10000 
    return error

def get_features(t,v,cai=None):

    ap_features = {}

    # Returns really large error value if cell AP is not valid 
    if ((min(v) > -52) or (max(v) < 0)):
        return 50000000 

    # Voltage/APD features#######################
    mdp = min(v)
    ap_features['mdp'] = mdp #added in
    max_p = max(v)
    apa = max_p - mdp
    dvdt_max = np.max(np.diff(v[0:100])/np.diff(t[0:100]))

    ap_features['Vm_peak'] = max_p
    ap_features['apa'] = apa
    ap_features['dvdt_max'] = dvdt_max

    for apd_pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        apd_val = calc_APD(t,v,apd_pct) 
        ap_features[f'apd{apd_pct}'] = apd_val
 
    if (ap_features['apd70'] - ap_features['apd80']) == 0:
        ap_features['triangulation'] = 0
    else:
        ap_features['triangulation'] = (ap_features['apd30'] - ap_features['apd40'])/(ap_features['apd70'] - ap_features['apd80'])
    
    ap_features['RMP'] = mdp #alex used this as well - found it in the get_paced_stats function in the figs_cell_objects script of the vc-optimization-cardiotoxicity repo
    #ap_features['RMP'] = np.mean([mdp, v[len(v)-1]]) #average between the mdp and the last point in v
    #ap_features['RMP'] = np.mean(v[len(v)-50:len(v)])

    if cai is not None: 
        # Calcium/CaT features######################## 
        max_cai = np.max(cai)
        min_cai = np.min(cai)
        max_cai_idx = np.argmax(cai)
        max_cai_time = t[max_cai_idx]
        cat_amp = np.max(cai) - np.min(cai)
        ap_features['cat_amp'] = cat_amp * 1e6 #make into nanomolar* 1e5 #added in multiplier since number is so small
        ap_features['cat_peak_time'] = max_cai_time
        ap_features['diastolic_cai'] = min_cai

        """
        for cat_pct in [10, 50, 90]:
            cat_recov = max_cai - (cat_amp * (cat_pct / 100))
            idx_catd = np.argmin(np.abs(cai[max_cai_idx:] - cat_recov)) #find the index value closest to cat_recov
            idx_catr = np.argmin(np.abs(cai[0:max_cai_idx] - cat_recov))
            catd_val = t[max_cai_idx+idx_catd] #decay time point
            catr_val = t[idx_catr] #rise time point

            ap_features[f'catd{cat_pct}'] = catd_val 
            ap_features[f'catr{cat_pct}'] = catr_val 

        ap_features['RT1050'] = ap_features['catr10']-ap_features['catr50'] 
        ap_features['RT1090'] = ap_features['catr10']-ap_features['catr90'] 
        ap_features['RT9010'] = ap_features['catd90']-ap_features['catd10'] 
        """

    return ap_features

def morph_error(base_dat, ind_dat, trace):
    points = list(range(0, 900, 1))
    trace_base = np.array([base_dat[trace][np.argmin(np.abs(points[i]-base_dat['t']))] for i in points])
    trace_base_norm = (trace_base-min(trace_base))/(max(trace_base)-min(trace_base))

    trace_ind = np.array([ind_dat[trace][np.argmin(np.abs(points[i]-ind_dat['t']))] for i in points])
    trace_ind_norm = (trace_ind-min(trace_base))/(max(trace_base)-min(trace_base))
    
    error = sum((trace_base_norm - trace_ind_norm)**2)
    return(error)

def mature_ipsc(mod_name, all_params = None, max_ik1_val=4, beats=5, l=5, s=2, start = 0, spon_prepace = 600, mature_prepace = 600):
    ik1_vals = list(np.arange(0.5, max_ik1_val+0.5, 0.5))
    if all_params is not None:
        all_params = {**all_params, **{'parameters.ik1_ishi_dc_scale':0.0}}
    else:
        all_params = {'parameters.ik1_ishi_dc_scale':0.0}

    for i in ik1_vals: 
        all_params['parameters.ik1_ishi_dc_scale'] = i
        dat_test, IC_test = run_model(mod_name, all_params = all_params, beats = 5, prepace= spon_prepace, stim = 0, start = start, length =l) #ipsc immature
        if max(dat_test['membrane.V']) < 0:
            dat_mature, IC_mature = run_model(mod_name, all_params = all_params, beats = beats, prepace = mature_prepace, stim = s, start = start, length = l) #ipsc mature
            if min(dat_mature['membrane.V']) < -65:
                return(dat_mature, i, IC_mature)

    return(0, 0, 0)

def calc_APD(t, v, apd_pct, ipsc = False):
    t = np.array(t)
    v = np.array(v)
    t = [i-t[0] for i in t]

    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    min_p_idx = np.argmin(v)
    apa = max_p - mdp
    repol_pot = max_p - apa * apd_pct/100
    if ipsc == True:
        idx_apd = np.argmin(np.abs(v[max_p_idx:min_p_idx] - repol_pot))
    else:
        idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
    apd_val = t[idx_apd+max_p_idx]

    return(apd_val) 

def get_ind_data(mod_name, all_params = None, artifact = False, get_labels = False):
    #mod, proto, x = myokit.load(path+model)
    #if ind is not None:
    #    for k, v in ind[0].items():
    #        mod['multipliers'][k].set_rhs(v)

    if mod_name == 'Kernik':
        v_label = 'membrane.V'
        t_label = 'engine.time'
        c_label = 'cai.Cai'
        ion_label = 'membrane.i_ion'
        stim_label = 'stimulus.i_stim'

        if artifact:
            model_path = './models/kernik_artifact_fixed.mmt'
        else:
            model_path = './models/kernik_leak_fixed.mmt'
        
    if mod_name == 'Paci':
        v_label = 'membrane.V'
        t_label = 'engine.time'
        c_label = 'calcium.Cai'
        ion_label = 'membrane.i_ion'
        stim_label = 'stimulus.i_stim'
    
        if artifact:
            model_path = './models/paci_artifact_ms_fixed.mmt'
        else:
            model_path = './models/paci_leak_ms_fixed.mmt'
    
    mod, proto, x = myokit.load(model_path)
    
    if all_params is not None:
        for k, scale in all_params.items():
            group, name = k.split('.')
            mod[group][name].set_rhs(scale)
    
    if get_labels:
        return mod, proto, v_label, t_label, c_label, ion_label, stim_label
    else:
        return mod, proto

def detect_abnormal_ap(t, v, rmp_bound = -52):
    # First cut off upstroke to assess repolarization phase
    v_max = np.argmax(v)
    v_0 = v[v_max:] 
    t_0 = t[v_max:] 

    rises_groups = []
    
    # Reported from Alex's in vitro data
    rmp_mean = -74.21
    rmp_std = 2.78


    if min(v_0) > rmp_mean+(2*rmp_std):
        #Assess repolarization Failure
        rises_groups.extend(list(range(v_max, len(v))))
    else:
        all_dvdt = np.diff(v_0)/np.diff(t_0)
        for i in range(0, len(all_dvdt)):
            m = all_dvdt[i]
            if m > 0 and v_0[i]>rmp_bound and v_0[i]<20:
                # if slope is greater than 0 and the voltage is over the bound, its an EAD
                rises_groups.append(v_max+i)
            #if v_0[i]<rmp_bound and m > 0.2:
                # if slope is greater than 3 (dvdt_max lower bound) and voltage is lower than the bound, its a phase 4 RF
            #    rises_groups.append(v_max+i)

    if rises_groups == []:
        info = "normal AP" 
        result = 0
    else:
        info = "abnormal AP"
        result = 1

    data = {'info': info, 'result':result, 'abnormal_idx':rises_groups}
    return(data)

def closest(lst, K): 
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def add_scalebar(axs, section, y_pos = -0.1):
    # FORMAT X AXIS
    if section == 0:
        xmin, xmax, ymin, ymax = axs.axis()
        scalebar = AnchoredSizeBar(axs.transData, 100, '100 ms', 'lower left', bbox_to_anchor = (0,y_pos), bbox_transform =axs.transAxes, pad=0.5, color='black', frameon=False, size_vertical=(ymax-ymin)*0.0001) #fontproperties=fontprops
        axs.add_artist(scalebar)
        axs.spines[['bottom']].set_visible(False)
        axs.tick_params(bottom=False)
        axs.tick_params(labelbottom=False)
    else:
        for i in list(range(0, len(section))):
            xmin, xmax, ymin, ymax = axs[section[i][0], section[i][1]].axis()
            scalebar = AnchoredSizeBar(axs[section[i][0], section[i][1]].transData, 100, '100 ms', 'lower left', bbox_to_anchor = (0,y_pos), bbox_transform =axs[section[i][0], section[i][1]].transAxes, pad=0.5, color='black', frameon=False, size_vertical=(ymax-ymin)*0.0001) #fontproperties=fontprops
            axs[section[i][0], section[i][1]].add_artist(scalebar)
            axs[section[i][0], section[i][1]].spines[['bottom']].set_visible(False)
            axs[section[i][0], section[i][1]].tick_params(bottom=False)
            axs[section[i][0], section[i][1]].tick_params(labelbottom=False)

def generate_alldata(get_raw_data, save_data_to = './', compressed = True, num_best_models = 4, trials = ['trial1', 'trial2', 'trial4', 'trial5', 'trial6', 'trial8', 'trial9', 'trial10']):

        if compressed:
            file_end = '_info.csv.bz2'
        else:
            file_end = '_info.csv'

        # First, combine all the raw data for each individual trial into one dataframe
        all_trials = []

        for t in list(range(0, len(trials))):
            try:
                print(trials[t])
                data = pd.read_csv(get_raw_data+trials[t]+file_end)
                data['trial'] = t
                all_trials.append(data)
            except:
                print('couldnt complete for: trial', t)
        
        all_trials = pd.concat(all_trials)

        all_trials = all_trials.drop_duplicates(subset=all_trials.filter(like='multiplier').columns.to_list()) #drop dublicates so all best individuals are unique 
        all_trials = all_trials[all_trials['t']!='50000000'] #filter out bad inds
        all_trials.to_csv(save_data_to+'all_data.csv', index=False)

        # Get extra data for best indiviuals
        best_data = all_trials.sort_values(by='fitness', ascending = True).reset_index().iloc[0:num_best_models]
        vcp_data = []
        for i in range(0, len(best_data)):
            ind = best_data.filter(like = 'multiplier').iloc[i].to_dict()
            ik1_val = best_data['ik1_dc'][i]
            try:
                vcp_data.append(collect_vcp_data(ind, ik1_val)) #drug_labels = ['Control']
            except:
                pass

        best_drug_data = pd.DataFrame(vcp_data)
        best_drug_data.to_csv(save_data_to+'best_drug_data.csv')

def cellml_to_mmt(model_cellml, model_mmt):
    i = myokit.formats.importer('cellml')
    mod=i.model(model_cellml) 
    myokit.save(model_mmt, mod)

def get_vcp_error(t_ind, i_out_ind, data_path = './data/all_cells.csv.bz2', normalize = True):
    t_ind = np.array(t_ind)
    i_out_ind = np.array(i_out_ind)
    
    all_cells = pd.read_csv(data_path)
    bounds = pd.read_csv('./data/bounds.csv')
    upper_bound = bounds['upper bound'].tolist()
    lower_bound = bounds['lower bound'].tolist()
    

    # Mean current trace
    t_mean_vcp = (np.array(all_cells[all_cells['cell']==0]['Time (s)'])*1000)
    #i_out_mean = [np.mean([upper_bound[i], lower_bound[i]]) for i in range(0, len(t_mean_vcp))]
    i_out_mean = bounds['mean trace'].tolist()
    std_trace = bounds['std_trace'].tolist()

    conds = ['I_Kr', 'I_CaL', 'I_Na', 'I_To', 'I_K1', 'I_F', 'I_Ks']
    max_p_curr = pd.read_csv('./data/max_paci_curs.csv')

    kristins_times = [{'Current':'I_Kr', 'Time Start':850, 'Time End':870}, 
                  {'Current':'I_CaL', 'Time Start':2450, 'Time End':2470},
                  {'Current':'I_Na', 'Time Start':2350, 'Time End':2370},
                  {'Current':'I_To', 'Time Start':3220, 'Time End':3240},
                  {'Current':'I_K1', 'Time Start':3880, 'Time End':3900},
                  {'Current':'I_F', 'Time Start':5515, 'Time End':5535},
                  {'Current':'I_Ks', 'Time Start':8640, 'Time End':8660}]

    max_p_curr_kristin = pd.DataFrame(kristins_times)

    error = 0

    for i in range(0, len(conds)):
        #start_time = ((max_p_curr[max_p_curr['Current']==conds[i]].reset_index()['Time Start'][0]*1000)-400)-30
        #end_time = ((max_p_curr[max_p_curr['Current']==conds[i]].reset_index()['Time End'][0]*1000)-400)+30
        start_time = max_p_curr_kristin[max_p_curr_kristin['Current']==conds[i]].reset_index()['Time Start'][0]
        end_time = max_p_curr_kristin[max_p_curr_kristin['Current']==conds[i]].reset_index()['Time End'][0]

        start_idx = np.argmin(np.abs(t_ind - start_time))
        end_idx = np.argmin(np.abs(t_ind - end_time))

        start_idx_mean = np.argmin(np.abs(t_mean_vcp - start_time))
        end_idx_mean = np.argmin(np.abs(t_mean_vcp - end_time))

        t_ind_cond = t_ind[start_idx:end_idx]
        iout_ind_cond = i_out_ind[start_idx:end_idx]

        t_mean_cond = np.array(t_mean_vcp[start_idx_mean:end_idx_mean])
        iout_mean_cond = np.array(i_out_mean[start_idx_mean:end_idx_mean])

        t_std_cond = np.array(t_mean_vcp[start_idx_mean:end_idx_mean])
        iout_std_cond = np.array(std_trace[start_idx_mean:end_idx_mean])

        if normalize == True:
            i_out_ind_cond_norm = (iout_ind_cond)/np.mean(iout_std_cond)
            iout_mean_cond_norm = (iout_mean_cond)/np.mean(iout_std_cond)
            error += sum((i_out_ind_cond_norm-iout_mean_cond_norm)**2)
        else:
            error += sum((iout_ind_cond-iout_mean_cond)**2)

        # VISUALIZE TRACES
        #plt.plot(t_ind_cond, iout_ind_cond, label = 'Ind')
        #plt.plot(t_mean_cond, iout_mean_cond, label = 'Mean')
        #plt.title(conds[i] + ' - error = '+str(sum((iout_ind_cond-iout_mean_cond)**2)))
        #plt.ylim([-30, 30])
        #plt.legend()
        #plt.show()

    return(error)

def return_vc_proto(scale=1, prestep_size=0):
    segments = [
            VCSegment(756.9, 6),
            VCSegment(7.3, -41),
            VCSegment(100.6, 8.5),
            VCSegment(500, -80),
            VCSegment(106.2, -81),
            VCSegment(103.7, -2, -34),
            VCSegment(500, -80),
            VCSegment(183, -87),
            VCSegment(101.9, -52, 14),
            VCSegment(500, -80),
            VCSegment(272.1, 54, -107),
            VCSegment(102.8, 60),
            VCSegment(500, -80),
            VCSegment(52.2, -76, -80),
            VCSegment(102.7, -120),
            VCSegment(500, -80),
            VCSegment(936.9, -120),
            VCSegment(94.9, -77),
            VCSegment(8, -118),
            VCSegment(500, -80),
            VCSegment(729.4, 55),
            VCSegment(996.6, 48),
            VCSegment(894.9, 59, 28),
            VCSegment(900, -80)
            ]

    if prestep_size != 0:
        segments = [VCSegment(prestep_size, -80)] + segments

    new_segments = []
    for seg in segments:
        if seg.end_voltage is None:
            new_segments.append(VCSegment(seg.duration*scale, seg.start_voltage*scale))
        else:
            new_segments.append(VCSegment(seg.duration*scale,
                                          seg.start_voltage*scale,
                                          seg.end_voltage*scale))

    return VCProtocol(new_segments)

def collect_vcp_data(ind, ik1_val, leak_params = {'voltageclamp.gLeak':0.5, 'cell.Cm':50}, artifact_params = {'voltageclamp.cm_est':50, 'voltageclamp.rseries':0.02, 'voltageclamp.rseries_est':0.02}, mod_name = 'Paci', stim = 6, length = 5, drug_labels = ['Cisapride', 'Verapamil', 'Quinidine', 'Quinine', 'Control']):
    biomarkers = ['dvdt_max','RMP', 'apa', 'apd10', 'apd20', 'apd30', 'apd60', 'apd90', 'triangulation']
    
    drugs = {'Cisapride':{'multipliers.i_kr_multiplier':(1-0.95), 'multipliers.i_cal_pca_multiplier':(1-0.01), 'multipliers.i_na_multiplier':(1-0.02), 'multipliers.i_to_multiplier':(1-0.13), 'multipliers.i_k1_multiplier':(1-0.05),'multipliers.i_ks_multiplier': (1-0.02)},
         'Verapamil':{'multipliers.i_kr_multiplier':(1-0.21), 'multipliers.i_cal_pca_multiplier':(1-0.39), 'multipliers.i_na_multiplier':(1-0.009), 'multipliers.i_to_multiplier':(1-0.01), 'multipliers.i_k1_multiplier':(1-0.03), 'multipliers.i_ks_multiplier': (1-0.03)},
         'Quinidine':{'multipliers.i_kr_multiplier':(1-0.89), 'multipliers.i_cal_pca_multiplier':(1-0.16), 'multipliers.i_na_multiplier':(1-0.1), 'multipliers.i_to_multiplier':(1-0.43), 'multipliers.i_k1_multiplier':(1-0.01), 'multipliers.i_ks_multiplier': (1-0.27)},
         'Quinine':{'multipliers.i_kr_multiplier':(1-0.72), 'multipliers.i_cal_pca_multiplier':(1-0.29), 'multipliers.i_na_multiplier':(1-0.28), 'multipliers.i_to_multiplier':(1-0.15), 'multipliers.i_k1_multiplier':(1-0.009), 'multipliers.i_f_multiplier': (1-0.32), 'multipliers.i_ks_multiplier': (1-0.2)}
         }
    
    all_data = {}
    for d in drug_labels:
        #print(d)
        if d == 'Control':
            ind_cp = ind
        else:
            ind_cp = ind.copy()
            #print(ind_cp)
            for k, v in drugs[d].items():
                ind_cp[k] = ind[k]*drugs[d][k]
                #print(str(k) +' times drug = ' + str(ind[k]) + '*' + str(drugs[d][k]))
                

        #dat, IC = run_model(ind = [ind_cp], beats = 1, dc = [{'ik1_ishi_dc_scale': ik1_val}], rseal = 0.2, stim = 1, length = 5, path = './models/', model = 'paci-2013-ventricular-leak-fixed.mmt')
        dat, IC = run_model(mod_name, all_params = {**ind_cp, **leak_params, **{'parameters.ik1_ishi_dc_scale': ik1_val}}, stim = stim, length = length)

        if mod_name == 'Kernik':
            cal_name = 'cai.Cai'
        else:
            cal_name = 'calcium.Cai'

        ind_biomarkers = get_features(dat['engine.time'], dat['membrane.V'], dat[cal_name])
        if ind_biomarkers == 50000000:
            ind_biomarkers = dict(zip(biomarkers, [50000000]*len(biomarkers)))
        else:
            ind_biomarkers = dict(zip(biomarkers, [ind_biomarkers[k] for k in biomarkers]))
        #t_vcp, v_vcp, i_out = run_vc_protocol(ind = [ind_cp], dc = [{'ik1_ishi_dc_scale':ik1_val}], rseal = 0.2, model = 'paci-2013-ventricular-leak-fixed.mmt')
        t_vcp, i_out, v_vcp = get_vc_artifact_response(mod_name, {**ind_cp, **leak_params, **artifact_params})
        e_vcp = get_vcp_error(t_vcp, i_out)

        all_data['t_'+d] = str(list(dat['engine.time']))
        all_data['v_'+d] = str(list(dat['membrane.V']))
        all_data['t_out_'+d] = str(list(t_vcp))
        all_data['i_out_'+d]= str(list(i_out))
        all_data['e_vcp'+d]= e_vcp

        for b in biomarkers:
            all_data[b+'_'+d] = ind_biomarkers[b]

    return(all_data)

def collect_drug_data(mod_name, all_params, ind, ik1_val, stim, length, beats = 1, drug_labels = ['Cisapride', 'Verapamil', 'Quinidine', 'Quinine', 'Control']):
    biomarkers = ['dvdt_max','RMP', 'apa', 'apd10', 'apd20', 'apd30', 'apd60', 'apd90', 'triangulation']
    
    drugs = {'Cisapride':{'multipliers.i_kr_multiplier':(1-0.95), 'multipliers.i_cal_pca_multiplier':(1-0.01), 'multipliers.i_na_multiplier':(1-0.02), 'multipliers.i_to_multiplier':(1-0.13), 'multipliers.i_k1_multiplier':(1-0.05),'multipliers.i_ks_multiplier': (1-0.02)},
         'Verapamil':{'multipliers.i_kr_multiplier':(1-0.21), 'multipliers.i_cal_pca_multiplier':(1-0.39), 'multipliers.i_na_multiplier':(1-0.009), 'multipliers.i_to_multiplier':(1-0.01), 'multipliers.i_k1_multiplier':(1-0.03), 'multipliers.i_ks_multiplier': (1-0.03)},
         'Quinidine':{'multipliers.i_kr_multiplier':(1-0.89), 'multipliers.i_cal_pca_multiplier':(1-0.16), 'multipliers.i_na_multiplier':(1-0.1), 'multipliers.i_to_multiplier':(1-0.43), 'multipliers.i_k1_multiplier':(1-0.01), 'multipliers.i_ks_multiplier': (1-0.27)},
         'Quinine':{'multipliers.i_kr_multiplier':(1-0.72), 'multipliers.i_cal_pca_multiplier':(1-0.29), 'multipliers.i_na_multiplier':(1-0.28), 'multipliers.i_to_multiplier':(1-0.15), 'multipliers.i_k1_multiplier':(1-0.009), 'multipliers.i_f_multiplier': (1-0.32), 'multipliers.i_ks_multiplier': (1-0.2)}
         }
    
    all_data = {}
    for d in drug_labels:
        #print(d)
        if d == 'Control':
            ind_cp = ind[0]
        else:
            ind_cp = ind[0].copy()
            #print(ind_cp)
            #print(ind)
            for k, v in drugs[d].items():
                ind_cp[k] = ind[0][k]*drugs[d][k]
                #print(str(k) +' times drug = ' + str(ind[0][k])+ '*' + str(drugs[d][k]))

        dat, IC = run_model(mod_name, all_params = {**ind_cp, **all_params, **{'parameters.ik1_ishi_dc_scale': ik1_val}}, beats = beats, stim = stim, length = length)

        all_data['t_'+d] = dat['engine.time']
        all_data['v_'+d] = dat['membrane.V']

    return(all_data)

def run_vc_protocol(ind = None, rseal = 0.5, dc = None, path = './models/', model='paci_artifact_ms_fixed.mmt', with_all_dat=False):
    # based on Alex's get_vc_artifact_response function from the RICP paper
    MOD_Cm = 45 #pF
    MOD_rseries = (1/rseal)/100 #.02 #Gohms
    MOD_gLeak = rseal #.5 which equals 2GOhm seal (since 1/0.5=2) 
    
    if model == 'kernik_artifact_fixed.mmt' or model == 'kernik_leak_fixed.mmt':
        #model_path = './mmt/kernik_artifact_fixed.mmt'
        mod = myokit.load_model(path+model)
        mod['geom']['Cm'].set_rhs(MOD_Cm)
    if model == 'paci_artifact_ms_fixed.mmt' or model == 'paci-2013-ventricular-leak-fixed.mmt':
        #model_path = './mmt/paci_artifact_ms_fixed.mmt'
        mod = myokit.load_model(path+model)
        mod['cell']['Cm'].set_rhs(MOD_Cm) 
    
    if model == 'kernik_artifact_fixed.mmt' or model == 'paci_artifact_ms_fixed.mmt':
        mod['voltageclamp']['cm_est'].set_rhs(MOD_Cm)
        mod['voltageclamp']['rseries'].set_rhs(MOD_rseries)
        mod['voltageclamp']['rseries_est'].set_rhs(MOD_rseries)
        mod['voltageclamp']['gLeak'].set_rhs(MOD_gLeak)
    else:
        mod['membrane']['gLeak'].set_rhs(MOD_gLeak)

    #for k, scale in all_params.items():
    #    group, name = k.split('.')
    #    model_value = mod[group][name].value()
    #    mod[group][name].set_rhs(model_value * scale)

    if dc is not None:
        for k, v in dc[0].items():
            mod['parameters'][k].set_rhs(v)
    
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    if model == 'kernik_artifact_fixed.mmt' or model == 'paci_artifact_ms_fixed.mmt':
        p = mod.get('engine.pace')
        p.set_binding(None)

        prestep = 50000
        vc_proto = return_vc_proto(prestep_size=prestep)

        proto = myokit.Protocol()
        proto.add_step(-80, prestep+10000)

        piecewise, segment_dict, t_max = vc_proto.get_myokit_protocol()

        ####
        p = mod.get('engine.pace')
        p.set_binding(None)

        new_seg_dict = {}
        for k, vol in segment_dict.items():
            new_seg_dict[k] = vol

        segment_dict = new_seg_dict

        mem = mod.get('voltageclamp')


        for v_name, st in segment_dict.items():
            v_new = mem.add_variable(v_name)
            v_new.set_rhs(st)

        vp = mem.add_variable('vp')
        vp.set_rhs(0)

        v_cmd = mod.get('voltageclamp.Vc')

        v_cmd.set_binding(None)
        vp.set_binding('pace')

        v_cmd.set_rhs(piecewise)

        sim = myokit.Simulation(mod, proto)

        times = np.arange(0, t_max, 0.1)

        sim.set_max_step_size(1)

        dat = sim.run(t_max, log_times=times)

        cm = mod['voltageclamp']['cm_est'].value()
        t = dat.time()
        i_out = [v / cm for v in dat['voltageclamp.Iout']]
        v = dat['voltageclamp.Vc']

        if with_all_dat:
            return times, i_out, v, dat
        else:
            return times, i_out, v
    else:
        prestep = 50000
        vc_proto = return_vc_proto(prestep_size=prestep)

        proto = myokit.Protocol()
        proto.add_step(-80, prestep+10000)

        piecewise, segment_dict, t_max = vc_proto.get_myokit_protocol()

        new_seg_dict = {}
        for k, vol in segment_dict.items():
            new_seg_dict[k] = vol

        segment_dict = new_seg_dict

        p = mod.get('engine.pace')
        p.set_binding(None)
        
        v = mod.get('membrane.V')
        v.demote()
        v.set_rhs(0)
        v.set_binding('pace') # Bind to the pacing mechanism

        mem = mod.get('membrane')
        v = mem.get('V')
        v.set_binding(None)

        for v_name, st in segment_dict.items():
            v_new = mem.add_variable(v_name)
            v_new.set_rhs(st)

        vp = mem.add_variable('vp')
        vp.set_rhs(0)
        vp.set_binding('pace')

        v.set_rhs(piecewise)

        sim = myokit.Simulation(mod, proto)
        sim.set_max_step_size(1)
        times = np.arange(0, t_max, 0.1)

        dat = sim.run(t_max, log_times=times)

        t = dat.time()
        i_out = dat['membrane.i_ion']
        v = dat['membrane.V']

        if with_all_dat:
            return times, i_out, v, dat
        else:
            return times, i_out, v

def get_vc_artifact_response(mod_name, all_params, with_all_dat=False):
    if mod_name == 'Kernik':
        model_path = './models/kernik_artifact_fixed.mmt'
        mod = myokit.load_model(model_path)
    if mod_name == 'Paci':
        model_path = './models/paci_artifact_ms_fixed.mmt'
        mod = myokit.load_model(model_path)

    for k, value in all_params.items():
        group, name = k.split('.')
        mod[group][name].set_rhs(value)

    p = mod.get('engine.pace')
    p.set_binding(None)

    prestep = 50000
    vc_proto = return_vc_proto(prestep_size=prestep)

    proto = myokit.Protocol()
    proto.add_step(-80, prestep+10000)

    piecewise, segment_dict, t_max = vc_proto.get_myokit_protocol()

    ####
    p = mod.get('engine.pace')
    p.set_binding(None)

    new_seg_dict = {}
    for k, vol in segment_dict.items():
        new_seg_dict[k] = vol

    segment_dict = new_seg_dict

    mem = mod.get('voltageclamp')

    for v_name, st in segment_dict.items():
        v_new = mem.add_variable(v_name)
        v_new.set_rhs(st)

    vp = mem.add_variable('vp')
    vp.set_rhs(0)

    v_cmd = mod.get('voltageclamp.Vc')
    v_cmd.set_binding(None)
    vp.set_binding('pace')

    v_cmd.set_rhs(piecewise)

    sim = myokit.Simulation(mod, proto)

    times = np.arange(0, t_max, 0.1)

    sim.set_max_step_size(1)

    dat = sim.run(t_max, log_times=times)

    cm = mod['voltageclamp']['cm_est'].value()
    t = dat.time()
    i_out = [v / cm for v in dat['voltageclamp.Iout']]
    v = dat['voltageclamp.Vc']

    times, i_out, v = change_vcp_start(times, i_out, v) #added by kristin

    if with_all_dat:
        return times, i_out, v, dat
    else:
        return times, i_out, v

def get_vc_leak_response(mod_name, all_params, with_all_dat=False, MOD_Cm = 45, MOD_rseries = 0.02, MOD_gLeak = 0.5):
    if mod_name == 'Kernik':
        model_path = './models/kernik_leak_fixed.mmt'
        mod = myokit.load_model(model_path)
        mod['geom']['Cm'].set_rhs(MOD_Cm)
    if mod_name == 'Paci':
        model_path = './models/paci_leak_ms_fixed.mmt'
        mod = myokit.load_model(model_path)
        mod['cell']['Cm'].set_rhs(MOD_Cm) 
    
    mod['voltageclamp']['gLeak'].set_rhs(MOD_gLeak)

    for k, scale in all_params.items():
        group, name = k.split('.')
        model_value = mod[group][name].value()
        mod[group][name].set_rhs(model_value * scale)

    prestep = 50000
    vc_proto = return_vc_proto(prestep_size=prestep)

    proto = myokit.Protocol()
    proto.add_step(-80, prestep+10000)

    piecewise, segment_dict, t_max = vc_proto.get_myokit_protocol()

    new_seg_dict = {}
    for k, vol in segment_dict.items():
        new_seg_dict[k] = vol

    segment_dict = new_seg_dict

    p = mod.get('engine.pace')
    p.set_binding(None)
    
    v = mod.get('membrane.V')
    v.demote()
    v.set_rhs(0)
    v.set_binding('pace') # Bind to the pacing mechanism

    mem = mod.get('membrane')
    v = mem.get('V')
    v.set_binding(None)

    for v_name, st in segment_dict.items():
        v_new = mem.add_variable(v_name)
        v_new.set_rhs(st)

    vp = mem.add_variable('vp')
    vp.set_rhs(0)
    vp.set_binding('pace')

    v.set_rhs(piecewise)

    sim = myokit.Simulation(mod, proto)
    sim.set_max_step_size(1)
    times = np.arange(0, t_max, 0.1)

    dat = sim.run(t_max, log_times=times)

    t = dat.time()
    i_out = dat['membrane.i_ion']
    v = dat['membrane.V']

    if with_all_dat:
        return times, i_out, v, dat
    else:
        return times, i_out, v

def change_vcp_start(t_vcp, i_out, v_vcp):
    start = np.where(t_vcp-50000 == np.min(np.abs((t_vcp-50000))))[0][0]
    t_vcp = t_vcp[start:]-t_vcp[start]
    i_out = i_out[start:]
    v_vcp = v_vcp[start:]
    return(t_vcp, i_out, v_vcp)

def get_ap(ap_dat):
    t = ap_dat['Time (s)'].values * 1000
    v = ap_dat['Voltage (V)'].values * 1000

    if (((v.max() - v.min()) < 20) or (v.max() < 0)):
        return v[60000:62500], 'flat'

    kernel_size = 100
    kernel = np.ones(kernel_size) / kernel_size
    v_smooth = np.convolve(v, kernel, mode='same')

    peak_idxs = find_peaks(np.diff(v_smooth), height=.1, distance=1000)[0]

    if len(peak_idxs) < 2:
        return v[0:2500], 'flat'
        #import pdb
        #pdb.set_trace()
        #return 

    idx_start = peak_idxs[0] - 50
    idx_end = idx_start + 2500
    #min_v = np.min(v[peak_idxs[0]:peak_idxs[1]])
    #min_idx = np.argmin(v[peak_idxs[0]:peak_idxs[1]])
    #search_space = [peak_idxs[0], peak_idxs[0] + min_idx]
    #amplitude = np.max(v[search_space[0]:search_space[1]]) - min_v
    #v_90 = min_v + amplitude * .1
    #idx_apd90 = np.argmin(np.abs(v[search_space[0]:search_space[1]] - v_90))

    return v[idx_start:idx_end], 'spont'

def plot_figure_only_ap():
    fig = plt.figure(figsize=(4.5, 4))
    fig.subplots_adjust(.15, .12, .95, .95)

    grid = fig.add_gridspec(1, 1)

    ax_spont = fig.add_subplot(grid[0])

    plot_cc(ax_spont, 'flat', 'k')
    plot_cc(ax_spont, 'spont', 'k')

    ax_spont.spines['top'].set_visible(False)
    ax_spont.spines['right'].set_visible(False)

    ax_spont.set_ylabel('Voltage (mV)')

    matplotlib.rcParams['pdf.fonttype'] = 42
    #plt.savefig('./figure-pdfs/f-ap_my_dat_hetero.pdf', transparent=True)

    plt.show()

def plot_cc(ax, cc_type, alph = 0.3, col=None):
    all_files = listdir('../ap-vc-correlations/data/cells')

    all_ap_features = []
    all_currs = []

    t = np.linspace(0, 250, 2500)

    n = 0

    for f in all_files:
        if '.DS' in f:
            continue

        ap_dat = pd.read_csv(f'../ap-vc-correlations/data/cells/{f}/Pre-drug_spont.csv')
        ap_dat, cc_shape = get_ap(ap_dat)

        if cc_shape == cc_type:
            if cc_type == 'spont':
                if col is None:
                    ax.plot(t, ap_dat, '#377eb8', alpha = alph, rasterized=True)
                else:
                    ax.plot(t, ap_dat, col, alpha = alph, rasterized=True)
                t_sp = t
                ap_dat_sp = ap_dat
            else:
                if col is None:
                    ax.plot(t, ap_dat, '#ff7f00', alpha = alph+0.2, rasterized=True)
                else:
                    ax.plot(t, ap_dat, col, alpha = alph+0.2, rasterized=True)

            #if f == '6_033021_4_alex_control':
                #ax.plot(t, ap_dat, 'pink', rasterized=True)
            #if f == '4_022421_1_alex_cisapride':
            #    ax.plot(t, ap_dat, 'pink', rasterized=True)


            #ax.plot(ap_dat, 'k', alpha = .3)
            n += 1
        else:
            continue

    #if cc_type == 'spont':
    #    ax.plot(t_sp, ap_dat_sp, 'pink')

    #ax.text(100, 40, f'n={n}')
    ax.set_ylim(-70, 50)
    #ax.set_xlabel('Time (ms)')

def get_pareto_df(path, weights, append = False, name = 'weight_analysis_adj.csv.bz2'):
    best_data_1 = []

    if append == True:
        og_data = pd.read_csv('./data/'+name)
        og_data = og_data.drop(['level_0'], axis = 1)
        for w in range(0, len(weights)):
            og_data = og_data[og_data['weights']!=weights[w]]
        best_data_1.append(og_data)

    for w in range(0, len(weights)):
        all_trials = pd.read_csv(path+'weight'+str(weights[w])+'_all_data.csv')
        best_data = all_trials.sort_values(by='fitness', ascending = True).reset_index().iloc[0:50]
        best_data['weights'] = weights[w]
        best_data_1.append(best_data)

    best_data_1 = pd.concat(best_data_1)
        
    if append == False:
        best_data_1 = best_data_1.reset_index()
    best_data_1.to_csv('./data/'+name, index=False)

def get_alldata(inputs):
    # GET ALL DATA FILE FOR EACH WEIGHT
    weight, data_folder = inputs
    trials_lst = []

    print(' ')
    print('weight = ', weight)

    for t in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if os.path.isfile(data_folder+'trial'+str(t)+'_'+str(weight)+'_info.csv'):
            trials_lst.append('trial'+str(t)+'_'+str(weight))
    
    print(trials_lst)
    
    generate_alldata(data_folder, trials = trials_lst, save_data_to=data_folder+'weight'+str(weight)+'_', compressed=False)

def plot_pareto(data, weights, axs, x_key, y_key, legend = True, x_label = '', y_label = '',  scale_x = 1, scale_y = 1, xlim = None, ylim = None):
    cmap = plt.get_cmap('turbo', len(weights))
    x_vals = []
    y_vals = []

    for i in range(0, len(weights)):
        w = weights[i]
        weight_data = data[data['weights']==w]
        y = np.mean((weight_data[y_key]/scale_y))
        x = np.mean(weight_data[x_key]/scale_x)
        y_vals.append(y)
        x_vals.append(x)
        axs.scatter(x, y, label = '\u03B1 = ' + str(w), color = cmap(i), zorder = 100)


    # Optimal Point
    x_min = x_vals[np.argmin(x_vals)]
    y_min = y_vals[np.argmin(y_vals)]
    axs.plot([x_min, x_min], [0, 200], color = 'gray', linestyle = 'dashed')
    axs.plot([0, 15000], [y_min, y_min], color = 'gray', linestyle = 'dashed')
    axs.scatter(x_min, y_min, color = 'gray')

    # Find alpha closest to optimal 
    dists = []
    for i in range(0, len(x_vals)):
        x = x_vals[i]
        y = y_vals[i]
        d = np.sqrt((x_min-x)**2+(y_min-y)**2)
        dists.append(d)
    axs.scatter(x_vals[np.argmin(dists)], y_vals[np.argmin(dists)], color = 'black', zorder = 0, s = 100)
    print('alpha:', weights[np.argmin(dists)])

    axs.set_xlabel(x_label, fontsize = 16)
    axs.set_ylabel(y_label, fontsize = 16)
    if legend == True:
        axs.legend(ncol = 2, frameon=False)

    std_x = np.std(x_vals)
    std_y = np.std(y_vals)
    axs.set_ylim([min(y_vals)-std_x, max(y_vals)+std_x])
    axs.set_xlim([min(x_vals)-std_y, max(x_vals)+std_y])


    # Format Plots
    if ylim is not None:
        axs.set_ylim(ylim)
    if xlim is not None:
        axs.set_xlim(xlim)

def plot_drug_response(best_drug_data, drug, axs, cell_stats, legend = False):

    # Plotting Alex's Data

    cell_stats = cell_stats[cell_stats['file']!='031621_2_verapamil'] #filter out this file since it is broken
    drug_cell_stats = cell_stats[cell_stats['drug_type']==drug].reset_index()
    paci_vcp_data = pd.read_csv('./data/paci_vcp_data.csv.bz2')
    kernik_vcp_data = pd.read_csv('./data/kernik_vcp_data.csv.bz2')

    ####### PLOT EXPERIMENTAL DATA #######
    # plot predrug
    i = 0
    #cell_predrug = pd.read_csv('./data/ipsc_csv/'+drug_cell_stats['file'][i]+'/pre-drug_paced.csv')
    #axs.plot(cell_predrug['Time (s)'], cell_predrug['Voltage (V)'], color = 'black', alpha = 0.3, label = 'Alex Data')

    # plot postdrug
    cell_postdrug = pd.read_csv('./data/ipsc_csv/'+drug_cell_stats['file'][i]+'/post-drug_paced.csv')
    axs.plot((np.array(cell_postdrug['Time (s)'])*1000)-145, np.array(cell_postdrug['Voltage (V)'])*1000, color = 'black', alpha = 0.1, label = 'Experimental\nData')

    #"""
    for i in drug_cell_stats['file'].tolist():
        #if i != '031121_3_control' and i !='031621_2_verapamil':

        # plot predrug
        #cell = pd.read_csv('./data/ipsc_csv/'+i+'/pre-drug_paced.csv')
        #axs.plot(cell['Time (s)'], cell['Voltage (V)'], color = 'black', alpha = 0.3)

        # plot postdrug
        cell = pd.read_csv('./data/ipsc_csv/'+i+'/post-drug_paced.csv')
        axs.plot((np.array(cell['Time (s)'])*1000)-145, np.array(cell['Voltage (V)'])*1000, color = 'black', alpha = 0.1)
    #"""


    ####### PLOT OPTIMIZED DATA #######
    #for i in range(0, len(best_drug_data['t_Cisapride'])):
    for i in [0]:
        # Formatting best individual
        t = np.array(eval(best_drug_data['t_'+'Control'][i]))
        v = np.array(eval(best_drug_data['v_'+'Control'][i]))
        t_drug = np.array(eval(best_drug_data['t_'+drug][i]))
        v_drug = np.array(eval(best_drug_data['v_'+drug][i]))

        if min(v_drug) < -52 or max(v_drug) > 0:
            ind = i
            if i == 0:
                #axs.plot(t+0.145, v, color = 'black', label = 'Best Inds')
                axs.plot(t_drug+0.145, v_drug, color = 'black', label = 'Adjusted')
            else:
                #axs.plot(t+0.145, v, color = 'black')
                axs.plot(t_drug+0.145, v_drug, color = 'black', label = 'Adjusted Model')

    axs.plot((np.array(eval(paci_vcp_data['t_'+drug][0]))), (np.array(eval(paci_vcp_data['v_'+drug][0]))), color = 'peru', label='Paci')
    axs.plot((np.array(eval(kernik_vcp_data['t_'+drug][0]))), (np.array(eval(kernik_vcp_data['v_'+drug][0]))), color = 'teal', label='Kernik')

    # Formatting Plot
    #axs.set_xlim([0.1, 1])
    if legend == True:
        axs.legend(frameon=False)
    axs.set_ylabel('Membrane Potential (mV)', fontsize = 16)
    axs.set_xlabel('Time (ms)', fontsize = 16)
    axs.set_title(drug, fontsize = 16)

def plot_biomarkers(best_drug_data_og, drugs, axs = None, biomarker = 'apd90', biomarker_label = '$APD_{90}$', percent_change = False):
    if axs is None:
        fig, axs = plt.subplots(1, figsize = (3,5), constrained_layout = True)

    alex_csv_predrug = pd.read_csv('./data/cell_stats.csv')
    alex_csv_predrug = alex_csv_predrug.rename(columns={"rmp": "RMP"})
    alex_csv_change = pd.read_csv('./data/cell_change_stats.csv')
    alex_csv_change = alex_csv_change.rename(columns={"rmp": "RMP"})
    paci_vcp_data = pd.read_csv('./data/paci_vcp_data.csv.bz2')
    kernik_vcp_data = pd.read_csv('./data/kernik_vcp_data.csv.bz2')

    for d in range(0, len(drugs)):
        adp90_change = alex_csv_change[alex_csv_change['drug_type']==drugs[d]][biomarker]*100
        apd90_pred = alex_csv_predrug[alex_csv_predrug['drug_type']==drugs[d]][biomarker]
        apd90_postd = ((adp90_change/100)*apd90_pred)+apd90_pred

        apd90_change_paci = ((paci_vcp_data[biomarker+'_'+drugs[d]][0]-paci_vcp_data[biomarker+'_Control'][0])/paci_vcp_data[biomarker+'_Control'][0])*100
        apd90_pred_paci = paci_vcp_data[biomarker+'_Control'][0]
        apd90_postd_paci = paci_vcp_data[biomarker+'_'+drugs[d]][0]

        apd90_change_kernik = ((kernik_vcp_data[biomarker+'_'+drugs[d]][0]-kernik_vcp_data[biomarker+'_Control'][0])/kernik_vcp_data[biomarker+'_Control'][0])*100
        apd90_pred_kernik = kernik_vcp_data[biomarker+'_Control'][0]
        apd90_postd_kernik = kernik_vcp_data[biomarker+'_'+drugs[d]][0]

        apd90_change_adj = ((best_drug_data_og[biomarker+'_'+drugs[d]][0]-best_drug_data_og[biomarker+'_Control'][0])/best_drug_data_og[biomarker+'_Control'][0])*100
        apd90_pred_adj = best_drug_data_og[biomarker+'_Control'][0]
        apd90_postd_adj = best_drug_data_og[biomarker+'_'+drugs[d]][0]
        
        # Make Plots
        if percent_change == True:
            axs.scatter([d]*len(adp90_change), adp90_change, color = 'gray', alpha = 0.2, s = 30)
            axs.scatter(d, apd90_change_paci, color = 'peru', s = 100)
            axs.scatter(d, apd90_change_adj, color = 'black', s = 100)
            axs.scatter(d, apd90_change_kernik, color = 'teal', s = 100)
            axs.errorbar(d, np.mean(adp90_change), yerr=2*np.std(adp90_change), fmt="_", color="gray", markersize=4, capsize=4, zorder = 0)

            # Format
            axs.set_xticks([-1]+list(range(0, len(drugs)))+[len(drugs)])
            axs.set_xticklabels(['']+drugs+[''], fontname="Arial", fontsize = 16, rotation = 60)
            axs.set_ylabel('% Change '+biomarker_label, fontname="Arial", fontsize = 16)
        else:
            axs.scatter([d]*len(apd90_postd), apd90_postd, color = 'gray', alpha = 0.2,  s = 30)
            axs.scatter(d, apd90_postd_paci, color = 'peru', s = 100)
            axs.scatter(d, apd90_postd_adj, color = 'black', s = 100)
            axs.scatter(d, apd90_postd_kernik, color = 'teal', s = 100)
            axs.errorbar(d, np.mean(apd90_postd), yerr=2*np.std(apd90_postd), fmt="_", color="gray", markersize=4, capsize=4, zorder = 0)

            # Format
            axs.set_xticks([-1]+list(range(0, len(drugs)))+[len(drugs)])
            axs.set_xticklabels(['']+drugs+[''], fontname="Arial", fontsize = 16, rotation = 60)
            axs.set_ylabel(biomarker_label, fontname="Arial", fontsize = 16)

def simulate_model(mod, proto, with_hold=True, sample_freq=0.00004):
    # Imported from the following github repo: https://github.com/Christini-Lab/nav_artifact/tree/main 
    if mod.time_unit().multiplier() == .001:
        scale = 1000
    else:
        scale = 1

    p = mod.get('engine.pace')
    p.set_binding(None)

    v_cmd = mod.get('voltageclamp.Vc')
    v_cmd.set_rhs(0)
    v_cmd.set_binding('pace') # Bind to the pacing mechanism

    # Run for 20 s before running the VC protocol
    if with_hold:
        holding_proto = myokit.Protocol()
        holding_proto.add_step(-.080*scale, 30*scale)
        sim = myokit.Simulation(mod, holding_proto)
        t_max = holding_proto.characteristic_time()
        sim.run(t_max)
        mod.set_state(sim.state())

    t_max = proto.characteristic_time()
    times = np.arange(0, t_max, sample_freq*scale)
    sim = myokit.Simulation(mod, proto)

    dat = sim.run(t_max, log_times=times)

    return dat, times

def get_iv_data(mod, dat, times):
    # Imported from the following github repo: https://github.com/Christini-Lab/nav_artifact/tree/main 
    iv_dat = {}

    cm = mod['voltageclamp']['cm_est'].value()
    i_out = [v/cm for v in dat['voltageclamp.Iout']]
    v = np.array(dat['voltageclamp.Vc'])
    step_idxs = np.where(np.diff(v) > .005)[0]

    v_steps = v[step_idxs + 10]
    iv_dat['Voltage'] = v_steps

    sample_period = times[1]
    if mod.time_unit().multiplier() == .001:
        scale = 1000
    else:
        scale = 1

    currs = []
    #for idx in step_idxs:
    #    currs.append(np.min(i_out[(idx+3):(idx+23)]))

    for idx in step_idxs:
        temp_currs = i_out[(idx+3):(idx+103)]
        x = find_peaks(-np.array(temp_currs), distance=5, width=4)

        if len(x[0]) < 1:
            currs.append(np.min(temp_currs))
        else:
            currs.append(temp_currs[x[0][0]])


    iv_dat['Current'] = currs

    return iv_dat

def mod_sim(param_vals):
    # Imported from the following github repo: https://github.com/Christini-Lab/nav_artifact/tree/main 

    proto = myokit.Protocol()

    for v in range(-90, 50, 2):
        proto.add_step(-100, 2000)
        proto.add_step(v, 20)

    gna = param_vals[0]
    rs = param_vals[1]
    cm = param_vals[2]
    comp = .8

    mod = myokit.load_model('./models/ord_na_lei.mmt')
    mod['INa']['g_Na_scale'].set_rhs(gna)

    mod['voltageclamp']['rseries'].set_rhs(rs)
    mod['voltageclamp']['rseries_est'].set_rhs(rs)

    cm_val = 15
    mod['voltageclamp']['cm_est'].set_rhs(cm)
    mod['model_parameters']['Cm'].set_rhs(cm)

    mod['voltageclamp']['alpha_c'].set_rhs(comp)
    mod['voltageclamp']['alpha_p'].set_rhs(comp)

    dat, times = simulate_model(mod, proto)

    iv_dat = get_iv_data(mod, dat, times)

    print('Done')

    return [iv_dat, param_vals]

def generate_dat(gna_vals):
    # Imported from the following github repo: https://github.com/Christini-Lab/nav_artifact/tree/main 
    # Changed slightly - instead of plotting random models, I keep rs and cm consistant but vary gna between 0 and 1.

    proto = myokit.Protocol()

    for v in range(-90, 50, 2):
        proto.add_step(-100, 2000)
        proto.add_step(v, 20)

    #gna_vals = lhs_array(.2, 5, n=num_mods, log=True)
    #rs_vals = lhs_array(4E-3, 15E-3, n=num_mods)
    #cm_vals = lhs_array(8, 22, n=num_mods)
    #comp = .8

    #gna_vals = [1, 0.75, 0.5, 0.25]
    rs_vals = [20.0 * 1e-3]*len(gna_vals)
    cm_vals = [60]*len(gna_vals)

    with Pool() as p:
        dat = p.map(mod_sim, np.array([gna_vals, rs_vals, cm_vals]).transpose())

    all_currents = []
    all_meta = []

    for curr_mod in dat:
        all_currents.append(curr_mod[0]['Current'])
        all_meta.append(curr_mod[1])

    all_sim_dat = pd.DataFrame(all_currents, columns=dat[0][0]['Voltage'])
    mod_meta = pd.DataFrame(all_meta, columns=['G_Na', 'Rs', 'Cm'])

    all_sim_dat.to_csv('./data/all_sim.csv', index=False)
    mod_meta.to_csv('./data/meta.csv', index=False)