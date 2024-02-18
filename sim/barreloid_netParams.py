'''
netParams.py 

Contributors: joao.moreira@downstate.edu, salvadordura@gmail.com
'''

#------------------------------------------------------------------------------
# --- LIBRARY IMPORTS
#------------------------------------------------------------------------------
import NetPyNE_BBP
import numpy as np
from netpyne import specs
import pickle, json
import sys
import math

import pandas as pd
import os

netParams = specs.NetParams()   # object of class NetParams to store the network parameters

try:
    from __main__ import cfg  # import SimConfig object with params from parent module
except:
    from barreloid_cfg import cfg

#------------------------------------------------------------------------------
# --- VERSION 
#------------------------------------------------------------------------------
netParams.version = 'S1L6_v0'

#------------------------------------------------------------------------------
#
# --- NETWORK PARAMETERS
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# --- General connectivity parameters
#------------------------------------------------------------------------------
netParams.scaleConnWeight = 1.0 # Connection weight scale factor (default if no model specified)
netParams.scaleConnWeightNetStims = 1.0 #0.5  # scale conn weight factor for NetStims
netParams.defaultThreshold = 0.0 # spike threshold, 10 mV is NetCon default, lower it for all cells

### reevaluate these values
netParams.defaultDelay = 2.0 # default conn delay (ms) # DEFAULT
netParams.propVelocity = 500.0 # propagation velocity (um/ms)
netParams.probLambda = 100.0  # length constant (lambda) for connection probability decay (um)

netParams.shape='cylinder'

netParams.defineCellShapes = True # JV 2021-02-23 - Added to fix the lack of the pt3d term in the cells, which make it unable to record i_membrane_

#------------------------------------------------------------------------------
# --- Load BBP circuit
#------------------------------------------------------------------------------
import Build_Net as BN

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- L6A cells and pop
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if cfg.addL6Apops:
    # --- Based on (Dash ... Crandall, 2022) and (Kwegyir-Afful, 2009)
    CT_pops=['L6A_activated','L6A_suppressed','L6A_sparse','L6A_silent']
    print('\n\t>>>     Adding L6A CT pops - ', CT_pops)
    netParams.cellParams.update(BN.BuildNetwork.getL6ACellTemplate( cellsFolder=cfg.NetPyNE_L6A_JSON_cells,file_format='pkl',cell_pop='L6A'))
    
    # --- testing using a new method that creates L6A and CC/IN pops together
    # netParams.popParams.update( BN.BuildNetwork.getL6ASubtypesPopTemplate(pops=CT_pops,cell_pop='L6A',  center_point=cfg.center_point,pop_flag='__pop',cell_flag='__cell',diversity=True,volume_shape=cfg.pop_shape,group_vecstims=True))
    netParams.popParams.update( BN.BuildNetwork.getL6PopTemplateCtCcIn(   pops=CT_pops,                 center_point=cfg.center_point,pop_flag='__pop',cell_flag='__cell',diversity=True,volume_shape=cfg.pop_shape,group_vecstims=True))

    secTarget = 'basal'     # secList
    
    count_cells=0
    for pop_name in netParams.popParams.keys():
        if 'L6A' in pop_name:
            for CT_pop in CT_pops: 
                if CT_pop in pop_name:
                    try:
                        try:
                            count_cells+=netParams.popParams[pop_name]['numCells']
                        except:
                            count_cells+=len(netParams.popParams[pop_name]['cellsList'])
                        print(pop_name, ' count_cells: ', count_cells)
                    except:
                        print('failed to count cells in ', pop_name, ' pop')
                        continue
    CT_cells= count_cells
    # CT_cells = sum([netParams.popParams[ct_pop+'__pop']['numCells'] for ct_pop in CT_pops])
    
    CC_cells = round((CT_cells/0.563)*0.437)
    IN_cells = round((CT_cells+CC_cells)*0.082737031498839)
    if cfg.addL6Asubpops: 
        print('\n\t>>>     Adding L6A subpops - ', cfg.L6Asubpops)
        for subpop in cfg.L6Asubpops:
            # print('\n\t>>>     Adding L6A subpop - ', subpop)
            netParams.cellParams.update(BN.BuildNetwork.getL6ACellTemplate( cellsFolder=cfg.NetPyNE_L6A_JSON_cells,file_format='pkl',cell_pop=subpop))
            netParams.popParams.update( BN.BuildNetwork.getL6PopTemplateCtCcIn(     pops=[subpop],center_point=cfg.center_point,pop_flag='__pop',cell_flag='__cell',diversity=True,volume_shape=cfg.pop_shape,))
else:
    if cfg.simplifyL6A: 
        netParams.cellParams.update(BN.BuildNetwork.getCellTemplate(    template='izhi',pops=['L6A'],cell_flag='__cell'))
        netParams.popParams.update( BN.BuildNetwork.getPopTemplate(     pops=['L6A'],center_point=cfg.center_point,pop_flag='__pop',cell_flag='__cell'))
        secTarget = 'soma_0'    # single section
    else:
        netParams.cellParams.update(BN.BuildNetwork.getL6ACellTemplate( cellsFolder=cfg.NetPyNE_L6A_JSON_cells,file_format='pkl'))
        # netParams.cellParams.update(BN.BuildNetwork.getL6ACellTemplate( cellsFolder=cfg.NetPyNE_L6A_JSON_cells))
        netParams.popParams.update( BN.BuildNetwork.getPopTemplate(     pops=['L6A'],center_point=cfg.center_point,pop_flag='__pop',cell_flag='__cell',diversity=True,volume_shape='cube'))
        secTarget = 'basal'     # secList
        # print('\t>>\tWarning: Add code to remove AXON segment from L6A cells')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- Synaptic Mechanisms
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --- Dummy exc and inh synMechs for testing
netParams.synMechParams = BN.BuildNetwork.getSynMechParams()

# --- Selecting type of MOD file to be used
if   cfg.modType == 'Det':              modAMPANMDA = 'DetAMPANMDA';                modGABA     = 'DetGABAAB'              # S1 Deterministic  implementation of the BBP mod files
elif cfg.modType == 'Prob_S1':          modAMPANMDA = 'ProbAMPANMDA_EMS_S1';        modGABA     = 'ProbGABAAB_EMS_S1'      # S1 Probabilistic  implementation of the BBP mod files
elif cfg.modType == 'Prob_original':    modAMPANMDA = 'ProbAMPANMDA_EMS_original';  modGABA     = 'ProbGABAAB_EMS_original' # original MOD from BBP model
else:                                   modAMPANMDA = 'ProbAMPANMDA_EMS';           modGABA     = 'ProbGABAAB_EMS'         # Original Thalamus implementation of the BBP mod files
print('\n\t>>\tMOD template\tAMPA: ',   modAMPANMDA, '\tGABA: ', modGABA)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- Model adjustments for in vitro condition
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --- Changes extracellular Ca2+ concentration for all sections in the biophysical cell models
if cfg.cao_secs is not None:
    print('\t>>\tChanging extracellular Ca2+ concentration to ', str(cfg.cao_secs))
    for biophys_cell in netParams.cellParams.keys():
        for sec in netParams.cellParams[biophys_cell]['secs'].keys():
            if 'ions' in netParams.cellParams[biophys_cell]['secs'][sec].keys():
                if 'ca' in netParams.cellParams[biophys_cell]['secs'][sec]['ions'].keys(): netParams.cellParams[biophys_cell]['secs'][sec]['ions']['ca']['o'] = cfg.cao_secs

# --- Rescale USE parameter (probability of synapse activation)
if cfg.rescaleUSE is not None:
    print('\t>>\tRescaling synaptic USE to ', str(cfg.rescaleUSE))
    for mech in netParams.synMechParams.keys():
        try:    netParams.synMechParams[mech]['Use']*=cfg.rescaleUSE
        except: continue

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- Connectivity
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if cfg.addL6Ainterconnections:
    print('\n\t>>Adding L6A interconnections')
    print('\n\t\t>Loading L6A syn mechs')
    S1L6SynMechs_dict       = NetPyNE_BBP.StoreParameters.getS1L6SynMechs()
    netParams.synMechParams.update(S1L6SynMechs_dict)

    print('\t\t>Loading L6A conns')
    S1L6Connectivity_dict   = NetPyNE_BBP.StoreParameters.getS1L6Connectivity()
    netParams.connParams.update(S1L6Connectivity_dict)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- Stimulation
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if cfg.addOrnsteinUhlenbeckIClamp:
    netParams.stimSourceParams['OrnsteinUhlenbeckNoise'] = {'type': 'IClamp', 'del': 0, 'dur': 1e9, 'amp': 0}

    pop_cellTypes=[netParams.popParams[pop]['cellType'] for pop in netParams.popParams.keys() if 'cellType' in netParams.popParams[pop].keys()]
    pop_cellTypes=list(set(pop_cellTypes))
    netParams.stimTargetParams['OrnsteinUhlenbeckNoise_stim'] = {'source': 'OrnsteinUhlenbeckNoise', 'sec':'soma_0', 'loc': 0.5, 'conds': {'cellType':pop_cellTypes}}

# --- Adds a current stim to thalamic populations
if cfg.add_current_stims:
    NetPyNE_BBP.Prompt.headerMsg('Adding Current stim (IClamp)')
    
    for pop_ind, pop in enumerate(netParams.popParams.keys()):
        if pop in cfg.current_stim_targets:
            if len(cfg.current_stim_amp)>1: 
                print(pop,' - IClamp amplitude: ',cfg.current_stim_amp[cfg.current_stim_targets.index(pop)])
                netParams.stimSourceParams['IClamp_'+str(pop_ind)] = {'type': 'IClamp', 'del': cfg.current_stim_start, 'dur': cfg.duration, 'amp': cfg.current_stim_amp[cfg.current_stim_targets.index(pop)]}
            else:                           netParams.stimSourceParams['IClamp_'+str(pop_ind)] = {'type': 'IClamp', 'del': cfg.current_stim_start, 'dur': cfg.duration, 'amp': cfg.current_stim_amp}
            netParams.stimTargetParams['IClamp_'+str(pop_ind)+'__'+pop] = {'source': 'IClamp_'+str(pop_ind), 'sec':'soma_0', 'loc': 0.5, 'conds': {'pop':pop}}

if cfg.addNoiseIClamp:
    for pop in cfg.NoiseIClampParams.keys():
        netParams.stimSourceParams['NoiseIClamp_source__'+pop] = {'type': 'IClamp', 'del': 0, 'dur': 1e9, 'amp': cfg.NoiseIClampParams[pop]['amp']}
        netParams.stimTargetParams['NoiseIClamp_target__'+pop] = {'source': 'NoiseIClamp_source__'+pop, 'sec':'soma_0', 'loc': 0.5, 'conds': {'pop':pop}}
        


if cfg.add_bkg_stim:
    for ind in range(len(cfg.bkg_rate)):
        netParams.stimSourceParams['bkg_'+str(ind)] = { 
            'type':     'NetStim', 
            'rate':     cfg.bkg_rate[ind], 
            'noise':    cfg.bkg_noise[ind],
            }
        netParams.stimTargetParams['bkg_'+str(ind)+'|'+cfg.bkg_pop[ind].split('__')[0]+'__pop'] = { 
            'source':   'bkg_'+str(ind), 
            'conds':    {'popType': cfg.bkg_pop[ind]}, 
            'weight':   cfg.bkg_weight[ind], 
            'delay':    cfg.bkg_delay[ind], 
            'synMech':  cfg.bkg_synMech[ind],
            'sec': 'soma_0'
            }


if cfg.add_Thal_L6A_activated_stim:
    netParams.stimSourceParams['TC_stim'] = { 
            'type':     'NetStim', 
            'rate':     20.0, 
            'start':    5000.0, 
            'number':   60.0, 
            'noise':    1.0,
            }
    netParams.stimTargetParams['TC_stim_Tar'] = { 
            'source':   'TC_stim', 
            'conds':    {'pop': 'L6A_activated__pop'}, 
            'weight':   cfg.current_stim_Thal_L6A_ac_amp, 
            'delay':    0.1, 
            'synMech':  'exc',
            'sec': 'soma_0'
            }
    netParams.stimSourceParams['TC_stim2'] = { 
            'type':     'NetStim', 
            'rate':     20.0, 
            'start':    5000.0, 
            'number':   60.0, 
            'noise':    1.0,
            }
    netParams.stimTargetParams['TC_stim_Tar2'] = { 
            'source':   'TC_stim2', 
            'conds':    {'pop': ['L6IN_LBC_bAC__pop', 'L6IN_LBC_bNA__pop', 'L6IN_LBC_cNA__pop', 'L6IN_MC_bAC__pop', 'L6IN_MC_bNA__pop', 'L6IN_MC_cAC__pop']}, 
            'weight':   cfg.current_stim_Thal_L6A_IN_amp, 
            'delay':    0.1, 
            'synMech':  'exc',
            'sec': 'soma_0'
            }

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- Network mods - Section for specific changes after network creation
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------


if cfg.removeConns:
    print('\n\n\n ---- REMOVING ALL CONNS FOR DEBUGGING --- \n\n\n')
    netParams.connParams={}

if cfg.singleCellPops:
    print('\n\n\n ---- SINGLE CELL POPS FOR DEBUGGING --- \n\n\n')
    if type(cfg.singleCellPops)==list:  pops=cfg.singleCellPops
    else:                               pops=netParams.popParams.keys()


    for pop in pops:
        if   ('VPM' in pop) or ('TRN' in pop):                      numCells=1
        elif ('L6A' in pop) or ('L6CC' in pop) or ('L6IN' in pop):  numCells=5
        else:                                                       numCells=1

        if 'cellsList' in netParams.popParams[pop].keys():
            if len(netParams.popParams[pop]['cellsList'])>=numCells: 
                netParams.popParams[pop]['cellsList']=netParams.popParams[pop]['cellsList'][0:numCells]
                print(netParams.popParams[pop]['cellsList'])
            else:
                print('nope: ', pop)
                print(netParams.popParams[pop]['cellsList'])
                # netParams.popParams[pop]['cellsList']=[netParams.popParams[pop]['cellsList'][0]]
            # netParams.popParams[pop]['numCells']=numCells
        elif 'density' in netParams.popParams[pop].keys():
            del netParams.popParams[pop]['density']
            netParams.popParams[pop]['numCells']=numCells
        elif 'numCells' in netParams.popParams[pop].keys():
            netParams.popParams[pop]['numCells']=numCells
        else:
            netParams.popParams[pop]['numCells']=numCells
            continue


if    cfg.simulateL6only:
    print('\n\n\n ---- Running L6A ONLY mode to inspect tune L6 connectivity --- \n\n\n')
    netParams.cellParams    = {cell_name: cell_params for cell_name, cell_params in netParams.cellParams.items() if 'L6'   in cell_name}
    netParams.popParams     = {pop_name:  pop_params  for pop_name,  pop_params  in netParams.popParams.items()  if 'L6'   in pop_name }
    netParams.connParams    = {conn_name: conn_params for conn_name, conn_params in netParams.connParams.items() if '_L6_' in conn_name}
    print('\n\n',netParams.cellParams.keys())
    print('\n\n',netParams.popParams.keys())
    print('\n\n',netParams.connParams.keys())
    # print('\n\n',netParams.synMechParams)
    
    for conn_name, conn_params in netParams.connParams.items():
        if 'EE_' not in conn_name:                
            if '_activated' in conn_name:
                netParams.connParams[conn_name]['probability'] =  cfg.L6A_activated_L6IN_probfractor + netParams.connParams[conn_name]['probability']
            elif '_suppressed' in conn_name:
                netParams.connParams[conn_name]['probability'] =  cfg.L6IN_L6A_suppressed_probfractor + netParams.connParams[conn_name]['probability']
        else:
            if '_suppressed' in conn_name:
                netParams.connParams[conn_name]['probability'] =  cfg.L6A_sup_sup_probfractor + netParams.connParams[conn_name]['probability']
