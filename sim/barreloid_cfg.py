"""
barreloid_cfg.py 

Contributors: joao.moreira@downstate.edu, salvadordura@gmail.com
"""

from netpyne import specs
import pickle
import NetPyNE_BBP
import json
import sys
import os
import numpy as np

saveFigFlag = '' # declared here, but altered afterwards

#------------------------------------------------------------------------------
#   Simulation options
#------------------------------------------------------------------------------
cfg       = specs.SimConfig()     # object of class SimConfig to store simulation configuration
cfg.duration        = 110 # (ms)
cfg.dt              = 0.025                 # Internal integration timestep to use
cfg.printRunTime    = 0.1 # (s)
cfg.printPopAvgRates = True

cfg.hParams         = {'v_init': -60, 'celsius': 34} # changing to -60 mV to remove initial bursts in "in vivo"-like sim
cfg.verbose         = False                 # Show detailed messages
# cfg.cvode_atol      = 1e-6
# cfg.cache_efficient = True
# cfg.random123 = True

# --- Must be (connRandomSecFromList=True) and (distributeSynsUniformly=False), so that conns are picked from a secList, and the list changes everytime
cfg.connRandomSecFromList   = True
cfg.distributeSynsUniformly = False

cfg.base_random_seed = 100000

cfg.rand123GlobalIndex = cfg.base_random_seed

cfg.seeds = {'conn': cfg.base_random_seed, 
             'stim': cfg.base_random_seed, 
             'loc':  cfg.base_random_seed, 
             'cell': cfg.base_random_seed}

#------------------------------------------------------------------------------
# Recording
#------------------------------------------------------------------------------
cfg.recordStep = 0.25              # Step size in ms to save data (eg. V traces, LFP, etc)
# cfg.recordStep = cfg.dt         # Step size in ms to save data (eg. V traces, LFP, etc)

cfg.recordTraces = {    'V_soma': {'sec': 'soma_0', 'loc': 0.5, 'var': 'v'},
                        # 'V_ptr': {'var': 'ptr'},
                        }
#------------------------------------------------------------------------------
# Path configuration
#------------------------------------------------------------------------------
cfg.convertCellMorphologies = False
cfg.loadCellModel           = True
cfg.convertCellModel        = True
cfg.saveCellModel           = True
cfg.plotSynLocation         = True

# cfg.base_dir = '/Users/joao'
cfg.base_dir=os.path.expanduser("~")

cfg.NetPyNE_rootFolder              = '..'
cfg.NetPyNE_JSON_cells              = cfg.NetPyNE_rootFolder+'/cells/netpyne_morphologies'
cfg.NetPyNE_templateCells           = cfg.NetPyNE_rootFolder+'/mod'
cfg.NetPyNE_exportedCells           = cfg.NetPyNE_rootFolder+'/cells/morphologies_swc'
cfg.NetPyNE_L6A_JSON_cells          = cfg.NetPyNE_rootFolder+'/cells/S1_BBP_cells/'
cfg.NetPyNE_network_template        = cfg.NetPyNE_rootFolder+'/conn/barreloid_network_template/network_template.json'

cfg.loadCircuitProperties           = True
cfg.saveCircuitProperties           = True
cfg.stored_circuit_path             = cfg.NetPyNE_rootFolder+'/conn/bbp_circuit_propeties/circuit_dict.pkl'

cfg.BBP_conn_properties             = cfg.NetPyNE_rootFolder+'/conn/calculate_BBP_conn_properties/BBP_conn_propeties.json'

#------------------------------------------------------------------------------
# Simulation Configuration
#------------------------------------------------------------------------------

cfg.modType                         = 'Prob_original'

#------------------------------------------------------------------------------
#   Network
#------------------------------------------------------------------------------
cfg.center_point = 500

cfg.re_rescale = 1
cfg.cao_secs            = 1.2
cfg.rescaleUSE          = 0.4029343148532312 * cfg.re_rescale # From BlueConfig file

cfg.simplifyL6A         = False
cfg.addL6Apops          = True
cfg.pop_shape           = 'cylinder'
cfg.addL6Asubpops       = True
cfg.L6Asubpops          = ['L6CC_TPC_L1_cAD','L6CC_UTPC_cAD','L6CC_BPC_cAD','L6CC_IPC_cAD','L6IN_LBC_bAC','L6IN_LBC_bNA','L6IN_LBC_cNA','L6IN_MC_bAC','L6IN_MC_bNA','L6IN_MC_cAC']

cfg.addL6Ainterconnections = True # --- IMPLEMENT THIS

cfg.removeConns         = False
cfg.singleCellPops      = True # bool or list of pops

cfg.simulateL6only      = True  # removes all pops that are not from L6 to tune connectivity

#------------------------------------------------------------------------------
#   Stimulation
#------------------------------------------------------------------------------

# --- NOTE: change from pop to cellType when we add cell diversity in the thalamus
cfg.addNoiseIClamp=True 

cfg.NoiseIClampParams={                        
                        'L6A_activated__pop':   {'amp':0.1175},
                        'L6A_suppressed__pop':  {'amp':0.13}, 
                        'L6A_sparse__pop':      {'amp':0.09}, 
                        'L6A_silent__pop':      {'amp':0.09},

                        'L6CC_TPC_L1_cAD__pop': {'amp':0.10},
                        'L6CC_UTPC_cAD__pop':   {'amp':0.08}, 
                        'L6CC_BPC_cAD__pop':    {'amp':0.10}, 
                        'L6CC_IPC_cAD__pop':    {'amp':0.08},

                        'L6IN_LBC_bAC__pop':    {'amp':0.08},
                        'L6IN_LBC_bNA__pop':    {'amp':0.03}, 
                        'L6IN_LBC_cNA__pop':    {'amp':0.025}, 
                        'L6IN_MC_bAC__pop':     {'amp':0.08}, 
                        'L6IN_MC_bNA__pop':     {'amp':0.025}, 
                        'L6IN_MC_cAC__pop':     {'amp':0.015},
                    
                    }

# --- All replaced by Ornstein Uhlenbeck combined with IClamp amplitude
cfg.addOrnsteinUhlenbeckIClamp  = False
cfg.add_current_stims           = False
cfg.add_bkg_stim                = False

if cfg.add_current_stims:
    # cfg.current_stim_targets    = ['VPM__pop']
    cfg.current_stim_targets    = ['VPM__pop', 'TRN__pop', 'L6A_activated__pop', 'L6IN__pop']
    # cfg.current_stim_targets    = ['VPM__pop','TRN__pop']
    cfg.current_stim_amp        = [0.1,        0.05,       0.075,                  0.05]
    cfg.current_stim_start      = 0
    cfg.current_stim_duration   = cfg.duration

if cfg.add_bkg_stim:
    cfg.bkg_rate    = [40,          200,            200,            200,]
    cfg.bkg_noise   = [1,           1,              1,              1,]
    cfg.bkg_weight  = [0.001,       0.0005,         0.0005,         0.0005,]
    cfg.bkg_delay   = [0,           0,              0,              0,]
    cfg.bkg_synMech = ['exc',       'exc',          'exc',          'exc',]
    cfg.bkg_pop     = ['L6A__cell', 'L6A__cell',    'VPM__cell',    'TRN__cell',] # 'cellType'

#------------------------------------------------------------------------------
#   Connectivity
#------------------------------------------------------------------------------

cfg.L6A_sup_sup_probfractor = '2.0*'

cfg.L6A_activated_L6IN_probfractor = '10.0*'
cfg.L6IN_L6A_suppressed_probfractor = '10.0*'

#------------------------------------------------------------------------------
#   stim_Thal_L6A
#------------------------------------------------------------------------------

cfg.add_Thal_L6A_activated_stim = True

cfg.current_stim_Thal_L6A_ac_amp = 0.001
cfg.current_stim_Thal_L6A_IN_amp = 0.005

#------------------------------------------------------------------------------
#   Save
#------------------------------------------------------------------------------

cfg.simLabel = 'v0_batch1'
cfg.saveFolder = '../data/'+cfg.simLabel
# cfg.filename            = 'barr_net_'          # Set file output name
cfg.savePickle          = False             # Save params, network and sim output to pickle file
cfg.saveJson            = False              # Save params, network and sim output to JSON file
cfg.saveDataInclude     = ['simData', 'simConfig', 'netParams', 'net'] # ['simData'] # 
cfg.saveCellConns       = True
if cfg.saveCellConns == False: NetPyNE_BBP.Prompt.headerMsg('THIS IS A DEVELOPMENT/DEBUG SESSION: CELL CONNS ARE NOT BEING SAVED - for final runs set cfg.saveCellConns = True')
# folderName              = 'barreloid_network'
# cfg.saveFolder          = '../data/barreloid_sims/'+folderName
cfg.saveCellSecs = True



#------------------------------------------------------------------------------
# --- Plotting
#------------------------------------------------------------------------------
cfg.saveFigPath = cfg.NetPyNE_rootFolder+'/figs/barreloid_figs'

cfg.analysis['plotRaster']  = {'figSize':(25, 20), 
                               'orderBy': 'y',
                               'marker' : 'o',
                               'markerSize':  15,
                               'timeRange': [0,cfg.duration],    
                               'saveFig': True} # Plot a raster


pops = [
            'L6A_activated__pop', 'L6A_suppressed__pop', 'L6A_sparse__pop', 'L6A_silent__pop',
            'L6CC_TPC_L1_cAD__pop', 'L6CC_UTPC_cAD__pop', 'L6CC_BPC_cAD__pop', 'L6CC_IPC_cAD__pop',
            'L6IN_LBC_bAC__pop', 'L6IN_LBC_bNA__pop', 'L6IN_LBC_cNA__pop', 'L6IN_MC_bAC__pop', 'L6IN_MC_bNA__pop', 'L6IN_MC_cAC__pop',
            # 'L6IN__pop',
            ]
# record_pops = [(pop,[0]) for pop in pops]
record_pops = pops
overlay=True


cfg.analysis['plotTraces']  = {
                                'include': record_pops, 
                                'overlay': overlay,
                                'oneFigPer': 'trace',
                                # 'include': [(pop,list(np.arange(0,15))) for pop in pops], 
                                # 'include': [(pop,[0]) for pop in pops], 
                                # 'include': ['all'], 
                                'ylim':[-90,60],
                                'figSize':[24,15],
                                # 'saveFig': cfg.saveFigPath+'/'+cfg.filename+'traces'+'.png',
                                'saveFig': True}

