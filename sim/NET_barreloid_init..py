"""
init.py

Starting script to run NetPyNE-based thalamus model for thesis.

Usage:
    python barreloid_init.py # Run simulation, optionally plot a raster

MPI usage:
    mpiexec -n 8 nrniv -python -mpi barreloid_init.py

Contributors: joao.moreira@downstate.edu, salvadordura@gmail.com
"""


# snippet of code to import matplotlib and dynamically switch backend to "MacOSX" for plotting
from pydoc import source_synopsis
import sys
from    matplotlib  import  pyplot  as plt
import matplotlib; matplotlib.use('agg')  # to avoid graphics error in servers
from netpyne import sim
import numpy as np

############################################################################################################
# --- Running simulation
cfg, netParams = sim.readCmdLineArgs(simConfigDefault='barreloid_cfg.py', netParamsDefault='barreloid_netParams.py')
addNoise        = True


############################################################################################################

def addNoiseIClamp(sim,variance = 0.001):
    import numpy as np
    print('\t---> Using Ornstein Uhlenbeck to add noise to the IClamp')
    import math
    from CurrentStim import CurrentStim as CS
    vecs_dict={}
    for cell_ind, cell in enumerate(sim.net.cells):
        vecs_dict.update({cell_ind:{'tvecs':{},'svecs':{}}})
        cell_seed      = sim.cfg.base_random_seed + cell.gid
        for stim_ind,stim in enumerate(sim.net.cells[cell_ind].stims):
            if 'NoiseIClamp' in stim['label']:
                try:        
                    mean = sim.cfg.NoiseIClampParams[sim.net.cells[cell_ind].tags['pop']]['amp']
                    # print('mean noise: ', mean, ' nA')
                except:     
                    mean = 0
                    # print('except mean noise: ', mean, ' nA')
                variance         = variance  # from BlueConfig file
                tvec,svec = CS.add_ornstein_uhlenbeck(tau=1e-9,sigma=math.sqrt(variance),mean=mean,duration=sim.cfg.duration,dt=0.25,seed=cell_seed,plotFig=False)
                vecs_dict[cell_ind]['tvecs'].update({stim_ind:tvec})
                vecs_dict[cell_ind]['svecs'].update({stim_ind:svec})

                vecs_dict[cell_ind]['svecs'][stim_ind].play(sim.net.cells[cell_ind].stims[stim_ind]['hObj']._ref_amp, vecs_dict[cell_ind]['tvecs'][stim_ind], True)
    return sim, vecs_dict


sim.initialize(
    simConfig = cfg, 	
    netParams = netParams)  				# create network object and set cfg and net params
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations
sim.net.connectCells()            			# create connections between cells based on params
sim.net.addStims() 							# add network stimulation

if addNoise:
    if sim.cfg.addNoiseIClamp: sim, vecs_dict = addNoiseIClamp(sim)

sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
sim.runSim()                      			# run parallel Neuron simulation  
sim.gatherData()                  			# gather spiking data and cell info from each node
sim.analyze()

cfg.allpops = ['L6A_activated__pop', 'L6A_suppressed__pop', 'L6A_sparse__pop', 'L6A_silent__pop',
               'L6CC_TPC_L1_cAD__pop', 'L6CC_UTPC_cAD__pop', 'L6CC_BPC_cAD__pop', 'L6CC_IPC_cAD__pop',
               'L6IN_LBC_bAC__pop', 'L6IN_LBC_bNA__pop', 'L6IN_LBC_cNA__pop', 
               'L6IN_MC_bAC__pop', 'L6IN_MC_bNA__pop', 'L6IN_MC_cAC__pop']
               
sim.analysis.plotShape(includePre=cfg.allpops, includePost = cfg.allpops, includeAxon=False, showSyns=False,
    cvar= 'voltage', dist=0.6, elev=95, azim=-90, 
    axisLabels=True, synStyle='o', 
    clim= [-55, -65], showFig=False, saveFig=True, synSize=2, figSize=(18, 18))


for popPre in cfg.allpops[0:1]:
    for popPost in cfg.allpops[5:6]:
        if popPre == popPost:
            sim.analysis.plot2Dnet(include= [popPre], figSize=(18, 18), fontSize=12, saveData=None,
                               saveFig= popPre + popPost + '_plot3D.png', showFig=False, view='xz', showConns=False)
        else:
            sim.analysis.plot2Dnet(include= [popPre, popPost], figSize=(18, 18), fontSize=12, saveData=None,
                               saveFig= popPre + popPost + '_plot3D.png', showFig=False, view='xz', showConns=False)

sim.analysis.plotConn(includePre=cfg.allpops, includePost = cfg.allpops, feature='convergence', figSize=(30, 12), fontSize=12, saveData=None, saveFig=True, showFig=False, graphType='bar')
sim.analysis.plotConn(includePre=cfg.allpops, includePost = cfg.allpops, feature='convergence', figSize=(24, 24), fontSize=12, saveData=None, saveFig=True, showFig=False, graphType='matrix')
   
