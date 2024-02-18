"""
batch.py 

Batch simulation for S1 model using NetPyNE

Contributors: salvadordura@gmail.com, fernandodasilvaborges@gmail.com
"""
from netpyne.batch import Batch
from netpyne import specs
import numpy as np

# ----------------------------------------------------------------------------------------------
# Custom
# ----------------------------------------------------------------------------------------------
def custom():
    params = specs.ODict()
    
    # params[('seeds', 'stim')] =  [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]
    
    params[('L6A_sup_sup_probfractor')] = ['4.0*', '6.0*']

    # params[('L6A_activated_L6IN_probfractor')] = ['10.0*']
    # params[('L6IN_L6A_suppressed_probfractor')] = ['10.0*']
    
    params[('current_stim_Thal_L6A_ac_amp')] = [0.00125]    

    params[('current_stim_Thal_L6A_IN_amp')] = [0.005]    

    b = Batch(params=params, netParamsFile='barreloid_netParams.py', cfgFile='barreloid_cfg.py')

    return b

# ----------------------------------------------------------------------------------------------
# Run configurations
# ----------------------------------------------------------------------------------------------
def setRunCfg(b, type='mpi_bulletin'):
    if type=='mpi_bulletin' or type=='mpi':
        b.runCfg = {'type': 'mpi_bulletin', 
            'script': 'init.py', 
            'skip': True}

    elif type=='mpi_direct':
        b.runCfg = {'type': 'mpi_direct',
            'cores': 7,
            'script': 'barreloid_init.py',
            'mpiCommand': '/opt/intel/oneapi/mpi/2021.10.0/bin/mpirun', # --use-hwthread-cpus
            'skip': True}

    elif type=='mpi_direct2':
        b.runCfg = {'type': 'mpi_direct',
            'mpiCommand': 'mpirun -n 8 ./x86_64/special -mpi -python init.py', # --use-hwthread-cpus
            'skip': True}

    elif type=='hpc_slurm_gcp':
        b.runCfg = {'type': 'hpc_slurm', 
            'allocation': 'default',
            'walltime': '72:00:00', 
            'nodes': 1,
            'coresPerNode': 40,
            'email': 'fernandodasilvaborges@gmail.com',
            'folder': '/home/ext_fernandodasilvaborges_gmail_/S1_HFO/sim/', 
            'script': 'init.py', 
            'mpiCommand': 'mpirun',
            'skipCustom': '_raster_gid.png'}

    elif type == 'hpc_slurm_Expanse_debug':
        b.runCfg = {'type': 'hpc_slurm',
                    'allocation': 'TG-IBN140002',
                    'partition': 'debug',
                    'walltime': '1:00:00',
                    'nodes': 1,
                    'coresPerNode': 4,
                    'email': 'fernandodasilvaborges@gmail.com',
                    'folder': '/home/fborges/S1_HFO/sim/',
                    'script': 'init.py',
                    'mpiCommand': 'mpirun',
                    'custom': '#SBATCH --mem=249325M\n#SBATCH --export=ALL\n#SBATCH --partition=debug',
                    'skip': True}

    elif type == 'hpc_slurm_largeExpanse':
        b.runCfg = {'type': 'hpc_slurm',
                    'allocation': 'TG-IBN140002',
                    'partition': 'large-shared',
                    'walltime': '8:00:00',
                    'nodes': 1,
                    'coresPerNode': 128,
                    'email': 'fernandodasilvaborges@gmail.com',
                    'folder': '/home/fborges/S1_HFO/sim/',
                    'script': 'init.py',
                    'mpiCommand': 'mpirun',
                    'custom': '#SBATCH --mem=512G\n#SBATCH --export=ALL\n#SBATCH --partition=large-shared',
                    'skip': True}
        
    elif type == 'hpc_slurm_Expanse':
        b.runCfg = {'type': 'hpc_slurm',
                    'allocation': 'TG-IBN140002',
                    'partition': 'compute',
                    'walltime': '4:00:00',
                    'nodes': 1,
                    'coresPerNode': 64,
                    'email': 'fernandodasilvaborges@gmail.com',
                    'folder': '/home/fborges/thalamus_netpyne-1/sim/',
                    'script': 'barreloid_init.py',
                    'mpiCommand': 'mpirun',
                    'custom': '#SBATCH --mem=48G\n#SBATCH --export=ALL\n#SBATCH --partition=compute',
                    'skip': True}
        
# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------
if __name__ == '__main__': 
    b = custom() #

    b.batchLabel = 'v1_batch5'  
    b.saveFolder = '../data/'+b.batchLabel
    b.method = 'grid'
    setRunCfg(b, 'mpi_direct') # setRunCfg(b, 'hpc_slurm_Expanse')
    b.run() # run batch
