import os

#from stable_baselines_master.stable_baselines.a2c import A2C
#from stable_baselines_master.stable_baselines.acer import ACER
#from stable_baselines_master.stable_baselines.acktr import ACKTR
#from stable_baselines_master.stable_baselines.deepq import DQN
from stable_baselines_master.stable_baselines.her import HER
from stable_baselines_master.stable_baselines.ppo2 import PPO2
#from stable_baselines_master.stable_baselines.td3 import TD3
from stable_baselines_master.stable_baselines.sac import SAC

# Load mpi4py-dependent algorithms only if mpi is installed.
try:
    import mpi4py
except ImportError:
    mpi4py = None

if mpi4py is not None:
    from stable_baselines_master.stable_baselines.ddpg import DDPG
#    from stable_baselines_master.stable_baselines.gail import GAIL
#    from stable_baselines_master.stable_baselines.ppo1 import PPO1
#    from stable_baselines_master.stable_baselines.trpo_mpi import TRPO
del mpi4py

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), 'version.txt')
with open(version_file, 'r') as file_handler:
    __version__ = file_handler.read().strip()
