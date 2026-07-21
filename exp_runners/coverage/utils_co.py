import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../')
sys.path.append(current_file_path + '/../../')

from glob import glob
import pandas as pd
import re
import numpy as np
from env_utils import EnvUtil, add_parser_commons, get_parser_to_args
from formats import FileFormat

trPath = f'{current_file_path}/data/model'
teDataRoot = f'{current_file_path}/data/test'
tePath = f'{current_file_path}/data/test/csv'
teDataPath = f'{current_file_path}/data/test/matlab'
teHistoryPath = f'{current_file_path}/data/test/log'
teDataBKPath = f'{current_file_path}/data/test/backup'

if not os.path.isdir(trPath): os.mkdir(f'{current_file_path}/data')
if not os.path.isdir(trPath): os.mkdir(trPath)
if not os.path.isdir(teDataRoot): os.mkdir(teDataRoot)
if not os.path.isdir(tePath): os.mkdir(tePath)
if not os.path.isdir(teDataPath): os.mkdir(teDataPath)
if not os.path.isdir(teHistoryPath): os.mkdir(teHistoryPath)
if not os.path.isdir(teDataBKPath): os.mkdir(teDataBKPath)

class COUtil(EnvUtil, FileFormat):
    trPath = f'{current_file_path}/data/model'
    tePath = f'{current_file_path}/data/test/csv'

    def __init__(self, trPath=trPath, tePath=tePath, density=None):
        super().__init__(trPath, tePath)
        if type(density) in [list, tuple]:
            if len(density) == 2:
                self.density = {'map':density[0], 'na':density[1], 'nt':density[2]} 
            else:
                self.density = {'map':density[0], 'na':density[1], 'nt':density[2]} 
        elif type(density) == int:
            # self.default_setup = {'map':10, 'na':density, 'nt':density} 
            raise Exception('Type density as iteratable form')
        
        elif density == None:
            pass
        
        self.num_points = 100
        self.teHistoryPath = teHistoryPath
        self.rScenario = re.compile(f'co')
        self.scenario_path = os.path.dirname(os.path.abspath(__file__))

    def calc_metric(self, args, trAvgReward, optimal, cntRc, cntRs, cntRm, cntRp, cntRv, cntRv2):
        trainMetric = trAvgReward / optimal
        if np.any(trainMetric > 1.0): raise Exception('Metric cannot overcome 1.0, please check \'capture\' and \'step\' count info ')
        return trainMetric

    def set_n_agents_by_density(self, args):
        args.n_preys = args.n_obstacles
        args = super().set_n_agents_by_density(args)
        return args

    def set_batchsize_epoch(self, args):
        maps=args.grid_size
        dm, dna, dnt = args.density
        r = maps/dm
        na = dna*r**2

        if 20 <= maps < 30:
            batchs = int(0.5*self.batch_size*na/8)
        elif 30 <= maps < 40:
            batchs = int(0.25*self.batch_size*na/8)
        elif 40 <= maps < 50:
            batchs = int(0.25*self.batch_size*na/8)
        elif 50 <= maps < 60:
            batchs = int(0.25*self.batch_size*na/8)
        elif 60 <= maps < 70:
            batchs = int(0.2*self.batch_size*na/8)
        elif 70 <= maps < 80:
            batchs = int(0.1*self.batch_size*na/8)
        else:
            batchs = int(self.batch_size*args.n_agents/8)

        tr = self.getTrainedMapForm(args, args.exp_name)
        prog = glob(f'{os.getcwd()}/data/model/{tr}/progress*.csv')
        totalEnvSteps = 0
        totalEpochs = 0
        for p in prog:
            try:
                df = pd.read_csv(p)
                totalEnvSteps = max(np.max(df['TotalEnvSteps']), totalEnvSteps)
                totalEpochs = max(np.max(df['Iteration']), totalEpochs)
            except:
                continue
            
        xEnvSteps = args.trEnvStep * self.EnvStep - totalEnvSteps
        n_epochs = totalEpochs + self.get_epoch_must_train(xEnvSteps, na, batchs) + 1

        return batchs, n_epochs
    
    def parser_init(self, parser):
        parser.add_argument('--map', type=int, default=10)
        parser.add_argument('--sen', type=int, default=2)
        parser.add_argument('--den', type=float, default=0.04)
        parser.add_argument('--loss', type=float, default=0, help='packet loss prob.')
        parser.add_argument('--fault', type=float, default=0, help='observation fault prob.')
        
        #! scenario specific
        parser.add_argument('--scenario', type=str, default='co')
        parser.add_argument('--add_clock', type=int, default=1, help='0: No / 1: add')
        parser.add_argument('--rewards', nargs='+', default=[2, 0.1, 0, 0.5], type=float)
        parser.add_argument('--capture_reward', type=float, default=2)
        parser.add_argument('--step_cost', type=float, default=0.1)
        parser.add_argument('--rm', type=float, default=0.0, help='each agent moving reward')
        parser.add_argument('--penalty', type=float, default=0.5)
        parser.add_argument('--lazy_penalty', type=float, default=0)
        parser.add_argument('--revisit_penalty', type=float, default=0)
        parser.add_argument('--final_reward', type=float, default=0)
        
        parser.add_argument('--density', nargs='+', type=int, default=[10,3,0])
        parser.add_argument('--grid_size', type=int, default=10)
        parser.add_argument('--n_groups', type=int, default=2, help='')
        parser.add_argument('--n_nodes', type=int, default=1, help='')
        parser.add_argument('--n_agents', '-n', type=int, default=3)

        parser.add_argument('--load', type=int, default=2)
        parser.add_argument('--Rsen', type=int, default=1)
        parser.add_argument('--max_env_steps', type=int, default=400)
        parser.add_argument('--n_obstacles', type=int, default=0, help='this will have a effect if you select: obstDep==pre')

        parser = add_parser_commons(parser)
        args = get_parser_to_args(parser)

        return args