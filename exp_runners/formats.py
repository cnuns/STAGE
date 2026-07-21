import argparse
import re
import collections
import os
from glob import glob
import pandas as pd
from custom_implement.utils import extracting_number, pill_last_path
from custom_implement.utils import send_to_trash
from testing import TEST_METRIC_VECTORS

class FileFormat():
    rTrainForm = re.compile(r'(gnn|gnnOL|smarl|smarlVP|smarlP)*.*TR')
    rTestQuery = re.compile(r'(gnn|gnnOL|smarl|smarlVP|smarlP)*.*TE')
    def __init__(self):
        pass
    
    @classmethod
    def getTrainForm(self, exp_name):
        """
        Extract only up to the train setup part from the file name.
        """

        trainForm = self.rTrainForm.search(exp_name).group()
        
        if trainForm:
            while trainForm[-1] in ['_', '.']:
                trainForm = trainForm[:-1]

            args = argparse.Namespace()
            args.exp_name = exp_name

            args = self.parameter_parsing(exp_name)
            args.trainForm = trainForm
            
            return trainForm
    
    @classmethod
    def getTrainedMapForm(self, args, trForm, testScale='TrTeSame', dens=None): # dens = {'map':10,'na':6, 'nt':6}
        trainForm = self.rTrainForm.search(trForm).group()
        return trainForm

    @classmethod
    def get_trainings(self, threshold):
        trainings = []
        if os.path.exists(self.trPath) and os.path.isdir(self.trPath):
            #print(f"The folder at '{path}' exists.")
            pass
        else:
            print(f"The folder at '{self.trPath}' does not exist.")
            raise Exception('./DB 폴더에 아무 폴더도 확인되지 않음. symlink 생성 확인할 것. 오류발생으로 exit.')

        train_dirs = glob(f'{self.trPath}/*')
        if len(train_dirs) == 0:
            # print('./DB 폴더에 학습된 모델 없음.')
            return trainings
        
        for model_name in train_dirs:
            itrs = glob(f'{model_name}/itrs/itr_*.pkl')

            if 0 <= len(itrs) != threshold:
                trainForm = self.getTrainForm(pill_last_path(model_name))
                #! if index == 0: raise Exception('index 오류')
                trainings.append(trainForm)
            else:
                # print('have to train more')
                pass
            
        return trainings

    @classmethod
    def getTestForm(self, query):
        testForm = self.rTestQuery.search(query).group()
        if testForm:
            while testForm[-1] in ['_', '.']:
                testForm = testForm[:-1]
            return testForm
        else:
            pass

    @classmethod
    def get_trained_list(self, threshold=0):
        trained = []
        if os.path.exists(self.trPath) and os.path.isdir(self.trPath):
            pass
        else:
            raise Exception(f"The trained model path at '{self.trPath}' does not exist.")

        train_dirs = glob(f'{self.trPath}/*')
        if len(train_dirs) == 0:
            return trained
        
        for model_name in train_dirs:
            if threshold != 'debug':
                thrEpoch = threshold
            else:
                thrEpoch = 1

            itrs = glob(f'{model_name}/itrs/itr_*.pkl')
            
            if thrEpoch<=len(itrs):
                itrs_sorted = sorted(itrs) # Sort file names in numerical order.
                itr_max = re.compile('itr_[0-9]+').findall(itrs_sorted[-1])[0]
                itr_max = int(re.compile('[0-9]+').findall(itr_max)[0])

                if thrEpoch <= itr_max :
                    # really trained and finished well
                    trainForm = self.getTrainForm(pill_last_path(model_name))

                    if trainForm not in trained:
                        trained.append(trainForm)
                else:

                    print(model_name)
                    raise Exception('error occur during training')
            
            else:
                pass
            
        return trained
    
    @classmethod
    def get_tested_list(self, args, n_threshold=50):
        tested = []
        n_threshold = 1 if n_threshold == 'debug' else n_threshold
        
        if not(os.path.exists(self.tePath) and os.path.isdir(self.tePath)):
            raise Exception(f"The trained model path at '{self.trPath}' does not exist.")

        test_files = glob(f'{self.tePath}/*.csv')
        for teFile in test_files:
            try:
                df_test = pd.read_csv(f'{teFile}')
            except pd.errors.EmptyDataError:
                continue
            
            valid = True
            columns = df_test.columns.tolist()
            for vector in TEST_METRIC_VECTORS:
                if vector in columns: continue
                else:
                    valid = False
                    break

            if valid:
                if n_threshold <= len(df_test):
                    testForm = self.getTestForm(teFile)
                    tested.append(testForm)

        return tested
    
    @classmethod
    def parameter_parsing(self, args):
        if type(args) == argparse.Namespace:
            exp_name = args.exp_name
        elif type(args) == str:
            exp_name = args
            args = argparse.Namespace()

        args_dict=vars(args)

        rArchi = re.compile(r'(gnn|gnnOL|smarl|smarlVP|smarlP)')
        rScen = re.compile(r'(pp|co)')
        rLinkInput = re.compile('in[0-9]+')
        rModel = re.compile('mo[0-9]+')
        rValuePolicy = re.compile('vp[0-9]+')
        rRewards = re.compile('rew[0-9.]+_[0-9.]+_[0-9.]+_[0-9.]+')
        rDensity = re.compile('den[0-9]+_[0-9.]+_[0-9.]+')
        rLoad = re.compile('load[0-9.]+')
        rHop = re.compile('L[0-9]+')
        rRsen = re.compile('Rsen[0-9]+')
        rTmax = re.compile('Tmax[0-9]+')
        
        rtrHomo = re.compile('trHo[0-9]+')
        rrHop = re.compile('rHop[0-9]+')
        rtrMap = re.compile('trMap[0-9]+')
        rtrRcom = re.compile('trRcom[0-9]+')
        rtrPl = re.compile('trpl-?[0-9.]+')
        rtrPf = re.compile('trpf[0-9.]+_[0-9.]+_[0-9.]+_[0-9.]+')
        
        rteMap = re.compile('teMap[0-9]+')
        rteRcom = re.compile('teRcom[0-9]+')
        rtePl = re.compile('tepl-?[0-9.]+')
        rtePf = re.compile('tepf[0-9.]+_[0-9.]+_[0-9.]+_[0-9.]+')
        
        keys = {
                'architecture': (rArchi, str, 1),
                'scenario': (rScen, str, 1),
                'n_min': (rLinkInput, int, 1),
                'model': (rModel, int, 1),
                'value_policy': (rValuePolicy, int, 1),
                
                'rewards': (rRewards, float, 4),
                'density': (rDensity, float, 3),
                'load':(rLoad, float, 1),
                'n_gcn_layers':(rHop, int, 1),
                'Rsen':(rRsen, int, 1),
                'max_env_steps': (rTmax, int, 1),
                
                'trHomo': (rtrHomo, int, 1),
                'rHop': (rrHop, int, 1),
                #'grid_size':(rtrMap, int, 1), #*
                'trMap': (rtrMap, float, 1),
                'trRcom': (rtrRcom, float, 1),
                'trpl': (rtrPl, float, 1),
                'trpf': (rtrPf, float, 4),

                'teMap': (rteMap, float, 1),
                'teRcom': (rteRcom, float, 1),
                'tepl': (rtePl, float, 1),
                'tepf': (rtePf, float, 4), 
                }

        for k, (rex, dtype_, n_elem) in keys.items():
            rst = rex.search(exp_name)
            if rst:
                rst = rst.group()
            else:
                continue
            
            if rst:
                if dtype_==str:
                    args_dict[k] = dtype_(rst)
                    continue
                
                number = extracting_number(rst,dtype_,n_elem)
                
                if k == 'index':
                    args_dict[k] = f'ID{str(number[0]).zfill(3)}'
                    continue
                
                if len(number) == n_elem == 1:
                    args_dict[k] = number[0]
                
                else:
                    args_dict[k] = number
                    if k == 'rewards':
                        args_dict['capture_reward'] = dtype_(number[0])
                        args_dict['step_cost'] = dtype_(number[1])
                        args_dict['rm'] = dtype_(number[2])
                        args_dict['penalty'] = dtype_(number[3])

                    if k in ['trpf', 'tepf']:
                        PfNmin, PfNmax, PfPmin, PfPmax= dtype_(number[0]), dtype_(number[1]), dtype_(number[2]), dtype_(number[3])
                        args_dict[k+'Nmin'] = PfNmin 
                        args_dict[k+'Nmax'] = PfNmax 
                        args_dict[k+'Pmin'] = PfPmin 
                        args_dict[k+'Pmax'] = PfPmax
                    
                
        d = vars(args)
        for k, v in d.items():
            try:
                if type(v) != str:
                    if float(v) == int(v):
                        d[k] = int(v)
            except:
                continue
                
        return args
    
    @classmethod
    def process_exp_name_default(self, args):
        exp_layout = collections.OrderedDict([
            ('{}', f'{args.exe}_{args.scenario}'),
            ('in{}', args.n_min),
            
            ('rew{}', f'{args.capture_reward}_{args.step_cost}_{args.rm}_{args.penalty}'),
            ('den{}', f'{args.density[0]}_{args.density[1]}_{args.density[2]}'),
            ('load{}', args.load),
            ('L{}', args.n_gcn_layers),
            ('Rsen{}', args.Rsen),
            ('Tmax{}', args.max_env_steps),
            ('trMap{}', args.trMap),
            ('trRcom{}', args.trRcom),
            ('trpl{}', args.trpl),
            ('trpf{}', f'{args.trpfNmax}_{args.trpfNmin}_{args.trpfPmax}_{args.trpfPmin}'),
            
            ('TR', ''),

            ('teMap{}', args.teMap),
            ('teRcom{}', args.teRcom),
            ('tepl{}', args.tepl),
            ('tepf{}', f'{args.tepfNmax}_{args.tepfNmin}_{args.tepfPmax}_{args.tepfPmin}'),
            ('TE', ''),
        ])
        exp_name = '_'.join([key.format(val) for key, val in exp_layout.items()])
        return exp_name

    def preSetting(self, args, scenario_name):
        if args.exp_name is None:
            exp_name = self.process_exp_name_default(args)
        else:
            exp_name = args.exp_name

        args.prefix = scenario_name
        id_suffix = ('_' + str(args.run_id)) if args.run_id != 0 else ''
        unseeded_exp_dir = './data/' + args.loc +'/' + exp_name[:-7]
        args.exp_dir = './data/' + args.loc +'/' + exp_name + id_suffix

        if args.mode == 'restore':
            if os.path.isfile(args.exp_dir + '/params.pkl'):
                pass
            else:
                send_to_trash(args.exp_dir)
                args.mode = 'train'

        # Enforce
        args.center_adv = False if args.entropy_method == 'max' else args.center_adv

        return args