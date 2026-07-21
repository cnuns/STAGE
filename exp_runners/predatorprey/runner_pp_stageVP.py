import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../')
sys.path.append(current_file_path + '/../../')

from custom_implement.utils import set_nn_device_with, check_nn_on_device
parent_folder = os.path.abspath(os.path.join(current_file_path, os.pardir, os.pardir))
sys.path.append(parent_folder)

import argparse
import joblib
import time
from types import SimpleNamespace
import torch
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.experiment.deterministic import set_seed
from envs import PredatorPreyWrapper

from stage.torch.algos import CentralizedMAPPO
from stage.experiment.local_runner_wrapper import LocalRunnerWrapper
from stage.sampler import CentralizedMAOnPolicyVectorizedSampler
from stage.torch.baselines import CommBaseCritic # baseline
from stage.torch.policies import CommCategoricalMLPPolicy # policy

from exp_runners.routine import *
from custom_implement.utils import set_nn_device_with, check_nn_on_device
from custom_implement.randomness import fix_randomness
from exp_runners.testing import *

from eval_pp import eval_simple, eval_model
try: from utils_pp import *
except: from .utils_pp import *

import signal
flag = [0]
def on_exit_signal(signum, frame):
    global flag
    flag[0] = 1
    print("[SubProcess End] Train end by signal")

signal.signal(signal.SIGINT, on_exit_signal)

def stageVP(args, env, ):
    policy = CommCategoricalMLPPolicy(
        env.spec,
        n_agents=args.n_agents,
        encoder_hidden_sizes=args.encoder_hidden_sizes,
        embedding_dim=args.embedding_dim,
        attention_type=args.attention_type,
        n_gcn_layers=args.n_gcn_layers,
        residual=bool(args.residual),
        gcn_bias=bool(args.gcn_bias),
        categorical_mlp_hidden_sizes=args.categorical_mlp_hidden_sizes,
        name='comm_categorical_mlp_policy',
        device=args.device,
    )
    
    baseline = CommBaseCritic(
        env.spec,
        n_agents=args.n_agents,
        encoder_hidden_sizes=args.encoder_hidden_sizes,
        embedding_dim=args.embedding_dim,
        decoder_hidden_sizes=args.decoder_hidden_sizes,
        attention_type=args.attention_type,
        n_gcn_layers=args.n_gcn_layers,
        residual=bool(args.residual),
        gcn_bias=bool(args.gcn_bias),
        aggregator_type=args.aggregator_type,
        device=args.device,
    )

    return policy, baseline

def run(args):
    args = pp.preSetting(args, scenario_name='predator')

    if args.mode == 'train':
        # making sequential log dir if name already exists
        @wrap_experiment(name=args.exp_name,
                         prefix=args.prefix,
                         log_dir=args.exp_dir,
                         snapshot_mode='gap_and_last',
                         snapshot_gap=1)
        
        def train_predatorprey(ctxt=None, args_dict=vars(args)):
            args = SimpleNamespace(**args_dict)
            
            set_seed(args.seed)
            fix_randomness(args.seed)

            env = PredatorPreyWrapper(
                centralized=True,
                grid_shape=(args.grid_size, args.grid_size),
                n_agents=args.n_agents,
                n_preys=args.n_preys,
                max_steps=args.max_env_steps,
                step_cost=args.step_cost,
                prey_capture_reward=args.capture_reward,
                penalty=args.penalty,
                other_agent_visible=bool(args.agent_visible),
                params = vars(args)
            )
            env = GarageEnv(env)

            runner = LocalRunnerWrapper(
                ctxt,
                eval=args.eval_during_training,
                n_eval_episodes=args.n_eval_episodes,
                eval_greedy=args.eval_greedy,
                eval_epoch_freq=args.eval_epoch_freq,
                save_env=env.pickleable,
            )
            
            policy, baseline = stageVP(args, env)

            # Set max_path_length <= max_steps
            # If max_path_length > max_steps, algo will pad obs
            # obs.shape = torch.Size([n_paths, algo.max_path_length, feat_dim])
            algo = CentralizedMAPPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=args.max_env_steps, # Notice
                discount=args.discount,
                center_adv=bool(args.center_adv),
                positive_adv=bool(args.positive_adv),
                gae_lambda=args.gae_lambda,
                policy_ent_coeff=args.ent,
                entropy_method=args.entropy_method,
                stop_entropy_gradient=True \
                   if args.entropy_method == 'max' else False,
                clip_grad_norm=args.clip_grad_norm,
                optimization_n_minibatches=args.opt_n_minibatches,
                optimization_mini_epochs=args.opt_mini_epochs,
                device=args.device,
            )

            # check neural networks on target device
            check_nn_on_device(algo, args.device)

            runner.setup(algo, env,
                sampler_cls=CentralizedMAOnPolicyVectorizedSampler, 
                sampler_args={'n_envs': args.n_envs},
                hybrid_mode=args.hybrid,
                devices=args.devices,
                flag=args.flag,
                )
            
            runner.train(n_epochs=args.n_epochs, batch_size=args.bs)

        train_predatorprey(args_dict=vars(args))

    elif args.mode in ['restore', 'eval', 'test']:
        exp_dir = model_path = f'{PATH_MODEL}/{args.model_to_loading}'

        set_seed(args.seed)
        fix_randomness(args.seed)

        if args.mode == 'restore':
            data = joblib.load(exp_dir + '/params.pkl')
            algo = data['algo']
            
            from stage.experiment.runner_utils import restore_training
            if args.policy_unfreeze_layers is not None:
                env = PredatorPreyWrapper(
                    centralized=True,
                    grid_shape=(args.grid_size, args.grid_size),
                    n_agents=args.n_agents,
                    n_preys=args.n_preys,
                    max_steps=args.max_env_steps,
                    step_cost=args.step_cost,
                    prey_capture_reward=args.capture_reward,
                    penalty=args.penalty,
                    other_agent_visible=bool(args.agent_visible),
                    params = vars(args)
                )
                env = GarageEnv(env)
                
                restore_training(exp_dir, args.exp_name, args, env_saved=False, env=env)
            else:
                env = data['env']
                restore_training(exp_dir, args.exp_name, args, env_saved=env.pickleable, env=env)

        elif args.mode == 'eval':
            data = joblib.load(exp_dir + '/params.pkl')
            algo = data['algo']
            
            set_policy_attributes(algo.policy, args)

            env = PredatorPreyWrapper(
                centralized=True,
                grid_shape=(args.grid_size, args.grid_size),
                n_agents=args.n_agents,
                n_preys=args.n_preys,
                max_steps=args.max_env_steps,
                step_cost=args.step_cost,
                prey_capture_reward=args.capture_reward,
                penalty=args.penalty,
                other_agent_visible=bool(args.agent_visible),
                params = vars(args)
            )

            device = args.device
            set_nn_device_with(algo, device, print_option=True)

            eval_simple(args, env, algo)
            
        elif args.mode == 'test':
            env = PredatorPreyWrapper(
                centralized=True,
                grid_shape=(args.grid_size, args.grid_size),
                n_agents=args.n_agents,
                n_preys=args.n_preys,
                max_steps=args.max_env_steps,
                step_cost=args.step_cost,
                prey_capture_reward=args.capture_reward,
                penalty=args.penalty,
                other_agent_visible=bool(args.agent_visible),
                params = vars(args)
            )
            
            test_and_output_result_file(args, env, flag, model_path, PATH_TEST, pp, eval_model)
            

if __name__ == '__main__':
    pp = PPUtil()
    try: PC_NAME = os.popen('echo %PC_NAME%').read().strip()
    except: PC_NAME = 'temp'
    PATH_MODEL = './data/model'
    PATH_TEST = './data/test'
    
    parser = argparse.ArgumentParser()
    args = pp.parser_init(parser)
    args.PC_NAME = PC_NAME
    args.EXEC_Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    args.exe = 'stageVP'
    
    args.preTrMap = [10] # train map sizes before transfer
    args.preTrHop = [1] # train hop nums before transfer
    args.preTrEnvStep = [3.5] # train env steps before transfer
    args.grid_size = args.trMap = 30 # last train map size
    args.n_gcn_layers = 2 # last train hop num
    trEnvStep = 7.0 # total env steps(x * 7.5 * 10 ** 6)
    args.max_env_steps = int((15*args.grid_size)+250)
    args.teMap = args.map
    args.policy_unfreeze_layers = ['all']
    args.value_unfreeze_layers = ['all']
    # args.debug = 1 # this will start training just 3-epochs and testing for each epochs
    #! Run by
    # python runner_pp_stageVP.py --cmd train --map 10 --sen 2 --den 0.03 --loss 0 --fault 0

    if 'train' in args.cmd:
        for map_size, hop_num, step in zip(args.preTrMap, args.preTrHop, args.preTrEnvStep):
            args.preMap = map_size
            args.preHop = hop_num
            args.trEnvStep = step
            args, is_trained = ready_to_train(args, pp, flag, debug=0)
            if not is_trained:
                run(args)
        
        if trEnvStep > args.trEnvStep:
            flag[0] = 0 # Terminate running by typing "Ctrl + C"

            args.preMap = None
            args.preHop = None
            args.trEnvStep = trEnvStep
            args, is_trained = ready_to_train(args, pp, flag, debug=0)
            if not is_trained:
                run(args)

    flag[0] = 0 # Terminate running by typing "Ctrl + C"
    
    if 'test' in args.cmd or 'eval' in args.cmd:
        args = ready_to_test(args, pp, debug=0)
        # args.render = 1 # if you want to render the environment while test, add this
        run(args)
    

