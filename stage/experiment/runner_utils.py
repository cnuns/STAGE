import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../../')

from custom_implement.utils import set_nn_device_with, set_policy_attributes

from garage.experiment.experiment import ExperimentContext, dump_json
from garage.experiment.deterministic import set_seed
from stage.experiment.local_runner_wrapper import LocalRunnerWrapper
from garage.torch.algos import make_optimizer
from stage.torch.modules.graph_conv_module import GraphConvolutionModule
# from tensorboardX import SummaryWriter
import dowel
from dowel import logger
import time
import socket
from datetime import datetime
import torch.nn as nn
import copy


def restore_training(log_dir, exp_name, args, env_saved=True, env=None):
    # tabular_log_file = os.path.join(log_dir, 'progress_restored.{}.{}.csv'.
    #     format(str(time.time())[:10], socket.gethostname()))
    # text_log_file = os.path.join(log_dir, 'debug_restored.{}.{}.log'.
    #     format(str(time.time())[:10], socket.gethostname()))
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M")

    tabular_log_file = os.path.join(log_dir, f'progress_restored_{date_time}.csv')
    text_log_file = os.path.join(log_dir, f'debug_restored.{date_time}.log')
    variant_log_file = os.path.join(log_dir, f'variant_restored_{date_time}.json')

    args.n_epochs = int(args.n_epochs)
    dump_json(variant_log_file, vars(args))
    logger.remove_all()
    if hasattr(logger, '_prefixes'):
        logger._prefixes = []
    logger.add_output(dowel.TextOutput(text_log_file))
    logger.add_output(dowel.CsvOutput(tabular_log_file))
    # logger.add_output(dowel.TensorBoardOutput(log_dir))
    logger.add_output(dowel.StdOutput())
    logger.push_prefix('[%s] ' % exp_name)

    ctxt = ExperimentContext(snapshot_dir=log_dir,
                             snapshot_mode='gap_and_last',
                             snapshot_gap=1)

    
    runner = LocalRunnerWrapper(
        ctxt,
        eval=args.eval_during_training,
        n_eval_episodes=args.n_eval_episodes,
        eval_greedy=args.eval_greedy,
        eval_epoch_freq=args.eval_epoch_freq,
        save_env=env_saved
    )
    saved = runner._snapshotter.load(log_dir, 'last')
    runner._setup_args = saved['setup_args']
    runner._train_args = saved['train_args']
    runner._stats = saved['stats']

    print(f'[RESTORE TRAIN] ####### Training seed has been changed: {runner._setup_args.seed} --> {args.seed + 1} #######')
    runner._setup_args.seed = runner._setup_args.seed + 1

    set_seed(runner._setup_args.seed)
    algo = saved['algo']
    algo.max_path_length = args.max_env_steps

    if len(algo.policy.gcn_layers) < args.n_gcn_layers: # hop num 변경
        algo.policy.gcn_layers.extend([GraphConvolutionModule(in_features=args.embedding_dim,
                                                              out_features=args.embedding_dim,
                                                              bias=args.gcn_bias,
                                                              id=i) for i in range(len(algo.policy.gcn_layers),args.n_gcn_layers)])
        algo.policy.layers = [algo.policy.encoder, algo.policy.attention_layer, algo.policy.gcn_layers, algo.policy.categorical_output_layer]
        algo._old_policy = copy.deepcopy(algo.policy)
        algo.baseline.gcn_layers.extend([GraphConvolutionModule(in_features=args.embedding_dim,
                                                                out_features=args.embedding_dim,
                                                                bias=args.gcn_bias,
                                                                id=i) for i in range(len(algo.baseline.gcn_layers),args.n_gcn_layers)])
        algo.baseline.layers = [algo.baseline.encoder, algo.baseline.attention_layer, algo.baseline.gcn_layers]

        algo._optimizer = make_optimizer(type(algo._optimizer),
                                         algo.policy,
                                         lr=algo._optimizer.param_groups[0]['lr'],
                                         eps=algo._optimizer.param_groups[0]['eps'],
                                         device=algo._optimizer.param_groups[0]['params'][0].device,
                                         )
        algo._baseline_optimizer = make_optimizer(type(algo._baseline_optimizer),
                                                  algo.baseline,
                                                  lr=algo._baseline_optimizer.param_groups[0]['lr'],
                                                  eps=algo._baseline_optimizer.param_groups[0]['eps'],
                                                  device=algo._baseline_optimizer.param_groups[0]['params'][0].device,
                                                  )

    # Compatibility patch
    if not hasattr(algo, '_clip_grad_norm'):
        setattr(algo, '_clip_grad_norm', args.clip_grad_norm)

    if env_saved:
        env = saved['env']
        env.env.Rcom_th = env.env.Rcom_th.cpu()

    #! set device to Networks
    device = args.device
    set_nn_device_with(algo, device)
    #! set device to Networks----------------------------------------------------------------------

    #! freeze layers ----------------------------------------------------------------------
    def print_modules_with_params_or_subnets(unfreeze_layers, model):
        print("Modules with parameters or submodules:")
        
        for name, module in model.named_modules():
            # 조건: 학습 파라미터가 있거나 nn.Module의 서브클래스인 경우
            if any(True for _ in module.parameters()) or isinstance(module, nn.Module):
                unfreeze = False
                
                if len(unfreeze_layers) == 0 or unfreeze_layers==[]:
                    unfreeze = False
                elif 'all' in unfreeze_layers:
                    unfreeze = True
                else:
                    for layer_name in unfreeze_layers:
                        if layer_name in name:
                            unfreeze = True
                            break
                    
                if unfreeze:
                    for param in module.parameters():
                        param.requires_grad = True
                    print(f"- Variable Name: {name}, Class Name: {module.__class__.__name__}, UnFreeze: {param.requires_grad}")
                        
                else:
                    for param in module.parameters():
                        param.requires_grad = False
                    print(f"- Variable Name: {name}, Class Name: {module.__class__.__name__}, UnFreeze: {param.requires_grad}")

    print_modules_with_params_or_subnets(args.policy_unfreeze_layers, algo.policy)
    print_modules_with_params_or_subnets(args.policy_unfreeze_layers, algo._old_policy)
    
    print_modules_with_params_or_subnets(args.value_unfreeze_layers, algo.baseline)
    if len(args.value_unfreeze_layers) == 0 or args.value_unfreeze_layers==[]:
        algo.baseline.unfreeze_layers = None
    if len(args.policy_unfreeze_layers) == 0 or args.policy_unfreeze_layers==[]:
        algo.policy.unfreeze_layers = None

    set_policy_attributes(algo.policy, args, algo.baseline)
    set_policy_attributes(algo._old_policy, args)

    runner.setup(env=env,
                 algo=algo,
                 sampler_cls=runner._setup_args.sampler_cls,
                 sampler_args=runner._setup_args.sampler_args,
                 hybrid_mode=args.hybrid,
                 devices=args.devices,
                 flag=args.flag,
                 )

    runner._train_args.start_epoch = runner._stats.total_epoch + 1
    runner._train_args.n_epochs = args.n_epochs # edit: Configure it to run up to n_epochs instead of executing n_epochs each time.
    runner._train_args.batch_size = args.bs
    
    print('\nRestored checkpoint from epoch #{}...'.format(runner._train_args.start_epoch))
    print('To be trained for additional {} epochs...'.format(args.n_epochs))
    print('Will be finished at epoch #{}...\n'.format(runner._train_args.n_epochs))

    return runner._algo.train(runner)