from stage.torch.policies.dec_categorical_mlp_policy \
    import DecCategoricalMLPPolicy

from stage.torch.policies.centralized_categorical_mlp_policy \
    import CentralizedCategoricalMLPPolicy

from stage.torch.policies.comm_categorical_mlp_policy \
    import CommCategoricalMLPPolicy


__all__ = [
    'DecCategoricalMLPPolicy', 
    'CentralizedCategoricalMLPPolicy',
    'CommCategoricalMLPPolicy',
]