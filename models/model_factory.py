from omegaconf import DictConfig
from .dsam.dsam_original import SpatioTemporalModel
from .braingnn.braingnn import Network as BrainGNN
from .bnt.bnt import BrainNetworkTransformer

def model_factory(config: DictConfig, dataloaders):
    if config.models.name in ["SpatioTemporalModel", "BrainGNN"]:
      return eval(config.models.name)(config, dataloaders)
    elif config.models.name in ["BrainNetworkTransformer"]:
        return eval(config.models.name)(config)
    else:
        return None