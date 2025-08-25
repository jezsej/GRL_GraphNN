from omegaconf import DictConfig
from .dsam.dsam_original import SpatioTemporalModel

def model_factory(config: DictConfig, dataloaders):
    if config.models.name in ["SpatioTemporalModel", "BNT", "BrainGNN"]:
      return eval(config.models.name)(config, dataloaders)
    else:
        return None