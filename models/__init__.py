def build_model(config):
    model_name = config['name'].lower()

    if model_name == 'braingnn':
        from models.braingnn.braingnn import BrainGNN
        return BrainGNN(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            pooling=config.get('pooling', 'attention')
        )

    elif model_name == 'dsam':
        from models.dsam.dsam import DSAM
        return DSAM(
            input_dim=config['input_dim'],
            spatial_dim=config['spatial_dim'],
            temporal_dim=config['temporal_dim'],
            num_classes=config['num_classes']
        )

    elif model_name == 'bnt':
        from models.bnt.bnt import BNT
        return BNT(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes']
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")
