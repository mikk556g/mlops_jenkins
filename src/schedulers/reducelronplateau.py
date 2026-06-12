from torch.optim.lr_scheduler import ReduceLROnPlateau


def reducelronplateau(optimizer, scheduler_config):

    if scheduler_config["name"].lower() == "reducelronplateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=scheduler_config["lr_factor"],
            patience=scheduler_config["lr_patience"],
        )
    else:
        raise ValueError("Unsupported scheduler")

    return scheduler
