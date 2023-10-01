"""
utils for saving
"""
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """saves .pth model to dir

    args:
    model: .pth model.
    target_dir: target_dir.
    model_name: "model_name.pth"

    example:
    save_model(model=model_0,
               target_dir="models",
               model_name="dummy_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)

def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None
                  ) -> torch.utils.tensorboard.writer.SummaryWriter():
    """creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.
    timestamp- YYYY-MM-DD

    Args:
        experiment_name (str): name of experiment.
        model_name (str): name of model.
        extra (str, optional): extra info to add. defaults None.

    Returns:
        instance of SummaryWriter() saving to log_dir.

    Example usage:
        - Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        - the above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)
