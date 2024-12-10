from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
from anotherJEPA3 import JEPA
import glob


def get_device():
    """Check for GPU availability."""
    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    # data_path = "/scratch/DL24FA"
    data_path = "/Volumes/PhData2/DeepLearning"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


def load_model(device):
    """Load or initialize the model."""
    state_dim = 128  # Ensure this matches the state_dim used during training
    action_dim = 2
    hidden_dim = 32  # Ensure this matches the hidden_dim used during training
    model = JEPA(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, ema_rate=0.99).to(device)
    model.load_state_dict(torch.load("trained_jepa.pth"))
    model.eval()
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model(device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
