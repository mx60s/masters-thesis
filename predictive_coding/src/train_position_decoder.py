import argparse
import re
import gc
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.optim
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize
from PIL import Image
from tqdm.auto import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from predictive_coding.src import PredictiveCoder, VestibularCoder, BottleneckCoder, PositionProbe


def main():
    parser = argparse.ArgumentParser(description="Predictive coding analysis and training script.")
    parser.add_argument("--dataset", type=str, default="env-pcs",
                        help="Name of the dataset directory. Examples: 'small-world-random', 'env-pcs', '0.1-traj-pcs', 'pc-images'.")
    parser.add_argument("--model_set", type=str, default="0.5-0.5",
                        help="Comma- or dash-separated floats specifying model config, e.g. '0.5-0.5' or '0.9-0.9'.")
    parser.add_argument("--seq_len", type=int, default=20,
                        help="Sequence length the model expects.")
    parser.add_argument("--num_actions", type=int, default=3,
                        help="Number of possible actions in the dataset.")
    parser.add_argument("--which_residual", type=int, default=2,
                        help="Which residual block to fetch latents from in the model.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Which device to run on (e.g. 'cuda:0', 'cuda:1', 'cpu').")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs to train the position probe.")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Training batch size for the position probe.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for the position probe.")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for the position probe.")
    parser.add_argument("--experiment_name", type=str, default="",
                        help="Override the experiment name, otherwise uses 'predictive-coding-<model_set>'.")
    args = parser.parse_args()

    current_file_path = Path(__file__).resolve()
    base_dir = current_file_path.parent.parent.parent
    datasets_dir = base_dir / "datasets"
    weights_dir = base_dir / "predictive_coding/weights"

    delimiter = ',' if ',' in args.model_set else '-'
    set_tuple = tuple(map(float, args.model_set.split(delimiter)))

    if args.experiment_name.strip():
        experiment = args.experiment_name
    else:
        experiment = f"predictive-coding-{delimiter.join(map(str, set_tuple))}-new-vest"

    seq_len = args.seq_len
    num_actions = args.num_actions
    dataset = args.dataset

    '''
    Load dataset
    '''
    dataset_path = datasets_dir / dataset
    folds = [p for p in dataset_path.glob("*") if p.is_dir()]
    images = []
    positions = []
    actions = []

    for fold in folds:
        if dataset == "0.1-traj-pcs":
            x, y = re.findall(r"(.*)_(.*)", fold.name)[0]
            trajs = [p for p in fold.glob("*") if p.is_dir()]
            for traj in trajs:
                action = torch.from_numpy(np.load(traj / "actions.npy"))
                actions.append(action)
                pos = torch.from_numpy(np.load(traj / "states.npy"))
                positions.append(pos)

                for tidx in range(seq_len):
                    img_path = traj / f"{tidx}.png"
                    image = Image.open(img_path)
                    image = Normalize([121.6697 / 255, 149.3242 / 255, 154.9510 / 255],
                                      [40.7521 / 255, 47.7267 / 255, 103.2739 / 255])(ToTensor()(image))
                    images.append(image)
        else:
            base = torch.empty(50, 3, 64, 64)
            if dataset in ["pc-images", "small-world-random-pc"]:
                action = torch.from_numpy(np.load(fold / "actions.npy"))
                if len(action) == 0:
                    continue
                actions.append(action)
                positions.append(torch.from_numpy(np.load(fold / "states.npy")))
            else:
                action = torch.from_numpy(np.load(fold / "actions.npz")["arr_0"])
                position = torch.from_numpy(np.load(fold / "state.npz")["arr_0"])
                rolls = np.random.randint(0, 50, size=8)
                for r in rolls:
                    rolled_action = torch.roll(action, -r, dims=0)[:seq_len]
                    rolled_position = torch.roll(position, -r, dims=0)[seq_len - 1]
                    actions.append(rolled_action)
                    positions.append(rolled_position)

            for tidx in range(17):
                img_path = fold / f"{tidx}.png"
                if not img_path.exists():
                    print(f"Missing image {img_path}")
                    continue
                image = Image.open(img_path)
                image = Normalize([121.6697 / 255, 149.3242 / 255, 154.9510 / 255],
                                  [40.7521 / 255, 47.7267 / 255, 103.2739 / 255])(ToTensor()(image))
                base[tidx] = image

            if dataset not in ["pc-images", "small-world-random-pc"]:
                for r in rolls:
                    rolled_seq = torch.roll(base, -r, dims=0)[:seq_len]
                    images.append(rolled_seq)

    images = torch.stack(images, dim=0)
    positions = torch.stack(positions, dim=0)
    actions = torch.stack(actions, dim=0)

    '''
    Load model
    '''
    model = VestibularCoder(
        in_channels=3,
        out_channels=3,
        layers=[2, 2, 2, 2],
        seq_len=seq_len,
        num_actions=2
    )
    ckpt_path = weights_dir / experiment / "best.ckpt"
    print(f"Loading model from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    model = model.to(args.device)

    '''
    Collect latents
    '''
    latents = []
    locations = []
    bsz = 100

    for idx in range(len(images) // bsz + 1):
        batch = images[bsz * idx:bsz * (idx + 1)].to(args.device)
        batch_actions = actions[bsz * idx:bsz * (idx + 1)].to(args.device)
        batch_positions = positions[bsz * idx:bsz * (idx + 1)]
        
        # If the model is trained with a fixed head direction, adjust here
        if len(set_tuple) == 2:
            # Example: for the "vestibular non head turn models"
            batch_actions = batch_actions[:, :, [0, 2]]

        if len(batch) == 0:
            break

        with torch.no_grad():
            feats = model.get_latents(batch, actions=batch_actions.float(), which=args.which_residual)[:, -1]
            latents.append(feats.cpu())

        locations.append(batch_positions)

    gc.collect()
    torch.cuda.empty_cache()

    latents = torch.cat(latents, dim=0)
    locations = torch.cat(locations, dim=0)[:, :2]

    # Normalize positions
    y0_mean = locations[:, 0].mean()
    y0_std = locations[:, 0].std()
    y1_mean = locations[:, 1].mean()
    y1_std = locations[:, 1].std()

    locations[:, 0] = (locations[:, 0] - y0_mean) / y0_std
    locations[:, 1] = (locations[:, 1] - y1_mean) / y1_std

    # Shuffle and split
    shuffle_idx = torch.randperm(len(latents))
    latents = latents[shuffle_idx]
    locations = locations[shuffle_idx]

    val_latents = latents[:400]
    train_latents = latents[400:]
    val_positions = locations[:400]
    train_positions = locations[400:]

    '''
    Train position probe
    '''
    net = PositionProbe().to(args.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    for epoch in tqdm(range(args.epochs)):
        batch_idx = np.arange(0, len(train_latents))
        np.random.shuffle(batch_idx)
        # Chop it into rows of batch_size
        batch_idx = batch_idx[: len(batch_idx) // args.batch_size * args.batch_size]
        batch_idx = batch_idx.reshape(-1, args.batch_size)

        model.train()
        for it, idx_ in enumerate(batch_idx):
            optimizer.zero_grad()
            batch = train_latents[idx_].float().to(args.device)
            pos = train_positions[idx_, :2].float().to(args.device)
            pred = net(batch)
            loss = F.mse_loss(pred, pos)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = net(val_latents.to(args.device)).cpu()
            val_loss = F.mse_loss(pred, val_positions)
            print(f"Epoch {epoch}, val MSE loss: {val_loss.item():.6f}", end='\r')

        scheduler.step()

    '''
    Evaluate and save
    '''
    with torch.no_grad():
        batch = val_latents.to(args.device)
        pc_pred = net(batch).cpu()
        # Undo normalization
        pc_pred[:, 0] = pc_pred[:, 0] * y0_std + y0_mean
        pc_pred[:, 1] = pc_pred[:, 1] * y1_std + y1_mean

        validation = torch.clone(val_positions)
        validation[:, 0] = validation[:, 0] * y0_std + y0_mean
        validation[:, 1] = validation[:, 1] * y1_std + y1_mean
        pc = torch.linalg.norm(pc_pred - validation, dim=1).numpy()

    print(f"\nFinished processing {experiment}-residual-{args.which_residual}")
    
    out_path = weights_dir / f"{experiment}/position_errors-residual-{args.which_residual}-nonlinear-multiview.npy"
    net_path = weights_dir / f"{experiment}/position-decoder-residual-{args.which_residual}-nonlinear-multiview.pth"

    np.save(out_path, pc)
    torch.save(net.state_dict(), net_path)
    print(f"Saved position errors to {out_path}")
    print(f"Saved position probe model to {net_path}")

if __name__ == "__main__":
    main()
