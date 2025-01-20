#!/usr/bin/env python

import argparse
import os
import re
import gc
from glob import glob
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#from predictive_coding.analysis import *
from predictive_coding.src import PredictiveCoder, VestibularCoder, Autoencoder

from torchvision.models import vgg16, VGG16_Weights

from torchvision.transforms import ToTensor, Normalize

def main():
    parser = argparse.ArgumentParser(description="Predictive coding analysis + linear probe script (scikit-learn).")
    parser.add_argument("--dataset", type=str, default="env-pcs",
                        help="Name of the dataset directory. Examples: 'small-world-random', 'env-pcs', '0.1-traj-pcs', 'pc-images'.")
    parser.add_argument("--model_set", type=str, default="0.5-0.5",
                        help="Comma- or dash-separated floats specifying model config, e.g. '0.5-0.5' or '0.1,0.9'.")
    parser.add_argument("--seq_len", type=int, default=20,
                        help="Sequence length the model expects.")
    parser.add_argument("--num_actions", type=int, default=3,
                        help="Number of possible actions in the dataset.")
    parser.add_argument("--which_residual", type=int, default=2,
                        help="Which residual block to fetch latents from in the model.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Which device to run on (e.g. 'cuda:0', 'cuda:1', 'cpu').")
    parser.add_argument("--experiment_name", type=str, default="",
                        help="Override the experiment name, otherwise uses 'predictive-coding-<model_set>'.")
    parser.add_argument("--model_type", type=str, default="vestibular",
                        choices=["vestibular", "predictive", "bottleneck", "vgg16", "autoencoder"],
                        help="Which model class to instantiate. E.g. 'vestibular' (VestibularCoder), 'predictive' (PredictiveCoder), etc.")
    parser.add_argument("--save_suffix", type=str, default="linear-svm",
                        help="Suffix for the saved .npy and .pkl files.")
    args = parser.parse_args()

    # Parse the model_set argument into a tuple of floats
    delimiter = ',' if ',' in args.model_set else '-'
    set_tuple = tuple(map(float, args.model_set.split(delimiter)))

    if args.experiment_name.strip():
        experiment = args.experiment_name
    else:
        experiment = f"predictive-coding-{delimiter.join(map(str, set_tuple))}"

    dataset = args.dataset
    seq_len = args.seq_len
    num_actions = args.num_actions

    np.random.seed(200)
    torch.manual_seed(200)

    '''
    Load dataset
    '''
    folds = glob(f'/home/mag/predictive-coding/datasets/{dataset}/*')
    images = []
    positions = []
    actions = []

    # TODO standardize different loading patterns
    for idx, fold in enumerate(folds):
        if dataset == '0.1-traj-pcs':
            x, y = re.findall(f'/home/mag/predictive-coding/datasets/{dataset}/(.*)_(.*)', fold)[0]
            trajs = glob(f'/home/mag/predictive-coding/datasets/{dataset}/{x}_{y}/*')
            for t, traj in enumerate(trajs):
                action = torch.from_numpy(np.load(f'{traj}/actions.npy'))
                actions.append(action)
                pos = torch.from_numpy(np.load(f'{traj}/states.npy'))
                positions.append(pos)

                for tidx in range(seq_len):
                    img_path = f'{traj}/{tidx}.png'
                    image = Image.open(img_path)
                    image = Normalize(
                        [121.6697/255, 149.3242/255, 154.9510/255],
                        [40.7521/255, 47.7267/255, 103.2739/255]
                    )(ToTensor()(image))
                    images.append(image)

        else:
            if dataset == 'pc-images' or dataset == 'small-world-random-pc':
                action = torch.from_numpy(np.load(f'{fold}/actions.npy'))
                if len(action) == 0:
                    continue
                actions.append(action)
                positions.append(torch.from_numpy(np.load(f'{fold}/states.npy')))
                base = torch.empty(50, 3, 64, 64)
                for tidx in range(50):
                    img_path = f'{fold}/{tidx}.png'
                    if not os.path.exists(img_path):
                        print(f"Missing image {img_path}")
                        continue
                    image = Image.open(img_path)
                    image = Normalize(
                        [121.6697/255, 149.3242/255, 154.9510/255],
                        [40.7521/255, 47.7267/255, 103.2739/255]
                    )(ToTensor()(image))
                    base[tidx] = image
                images.append(base[:seq_len])
            else:
                # e.g. 'env-pcs' or 'small-world-random'
                action = torch.from_numpy(np.load(f'{fold}/actions.npz')['arr_0'])
                position = torch.from_numpy(np.load(f'{fold}/state.npz')['arr_0'])

                rolls = np.random.randint(0, 50, size=8)
                for r in rolls:
                    rolled_action = torch.roll(action, -r, dims=0)[:seq_len]
                    rolled_position = torch.roll(position, -r, dims=0)[seq_len - 1]
                    actions.append(rolled_action)
                    positions.append(rolled_position)

                # Load 50 images
                base = torch.empty(50, 3, 64, 64)
                for tidx in range(50):
                    img_path = f'{fold}/{tidx}.png'
                    if not os.path.exists(img_path):
                        print(f"Missing image {img_path}")
                        continue
                    image = Image.open(img_path)
                    image = Normalize(
                        [121.6697/255, 149.3242/255, 154.9510/255],
                        [40.7521/255, 47.7267/255, 103.2739/255]
                    )(ToTensor()(image))
                    base[tidx] = image

                # Create rolled image sequences
                for r in rolls:
                    rolled_seq = torch.roll(base, -r, dims=0)[:seq_len]
                    images.append(rolled_seq)

    images = torch.stack(images, dim=0)
    positions = torch.stack(positions, dim=0)
    actions = torch.stack(actions, dim=0)

    '''
    Build/load model
    '''
    ckpt_path = f"./experiments/{experiment}/best.ckpt"
    print(f"Experiment name: {experiment}")
    print(f"Loading model from: {ckpt_path}")

    if args.model_type == "vgg16":
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)
    elif args.model_type == "autoencoder":
        model = Autoencoder(in_channels=3, out_channels=3, layers=[2, 2, 2, 2], seq_len=seq_len)
        model.load_state_dict(torch.load(ckpt_path))
    elif args.model_type == "predictive":
        model = PredictiveCoder(in_channels=3, out_channels=3, layers=[2, 2, 2, 2], seq_len=seq_len)
        model.load_state_dict(torch.load(ckpt_path))
    elif args.model_type == "vestibular":
        model = VestibularCoder(in_channels=3, out_channels=3, layers=[2, 2, 2, 2],
                                seq_len=seq_len, num_actions=num_actions)
        model.load_state_dict(torch.load(ckpt_path))
    else:
        raise ValueError(f"This script currently only supports vgg16, autoencoder, predictive coder, or vestibular coder. You specified '{args.model_type}'.")

    model.eval()
    model = model.to(args.device)

    '''
    Collect latents
    '''
    bsz = 100
    latents = []
    collected_pos = []

    for idx in range(len(images) // bsz + 1):
        batch = images[bsz * idx : bsz * (idx + 1)].to(args.device)
        batch_actions = actions[bsz * idx : bsz * (idx + 1)].to(args.device)
        batch_pos = positions[bsz * idx : bsz * (idx + 1)]

        if len(batch) == 0:
            break

        with torch.no_grad():
            if args.model_type == "vgg16":
                # just get features of the last frame in the sequence
                features = model.features(batch[:, -1])
                latents.append(features.view(features.size(0), -1).cpu())
            elif args.model_type == "autoencoder":
                features = model.get_latents(batch[:, -1])
                latents.append(features.cpu())
            else:
                # For predictive / vestibular coders:
                feats = model.get_latents(batch, actions=batch_actions.float(), which=args.which_residual)
                latents.append(feats[:, -1].cpu())  # We use the last step in the sequence

        collected_pos.append(batch_pos)

    latents = torch.cat(latents, dim=0).cpu().numpy()
    y = torch.cat(collected_pos, dim=0).cpu().numpy()[:, :2]  # Discard head direction

    print(f"Latents shape: {latents.shape}")
    print(f"Positions shape: {y.shape}")

    X = latents.reshape(latents.shape[0], -1)

    x_scaler = StandardScaler()
    X = x_scaler.fit_transform(X)

    # Normalize position data
    y0_mean, y0_std = y[:, 0].mean(), y[:, 0].std()
    y1_mean, y1_std = y[:, 1].mean(), y[:, 1].std()
    y[:, 0] = (y[:, 0] - y0_mean) / y0_std
    y[:, 1] = (y[:, 1] - y1_mean) / y1_std

    def denormalize_y(yn):
        """Given a (N,2) array of normalized positions, return them in original scale."""
        y_denorm = np.copy(yn)
        y_denorm[:, 0] = y_denorm[:, 0] * y0_std + y0_mean
        y_denorm[:, 1] = y_denorm[:, 1] * y1_std + y1_mean
        return y_denorm

    linear_svr = MultiOutputRegressor(LinearSVR(random_state=0, tol=1e-5, C=0.1, epsilon=0.1))

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_distances = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        linear_svr.fit(X_train, y_train)
        y_pred = linear_svr.predict(X_val)

        y_val_denorm = denormalize_y(y_val)
        y_pred_denorm = denormalize_y(y_pred)

        distances = np.linalg.norm(y_val_denorm - y_pred_denorm, axis=1)
        fold_distances.append(distances)

        print(f"Fold {fold+1} - Mean distance error: {distances.mean():.4f}")

    all_distances = np.concatenate(fold_distances)
    mean_distance = np.mean(all_distances)
    median_distance = np.median(all_distances)
    std_distance = np.std(all_distances)

    print("-------------------------------------------------------")
    print(f"Overall Mean distance error: {mean_distance:.4f}")
    print(f"Overall Median distance error: {median_distance:.4f}")
    print(f"Standard deviation of distance error: {std_distance:.4f}")
    print("-------------------------------------------------------")

    # Finally, fit the model on the entire dataset
    linear_svr.fit(X, y)

    # Save
    out_dir = f"./experiments/{experiment}"
    os.makedirs(out_dir, exist_ok=True)

    distances_path = os.path.join(out_dir, f"position_errors-residual-{args.which_residual}-{args.save_suffix}.npy")
    model_path = os.path.join(out_dir, f"position_decoder-residual-{args.which_residual}-{args.save_suffix}.pkl")
    scaler_path = os.path.join(out_dir, f"position_decoder-residual-{args.which_residual}-{args.save_suffix}-scaler.pkl")

    np.save(distances_path, all_distances)
    print(f"Saved cross-val distances to {distances_path}")

    with open(model_path, "wb") as f:
        pickle.dump(linear_svr, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved linear probe model to {model_path}")

    with open(scaler_path, "wb") as f:
        pickle.dump(x_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved input scaler to {scaler_path}")

if __name__ == "__main__":
    main()
