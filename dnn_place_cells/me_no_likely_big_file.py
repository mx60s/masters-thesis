import argparse
from tqdm import tqdm
import numpy as np
import os

def memory_efficient_ablation(input_file, output_file, keep_ratio=0.5, chunk_size=1000):
    """
    Randomly sample a set of DNN activations.
    """
    array = np.load(input_file, mmap_mode='r')
    shape = array.shape
    dtype = array.dtype

    n_units_to_keep = int(shape[1] * keep_ratio)

    units_to_keep = np.random.choice(shape[1], n_units_to_keep, replace=False)
    units_to_keep.sort()

    with open(output_file, 'wb') as f:
        np.lib.format.write_array_header_1_0(f, np.lib.format.header_data_from_array_1_0(np.empty((shape[0], n_units_to_keep), dtype=dtype)))

        for i in tqdm(range(0, shape[0], chunk_size)):
            chunk = array[i:i+chunk_size, units_to_keep]
            chunk.tofile(f)

    print(f"Ablated array saved to {output_file}")
    print(f"New shape: {(shape[0], n_units_to_keep)}")

def main():
    parser = argparse.ArgumentParser(description="Perform memory-efficient ablation on DNN activations.")
    parser.add_argument("input_file", type=str, help="Path to the input .npy file.")
    parser.add_argument("output_file", type=str, help="Path to save the ablated .npy file.")
    parser.add_argument("--keep_ratio", type=float, default=0.5, help="Fraction of units to keep (default: 0.5).")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of data chunks to process at a time (default: 1000).")

    args = parser.parse_args()

    memory_efficient_ablation(args.input_file, args.output_file, args.keep_ratio, args.chunk_size)

    try:
        loaded_array = np.load(args.output_file)
        print(f"Successfully loaded the ablated array. Shape: {loaded_array.shape}")
    except Exception as e:
        print(f"Error loading the ablated array: {e}")

if __name__ == "__main__":
    main()
