# From Luo et al, edited to include predictive coding files

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
import seaborn as sns
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image \
    import load_img, img_to_array
from tqdm import tqdm
import utils
import models

def generate_random_data(
        data_path,
        movement_mode,
        x_min,
        x_max,
        y_min,
        y_max,
        multiplier,
        n_rotations,
        random_seed,
        image_shape=(224, 224, 3),
    ):
    """Generate random data for testing purposes"""

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # generate random data
    np.random.seed(random_seed)

    if movement_mode == '1d':
        n_stops_x = len( range(x_min*multiplier, x_max*multiplier+1) )
        n_samples = n_stops_x
    elif movement_mode == '2d':
        n_stops_x = len( range(x_min*multiplier, x_max*multiplier+1) )
        n_stops_y = len( range(y_min*multiplier, y_max*multiplier+1) )
        n_samples = (n_stops_x * n_stops_y) * n_rotations
    print(f'n_samples: {n_samples}')

    batch_x = np.random.rand(n_samples, *image_shape)
    
    # save random data as png
    for i in range(n_samples):
        img = Image.fromarray( (batch_x[i]*255).astype('uint8') )
        img.save(f'{data_path}/{i}.png')


def load_preprocessed_data(
        config,
        data_path, 
        movement_mode,
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        multiplier,
        n_rotations,
        preprocess_func=None,
        color_mode='rgb', 
        target_size=(224, 224), 
        image_shape=(224, 224, 3),
        interpolation='nearest',
        data_format='channels_last',
        dtype=K.floatx(),
    ):
    if movement_mode == '1d':
        n_stops_x = len( range(env_x_min*multiplier, env_x_max*multiplier+1) )
        n_samples = n_stops_x

    elif movement_mode == '2d':
        #n_stops_x = len( range(env_x_min*multiplier, env_x_max*multiplier+1) )
        #n_stops_y = len( range(env_y_min*multiplier, env_y_max*multiplier+1) )
        #n_samples = (n_stops_x * n_stops_y) * n_rotations
        n_samples =  len(os.listdir(data_path))
    
    if 'vit' in config['model_name']:
        image_shape = (3, 224, 224)
        data_format = 'channels_first'
    else:
        image_shape = (224, 224, 3)
        data_format = 'channels_last'

    print(f'[Check] data loader - n_samples: {n_samples}')
    print(f'[Check] preprocessing...')
    # build batch of image data
    batch_x = np.zeros((n_samples,) + image_shape, dtype=dtype)
    for i in tqdm(range(n_samples)):
        fpath = f'{data_path}/{i}.png'
        # PIL.Image
        img = load_img(
            fpath,
            color_mode=color_mode,
            target_size=target_size,
            interpolation=interpolation)
        x = img_to_array(img, data_format=data_format)

        # Pillow images should be closed after `load_img`,
        # but not PIL images.
        if hasattr(img, 'close'):
            img.close()
        if preprocess_func:
            if 'vit' in config['model_name']:
                x = preprocess_func(
                        x, return_tensors="tf"
                    )['pixel_values']
            else:
                x = preprocess_func(x)
        batch_x[i] = x
    
    print(f'[Check] data loader - batch_x.shape: {batch_x.shape}')
    return batch_x


def load_decoding_targets(
        movement_mode,
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        multiplier,
        n_rotations
    ):
    """
    Produce coordinates as targets for training/testing

    return:
        A list of lists, where each sublist is a coordinate
        corresponding to a frame and a rotation.
    """

    targets_true = []
    if movement_mode == '1d':
        NotImplementedError()

    elif movement_mode == '2d':
        # same idea as generating the frames in Unity
        # so we get decimal coords in between the grid points
        for i in range(env_x_min*multiplier, env_x_max*multiplier+1):
            for j in range(env_y_min*multiplier, env_y_max*multiplier+1):
                for k in range(n_rotations):
                    targets_true.append([i/multiplier, j/multiplier, k])
    
    return np.array(targets_true)


def load_decoding_targets_border_distance(
        movement_mode,
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        multiplier,
        n_rotations
    ):
    """
    Produce distance to nearest wall as targets for training/testing

    return:
        A list of lists, where each sublist is the nearest distance
        to a wall. Notice, given a location, the nearest distance
        to a wall is computed regardless of the direction of the
        agent. In other words, while there might be multiple walls 
        with the same distance to the agent, the nearest distance 
        should still be unique and the same.
    """
    targets_true = []
    if movement_mode == '1d':
        raise NotImplementedError()

    elif movement_mode == '2d':
        # The 2D env is a square grid (X-Z plane in Unity)
        # Both X and Y axes are from -5 to 5 (inc. walls).
        # The agent's furthest reachable position on the grid 
        # are between [-env_x_min, env_x_max] inclusive along each axis.
        # More specifically, the agent can move to any position
        # on each axis in range(env_x_min*multiplier, env_x_max*multiplier+1), if x-axis.
        # The agent also rotates itself a number of times.

        # For a given position, the nearest distance to a wall can be computed 
        # and is unique regardless of the direction of the agent (there might be multiple
        # walls with the same distance to the agent, but the nearest distance should still
        # be unique and the same).

        # We compute and collect the nearest distance to a wall for each position
        # on the grid and for each rotation of the agent.
        env_x_min_wall = env_x_min - 1
        env_x_max_wall = env_x_max + 1
        env_y_min_wall = env_y_min - 1
        env_y_max_wall = env_y_max + 1
        totalLen = env_x_max_wall - env_x_min_wall
        for i in range(env_x_min*multiplier, env_x_max*multiplier+1):
            for j in range(env_y_min*multiplier, env_y_max*multiplier+1):
                for k in range(n_rotations):
                    # along each axis, we compute the distance to the nearest wall
                    # and take the minimum of the two.
                    nearest_wall_x = min(
                        abs(i/multiplier - env_x_min_wall), 
                        abs(i/multiplier - env_x_max_wall)
                    )
                    nearest_wall_y = min(
                        abs(j/multiplier - env_y_min_wall), 
                        abs(j/multiplier - env_y_max_wall)
                    )
                    nearest_wall = min(nearest_wall_x, nearest_wall_y)
                    targets_true.append([nearest_wall])

    return np.array(targets_true)


def load_full_dataset_model_reps(config, model, preprocessed_data, batch_size=32, batch_saving=False):
    """
    Generic function to produce model representations
    for the full dataset (before train/test split),
    using batch processing to avoid OOM errors.
    """
    print(f'[Check] producing model representations...')
    
    if config['model_name'] == 'none':
        model_reps = preprocessed_data.reshape(preprocessed_data.shape[0], -1)
        print(f'raw image input shape: {model_reps.shape}')
        return model_reps

    env = config['unity_env']
    mode = config['movement_mode']
    name = config['model_name']
    layer = config['output_layer']
    p = f'results/{env}/{mode}/uniform/{name}/{layer}'
    
    if os.path.exists(f'{p}/model_reps.npy'):
        print("Found saved model reps")
        return np.load(f'{p}/model_reps.npy')

    # Create a temporary directory to store batch files
    if batch_saving:
        print("batch saving")
        temp_dir = f'results/temp_model_reps/{env}/{name}/{layer}'
        os.makedirs(temp_dir, exist_ok=True)

    total_samples = preprocessed_data.shape[0]
    
    if batch_saving:
        model_reps = []
        for i in tqdm(range(0, total_samples, batch_size)):
            batch = preprocessed_data[i:i+batch_size]
            
            if 'vit' in config['model_name']:
                if config['model_name'] in ['vit_b16', 'vit_b16_untrained']:
                    layer_index = int(config['output_layer'][6:])
                    batch_reps = model(
                        batch, training=False, 
                        output_hidden_states=True
                    ).hidden_states[layer_index].numpy()
            else:
                batch_reps = model.predict(batch, verbose=0)
            
            if len(batch_reps.shape) > 2:
                batch_reps = batch_reps.reshape(batch_reps.shape[0], -1)
                
            model_reps.append(batch_reps)
            
            if i % (batch_size * 20) == 0 or i + batch_size >= total_samples:   
                print(f'Saving batch {i}')
                # Save batch representations to disk
                model_reps = np.concatenate(model_reps, axis=0)
                np.save(os.path.join(temp_dir, f'batch_{i + len(model_reps)}.npy'), model_reps)
                del model_reps
                model_reps = []
        
        # Clear model from memory
        del model
        K.clear_session()

        # Now, concatenate all batches into a single file
        print("Concatenating all batches into a single file...")
        concatenate_large_arrays(temp_dir, f'{p}/model_reps.npy')

        # Optionally, remove the temporary batch files
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        
        return None

    else:
        model_reps = []
        for i in tqdm(range(0, total_samples, batch_size)):
            batch = preprocessed_data[i:i+batch_size]
            
            if 'vit' in config['model_name']:
                if config['model_name'] in ['vit_b16', 'vit_b16_untrained']:
                    layer_index = int(config['output_layer'][6:])
                    batch_reps = model(
                        batch, training=False, 
                        output_hidden_states=True
                    ).hidden_states[layer_index].numpy()
            else:
                batch_reps = model.predict(batch, verbose=0)
            
            if len(batch_reps.shape) > 2:
                batch_reps = batch_reps.reshape(batch_reps.shape[0], -1)
                
            model_reps.append(batch_reps)
        
        # Clear model from memory
        del model
        K.clear_session()

        model_reps = np.concatenate(model_reps, axis=0)
        print("Concatenated")
        os.makedirs(p, exist_ok=True)
        np.save(f'{p}/model_reps.npy', model_reps)
        return model_reps

def concatenate_large_arrays(directory, output_file):
    """
    Concatenate all .npy files in the given directory into a single large .npy file.
    Uses memory mapping for efficiency, but saves the final result in a standard numpy format.
    """
    batch_files = sorted([f for f in os.listdir(directory) if f.endswith('.npy')],
                         key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Determine total shape and dtype
    sample_array = np.load(os.path.join(directory, batch_files[0]))
    dtype = sample_array.dtype
    shape = list(sample_array.shape)
    shape[0] = 0
    del sample_array

    for batch_file in batch_files:
        batch = np.load(os.path.join(directory, batch_file))
        shape[0] += batch.shape[0]
        del batch

    # Create a temporary memmap file
    temp_memmap_file = output_file + '.temp'
    mmap_array = np.memmap(temp_memmap_file, dtype=dtype, mode='w+', shape=tuple(shape))

    # Copy data from each file to the memory-mapped array
    start_idx = 0
    for i, batch_file in enumerate(batch_files):
        batch = np.load(os.path.join(directory, batch_file))
        end_idx = start_idx + batch.shape[0]
        mmap_array[start_idx:end_idx] = batch
        start_idx = end_idx
        del batch
        print(f"Processed file {i+1}/{len(batch_files)}: {batch_file}")

    mmap_array.flush()

    # Save the memmap array to a standard numpy file
    print(f"Saving final array to {output_file}...")
    np.save(output_file, mmap_array)

    # Clean up
    del mmap_array
    os.remove(temp_memmap_file)

    print(f"All data concatenated and saved to {output_file}")
    print(f"Final array shape: {tuple(shape)}")


def load_model_layers(model_name):
    """
    Given `model_name`, return the main
    layers of the model that are of interest
    for analysis.
    """
    models_and_layers = {
        'simclrv2_r50_1x_sk0':
            [
                'block_group1',
                'block_group2',
                'block_group4',
                'final_avg_pool',
            ],
        'vgg16':
            [   
                'block2_pool',
                'block4_pool',
                'block5_pool',
                'fc2',
            ],
        'vgg16_untrained':
            [
                'block2_pool',
                'block4_pool',
                'block5_pool',
                'fc2',
            ],
        'resnet50':
            [
                'conv2_block3_out',
                'conv4_block6_out',
               'conv5_block2_out',
                'avg_pool',
            ],
        'resnet50_untrained':
            [
                'conv2_block3_out',
                'conv4_block6_out',
                'conv5_block2_out',
                'avg_pool',
            ],
        'vit_b16':
            [   
                'layer_3',
                'layer_6',
                'layer_9',
                'layer_12',
            ],
        'vit_b16_untrained':
            [
                'layer_3',
                'layer_6',
                'layer_9',
                'layer_12',
            ],
        'predictive-coding-0.1-0.1':
        [
            'block2_pool',
            'final',
            'post_pred'
        ],
        'predictive-coding-0.9-0.9':
        [
            'block2_pool',
            'final',
            'post_pred'
        ],
        'predictive-coding-0.1-0.1-0.0':
        [
            'block2_pool',
            'final',
            'post_pred'
        ],
        'predictive-coding-small-world-new':
        [
            'block2_pool',
            'final',
            'post_pred'
        ]
    }
    return models_and_layers[model_name]
    

def load_envs_dict(model_name, envs):
    model_layers = load_model_layers(model_name)
    # gradient cmap in warm colors in a list
    # cmaps = sns.color_palette("Reds", len(model_layers)).as_hex()[::-1]
    cmaps = ['#FFDCA2', '#FF9799', '#C63264', '#4E0362']
    if len(envs) == 1:
        prefix = f'{envs[0]}'
    else:
        raise NotImplementedError
        # TODO: 
        # 1. env28 is not flexible.
        # 2. cannot work with across different envs (e.g. decorations.)

    envs_dict = {}
    for output_layer in model_layers:
        envs_dict[
            f'{prefix}_2d_{model_name}_{output_layer}'
        ] = {
            'name': f'{prefix}',
            'n_walls': 4,
            'output_layer': output_layer,
            'color': cmaps.pop(0),
        }
    return envs_dict


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    envs = ['biggest_world_r17']
    model_name = 'resnet50'

    envs_dict = load_envs_dict(model_name, envs)
    config_versions=list(envs_dict.keys())

    for cv in config_versions:
        config = utils.load_config(cv)
    
        model, preprocess_func = models.load_model(
                config['model_name'], config['output_layer'])
        
        preprocessed_data = load_preprocessed_data(
            config=config,
            data_path=\
                f"data/minecraft/{config['unity_env']}",
                #f"{config['movement_mode']}", 
            movement_mode=config['movement_mode'],
            env_x_min=config['env_x_min'],
            env_x_max=config['env_x_max'],
            env_y_min=config['env_y_min'],
            env_y_max=config['env_y_max'],
            multiplier=config['multiplier'],
            n_rotations=config['n_rotations'],
            preprocess_func=preprocess_func,
        )
        
        load_full_dataset_model_reps(config, model, preprocessed_data, batch_saving=False)
    
    #targets_true = load_decoding_targets_border_distance(
    #    movement_mode='2d',
    #    env_x_min=-4,
    #    env_x_max=4,
    #    env_y_min=-4,
    #    env_y_max=4,
    #    multiplier=2,
    #    n_rotations=24
    #)
