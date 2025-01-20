from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def _unit_chart_type_classification(unit_chart_info):
    """
    Given a unit_chart_info, classify the units into different types,
    and return the indices of units by type or combo of types.
    """
    dead_units_indices = []
    max_num_clusters = np.max(unit_chart_info[:, 1])  # global max used for setting xaxis.
    num_clusters = np.zeros(max_num_clusters+1)
    cluster_sizes = []
    cluster_peaks = []
    border_cell_indices = []
    place_cells_indices = []
    direction_cell_indices = []
    active_no_type_indices = []

    for unit_index in range(unit_chart_info.shape[0]):
        if unit_chart_info[unit_index, 0] == 0:
            dead_units_indices.append(unit_index)
        else:
            num_clusters[int(unit_chart_info[unit_index, 1])] += 1
            cluster_sizes.extend(unit_chart_info[unit_index, 2])
            cluster_peaks.extend(unit_chart_info[unit_index, 3])

            if unit_chart_info[unit_index, 1] > 0:
                place_cells_indices.append(unit_index)
                is_place_cell = True
            else:
                is_place_cell = False

            if unit_chart_info[unit_index, 10] > 0.47:
                direction_cell_indices.append(unit_index)
                is_direction_cell = True
            else:
                is_direction_cell = False

            if unit_chart_info[unit_index, 9] > 0.5:
                border_cell_indices.append(unit_index)
                is_border_cell = True
            else:
                is_border_cell = False

            if not (is_place_cell or is_direction_cell or is_border_cell):
                active_no_type_indices.append(unit_index)

    # plot
    n_dead_units = len(dead_units_indices)
    n_active_units = unit_chart_info.shape[0] - n_dead_units

    # Collect the indices of units that are all three types
    # (place + border + direction)
    place_border_direction_cells_indices = \
        list(set(place_cells_indices) & set(border_cell_indices) & set(direction_cell_indices))
    
    # Collect the indices of units that are two types (inc. three types)
    # (place + border cells)
    # (place + direction cells)
    # (border + direction cells)
    place_and_border_cells_indices = \
        list(set(place_cells_indices) & set(border_cell_indices))
    place_and_direction_cells_indices = \
        list(set(place_cells_indices) & set(direction_cell_indices))
    border_and_direction_cells_indices = \
        list(set(border_cell_indices) & set(direction_cell_indices))
    
    # Collect the indices of units that are only two types
    # (place  + border - direction),
    # (place  + direction   - border),
    # (border + direction   - place)
    place_and_border_not_direction_cells_indices = \
        list(set(place_and_border_cells_indices) - set(place_border_direction_cells_indices))
    place_and_direction_not_border_cells_indices = \
        list(set(place_and_direction_cells_indices) - set(place_border_direction_cells_indices))
    border_and_direction_not_place_cells_indices = \
        list(set(border_and_direction_cells_indices) - set(place_border_direction_cells_indices))
    
    # Collect the indices of units that are exclusive 
    # place cells, 
    # border cells, 
    # direction cells
    exclusive_place_cells_indices = \
        list(set(place_cells_indices) - (set(place_and_border_cells_indices) | set(place_and_direction_cells_indices)))
    exclusive_border_cells_indices = \
        list(set(border_cell_indices) - (set(place_and_border_cells_indices) | set(border_and_direction_cells_indices)))
    exclusive_direction_cells_indices = \
        list(set(direction_cell_indices) - (set(place_and_direction_cells_indices) | set(border_and_direction_cells_indices)))

    results =  {
        'dead_units_indices': dead_units_indices,
        'place_border_direction_cells_indices': place_border_direction_cells_indices,
        'place_and_border_not_direction_cells_indices': place_and_border_not_direction_cells_indices,
        'place_and_direction_not_border_cells_indices': place_and_direction_not_border_cells_indices,
        'border_and_direction_not_place_cells_indices': border_and_direction_not_place_cells_indices,
        'exclusive_place_cells_indices': exclusive_place_cells_indices,
        'exclusive_border_cells_indices': exclusive_border_cells_indices,
        'exclusive_direction_cells_indices': exclusive_direction_cells_indices,
        'active_no_type_indices': active_no_type_indices,
    }
    
    assert unit_chart_info.shape[0] == sum([len(v) for v in results.values()])

    # Check all values are mutually exclusive
    for key, value in results.items():
        for key2, value2 in results.items():
            if key != key2:
                assert len(set(value) & set(value2)) == 0, f'{key} and {key2} have common elements'

    return results

if __name__ == '__main__':
    model_specifiers = {
        #'resnet50' : ['conv4_block6_out'],
        'vit_b16' : ['layer_3', 'layer_6', 'layer_9', 'layer_12']
    }
    
    env = 'small_world_r17'
    to_plot = 200

    for model_name, layer_names in model_specifiers.items():
        for layer_name in layer_names:
            print(f'Plotting marginal tuning for {model_name}, {layer_name}')
            
            fig = plt.figure(figsize=(20, 5))
            gs = fig.add_gridspec(
                nrows=1, ncols=1
            )
            
            results_path = f'/home/mag/Space/results/{env}/2d/uniform/{model_name}/inspect_units/unit_chart/{layer_name}'
            unit_chart_info = np.load(f'{results_path}/unit_chart.npy', allow_pickle=True)
            
            unit_indices_by_types = _unit_chart_type_classification(unit_chart_info)
                
            n_exc_place_cells = len(unit_indices_by_types["exclusive_place_cells_indices"])
            n_exc_border_cells = len(unit_indices_by_types["exclusive_border_cells_indices"])
            n_exc_direction_cells = len(unit_indices_by_types["exclusive_direction_cells_indices"])
            n_place_and_border_not_direction_cells = len(unit_indices_by_types["place_and_border_not_direction_cells_indices"])
            n_place_and_direction_not_border_cells = len(unit_indices_by_types["place_and_direction_not_border_cells_indices"])
            n_border_and_direction_not_place_cells = len(unit_indices_by_types["border_and_direction_not_place_cells_indices"])
            n_place_border_direction_cells = len(unit_indices_by_types["place_border_direction_cells_indices"])
            n_active_no_type_cells = len(unit_indices_by_types["active_no_type_indices"])
            
            sum_n_cells = \
                n_exc_place_cells + \
                n_exc_border_cells + \
                n_exc_direction_cells + \
                n_place_and_border_not_direction_cells + \
                n_place_and_direction_not_border_cells + \
                n_border_and_direction_not_place_cells + \
                n_place_border_direction_cells + \
                n_active_no_type_cells
            
            labels = []
            n_cells = []
            
            # collect the number of cells for each type
            # only if n of cells > 0 and collect the labels
            if n_exc_place_cells/sum_n_cells >= 0.01:
                n_cells.append(n_exc_place_cells)
                labels.append('P')
            
            if n_exc_border_cells/sum_n_cells >= 0.01:
                n_cells.append(n_exc_border_cells)
                labels.append('B')
            
            if n_exc_direction_cells/sum_n_cells >= 0.01:
                n_cells.append(n_exc_direction_cells)
                labels.append('D')
            
            if n_place_and_border_not_direction_cells/sum_n_cells >= 0.01:
                n_cells.append(n_place_and_border_not_direction_cells)
                labels.append('P+B')
            
            if n_place_and_direction_not_border_cells/sum_n_cells >= 0.01:
                n_cells.append(n_place_and_direction_not_border_cells)
                labels.append('P+D')
            
            if n_border_and_direction_not_place_cells/sum_n_cells >= 0.01:
                n_cells.append(n_border_and_direction_not_place_cells)
                labels.append('B+D')
            
            if n_place_border_direction_cells/sum_n_cells >= 0.01:
                n_cells.append(n_place_border_direction_cells)
                labels.append('P+B+D')
                
            if n_active_no_type_cells/sum_n_cells >= 0.01:
                n_cells.append(n_active_no_type_cells)
                labels.append('Active (no type)')
                
            # And lastly, the dead units
            n_cells.append(len(unit_indices_by_types["dead_units_indices"]))
            labels.append('Inactive')
                
            # make sure plt.cm.Pastel1.colors are consistent across layers
            # for each type of cells.
            colors = []
            for label in labels:
                if label == 'P':
                    colors.append(plt.cm.Pastel1.colors[1])
                elif label == 'B':
                    colors.append(plt.cm.Pastel1.colors[0])
                elif label == 'D':
                    colors.append(plt.cm.Pastel1.colors[2])
                elif label == 'P+B':
                    colors.append(plt.cm.Pastel1.colors[3])
                elif label == 'P+D':
                    colors.append(plt.cm.Pastel1.colors[4])
                elif label == 'B+D':
                    colors.append(plt.cm.Pastel1.colors[5])
                elif label == 'P+B+D':
                    colors.append(plt.cm.Pastel1.colors[6])
                elif label == 'Inactive':
                    colors.append("grey")
                elif label == 'Active (no type)':
                    colors.append(plt.cm.Pastel1.colors[7])
            
            ax = fig.add_subplot(gs[0])
            
            # Calculate percentages and exclude labels with 0 percentage
            total_cells = sum(n_cells)
            percentages = [round((cell / total_cells) * 100) for cell in n_cells]
            filtered_labels = [
                label if percentage > 0 else '' \
                    for label, percentage in zip(labels, percentages)
            ]
            
            ax.pie(
                n_cells,
                autopct=lambda p: '{:.0f}'.format(round(p)) if p >= 1 else '',
                labels=filtered_labels,
                colors=colors,
                explode=[0.1]*len(labels),
                textprops={'fontsize': 14},
            )
        
            if model_name == 'vgg16':
                model_name_plot = f'VGG-16 Small World'
                
                if layer_name == 'block2_pool':
                    output_layer_plot = 'Early (block2_pool)'
                elif layer_name == 'block4_pool':
                    output_layer_plot = 'Mid (block4_pool)'
                elif layer_name == 'block5_pool':
                    output_layer_plot = 'Late (block5_pool)'
                elif layer_name == 'fc2':
                    output_layer_plot = 'Penultimate (fc2)'
            
            elif model_name == 'resnet50':
                model_name_plot = f'ResNet-50 {env}' # 'conv2_block3_out', 'conv5_block2_out', 'avg_pool'
                
                if layer_name == 'conv2_block3_out':
                    output_layer_plot = 'Early (conv2_block3_out)'
                elif layer_name == 'conv4_block6_out':
                    output_layer_plot = 'Mid (conv4_block6_out)'
                elif layer_name == 'conv5_block2_out':
                    output_layer_plot = 'Late (conv5_block2_out)'
                elif layer_name == 'avg_pool':
                    output_layer_plot = 'Penultimate (avg_pool)'
        
            elif model_name == 'vit_b16':
                model_name_plot = f'ViT b16 {env}'
                
                if layer_name == 'layer_3':
                    output_layer_plot = '(Early) Layer 3'
                elif layer_name == 'layer_6':
                    output_layer_plot = '(Middle) Layer 6'
                elif layer_name == 'layer_9':
                    output_layer_plot = '(Late) Layer 9'
                elif layer_name == 'layer_12':
                    output_layer_plot = '(Penultimate) Layer 12'
                        
            elif model_name[:17] == 'predictive-coding':
                model_name_plot = f'Predictive Coder 0.1-0.1-0.0'
            
                # fix
                if layer_name == 'block2_pool':
                    output_layer_plot = 'Early (block2_pool)'
                elif layer_name == 'final':
                    output_layer_plot = 'Encoder Output'
                elif layer_name == 'post_pred':
                    output_layer_plot = 'Post-Prediction'
                else:
                    output_layer_plot = layer_name
            else:
                output_layer_plot = '??'
            
            ax.set_title(f'{output_layer_plot}', fontweight='bold', fontsize=16)
        
            # remove left, top and right
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # make subplot space wider
            plt.subplots_adjust(wspace=0.8)
        
            save_path = f'figs/{env}/2d/uniform/{model_name}/marginal_tuning/{layer_name}_place_cells_oumc.pdf'
        
            with PdfPages(save_path) as pdf:
                pdf.savefig(fig)
                plt.close(fig)
                
                index_key = 'exclusive_place_cells_indices'
                
                results_path = f'/home/mag/Space/results/{env}/2d/uniform/{model_name}'
                
                model_reps = np.load(f'{results_path}/{layer_name}/model_reps.npy')
                
                if model_reps.shape[1] > 8000:
                    sampled_indices = np.load(f'/home/mag/Space/results/sample_indices_{model_name}_{layer_name}.npy')
                    model_reps = model_reps[:, sampled_indices]
                
                n_bins = 17
                n_rotations = 17
                
                for i in tqdm(range(to_plot)):
                    idx = unit_indices_by_types[index_key][i]
                    heatmap = model_reps[:, idx].reshape((n_bins, n_bins, n_rotations))
                    x_marginal = np.sum(heatmap, axis=0)
                    y_marginal = np.sum(heatmap, axis=1)
            
                    summed_rotation_map = np.sum(heatmap, axis=2)
                    
                    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
                
                    img = axs[0].imshow(x_marginal, origin='lower', aspect='auto', extent=[0, n_bins, 0, n_rotations], cmap='hot')
                    fig.colorbar(img, ax=axs[0], label='Marginal Unit Activation')
                    axs[0].set_xlabel('X Position')
                    axs[0].set_ylabel('Rotation Index')
                    
                    img = axs[1].imshow(y_marginal, origin='lower', aspect='auto', extent=[0, n_bins, 0, n_rotations], cmap='hot')
                    fig.colorbar(img, ax=axs[1], label='Marginal Unit Activation')
                    axs[1].set_xlabel('Y Position')
                    axs[1].set_ylabel('Rotation Index')
            
                    img = axs[2].imshow(summed_rotation_map, extent=[0, n_bins, 0, n_bins],
                               origin='lower', aspect='auto', cmap='hot')
                    fig.colorbar(img, label='Summed Unit Activation')
                    axs[2].set_xlabel('X Position')
                    axs[2].set_ylabel('Y Position')
            
                    fig.suptitle(f'Neuron {idx}', fontsize=12)
                    
                    plt.tight_layout(h_pad=3.0)
                    pdf.savefig(fig)
                    plt.close(fig)
