# Edited from Luo et al.
# Edits mainly include implementing the full place cell determination code from Tanni et al.

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import logging
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import scores

""" 
Stuff from Tanni et. al.
"""

from copy import deepcopy

import numpy as np
from scipy import ndimage
import cv2 as cv
from skimage.measure import regionprops

from scipy.stats import pearsonr
from tqdm import tqdm

#from barrylab_ephys_analysis.spatial.similarity import spatial_correlation

def spatial_correlation(map_1, map_2, min_included_value=0.01, mask=None, min_bins=4, bin_wise=False,
                        min_bin_samples=0):
    """Returns Pearson correlation of values in map_1 and map_2.

    Bins are ignored where any value in either map falls below `min_included_value` parameter
    or has the value `numpy.nan`

    :param numpy.ndarray map_1: shape (n_xbins, n_ybins) or (n_xbins, n_ybins, n_maps)
    :param numpy.ndarray map_2: shape (n_xbins, n_ybins) or (n_xbins, n_ybins, n_maps)
    :param float min_included_value: minimum values for bin to be included. If value is below
        this in any map, that bin is excluded computation. Default is 0.01
    :param numpy.ndarray mask: shape (n_xbins, n_ybins) boolean array specifying which bins to use
    :param int min_bins: minimum  number of bins that must remain to correlate, otherwise
        returns (numpy.nan, numpy.nan). Default is 4.
    :param bool bin_wise: if True, r-value is computed `n_maps` of `map_1` and `map_2`.
    :param int min_bin_samples: minimum number of samples per bin that must be valid for computing
        the r-value in case where `bin_wise=True`. Bins with fewer valid samples are set to `numpy.nan`.
    :return: output from :py:func:`scipy.stats.pearsonr` or numpy.ndarray of shape `map_1.shape[:-1]` with rho values
    """

    if map_1.ndim == 2:
        map_1 = map_1[:, :, np.newaxis]
    if map_2.ndim == 2:
        map_2 = map_2[:, :, np.newaxis]

    if map_1.shape[1] != map_2.shape[1]:
        raise ValueError('Number of maps does not match for map_1 {} and map_2 {}'.format(map_1.shape, map_2.shape))
    if not (mask is None) and (mask.ndim != 2):
        raise ValueError('Incorrect mask shape {}'.format(mask.shape))
    if not (mask is None) and (mask.shape[0] != map_1.shape[0] or mask.shape[1] != map_1.shape[1]):
        raise ValueError('Mask shape {} does not match with map shape {}'.format(mask.shape, map_1.shape))

    # Put mask to correct shape so broadcasting works also with stacked ratemaps
    if not (mask is None):
        mask = mask[:, :, np.newaxis]

    # Create masks based on np.nan values
    ignore_1 = np.isnan(map_1)
    ignore_2 = np.isnan(map_2)

    # Combine np.nan masks with min_included_value mask
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered in less')
        ignore_1 = np.logical_or(ignore_1, map_1 < min_included_value)
        ignore_2 = np.logical_or(ignore_2, map_2 < min_included_value)

    # Combine ignore masks of map_1 and map_2 to union ignore mask
    ignore = np.logical_or(ignore_1, ignore_2)

    # Convert ignore mask to inclusion mask and combine with mask parameter if provided
    mask = ~ignore if mask is None else np.logical_and(~ignore, mask)

    # If bin_wise is not True, return pearsonr across the whole array
    if not bin_wise:

        # If no bins are included, abort and return (np.nan, np.nan)
        if np.count_nonzero(mask) < min_bins:
            return np.nan, np.nan

        return pearsonr(map_1[mask], map_2[mask])

    # Otherwise, compute pearson correlation r-value for between the two maps along the last dimension

    r_values = np.zeros(map_1.shape[:-1], dtype=np.float32) * np.nan

    for i_x in range(r_values.shape[1]):
        for i_y in range(r_values.shape[0]):

            if np.sum(mask[i_y, i_x, :]) < min_bin_samples:
                continue

            r_values[i_y, i_x] = pearsonr(map_1[i_y, i_x, np.where(mask[i_y, i_x, :])[0]],
                                          map_2[i_y, i_x, np.where(mask[i_y, i_x, :])[0]])[0]

    return r_values


def compute_field_stability(field_ratemap, spike_rates_1, spike_rates_2,
                            min_included_value, min_bins):
    """Returns the pearson correlation of two ratemaps at location of field ratemap.
    Excludes all bins that are numpy.nan or < 0 in field ratemap.

    :param numpy.ndarray ratemap: shape (n_ybins, n_xbins) array with np.nan outside field
    :param spike_rates_1: shape (n_ybins, n_xbins) array
    :param spike_rates_2: shape (n_ybins, n_xbins) array
    :param min_included_value: minimum value in spike_rates_1 and spike_rates_2 for bin to be included
    :param min_bins: minimum number of bins that must remain to compute correlation, else returns numpy.nan
    :return: rho
    :rtype: float
    """
    field_ratemap = field_ratemap.copy()
    field_ratemap[np.isnan(field_ratemap)] = 0
    return spatial_correlation(spike_rates_1, spike_rates_2,
                               min_included_value=min_included_value,
                               mask=(field_ratemap > 0),
                               min_bins=min_bins)[0]


def primary_filter(ratemap, min_area_bins=0, min_peak_value=0):
    """Returns True if ratemap passes filter criteria, False otherwise

    :param numpy.ndarray ratemap: shape (n_ybins, n_xbins) array with np.nan outside field
    :param int min_area_bins: number of non-zero bins in ratemap required to pass
    :param float min_peak_value: minimum required maximum value in ratemap to pass
    :return: pass
    :rtype: bool
    """
    if np.count_nonzero(ratemap) < min_area_bins:
        return False
    if np.nanmax(ratemap) < min_peak_value:
        return False

    return True


def secondary_filter(ratemap, max_area_bins, stability_ratemaps,
                     min_stability, stability_kwargs):
    """Returns True if ratemap passes filter criteria, False otherwise

    :param numpy.ndarray ratemap: shape (n_ybins, n_xbins) array with np.nan outside field
    :param int max_area_bins: maximum number of bins greater than 0 in ratemap allowed to pass
    :param tuple stability_ratemaps: tuple with two ratemaps that are used for computing the stability
        of a field_ratemap after masking them with the non-numpy.nan elements in field_ratemap
    :param float min_stability: minimum required stability in stability_ratemaps to pass
    :param dict stability_kwargs: passed on to :py:func:`compute_field_stability`
    :return: pass
    :rtype: bool
    """
    if np.count_nonzero(ratemap) > max_area_bins:
        return False

    if 'min_included_value' in stability_kwargs and stability_kwargs['min_included_value'] <= 0:
        raise Exception('This module uses 0 values as indication of outside of bin areas.\n'
                        + 'Therefore, min_included_value must be above 0 for stability computation,\n'
                        + 'but is currently {}'.format(stability_kwargs['min_included_value']))
    stability = compute_field_stability(ratemap, stability_ratemaps[0],
                                        stability_ratemaps[1], **stability_kwargs)
    if np.isnan(stability) or stability < min_stability:
        return False

    return True


def create_field_ratemap(ratemap, field_idx):
    """Returns a copy of input `ratemap`, where fields not True in `field_idx` are set to `0`.

    :param numpy.ndarray ratemap: shape (n_ybins, n_xbins)
    :param numpy.ndarray field_idx: boolean array same shape as ratemap, specifying which elements
        belong to the field.
    :return: field_ratemap
    :rtype: numpy.ndarray
    """
    field_ratemap = ratemap.copy()
    field_ratemap[~field_idx] = 0

    return field_ratemap


def get_field_map(ratemap):
    """Returns the contiguous region map as identified with :py:func:`scipy.ndimage.label`.

    :param numpy.ndarray ratemap: shape (n_ybins, n_xbins)
    :return: same as input but integer identifiers marking which elements belong to which proximal group
    :rtype: numpy.ndarray
    """
    return ndimage.label(ratemap > 0)[0]


def get_filtered_subfield_ratemaps(ratemap, primary_filter_kwargs):
    """Returns a list containing a copy of ratemap for each field that passes the primary filter
    :py:func:`spatial.fields.primary_filter` where values outside the field are set `0`.

    :param numpy.ndarray ratemap: shape (n_ybins, n_xbins)
    :param dict primary_filter_kwargs: see :py:func:`spatial.field.primary_filter` keyword arguments.
        filter_kwargs is passed on to that function as `**filter_kwargs` after `ratemap` argument.
    :return: field_ratemaps
    :rtype: list
    """
    field_map = get_field_map(ratemap)

    field_nrs = np.unique(field_map)[1:]  # ignores the 0 background field

    # If no fields were detected, return no fields
    if len(field_nrs) == 0:
        return []

    field_ratemaps = []
    for field_nr in field_nrs:
        field_ratemap = create_field_ratemap(ratemap, field_map == field_nr)
        if primary_filter(field_ratemap, **primary_filter_kwargs):
            field_ratemaps.append(field_ratemap)

    return field_ratemaps


def detect_field_candidates(ratemap, base_threshold, threshold_step, primary_filter_kwargs):
    """Returns a list of field_candidates that passed primary_filter.

    The returned field_candidates list is a nested list with the following structure:
        - Each element in list contains two elements.
        - The first element is the origin index within field_candidates. This is None for first subfields,
          but int after.
        - The second element is a list of ratemaps with increasingly higher threshold, that all
          have only a single continugous region. The final element in the ratemap list is None,
          if no further subfields were found with the next threshold (none passed the primary_filter).
        - If more than one subfield is found, these are separately appended to field_candidates list following
          the same structure and using the origin index of the previous level where they were detected.

    - The field_candidates list is populated in an iterative fashion. Each iteration the ratemap threshold is increase
      by threshold_step.
    - If a single new field passes primary_filter, it is appended to the list of ratemaps for that level and
      procedure is repeated.
    - If no new fields pass primary_filter, the ratemap list is ended with None. This causes the loop to start
      the procedure on the next level in the field_candidates list.
    - New levels are added to the field_candidates list each time two or more fields are detected and pass the
      primary_filter. In that case, after the new fields are appended to field_candidates list,
      the loop moves on to next level.
    - As a result the loop extracts all possible field_candidates and also forms links between them.
      This allows working backwards through this list along the order of inheritance (field and subfield links).

    :param numpy.ndarray ratemap: shape (n_ybins, n_xbins) ratemap. Any numpy.nan elements should
        be replaced with zeros before passing ratemap to this function
    :param float base_threshold: baseline threshold level from which to start detecting fields
    :param float threshold_step: threshold shift at each iteration
    :param dict primary_filter_kwargs: see :py:func:`spatial.field.primary_filter` keyword arguments.
        filter_kwargs is passed on to that function as `**filter_kwargs` after `ratemap` argument.
    :return: field_candidates
    :rtype: list
    """

    # Use a copy of original input ratemap, threshold and threshold the ratemap.
    ratemap = ratemap.copy()
    current_threshold = deepcopy(base_threshold)
    ratemap[ratemap < current_threshold] = 0

    # Find all subfields in the ratemap at baseline threshold
    subfields = get_filtered_subfield_ratemaps(ratemap, primary_filter_kwargs)

    # If no subfields were found, return an empty list
    if len(subfields) == 0:
        return []

    # Create fields list with the initial subfields
    field_candidates = []
    for subfield in subfields:
        field_candidates.append((None, [subfield]))

    current_level = 0

    # Continuoue the loop until the last level ratemap list ends with None
    while not (field_candidates[-1][1][-1] is None):

        # If current level ratemaplist ends with None, move to next level
        if field_candidates[current_level][1][-1] is None:
            current_level += 1

        # Increase current_threshold and create a copy of current_ratemap with the current_threshold
        current_threshold += threshold_step
        current_ratemap = field_candidates[current_level][1][-1].copy()
        current_ratemap[current_ratemap < current_threshold] = 0

        # Find subfields for current level
        subfields = get_filtered_subfield_ratemaps(current_ratemap, primary_filter_kwargs)

        if len(subfields) == 0:
            # If no subfields were found, end the current level ratemap list with None
            field_candidates[current_level][1].append(None)

        elif len(subfields) == 1:
            # If a single field was found, append to current level ratemap list
            field_candidates[current_level][1].append(subfields[0])

        else:
            # If more than one field was found, append these to field_candidates list
            for subfield in subfields:
                field_candidates.append((current_level, [subfield]))
            # If more than one field was found, move to next level
            current_level += 1

    return field_candidates


def extract_fields_from_field_candidates(field_candidates, secondary_filter_kwargs):
    """Returns the field_ratemap with lowest threshold of each field_candidate that
    passes the :py:func:`secondary_filter`.

    Iterates through field_candidates in reverse order of detection in :py:func:`detect_field_candidates`.
    Ignores any field_candiate elements that were the origin of sub-fields that pass secondary_filter.

    :param list field_candidates: output from :py:func:`detect_field_candidates`
    :param dict secondary_filter_kwargs: see :py:func:`spatial.field.secondary_filter` keyword arguments.
        filter_kwargs is passed on to that function as `**filter_kwargs` after `ratemap` argument.
    :return: field_ratemaps
    :rtype: list
    """
    field_ratemap_dicts = []

    # Loop through levels of the field_candidates list starting from the last
    for current_level in range(len(field_candidates))[::-1]:

        # Get the field_candidate element for the current_level
        field_candidate = field_candidates[current_level]

        # Find out if current level field_candidate has any sub-fields already passed the secondary_filter
        n_subfields = 0
        for field_ratemap_dict in field_ratemap_dicts:
            if field_ratemap_dict['origin'] == current_level:
                n_subfields += 1

        # If more than one subfield has been identified for this field_candidate,
        # pass origin of this field_candidate to detected subfields and skip processing this field_candidate.
        if n_subfields > 1:
            for field_ratemap_dict in field_ratemap_dicts:
                if field_ratemap_dict['origin'] == current_level:
                    field_ratemap_dict['origin'] = field_candidate[0]
            continue

        field_ratemap = None

        # Loop through the ratemaps of this field_candiate
        for field_candidate_ratemap in field_candidate[1][::-1]:

            # field_candidate_ratemap lists can end in None. Ignore these elements.
            if field_candidate_ratemap is None:
                continue

            if secondary_filter(field_candidate_ratemap, **secondary_filter_kwargs):
                # If a ratemap passes the secondary_filter, overwrite the field_ratemap
                # This way final field_ratemap is the one detected with lowest threshold
                # but still passes the secondary_filter.
                field_ratemap = field_candidate_ratemap

        if field_ratemap is None:

            if n_subfields == 1:
                # If no field_ratemap passed the secondary_filter fo this field_candidate,
                # but this field_candidate had one subfield passing through the filter earlier
                # assign the origin of that subfield to be the origin of current field_candidate.
                subfield_index = [field_ratemap_dict['origin']
                                  for field_ratemap_dict in field_ratemap_dicts].index(current_level)
                field_ratemap_dicts[subfield_index]['origin'] = field_candidate[0]

        else:

            # If a field_ratemap did pass the secondary_filter, append it to field_ratemap_dicts
            field_ratemap_dicts.append({'ratemap': field_ratemap,
                                        'origin': field_candidate[0]})

            if n_subfields == 1:
                # Remove the single subfield of the current field_candidate from field_ratemap_dicts list
                subfield_index = [field_ratemap_dict['origin']
                                  for field_ratemap_dict in field_ratemap_dicts].index(current_level)
                del field_ratemap_dicts[subfield_index]

    field_ratemaps = [field_ratemap_dict['ratemap'] for field_ratemap_dict in field_ratemap_dicts
                      if 'ratemap' in field_ratemap_dict]

    return field_ratemaps


def detect_fields(ratemap, stability_ratemaps, base_threshold, threshold_step,
                  primary_filter_kwargs, secondary_filter_kwargs):
    """Returns a list of copies of input `ratemap` with every value except those in field
    replaced by `numpy.nan`.

    :param numpy.ndarray ratemap: shape (n_ybins, n_xbins) ratemap for fields to be detected.
        Values to be ignored should be set to numpy.nan.
    :param tuple stability_ratemaps: tuple with two ratemaps that are used for computing the stability
        of a field_ratemap after masking them with the non-numpy.nan elements in field_ratemap
    :param float base_threshold: values below this are set to numpy.nan and ignored
    :param float threshold_step: the step in ratemap values for iterative detection of fields.
    :param dict primary_filter_kwargs: see :py:func:`spatial.field.primary_filter` keyword arguments.
        filter_kwargs is passed on to that function as `**filter_kwargs` after `ratemap` argument.
    :param dict secondary_filter_kwargs: see :py:func:`spatial.field.secondary_filter` keyword arguments.
        `'max_relative_bins'` element is replaced with `'max_area_bins'` computed based on the `ratemap`.
        secondary_filter_kwargs is then passed on to that function as `**filter_kwargs` after `ratemap` argument.
    :return: field_ratemaps list of ratemaps where values outside field are numpy.nan
    :rtype: list
    """
    # Add stability_ratemaps and max_area_bins to a copy of secondary_filter_kwargs
    secondary_filter_kwargs = deepcopy(secondary_filter_kwargs)
    #print(secondary_filter_kwargs)
    secondary_filter_kwargs['stability_ratemaps'] = stability_ratemaps
    secondary_filter_kwargs['max_area_bins'] = \
        np.sum(~np.isnan(ratemap)) * secondary_filter_kwargs.pop('max_relative_bins')

    # Ensure original ratemap is not modified
    ratemap = ratemap.copy()

    # Ensure ratemap is in float dtype to support nans that are required in final output format
    if not isinstance(ratemap.dtype, np.floating):
        ratemap = ratemap.astype(np.float64)

    # Set numpy.nan values to 0 for compatibility with field detection methods.
    ratemap[np.isnan(ratemap)] = 0

    # Detect field candidates and extract those that pass all filters
    field_candidates = detect_field_candidates(ratemap, base_threshold, threshold_step, primary_filter_kwargs)
    field_ratemaps = extract_fields_from_field_candidates(field_candidates, secondary_filter_kwargs)

    # Set field_ratemap values of 0 to numpy.nan to indicate outside of field areas
    for i, field_ratemap in enumerate(field_ratemaps):
        field_ratemaps[i][field_ratemap == 0] = np.nan

    return field_ratemaps


def get_field_regionprops(field_ratemap, data_x_values=None, data_y_values=None,
                          orientation_x_axis_0_to_pi=True):
    """Returns output from :py:func:`skimage.measure.regionprops` for the boolean field map.

    Note, if data_x_values and data_y_values (both or neither must be provided), these are used
    to estimate corresponding centroid_x, centroid_y, minor_axis and major_axis values.

    Note, data_x_values and data_y_values must have equal and constant spacing in values,
    as the correction on minor_axis and major_axis length are based on this assumption.

    :param numpy.ndarray field_ratemap: shape (n_ybins, n_xbins) ratemap with numpy.nan values
        outside the field.
    :param numpy.ndarray data_x_values: if provided, to estimate position and axes length values
    :param numpy.ndarray data_y_values: if provided, to estimate position and axes length values
    :param bool orientation_x_axis_0_to_pi: if True (default), orientation value is set to range from
        0 to pi, increasing clock-wise from parallel to x-axis.
    :return: dictionary with selection of field properties returned by :py:func:`skimage.measure.regionprops`
    :rtype: dict
    """
    if (data_x_values is None) != (data_y_values is None):
        raise ValueError('data_x_values and data_y_values must both be provided or neither.')

    if not (data_x_values is None):
        x_offset = data_x_values[0]
        y_offset = data_y_values[0]
        pixel_edge_length = data_x_values[1] - data_x_values[0]
    else:
        x_offset = None
        y_offset = None
        pixel_edge_length = None

    out = regionprops(np.uint8(~np.isnan(field_ratemap)), coordinates='rc')[0]
    return {'area': out.area,
            'centroid_x': (out.centroid[1] if (data_x_values is None)
                           else out.centroid[1] * pixel_edge_length + x_offset),
            'centroid_y': (out.centroid[0] if (data_y_values is None)
                           else out.centroid[0] * pixel_edge_length + y_offset),
            'eccentricity': out.eccentricity,
            'minor_axis': (out.minor_axis_length if pixel_edge_length is None
                           else out.minor_axis_length * pixel_edge_length),
            'major_axis': (out.major_axis_length if pixel_edge_length is None
                           else out.major_axis_length * pixel_edge_length),
            'orientation': ((-out.orientation + np.pi / 2.) if orientation_x_axis_0_to_pi
                            else out.orientation)}


def compute_field_contour(field_ratemap, flipud=True):
    """Returns contour points of the field in the field_ratemap.

    Assumes input has only one contiguous region of values above 0 and not `numpy.nan`,

    :param numpy.ndarray field_ratemap: shape (n_ybins, n_xbins) ratemap with numpy.nan values
        outside the field.
    :return: contours list of contour points
    :param bool flipud: if True (default), the field_ratemap array is flipped prior to extracting
        the contour. This yields correct contour point values relative to field_ratemap array indices.
    :rtype: list
    """
    # The ratemap needs to be flipped on the vertical axis, because OpenCV starts counting
    # y-axis positions from bottom edge, but our analysis assumes position values are counted
    # ascending from the top edge of the ratemap array.
    if flipud:
        field_ratemap = np.flipud(field_ratemap.copy())

    # Convert input to binary
    field_ratemap[np.isnan(field_ratemap)] = 0
    field_binary_map = np.uint8(field_ratemap > 0)

    # Only keep the first contour from output list, as input should only have one region
    contour = cv.findContours(field_binary_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]

    return contour


def compute_field_ellipse(field_ratemap, flipud=False):
    """

    :param numpy.ndarray field_ratemap: shape (n_ybins, n_xbins) ratemap with numpy.nan values
        outside the field.
    :param bool flipud: if False (default), ratemap is not flipped along first dimension for
        detecting contour prior to ellipse fitting. This yields correct ellipse coordinates
        relative to field_ratemap array indices.
    :return:
    """

    contour = compute_field_contour(field_ratemap, flipud=flipud)

    (x, y), (MA, ma), angle = cv.fitEllipse(contour)

    return {'x': x, 'y': y, 'major_axis_width': MA, 'minor_axis_width': ma, 'angle': angle}

"""
Oldish stuff starts here
"""

def _is_dead_unit(heatmap):
    """
    Given a unit's 2D heatmap, check if it is a dead unit.
    """
    # return np.allclose(heatmap, 0)

    # unit is dead if less than 1% of the heatmap is active
    return np.sum(heatmap > 0) < 0.01 * heatmap.shape[0] * heatmap.shape[1]


def _compute_single_heatmap_fields_info(
        model_reps, 
        unit_index,
        env_x_max,
        env_y_max,
        env_x_min,
        env_y_min,
        multiplier,
        pixel_min_threshold, 
        pixel_max_threshold
    ):
    """
    Given a 2D heatmap of a unit, compute:
        num_clusters, num_pixels_in_clusters, max_value_in_clusters, \
            mean_value_in_clusters, var_value_in_clusters, heatmap_thresholded
    """
    
    """
    MAGGIE CHANGES:
    - Using Tanni et. al. code found here: https://github.com/Barry-lab/Publication_TanniDeCothiBarry2022/blob/main/electrophysiology_analysis/barrylab_ephys_analysis/spatial/fields.py
    - Now accepts non-summed model reps instead of the heatmap and does the summing myself, splitting into two for the stability check
    """
    #model_reps = (model_reps - np.min(model_reps)) / (np.max(model_reps) - np.min(model_reps))
    split = model_reps.shape[1] // 2
    summed1 = np.sum(model_reps[:, :split], axis=1)
    summed2 = np.sum(model_reps[:, split:], axis=1)
    totalsum = np.sum(model_reps, axis=1)
    
    ratemap1 = summed1.reshape(
                (env_x_max*multiplier-env_x_min*multiplier+1, 
                env_y_max*multiplier-env_y_min*multiplier+1) )
    ratemap2 = summed2.reshape(
                (env_x_max*multiplier-env_x_min*multiplier+1, 
                env_y_max*multiplier-env_y_min*multiplier+1) )
                
    totalmap = totalsum.reshape(
                (env_x_max*multiplier-env_x_min*multiplier+1, 
                env_y_max*multiplier-env_y_min*multiplier+1) )
    
    min_value = np.min(totalmap)
    max_value = np.max(totalmap)
    #print('mm', max_value, min_value)
    #totalmap = (totalmap - min_value) / (max_value - min_value)
    #ratemap1 = (ratemap1 - min_value) / (max_value - min_value)
    #ratemap2 = (ratemap2 - min_value) / (max_value - min_value)

    base_threshold = np.mean(totalmap)  # adjustable
    threshold_step = 0.1 * np.std(totalmap)  # adjustable
    
    primary_filter_kwargs = {
        'min_area_bins': 9,  # Minimum 3x3 grid
        'min_peak_value': np.max(totalmap) * 0.5  # 50% of max value
    }
    
    secondary_filter_kwargs = {
        'max_relative_bins': 0.5,  # Maximum 50% of total area
        'min_stability': 0.3,  # Minimum correlation of 0.5
        'stability_kwargs': {
            'min_included_value': np.mean(totalmap) / 2,  # Or another suitable value
            'min_bins': 9  # Minimum 3x3 grid for stability calculation   
        }
    }
    
    field_ratemaps = detect_fields(totalmap, (ratemap1, ratemap2), 
                                   base_threshold, threshold_step,
                                   primary_filter_kwargs, secondary_filter_kwargs)
                                   
    num_clusters = np.array([len(field_ratemaps)])
    num_pixels_in_clusters = []
    max_value_in_clusters = []
    mean_value_in_clusters = []
    var_value_in_clusters = []
    
    for field in field_ratemaps:
        valid_values = field[~np.isnan(field)]
        num_pixels_in_clusters.append(len(valid_values))
        max_value_in_clusters.append(np.max(valid_values))
        mean_value_in_clusters.append(np.mean(valid_values))
        var_value_in_clusters.append(np.var(valid_values))
        
    if num_clusters[0] == 0:
        num_pixels_in_clusters = np.array([0])
        max_value_in_clusters = np.array([0])
        mean_value_in_clusters = np.array([0])
        var_value_in_clusters = np.array([0])
        #max_value_indices_in_clusters = np.array([(0, 0)])
    else:
        max_value_in_clusters = np.array(max_value_in_clusters)
        mean_value_in_clusters = np.array(mean_value_in_clusters)
        var_value_in_clusters = np.array(var_value_in_clusters)
        #max_value_indices_in_clusters = np.array(max_value_indices_in_clusters)
    
    # return num_clusters, num_pixels_in_clusters, max_value_in_clusters, \
    #    mean_value_in_clusters, var_value_in_clusters, heatmap_thresholded, mean_angle
    return num_clusters, num_pixels_in_clusters, max_value_in_clusters, \
        mean_value_in_clusters, var_value_in_clusters, None
    
    """
    Old code:
    
    scaler = MinMaxScaler()
    # normalize to [0, 1]
    heatmap_normalized = scaler.fit_transform(heatmap)  
    # convert to [0, 255]      
    heatmap_gray = (heatmap_normalized * 255).astype(np.uint8)
    # compute activity threshold as the mean of the heatmap
    activity_threshold = np.mean(heatmap_gray)

    _, heatmap_thresholded = cv2.threshold(
        heatmap_gray, activity_threshold, 
        255, cv2.THRESH_BINARY
    )

    # num_labels=4,
    # num_labels includes background
    # labels \in (17, 17)
    # stats \in (4, 5): [left, top, width, height, area] for each label
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(heatmap_thresholded)

    # Create a mask to filter clusters based on pixel thresholds
    # e.g. mask=[False, True, False, True] for each label (i.e. a cluster)
    mask = (stats[:, cv2.CC_STAT_AREA] >= pixel_min_threshold) & \
            (stats[:, cv2.CC_STAT_AREA] <= pixel_max_threshold)
    # set background to False regardless of pixel thresholds
    mask[0] = False
        
    # Filter the stats and labels based on the mask
    # filtered_stats.shape (2, 5)
    filtered_stats = stats[mask]

    # For labels with mask=True, keep the label, otherwise set to 0
    # this in fact will include 0, but we want 1, 3 only
    # so when using `filtered_labels` to extract max value in each cluster
    # we need to exclude 0
    filtered_labels = np.where(np.isin(labels, np.nonzero(mask)[0]), labels, 0)

    # Count the number of clusters that meet the criteria
    num_clusters = np.array([filtered_stats.shape[0]])

    # Get the number of pixels in each cluster
    num_pixels_in_clusters = filtered_stats[:, cv2.CC_STAT_AREA]

    # Get the max/mean/var value in heatmap based on each cluster
    max_value_in_clusters = []
    mean_value_in_clusters = []
    var_value_in_clusters = []
    for label in np.unique(filtered_labels):
        if label != 0:
            max_value_in_clusters.append(
                np.around(
                    np.max(heatmap[filtered_labels == label]), 1
                )
            )
            mean_value_in_clusters.append(
                np.around(
                    np.mean(heatmap[filtered_labels == label]), 1
                )
            )
            var_value_in_clusters.append(
                np.around(
                    np.var(heatmap[filtered_labels == label]), 1
                )
            )
            
    # Add 0 to `num_pixels_in_clusters` and `max_value_in_clusters`
    # in case `num_clusters` is 0. This is helpful when we want to
    # plot fields info against coef, as no matter if there is a cluster
    # for a unit, there is always a coef for that unit.
    if num_clusters[0] == 0:
        num_pixels_in_clusters = np.array([0])
        max_value_in_clusters = np.array([0])
        mean_value_in_clusters = np.array([0])
        var_value_in_clusters = np.array([0])
    else:
        max_value_in_clusters = np.array(max_value_in_clusters)
        mean_value_in_clusters = np.array(mean_value_in_clusters)
        var_value_in_clusters = np.array(var_value_in_clusters)

    colors = np.arange(150, dtype=int).tolist()
    for label in np.unique(filtered_labels):
        if label != 0:
            # create a mask for each label
            mask = np.where(filtered_labels == label, 255, 0).astype(np.uint8)
            # find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # draw contours
            cv2.drawContours(heatmap_thresholded, contours, -1, colors[label-1], 1)
            
    """

    return num_clusters, num_pixels_in_clusters, max_value_in_clusters, \
        mean_value_in_clusters, var_value_in_clusters, heatmap_thresholded


def _compute_single_heatmap_grid_scores(activation_map, smooth=False):
    # mask parameters
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())

    scorer = scores.GridScorer(
        len(activation_map),        # nbins
        [0, len(activation_map)-1], # coords_range
        masks_parameters            # parameters for the masks
    )

    score_60, score_90, max_60_mask, max_90_mask, sac = \
        scorer.get_scores(activation_map)
    
    return score_60, score_90, max_60_mask, max_90_mask, sac, scorer


def _compute_single_heatmap_border_scores(activation_map, db=3):
    """
    Banino et al. 2018 uses db=3.
    """
    num_bins = activation_map.shape[0]
    
    # Compute c (average activity for bins further than db bins from any wall)
    c = np.mean([
        activation_map[i, j]
        for i in range(db, num_bins - db)
        for j in range(db, num_bins - db)
    ])

    wall_scores = []

    # Compute the average activation for each wall
    for i in range(4):
        if i == 0:
            # Top wall
            activations = activation_map[:db, :]
        elif i == 1:
            # Right wall
            activations = activation_map[:, -db:]
        elif i == 2:
            # Bottom wall
            activations = activation_map[-db:, :]
        elif i == 3:
            # Left wall
            activations = activation_map[:, :db]

        bi = np.mean(activations)
        wall_scores.append((bi - c) / (bi + c))

    return np.max(wall_scores)


def _compute_single_heatmap_directional_scores(activation_maps):
    """
    Args:
        `activation_maps` correspond to the un-summed activation maps
        for a single unit across rotations \in (n_locations, n_rotations)

    - num of angular bins in Banino here becomes `n_rotations`.
    - based on Banino eq, we need to convert each n_rotations to 
        `alpha_i` which is angle.
    - the intensity `beta_i` of an angle is the average activation 
        across all locations for that angle.
    """
    #mask = ~np.all(activation_maps == 0, axis=1) # maggie change
    #activation_maps = activation_maps[mask]

    activation_maps = (activation_maps - np.min(activation_maps)) / (np.max(activation_maps) - np.min(activation_maps))
    
    # model_reps \in (n_locations, n_rotations, n_features)
    # activation_maps \in (n_locations, n_rotations)
    num_bins = activation_maps.shape[1]
    alphas = np.linspace(0, 2*np.pi, num=num_bins, endpoint=False)
    betas = np.mean(activation_maps, axis=0) # mean direction for this spatial bin

    # given a rotation, we can compute alpha_i and beta_i
    # which are used to compute r_i in the eq.
    # we collect r_i for each rotation and compute the mean
    # vector, whose length is used as the directional score.
    polar_plot_coords = [] # (n_rotations, 2)
    per_rotation_vector_length = []
    for alpha_i, beta_i in zip(alphas, betas):
        polar_plot_coords.append(
            [beta_i*np.cos(alpha_i), beta_i*np.sin(alpha_i)]
        )
        per_rotation_vector_length.append(
            np.linalg.norm([beta_i*np.cos(alpha_i), beta_i*np.sin(alpha_i)])
        )
    
    # to compute mean vector length,
    # first we compute the sum of r_i normed by sum of beta_i
    r_normed_by_beta = np.sum(
        np.array(polar_plot_coords), axis=0) / np.sum(betas)

    # then we compute the length of the mean vector
    mean_vector_length = np.linalg.norm(r_normed_by_beta)
    logging.info(f'[Check] mean_vector_length: {mean_vector_length}')
    return mean_vector_length, per_rotation_vector_length