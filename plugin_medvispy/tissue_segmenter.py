import numpy as np
from scipy import ndimage
from skimage.morphology import binary_closing, remove_small_objects, ball, binary_opening, skeletonize
from skimage.filters import gaussian, threshold_otsu
import gc
from numba import jit, prange
from skimage.filters import frangi
from skimage.filters import sobel  
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from skimage import measure 
from skimage.draw import line_nd  
from skimage.measure import regionprops, label


# Optimized partial volume estimation function using Numba.
# Estimates soft tissue class probabilities per voxel using Gaussian models and local context.
@jit(nopython=True)
def pv_estimation_jit(mri_data, mask, hard_labels, means, variances, shape, n_classes, weights_array, iterations, crack_mean, crack_var):
    pve_probs = np.zeros((shape[0], shape[1], shape[2], n_classes), dtype=np.float32)
    
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if mask[x, y, z] > 0:
                    label = hard_labels[x, y, z]

                    # For known classes, assign hard label
                    if label < n_classes:
                        pve_probs[x, y, z, label] = 1.0
                    else:
                        voxel_val = mri_data[x, y, z]
                        probs = np.zeros(n_classes, dtype=np.float32)

                        # Compute Gaussian likelihoods for each class
                        for c in range(n_classes):
                            diff = voxel_val - means[c]
                            probs[c] = weights_array[c] * np.exp(-0.5 * (diff**2) / variances[c]) / np.sqrt(2 * np.pi * variances[c])

                        # Boost CSF probability in homogeneous low-intensity regions
                        if label == 1:  # CSF class
                            local_window = mri_data[max(0, x-2):min(shape[0], x+3),
                                                    max(0, y-2):min(shape[1], y+3),
                                                    max(0, z-2):min(shape[2], z+3)]
                            local_mean = np.mean(local_window)
                            local_std = np.std(local_window)
                            if local_std < 0.05 and local_mean < means[1] * 1.1:
                                probs[1] *= 2.5

                        # Enhance Skull probability in presence of potential cracks
                        if label == 3:  # Skull class
                            local_window = mri_data[max(0, x-1):min(shape[0], x+2),
                                                    max(0, y-1):min(shape[1], y+2),
                                                    max(0, z-1):min(shape[2], z+2)]
                            local_std = np.std(local_window)
                            crack_prob = (np.exp(-0.5*(voxel_val - crack_mean)**2 / crack_var) *
                                          (1 - np.exp(-local_std / 0.1)))
                            probs[3] *= (1.0 + 6.0 * crack_prob)

                        # Normalize to get probability distribution
                        probs_sum = np.sum(probs) + 1e-6
                        if probs_sum > 0:
                            pve_probs[x, y, z, :] = probs / probs_sum
                            
    return pve_probs

    
# Optimized MRF (Markov Random Field) computation using Numba and 26-neighborhood averaging.
# Computes spatial smoothing term for a specific class c using neighboring probabilities and edge suppression.
@jit(nopython=True, parallel=True)
def compute_mrf_term_fast(prob_maps, mask, beta, edge_map, c):
    mrf_terms = np.zeros(prob_maps.shape[:3], dtype=np.float32)
    
    for x in prange(1, prob_maps.shape[0]-1):
        for y in range(1, prob_maps.shape[1]-1):
            for z in range(1, prob_maps.shape[2]-1):
                if mask[x, y, z]:
                    # Reduce smoothing strength near edges
                    edge_weight = 1 - 0.7 * edge_map[x, y, z]

                    # 26-neighborhood for class 'c' probability smoothing
                    neighbors = prob_maps[x-1:x+2, y-1:y+2, z-1:z+2, c]
                    mrf_terms[x, y, z] = beta * edge_weight * np.mean(neighbors)
                    
    return mrf_terms


# Class for segmenting CSF, Scalp, Skull, and Background tissues in MRI images
class TissueSegmenter:
    def __init__(self):
        # Initialize tissue-specific beta values for MRF regularization
        self.beta_params = {
            'csf': 1.0,         # Strong smoothing for CSF
            'skull': 0.2,       # Minimal smoothing for skull due to sharp boundaries
            'scalp': 0.7,       # Moderate smoothing
            'background': 0.2   # Minimal smoothing for background
        }

        self.edge_map = None  # Edge map for adaptive smoothing control

        self.weights = None       # Mixture weights for tissue probability estimation
        self.thresholds = None    # Optional threshold values (can be set later)
        self.means = None         # Intensity means for each tissue class
        self.variances = None     # Intensity variances for each tissue class
        self.tissue_types = ['background', 'csf', 'scalp', 'skull']  # Class labels

    def compute_tissue_thresholds(self, templates):
        """Hybrid threshold calculation with crack-preserving defaults"""
        # Base thresholds (optimized values from your best static version)
        base_thresholds = {
            'csf': 0.00002,  # Very low for CSF fluid detection
            'scalp': 0.2,   # Medium for scalp tissue
            'skull': 0.10    # Low to preserve thin structures
        }
        
        # Dynamic adjustment factor based on template quality
        self.thresholds = {
            'csf': max(base_thresholds['csf'], 
                    0.7 * self.compute_probability_threshold(templates['csf'], 60)),
            'scalp': min(base_thresholds['scalp'], 
                    1.3 * self.compute_probability_threshold(templates['scalp'], 25)),
            'skull': max(base_thresholds['skull'], 
                    0.5 * self.compute_probability_threshold(templates['skull'], 15))
        }
        
    # Calculate a percentile-based threshold from a non-zero tissue probability template
    def compute_probability_threshold(self, tissue_template, percentile):
        # Extract non-zero probability values
        probability_values = tissue_template[tissue_template > 0]

        # Return a default value if the template is empty
        if len(probability_values) == 0:
            return 0.1

        # Compute the percentile-based threshold
        return np.percentile(probability_values, percentile)

    # Compute the mean and standard deviation of MRI intensities for a specific tissue
    def compute_tissue_intensity_stats(self, mri_data, tissue_mask):
        # Extract intensity values where the tissue mask is active
        tissue_intensity_values = mri_data[tissue_mask > 0]

        # Compute mean intensity, or use default if no values found
        mean_intensity = np.mean(tissue_intensity_values) if len(tissue_intensity_values) > 0 else 0.5

        # Compute standard deviation, or use default if no values found
        std_intensity = np.std(tissue_intensity_values) if len(tissue_intensity_values) > 0 else 0.1

        return mean_intensity, std_intensity

    # Compute tissue-specific weights based on intensity statistics and thresholds
    def compute_tissue_weights(self, mri_data, templates):
        # Ensure thresholds have been computed
        if self.thresholds is None:
            raise ValueError("Thresholds must be computed before computing tissue weights.")
        
        # Create tissue masks by applying thresholds to the probability templates
        csf_mask = templates['csf'] > self.thresholds['csf']
        scalp_mask = templates['scalp'] > self.thresholds['scalp']
        skull_mask = templates['skull'] > self.thresholds['skull']
        
        # Calculate intensity statistics (mean and std) for each tissue
        csf_mean, csf_std = self.compute_tissue_intensity_stats(mri_data, csf_mask)
        scalp_mean, scalp_std = self.compute_tissue_intensity_stats(mri_data, scalp_mask)
        skull_mean, skull_std = self.compute_tissue_intensity_stats(mri_data, skull_mask)

        # Prevent division by zero in weight calculations
        csf_mean = max(csf_mean, 1e-6)
        scalp_mean = max(scalp_mean, 1e-6)
        skull_mean = max(skull_mean, 1e-6)
        
        # Define class weights based on normalized variability (std / mean)
        self.weights = {
            'csf': 4.0 + (csf_std / csf_mean),
            'scalp': 7.0 + (scalp_std / scalp_mean),
            'skull': 9.0 + (skull_std / skull_mean),
            'background': 0.01  # Minimal contribution from background
        }

        
    # Preprocess MRI data by log transformation and min-max normalization
    def preprocess_mri(self, mri_data):
        # Replace NaNs with 0 to avoid computational errors
        mri_data = np.nan_to_num(mri_data, 0)

        # Apply log transformation to reduce intensity range skewness
        mri_log = np.where(mri_data > 0, np.log(mri_data + 1.0), 0.0)

        # Normalize to [0, 1] using min-max scaling
        normalized_mri = (mri_log - np.min(mri_log)) / (np.max(mri_log) - np.min(mri_log) + 1e-6)

        return normalized_mri.astype(np.float32)

    # Enhance the contrast of a specific tissue region in MRI based on intensity and prior probability
    def enhance_tissue(self, mri_data, tissue_template, tissue_type):
        enhanced = mri_data.copy()

        # Compute tissue intensity statistics from high-confidence regions
        tissue_mean, tissue_std = self.compute_tissue_intensity_stats(
            mri_data, tissue_template > self.thresholds[tissue_type]
        )

        # Define tissue mask using thresholded template and intensity closeness
        tissue_mask = (tissue_template > self.thresholds[tissue_type]) & \
                    (np.abs(mri_data - tissue_mean) < 2 * tissue_std)

        # Compute enhancement factor based on intensity deviation from mean
        enhancement_factor = 1 + np.abs(mri_data[tissue_mask] - tissue_mean) / (tissue_std + 1e-6)

        # Clamp the enhancement to a reasonable range to prevent artifacts
        enhancement_factor = np.clip(enhancement_factor, 0.8, 2.0)

        # Apply weighted enhancement to selected voxels
        enhanced[tissue_mask] *= self.weights[tissue_type] * enhancement_factor

        # Normalize the enhanced volume to [0, 1]
        enhanced = (enhanced - np.min(enhanced)) / (np.max(enhanced) - np.min(enhanced) + 1e-6)

        return enhanced


   # Perform Partial Volume Estimation using prior templates, intensity models, and spatial refinement
    def pv_estimation(self, mri_data, mask, hard_labels, means, variances, templates, n_classes=4, iterations=20):
        shape = mri_data.shape

        # Convert weights to array for fast access
        weights_array = np.array([self.weights[t] for t in self.tissue_types], dtype=np.float32)

        # Estimate skull crack intensity distribution
        crack_mean, crack_var = self.compute_crack_stats(mri_data, templates['skull'])

        # Run JIT-optimized partial volume estimation
        pve_probs = pv_estimation_jit(
            mri_data, mask, hard_labels, np.array(means), np.array(variances),
            shape, n_classes, weights_array, iterations, crack_mean, crack_var
        )
        return pve_probs


    # Compute mean and variance of MRI intensity incrementally (memory-efficient)
    def compute_incremental_stats(self, mri_norm, mask, batch_size=50):
        n_voxels = np.sum(mask)
        mean_sum = 0.0
        m2 = 0.0
        count = 0

        # Iterate over batches along first dimension to reduce memory load
        for start in range(0, mri_norm.shape[0], batch_size):
            end = min(start + batch_size, mri_norm.shape[0])
            batch_data = mri_norm[start:end]
            batch_mask = mask[start:end]

            # Extract masked voxel values
            batch_values = batch_data[batch_mask > 0]

            # Welford's algorithm for online variance calculation
            for value in batch_values:
                count += 1
                delta = value - mean_sum
                mean_sum += delta / count
                delta2 = value - mean_sum
                m2 += delta * delta2

            gc.collect()  # Optional: manual garbage collection for large 3D arrays

        mean_value = mean_sum
        var_value = m2 / (count - 1) if count > 1 else 0.1

        return mean_value, var_value + 1e-6

    # Perform probabilistic tissue segmentation using MRF, Gaussian modeling, and prior templates
    def segment_tissues(self, mri_norm, mri_enhanced, templates):
        self.edge_map = sobel(mri_norm)  # Compute edge map to reduce smoothing near edges

        n_classes = 4
        shape = mri_norm.shape
        mask = (mri_norm > 0).astype(np.uint8)

        # Initialize probability maps using template priors and custom weights
        prob_maps = np.ones((*shape, n_classes), dtype=np.float32)
        atlas_max = np.max(np.stack([templates['csf'], templates['scalp'], templates['skull']], axis=0), axis=0)
        prob_maps[..., 0] = np.where(mask, np.clip((1 - atlas_max), 0, 1) * self.weights['background'] + 0.005, 0)
        prob_maps[..., 1] = np.where(mask, templates['csf'] * self.weights['csf'] * 0.7 + 0.2, 0)
        prob_maps[..., 2] = np.where(mask, templates['scalp'] * self.weights['scalp'] * 0.15 + 0.1, 0)
        prob_maps[..., 3] = np.where(mask, templates['skull'] * self.weights['skull'] * 0.4 + 0.1, 0)

        # Normalize initial probability maps
        prob_sums = np.sum(prob_maps, axis=-1, keepdims=True)
        prob_maps = np.where(prob_sums > 0, prob_maps / (prob_sums + 1e-6), 0)

        # Initialize Gaussian parameters if not already computed
        if self.means is None or self.variances is None:
            mean_value, var_value = self.compute_incremental_stats(mri_enhanced, mask)
            self.means = [mean_value] * n_classes
            self.variances = [var_value] * n_classes

        max_iter_main = 15
        for iter in range(max_iter_main):
            print(f"Main Iteration {iter + 1}/{max_iter_main}")
            old_prob_maps = prob_maps.copy()
            mrf_terms = np.zeros_like(prob_maps)

            # Compute MRF regularization terms per class
            for c in range(n_classes):
                mrf_terms[..., c] = compute_mrf_term_fast(
                    prob_maps, mask, self.beta_params[self.tissue_types[c]], self.edge_map, c
                )

            gaussian_terms = np.zeros_like(prob_maps)
            # Compute log-Gaussian likelihoods
            for c in range(n_classes):
                gaussian_terms[..., c] = -0.5 * (
                    ((mri_norm - self.means[c]) ** 2) / self.variances[c]
                    + np.log(self.variances[c])  # Log-space Gaussian
                )
                # Combine MRF and Gaussian likelihoods, stabilize exponential range
                prob_maps[..., c] = np.exp(
                    np.clip(mrf_terms[..., c] + gaussian_terms[..., c], -30, 30)
                ) + 1e-8

            # Reweight using prior templates
            for c in range(n_classes):
                if c == 1:
                    prob_maps[..., c] *= templates['csf'] * self.weights['csf'] * 1.4
                elif c == 2:
                    prob_maps[..., c] *= templates['scalp'] * self.weights['scalp'] * 1.2
                elif c == 3:
                    prob_maps[..., c] *= templates['skull'] * self.weights['skull'] * 1.5

            # Normalize probabilities again
            prob_sums = np.sum(prob_maps, axis=-1, keepdims=True)
            prob_maps = np.where(prob_sums > 0, prob_maps / (prob_sums + 1e-6), 0)

            # Update Gaussian stats based on new soft assignments
            new_means = []
            new_vars = []
            for c in range(n_classes):
                mask_c = prob_maps[..., c] > 0.2
                if np.any(mask_c):
                    mean_c = np.mean(mri_norm[mask_c])
                    var_c = np.var(mri_norm[mask_c]) + 1e-6
                else:
                    print(f"WARNING: No voxels above threshold in class {c}. Reusing previous mean/variance.")
                    mean_c = self.means[c] if self.means else 0.5
                    var_c = self.variances[c] if self.variances else 0.1
                new_means.append(mean_c)
                new_vars.append(max(var_c, 0.005))  # Avoid very low variance
            self.means = new_means
            self.variances = new_vars

            # Check convergence
            diff = np.mean(np.abs(prob_maps - old_prob_maps))
            print(f"Change: {diff:.6f}")
            if diff < 1e-4 and iter > 4:
                break

        # Final hard label and refined probability estimation
        hard_labels = np.argmax(prob_maps, axis=-1)
        pve_probs = self.pv_estimation(mri_norm, mask, hard_labels, self.means, self.variances, templates)
        labels = np.argmax(pve_probs, axis=-1)

        return labels, mri_norm

    # Refine segmentation mask by processing CSF, Skull, and Scalp regions and combining results
    def post_process_segmentation(self, segmentation, mri_data):
        INTENSITY_VALUES = {0: 0, 1: 64, 2: 175, 3: 255}
        struct = ball(2)
        
        # Process CSF: fill holes, close gaps, remove small objects, dilate, and fill holes again
        csf_mask = segmentation == 1
        csf_filled = ndimage.binary_fill_holes(csf_mask)  # Fill internal holes in CSF regions
        csf_closed = ndimage.binary_closing(csf_filled, structure=struct, iterations=1)  # Close small gaps
        csf_cleaned = remove_small_objects(csf_closed, min_size=200)  # Remove small isolated objects
        csf_dilated = ndimage.binary_dilation(csf_cleaned, structure=struct, iterations=1)  # Dilate for better connectivity
        csf_final = ndimage.binary_fill_holes(csf_dilated)  # Final hole filling
        
        # Process Skull with a dedicated post-processing function
        skull_mask = segmentation == 3
        skull_final = self.post_process_skull(skull_mask, mri_data)
        
        # Process Scalp by closing small gaps
        scalp_mask = segmentation == 2
        scalp_processed = ndimage.binary_closing(scalp_mask, structure=struct)
        
        # Combine all processed masks into the final result with assigned intensity values
        result = np.zeros_like(segmentation, dtype=np.uint8)
        result[scalp_processed] = INTENSITY_VALUES[1]
        result[csf_final] = INTENSITY_VALUES[2]
        result[skull_final] = INTENSITY_VALUES[3]        
        
        return result

    
    # Enhanced detection of skull cracks using multi-scale Frangi filtering and morphological operations
    def enhanced_crack_detection(self, mri_data, skull_mask):
        skull_values = mri_data[skull_mask]
        if len(skull_values) == 0:
            return np.zeros_like(skull_mask, dtype=bool)

        # Identify crack candidates based on intensity threshold (20th percentile)
        q20 = np.percentile(skull_values, 20)
        crack_candidates = (mri_data < q20) & skull_mask

        # Compute line-like structures with Frangi filter at multiple scales
        line_scores = np.zeros_like(mri_data, dtype=np.float32)
        for sigma in [0.2, 0.5, 0.9]:
            response = frangi(mri_data, sigmas=[sigma], alpha=0.2, beta=0.7, gamma=12, black_ridges=False)
            line_scores = np.maximum(line_scores, response)

        # Smooth response and threshold using Otsu within skull region
        smoothed = gaussian(line_scores, sigma=0.8)
        line_thresh = threshold_otsu(smoothed[skull_mask])
        line_mask = smoothed > (0.9 * line_thresh)

        # Combine detected cracks and candidates, then clean with morphological closing and opening
        combined = (line_mask | crack_candidates) & skull_mask
        combined = binary_closing(combined, ball(1))
        combined = binary_opening(combined, ball(1))

        # Skeletonize to get thin crack lines
        thin_cracks = skeletonize(combined)

        # Keep only sufficiently long crack segments (major axis length >= 7)
        labeled = label(thin_cracks)
        props = regionprops(labeled)
        final_cracks = np.zeros_like(thin_cracks)
        for prop in props:
            if prop.axis_major_length >= 7:
                final_cracks[labeled == prop.label] = 1

        return final_cracks.astype(np.uint8)

    
    # Connect broken crack segments in 3D using a minimum spanning tree to bridge close gaps
    def refine_crack_connections(self, segmentation, crack_map):
        
        labeled = label(crack_map)
        props = regionprops(labeled)
        
        if len(props) > 1:
            # Compute centroids and pairwise distances of crack segments
            centroids = np.array([prop.centroid for prop in props])
            dist_matrix = squareform(pdist(centroids))
            
            # Generate MST to find shortest connections between segments
            mst = minimum_spanning_tree(dist_matrix)
            
            for i, j in zip(*mst.nonzero()):
                voxel_size = 1.0  # mm per pixel (adjust based on data)
                max_connection_distance = 5.0 / voxel_size  # max 5mm to connect
                
                if dist_matrix[i, j] < max_connection_distance:
                    start = tuple(np.round(props[i].centroid).astype(int))
                    end = tuple(np.round(props[j].centroid).astype(int))
                    
                    # Get coordinates of line connecting two segment centroids
                    line_coords = line_nd(start, end, endpoint=True)
                    
                    # Clip coordinates to within image volume bounds
                    line_coords = (
                        np.clip(line_coords[0], 0, segmentation.shape[0] - 1),
                        np.clip(line_coords[1], 0, segmentation.shape[1] - 1),
                        np.clip(line_coords[2], 0, segmentation.shape[2] - 1)
                    )
                    
                    segmentation[line_coords] = 255  # Assign skull label to connecting line
        
        return segmentation

    
    #Polished skull post-processing with better crack preservation
    def post_process_skull(self, skull_mask, mri_data):
        # Step 1: Crack Detection
        crack_map = self.enhanced_crack_detection(mri_data, skull_mask)

        # Step 2: Gentle smoothing of skull (adaptive morphological opening)
        struct = ball(1)
        smoothed_skull = binary_opening(skull_mask, struct)

        # Step 3: Merge cracks carefully
        combined_skull = np.where(crack_map, skull_mask, smoothed_skull)

        # Step 4: Fill small gaps inside skull (helps thin bone preservation)
        filled_skull = ndimage.binary_fill_holes(combined_skull)

        # Step 5: Remove noise but **preserve connected thin structures**
        cleaned_skull = remove_small_objects(filled_skull, min_size=100)

        # Step 6: Optional: crack reinforcement (prevent small cracks from disappearing)
        if np.sum(crack_map) > 0:
            cleaned_skull[crack_map > 0] = 1

        return cleaned_skull
