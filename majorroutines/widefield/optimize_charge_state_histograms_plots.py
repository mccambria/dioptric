# # -*- coding: utf-8 -*-
# """
# Real-time acquisition, histogram analysis, and SLM weight adjustment.

# Created on Oct 26, 2024

# @author: sbcahnd
# """
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
# from scipy.optimize import curve_fit
# from skimage.filters import threshold_otsu, threshold_triangle, threshold_li
# from collections import defaultdict
# import seaborn as sns
# import os
# from scipy.stats import norm
# from sklearn.mixture import GaussianMixture
# import numpy as np
# from collections import defaultdict
# from utils import kplotlib as kpl
# from utils import data_manager as dm
# from utils import widefield as widefield

# os.environ['OMP_NUM_THREADS'] = '9'

# # Define the 2D Gaussian function
# def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
#     x, y = xy
#     xo, yo = float(xo), float(yo)
#     a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (2 * sigma_y**2)
#     b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
#     c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (2 * sigma_y**2)
#     g = offset + amplitude * np.exp(-(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
#     return g.ravel()

# # Function to fit a 2D Gaussian around NV coordinates
# def fit_gaussian(image, coord, window_size=2):
#     x0, y0 = coord
#     img_shape_y, img_shape_x = image.shape
#     x_min = max(int(x0 - window_size), 0)
#     x_max = min(int(x0 + window_size + 1), img_shape_x)
#     y_min = max(int(y0 - window_size), 0)
#     y_max = min(int(y0 + window_size + 1), img_shape_y)

#     if (x_max - x_min) <= 1 or (y_max - y_min) <= 1:
#         print(f"Invalid cutout for NV at ({x0}, {y0}): Region too small or out of bounds")
#         return x0, y0, 0

#     x_range = np.arange(x_min, x_max)
#     y_range = np.arange(y_min, y_max)
#     x, y = np.meshgrid(x_range, y_range)
#     image_cutout = image[y_min:y_max, x_min:x_max]

#     if image_cutout.size == 0:
#         print(f"Zero-size cutout for NV at ({x0}, {y0})")
#         return x0, y0, 0

#     image_cutout = (image_cutout - np.min(image_cutout)) / (np.max(image_cutout) - np.min(image_cutout))
#     initial_guess = (1, x0, y0, 3, 3, 0, 0)

#     try:
#         bounds = (
#             (0, x_min, y_min, 0, 0, -np.pi, 0),
#             (np.inf, x_max, y_max, np.inf, np.inf, np.pi, np.inf),
#         )
#         popt, _ = popt, _ = curve_fit(gaussian_2d, (x, y), image_cutout.ravel(), p0=initial_guess, bounds=bounds, maxfev=5000)
#         amplitude, fitted_x, fitted_y, _, _, _, _ = popt
#         return fitted_x, fitted_y, amplitude
#     except Exception as e:
#         print(f"Fit failed for NV at ({x0}, {y0}): {e}")
#         return x0, y0, 0

# def calculate_fwhm(sigma_x, sigma_y):
#     return 2.355 * np.array([sigma_x, sigma_y])

# def fit_gaussian_and_calculate_fwhm(image, coord, window_size=6):
#     fitted_x, fitted_y, amplitude = fit_gaussian(image, coord, window_size)
#     if amplitude > 0:
#         x0, y0 = coord
#         img_shape_y, img_shape_x = image.shape
#         x_min = max(int(x0 - window_size), 0)
#         x_max = min(int(x0 + window_size + 1), img_shape_x)
#         y_min = max(int(y0 - window_size), 0)
#         y_max = min(int(y0 + window_size + 1), img_shape_y)
#         x_range = np.arange(x_min, x_max)
#         y_range = np.arange(y_min, y_max)
#         x, y = np.meshgrid(x_range, y_range)
#         image_cutout = image[y_min:y_max, x_min:x_max]
#         initial_guess = (1, fitted_x, fitted_y, 3, 3, 0, 0)
#         try:
#             popt, _ = curve_fit(gaussian_2d, (x, y), image_cutout.ravel(), p0=initial_guess)
#             _, _, _, sigma_x, sigma_y, _, _ = popt
#             fwhm = calculate_fwhm(sigma_x, sigma_y)
#             return fitted_x, fitted_y, amplitude, fwhm
#         except Exception as e:
#             print(f"FWHM fit failed for NV at ({x0}, {y0}): {e}")
#             return fitted_x, fitted_y, amplitude, [0, 0]
#     return fitted_x, fitted_y, amplitude, [0, 0]

# def threshold_counts(nv_list, sig_counts, ref_counts=None, method='otsu', plot=False, **kwargs):
#     """Threshold counts for NVs using various methods, including GMM for bimodal data."""

#     from skimage.filters import threshold_otsu, threshold_triangle, threshold_li
#     import numpy as np

#     # Dictionary to map methods to functions
#     threshold_methods = {
#         'otsu': threshold_otsu,
#         'triangle': threshold_triangle,
#         'entropy': threshold_li,
#         'gmm': fit_bimodal_gmm  # Assumes fit_bimodal_gmm returns a threshold
#     }

#     if method not in threshold_methods:
#         raise ValueError(f"Unknown thresholding method: {method}")

#     num_nvs = len(nv_list)
#     sig_thresholds, ref_thresholds = [], []

#     for nv_ind in range(num_nvs):
#         combined_counts = ref_counts[nv_ind].flatten() if ref_counts is not None else sig_counts[nv_ind].flatten()

#         # Apply the chosen method
#         if method == 'gmm':
#             gmm, threshold = fit_bimodal_gmm(combined_counts)
#             if plot:
#                 plot_gmm_fit(combined_counts, gmm, title=f"GMM Fit for NV {nv_ind}")
#         else:
#             threshold_func = threshold_methods[method]
#             threshold = threshold_func(combined_counts, **kwargs)

#         sig_thresholds.append(threshold)

#         # Optional ref threshold calculation
#         if ref_counts is not None:
#             ref_data = ref_counts[nv_ind].flatten()
#             if method == 'gmm':
#                 _, ref_threshold = fit_bimodal_gmm(ref_data)
#             else:
#                 ref_threshold = threshold_func(ref_data, **kwargs)
#             ref_thresholds.append(ref_threshold)

#     return sig_thresholds, ref_thresholds if ref_counts is not None else None


# def fit_bimodal_gmm(data, n_components=2, random_state=42):
#     """Fit a Gaussian Mixture Model (GMM) to data and return the GMM and threshold."""
#     from sklearn.mixture import GaussianMixture
#     import numpy as np

#     gmm = GaussianMixture(n_components=n_components, random_state=random_state)
#     data_reshaped = data.reshape(-1, 1)
#     gmm.fit(data_reshaped)

#     # Calculate threshold as the midpoint between the means of the two components
#     means = np.sort(gmm.means_.flatten())
#     threshold = np.mean(means) if len(means) == 2 else means[0]  # Simple logic for two components

#     return gmm, threshold


# def plot_gmm_fit(data, gmm, sig_counts=None, title="GMM Fit on Data", plot_components=True):
#     """Plot data histogram and fitted GMM components using Seaborn for visualization."""
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import seaborn as sns
#     from scipy.stats import norm

#     sns.set(style="whitegrid")
#     x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)
#     log_prob = gmm.score_samples(x)
#     pdf = np.exp(log_prob)

#     plt.figure(figsize=(12, 7))
#     sns.histplot(data, bins=30, kde=False, color="gray", label="Data", stat="density", alpha=0.6)

#     if sig_counts is not None:
#         sns.histplot(sig_counts, bins=30, kde=False, color="blue", label="Signal Data", stat="density", alpha=0.4)

#     plt.plot(x, pdf, color='black', lw=2, label='GMM Fit')

#     if plot_components:
#         means = gmm.means_.flatten()
#         covariances = gmm.covariances_.flatten()
#         weights = gmm.weights_.flatten()
#         for mean, covariance, weight in zip(means, covariances, weights):
#             std_dev = np.sqrt(covariance)
#             component_pdf = weight * norm.pdf(x, loc=mean, scale=std_dev)
#             plt.plot(x, component_pdf, '--', lw=2, label=f'Component (mean={mean:.2f})')

#     plt.title(title, fontsize=14)
#     plt.xlabel('Data Value', fontsize=12)
#     plt.ylabel('Density', fontsize=12)
#     plt.legend()
#     plt.show()


# def calculate_area_under_gmm(gmm, data):
#     """Calculate the area under each Gaussian component for a bimodal GMM."""
#     import numpy as np
#     from scipy.stats import norm

#     means = gmm.means_.flatten()
#     covariances = gmm.covariances_.flatten()
#     weights = gmm.weights_.flatten()

#     # Sort components for consistency
#     sorted_indices = np.argsort(means)
#     means = means[sorted_indices]
#     covariances = covariances[sorted_indices]
#     weights = weights[sorted_indices]

#     areas = []
#     for mean, covariance, weight in zip(means, covariances, weights):
#         std_dev = np.sqrt(covariance)
#         distribution = norm(loc=mean, scale=std_dev)
#         area = weight * np.sum(distribution.pdf(data))
#         areas.append(area)
#     return areas

# def calculate_metrics(nv_list, raw_data, method='otsu', plot=False, **kwargs):
#     """
#     Calculate SNR and fidelity for each NV using various methods such as GMM, Otsu, etc.,
#     with fidelity determined using combined counts from signal and reference data.

#     Parameters:
#     - nv_list (list): List of NV centers.
#     - raw_data (dict): Dictionary containing raw data, expected to have 'counts'.
#     - method (str): Method for fitting/thresholding. Supported: 'gmm', 'otsu', 'triangle', 'entropy'.
#     - plot (bool): Flag to enable or disable plotting for fits. Default is False.
#     - kwargs (dict): Additional arguments for thresholding methods.

#     Returns:
#     - dict: Metrics for each NV, including SNR, fidelity, and state probabilities.
#     """
#     from collections import defaultdict
#     import numpy as np
#     from skimage.filters import threshold_otsu, threshold_triangle, threshold_li

#     metrics = defaultdict(lambda: {"snr": [], "fidelity": [], "probability_nv_minus": [], "probability_nv_zero": []})
#     num_nvs = len(nv_list)
#     counts = np.array(raw_data["counts"])
#     sig_counts = counts[0]  # Assuming first index contains signal counts
#     ref_counts = counts[1]  # Assuming second index contains reference counts

#     for nv_ind in range(num_nvs):
#         # Combine signal and reference counts for thresholding
#         combined_counts = np.concatenate([sig_counts[nv_ind].flatten(), ref_counts[nv_ind].flatten()])

#         # Validate data
#         if len(combined_counts) == 0:
#             print(f"Warning: No data for NV index {nv_ind}. Skipping.")
#             continue

#         try:
#             # Determine threshold using the specified method on combined counts
#             if method == 'otsu':
#                 threshold = threshold_otsu(combined_counts)
#             elif method == 'triangle':
#                 threshold = threshold_triangle(combined_counts)
#             elif method == 'entropy':
#                 threshold = threshold_li(combined_counts)
#             else:
#                 raise ValueError(f"Method '{method}' is not supported.")

#             # Calculate probabilities using combined counts
#             p_nv_minus = np.sum(combined_counts >= threshold) / len(combined_counts)
#             p_nv_zero = 1 - p_nv_minus

#             # Calculate error rates based on thresholded counts
#             epsilon_0 = 1 - p_nv_minus  # False negative rate
#             epsilon_1 = p_nv_zero       # False positive rate

#             # Calculate Fidelity
#             fidelity = 1 - 0.5 * (epsilon_0 + epsilon_1)

#             # Calculate SNR using photon counts (assuming shot noise)
#             alpha_0 = np.mean(combined_counts[combined_counts >= threshold])  # Mean count for NV⁻ (|0>)
#             alpha_1 = np.mean(combined_counts[combined_counts < threshold])   # Mean count for NV⁰ (|1>)
#             if np.isnan(alpha_0) or np.isnan(alpha_1) or alpha_0 <= 0 or alpha_1 <= 0:
#                 print(f"Invalid alpha values for NV index {nv_ind}. Skipping.")
#                 continue

#             snr = (alpha_0 - alpha_1) / np.sqrt(alpha_0 + alpha_1)

#             # Append metrics
#             metrics[nv_ind]["snr"].append(snr)
#             metrics[nv_ind]["fidelity"].append(fidelity)
#             metrics[nv_ind]["probability_nv_minus"].append(p_nv_minus)
#             metrics[nv_ind]["probability_nv_zero"].append(p_nv_zero)

#         except Exception as e:
#             print(f"Error processing NV index {nv_ind}: {e}")
#             metrics[nv_ind]["snr"].append(None)
#             metrics[nv_ind]["fidelity"].append(None)
#             metrics[nv_ind]["probability_nv_minus"].append(None)
#             metrics[nv_ind]["probability_nv_zero"].append(None)

#     return metrics


# def save_high_res_figure(path, dpi=300, bbox_inches='tight', **kwargs):
#     """
#     Save a Matplotlib figure to the specified path with high resolution.

#     Parameters:
#     - path (str): Full file path to save the figure, including the file name and extension.
#     - dpi (int, optional): Dots per inch for the figure resolution. Default is 300.
#     - bbox_inches (str, optional): Bounding box option for tight layout. Default is 'tight'.
#     - **kwargs: Additional keyword arguments to pass to plt.savefig().
#     """
#     # Get the directory from the path
#     directory = os.path.dirname(path)

#     # Create the directory if it does not exist
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     # Save the figure
#     plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
#     print(f"Figure saved at {path} with resolution {dpi} DPI.")

# def visualize_metrics(metrics, datasets):
#     """Visualize SNR, fidelity, threshold, and probabilities for NV⁻ and NV⁰ for each NV across datasets."""
#     num_nvs = len(metrics)


#     # Convert metrics to arrays, ensuring data consistency
#     snr_data = np.array([metrics[i].get("snr", [None]) for i in range(num_nvs)])
#     fidelity_data = np.array([metrics[i].get("fidelity", [None]) for i in range(num_nvs)])
#     probability_nv_minus = np.array([metrics[i].get("probability_nv_minus", [None]) for i in range(num_nvs)])
#     probability_nv_zero = np.array([metrics[i].get("probability_nv_zero", [None]) for i in range(num_nvs)])

#     # Ensure arrays are not empty or filled with None values
#     snr_data = np.nan_to_num(np.array(snr_data, dtype=float), nan=np.nan)
#     fidelity_data = np.nan_to_num(np.array(fidelity_data, dtype=float), nan=np.nan)
#     probability_nv_minus = np.nan_to_num(np.array(probability_nv_minus, dtype=float), nan=np.nan)
#     probability_nv_zero = np.nan_to_num(np.array(probability_nv_zero, dtype=float), nan=np.nan)
#     # Compact Plotting for SNR, Fidelity, Threshold, and Probability metrics in a single row (4 columns)
#     plt.figure(figsize=(6, 6))

#     # NV⁻ Probability Heatmap
#     plt.subplot(1, 4, 1)
#     sns.heatmap(probability_nv_minus, cmap="Blues", annot=False, cbar_kws={"label": "Probability (NV⁻)"}, square=True)
#     plt.title("NV⁻ Probability", fontsize=10)
#     # plt.xlabel("Datasets", fontsize=9)
#     plt.ylabel("NV Index", fontsize=9)
#     plt.xticks([], fontsize=8)
#     plt.yticks(fontsize=8)  # Hide y-ticks for compactness
#     plt.gca().set_aspect(0.2)

#     # SNR Heatmap
#     plt.subplot(1, 4, 2)
#     sns.heatmap(snr_data, cmap="coolwarm", annot=False, cbar_kws={"label": "SNR"}, square=True)
#     plt.title("SNR", fontsize=10)
#     # plt.xlabel("Datasets", fontsize=9)
#     plt.xticks([], fontsize=8)
#     plt.yticks([], fontsize=8)
#     plt.gca().set_aspect(0.2)

#     # Fidelity Heatmap
#     plt.subplot(1, 4, 3)
#     sns.heatmap(fidelity_data, cmap="viridis", annot=False, cbar_kws={"label": "Fidelity"}, square=True)
#     plt.title("Fidelity", fontsize=10)
#     # plt.xlabel("Datasets", fontsize=9)
#     plt.xticks([], fontsize=8)
#     plt.yticks([], fontsize=8)  # Hide y-ticks for compactness
#     plt.gca().set_aspect(0.2)

#     # Threshold Heatmap
#     # plt.subplot(1, 4, 4)
#     # sns.heatmap(threshold_data, cmap="YlGnBu", annot=False, cbar_kws={"label": "Threshold"}, square=True)
#     # plt.title("Threshold", fontsize=10)
#     # # plt.xlabel("Datasets", fontsize=9)
#     # plt.xticks([], fontsize=8)
#     # plt.yticks([], fontsize=8)  # Hide y-ticks for compactness
#     # plt.gca().set_aspect(0.2)
#     # # Save the current figure with high resolution
#     # path = r"C:\Users\Saroj Chand\OneDrive\Documents\charge_state.png"
#     # save_high_res_figure(path)
#     # # Tight layout adjustment for compact fit
#     # plt.tight_layout(pad=0.01)
#     # plt.show()


#     # import matplotlib.pyplot as plt
#     # Scatter plot of SNR vs Fidelity with NV index labels
#     # Scatter plot of SNR vs Fidelity with NV index labels
#     cmap = plt.cm.get_cmap('viridis', snr_data.shape[1])
#     plt.figure(figsize=(6, 5))
#     for i in range(snr_data.shape[1]):
#         color = cmap(i)  # Assign a color from the colormap
#         plt.scatter(snr_data[:, i], fidelity_data[:, i], label=f"Dataset {i+1}", alpha=0.8, color=color, s=60, edgecolors="k")

#     # Set labels and title with elegant styling for SNR vs Fidelity
#     plt.xlabel("SNR", fontsize=12)
#     plt.ylabel("Fidelity", fontsize=12)
#     plt.title("SNR vs Fidelity Across Datasets", fontsize=14)


#     # Save the SNR vs Fidelity plot
#     path_fidelity = r"C:\Users\Saroj Chand\OneDrive\Documents\snr_vs_fidelity.png"
#     save_high_res_figure(path_fidelity)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.show()

#     # Scatter plot of Probability (NV⁻) vs SNR
#     plt.figure(figsize=(6, 5))
#     for i in range(snr_data.shape[1]):
#         color = cmap(i)
#         plt.scatter(snr_data[:, i], probability_nv_minus[:, i], label=f"Dataset {i+1}", alpha=0.8, color=color, s=60, edgecolors="k")

#     # Set labels and title with elegant styling for Probability vs SNR
#     plt.xlabel("SNR", fontsize=12)
#     plt.ylabel("Probability (NV⁻)", fontsize=12)
#     plt.title("Probability (NV⁻) vs SNR Across Datasets", fontsize=14)

#     # Save the Probability vs SNR plot
#     path_prob = r"C:\Users\Saroj Chand\OneDrive\Documents\prob_vs_snr.png"
#     save_high_res_figure(path_prob)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.show()

#     # Scatter plot of Probability (NV⁻) vs Fidelity
#     plt.figure(figsize=(6, 5))
#     for i in range(snr_data.shape[1]):
#         color = cmap(i)
#         plt.scatter(fidelity_data[:, i], probability_nv_minus[:, i], label=f"Dataset {i+1}", alpha=0.8, color=color, s=60, edgecolors="k")

#     # Set labels and title with elegant styling for Probability vs Fidelity
#     plt.xlabel("Fidelity", fontsize=12)
#     plt.ylabel("Probability (NV⁻)", fontsize=12)
#     plt.title("Probability (NV⁻) vs Fidelity Across Datasets", fontsize=14)

#     # Save the Probability vs Fidelity plot
#     path_prob_fidelity = r"C:\Users\Saroj Chand\OneDrive\Documents\prob_vs_fidelity.png"
#     save_high_res_figure(path_prob_fidelity)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.show()

# def process_multiple_datasets(dataset_ids):
#     """Process multiple datasets using dm to load data by ID."""
#     all_metrics = []

#     for dataset_id in dataset_ids:
#         # Load data from dm
#         raw_data = dm.get_raw_data(file_id=dataset_id)

#         # Extract nv_list from raw_data
#         nv_list = raw_data["nv_list"]

#         # Calculate metrics for the loaded dataset
#         metrics = calculate_metrics(nv_list, raw_data)
#         all_metrics.append(metrics)

#     # Combine metrics across datasets
#     combined_metrics = defaultdict(lambda: {"snr": [], "fidelity": [], "threshold": [], "probability_nv_minus": [], "probability_nv_zero": []})
#     for dataset_metrics in all_metrics:
#         for nv_ind, nv_metrics in dataset_metrics.items():
#             combined_metrics[nv_ind]["snr"].extend(nv_metrics["snr"])
#             combined_metrics[nv_ind]["fidelity"].extend(nv_metrics["fidelity"])
#             combined_metrics[nv_ind]["threshold"].extend(nv_metrics.get("threshold", []))
#             combined_metrics[nv_ind]["probability_nv_minus"].extend(nv_metrics["probability_nv_minus"])
#             combined_metrics[nv_ind]["probability_nv_zero"].extend(nv_metrics["probability_nv_zero"])

#     # Visualize metrics
#     visualize_metrics(combined_metrics, dataset_ids)

#     # Select the best dataset based on thresholds
#     best_dataset_index = select_best_dataset(combined_metrics)
#     return best_dataset_index

# def select_best_dataset(metrics, snr_threshold=1.0, fidelity_threshold=0.8):
#     """Selects the best dataset based on SNR and fidelity thresholds."""
#     best_dataset = None
#     max_valid_nvs = 0

#     for dataset_idx in range(len(metrics[0]["snr"])):
#         valid_nvs = sum(
#             1 for nv in metrics if metrics[nv]["snr"][dataset_idx] > snr_threshold
#             and metrics[nv]["fidelity"][dataset_idx] > fidelity_threshold
#         )

#         if valid_nvs > max_valid_nvs:
#             max_valid_nvs = valid_nvs
#             best_dataset = dataset_idx

#     print(f"Best dataset based on thresholds: Dataset {best_dataset + 1} with {max_valid_nvs} NVs meeting criteria.")
#     return best_dataset


# # Function to visualize SNR, Fidelity, and FWHM in 2D real space with color scale for first 11 NVs
# def visualize_metrics_in_2d_real_space(nv_list, metrics, fwhm_matrix):
#     snr_values = np.array([metrics[i]["snr"][0] for i in range(len(nv_list))])
#     fidelity_values = np.array([metrics[i]["fidelity"][0] for i in range(len(nv_list))])
#     avg_fwhm = fwhm_matrix[:len(nv_list)].mean(axis=1)

#     # Visualization for SNR in 2D Real Space
#     plt.figure(figsize=(8, 6))
#     plt.scatter([coord[0] for coord in nv_list], [coord[1] for coord in nv_list], c=snr_values, cmap='viridis', s=80, edgecolor='k', alpha=0.8)
#     plt.colorbar(label='SNR')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.title('SNR in 2D Real Space')
#     plt.show()

#     # Visualization for Fidelity in 2D Real Space
#     plt.figure(figsize=(8, 6))
#     plt.scatter([coord[0] for coord in nv_list], [coord[1] for coord in nv_list], c=fidelity_values, cmap='plasma', s=80, edgecolor='k', alpha=0.8)
#     plt.colorbar(label='Fidelity')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.title('Fidelity in 2D Real Space')
#     plt.show()

#     # Visualization for Average FWHM in 2D Real Space
#     plt.figure(figsize=(8, 6))
#     plt.scatter([coord[0] for coord in nv_list], [coord[1] for coord in nv_list], c=avg_fwhm, cmap='coolwarm', s=80, edgecolor='k', alpha=0.8)
#     plt.colorbar(label='Average FWHM (pixels)')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.title('Average FWHM in 2D Real Space')
#     plt.show()

# def remove_outliers(data, threshold=1.5):
#     """Remove outliers based on the IQR method."""
#     q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
#     iqr = q3 - q1
#     lower_bound, upper_bound = q1 - threshold * iqr, q3 + threshold * iqr
#     return (data >= lower_bound) & (data <= upper_bound)

# def visualize_fwhm_real_space(nv_list, fwhm_matrix):
#     """Visualize NV FWHM in real space with color scaling."""
#     avg_fwhm = np.mean(fwhm_matrix, axis=1)
#     mask = remove_outliers(avg_fwhm)
#     filtered_nv_list = [nv_list[i] for i in range(len(nv_list)) if mask[i]]
#     filtered_avg_fwhm = avg_fwhm[mask]

#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(
#         [coord[0] for coord in filtered_nv_list],
#         [coord[1] for coord in filtered_nv_list],
#         c=filtered_avg_fwhm, cmap='viridis', s=80, alpha=0.7, edgecolor='k'
#     )
#     plt.colorbar(scatter, label='Average FWHM (pixels)')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.title('Visualization of NV FWHM in Real Space')
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.show()

# def visualize_fitting_results(img_array, nv_coords, sigma=3.0, title="24ms, Ref"):
#     """Visualize fitted NV coordinates on an image array."""
#     fig, ax = plt.subplots()
#     kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
#     for idx, coord in enumerate(nv_coords):
#         circ = Circle(coord, sigma, color="lightblue", fill=False, linewidth=0.5)
#         ax.add_patch(circ)
#         ax.text(coord[0], coord[1] - sigma - 1, str(idx), color="white", fontsize=8, ha="center")
#     plt.show()

# def process_and_visualize_fwhm(nv_list, img_array):
#     """Process NV FWHM and visualize in real space."""
#     fwhm_matrix = [fit_gaussian_and_calculate_fwhm(img_array, coord)[-1] for coord in nv_list]
#     fwhm_matrix = np.array(fwhm_matrix)
#     visualize_fwhm_real_space(nv_list, fwhm_matrix)


# def visualize_fidelity_vs_fwhm(metrics, fwhm_matrix, nv_list):
#     """Visualize Fidelity vs Average FWHM for each NV."""
#     avg_fwhm = np.mean(fwhm_matrix, axis=1)  # Calculate average FWHM (mean of FWHM_x and FWHM_y)
#     mask = remove_outliers(avg_fwhm)
#     filtered_nv_list = [nv_list[i] for i in range(len(nv_list)) if mask[i]]
#     # Extract fidelity values for the first dataset (or adjust as needed)
#     fidelity_values = np.array([metrics[i]["fidelity"][0] for i in range(len(filtered_nv_list))])
#     filtered_avg_fwhm = avg_fwhm[mask]
#     # Create a scatter plot with color scale for average FWHM
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(
#         filtered_avg_fwhm,
#         fidelity_values,
#         c=fidelity_values,
#         cmap='viridis',
#         s=80,
#         alpha=0.7,
#         edgecolor='k'
#     )
#     plt.colorbar(scatter, label='Fidelity')
#     plt.xlabel('Average FWHM (pixels)', fontsize=12)
#     plt.ylabel('Fidelity', fontsize=12)
#     plt.title('Fidelity vs Average FWHM', fontsize=14)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.show(block=False)

# def visualize_nv_minus_probability_vs_fwhm(metrics, fwhm_matrix, nv_list):
#     """Visualize NV⁻ Probability vs Average FWHM for each NV."""

#     avg_fwhm = np.mean(fwhm_matrix, axis=1)  # Calculate average FWHM (mean of FWHM_x and FWHM_y)
#     mask = remove_outliers(avg_fwhm)
#     filtered_avg_fwhm = avg_fwhm[mask]
#     filtered_nv_list = [nv_list[i] for i in range(len(nv_list)) if mask[i]]
#     # Extract fidelity values for the first dataset (or adjust as needed)
#     filtered_avg_fwhm = avg_fwhm[mask]
#     probability_nv_minus = np.array([metrics[i]["probability_nv_minus"][0] for i in range(len(filtered_nv_list))])
#     avg_fwhm = np.mean(fwhm_matrix, axis=1)  # Calculate average FWHM (mean of FWHM_x and FWHM_y)

#     # Create a scatter plot with color scale for NV⁻ Probability
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(
#         filtered_avg_fwhm,
#         probability_nv_minus,
#         c=probability_nv_minus,
#         cmap='plasma',
#         s=80,
#         alpha=0.7,
#         edgecolor='k'
#     )
#     plt.colorbar(scatter, label='NV⁻ Probability')
#     plt.xlabel('Average FWHM (pixels)', fontsize=12)
#     plt.ylabel('NV⁻ Probability', fontsize=12)
#     plt.title('NV⁻ Probability vs Average FWHM', fontsize=14)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.show()


# # data = dm.get_raw_data(file_id=1694279622270, load_npz=True)
# data = dm.get_raw_data(file_id=1694881386605, load_npz=True)
# nv_list = data["nv_list"]
# valid_coords = [(nv.coords['pixel'][0], nv.coords['pixel'][1]) for nv in nv_list if isinstance(nv.coords.get('pixel', []), (list, tuple)) and len(nv.coords['pixel']) >= 2]
# img_array = np.array(data["ref_img_array"])
# fwhm_matrix = [fit_gaussian_and_calculate_fwhm(img_array, coord)[-1] for coord in valid_coords]
# fwhm_matrix = np.array(fwhm_matrix)
# metrics = calculate_metrics(nv_list, data)

# # Example usage with dataset IDs
# dataset_ids = [
#     # 1688554695897,
#     1694279622270,
#     # 1688505772462,
# ]
# # visualize_metrics(metrics, datasets)
# # visualize_fidelity_vs_fwhm(metrics, fwhm_matrix, nv_list)
# # visualize_nv_minus_probability_vs_fwhm(metrics, fwhm_matrix, nv_list)
# # Process the datasets
# best_dataset = process_multiple_datasets(dataset_ids)
# # print(f"Proceed with Dataset {best_dataset}")


# # # Create a figure for displaying equations only
# # plt.figure(figsize=(8, 6))

# # # Adding equations for SNR and Fidelity with definitions of ε₀ and ε₁
# # plt.text(
# #     0.5, 0.8, r"$\text{SNR} = \frac{p_{\text{NV}^- | \text{NV}^-} - p_{\text{NV}^- | \text{NV}^0}}{\sqrt{p_{\text{NV}^- | \text{NV}^-}(1 - p_{\text{NV}^- | \text{NV}^-}) + p_{\text{NV}^- | \text{NV}^0}(1 - p_{\text{NV}^- | \text{NV}^0})}}$",
# #     ha="center", color="black", fontsize=14,
# #     bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3")
# # )

# # plt.text(
# #     0.5, 0.6, r"$\text{If } \epsilon_{\text{NV}^-} = \epsilon_{\text{NV}^0}, \text{ SNR} = \frac{2\mathcal{F} - 1}{\sqrt{2\mathcal{F}(1 - \mathcal{F})}}$",
# #     ha="center", color="black", fontsize=14,
# #     bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3")
# # )

# # plt.text(
# #     0.5, 0.4, r"$\text{Fidelity } \mathcal{F} = 1 - \frac{1}{2} (\epsilon_{\text{NV}^-} + \epsilon_{\text{NV}^0})$",
# #     ha="center", color="black", fontsize=14,
# #     bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3")
# # )

# # plt.text(
# #     0.5, 0.2, r"$\text{where } \epsilon_{\text{NV}^-} = 1 - p_{\text{NV}^- | \text{NV}^-}, \; \epsilon_{\text{NV}^0} = p_{\text{NV}^- | \text{NV}^0}$",
# #     ha="center", color="black", fontsize=14,
# #     bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3")
# # )

# # # Remove axis
# # plt.axis('off')

# # # Save the figure with equations
# # path_equations = r"C:\Users\Saroj Chand\OneDrive\Documents\equations_only.png"
# # plt.savefig(path_equations, dpi=300, bbox_inches='tight')
# # plt.show()

# # #
# # img_array = np.array(data["ref_img_array"])
# # # img_array = -np.array(data["diff_img_array"])
# # nv_coordinates, integrated_intensities = load_nv_coords(
# #     file_path="slmsuite/nv_blob_detection/nv_blob_filtered_144nvs.npz"
# # )
# # nv_coordinates = nv_coordinates.tolist()
# # integrated_intensities = integrated_intensities.tolist()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.filters import threshold_li, threshold_otsu, threshold_triangle

from utils import data_manager as dm
from utils import widefield as widefield


def threshold_counts(nv_list, sig_counts, ref_counts=None, method="otsu"):
    """Threshold counts for NVs based on the selected method."""

    num_nvs = len(nv_list)
    sig_thresholds, ref_thresholds = [], []

    # Process thresholds based on the selected method
    for nv_ind in range(num_nvs):
        combined_counts = (
            np.append(sig_counts[nv_ind].flatten(), ref_counts[nv_ind].flatten())
            if ref_counts is not None
            else sig_counts[nv_ind].flatten()
        )

        # Choose method for thresholding
        if method == "otsu":
            threshold = threshold_otsu(combined_counts)
        elif method == "triangle":
            threshold = threshold_triangle(combined_counts)
        elif method == "entropy":
            threshold = threshold_li(combined_counts)
        else:
            raise ValueError(f"Unknown thresholding method: {method}")

        # Append threshold to the appropriate list
        sig_thresholds.append(threshold)

        # Optional: Compute ref threshold if needed separately
        if ref_counts is not None:
            ref_threshold = (
                threshold_otsu(ref_counts[nv_ind].flatten())
                if method == "otsu"
                else (
                    threshold_triangle(ref_counts[nv_ind].flatten())
                    if method == "triangle"
                    else threshold_li(ref_counts[nv_ind].flatten())
                )
            )
            ref_thresholds.append(ref_threshold)

    return sig_thresholds, ref_thresholds if ref_counts is not None else None


def calculate_metrics(nv_list, raw_data, method="otsu"):
    """Calculate SNR, fidelity, and ETX for each NV across multiple datasets."""
    metrics = defaultdict(lambda: {"snr": [], "fidelity": [], "etx": []})
    num_nvs = len(nv_list)
    counts = np.array(raw_data["counts"])

    # Compute optimal thresholds for all NVs using both sig and ref counts
    sig_thresholds, _ = threshold_counts(nv_list, counts[0], counts[1], method=method)

    for nv_ind in range(num_nvs):
        sig_counts_list = counts[0, nv_ind].flatten()
        ref_counts_list = counts[1, nv_ind].flatten()

        # Calculate SNR
        noise = np.sqrt(np.var(ref_counts_list) + np.var(sig_counts_list))
        signal = np.mean(ref_counts_list) - np.mean(sig_counts_list)
        snr = signal / noise if noise != 0 else 0

        # Calculate fidelity using the combined threshold
        threshold = sig_thresholds[nv_ind]
        fidelity = np.sum(sig_counts_list < threshold) / len(sig_counts_list)

        # Append metrics
        metrics[nv_ind]["snr"].append(snr)
        metrics[nv_ind]["fidelity"].append(fidelity)
        metrics[nv_ind]["etx"].append(threshold)

    return metrics


def visualize_metrics(metrics, datasets):
    """Visualize SNR, fidelity, threshold, and probabilities for NV⁻ and NV⁰ for each NV across datasets."""
    num_nvs = len(metrics)

    # Convert metrics to arrays, ensuring data consistency
    snr_data = np.array([metrics[i].get("snr", [None]) for i in range(num_nvs)])
    fidelity_data = np.array(
        [metrics[i].get("fidelity", [None]) for i in range(num_nvs)]
    )
    probability_nv_minus = np.array(
        [metrics[i].get("probability_nv_minus", [None]) for i in range(num_nvs)]
    )
    probability_nv_zero = np.array(
        [metrics[i].get("probability_nv_zero", [None]) for i in range(num_nvs)]
    )

    # Ensure arrays are not empty or filled with None values
    snr_data = np.nan_to_num(np.array(snr_data, dtype=float), nan=np.nan)
    fidelity_data = np.nan_to_num(np.array(fidelity_data, dtype=float), nan=np.nan)
    probability_nv_minus = np.nan_to_num(
        np.array(probability_nv_minus, dtype=float), nan=np.nan
    )
    probability_nv_zero = np.nan_to_num(
        np.array(probability_nv_zero, dtype=float), nan=np.nan
    )
    # Compact Plotting for SNR, Fidelity, Threshold, and Probability metrics in a single row (4 columns)
    plt.figure(figsize=(6, 6))

    # NV⁻ Probability Heatmap
    plt.subplot(1, 4, 1)
    sns.heatmap(
        probability_nv_minus,
        cmap="Blues",
        annot=False,
        cbar_kws={"label": "Probability (NV⁻)"},
        square=True,
    )
    plt.title("NV⁻ Probability", fontsize=10)
    # plt.xlabel("Datasets", fontsize=9)
    plt.ylabel("NV Index", fontsize=9)
    plt.xticks([], fontsize=8)
    plt.yticks(fontsize=8)  # Hide y-ticks for compactness
    plt.gca().set_aspect(0.2)

    # SNR Heatmap
    plt.subplot(1, 4, 2)
    sns.heatmap(
        snr_data, cmap="coolwarm", annot=False, cbar_kws={"label": "SNR"}, square=True
    )
    plt.title("SNR", fontsize=10)
    # plt.xlabel("Datasets", fontsize=9)
    plt.xticks([], fontsize=8)
    plt.yticks([], fontsize=8)
    plt.gca().set_aspect(0.2)

    # Fidelity Heatmap
    plt.subplot(1, 4, 3)
    sns.heatmap(
        fidelity_data,
        cmap="viridis",
        annot=False,
        cbar_kws={"label": "Fidelity"},
        square=True,
    )
    plt.title("Fidelity", fontsize=10)
    # plt.xlabel("Datasets", fontsize=9)
    plt.xticks([], fontsize=8)
    plt.yticks([], fontsize=8)  # Hide y-ticks for compactness
    plt.gca().set_aspect(0.2)

    # Threshold Heatmap
    # plt.subplot(1, 4, 4)
    # sns.heatmap(threshold_data, cmap="YlGnBu", annot=False, cbar_kws={"label": "Threshold"}, square=True)
    # plt.title("Threshold", fontsize=10)
    # # plt.xlabel("Datasets", fontsize=9)
    # plt.xticks([], fontsize=8)
    # plt.yticks([], fontsize=8)  # Hide y-ticks for compactness
    # plt.gca().set_aspect(0.2)
    # # Save the current figure with high resolution
    # path = r"C:\Users\Saroj Chand\OneDrive\Documents\charge_state.png"
    # save_high_res_figure(path)
    # # Tight layout adjustment for compact fit
    # plt.tight_layout(pad=0.01)
    # plt.show()

    # import matplotlib.pyplot as plt
    # Scatter plot of SNR vs Fidelity with NV index labels
    # Scatter plot of SNR vs Fidelity with NV index labels
    cmap = plt.cm.get_cmap("viridis", snr_data.shape[1])
    plt.figure(figsize=(6, 5))
    for i in range(snr_data.shape[1]):
        color = cmap(i)  # Assign a color from the colormap
        plt.scatter(
            snr_data[:, i],
            fidelity_data[:, i],
            label=f"Dataset {i+1}",
            alpha=0.8,
            color=color,
            s=60,
            edgecolors="k",
        )

    # Set labels and title with elegant styling for SNR vs Fidelity
    plt.xlabel("SNR", fontsize=12)
    plt.ylabel("Fidelity", fontsize=12)
    plt.title("SNR vs Fidelity Across Datasets", fontsize=14)

    # Save the SNR vs Fidelity plot
    # path_fidelity = r"C:\Users\Saroj Chand\OneDrive\Documents\snr_vs_fidelity.png"
    # save_high_res_figure(path_fidelity)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.show()

    # Scatter plot of Probability (NV⁻) vs SNR
    plt.figure(figsize=(6, 5))
    for i in range(snr_data.shape[1]):
        color = cmap(i)
        plt.scatter(
            snr_data[:, i],
            probability_nv_minus[:, i],
            label=f"Dataset {i+1}",
            alpha=0.8,
            color=color,
            s=60,
            edgecolors="k",
        )

    # Set labels and title with elegant styling for Probability vs SNR
    plt.xlabel("SNR", fontsize=12)
    plt.ylabel("Probability (NV⁻)", fontsize=12)
    plt.title("Probability (NV⁻) vs SNR Across Datasets", fontsize=14)

    # Save the Probability vs SNR plot
    # path_prob = r"C:\Users\Saroj Chand\OneDrive\Documents\prob_vs_snr.png"
    # save_high_res_figure(path_prob)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.show()

    # Scatter plot of Probability (NV⁻) vs Fidelity
    plt.figure(figsize=(6, 5))
    for i in range(snr_data.shape[1]):
        color = cmap(i)
        plt.scatter(
            fidelity_data[:, i],
            probability_nv_minus[:, i],
            label=f"Dataset {i+1}",
            alpha=0.8,
            color=color,
            s=60,
            edgecolors="k",
        )

    # Set labels and title with elegant styling for Probability vs Fidelity
    plt.xlabel("Fidelity", fontsize=12)
    plt.ylabel("Probability (NV⁻)", fontsize=12)
    plt.title("Probability (NV⁻) vs Fidelity Across Datasets", fontsize=14)

    # Save the Probability vs Fidelity plot
    # path_prob_fidelity = r"C:\Users\Saroj Chand\OneDrive\Documents\prob_vs_fidelity.png"
    # save_high_res_figure(path_prob_fidelity)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.show()


# def visualize_metrics(metrics, datasets):
#     """Visualize SNR, fidelity, and ETX for each NV across datasets."""
#     num_nvs = len(metrics)

#     # Convert metrics to arrays for easier plotting
#     snr_data = np.array([metrics[i]["snr"] for i in range(num_nvs)])
#     fidelity_data = np.array([metrics[i]["fidelity"] for i in range(num_nvs)])
#     etx_data = np.array([metrics[i]["etx"] for i in range(num_nvs)])

#     # Plotting SNR and Fidelity across NVs and datasets
#     plt.figure(figsize=(14, 5))

#     # SNR Heatmap
#     plt.subplot(1, 3, 1)
#     sns.heatmap(snr_data, cmap="coolwarm", annot=False, cbar_kws={"label": "SNR"})
#     plt.title("SNR Across NVs and Datasets")
#     plt.xlabel("Datasets")
#     plt.ylabel("NV Index")

#     # Fidelity Heatmap
#     plt.subplot(1, 3, 2)
#     sns.heatmap(fidelity_data, cmap="viridis", annot=False, cbar_kws={"label": "Fidelity"})
#     plt.title("Fidelity Across NVs and Datasets")
#     plt.xlabel("Datasets")

#     # ETX Heatmap
#     plt.subplot(1, 3, 3)
#     sns.heatmap(etx_data, cmap="YlGnBu", annot=False, cbar_kws={"label": "ETX"})
#     plt.title("ETX Across NVs and Datasets")
#     plt.xlabel("Datasets")

#     plt.tight_layout()
#     plt.show()

#     # Scatter plot of SNR vs Fidelity for dataset selection
#     plt.figure(figsize=(10, 6))
#     for i in range(snr_data.shape[1]):
#         plt.scatter(snr_data[:, i], fidelity_data[:, i], label=f"Dataset {i+1}", alpha=0.7)
#     plt.xlabel("SNR")
#     plt.ylabel("Fidelity")
#     plt.legend()
#     plt.title("SNR vs Fidelity Across Datasets")
#     plt.show()


def select_best_dataset(metrics, snr_threshold=1.0, fidelity_threshold=0.9):
    """Selects the best dataset based on SNR and fidelity thresholds."""
    best_dataset = None
    max_valid_nvs = 0

    for dataset_idx in range(len(metrics[0]["snr"])):
        valid_nvs = sum(
            1
            for nv in metrics
            if metrics[nv]["snr"][dataset_idx] > snr_threshold
            and metrics[nv]["fidelity"][dataset_idx] > fidelity_threshold
        )

        if valid_nvs > max_valid_nvs:
            max_valid_nvs = valid_nvs
            best_dataset = dataset_idx

    print(
        f"Best dataset based on thresholds: Dataset {best_dataset + 1} with {max_valid_nvs} NVs meeting criteria."
    )
    return best_dataset


def process_multiple_datasets(dataset_ids):
    """Process multiple datasets using dm to load data by ID."""
    all_metrics = []

    for dataset_id in dataset_ids:
        # Load data from dm
        raw_data = dm.get_raw_data(file_id=dataset_id)

        # Extract nv_list from raw_data
        nv_list = raw_data["nv_list"]

        # Calculate metrics for the loaded dataset
        metrics = calculate_metrics(nv_list, raw_data)
        all_metrics.append(metrics)

    # Combine metrics across datasets
    combined_metrics = defaultdict(lambda: {"snr": [], "fidelity": [], "etx": []})
    for dataset_metrics in all_metrics:
        for nv_ind, nv_metrics in dataset_metrics.items():
            combined_metrics[nv_ind]["snr"].extend(nv_metrics["snr"])
            combined_metrics[nv_ind]["fidelity"].extend(nv_metrics["fidelity"])
            combined_metrics[nv_ind]["etx"].extend(nv_metrics["etx"])

    # Visualize metrics
    visualize_metrics(combined_metrics, dataset_ids)

    # Select the best dataset based on thresholds
    best_dataset_index = select_best_dataset(combined_metrics)
    return best_dataset_index


# Example usage with dataset IDs
dataset_ids = [
    1694881386605,
    # 1688554695897,
    # 1688505772462,
    # Add more dataset IDs as needed
]

# Process the datasets
best_dataset = process_multiple_datasets(dataset_ids)
print(f"Proceed with Dataset {best_dataset + 1}")
