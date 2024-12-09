
from utils import data_manager as dm
from utils import kplotlib as kpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from scipy.optimize import curve_fit, least_squares
from scipy.ndimage import gaussian_filter1d
from utils import widefield as widefield


def exp_decay(tau, norm, rate, offset=0):
    """Exponential decay function."""
    return norm * np.exp(-rate * tau) + offset

def double_exp_decay(tau, norm1, rate1, norm2, rate2, offset=0):
    """Double exponential decay function for more complex fitting scenarios."""
    return norm1 * np.exp(-rate1 * tau) + norm2 * np.exp(-rate2 * tau) + offset

def robust_fit(tau, y, yerr=None, model='single'):
    """Performs robust fitting using least squares with different loss functions."""
    if not np.isfinite(y).all() or not np.isfinite(tau).all():
        raise ValueError("Input data contains non-finite values.")

    epsilon = 1e-10  # Small value to prevent division by zero
    
    if model == 'single':
        def residuals(params):
            return (exp_decay(tau, *params) - y) / (yerr + epsilon if yerr is not None else 1)
        initial_guess = [y[0], 1 / (tau[-1] - tau[0]), 0]
    elif model == 'double':
        def residuals(params):
            return (double_exp_decay(tau, *params) - y) / (yerr + epsilon if yerr is not None else 1)
        initial_guess = [y[0] / 2, 1 / (tau[-1] - tau[0]), y[0] / 2, 1 / (tau[-1] - tau[0]), 0]
    else:
        raise ValueError("Invalid model specified.")

    # print("Initial guess:", initial_guess)  # Debugging print statement

    result = least_squares(residuals, initial_guess, loss='soft_l1')
    return result.x, result.success

def process_and_fit_data(data, use_double_fit=False):
    """Processes and fits NV relaxation data with robust fitting."""
    nv_list = data["nv_list"]
    taus = np.array(data["taus"]) / 1e6  # Convert ns to ms
    counts = np.array(data["states"])
    sig_counts, ref_counts = counts[0], counts[1]  # Extract the two states

    # Process counts using widefield's process_counts function
    sig_avg_counts, sig_avg_counts_ste, _ = widefield.process_counts(
        nv_list, sig_counts, threshold=False
    )
    ref_avg_counts, ref_avg_counts_ste, _ = widefield.process_counts(
        nv_list, ref_counts, threshold=False
    )

    # Compute the difference between the states
    diff_counts = ref_avg_counts - sig_avg_counts
    diff_counts_ste = np.sqrt(sig_avg_counts_ste**2 + ref_avg_counts_ste**2)

    # Normalize counts (if normalization is desired; otherwise, skip this step)
    norm_counts = diff_counts #/ np.max(ref_avg_counts, axis=1, keepdims=True)
    norm_counts_ste = diff_counts_ste #/ np.max(ref_avg_counts, axis=1, keepdims=True)

    fit_params, fit_functions, residuals = [], [], []

    for nv_idx in range(len(nv_list)):
        nv_counts = gaussian_filter1d(norm_counts[nv_idx], sigma=1)  # Smoothing for stability
        nv_counts_ste = norm_counts_ste[nv_idx]

        # Try single exponential first
        params, success = robust_fit(taus, nv_counts, yerr=nv_counts_ste, model='single')
        if not success and use_double_fit:
            params, success = robust_fit(taus, nv_counts, yerr=nv_counts_ste, model='double')

        fit_curve = exp_decay(taus, *params) if len(params) == 3 else double_exp_decay(taus, *params)
        fit_params.append(params)
        fit_functions.append(lambda t, p=params: exp_decay(t, *p[:3]) if len(p) == 3 else double_exp_decay(t, *p))
        residuals.append(nv_counts - fit_curve)

    return fit_params, fit_functions, residuals, taus, norm_counts, norm_counts_ste, nv_list

def plot_fitted_data(nv_list, taus, norm_counts, norm_counts_ste, fit_functions):
    """Plot for raw data with fitted curves using Seaborn style, including NV index labels."""
    sns.set(style="whitegrid")
    num_nvs = len(nv_list)
    num_cols = 6
    num_rows = int(np.ceil(num_nvs / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    axes = axes.flatten()

    for nv_idx in range(num_nvs):
        ax = axes[nv_idx]
        
        # Plot data with Seaborn's lineplot for a cleaner look
        sns.lineplot(
            x=taus,
            y=norm_counts[nv_idx],
            ax=ax,
            marker='o',
            color='blue',
            label=f'NV {nv_idx + 1}',
            lw=1,
            markersize=4,
        )
        
        # Plot error bars separately for clarity
        ax.errorbar(taus, norm_counts[nv_idx], yerr=norm_counts_ste[nv_idx], fmt='o', alpha=0.5, ecolor='gray')

        # Plot fitted curve if available
        if fit_functions[nv_idx]:
            fit_curve = fit_functions[nv_idx](taus)
            sns.lineplot(
                x=taus,
                y=fit_curve,
                ax=ax,
                color='red',
                # label='Fit',
                lw=2,
            )
        
        # Add NV index within the plot at the center
        # ax.text(
        #     0.5, 0.5, f'NV {nv_idx + 1}',
        #     transform=ax.transAxes, fontsize=10, va='center', ha='center',
        #     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        # )
        
        # Only show y-axis label on the leftmost subplots
        # if nv_idx % num_cols == 0:
        #     ax.set_ylabel('Normalized Counts')
        # else:
        #     ax.set_yticklabels([])  # Remove y-tick labels for compactness
        ax.set_yticklabels([])

        # Only show x-axis label on the bottommost subplots
        if nv_idx >= (num_rows - 1) * num_cols:
            ax.set_xlabel('Relaxation Time (ms)')
        else:
            ax.set_xticklabels([])  # Remove x-tick labels for compactness

        ax.legend(fontsize='small')

    # Hide unused subplots
    for ax in axes[num_nvs:]:
        ax.axis('off')

    # plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.01, wspace=0.01)

    # Save the figure with a timestamped filename
    plt.tight_layout()
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    file_name = dm.get_file_name(file_id=file_id)
    file_path = dm.get_file_path(__file__, file_name, f"{file_id}_{date_time_str}")
    # kpl.show(block=True)
    dm.save_figure(fig, file_path)
    # plt.close(fig_fitting)
    # plt.show(block=True) 

def plot_fitted_data_separately(nv_list, taus, norm_counts, norm_counts_ste, fit_functions):
    """Plot separate figures for each NV with fitted curves using Seaborn style."""
    sns.set(style="whitegrid")

    for nv_idx, nv_name in enumerate(nv_list):
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Plot data with Seaborn's lineplot for a cleaner look
        sns.scatterplot(
            x=taus,
            y=norm_counts[nv_idx],
            ax=ax,
            color='blue',
            label=f'NV {nv_idx + 1}',
            s=50,  # Size of markers
            alpha=0.7,
        )
        
        # Plot error bars separately for clarity
        ax.errorbar(taus, norm_counts[nv_idx], yerr=norm_counts_ste[nv_idx], fmt='o', alpha=0.5, ecolor='gray')

        # Plot fitted curve if available
        if fit_functions[nv_idx]:
            fit_curve = fit_functions[nv_idx](taus)
            sns.lineplot(
                x=taus,
                y=fit_curve,
                ax=ax,
                color='red',
                # label='Fit',
                lw=2,
            )

        # Add titles and labels
        ax.set_title(f'NV {nv_idx + 1} Relaxation Data', fontsize=12)
        ax.set_xlabel('Relaxation Time (ms)')
        ax.set_ylabel('Normalized Counts')
        ax.legend(fontsize='small')
        ax.grid(True, linestyle="--", linewidth=0.5)

        # Save each figure with a unique name
        now = datetime.now()
        date_time_str = now.strftime("%Y%m%d_%H%M%S")
        file_name = f'nv_{nv_idx + 1}_fitted_plot_{date_time_str}.png'  # Example filename
        # fig.savefig(file_name)
        # plt.close(fig)  # Close the figure after saving to avoid display issues
        plt.show(block=True)

    print("Plots created and saved.")


# Usage Example
if __name__ == "__main__":
    kpl.init_kplotlib()
    file_id=1700199606781
    # file_id=1700092000259
    # file_id= 1550610460299
    # data = dm.get_raw_data(file_id=1550610460299)  # Example file ID
    data = dm.get_raw_data(file_id=file_id)  # Example file ID
    fit_params, fit_functions, residuals, taus, norm_counts, norm_counts_ste, nv_list = process_and_fit_data(data, use_double_fit=True)
    # plot_fitted_data(nv_list, taus, norm_counts, norm_counts_ste, fit_functions)
    plot_fitted_data_separately(nv_list, taus, norm_counts, norm_counts_ste, fit_functions)
