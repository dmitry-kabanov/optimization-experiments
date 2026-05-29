#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib
# Use non-interactive Agg backend to avoid GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the exact solution function from the exact module
from exact import compute_viscous_burgers_with_ic_and_periodic_bc_spectral

def load_predictions(out_dir):
    """
    Finds directories in out_dir that have split_key=true and loads predictions.npy.
    Categorizes them by implementation (scipy_optimize vs optim_jl) and gtol.
    """
    scipy_preds = {}
    optim_preds = {}
    
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Output directory '{out_dir}' does not exist.")
        
    for d in os.listdir(out_dir):
        dir_path = os.path.join(out_dir, d)
        if not os.path.isdir(dir_path) or os.path.islink(dir_path):
            continue
            
        config_path = os.path.join(dir_path, "config.json")
        pred_path = os.path.join(dir_path, "predictions.npy")
        
        if not os.path.exists(config_path) or not os.path.exists(pred_path):
            continue
            
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            continue
            
        impl = config.get("impl")
        split_key = config.get("split_key")
        gtol = config.get("gtol")
        
        # Check if split_key is true
        if split_key is True:
            try:
                # Load array and flatten to 1D
                pred_data = np.load(pred_path).squeeze()
                if impl == "scipy_optimize":
                    scipy_preds[gtol] = pred_data
                elif impl == "optim_jl":
                    optim_preds[gtol] = pred_data
            except Exception as e:
                print(f"Warning: Failed to load predictions from {pred_path}: {e}")
                continue
                
    return scipy_preds, optim_preds

def get_pred_by_tol(preds_dict, tol):
    """
    Helper to robustly find a prediction array in a dictionary by float tolerance key.
    """
    for k, v in preds_dict.items():
        if np.isclose(k, tol):
            return v
    return None

def main():
    out_dir = "_output"
    print("Loading predictions from output directory...")
    scipy_preds, optim_preds = load_predictions(out_dir)
    
    # 3. Compute exact solution of the Burgers' equation
    # Parameters are exactly those in exact.py's __main__ block:
    #   x, dx = np.linspace(0, 2, 256, endpoint=False, retstep=True)
    #   u0 = 0.5 - 0.25 * np.sin(np.pi * x)
    #   nu = 0.01 / np.pi
    #   tfinal = 2.0
    print("Computing exact solution...")
    x_exact, dx_exact = np.linspace(0, 2, 256, endpoint=False, retstep=True)
    u0_exact = 0.5 - 0.25 * np.sin(np.pi * x_exact)
    nu_exact = 0.01 / np.pi
    tfinal_exact = 2.0
    exact_sol = compute_viscous_burgers_with_ic_and_periodic_bc_spectral(
        u0_exact, x_exact, dx_exact, tfinal_exact, nu_exact
    )
    
    # Grid used for prediction values (endpoint=True by default in train script)
    x_pred = np.linspace(0.0, 2.0, 256)
    
    # Tolerances to plot
    tols = [1e-3, 1e-4, 1e-5]
    
    # 4. Create Matplotlib figure with premium styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#555555'
    plt.rcParams['ytick.color'] = '#555555'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    
    # Color scheme and styles
    exact_color = '#1C1C1C' # Rich black for exact reference
    colors_by_tol = {
        1e-3: '#E76F51', # Warm Coral
        1e-4: '#2A9D8F', # Sage Teal
        1e-5: '#457B9D'  # Slate Blue
    }
    
    linestyles_by_tol = {
        1e-3: '--',  # Dashed
        1e-4: '-.',  # Dash-dot
        1e-5: ':'    # Dotted
    }
    
    labels_by_tol = {
        1e-3: r'PINN ($\mathrm{gtol}=10^{-3}$)',
        1e-4: r'PINN ($\mathrm{gtol}=10^{-4}$)',
        1e-5: r'PINN ($\mathrm{gtol}=10^{-5}$)'
    }
    
    # Plotting helper
    def plot_impl(ax, preds_dict, title):
        # Plot exact solution
        ax.plot(x_exact, exact_sol, color=exact_color, linestyle='-', linewidth=2.0, label='Exact solution')
        
        # Plot predictions for each gtol
        for tol in tols:
            pred_arr = get_pred_by_tol(preds_dict, tol)
            if pred_arr is not None:
                ax.plot(
                    x_pred, 
                    pred_arr, 
                    color=colors_by_tol[tol], 
                    linestyle=linestyles_by_tol[tol], 
                    linewidth=1.8, 
                    label=labels_by_tol[tol]
                )
            else:
                print(f"Warning: No prediction found for tol={tol} in {title}")
                
        # Design & labels
        # ax.set_title(title, fontsize=14, pad=15, fontweight='bold', color='#2C3E50')
        ax.set_xlabel(r'$x$', fontsize=12, labelpad=8)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#E5E5E5')
        
        # Clean look: remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    # Plot left: scipy_optimize
    plot_impl(ax1, scipy_preds, 'SciPy Optimize (BFGS)')
    ax1.set_ylabel(r'$u(t=2, x)$', fontsize=12, labelpad=8)
    
    # Plot right: optim_jl
    plot_impl(ax2, optim_preds, 'Optim.jl (BFGS)')
    
    # For the left subfigure put "A" in the left top corner of the plot,
    # and for the right subfigure put "B" in the left top corner.
    # We place them at 3% from left, 95% from bottom in axes coordinates.
    ax1.text(0.03, 0.95, 'A', transform=ax1.transAxes, fontsize=18, fontweight='bold', va='top', ha='left',
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'))
    ax2.text(0.03, 0.95, 'B', transform=ax2.transAxes, fontsize=18, fontweight='bold', va='top', ha='left',
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Legend for the subfigures
    ax1.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#E5E5E5', framealpha=0.9, fontsize=10)
    ax2.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#E5E5E5', framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    
    # 5. Save the figure as vbe-bfgs-predictions.pdf
    pdf_filename = "vbe-bfgs-predictions.pdf"
    plt.savefig(pdf_filename, dpi=300)
    print(f"Figure successfully saved to: {pdf_filename}")

if __name__ == "__main__":
    main()
