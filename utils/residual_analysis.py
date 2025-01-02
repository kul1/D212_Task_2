# File: utils/residual_analysis.py

import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_rse(model, y_true):
    """
    Calculate Residual Standard Error (RSE) for a given model.

    Parameters:
    - model: The fitted regression model (from OLS).
    - y_true: The actual target values.

    Returns:
    - rse: Residual Standard Error value.
    """
    # Check if the model has fitted values to avoid calculation for invalid models
    if hasattr(model, 'fittedvalues'):
        residuals = y_true - model.fittedvalues
        rss = np.sum(np.square(residuals))
        rse = np.sqrt(rss / (len(y_true) - len(model.params)))
        return rse
    else:
        print("Warning: Model does not have fitted values. Unable to calculate RSE.")
        return None

def plot_residuals(model, title, y_true, config, method_name):
    """
    Plot residuals for a given regression model.

    Parameters:
    - model: The fitted regression model.
    - title: The title of the plot.
    - y_true: The actual target values.
    - config: Configuration containing file paths for saving results.
    - method_name: Method name to append to the plot filename.
    """
    # Check if the model has fitted values
    if hasattr(model, 'fittedvalues'):
        residuals = y_true - model.fittedvalues

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(model.fittedvalues, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.title(title)

        # Ensure results directory exists
        residual_plot_dir = os.path.join(config.RESULTS_DIR, 'visuals')
        os.makedirs(residual_plot_dir, exist_ok=True)

        # Save plot with unique name using method_name
        residual_plot_filename = f'residual_plot_{method_name}.png'
        residual_plot_path = os.path.join(residual_plot_dir, residual_plot_filename)
        plt.savefig(residual_plot_path)
        plt.close()
        print(f"Residual plot saved to: {residual_plot_path}")
    else:
        print(f"Warning: Model does not have fitted values. Unable to plot residuals for {title}.")

