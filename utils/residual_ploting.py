# File Path: utils/residual_plotting.py

from utils.save_results import save_residual_plot

def plot_all_residuals(reduced_results_dict, y, config):
    """Generates and saves residual plots for all reduced models."""
    for method, (reduced_model, _) in reduced_results_dict.items():
        if isinstance(reduced_model, sm.regression.linear_model.RegressionResultsWrapper):
            residuals = y - reduced_model.fittedvalues
            save_residual_plot(reduced_model, residuals, method, config)
        else:
            print(f"Warning: Model does not have fitted values. Unable to plot residuals for {method.capitalize()}.")

