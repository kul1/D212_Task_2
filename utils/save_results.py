import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from utils.config_loader import load_config

# Load the config globally for use throughout the module
config = load_config()

def ensure_directory_exists(directory):
    """Ensure that a directory exists; create it if it doesn't."""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        print(f"Directory checked/created: {directory}")
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")

def save_anova_results(anova_results):
    try:
        bivariant_dir = config.BIVARIANT_DIR  # Fetch from config
        ensure_directory_exists(bivariant_dir)
        anova_results_df = pd.concat(anova_results).reset_index()
        anova_results_df.columns = ['Variable', 'index', 'df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)']
        anova_results_df.to_csv(os.path.join(bivariant_dir, 'anova_results.csv'), index=False)
        print("ANOVA results saved to 'bivariant/anova_results.csv'")
    except Exception as e:
        print(f"Error saving ANOVA results: {e}")

def save_chi_square_results(chi_square_results, file_name='chi_square_results.csv'):
    try:
        bivariant_dir = config.BIVARIANT_DIR  # Fetch from config
        ensure_directory_exists(bivariant_dir)
        file_path = os.path.join(bivariant_dir, file_name)
        with open(file_path, 'w') as file:
            file.write("### Chi-Square Test Results ###\n\n")
            for key, value in chi_square_results.items():
                file.write(f'{key}: Chi2={value[0]}, p-value={value[1]}, dof={value[2]}\n')
                file.write("Expected frequencies:\n" + str(value[3]) + "\n\n")
        print(f"Chi-Square test results saved to '{file_path}'")
    except Exception as e:
        print(f"Error saving Chi-Square results: {e}")

def save_residual_plots_and_rse(residuals, rse, model, model_name='initial'):
    try:
        assumptions_dir = config.ASSUMPTIONS_DIR  # Fetch from config
        ensure_directory_exists(assumptions_dir)

        # Save Residual Standard Error
        rse_filepath = os.path.join(assumptions_dir, f'{model_name}_rse.txt')
        with open(rse_filepath, 'w') as file:
            file.write(f"Residual Standard Error: {rse}\n")
        print(f"Residual Standard Error saved to {rse_filepath}")

        # Residuals vs Fitted plot
        plt.figure(figsize=(10, 6))
        sns.residplot(x=model.fittedvalues, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 2})
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residuals vs Fitted - {model_name.capitalize()} Model')
        residuals_vs_fitted_filepath = os.path.join(assumptions_dir, f'{model_name}_residuals_vs_fitted.png')
        plt.savefig(residuals_vs_fitted_filepath)
        plt.close()
        print(f"Residuals vs Fitted plot saved to {residuals_vs_fitted_filepath}")

        # Q-Q plot
        qq_plot_filepath = os.path.join(assumptions_dir, f'{model_name}_qq_plot.png')
        sm.qqplot(residuals, line='45')
        plt.title(f'Q-Q Plot - {model_name.capitalize()} Model')
        plt.savefig(qq_plot_filepath)
        plt.close()

        print(f"Q-Q Plot saved to {qq_plot_filepath}")
    except Exception as e:
        print(f"Error saving residual plots and RSE: {e}")

def save_initial_regression_results(model, mse, r_squared, model_name='initial'):
    try:
        results_path = config.RESULTS_DIR  # Fetch from config
        ensure_directory_exists(results_path)

        with open(os.path.join(results_path, f'{model_name}_regression_summary.txt'), 'w') as file:
            file.write(model.summary().as_text())

        with open(os.path.join(results_path, f'{model_name}_mse_r_squared.txt'), 'w') as file:
            file.write(f"Mean Squared Error: {mse}\n")
            file.write(f"R-squared: {r_squared}\n")

        print(f"{model_name.capitalize()} regression results saved.")
    except Exception as e:
        print(f"Error saving initial regression results: {e}")

def save_significant_vars(selected_vars, model_name='lasso'):
    try:
        results_path = config.RESULTS_DIR  # Fetch from config
        ensure_directory_exists(results_path)

        with open(os.path.join(results_path, f'{model_name}_significant_vars.txt'), 'w') as file:
            file.write("Significant Variables:\n")
            for var in selected_vars:
                file.write(f"{var}\n")
        print(f"Significant variables for {model_name} model saved.")
    except Exception as e:
        print(f"Error saving significant variables for {model_name} model: {e}")

def save_vif_results(vif_results, file_name='vif_results.csv'):
    try:
        prepared_dir = config.PREPARED_DATA_DIR  # Fetch from config
        ensure_directory_exists(prepared_dir)
        file_path = os.path.join(prepared_dir, file_name)
        vif_results.to_csv(file_path, index=False)
        print(f"VIF results saved to '{file_path}'")
    except Exception as e:
        print(f"Error saving VIF results: {e}")

def finalize_and_save_results(initial_model, reduced_model, final_vars, reduction_method=None):
    """
    Finalizes and saves results for both initial and reduced models.

    Parameters:
    - initial_model: The initial regression model (before any variable reduction).
    - reduced_model: The reduced regression model (after variable selection).
    - final_vars: The variables selected for the reduced model.
    - reduction_method: The reduction method used (e.g., 'lasso', 'backward_elimination').
    """
    try:
        # Save initial model results
        save_initial_regression_results(initial_model, initial_model.mse_resid, initial_model.rsquared, model_name='initial')

        if reduced_model:
            save_initial_regression_results(reduced_model, reduced_model.mse_resid, reduced_model.rsquared, model_name=f'reduced_{reduction_method}')
            print(f"\n### Comparison ###")
            print(f"Initial Model R-squared: {initial_model.rsquared}, MSE: {initial_model.mse_resid}")
            print(f"Reduced Model R-squared: {reduced_model.rsquared}, MSE: {reduced_model.mse_resid}")
        else:
            print("\n### No Reduced Model to Compare ###")

    except Exception as e:
        print(f"Error finalizing and saving results: {e}")
