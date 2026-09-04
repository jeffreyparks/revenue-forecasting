"""
Model Diagnostics Viewer

A Streamlit app for examining fitted statsmodels regression objects.
Loads models from joblib files and displays:
- Model information (brand, model, estimator, base estimator, parameters)
- Coefficient table with standard errors, t-values, and p-values
- Model fit statistics (R², Adj R², F-statistic, AIC, BIC)
- Residual analysis plots (residuals vs fitted, Q-Q plot, histogram, Cook's distance)
- LLM-powered analysis and recommendations

Usage:
    streamlit run analyze_models.py

Requirements:
    - Fitted model objects saved as joblib files in data/output/
    - Models stored as DataFrames with columns:
        - brand, model, response: Model identifiers
        - estimator: User-defined unique estimator name (e.g., 'RLM_Tukey')
        - base_estimator: Estimator type code (e.g., 'OLS', 'WLS', 'RLM', 'PCA')
        - params: JSON string of parameters used (optional)
        - R2, R2_CV, n_obs, n_outliers, n_features, wls_k: Model metrics
        - SW_stat, SW_p, BP_stat, BP_p, DW, cooks_n, VIF: Diagnostic stats
        - model_obj: Fitted Pipeline object
    - ANTHROPIC_API_KEY environment variable for LLM analysis
"""

import io
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import anthropic
import matplotlib.pyplot as plt
import openai
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from joblib import load as joblib_load
from scipy import stats

# Load environment variables from .env file
load_dotenv()

# Suppress numpy warnings from statsmodels influence calculations
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add parent directory to path to import functions
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import lr_models so joblib can find custom model classes (StatsmodelsOLS, StatsmodelsWLS)
import functions.lr_models  # noqa: E402, F401

# Constants
APP_DIR = Path(__file__).parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "output"
FIGURE_SIZE = (12, 5)
COOKS_THRESHOLD_FACTOR = 4  # Cook's distance threshold = factor / n
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
OPENAI_MODEL = "gpt-4o"
LLM_PROVIDERS = ["Anthropic (Claude)", "OpenAI (GPT-4o)"]


def safe_get_resid(results):
    """Safely get residuals from a results object. Returns None if unavailable."""
    try:
        return results.resid
    except AttributeError:
        return None


def safe_get_fittedvalues(results):
    """Safely get fitted values from a results object. Returns None if unavailable."""
    try:
        return results.fittedvalues
    except AttributeError:
        # Try to compute from model
        try:
            return results.predict(results.model.exog)
        except (AttributeError, TypeError):
            return None


def is_regularized_results(results) -> bool:
    """Check if results is a RegularizedResults object (lacks standard errors)."""
    return not hasattr(results, 'bse') or results.__class__.__name__ == 'RegularizedResults'


def get_fit_stats(results) -> dict:
    """
    Safely extract fit statistics from a statsmodels results object.
    
    Handles OLS/WLS (which have rsquared, aic, etc.), RLM models,
    and RegularizedResults (which lack most statistics).
    
    Returns dict with keys: rsquared, rsquared_adj, fvalue, f_pvalue, aic, bic, llf
    """
    stats = {}
    
    # Check if this is a non-standard results object (RLM or Regularized)
    has_rsquared = hasattr(results, 'rsquared')
    is_regularized = is_regularized_results(results)
    
    if not has_rsquared or is_regularized:
        # Compute pseudo-R² from residuals
        resid = safe_get_resid(results)
        try:
            y = results.model.endog
            y_mean = np.mean(y)
            ss_tot = np.sum((y - y_mean) ** 2)
            if resid is not None:
                ss_res = np.sum(resid ** 2)
                pseudo_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            else:
                pseudo_r2 = np.nan
            
            n = len(y)
            p = len(results.params) - 1  # number of predictors (excluding intercept)
            pseudo_r2_adj = 1 - (1 - pseudo_r2) * (n - 1) / (n - p - 1) if n > p + 1 and not np.isnan(pseudo_r2) else np.nan
        except (AttributeError, TypeError):
            pseudo_r2 = np.nan
            pseudo_r2_adj = np.nan
        
        stats['rsquared'] = pseudo_r2
        stats['rsquared_adj'] = pseudo_r2_adj
        stats['fvalue'] = np.nan
        stats['f_pvalue'] = np.nan
        stats['aic'] = np.nan
        stats['bic'] = np.nan
        stats['llf'] = np.nan
        stats['is_rlm'] = True  # Treat as RLM/non-standard for display purposes
    else:
        stats['rsquared'] = results.rsquared
        stats['rsquared_adj'] = results.rsquared_adj
        stats['fvalue'] = results.fvalue
        stats['f_pvalue'] = results.f_pvalue
        stats['aic'] = results.aic
        stats['bic'] = results.bic
        stats['llf'] = results.llf
        stats['is_rlm'] = False
    
    return stats


def format_stat(value, fmt='.4f') -> str:
    """Format a statistic value, returning 'N/A' for NaN values."""
    if pd.isna(value):
        return 'N/A'
    return f'{value:{fmt}}'


def get_diagnostics_from_row(row: pd.Series, results) -> dict:
    """Extract pre-computed regression diagnostics from model row."""
    n_inf = row['cooks_n']
    resid = safe_get_resid(results)
    mean_resid = np.mean(resid) if resid is not None else np.nan
    return {
        'shapiro_stat': row['SW_stat'],
        'shapiro_pvalue': row['SW_p'],
        'bp_stat': row['BP_stat'],
        'bp_pvalue': row['BP_p'],
        'dw_stat': row['DW'],
        'n_influential': int(n_inf) if pd.notna(n_inf) else 'N/A',
        'max_vif': row['VIF'],
        'mean_resid': mean_resid,
    }


def build_coefficient_table(results) -> str:
    """Format coefficient table as markdown for LLM prompt."""
    rows = []
    
    # Check if we have standard errors available
    try:
        bse_vals = results.bse.values
        tvals = results.tvalues.values
        pvals = results.pvalues.values
        has_inference = True
    except AttributeError:
        n_params = len(results.params)
        bse_vals = [np.nan] * n_params
        tvals = [np.nan] * n_params
        pvals = [np.nan] * n_params
        has_inference = False
    
    for var, coef, se, t, p in zip(
        results.params.index,
        results.params.values,
        bse_vals,
        tvals,
        pvals
    ):
        if has_inference:
            sig = "Yes" if p < 0.05 else "No"
            rows.append(f"| {var} | {coef:.4f} | {se:.4f} | {t:.2f} | {p:.4f} | {sig} |")
        else:
            rows.append(f"| {var} | {coef:.4f} | N/A | N/A | N/A | N/A |")
    return "\n".join(rows)


def build_llm_prompt(row: pd.Series, results, diagnostics: dict) -> str:
    """Build the LLM analysis prompt with model data."""
    coef_table = build_coefficient_table(results)
    fit_stats = get_fit_stats(results)
    model_type = "Robust Linear Model (RLM)" if fit_stats['is_rlm'] else "OLS/WLS"
    
    # Get model identification fields
    estimator_name = row.get('estimator', 'unknown')
    base_estimator = row.get('base_estimator', None)
    params = row.get('params', None)
    
    # Build model info string
    model_info_parts = [f"**Estimator:** {estimator_name}"]
    if base_estimator and base_estimator != estimator_name:
        model_info_parts.append(f"**Base Estimator:** {base_estimator}")
    if params:
        model_info_parts.append(f"**Parameters:** {params}")
    model_info_str = "\n- ".join(model_info_parts)
    
    prompt = f"""You are a senior marketing analytics consultant specializing in media mix modeling and marketing ROI optimization for retail brands. Analyze the following linear regression model and provide actionable recommendations.

## MODEL CONTEXT
- **Brand:** {row['brand']}
- **Model:** {row['model']}
- {model_info_str}
- **Model Type:** {model_type}
- **Response Variable:** {row['response']} (nd = net demand revenue, visits = site visits)
- **Sample Size:** {row['n_obs']} observations ({row['n_outliers']} outliers removed via z-score threshold)

## MODEL PERFORMANCE
- **In-Sample Score:** {row['R2']:.4f}
- **Cross-Validated Score:** {row['R2_CV']:.4f}
- **Adjusted R²:** {format_stat(fit_stats['rsquared_adj'])}{'  (pseudo-R² for RLM)' if fit_stats['is_rlm'] else ''}
- **F-statistic:** {format_stat(fit_stats['fvalue'], '.2f')}{f" (p-value: {format_stat(fit_stats['f_pvalue'], '.4e')})" if not fit_stats['is_rlm'] else ' (N/A for RLM)'}
- **AIC:** {format_stat(fit_stats['aic'], '.2f')}
- **BIC:** {format_stat(fit_stats['bic'], '.2f')}

## COEFFICIENT SUMMARY
| Variable | Coefficient | Std Error | t-value | p-value | Significant |
|----------|-------------|-----------|---------|---------|-------------|
{coef_table}

## RESIDUAL DIAGNOSTICS
- **Residual Normality (Shapiro-Wilk p-value):** {diagnostics['shapiro_pvalue']:.4f}
- **Heteroscedasticity (Breusch-Pagan p-value):** {diagnostics['bp_pvalue']:.4f}
- **Autocorrelation (Durbin-Watson):** {diagnostics['dw_stat']:.4f}
- **Influential Points (Cook's D > 4/n):** {diagnostics['n_influential']} observations
- **Multicollinearity (Max VIF):** {diagnostics['max_vif']:.2f}
- **Mean Residual:** {diagnostics['mean_resid']:.4f}

## PROVIDE THE FOLLOWING ANALYSIS:

### 1. Model Quality Assessment
Rate this model's overall quality (Poor/Fair/Good/Excellent) and explain why. Consider predictive power, statistical validity, and practical utility for forecasting.

### 2. Key Drivers Analysis
Identify the 3-5 most impactful variables (by magnitude and significance). For promotional variables, interpret what this means for campaign planning.

### 3. Statistical Concerns
Flag any issues with:
- Multicollinearity (if many promos are insignificant)
- Heteroscedasticity or non-normality (residual issues)
- Overfitting (high R² with low sample size or many features)
- Negative or counterintuitive coefficients

### 4. Actionable Recommendations
Provide 3-5 specific recommendations to improve this model, such as:
- Feature engineering suggestions (interactions, transformations)
- Variables to consider adding or removing
- Data quality improvements needed
- Alternative modeling approaches if appropriate

### 5. Business Implications
What does this model suggest about the effectiveness of this channel for this brand? Any recommendations for media investment optimization?

Format your response with clear headers and bullet points. Be specific and quantitative where possible."""
    
    return prompt


def call_llm_analysis(prompt: str, provider: str = "Anthropic (Claude)") -> str:
    """Call the selected LLM API for model analysis."""
    
    if provider == "OpenAI (GPT-4o)":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY environment variable not set."
        
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"
    else:
        # Default to Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return "Error: ANTHROPIC_API_KEY environment variable not set."
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error calling Anthropic API: {str(e)}"


class PDFReport(FPDF):
    """Custom PDF class for model diagnostics report."""
    
    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, self.title, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()} | Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C')
    
    def add_section(self, title: str, content: str):
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, content)
        self.ln(3)


def generate_pdf_report(row: pd.Series, results, diagnostics: dict, llm_analysis: str) -> bytes:
    """Generate a PDF report with model diagnostics and LLM analysis."""
    estimator_name = row.get('estimator', 'unknown')
    base_estimator = row.get('base_estimator', None)
    params = row.get('params', None)
    
    pdf = PDFReport(f"Model Analysis: {row['brand']} | {row['model']} | {estimator_name}")
    pdf.add_page()
    
    # Model Information - include base_estimator and params if different
    model_lines = [
        f"Brand: {row['brand']}",
        f"Model: {row['model']}",
        f"Estimator: {estimator_name}",
    ]
    if base_estimator and base_estimator != estimator_name:
        model_lines.append(f"Base Estimator: {base_estimator}")
    if params:
        model_lines.append(f"Parameters: {params}")
    model_lines.extend([
        f"Response Variable: {row['response']}",
        f"In-Sample Score: {row['R2']:.4f}",
        f"CV Score: {row['R2_CV']:.4f}",
        f"Observations: {row['n_obs']} ({row['n_outliers']} outliers removed)",
    ])
    model_info = "\n".join(model_lines)
    pdf.add_section("Model Information", model_info)
    
    # Model Fit Statistics
    stats = get_fit_stats(results)
    fit_stats_text = f"""R-squared: {format_stat(stats['rsquared'])}{'  (pseudo-R² for RLM)' if stats['is_rlm'] else ''}
    Adjusted R-squared: {format_stat(stats['rsquared_adj'])}
    F-statistic: {format_stat(stats['fvalue'], '.2f')}{f" (p-value: {format_stat(stats['f_pvalue'], '.4e')})" if not stats['is_rlm'] else ' (N/A for RLM)'}
    AIC: {format_stat(stats['aic'], '.2f')}
    BIC: {format_stat(stats['bic'], '.2f')}"""
    pdf.add_section("Model Fit Statistics", fit_stats_text)
    
    # Residual Diagnostics
    resid_diag = f"""Shapiro-Wilk p-value (normality): {format_stat(diagnostics['shapiro_pvalue'])}
    Breusch-Pagan p-value (heteroscedasticity): {format_stat(diagnostics['bp_pvalue'])}
    Durbin-Watson (autocorrelation): {format_stat(diagnostics['dw_stat'])}
    Influential Points (Cook's D > 4/n): {diagnostics['n_influential']}
    Max VIF (multicollinearity): {format_stat(diagnostics['max_vif'], '.2f')}
    Mean Residual: {format_stat(diagnostics['mean_resid'])}"""
    pdf.add_section("Residual Diagnostics", resid_diag)
    
    # Coefficients summary (top 10 most significant or by absolute coefficient)
    try:
        pvals = results.pvalues.values
        coef_df = pd.DataFrame({
            'Variable': results.params.index,
            'Coefficient': results.params.values,
            'p-value': pvals,
        }).sort_values('p-value').head(10)
        
        coef_text = "Top 10 Most Significant Coefficients:\n"
        for _, c in coef_df.iterrows():
            coef_text += f"  {c['Variable']}: {c['Coefficient']:.4f} (p={c['p-value']:.4f})\n"
    except AttributeError:
        # RegularizedResults - sort by absolute coefficient value instead
        coef_df = pd.DataFrame({
            'Variable': results.params.index,
            'Coefficient': results.params.values,
        })
        coef_df['abs_coef'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values('abs_coef', ascending=False).head(10)
        
        coef_text = "Top 10 Coefficients (by magnitude, p-values not available for regularized models):\n"
        for _, c in coef_df.iterrows():
            coef_text += f"  {c['Variable']}: {c['Coefficient']:.4f}\n"
    pdf.add_section("Key Coefficients", coef_text)
    
    # LLM Analysis
    pdf.add_page()
    pdf.add_section("LLM Analysis & Recommendations", llm_analysis)
    
    return bytes(pdf.output())


OLD_SCHEMA_COLUMNS = {'channel', 'base_model'}
NEW_SCHEMA_REQUIRED = {'model', 'estimator', 'base_estimator'}


@st.cache_resource
def load_models(filepath: str) -> pd.DataFrame:
    """Load fitted models from a joblib file."""
    df = joblib_load(filepath)
    if OLD_SCHEMA_COLUMNS & set(df.columns):
        st.warning(
            f"`{Path(filepath).name}` uses the old schema "
            f"(`channel`, `model`, `base_model`). Only files built with the "
            f"202607+ schema (`model`, `estimator`, `base_estimator`) are supported. "
            f"Results may be incorrect."
        )
    return df


@st.cache_data
def load_csv_index() -> pd.DataFrame:
    """Load a lightweight discovery index from all new-schema CSV files in DATA_DIR.

    Returns a DataFrame with columns: brand, model, response, run.
    Files lacking the 'estimator' column (old schema) are silently skipped.
    """
    frames = []
    for csv_path in sorted(DATA_DIR.glob("models_*.csv")):
        try:
            df = pd.read_csv(csv_path)
            if 'estimator' not in df.columns:
                continue  # old schema — skip
            frames.append(
                df[['brand', 'model', 'response']]
                .drop_duplicates()
                .assign(run=csv_path.stem)
            )
        except Exception:
            continue
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=['brand', 'model', 'response', 'run'])
    )


def get_model_options(df_models: pd.DataFrame) -> list[str]:
    """Generate display labels for model selector including estimator type."""
    return df_models.apply(
        lambda row: f"{row['brand']} | {row['model']} | {row['response']} | {row['estimator']}", 
        axis=1
    ).tolist()


def get_model_variant_options(df_models: pd.DataFrame) -> list[str]:
    """Generate display labels for model variant selector."""
    def format_row(row):
        label = row['estimator']
        if pd.notna(row.get('base_estimator')) and row['base_estimator'] != row['estimator']:
            label += f" ({row['base_estimator']})"
        if pd.notna(row.get('params')):
            label += f" [{row['params']}]"
        return label
    return df_models.apply(format_row, axis=1).tolist()


def get_unique_brands(df_models: pd.DataFrame) -> list[str]:
    """Get sorted unique brands from models DataFrame."""
    return sorted(df_models['brand'].unique().tolist())


def get_unique_models(df_models: pd.DataFrame, brand: str = None) -> list[str]:
    """Get sorted unique model names, optionally filtered by brand."""
    if brand:
        return sorted(df_models[df_models['brand'] == brand]['model'].unique().tolist())
    return sorted(df_models['model'].unique().tolist())


def filter_models(df_models: pd.DataFrame, brand: str, model: str) -> pd.DataFrame:
    """Filter models DataFrame by brand and model name."""
    filtered = df_models
    if brand:
        filtered = filtered[filtered['brand'] == brand]
    if model:
        filtered = filtered[filtered['model'] == model]
    return filtered


def display_model_info(row: pd.Series) -> None:
    """Display model metadata."""
    st.subheader("Model Information")
    
    # Get model identifiers from data
    estimator_name = row.get('estimator', 'unknown')
    base_estimator = row.get('base_estimator', None)
    params = row.get('params', None)
    
    # Detect model type details from pipeline for additional info
    pipe = row['model_obj']
    model_step = pipe.named_steps['model']
    model_class = model_step.__class__.__name__
    
    # Build model type detail string
    model_detail = None
    if model_class == 'StatsmodelsWLS':
        estimated_k = getattr(model_step, 'estimated_k_', None)
        if estimated_k:
            model_detail = f'k={estimated_k:.2f}'
    elif model_class == 'StatsmodelsRegularizedOLS':
        is_ols_refit = getattr(model_step, 'is_ols_refit_', False)
        if is_ols_refit:
            n_selected = len(getattr(model_step, 'selected_features_', [])) - 1
            model_detail = f'{n_selected} features selected'
    elif model_class == 'StatsmodelsPCAOLS':
        n_components = len(getattr(model_step, 'component_names_', []))
        var_explained = getattr(model_step, 'cumulative_variance_', [0])[-1] * 100
        model_detail = f'{n_components} PCs, {var_explained:.0f}% var'
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Brand:** {row['brand']}")
        st.write(f"**Model:** {row['model']}")
        st.write(f"**Response:** {row['response']}")
    with col2:
        st.write(f"**Estimator:** {estimator_name}")
        if base_estimator and base_estimator != estimator_name:
            st.write(f"**Base Estimator:** {base_estimator}")
        if params:
            st.write(f"**Parameters:** {params}")
        elif model_detail:
            st.write(f"**Details:** {model_detail}")
        st.write(f"**R² / CV R²:** {row['R2']:.4f} / {row['R2_CV']:.4f}")
        st.write(f"**Observations:** {row['n_obs']} ({row['n_outliers']} outliers removed)")


def display_coefficients(results, model_step=None) -> None:
    """Display regression coefficients table."""
    st.subheader("Regression Coefficients")
    
    # Show model-specific info
    if model_step is not None:
        model_class = model_step.__class__.__name__
        
        # For Lasso/Regularized: show selected features if refitted
        if model_class == 'StatsmodelsRegularizedOLS':
            is_refit = getattr(model_step, 'is_ols_refit_', False)
            if is_refit:
                selected = getattr(model_step, 'selected_feature_names_', [])
                st.success(f"✓ OLS refit on {len(selected)} selected features")
            else:
                st.warning("⚠ No OLS refit - showing regularized coefficients (no inference available). "
                          "Use `refit=True` in pipeline_regularized for full OLS results.")
        
        # For PCA: show component info
        elif model_class == 'StatsmodelsPCAOLS':
            n_comp = len(getattr(model_step, 'component_names_', []))
            var_exp = getattr(model_step, 'cumulative_variance_', [0])[-1] * 100
            st.info(f"PCA regression: {n_comp} components explaining {var_exp:.1f}% variance")
            
            # Show feature importance in expander
            with st.expander("Feature Importance (via PCA loadings)"):
                importance = model_step.get_original_feature_importance()
                if importance is not None:
                    st.dataframe(importance.head(15).style.format({
                        'contribution': '{:.4f}',
                        'direction': '{:.4f}',
                        'normalized_contribution': '{:.2%}'
                    }))
    
    # Check if this is a RegularizedResults (no standard errors available)
    try:
        bse = results.bse.values
        tvalues = results.tvalues.values
        pvalues = results.pvalues.values
        conf_int = results.conf_int()
        conf_lower = conf_int[0].values
        conf_upper = conf_int[1].values
        has_inference = True
    except AttributeError:
        # RegularizedResults doesn't have standard errors
        n_params = len(results.params)
        bse = [np.nan] * n_params
        tvalues = [np.nan] * n_params
        pvalues = [np.nan] * n_params
        conf_lower = [np.nan] * n_params
        conf_upper = [np.nan] * n_params
        has_inference = False
    
    coef_df = pd.DataFrame({
        'Variable': results.params.index,
        'Coefficient': results.params.values,
        'Std Error': bse,
        't-value': tvalues,
        'P>|t|': pvalues,
        '[0.025': conf_lower,
        '0.975]': conf_upper
    })
    
    if not has_inference:
        st.info("Note: Standard errors and inference statistics are not available for regularized regression models.")
    
    # Format numeric columns to 4 decimal places
    styled_df = coef_df.style.format({
        'Coefficient': '{:.4f}',
        'Std Error': '{:.4f}',
        't-value': '{:.4f}',
        'P>|t|': '{:.4f}',
        '[0.025': '{:.4f}',
        '0.975]': '{:.4f}'
    })
    st.dataframe(styled_df, width=800)


def display_fit_statistics(results) -> None:
    """Display model fit statistics."""
    st.subheader("Model Fit Statistics")
    stats = get_fit_stats(results)
    
    # Build metrics list, marking RLM-specific items
    metrics = ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)', 'AIC', 'BIC', 'Log-Likelihood']
    values = [
        stats['rsquared'],
        stats['rsquared_adj'],
        stats['fvalue'],
        stats['f_pvalue'],
        stats['aic'],
        stats['bic'],
        stats['llf']
    ]
    
    # If RLM, add note to R-squared metrics
    if stats['is_rlm']:
        metrics[0] = 'R-squared (pseudo)'
        metrics[1] = 'Adj. R-squared (pseudo)'
    
    stats_df = pd.DataFrame({'Metric': metrics, 'Value': values})
    
    # Custom formatter that handles NaN
    def format_value(v):
        return 'N/A' if pd.isna(v) else f'{v:.4f}'
    
    stats_df['Value'] = stats_df['Value'].apply(format_value)
    st.dataframe(stats_df, width=400)


def display_diagnostic_tests(row: pd.Series) -> None:
    """Display diagnostic test statistics."""
    st.subheader("Diagnostic Tests")
    
    # Helper to safely format values that may be NaN
    def safe_format(val, fmt='.4f'):
        if pd.isna(val):
            return 'N/A'
        return f'{val:{fmt}}'
    
    # Color logic - default to neutral if NaN
    shapiro_pval = row['SW_p']
    bp_pval = row['BP_p']
    dw_val = row['DW']
    vif_val = row['VIF']
    
    shapiro_color = "red" if pd.notna(shapiro_pval) and shapiro_pval < 0.05 else ("green" if pd.notna(shapiro_pval) else "gray")
    bp_color = "red" if pd.notna(bp_pval) and bp_pval < 0.05 else ("green" if pd.notna(bp_pval) else "gray")
    dw_color = "green" if pd.notna(dw_val) and 1.5 <= dw_val <= 2.5 else ("red" if pd.notna(dw_val) else "gray")
    vif_color = "red" if pd.notna(vif_val) and vif_val > 10 else ("orange" if pd.notna(vif_val) and vif_val > 5 else ("green" if pd.notna(vif_val) else "gray"))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Normality**")
        st.markdown(f"Shapiro-Wilk Stat: {safe_format(row['SW_stat'])}")
        st.markdown(f"p-value: <span style='color:{shapiro_color}'>{safe_format(shapiro_pval)}</span>", unsafe_allow_html=True)
    with col2:
        st.markdown("**Hetsk**")
        st.markdown(f"Breusch-Pagan Stat: {safe_format(row['BP_stat'])}")
        st.markdown(f"p-value: <span style='color:{bp_color}'>{safe_format(bp_pval)}</span>", unsafe_allow_html=True)
    with col3:
        st.markdown("**Independence**")
        st.markdown(f"Durbin-Watson Stat: <span style='color:{dw_color}'>{safe_format(dw_val)}</span>", unsafe_allow_html=True)
        n_inf = row['cooks_n']
        st.markdown(f"Influential Points: {int(n_inf) if pd.notna(n_inf) else 'N/A'}")
    with col4:
        st.markdown("**Multicollin**")
        st.markdown(f"Max VIF: <span style='color:{vif_color}'>{safe_format(vif_val, '.2f')}</span>", unsafe_allow_html=True)


def plot_residuals_vs_fitted(results) -> None:
    """Plot residuals vs fitted values."""
    resid = safe_get_resid(results)
    fitted = safe_get_fittedvalues(results)
    
    if resid is None or fitted is None:
        st.info("Residuals vs Fitted plot not available for this model type.")
        return
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.scatterplot(x=fitted, y=resid, ax=ax)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted')
    st.pyplot(fig)
    plt.close(fig)


def plot_qq(results) -> None:
    """Plot Q-Q plot of residuals."""
    resid = safe_get_resid(results)
    
    if resid is None:
        st.info("Q-Q plot not available for this model type.")
        return
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title('Normal Q-Q')
    st.pyplot(fig)
    plt.close(fig)


def plot_residuals_histogram(results) -> None:
    """Plot histogram of residuals."""
    resid = safe_get_resid(results)
    
    if resid is None:
        st.info("Residuals histogram not available for this model type.")
        return
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.histplot(resid, kde=True, ax=ax)
    ax.set_xlabel('Residuals')
    ax.set_title('Residuals Distribution')
    st.pyplot(fig)
    plt.close(fig)


def plot_cooks_distance(results) -> None:
    """Plot Cook's distance with influential points annotated."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    try:
        influence = results.get_influence()
        cooks_d = influence.cooks_distance[0]
        threshold = COOKS_THRESHOLD_FACTOR / len(cooks_d)
        
        ax.stem(range(len(cooks_d)), cooks_d, markerfmt=',')
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Observation')
        ax.set_ylabel("Cook's Distance")
        ax.set_title(f"Cook's Distance (threshold = {COOKS_THRESHOLD_FACTOR}/n)")
        
        # Annotate influential points
        influential_indices = np.where(cooks_d > threshold)[0]
        for idx in influential_indices:
            ax.annotate(
                f'{idx}', 
                xy=(idx, cooks_d[idx]), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=9,
                alpha=0.8
            )
    except (AttributeError, ValueError, np.linalg.LinAlgError):
        ax.text(0.5, 0.5, "Cook's Distance\nnot available\nfor this model type (WLS)", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Cook's Distance (N/A)")
    
    st.pyplot(fig)
    plt.close(fig)


def display_residual_analysis(results) -> None:
    """Display all residual analysis plots."""
    st.subheader("Residual Analysis")
    plot_residuals_vs_fitted(results)
    plot_qq(results)
    plot_residuals_histogram(results)
    plot_cooks_distance(results)


def main():
    """Main application entry point."""
    st.set_page_config(layout="wide", page_title="Model Diagnostics Viewer")
    
    # Custom CSS to set max width to 1200px
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1700px;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Model Diagnostics Viewer")
    
    # Load CSV discovery index — lightweight, no joblib loading yet
    csv_index = load_csv_index()

    if csv_index.empty:
        st.error(
            f"No compatible model CSV files found in {DATA_DIR}. "
            "Ensure models have been trained with the 202607+ schema."
        )
        return

    # Cascading selectors: Brand → Model → Response → Runs
    st.subheader("Select Model")
    col_brand, col_model = st.columns(2)

    with col_brand:
        brand_options = sorted(csv_index['brand'].unique().tolist())
        selected_brand = st.selectbox("Brand:", brand_options)

    with col_model:
        brand_models = sorted(
            csv_index[csv_index['brand'] == selected_brand]['model'].unique().tolist()
        )
        selected_model = st.selectbox("Model:", brand_models)

    # Response: auto-hide when only one value exists for the selected brand+model
    response_options = sorted(
        csv_index[
            (csv_index['brand'] == selected_brand) &
            (csv_index['model'] == selected_model)
        ]['response'].unique().tolist()
    )
    if len(response_options) > 1:
        selected_response = st.selectbox("Response Variable:", response_options)
    else:
        selected_response = response_options[0] if response_options else None

    if not selected_response:
        st.warning("No response variable found for the selected brand and model.")
        return

    # Run selector: only runs that contain the chosen brand+model+response
    available_runs = sorted(
        csv_index[
            (csv_index['brand'] == selected_brand) &
            (csv_index['model'] == selected_model) &
            (csv_index['response'] == selected_response)
        ]['run'].unique().tolist(),
        reverse=True,  # most recent first (stems sort lexicographically by date)
    )

    if not available_runs:
        st.warning("No runs found for the selected combination.")
        return

    run_labels = [r.replace('models_', '') for r in available_runs]
    stem_by_label = dict(zip(run_labels, available_runs))

    st.subheader("Select Runs to Compare")
    selected_run_labels = st.multiselect(
        "Select up to 3 runs:",
        run_labels,
        default=[run_labels[0]] if run_labels else [],
        max_selections=3,
    )

    if not selected_run_labels:
        st.info("Please select at least one run to analyze.")
        return

    # Load selected joblib files; collect all estimator rows matching the selection
    run_data = []
    for run_label in selected_run_labels:
        stem = stem_by_label[run_label]
        filepath = DATA_DIR / f"{stem}.joblib"
        if not filepath.exists():
            st.warning(f"Joblib file not found: {filepath.name}")
            continue
        df_run = load_models(str(filepath))
        matches = df_run[
            (df_run['brand'] == selected_brand) &
            (df_run['model'] == selected_model) &
            (df_run['response'] == selected_response)
        ]
        for _, row in matches.iterrows():
            estimator_name = row.get('estimator', 'unknown')
            base_estimator = row.get('base_estimator', None)
            params = row.get('params', None)
            pipe = row['model_obj']
            model_step = pipe.named_steps['model']
            results = model_step.results_
            run_data.append({
                'filename': f"{stem}.joblib",
                'run_label': run_label,
                'row': row,
                'results': results,
                'model_step': model_step,
                'estimator_name': estimator_name,
                'base_estimator': base_estimator,
                'params': params,
            })

    if not run_data:
        st.warning("No matching models found in the selected runs.")
        return
    
    # Display comparison across columns
    st.divider()
    st.subheader("Model Comparison")

    num_models = len(run_data)
    columns = st.columns(num_models)

    for idx, data in enumerate(run_data):
        with columns[idx]:
            run_label = data['run_label']
            estimator_name = data['estimator_name']
            base_estimator = data['base_estimator']
            params = data['params']
            row = data['row']
            results = data['results']
            model_step = data['model_step']

            st.markdown(f"**{run_label}**")
            header_parts = [estimator_name]
            if base_estimator and base_estimator != estimator_name:
                header_parts.append(f"({base_estimator})")
            if params:
                header_parts.append(f"[{params}]")
            st.markdown(f"*{' '.join(header_parts)}*")

            # Display model info
            display_model_info(row)
            display_coefficients(results, model_step=model_step)
            display_fit_statistics(results)
            display_diagnostic_tests(row)
    
    # Residual plots in columns
    st.divider()
    st.subheader("Residual Analysis")
    
    # Residuals vs Fitted
    st.markdown("**Residuals vs Fitted Values**")
    res_cols = st.columns(num_models)
    for idx, data in enumerate(run_data):
        with res_cols[idx]:
            plot_residuals_vs_fitted(data['results'])
    
    # Q-Q plots
    st.markdown("**Normal Q-Q Plot**")
    qq_cols = st.columns(num_models)
    for idx, data in enumerate(run_data):
        with qq_cols[idx]:
            plot_qq(data['results'])
    
    # Residuals histogram
    st.markdown("**Residuals Distribution**")
    hist_cols = st.columns(num_models)
    for idx, data in enumerate(run_data):
        with hist_cols[idx]:
            plot_residuals_histogram(data['results'])
    
    # Cook's distance
    st.markdown("**Cook's Distance**")
    cooks_cols = st.columns(num_models)
    for idx, data in enumerate(run_data):
        with cooks_cols[idx]:
            plot_cooks_distance(data['results'])
    
    # LLM Analysis Section
    st.divider()
    st.subheader("LLM Analysis")
    
    if num_models > 1:
        def format_llm_option(d):
            parts = [d['run_label'], d['estimator_name']]
            if d.get('base_estimator') and d['base_estimator'] != d['estimator_name']:
                parts.append(f"({d['base_estimator']})")
            if d.get('params'):
                parts.append(f"[{d['params']}]")
            return " | ".join(parts[:2]) + (" " + " ".join(parts[2:]) if len(parts) > 2 else "")
        
        llm_model_options = [format_llm_option(d) for d in run_data]
        llm_selected_idx = st.selectbox(
            "Select model for LLM analysis:",
            range(len(llm_model_options)),
            format_func=lambda i: llm_model_options[i]
        )
        llm_data = run_data[llm_selected_idx]
    else:
        llm_data = run_data[0]
    
    # LLM Provider selector
    llm_provider = st.selectbox("Select LLM Provider:", LLM_PROVIDERS)
    
    # Initialize session state for LLM analysis
    if 'llm_analysis' not in st.session_state:
        st.session_state.llm_analysis = None
    if 'current_model_key' not in st.session_state:
        st.session_state.current_model_key = None
    
    # Generate a unique key for the current model selection
    row = llm_data['row']
    results = llm_data['results']
    model_key = f"{llm_data['filename']}_{row['brand']}_{row['model']}_{row['response']}_{llm_data['estimator_name']}"
    
    # Reset analysis if model selection changed
    if st.session_state.current_model_key != model_key:
        st.session_state.llm_analysis = None
        st.session_state.current_model_key = model_key
    
    # Get pre-computed diagnostics from joblib (needed for both LLM and PDF)
    diagnostics = get_diagnostics_from_row(row, results)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Get LLM Analysis", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing model with {llm_provider}..."):
                prompt = build_llm_prompt(row, results, diagnostics)
                st.session_state.llm_analysis = call_llm_analysis(prompt, llm_provider)
    
    # Display LLM analysis if available
    if st.session_state.llm_analysis:
        st.markdown(st.session_state.llm_analysis)
        
        # PDF Export button
        with col2:
            pdf_bytes = generate_pdf_report(row, results, diagnostics, st.session_state.llm_analysis)
            filename = f"model_analysis_{row['brand']}_{row['model']}_{llm_data['estimator_name']}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            filename = filename.replace(" ", "_").replace("/", "-")
            st.download_button(
                label="Save as PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.info("Click 'Get LLM Analysis' to generate AI-powered insights and recommendations for this model.")


if __name__ == "__main__":
    main()