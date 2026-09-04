"""
Backtest Results Viewer

Streamlit app to explore backtest results by brand/model.
Displays model scores and time series of actuals vs predictions.

Usage:
    streamlit run app/backtest_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path for unpickling pipeline objects
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page config
st.set_page_config(
    page_title="Backtest Results",
    page_icon="📊",
    layout="wide"
)

# Bias is estimated on only the most recent weeks so older, less stable
# forecasts do not distort the correction applied to the upcoming forecast.
BIAS_WINDOW_WEEKS = 8
MIN_BIAS_WEEKS = 4

# --- Session State Initialization ---
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = {}  # {(brand, model_key, estimator): {...metrics...}}
if 'current_backtest' not in st.session_state:
    st.session_state.current_backtest = None
if 'edited_bias' not in st.session_state:
    st.session_state.edited_bias = {}  # {(brand, model_key, estimator): edited_bias_value}
if 'edited_params' not in st.session_state:
    st.session_state.edited_params = {}  # {(brand, model_key, estimator): params_string}
if 'bias_windows' not in st.session_state:
    st.session_state.bias_windows = {}  # {(brand, model_key): bias window in weeks}
if 'selected_brand' not in st.session_state:
    st.session_state.selected_brand = None
if 'selected_model_key' not in st.session_state:
    st.session_state.selected_model_key = None

st.title("📊 Backtest Results Viewer")

# --- Data Loading ---

@st.cache_data
def load_backtest_data(backtest_folder: Path):
    """Load backtest scores and predictions from a folder."""
    scores_path = backtest_folder / 'backtest_scores.joblib'
    predictions_path = backtest_folder / 'backtest_predictions.csv'
    
    df_scores = joblib.load(scores_path)
    df_predictions = pd.read_csv(predictions_path, parse_dates=['fiscal_week'])
    
    return df_scores, df_predictions


def get_available_backtests(output_dir: Path) -> list[Path]:
    """Get list of available backtest folders sorted by most recent."""
    folders = [f for f in output_dir.iterdir() if f.is_dir() and f.name.startswith('backtest_')]
    return sorted(folders, reverse=True)


# --- Sidebar: Data Selection ---

st.sidebar.header("Data Selection")

output_dir = PROJECT_ROOT / 'data' / 'backtest'
backtest_folders = get_available_backtests(output_dir)

if not backtest_folders:
    st.error("No backtest results found. Run the backtest notebook first.")
    st.stop()

selected_folder = st.sidebar.selectbox(
    "Backtest Run",
    backtest_folders,
    format_func=lambda x: x.name
)

# Track current backtest (selections are preserved across backtest runs)
st.session_state.current_backtest = selected_folder.name

# Extract timestamp from folder name (e.g., backtest_20260305_141347 -> 20260305_141347)
backtest_timestamp = selected_folder.name.replace('backtest_', '')

# Load data
df_scores, df_predictions = load_backtest_data(selected_folder)

# --- Sidebar: Page Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Page", ["Model Review", "Export"], label_visibility="collapsed")

# Show selection count in sidebar
if st.session_state.selected_models:
    st.sidebar.markdown(f"**Selected Models:** {len(st.session_state.selected_models)}")


# --- Helper Functions ---

def filter_to_recent_weeks(preds: pd.DataFrame, n_weeks: int) -> tuple[pd.DataFrame, list]:
    """Restrict predictions to the last n_weeks fiscal weeks actually present."""
    weeks = sorted(preds['fiscal_week'].unique())
    window = weeks[-n_weeks:] if n_weeks > 0 else weeks
    return preds[preds['fiscal_week'].isin(window)].copy(), window


def calc_prediction_metrics(preds):
    """Calculate Bias % and CI Coverage for a group of predictions."""
    # Bias % = ratio of sums (positive = over-predicting); robust to near-zero weeks
    actual_total = preds['revenue_actual'].sum()
    if len(preds) >= MIN_BIAS_WEEKS and actual_total > 0:
        bias_pct = (preds['revenue_pred'].sum() / actual_total - 1) * 100
    else:
        bias_pct = np.nan
    
    # CI Coverage: % of actuals within the CI
    if preds['revenue_pred_ci_lo'].notna().any():
        within_ci = (
            (preds['revenue_actual'] >= preds['revenue_pred_ci_lo']) & 
            (preds['revenue_actual'] <= preds['revenue_pred_ci_hi'])
        )
        ci_coverage = within_ci.mean() * 100
    else:
        ci_coverage = np.nan
    
    return pd.Series({'Bias %': bias_pct, 'CI_Cov': ci_coverage})


def format_revenue(val):
    """Format revenue with K/M suffixes."""
    if pd.isna(val):
        return ''
    if abs(val) >= 1_000_000:
        return f'{val/1_000_000:.1f}M'
    elif abs(val) >= 1_000:
        return f'{val/1_000:.1f}K'
    else:
        return f'{val:.0f}'


def format_error_pct(val):
    """Format error percentage with + sign for positive values."""
    if pd.isna(val):
        return ''
    return f'{val:+.1f}%'


def apply_selections_from_df(df_load: pd.DataFrame) -> int:
    """Restore session state selected_models/edited_bias/edited_params/bias_windows from a CSV DataFrame."""
    st.session_state.selected_models = {}
    st.session_state.edited_bias = {}
    st.session_state.edited_params = {}
    st.session_state.bias_windows = {}
    for row in df_load.to_dict('records'):
        model_label = row['estimator']
        key = (row['brand'], row['model'], model_label)
        bias_val = row.get('bias_pct', np.nan)
        params_val = row.get('params', '')
        if pd.isna(params_val):
            params_val = ''
        weeks_val = row.get('bias_weeks', BIAS_WINDOW_WEEKS)
        weeks_val = int(weeks_val) if pd.notna(weeks_val) else BIAS_WINDOW_WEEKS
        st.session_state.selected_models[key] = {
            'brand': row['brand'],
            'model_key': row['model'],
            'base_estimator': row['base_estimator'],
            'estimator': model_label,
            'bias_pct': bias_val,
            'bias_weeks': weeks_val,
            'params': params_val,
        }
        st.session_state.edited_bias[key] = bias_val
        st.session_state.edited_params[key] = params_val
        st.session_state.bias_windows[(row['brand'], row['model'])] = weeks_val
    return len(df_load)

# =============================================================================
# MODEL REVIEW PAGE
# =============================================================================
if page == "Model Review":
    st.sidebar.header("Filters")

    brands = sorted(df_scores['brand'].unique())
    
    # Preserve brand selection if available in current dataset
    brand_index = 0
    if st.session_state.selected_brand in brands:
        brand_index = brands.index(st.session_state.selected_brand)
    
    selected_brand = st.sidebar.selectbox("Brand", brands, index=brand_index)
    st.session_state.selected_brand = selected_brand

    model_keys = sorted(df_scores[df_scores['brand'] == selected_brand]['model'].unique())
    
    # Preserve model_key selection if available for current brand
    model_key_index = 0
    if st.session_state.selected_model_key in model_keys:
        model_key_index = model_keys.index(st.session_state.selected_model_key)
    
    selected_model_key = st.sidebar.selectbox("Model", model_keys, index=model_key_index)
    st.session_state.selected_model_key = selected_model_key

    # Bias window is remembered per brand/model; widget key is scoped so switching
    # models shows that model's setting rather than the previous widget value.
    window_key = (selected_brand, selected_model_key)
    bias_window = st.sidebar.number_input(
        "Bias window (weeks)",
        min_value=2,
        max_value=52,
        value=st.session_state.bias_windows.get(window_key, BIAS_WINDOW_WEEKS),
        step=1,
        help="Number of most recent weeks used to estimate Bias % and CI Cov %",
        key=f"bias_window_{selected_brand}_{selected_model_key}"
    )
    st.session_state.bias_windows[window_key] = int(bias_window)

    # Filter data
    mask_scores = (df_scores['brand'] == selected_brand) & (df_scores['model'] == selected_model_key)
    mask_preds = (df_predictions['brand'] == selected_brand) & (df_predictions['model'] == selected_model_key)

    df_scores_filtered = df_scores[mask_scores].copy()
    df_predictions_filtered = df_predictions[mask_preds].copy()

    # --- Main Content ---
    st.subheader(f"{selected_brand} / {selected_model_key}")

    # Determine display column: 'estimator' contains estimator display name (e.g., 'RLM_Huber')
    # 'base_estimator' contains shortcode for PIPELINE_MAP (e.g., 'RLM')
    # 'model' is the pipe-delimited data key (channel | funnel | tactic)
    display_col = 'estimator'
    has_base_estimator = 'base_estimator' in df_scores_filtered.columns
    
    # Get available models and fiscal months
    models = df_scores_filtered[display_col].unique()
    fiscal_months = sorted(df_scores_filtered['fiscal_month'].unique())
    
    # Check if params column exists (new backtest format)
    has_params = 'params' in df_scores_filtered.columns

    st.write(f"**Fiscal Months:** {', '.join(fiscal_months)}")
    st.write(f"**Models:** {', '.join(models)}")

    # --- Model Summary Grid ---
    st.markdown("### Model Comparison")

    # Get scores from most recent fiscal month only
    most_recent_month = max(fiscal_months)
    df_recent = df_scores_filtered[df_scores_filtered['fiscal_month'] == most_recent_month].copy()
    
    summary_cols = ['R2', 'R2_CV', 'RMSE', 'RMSE_CV', 'MAPE', 'MAPE_CV', 'BP_p', 'VIF']
    summary_cols = [c for c in summary_cols if c in df_recent.columns]

    # Use most recent month values directly (one row per model)
    df_summary = df_recent[[display_col] + summary_cols].copy()
    df_summary = df_summary.rename(columns={display_col: 'Model'})
    
    # Add base estimator column for export (PIPELINE_MAP lookup)
    if has_base_estimator:
        # Map display label to base estimator shortcode
        label_to_base = df_recent.groupby('estimator')['base_estimator'].first().to_dict()
        df_summary['Base Estimator'] = df_summary['Model'].map(label_to_base)
    else:
        df_summary['Base Estimator'] = df_summary['Model']
    
    # Add params column if available (from most recent month)
    if has_params:
        params_by_model = df_recent[[display_col, 'params']].copy()
        params_by_model = params_by_model.rename(columns={display_col: 'Model', 'params': 'Params'})
        df_summary = df_summary.merge(params_by_model, on='Model', how='left')

    # Calculate prediction metrics on the recent-weeks window only
    df_preds_bias_window, bias_weeks = filter_to_recent_weeks(df_predictions_filtered, bias_window)
    pred_metrics = df_preds_bias_window.groupby(display_col).apply(calc_prediction_metrics).reset_index()
    pred_metrics = pred_metrics.rename(columns={display_col: 'Model', 'CI_Cov': 'CI Cov %'})

    df_summary = df_summary.merge(pred_metrics, on='Model', how='left')

    # Order by model
    model_order = list(models)
    df_summary['_sort'] = df_summary['Model'].apply(lambda x: model_order.index(x) if x in model_order else 999)
    df_summary = df_summary.sort_values('_sort').drop(columns=['_sort'])

    # Reorder columns: Model, Base Estimator, Params first, then the rest
    preferred_order = ['Model', 'Base Estimator', 'Params']
    other_cols = [c for c in df_summary.columns if c not in preferred_order]
    final_order = [c for c in preferred_order if c in df_summary.columns] + other_cols
    df_summary = df_summary[final_order]

    # Format for display
    df_summary_display = df_summary.copy()
    
    # Add "Use" column based on current session state
    df_summary_display['Use'] = df_summary_display['Model'].apply(
        lambda m: (selected_brand, selected_model_key, m) in st.session_state.selected_models
    )
    
    # Multiply R2 columns by 100 for percentage display (stored as 0-1)
    r2_cols = ['R2', 'R2_CV']
    for col in r2_cols:
        if col in df_summary_display.columns:
            df_summary_display[col] = df_summary_display[col] * 100
    
    # Reorder to put Use first
    use_first_order = ['Use'] + [c for c in df_summary_display.columns if c != 'Use']
    df_summary_display = df_summary_display[use_first_order]
    
    # Column config for data_editor
    column_config = {
        'Use': st.column_config.CheckboxColumn('Use', default=False),
        'Model': st.column_config.TextColumn('Model', disabled=True),
        'Base Estimator': st.column_config.TextColumn('Base Estimator', disabled=True),
        'Params': st.column_config.TextColumn('Params', disabled=True),
        'R2': st.column_config.NumberColumn('R2', format='%.2f%%', disabled=True),
        'R2_CV': st.column_config.NumberColumn('R2_CV', format='%.2f%%', disabled=True),
        'RMSE': st.column_config.NumberColumn('RMSE', format='%.2f', disabled=True),
        'RMSE_CV': st.column_config.NumberColumn('RMSE_CV', format='%.2f', disabled=True),
        'MAPE': st.column_config.NumberColumn('MAPE', format='%.2f%%', disabled=True),
        'MAPE_CV': st.column_config.NumberColumn('MAPE_CV', format='%.2f%%', disabled=True),
        'BP_p': st.column_config.NumberColumn('BP_p', format='%.4f', disabled=True),
        'VIF': st.column_config.NumberColumn('VIF', format='%.2f', disabled=True),
        'Bias %': st.column_config.NumberColumn(f'Bias % ({bias_window}W)', format='%.2f%%', disabled=True),
        'CI Cov %': st.column_config.NumberColumn(f'CI Cov % ({bias_window}W)', format='%.2f%%', disabled=True),
    }
    
    edited_summary = st.data_editor(
        df_summary_display,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key=f"model_comparison_{backtest_timestamp}_{selected_brand}_{selected_model_key}"
    )
    
    if bias_weeks:
        week_range = f"{pd.Timestamp(bias_weeks[0]):%Y-%m-%d} to {pd.Timestamp(bias_weeks[-1]):%Y-%m-%d}"
    else:
        week_range = 'no data'
    st.caption(f"Bias % and CI Cov % measured over the last {len(bias_weeks)} weeks ({week_range}). "
               f"Positive bias = over-forecast. Blank when fewer than {MIN_BIAS_WEEKS} weeks of history.")
    
    # Process checkbox changes
    for idx, row in edited_summary.iterrows():
        estimator_label = row['Model']
        model_key = (selected_brand, selected_model_key, estimator_label)
        use_model = row['Use']
        
        # Get metrics for this model
        model_summary_row = df_summary[df_summary['Model'] == estimator_label].iloc[0]
        
        if use_model and model_key not in st.session_state.selected_models:
            # Get original params from backtest if available
            original_params = ''
            if has_params:
                backtest_params = df_summary[df_summary['Model'] == estimator_label]['Params'].iloc[0]
                original_params = backtest_params if pd.notna(backtest_params) else ''
            
            # Add to selections with metrics
            base_estimator = df_summary[df_summary['Model'] == estimator_label]['Base Estimator'].iloc[0]
            st.session_state.selected_models[model_key] = {
                'brand': selected_brand,
                'model_key': selected_model_key,
                'base_estimator': base_estimator,
                'estimator': estimator_label,
                'bias_pct': model_summary_row.get('Bias %', np.nan),
                'bias_weeks': int(bias_window),
                'params': original_params,
            }
            st.session_state.edited_bias[model_key] = model_summary_row.get('Bias %', np.nan)
            st.session_state.edited_params[model_key] = original_params
            st.rerun()
        elif not use_model and model_key in st.session_state.selected_models:
            del st.session_state.selected_models[model_key]
            if model_key in st.session_state.edited_bias:
                del st.session_state.edited_bias[model_key]
            if model_key in st.session_state.edited_params:
                del st.session_state.edited_params[model_key]
            st.rerun()

    # --- Display each model's details ---
    for estimator in models:
        st.markdown("---")
        
        # Create key for this brand/model_key/estimator combo
        model_key = (selected_brand, selected_model_key, estimator)
        
        # Header with estimator name and optional params
        model_params = df_summary[df_summary['Model'] == estimator]['Params'].iloc[0] if has_params else None
        params_suffix = f" `{model_params}`" if model_params and str(model_params).strip() else ""
        st.markdown(f"### {estimator}{params_suffix}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Get scores for this estimator
            model_scores = df_scores_filtered[df_scores_filtered[display_col] == estimator]
            
            # Month-by-month bias, so the windowed summary figure is explainable
            monthly_bias = (
                df_predictions_filtered[df_predictions_filtered[display_col] == estimator]
                .groupby('fiscal_month')
                .apply(calc_prediction_metrics)
                .reset_index()[['fiscal_month', 'Bias %']]
            )
            model_scores = model_scores.merge(monthly_bias, on='fiscal_month', how='left')
            
            # Display score table
            score_cols = ['fiscal_month', 'Bias %', 'RMSE', 'RMSE_CV', 'MAPE', 'MAPE_CV', 'R2', 'R2_CV', 'BP_stat', 'BP_p', 'VIF']
            score_cols = [c for c in score_cols if c in model_scores.columns]
            
            st.dataframe(
                model_scores[score_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                column_config={'Bias %': st.column_config.NumberColumn('Bias %', format='%.1f%%')}
            )
        
        with col2:
            # Prepare data for all charts
            model_preds = df_predictions_filtered[df_predictions_filtered[display_col] == estimator].copy()
            model_preds = model_preds.sort_values('fiscal_week')
            
            # Calculate ROAS columns
            model_preds['roas_actual'] = model_preds['revenue_actual'] / model_preds['spend'].replace(0, np.nan)
            model_preds['roas_pred'] = model_preds['revenue_pred'] / model_preds['spend'].replace(0, np.nan)
            
            if not model_preds['revenue_pred_ci_lo'].isna().all():
                model_preds['roas_ci_lo'] = model_preds['revenue_pred_ci_lo'] / model_preds['spend'].replace(0, np.nan)
                model_preds['roas_ci_hi'] = model_preds['revenue_pred_ci_hi'] / model_preds['spend'].replace(0, np.nan)
            
            # Calculate error percentages: (pred - actual) / actual * 100
            model_preds['revenue_error_pct'] = ((model_preds['revenue_pred'] - model_preds['revenue_actual']) / model_preds['revenue_actual'].replace(0, np.nan) * 100).round(1)
            model_preds['roas_error_pct'] = ((model_preds['roas_pred'] - model_preds['roas_actual']) / model_preds['roas_actual'].replace(0, np.nan) * 100).round(1)
            
            # Format revenue values with K/M suffixes for hover
            model_preds['revenue_actual_fmt'] = model_preds['revenue_actual'].apply(format_revenue)
            model_preds['revenue_pred_fmt'] = model_preds['revenue_pred'].apply(format_revenue)
            
            # Format error percentages with + sign for positive values
            model_preds['revenue_error_fmt'] = model_preds['revenue_error_pct'].apply(format_error_pct)
            model_preds['roas_error_fmt'] = model_preds['roas_error_pct'].apply(format_error_pct)
            
            # Create tabs for Time Series vs Spend Response views
            tab_time, tab_spend = st.tabs(["📈 Time Series", "💰 Spend Response"])
            
            # --- TIME SERIES TAB ---
            with tab_time:
                # Net Demand vs Time
                fig = go.Figure()
                
                # Add Spend as light grey columns in background (secondary y-axis)
                fig.add_trace(go.Bar(
                    x=model_preds['fiscal_week'],
                    y=model_preds['spend'],
                    name='Spend',
                    marker_color='rgba(230, 230, 230, 0.2)',
                    yaxis='y2',
                    visible='legendonly',
                    hovertemplate='Spend: %{y:,.0f}<extra></extra>'
                ))
                
                if not model_preds['revenue_pred_ci_lo'].isna().all():
                    fig.add_trace(go.Scatter(
                        x=pd.concat([model_preds['fiscal_week'], model_preds['fiscal_week'][::-1]]),
                        y=pd.concat([model_preds['revenue_pred_ci_hi'], model_preds['revenue_pred_ci_lo'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(147, 112, 219, 0.35)',
                        line=dict(color='rgba(147, 112, 219, 0.5)'),
                        hoverinfo='skip',
                        showlegend=True,
                        name='90% CI'
                    ))
                
                fig.add_trace(go.Scatter(
                    x=model_preds['fiscal_week'],
                    y=model_preds['revenue_actual'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6),
                    customdata=model_preds['revenue_actual_fmt'],
                    hovertemplate='Actual: %{customdata}<extra></extra>'
                ))
                
                fig.add_trace(go.Scatter(
                    x=model_preds['fiscal_week'],
                    y=model_preds['revenue_pred'],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    marker=dict(size=6),
                    customdata=np.column_stack([model_preds['revenue_pred_fmt'], model_preds['revenue_error_fmt']]),
                    hovertemplate='Predicted: %{customdata[0]}<br><span style="color:#FF7043;font-weight:bold">Error: %{customdata[1]}</span><extra></extra>'
                ))
                
                fig.update_layout(
                    title="Net Demand vs Time",
                    xaxis_title="Week",
                    yaxis_title="Net Demand",
                    yaxis2=dict(
                        title="Spend",
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"fig_nd_time_{backtest_timestamp}_{model_key}")
                
                # ROAS vs Time
                fig_roas = go.Figure()
                
                # Add Spend as light grey columns in background (secondary y-axis)
                fig_roas.add_trace(go.Bar(
                    x=model_preds['fiscal_week'],
                    y=model_preds['spend'],
                    name='Spend',
                    marker_color='rgba(230, 230, 230, 0.2)',
                    yaxis='y2',
                    visible='legendonly',
                    hovertemplate='Spend: %{y:,.0f}<extra></extra>'
                ))
                
                if not model_preds['revenue_pred_ci_lo'].isna().all():
                    fig_roas.add_trace(go.Scatter(
                        x=pd.concat([model_preds['fiscal_week'], model_preds['fiscal_week'][::-1]]),
                        y=pd.concat([model_preds['roas_ci_hi'], model_preds['roas_ci_lo'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(147, 112, 219, 0.35)',
                        line=dict(color='rgba(147, 112, 219, 0.5)'),
                        hoverinfo='skip',
                        showlegend=True,
                        name='90% CI'
                    ))
                
                fig_roas.add_trace(go.Scatter(
                    x=model_preds['fiscal_week'],
                    y=model_preds['roas_actual'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6),
                    hovertemplate='Actual: %{y:.2f}<extra></extra>'
                ))
                
                fig_roas.add_trace(go.Scatter(
                    x=model_preds['fiscal_week'],
                    y=model_preds['roas_pred'],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    marker=dict(size=6),
                    customdata=model_preds['roas_error_fmt'],
                    hovertemplate='Predicted: %{y:.2f}<br><span style="color:#FF7043;font-weight:bold">Error: %{customdata}</span><extra></extra>'
                ))
                
                fig_roas.update_layout(
                    title="ROAS vs Time",
                    xaxis_title="Week",
                    yaxis_title="ROAS",
                    yaxis2=dict(
                        title="Spend",
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                    height=300
                )
                
                st.plotly_chart(fig_roas, use_container_width=True, key=f"fig_roas_time_{backtest_timestamp}_{model_key}")
            
            # --- SPEND RESPONSE TAB ---
            with tab_spend:
                # Sort by spend for response curve visualization
                model_preds_spend = model_preds.sort_values('spend')
                
                # Net Demand vs Spend
                fig_nd_spend = go.Figure()
                
                # CI band for Net Demand vs Spend
                if not model_preds_spend['revenue_pred_ci_lo'].isna().all():
                    fig_nd_spend.add_trace(go.Scatter(
                        x=pd.concat([model_preds_spend['spend'], model_preds_spend['spend'][::-1]]),
                        y=pd.concat([model_preds_spend['revenue_pred_ci_hi'], model_preds_spend['revenue_pred_ci_lo'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(147, 112, 219, 0.35)',
                        line=dict(color='rgba(147, 112, 219, 0.5)'),
                        hoverinfo='skip',
                        showlegend=True,
                        name='90% CI'
                    ))
                
                fig_nd_spend.add_trace(go.Scatter(
                    x=model_preds_spend['spend'],
                    y=model_preds_spend['revenue_actual'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6),
                    customdata=model_preds_spend['revenue_actual_fmt'],
                    hovertemplate='Actual: %{customdata}<extra></extra>'
                ))
                
                fig_nd_spend.add_trace(go.Scatter(
                    x=model_preds_spend['spend'],
                    y=model_preds_spend['revenue_pred'],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    marker=dict(size=6),
                    customdata=np.column_stack([model_preds_spend['revenue_pred_fmt'], model_preds_spend['revenue_error_fmt']]),
                    hovertemplate='Predicted: %{customdata[0]}<br><span style="color:#FF7043;font-weight:bold">Error: %{customdata[1]}</span><extra></extra>'
                ))
                
                fig_nd_spend.update_layout(
                    title="Net Demand vs Spend",
                    xaxis_title="Spend",
                    yaxis_title="Net Demand",
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                    height=300
                )
                
                st.plotly_chart(fig_nd_spend, use_container_width=True, key=f"fig_nd_spend_{backtest_timestamp}_{model_key}")
                
                # ROAS vs Spend
                fig_roas_spend = go.Figure()
                
                # CI band for ROAS vs Spend
                if not model_preds_spend['revenue_pred_ci_lo'].isna().all():
                    fig_roas_spend.add_trace(go.Scatter(
                        x=pd.concat([model_preds_spend['spend'], model_preds_spend['spend'][::-1]]),
                        y=pd.concat([model_preds_spend['roas_ci_hi'], model_preds_spend['roas_ci_lo'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(147, 112, 219, 0.35)',
                        line=dict(color='rgba(147, 112, 219, 0.5)'),
                        hoverinfo='skip',
                        showlegend=True,
                        name='90% CI'
                    ))
                
                fig_roas_spend.add_trace(go.Scatter(
                    x=model_preds_spend['spend'],
                    y=model_preds_spend['roas_actual'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6),
                    hovertemplate='Actual: %{y:.2f}<extra></extra>'
                ))
                
                fig_roas_spend.add_trace(go.Scatter(
                    x=model_preds_spend['spend'],
                    y=model_preds_spend['roas_pred'],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    marker=dict(size=6),
                    customdata=model_preds_spend['roas_error_fmt'],
                    hovertemplate='Predicted: %{y:.2f}<br><span style="color:#FF7043;font-weight:bold">Error: %{customdata}</span><extra></extra>'
                ))
                
                fig_roas_spend.update_layout(
                    title="ROAS vs Spend",
                    xaxis_title="Spend",
                    yaxis_title="ROAS",
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                    height=300
                )
                
                st.plotly_chart(fig_roas_spend, use_container_width=True, key=f"fig_roas_spend_{backtest_timestamp}_{model_key}")


# =============================================================================
# EXPORT PAGE
# =============================================================================
else:  # page == "Export"
    st.subheader("Export Selected Models")
    
    # --- Load Saved Selections ---
    st.markdown("### Load Saved Selections")
    
    # Find existing model_selections CSVs
    train_dir = PROJECT_ROOT / 'data' / 'train'
    existing_selections = sorted(
        [f for f in train_dir.glob('model_selections_*.csv')],
        reverse=True
    )
    
    load_col1, load_col2 = st.columns([1, 1])
    
    with load_col1:
        st.markdown("**From existing file:**")
        if existing_selections:
            selected_file = st.selectbox(
                "Select saved file",
                options=[None] + existing_selections,
                format_func=lambda x: "-- Select --" if x is None else x.name,
                key="load_selection_file"
            )
            if st.button("📂 Load Selected File", disabled=selected_file is None):
                try:
                    df_load = pd.read_csv(selected_file)
                    n = apply_selections_from_df(df_load)
                    st.success(f"Loaded {n} model selections from {selected_file.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        else:
            st.info("No saved model_selections files found in data/train/")
    
    with load_col2:
        st.markdown("**Or upload a file:**")
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=['csv'],
            key="upload_selection_file",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            if st.button("📂 Load Uploaded File"):
                try:
                    df_load = pd.read_csv(uploaded_file)
                    required_cols = ['brand', 'model', 'estimator', 'base_estimator']
                    missing = [c for c in required_cols if c not in df_load.columns]
                    if missing:
                        st.error(f"Missing required columns: {missing}")
                    else:
                        n = apply_selections_from_df(df_load)
                        st.success(f"Loaded {n} model selections from uploaded file")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {e}")
    
    st.markdown("---")
    
    # --- Current Selections ---
    st.markdown("### Current Selections")
    
    if not st.session_state.selected_models:
        st.info("No models selected. Go to 'Model Review' and check 'Use this Model' for models you want to export.")
    else:
        # Build export dataframe
        export_data = []
        for key, metrics in st.session_state.selected_models.items():
            brand, model_key, estimator_label = key
            # Use base_estimator for pipeline lookup in training
            base_estimator = metrics.get('base_estimator', estimator_label)  # Fallback to label if not stored
            row = {
                'brand': brand,
                'model': model_key,
                'estimator': estimator_label,
                'base_estimator': base_estimator,
                '_label': estimator_label,  # Hidden column for key reconstruction
                'bias_pct': st.session_state.edited_bias.get(key, metrics.get('bias_pct', np.nan)),
                'bias_weeks': metrics.get('bias_weeks', BIAS_WINDOW_WEEKS),
                'params': st.session_state.edited_params.get(key, ''),
            }
            export_data.append(row)
        
        df_export = pd.DataFrame(export_data)
        
        # Columns for CSV export (exclude _label)
        export_cols = ['brand', 'model', 'estimator', 'base_estimator', 'bias_pct', 'bias_weeks', 'params']
        
        # Button row
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 1.5, 2.5])
        
        with btn_col1:
            if st.button("🔄 Refresh", help="Reset all Bias % and Params values to original"):
                # Reset edited_bias and edited_params to original values from selected_models
                for key, metrics in st.session_state.selected_models.items():
                    st.session_state.edited_bias[key] = metrics.get('bias_pct', np.nan)
                    st.session_state.edited_params[key] = metrics.get('params', '')
                st.rerun()
        
        with btn_col2:
            # Save CSV to data/train directory
            filename = f"model_selections_{backtest_timestamp}_backtest.csv"
            export_path = PROJECT_ROOT / 'data' / 'train' / filename
            if st.button("📥 Export CSV", help=f"Save to data/train/{filename}"):
                df_export[export_cols].to_csv(export_path, index=False)
                st.success(f"Saved to: data/train/{filename}")
        
        with btn_col3:
            if st.button("🗑️ Remove All", help="Clear all model selections"):
                st.session_state.selected_models = {}
                st.session_state.edited_bias = {}
                st.session_state.edited_params = {}
                st.rerun()
        
        st.markdown("---")
        st.markdown("**Edit bias_pct and params below** (brand/model/estimator are read-only):")
        st.caption("params accepts dict format, e.g.: `{'method':'elastic_net', 'L1_wt':1.0}`")
        
        # Display editable table using data_editor
        # bias_pct and params are editable
        column_config = {
            'brand': st.column_config.TextColumn('brand', disabled=True),
            'model': st.column_config.TextColumn('model', disabled=True),
            'estimator': st.column_config.TextColumn('estimator', disabled=True),
            'base_estimator': st.column_config.TextColumn('base_estimator', disabled=True),
            '_label': None,  # Hide this column
            'bias_pct': st.column_config.NumberColumn('bias_pct', disabled=False, format="%.1f"),
            'bias_weeks': st.column_config.NumberColumn('bias_weeks', disabled=True, format="%d",
                                                        help="Weeks of backtest history the bias was measured over"),
            'params': st.column_config.TextColumn('params', disabled=False, width='large'),
        }
        
        edited_df = st.data_editor(
            df_export,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="export_editor"
        )
        
        # Update edited_bias from data_editor changes
        for idx, row in edited_df.iterrows():
            # Use _label (estimator display label) for key reconstruction
            key = (row['brand'], row['model'], row['_label'])
            new_bias = row['bias_pct']
            new_params = row.get('params', '')
            if key in st.session_state.edited_bias:
                st.session_state.edited_bias[key] = new_bias
            if key in st.session_state.edited_params:
                st.session_state.edited_params[key] = new_params if pd.notna(new_params) else ''
        
        st.caption(f"Dataset: {selected_folder.name}")


# --- Footer ---
st.markdown("---")
st.caption(f"Data loaded from: {selected_folder.name}")
