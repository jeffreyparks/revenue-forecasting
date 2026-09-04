import pandas as pd
import numpy as np
from scipy import stats

from functions.config import RESPONSE_VAR, RESPONSE_VARS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.robust import norms as robust_norms
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression


# PIPELINE_MAP is defined at the bottom of this module after all functions are declared


class StatsmodelsOLS(BaseEstimator, RegressorMixin):

    def __init__(self, add_constant=True):
        self.add_constant = add_constant
        self.results_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y):
        # Statsmodels requires an explicit constant for the intercept
        X_df = pd.DataFrame(X)
        if self.add_constant:
            X_df = sm.add_constant(X_df, has_constant='add')
        
        self.model_ = sm.OLS(y, X_df)
        self.results_ = self.model_.fit()
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns
        return self

    def predict(self, X):
        X_df = pd.DataFrame(X)
        if self.add_constant:
            X_df = sm.add_constant(X_df, has_constant='add')
        return self.results_.predict(X_df)

    def get_intervals(self, X, alpha=0.1):
        X_df = pd.DataFrame(X)
        if self.add_constant:
            X_df = sm.add_constant(X_df, has_constant='add')
        
        # Returns a summary_frame with mean_ci and obs_ci
        prediction_obj = self.results_.get_prediction(X_df)
        return prediction_obj.summary_frame(alpha=alpha)


class StatsmodelsPLS(BaseEstimator, RegressorMixin):
    """
    Partial Least Squares Regression (PLS) for sklearn pipeline compatibility.
    
    Wraps sklearn's PLSRegression and fits statsmodels OLS on the PLS scores
    to provide proper inference (summary, p-values, prediction intervals).
    
    Parameters
    ----------
    add_constant : bool
        Whether to add an intercept term to OLS on scores. Default True.
    n_components : int
        Number of PLS components to extract. Default 2.
    scale : bool
        Whether to scale X and y before PLS. Default True.
    
    Attributes
    ----------
    pls_ : sklearn.cross_decomposition.PLSRegression
        Fitted PLS transformer.
    results_ : statsmodels RegressionResults
        Full OLS results on PLS scores with summary(), predict(), conf_int().
    x_scores_ : np.ndarray
        X scores (latent variables) from PLS fit.
    x_loadings_ : np.ndarray
        X loadings matrix showing feature contributions to components.
    coef_ : np.ndarray
        PLS regression coefficients (original feature space).
    component_names_ : list
        Names of PLS components ('PLS1', 'PLS2', ...).
    feature_names_in_ : list
        Original feature names.
    """
    def __init__(self, add_constant=True, n_components=2, scale=True):
        self.add_constant = add_constant
        self.n_components = n_components
        self.scale = scale
        self.pls_ = None
        self.model_ = None
        self.results_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.component_names_ = None
        self.coef_ = None
        self.x_weights_ = None
        self.x_loadings_ = None
        self.y_loadings_ = None
        self.x_scores_ = None
        self.y_scores_ = None
        self.resid_ = None

    def fit(self, X, y):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y_arr = np.asarray(y).ravel()
        
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = X_df.columns.tolist() if hasattr(X_df, 'columns') else None

        # Step 1: Fit PLS and transform X to scores
        self.pls_ = PLSRegression(n_components=self.n_components, scale=self.scale)
        X_scores = self.pls_.fit_transform(X_df, y_arr)[0]  # Returns (X_scores, Y_scores)
        
        # Store PLS attributes
        self.coef_ = self.pls_.coef_.flatten()
        self.x_weights_ = self.pls_.x_weights_
        self.x_loadings_ = self.pls_.x_loadings_
        self.y_loadings_ = self.pls_.y_loadings_
        self.x_scores_ = self.pls_.x_scores_
        self.y_scores_ = self.pls_.y_scores_
        
        # Create component names
        self.component_names_ = [f'PLS{i+1}' for i in range(self.n_components)]
        
        # Step 2: Create DataFrame with component names for OLS
        X_scores_df = pd.DataFrame(X_scores, columns=self.component_names_)
        
        # Step 3: Add constant for intercept
        if self.add_constant:
            X_scores_df = sm.add_constant(X_scores_df, has_constant='add')
        
        # Step 4: Fit OLS on PLS scores
        self.model_ = sm.OLS(y_arr, X_scores_df)
        self.results_ = self.model_.fit()
        self.resid_ = self.results_.resid
        
        return self

    def predict(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Transform X through PLS to get scores
        X_scores = self.pls_.transform(X_df)
        X_scores_df = pd.DataFrame(X_scores, columns=self.component_names_)
        
        if self.add_constant:
            X_scores_df = sm.add_constant(X_scores_df, has_constant='add')
        
        return self.results_.predict(X_scores_df)

    def get_intervals(self, X, alpha=0.1):
        """
        Get prediction intervals from OLS on PLS scores.
        
        Returns full statsmodels prediction intervals since we have
        a proper RegressionResults object.
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Transform X through PLS to get scores
        X_scores = self.pls_.transform(X_df)
        X_scores_df = pd.DataFrame(X_scores, columns=self.component_names_)
        
        if self.add_constant:
            X_scores_df = sm.add_constant(X_scores_df, has_constant='add')
        
        prediction_obj = self.results_.get_prediction(X_scores_df)
        return prediction_obj.summary_frame(alpha=alpha)

    def summary(self):
        """Return OLS summary on PLS components."""
        if self.results_ is not None:
            return self.results_.summary()
        return None

    def get_loadings(self):
        """
        Get PLS loadings matrix (features × components).
        
        Returns
        -------
        pd.DataFrame
            Loadings showing how original features contribute to each component.
        """
        if self.x_loadings_ is None:
            return None
        return pd.DataFrame(
            self.x_loadings_,
            index=self.feature_names_in_,
            columns=self.component_names_
        )

    def get_original_coefficients(self):
        """
        Get PLS coefficients in original feature space.
        
        Returns
        -------
        pd.Series
            Coefficients for each original feature.
        """
        if self.coef_ is None:
            return None
        return pd.Series(
            self.coef_,
            index=self.feature_names_in_,
            name='pls_coefficient'
        )
    

class StatsmodelsWLS(BaseEstimator, RegressorMixin):
    """
    Statsmodels WLS estimator for sklearn Pipeline compatibility.
    
    Computes weights as 1/(weight_col^weight_power) to handle heteroskedasticity
    where variance scales with the weight column (e.g., spend).
    
    Parameters
    ----------
    add_constant : bool
        Whether to add an intercept term. Default True.
    weight_col : str
        Column name to use for weight calculation. Default 'spend'.
    weight_power : float or None
        Power for weight calculation:
        - 1: Variance ∝ spend (std ∝ sqrt(spend))
        - 2: Variance ∝ spend² (std ∝ spend, constant coefficient of variation)
        - None: Auto-estimate k from OLS residuals during fit (recommended)
        Default None (auto-estimate).
    min_obs_for_estimation : int
        Minimum observations required to estimate k. If fewer, falls back to k=2.
        Default 10.
    """

    def __init__(self, add_constant=True, weight_col='spend', weight_power=None, 
                 min_obs_for_estimation=10):
        self.add_constant = add_constant
        self.weight_col = weight_col
        self.weight_power = weight_power
        self.min_obs_for_estimation = min_obs_for_estimation
        self.results_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.estimated_k_ = None  # Stores estimated k if auto-estimated

    def _estimate_variance_power(self, X, y):
        """
        Estimate k in Var(ε) = σ² * weight_col^k using log-log regression.
        
        Returns estimated k, or 2.0 as fallback if estimation fails.
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        weight_vals = X_df[self.weight_col].values
        n = len(y)
        
        # Need minimum observations for reliable estimation
        if n < self.min_obs_for_estimation:
            return 2.0
        
        # Stage 1: Fit OLS to get residuals
        X_ols = sm.add_constant(X_df) if self.add_constant else X_df
        try:
            ols_fit = sm.OLS(y, X_ols).fit()
            residuals = ols_fit.resid
        except Exception:
            return 2.0
        
        # Stage 2: Regress log(residuals²) on log(weight_col)
        # Filter out zero/negative values to avoid log(0)
        mask = weight_vals > 0
        if mask.sum() < self.min_obs_for_estimation:
            return 2.0
        
        try:
            log_resid_sq = np.log(residuals[mask] ** 2)
            log_weight = np.log(weight_vals[mask])
            X_var = sm.add_constant(log_weight)
            var_fit = sm.OLS(log_resid_sq, X_var).fit()
            k = var_fit.params.iloc[1]  # slope = k
            
            # Sanity check: k should be positive and reasonable
            if np.isnan(k) or k < 0 or k > 4:
                return 2.0
            return k
        except Exception:
            return 2.0

    def fit(self, X, y):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Extract weight column values
        if self.weight_col not in X_df.columns:
            raise ValueError(f"Weight column '{self.weight_col}' not found in X. "
                           f"Available columns: {list(X_df.columns)}")
        
        weight_vals = X_df[self.weight_col].values
        
        # Determine weight power: use provided or estimate
        if self.weight_power is None:
            self.estimated_k_ = self._estimate_variance_power(X_df, y)
            k = self.estimated_k_
        else:
            self.estimated_k_ = None
            k = self.weight_power
        
        # Compute weights: 1/spend^k
        # Use max() to avoid division by zero or very small values
        weights = 1 / (np.maximum(np.abs(weight_vals), 1e-8) ** k)
        
        # Store features info
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = X_df.columns if hasattr(X_df, 'columns') else None
        
        # Add constant for intercept
        if self.add_constant:
            X_df = sm.add_constant(X_df, has_constant='add')
        
        # Fit WLS
        self.model_ = sm.WLS(y, X_df, weights=weights)
        self.results_ = self.model_.fit()
        return self

    def predict(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        if self.add_constant:
            X_df = sm.add_constant(X_df, has_constant='add')
        return self.results_.predict(X_df)

    def get_intervals(self, X, alpha=0.1):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        if self.add_constant:
            X_df = sm.add_constant(X_df, has_constant='add')
        prediction_obj = self.results_.get_prediction(X_df)
        return prediction_obj.summary_frame(alpha=alpha)


class StatsmodelsRegularizedOLS(BaseEstimator, RegressorMixin):
    """
    Statsmodels Regularized OLS estimator for sklearn Pipeline compatibility.
    
    Supports Ridge (L2), Lasso (L1), and Elastic Net regularization via
    statsmodels' fit_regularized() method.
    
    Parameters
    ----------
    add_constant : bool
        Whether to add an intercept term. Default True.
    alpha : float
        The penalty weight / regularization strength. Higher values = more
        regularization = smaller coefficients. Default 1.0.
    L1_wt : float
        The elastic net mixing parameter, between 0 and 1:
        - L1_wt=0: Ridge regression (pure L2 penalty)
        - L1_wt=1: Lasso regression (pure L1 penalty)
        - 0 < L1_wt < 1: Elastic Net (mix of L1 and L2)
        Default 0 (Ridge).
    start_params : array-like or None
        Starting values for optimization. If None, uses OLS estimates.
        Default None.
    profile_scale : bool
        If True, the scale parameter is profiled out of the likelihood.
        Default False.
    refit : bool
        If True and L1_wt > 0 (Lasso/Elastic Net), refit using standard OLS
        on the selected (non-zero) features. This gives you a full 
        RegressionResults object with summary(), get_prediction(), conf_int(),
        etc. Removes shrinkage bias from coefficients. Default False.
    maxiter : int
        Maximum iterations for coordinate descent algorithm. Default 1000.
    cnvrg_tol : float
        Convergence tolerance for coordinate descent. Default 1e-8.
    zero_tol : float
        Coefficients below this threshold are set to zero (for Lasso/Elastic Net).
        Default 1e-8.
    
    Attributes
    ----------
    is_ols_refit_ : bool
        True if OLS refit was performed (refit=True with L1_wt > 0).
    selected_features_ : list
        Column names of features kept after Lasso selection.
    selected_feature_names_ : list
        Feature names excluding 'const' (property).
        
    Notes
    -----
    The penalty function is: alpha * (L1_wt * ||beta||_1 + 0.5 * (1 - L1_wt) * ||beta||_2^2)
    
    Unlike sklearn's regularized estimators, statsmodels does NOT automatically
    standardize features. For comparable regularization across features with
    different scales, consider standardizing in the pipeline.
    
    The intercept (constant term) is NOT penalized by default.
    
    Examples
    --------
    >>> # Ridge regression (L2 penalty)
    >>> model = StatsmodelsRegularizedOLS(alpha=1.0, L1_wt=0)
    >>> model.fit(X, y)
    >>> 
    >>> # Lasso for feature selection with OLS refit
    >>> model = StatsmodelsRegularizedOLS(alpha=0.1, L1_wt=1, refit=True)
    >>> model.fit(X, y)
    >>> print(model.results_.summary())  # Full OLS summary!
    >>> print(model.selected_feature_names_)  # Which features were kept
    >>> 
    >>> # Elastic Net (50% L1, 50% L2)
    >>> model = StatsmodelsRegularizedOLS(alpha=0.5, L1_wt=0.5)
    """

    def __init__(self, add_constant=True, alpha=1.0, L1_wt=0, start_params=None,
                 profile_scale=False, refit=False, maxiter=1000, cnvrg_tol=1e-8, 
                 zero_tol=1e-8):
        self.add_constant = add_constant
        self.alpha = alpha
        self.L1_wt = L1_wt
        self.start_params = start_params
        self.profile_scale = profile_scale
        self.refit = refit
        self.maxiter = maxiter
        self.cnvrg_tol = cnvrg_tol
        self.zero_tol = zero_tol
        self.results_ = None
        self.model_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.resid_ = None  # Store residuals manually
        self.scale_ = None  # Store residual standard error
        self.df_resid_ = None  # Store degrees of freedom
        self.is_ols_refit_ = False  # True if refitted with OLS
        self.selected_features_ = None  # Features kept after Lasso selection
        self.all_columns_ = None  # All columns including constant

    def fit(self, X, y):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y_arr = np.asarray(y).ravel()
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = X_df.columns.tolist() if hasattr(X_df, 'columns') else None
        
        if self.add_constant:
            X_df = sm.add_constant(X_df, has_constant='add')
        
        # Store all column names for prediction
        self.all_columns_ = X_df.columns.tolist()
        
        # Stage 1: Fit regularized model
        self.model_ = sm.OLS(y_arr, X_df)
        reg_results = self.model_.fit_regularized(
            alpha=self.alpha,
            L1_wt=self.L1_wt,
            start_params=self.start_params,
            profile_scale=self.profile_scale,
            refit=False,  # We handle refit ourselves
            maxiter=self.maxiter,
            cnvrg_tol=self.cnvrg_tol,
            zero_tol=self.zero_tol
        )
        
        # Stage 2: If refit=True and using L1 (Lasso/Elastic Net), refit OLS on selected features
        if self.refit and self.L1_wt > 0:
            # Identify non-zero coefficients
            nonzero_mask = np.abs(reg_results.params) > self.zero_tol
            self.selected_features_ = [col for col, keep in zip(X_df.columns, nonzero_mask) if keep]
            
            if len(self.selected_features_) > 0:
                # Refit standard OLS on selected features only
                X_selected = X_df[self.selected_features_]
                self.model_ = sm.OLS(y_arr, X_selected)
                self.results_ = self.model_.fit()  # Full RegressionResults!
                self.is_ols_refit_ = True
                self.resid_ = self.results_.resid
            else:
                # No features selected - keep regularized results
                self.results_ = reg_results
                self.selected_features_ = self.all_columns_
                self.is_ols_refit_ = False
                y_pred = self.results_.predict(X_df)
                self.resid_ = y_arr - y_pred
        else:
            # No refit - use regularized results
            self.results_ = reg_results
            self.selected_features_ = self.all_columns_
            self.is_ols_refit_ = False
            y_pred = self.results_.predict(X_df)
            self.resid_ = y_arr - y_pred
        
        # Compute scale (residual standard error)
        n = len(self.resid_)
        p = len(self.selected_features_)
        self.df_resid_ = n - p
        if self.df_resid_ > 0:
            self.scale_ = np.sqrt(np.sum(self.resid_ ** 2) / self.df_resid_)
        else:
            self.scale_ = np.nan
        
        return self

    def predict(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        if self.add_constant:
            X_df = sm.add_constant(X_df, has_constant='add')
        
        # Use only selected features if OLS refit was performed
        if self.is_ols_refit_:
            X_df = X_df[self.selected_features_]
        
        return self.results_.predict(X_df)

    def get_intervals(self, X, alpha=0.1):
        """
        Get prediction intervals for regularized or refitted OLS regression.
        
        If refit=True was used with Lasso, this returns proper OLS intervals.
        Otherwise, returns approximate intervals using residual standard error.
        
        Parameters
        ----------
        X : array-like
            Feature matrix for prediction.
        alpha : float
            Significance level for intervals (default 0.1 = 90% intervals).
        
        Returns
        -------
        pd.DataFrame
            DataFrame with mean predictions and intervals.
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        if self.add_constant:
            X_df = sm.add_constant(X_df, has_constant='add')
        
        # Use only selected features if OLS refit was performed
        if self.is_ols_refit_:
            X_df = X_df[self.selected_features_]
            # Use proper OLS prediction intervals
            prediction_obj = self.results_.get_prediction(X_df)
            return prediction_obj.summary_frame(alpha=alpha)
        
        # Fallback: approximate intervals for regularized results
        predictions = self.results_.predict(X_df)
        
        scale = self.scale_
        df_resid = self.df_resid_
        
        from scipy import stats as scipy_stats
        if scale is not None and not np.isnan(scale) and df_resid > 0:
            t_val = scipy_stats.t.ppf(1 - alpha/2, df_resid)
            margin = t_val * scale
            mean_se = scale / np.sqrt(len(predictions))
        else:
            t_val = np.nan
            margin = np.nan
            mean_se = np.nan
        
        return pd.DataFrame({
            'mean': predictions,
            'mean_se': mean_se,
            'mean_ci_lower': predictions - t_val * mean_se if not np.isnan(t_val) else np.nan,
            'mean_ci_upper': predictions + t_val * mean_se if not np.isnan(t_val) else np.nan,
            'obs_ci_lower': predictions - margin,
            'obs_ci_upper': predictions + margin,
        })

    @property
    def coef_(self):
        """Return coefficients (excluding intercept if present)."""
        if self.results_ is None:
            return None
        params = self.results_.params
        if self.add_constant:
            return params[1:]  # Skip constant
        return params

    @property
    def intercept_(self):
        """Return intercept if add_constant=True, else 0."""
        if self.results_ is None:
            return None
        if self.add_constant:
            return self.results_.params[0]
        return 0.0

    @property
    def n_nonzero_coefs_(self):
        """Return count of non-zero coefficients (useful for Lasso/Elastic Net)."""
        if self.results_ is None:
            return None
        coefs = self.coef_
        return np.sum(np.abs(coefs) > self.zero_tol)

    @property
    def selected_feature_names_(self):
        """Return names of features kept after Lasso selection (excludes constant)."""
        if self.selected_features_ is None:
            return None
        return [f for f in self.selected_features_ if f != 'const']


class StatsmodelsRLM(BaseEstimator, RegressorMixin):
    """
    Statsmodels Robust Linear Model (RLM) estimator for sklearn Pipeline compatibility.
    
    Uses M-estimation to down-weight outliers, providing robust regression
    that is less sensitive to influential points and heavy-tailed distributions.
    
    Parameters
    ----------
    add_constant : bool
        Whether to add an intercept term. Default True.
    norm : str or statsmodels.robust.norms object
        M-estimator norm function. Options:
        - 'huber' or 'HuberT': Moderate outlier down-weighting (default)
        - 'tukey' or 'TukeyBiweight': Aggressive outlier rejection
        - 'andrews' or 'AndrewWave': Redescending, similar to Tukey
        - 'ramsay' or 'RamsayE': Exponential decay
        - 'hampel' or 'Hampel': Three-part redescending
        - 'leastsquares' or 'LeastSquares': No down-weighting (OLS equivalent)
        - Or pass a statsmodels.robust.norms object directly
        Default 'huber'.
    scale_est : str
        Scale estimator for residuals. Options:
        - 'mad': Median Absolute Deviation (default, highly robust)
        - 'huber': Huber's Proposal 2 scale estimate
        - 'stand_mad': Standardized MAD (×1.4826 for normal efficiency)
        Default 'mad'.
    tune : float or None
        Tuning constant for the norm function. If None, uses norm default:
        - HuberT: 1.345 (95% efficiency under normality)
        - TukeyBiweight: 4.685 (95% efficiency)
        - AndrewWave: 1.339 * pi
        Default None.
    maxiter : int
        Maximum iterations for IRLS. Default 50.
    """

    # Mapping of string names to norm classes
    NORM_MAP = {
        'huber': robust_norms.HuberT,
        'hubert': robust_norms.HuberT,
        'tukey': robust_norms.TukeyBiweight,
        'tukeybiweight': robust_norms.TukeyBiweight,
        'bisquare': robust_norms.TukeyBiweight,
        'andrews': robust_norms.AndrewWave,
        'andrewwave': robust_norms.AndrewWave,
        'ramsay': robust_norms.RamsayE,
        'ramsaye': robust_norms.RamsayE,
        'hampel': robust_norms.Hampel,
        'leastsquares': robust_norms.LeastSquares,
    }

    def __init__(self, add_constant=True, norm='huber', scale_est='mad', 
                 tune=None, maxiter=50):
        self.add_constant = add_constant
        self.norm = norm
        self.scale_est = scale_est
        self.tune = tune
        self.maxiter = maxiter
        self.results_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def _get_norm(self):
        """Convert norm parameter to statsmodels norm object."""
        if isinstance(self.norm, str):
            norm_key = self.norm.lower().replace('_', '')
            if norm_key not in self.NORM_MAP:
                raise ValueError(
                    f"Unknown norm '{self.norm}'. Available: {list(self.NORM_MAP.keys())}"
                )
            norm_class = self.NORM_MAP[norm_key]
            # Apply custom tuning constant if provided
            if self.tune is not None:
                return norm_class(c=self.tune)
            return norm_class()
        else:
            # Assume it's already a norm object
            return self.norm

    def _get_scale_est(self):
        """Get scale estimator string for statsmodels."""
        scale_map = {
            'mad': 'mad',
            'huber': 'HuberScale',
            'stand_mad': 'stand_mad',
        }
        return scale_map.get(self.scale_est.lower(), 'mad')

    def fit(self, X, y):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = X_df.columns if hasattr(X_df, 'columns') else None
        
        if self.add_constant:
            X_df = sm.add_constant(X_df, has_constant='add')
        
        norm_obj = self._get_norm()
        self.model_ = sm.RLM(y, X_df, M=norm_obj)
        self.results_ = self.model_.fit(
            scale_est=self._get_scale_est(),
            maxiter=self.maxiter
        )
        return self

    def predict(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        if self.add_constant:
            X_df = sm.add_constant(X_df, has_constant='add')
        return self.results_.predict(X_df)

    def get_intervals(self, X, alpha=0.1):
        """
        Get prediction intervals using robust scale estimate.
        
        Note: RLM doesn't have built-in prediction intervals like OLS.
        This uses the robust scale estimate to construct approximate intervals.
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        if self.add_constant:
            X_df = sm.add_constant(X_df, has_constant='add')
        
        predictions = self.results_.predict(X_df)
        
        # Use robust scale estimate for interval width
        scale = self.results_.scale
        
        # Calculate t-value for confidence level
        from scipy import stats as scipy_stats
        df_resid = self.results_.df_resid
        t_val = scipy_stats.t.ppf(1 - alpha/2, df_resid)
        
        # Approximate prediction interval (conservative)
        # For prediction intervals, we need to account for both
        # estimation uncertainty and future observation variability
        margin = t_val * scale
        
        # Return DataFrame matching OLS format
        return pd.DataFrame({
            'mean': predictions,
            'mean_se': scale / np.sqrt(len(predictions)),  # Approximate
            'mean_ci_lower': predictions - t_val * scale / np.sqrt(len(predictions)),
            'mean_ci_upper': predictions + t_val * scale / np.sqrt(len(predictions)),
            'obs_ci_lower': predictions - margin,
            'obs_ci_upper': predictions + margin,
        })

    @property
    def weights_(self):
        """Return the weights assigned to each observation during fitting."""
        if self.results_ is None:
            return None
        return self.results_.weights
    

class StatsmodelsPCAOLS(BaseEstimator, RegressorMixin):
    """
    PCA dimensionality reduction followed by statsmodels OLS regression.
    
    Addresses multicollinearity by transforming correlated features into
    uncorrelated principal components, then fitting OLS on components.
    Returns a full RegressionResults object with proper inference.
    
    Parameters
    ----------
    add_constant : bool
        Whether to add an intercept term. Default True.
    n_components : int, float, or None
        Number of components to keep:
        - int: Exact number of components
        - float (0 < n < 1): Minimum variance explained (e.g., 0.95 = 95%)
        - None: Keep all components (min(n_samples, n_features))
        Default 0.95 (keep components explaining 95% of variance).
    standardize : bool
        Whether to standardize features before PCA. Should almost always
        be True since PCA is sensitive to scale. Default True.
    
    Attributes
    ----------
    pca_ : sklearn.decomposition.PCA
        Fitted PCA transformer.
    scaler_ : sklearn.preprocessing.StandardScaler or None
        Fitted scaler (if standardize=True).
    results_ : statsmodels RegressionResults
        Full OLS results object with summary(), predict(), conf_int(), etc.
    loadings_ : pd.DataFrame
        PCA loadings matrix (features × components). Shows how original
        features contribute to each component.
    variance_explained_ : np.array
        Variance explained by each component (ratio).
    cumulative_variance_ : np.array
        Cumulative variance explained.
    component_names_ : list
        Names of the principal components ('PC1', 'PC2', ...).
    feature_names_in_ : list
        Original feature names before PCA.
    
    Notes
    -----
    PCA Best Practices:
    1. Always standardize before PCA (features on different scales will
       dominate variance otherwise)
    2. Choose n_components based on variance explained or scree plot
    3. Components are orthogonal → no multicollinearity, VIF = 1.0
    4. Coefficients are on components, not original features
    5. Use loadings_ to interpret what each component represents
    
    Component Interpretation:
    - High positive loading: feature increases with component
    - High negative loading: feature decreases with component
    - Near-zero loading: feature doesn't contribute to component
    
    Examples
    --------
    >>> model = StatsmodelsPCAOLS(n_components=0.95)
    >>> model.fit(X, y)
    >>> print(model.results_.summary())  # Full OLS summary
    >>> print(model.loadings_)  # Feature contributions to components
    >>> print(model.variance_explained_)  # Variance per component
    """

    def __init__(self, add_constant=True, n_components=0.95, standardize=True):
        self.add_constant = add_constant
        self.n_components = n_components
        self.standardize = standardize
        self.pca_ = None
        self.scaler_ = None
        self.results_ = None
        self.model_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.loadings_ = None
        self.variance_explained_ = None
        self.cumulative_variance_ = None
        self.component_names_ = None

    def fit(self, X, y):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y_arr = np.asarray(y).ravel()
        
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = X_df.columns.tolist() if hasattr(X_df, 'columns') else None
        
        # Step 1: Standardize (recommended for PCA)
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_df)
        else:
            X_scaled = X_df.values if hasattr(X_df, 'values') else X_df
        
        # Step 2: PCA
        self.pca_ = PCA(n_components=self.n_components)
        X_pca = self.pca_.fit_transform(X_scaled)
        
        # Store PCA diagnostics
        n_components_kept = self.pca_.n_components_
        self.component_names_ = [f'PC{i+1}' for i in range(n_components_kept)]
        self.variance_explained_ = self.pca_.explained_variance_ratio_
        self.cumulative_variance_ = np.cumsum(self.variance_explained_)
        
        # Create loadings matrix (features × components)
        self.loadings_ = pd.DataFrame(
            self.pca_.components_.T,
            index=self.feature_names_in_,
            columns=self.component_names_
        )
        
        # Step 3: Create DataFrame with component names
        X_pca_df = pd.DataFrame(X_pca, columns=self.component_names_)
        
        # Step 4: Add constant for intercept
        if self.add_constant:
            X_pca_df = sm.add_constant(X_pca_df, has_constant='add')
        
        # Step 5: Fit OLS
        self.model_ = sm.OLS(y_arr, X_pca_df)
        self.results_ = self.model_.fit()  # Full RegressionResults!
        
        return self

    def predict(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Transform through pipeline
        if self.standardize:
            X_scaled = self.scaler_.transform(X_df)
        else:
            X_scaled = X_df.values if hasattr(X_df, 'values') else X_df
        
        X_pca = self.pca_.transform(X_scaled)
        X_pca_df = pd.DataFrame(X_pca, columns=self.component_names_)
        
        if self.add_constant:
            X_pca_df = sm.add_constant(X_pca_df, has_constant='add')
        
        return self.results_.predict(X_pca_df)

    def get_intervals(self, X, alpha=0.1):
        """
        Get prediction intervals from OLS results.
        
        Returns full statsmodels prediction intervals since we have
        a proper RegressionResults object.
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Transform through pipeline
        if self.standardize:
            X_scaled = self.scaler_.transform(X_df)
        else:
            X_scaled = X_df.values if hasattr(X_df, 'values') else X_df
        
        X_pca = self.pca_.transform(X_scaled)
        X_pca_df = pd.DataFrame(X_pca, columns=self.component_names_)
        
        if self.add_constant:
            X_pca_df = sm.add_constant(X_pca_df, has_constant='add')
        
        # Full OLS prediction intervals
        prediction_obj = self.results_.get_prediction(X_pca_df)
        return prediction_obj.summary_frame(alpha=alpha)

    def get_component_coefficients(self):
        """
        Get regression coefficients on principal components.
        
        Returns
        -------
        pd.Series
            Coefficients for each PC (and intercept if add_constant=True).
        """
        if self.results_ is None:
            return None
        return pd.Series(self.results_.params, 
                        index=self.results_.params.index,
                        name='coefficient')

    def get_original_feature_importance(self):
        """
        Approximate feature importance by combining PCA loadings with 
        component coefficients.
        
        Returns
        -------
        pd.DataFrame
            For each original feature:
            - contribution: Sum of |loading × coefficient| across components
            - direction: Net direction (positive = increases response)
            
        Notes
        -----
        This is an approximation. The true relationship between original
        features and response is mediated through components. Use loadings_
        for detailed interpretation.
        """
        if self.results_ is None or self.loadings_ is None:
            return None
        
        # Get coefficients for components only (exclude intercept)
        coefs = self.results_.params
        if self.add_constant:
            coefs = coefs[1:]  # Skip 'const'
        
        # Weighted loadings: loading × coefficient
        weighted_loadings = self.loadings_.mul(coefs.values, axis=1)
        
        # Sum across components for net effect
        net_effect = weighted_loadings.sum(axis=1)
        
        # Absolute contribution (ignoring direction)
        abs_contribution = weighted_loadings.abs().sum(axis=1)
        
        return pd.DataFrame({
            'contribution': abs_contribution,
            'direction': net_effect,
            'normalized_contribution': abs_contribution / abs_contribution.sum()
        }).sort_values('contribution', ascending=False)

    def scree_data(self):
        """
        Get data for scree plot visualization.
        
        Returns
        -------
        pd.DataFrame
            component, variance_explained, cumulative_variance for plotting.
        """
        if self.variance_explained_ is None:
            return None
        return pd.DataFrame({
            'component': self.component_names_,
            'variance_explained': self.variance_explained_,
            'cumulative_variance': self.cumulative_variance_
        })


# =============================================================================
# PIPELINE DEFINITIONS
# =============================================================================

def pipeline_ols_std(std_cols=None):
    """
    Default OLS pipeline with StandardScaler.
    
    Parameters
    ----------
    std_cols : list, optional
        Columns to standardize. Defaults to ['spend'].
    
    Returns
    -------
    Pipeline
        Sklearn pipeline with column transformer and StatsmodelsOLS.
    """
    if std_cols is None:
        std_cols = ['spend']
    
    col_trans = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), std_cols)
        ], 
        remainder='passthrough',
        verbose_feature_names_out=False 
    ).set_output(transform='pandas')
    
    return Pipeline([
        ('col_trans', col_trans),
        ('model', StatsmodelsOLS()),
    ])


def pipeline_ols_pwr(pwr_cols=None, method='yeo-johnson', std=True):
    """
    OLS pipeline with Power Transform and standardization

    Parameters
    ----------
    pwr_cols = list, optional
        Columns to power transform. Defaults to ['spend'].

    method = str, optional
        'yeo-johnson' or 'box-cox'

    std = boolean, optional
        Apply standardization after power transform

    Returns
    -------
    Pipeline
        SKlearn pipeline
    """
    if pwr_cols is None:
        pwr_cols = ['spend']

    col_trans = ColumnTransformer(
        transformers=[
            ('power', PowerTransformer(method=method, standardize=std), pwr_cols)
        ], 
        remainder='passthrough',
        verbose_feature_names_out=False 
    ).set_output(transform='pandas')

    return Pipeline([
        ('col_trans', col_trans),
        ('model', StatsmodelsOLS()),
    ])


def pipeline_wls(weight_col='spend', weight_power=None):
    """
    Weighted Least Squares pipeline with automatic weight computation.
    
    Computes weights as 1/weight_col^weight_power to handle heteroskedasticity
    where variance scales with the weight column (typically spend).
    
    Parameters
    ----------
    weight_col : str
        Column to use for weight calculation. Default 'spend'.
    weight_power : float or None
        Power for weight calculation:
        - None: Auto-estimate k from OLS residuals during fit (recommended)
        - 1: Variance ∝ spend (std ∝ sqrt(spend))
        - 2: Variance ∝ spend² (std ∝ spend, constant CV)
        Default None (auto-estimate per brand/model).
    
    Returns
    -------
    Pipeline
        Sklearn pipeline with WLS model.
    
    Notes
    -----
    When weight_power=None, the model estimates k during fit() by:
    1. Fitting OLS to get residuals
    2. Regressing log(residuals²) on log(spend) to estimate k
    3. Computing weights as 1/spend^k
    
    This means each brand/model gets its own estimated k value automatically.
    The estimated k is stored in pipe.named_steps['model'].estimated_k_
    
    This pipeline does NOT scale features, since:
    1. Weights handle the variance heterogeneity
    2. Scaling would complicate weight computation
    3. Coefficients remain interpretable as "per unit spend"
    
    Examples
    --------
    >>> # Auto-estimate k for each brand/model (recommended)
    >>> df_models = lr.train_models(df_train, pipeline_fn=lr.pipeline_wls)
    >>> 
    >>> # Check estimated k for a specific model
    >>> k = df_models.iloc[0]['model_obj'].named_steps['model'].estimated_k_
    >>> 
    >>> # Fixed weight_power if you prefer
    >>> from functools import partial
    >>> df_models = lr.train_models(df_train, pipeline_fn=partial(lr.pipeline_wls, weight_power=2))
    """
    return Pipeline([
        ('model', StatsmodelsWLS(
            weight_col=weight_col,
            weight_power=weight_power
        )),
    ])


def pipeline_rlm(norm='huber', scale_est='mad', tune=None, std_cols=None):
    """
    Robust Linear Model pipeline with configurable M-estimator.
    
    Uses iteratively reweighted least squares (IRLS) to down-weight
    outliers, providing resistance to influential points.
    
    Parameters
    ----------
    norm : str
        M-estimator norm function:
        - 'huber': Moderate down-weighting, preserves some outlier influence
                   Good for data with occasional moderate outliers
        - 'tukey': Aggressive rejection, zero weight for extreme outliers
                   Good for heavy contamination (>5% outliers)
        - 'andrews': Redescending estimator, smooth transition
        - 'hampel': Three-part redescending, configurable thresholds
        - 'ramsay': Exponential decay for light-tailed outliers
        Default 'huber'.
    scale_est : str
        Scale estimator:
        - 'mad': Median Absolute Deviation (default, highly robust)
        - 'huber': Huber's Proposal 2 (less robust, more efficient)
        - 'stand_mad': Standardized MAD for asymptotic normality
        Default 'mad'.
    tune : float or None
        Tuning constant for the norm. Higher = less down-weighting.
        Common values:
        - HuberT: 1.345 (default), 1.0 (more robust), 2.0 (less robust)
        - TukeyBiweight: 4.685 (default), 3.0 (more robust), 6.0 (less robust)
        Default None (use norm's default).
    std_cols : list or None
        Columns to standardize before fitting. Default None (no standardization).
        Note: RLM is scale-equivariant, so standardization is optional.
    
    Returns
    -------
    Pipeline
        Sklearn pipeline with RLM model.
    
    Notes
    -----
    Key differences from OLS/WLS:
    - Automatically down-weights outliers based on residual magnitude
    - More robust coefficient estimates when data has influential points
    - Weights can be inspected via: pipe.named_steps['model'].weights_
    
    Norm selection guide:
    - Shapiro p-value low + few influential points → 'huber'
    - Many influential points (>10%) → 'tukey'
    - Heteroskedasticity present → consider WLS instead or WLS+RLM hybrid
    
    Examples
    --------
    >>> # Default Huber norm (moderate robustness)
    >>> df_models = lr.train_models(df_train, pipeline_fn=lr.pipeline_rlm)
    >>> 
    >>> # Tukey for aggressive outlier rejection
    >>> from functools import partial
    >>> df_models = lr.train_models(
    ...     df_train, 
    ...     pipeline_fn=partial(lr.pipeline_rlm, norm='tukey')
    ... )
    >>> 
    >>> # Custom tuning constant (more aggressive Huber)
    >>> df_models = lr.train_models(
    ...     df_train,
    ...     pipeline_fn=partial(lr.pipeline_rlm, norm='huber', tune=1.0)
    ... )
    >>> 
    >>> # Check which observations were down-weighted
    >>> weights = df_models.iloc[0]['model_obj'].named_steps['model'].weights_
    >>> outlier_mask = weights < 0.5  # Significantly down-weighted
    """
    steps = []
    
    if std_cols is not None:
        col_trans = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), std_cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform='pandas')
        steps.append(('col_trans', col_trans))
    
    steps.append(('model', StatsmodelsRLM(
        norm=norm,
        scale_est=scale_est,
        tune=tune
    )))
    
    return Pipeline(steps)


def pipeline_regularized(alpha=1.0, L1_wt=0, std_cols=None, refit=False):
    """
    Regularized OLS pipeline supporting Ridge, Lasso, and Elastic Net.
    
    Uses statsmodels' fit_regularized() to apply L1/L2 penalties.
    
    Parameters
    ----------
    alpha : float
        Regularization strength. Higher values = more regularization.
        Common ranges:
        - Ridge: 0.1 to 100+
        - Lasso: 0.001 to 1
        - Choose via cross-validation or held-out test set
        Default 1.0.
    L1_wt : float
        The elastic net mixing parameter, between 0 and 1:
        - L1_wt=0: Ridge regression (pure L2 penalty) - shrinks coefficients
        - L1_wt=1: Lasso regression (pure L1 penalty) - can zero out coefficients
        - 0 < L1_wt < 1: Elastic Net - combines both behaviors
        Default 0 (Ridge).
    std_cols : list or None
        Columns to standardize before fitting. Default None.
        Note: Standardization is RECOMMENDED for regularized regression to
        ensure all features are penalized equally regardless of scale.
    refit : bool
        If True and L1_wt > 0 (Lasso/Elastic Net), refit using standard OLS
        on the selected (non-zero) features. This gives you:
        - Full RegressionResults with summary(), get_prediction(), conf_int()
        - Unbiased coefficient estimates (no shrinkage)
        - Proper prediction intervals
        Default False.
    
    Returns
    -------
    Pipeline
        Sklearn pipeline with optional standardization and regularized OLS.
    
    Notes
    -----
    Regularization Selection Guide:
    - Ridge (L1_wt=0): Use when you want to keep all features but shrink 
      coefficients toward zero. Good for multicollinearity.
    - Lasso (L1_wt=1): Use for feature selection. Some coefficients will be
      exactly zero. Good when you expect only a subset of features matter.
    - Elastic Net (0 < L1_wt < 1): Combines both. Use when you want some
      feature selection but also want correlated features grouped together.
    
    Alpha (penalty strength) should be tuned via cross-validation for best
    performance. Higher alpha = more regularization = simpler model.
    
    Examples
    --------
    >>> # Ridge regression with standardization
    >>> from functools import partial
    >>> df_models = lr.train_models(
    ...     df_train,
    ...     pipeline_fn=partial(lr.pipeline_regularized, alpha=1.0, L1_wt=0, std_cols=['spend'])
    ... )
    >>> 
    >>> # Lasso for feature selection with OLS refit (recommended)
    >>> df_models = lr.train_models(
    ...     df_train,
    ...     pipeline_fn=partial(lr.pipeline_regularized, alpha=0.1, L1_wt=1, std_cols=['spend'], refit=True)
    ... )
    >>> 
    >>> # Access full OLS results after Lasso selection
    >>> model_obj = df_models.iloc[0]['model_obj'].named_steps['model']
    >>> print(model_obj.results_.summary())  # Full OLS summary
    >>> print(model_obj.selected_feature_names_)  # Which features were kept
    >>> 
    >>> # Elastic Net (50% L1, 50% L2)
    >>> df_models = lr.train_models(
    ...     df_train,
    ...     pipeline_fn=partial(lr.pipeline_regularized, alpha=0.5, L1_wt=0.5, std_cols=['spend'])
    ... )
    """
    steps = []
    
    if std_cols is not None:
        col_trans = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), std_cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform='pandas')
        steps.append(('col_trans', col_trans))
    
    steps.append(('model', StatsmodelsRegularizedOLS(
        alpha=alpha,
        L1_wt=L1_wt,
        refit=refit
    )))
    
    return Pipeline(steps)


def pipeline_pca_ols(n_components=0.95, standardize=True):
    """
    PCA dimensionality reduction + OLS pipeline for multicollinearity.
    
    Transforms correlated features into uncorrelated principal components,
    then fits OLS on components. Returns full RegressionResults with proper
    inference (summary, p-values, prediction intervals).
    
    Parameters
    ----------
    n_components : int, float, or None
        Number of components to keep:
        - int: Exact number of components (e.g., 5)
        - float (0 < n < 1): Minimum variance explained (e.g., 0.95 = 95%)
        - None: Keep all components
        Default 0.95 (95% variance explained).
    standardize : bool
        Whether to standardize features before PCA. Highly recommended
        since PCA is sensitive to feature scales. Default True.
    
    Returns
    -------
    Pipeline
        Sklearn pipeline with PCA-OLS model.
    
    Notes
    -----
    When to use PCA regression:
    - High VIF (> 10) indicating multicollinearity
    - Many correlated predictor variables
    - Want proper OLS inference (unlike Ridge)
    - Willing to sacrifice coefficient interpretability
    
    Advantages over Ridge/Lasso:
    - Full RegressionResults object (summary, p-values, intervals)
    - Components are guaranteed uncorrelated (VIF = 1.0)
    - Can visualize variance explained
    
    Disadvantages:
    - Coefficients are on components, not original features
    - Harder to interpret (use loadings_ and get_original_feature_importance())
    
    Examples
    --------
    >>> # Keep 95% variance (recommended starting point)
    >>> df_models = lr.train_models(
    ...     df_train,
    ...     pipeline_fn=lr.pipeline_pca_ols
    ... )
    >>> 
    >>> # Exact number of components
    >>> from functools import partial
    >>> df_models = lr.train_models(
    ...     df_train,
    ...     pipeline_fn=partial(lr.pipeline_pca_ols, n_components=5)
    ... )
    >>> 
    >>> # Access PCA diagnostics
    >>> model_obj = df_models.iloc[0]['model_obj'].named_steps['model']
    >>> print(model_obj.results_.summary())  # Full OLS summary
    >>> print(model_obj.variance_explained_)  # Variance per component
    >>> print(model_obj.loadings_)  # Feature contributions
    >>> print(model_obj.get_original_feature_importance())  # Approx feature importance
    >>> 
    >>> # Scree plot data
    >>> scree_df = model_obj.scree_data()
    """
    return Pipeline([
        ('model', StatsmodelsPCAOLS(
            n_components=n_components,
            standardize=standardize
        )),
    ])


def pipeline_pls(n_components=2, scale=True):
    """
    Partial Least Squares (PLS) regression pipeline.
    
    Extracts latent components that maximize covariance between X and y,
    then fits OLS on those components for proper inference.
    
    Parameters
    ----------
    n_components : int
        Number of PLS components to extract. Default 2.
        Should be <= min(n_samples, n_features).
    scale : bool
        Whether to scale X and y before PLS. Default True.
    
    Returns
    -------
    Pipeline
        Sklearn pipeline with PLS model.
    
    Notes
    -----
    PLS vs PCA:
    - PCA: Finds directions that maximize variance in X
    - PLS: Finds directions that maximize covariance between X and y
    
    PLS is often preferred when:
    - Predictors are highly collinear
    - Number of predictors exceeds observations
    - Want dimensionality reduction that considers y
    
    Examples
    --------
    >>> # Default 2 components
    >>> df_models = lr.train_models(
    ...     df_train,
    ...     pipeline_fn=lr.pipeline_pls
    ... )
    >>> 
    >>> # More components
    >>> from functools import partial
    >>> df_models = lr.train_models(
    ...     df_train,
    ...     pipeline_fn=partial(lr.pipeline_pls, n_components=3)
    ... )
    >>> 
    >>> # Access PLS diagnostics
    >>> model_obj = df_models.iloc[0]['model_obj'].named_steps['model']
    >>> print(model_obj.results_.summary())  # OLS summary on PLS scores
    >>> print(model_obj.get_loadings())  # Feature contributions
    >>> print(model_obj.get_original_coefficients())  # PLS coefficients
    """
    return Pipeline([
        ('model', StatsmodelsPLS(
            n_components=n_components,
            scale=scale
        )),
    ])


# =============================================================================
# TRAINING HELPERS
# =============================================================================

def _train_single_model(df, y_col, pipeline_fn, brand, channel, estimator=None,
                        base_estimator=None, params_str=None,
                        metric='r2', threshold=3, model_params=None):
    """
    Train a single model for one brand/channel/response combination.
    
    Internal helper used by train_models() and train_selected_models().
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data filtered to specific brand/channel.
    y_col : str
        Response variable (configured via RESPONSE_VAR in config.py).
    pipeline_fn : callable
        Factory function that returns a Pipeline.
    brand : str
        Brand name.
    channel : str
        Channel name.
    estimator : str, optional
        User-defined unique estimator name (e.g., 'RLM_Tukey').
    base_estimator : str, optional
        Estimator type code (e.g., 'OLS', 'WLS', 'RLM', 'PCA').
    params_str : str, optional
        JSON string of parameters used for this model.
    metric : str
        Scoring metric for cross-validation. Default 'r2'.
    threshold : int
        Z-score threshold for outlier removal. Default 3.
    model_params : dict, optional
        Additional parameters to pass to the pipeline factory function.
    
    Returns
    -------
    dict or None
        Model record with diagnostics and fitted pipeline, or None if no data.
    """
    if len(df) == 0:
        return None
    
    # Pre-pipeline processing
    df, n_obs, n_outliers = z_outliers(df.copy(), x='spend', y=y_col, threshold=threshold)
    
    # Create pipeline with optional params
    if model_params:
        pipe = pipeline_fn(**model_params)
    else:
        pipe = pipeline_fn()
    
    cols_to_drop = ['brand', 'model']
    if 'weekstart' in df.columns:
        cols_to_drop.append('weekstart')
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Fit model
    # Drop the target and any other response variables (e.g. 'visits' when fitting 'nd')
    other_responses = [v for v in RESPONSE_VARS if v != y_col]
    X = df.drop(columns=[y_col] + [c for c in other_responses if c in df.columns])
    y = df[y_col].to_frame()
    pipe.fit(X, y)

    # Get diagnostics
    results = pipe.named_steps['model'].results_
    model_step = pipe.named_steps['model']
    
    # Cross-validate
    CV_SPLITS = int(np.sqrt(results.nobs)) # adaptive k, sqrt approach
    cv = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=10) # orig
    scores = cross_val_score(pipe, X, y, scoring=metric, cv=cv)
    
    # Get residuals
    if hasattr(model_step, 'resid_') and model_step.resid_ is not None:
        resid = model_step.resid_
    else:
        try:
            resid = results.resid
        except AttributeError:
            resid = None
    
    shapiro = calc_shapiro_wilk(results, resid=resid)
    bp = calc_breusch_pagan(results, resid=resid)
    dw = calc_durbin_watson(results, resid=resid)
    cooks = calc_cooks_distance(results)
    
    # Compute in-sample score
    has_rsquared = hasattr(results, 'rsquared_adj')
    
    if metric == 'r2':
        if has_rsquared:
            score = results.rsquared_adj
        else:
            y_mean = y.values.mean()
            ss_tot = np.sum((y.values - y_mean) ** 2)
            ss_res = np.sum(resid ** 2) if resid is not None else np.nan
            score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    elif metric in ('neg_mean_squared_error', 'neg_root_mean_squared_error'):
        mse = np.mean(resid ** 2) if resid is not None else np.nan
        score = -np.sqrt(mse) if 'root' in metric else -mse
    else:
        if has_rsquared:
            score = results.rsquared_adj
        else:
            y_mean = y.values.mean()
            ss_tot = np.sum((y.values - y_mean) ** 2)
            ss_res = np.sum(resid ** 2) if resid is not None else np.nan
            score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Get estimated k if using WLS with auto-estimation
    estimated_k = getattr(model_step, 'estimated_k_', None)
    
    # Build model record
    model_record = {
        'brand': brand,
        'model': channel,
        'response': y_col,
        'estimator': estimator,        # User-defined unique name (e.g., 'RLM_Tukey')
        'base_estimator': base_estimator,   # Estimator type code (e.g., 'RLM', 'OLS', 'PCA')
        'params': params_str,       # JSON string of parameters used
        'R2': round(score, 4) if not np.isnan(score) else np.nan,
        'R2_CV': round(scores.mean(), 4),
        'n_obs': n_obs,
        'n_outliers': n_outliers,
        'n_features': pipe.named_steps['model'].n_features_in_,
        'wls_k': round(estimated_k, 3) if estimated_k is not None else None,
        'SW_stat': shapiro['statistic'],
        'SW_p': shapiro['pvalue'],
        'BP_stat': bp['statistic'],
        'BP_p': bp['pvalue'],
        'DW': dw['statistic'],
        'cooks_n': cooks['n_influential'],
        'VIF': calc_vif(X),
        'model_obj': pipe,
    }
    
    return model_record


def train_models(df_train, metric='r2', threshold=3, pipeline_fn=None):
    """
    Train Linear Regression models for each brand/channel/funnel combination.
    Uses Statsmodels OLS with standardization and RepeatedKFold cross-validation.

    Parameters
    ----------
    df_train : pd.DataFrame
    Training data with 'brand', 'model', 'nd_type', 'spend', and response variable.
    metric : str
        Scoring metric for cross-validation. Default 'r2'.
    threshold : int
        Z-score threshold for outlier removal. Default 3.
    pipeline_fn : callable, optional
        Factory function that returns a Pipeline. Use functools.partial for parameterization.
        Defaults to pipeline_ols_std(). Example: partial(pipeline_ols_std, std_cols=['spend', 'other'])

    Returns
    -------
    pd.DataFrame
        Model diagnostics and fitted Pipeline objects per brand/channel/funnel.
    """
    if pipeline_fn is None:
        pipeline_fn = pipeline_ols_std

    models = []

    breaks = list(
        df_train[['brand', 'model']]
        .drop_duplicates()
        .itertuples(index=False, name=None))

    for brand, model_key in breaks:
        df = df_train[
            (df_train['brand'] == brand) & 
            (df_train['model'] == model_key)
        ]
        
        record = _train_single_model(
            df=df,
            y_col=RESPONSE_VAR,
            pipeline_fn=pipeline_fn,
            brand=brand,
            channel=model_key,
            metric=metric,
            threshold=threshold
        )
        
        if record is not None:
            models.append(record)

    return pd.DataFrame(models)


def train_selected_models(df_train, df_selection, metric='r2', threshold=3, pipeline_map=None):
    """
    Train models for specific brand/model combinations from a selection file.
    
    This function trains only the combinations specified in df_selection, allowing
    multiple different estimators (OLS, WLS, RLM, PCA) per brand/model.
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training data with 'brand', 'model', 'spend', and response variable.
    df_selection : pd.DataFrame
        Model selections with columns:
        - 'brand': Brand name
        - 'model': Model key (channel | funnel | tactic)
        - 'estimator': Unique user-defined estimator name (e.g., 'RLM_Tukey'). Must be unique per run.
        - 'base_estimator': Estimator type code (e.g., 'OLS', 'WLS', 'RLM', 'PCA'). Maps to pipeline.
        - 'params' (optional): Dict-format string for pipeline parameters.
    metric : str
        Scoring metric for cross-validation. Default 'r2'.
    threshold : int
        Z-score threshold for outlier removal. Default 3.
    pipeline_map : dict, optional
        Maps base model names to pipeline factory functions.
        Defaults to module-level PIPELINE_MAP.
    
    Returns
    -------
    pd.DataFrame
        Model diagnostics and fitted Pipeline objects with columns:
        - 'model': User-defined model name (unique identifier)
        - 'base_estimator': Estimator type code (OLS, WLS, etc.)
        - 'params': String representation of parameters used
    
    Examples
    --------
    >>> df_selection = pd.read_csv('data/train/model_selections.csv')
    >>> df_models = lr.train_selected_models(df_train, df_selection)
    """
    import ast
    import json
    
    if pipeline_map is None:
        pipeline_map = PIPELINE_MAP
    
    if pipeline_map is None:
        raise ValueError("PIPELINE_MAP not initialized. Import lr_models after module fully loads.")
    
    models = []
    estimators_seen = set()  # Track (brand, model_key, estimator) tuples for uniqueness
    
    for _, row in df_selection.iterrows():
        brand = row['brand']
        model_key = row['model']
        
        estimator = row['estimator']
        base_estimator = row['base_estimator']
        
        # Validate estimator name uniqueness within brand/model_key
        selection_key = (brand, model_key, estimator)
        if selection_key in estimators_seen:
            print(f"WARNING: Duplicate estimator name '{estimator}' for {brand}/{model_key}. Estimator names must be unique within brand/model.")
        estimators_seen.add(selection_key)
        
        # Parse optional model parameters from 'params' column
        model_params = None
        params_str_display = None
        if 'params' in row and pd.notna(row['params']) and str(row['params']).strip():
            try:
                model_params = ast.literal_eval(str(row['params']))
                if not isinstance(model_params, dict):
                    print(f"WARNING: params for {brand}/{model_key} is not a dict, ignoring")
                    model_params = None
                else:
                    params_str_display = json.dumps(model_params, sort_keys=True)
            except (ValueError, SyntaxError) as e:
                print(f"WARNING: Could not parse params for {brand}/{model_key}: {e}")
                model_params = None
        
        # Get pipeline factory function using base_estimator
        if base_estimator not in pipeline_map:
            print(f"WARNING: Unknown base estimator '{base_estimator}' for {brand}/{model_key}, skipping")
            continue
        
        pipeline_fn = pipeline_map[base_estimator]
        params_info = f" with params {model_params}" if model_params else ""
        print(f"Training {estimator} ({base_estimator}) for {brand} / {model_key}{params_info}...")
        
        df = df_train[
            (df_train['brand'] == brand) & 
            (df_train['model'] == model_key)
        ]
        
        if len(df) == 0:
            print(f"  WARNING: No training data for {brand}/{model_key}/{RESPONSE_VAR}")
            continue
        
        record = _train_single_model(
            df=df,
            y_col=RESPONSE_VAR,
            pipeline_fn=pipeline_fn,
            brand=brand,
            channel=model_key,
            estimator=estimator,
            base_estimator=base_estimator,
            params_str=params_str_display,
            metric=metric,
            threshold=threshold,
            model_params=model_params
        )
        
        if record is not None:
            models.append(record)
    
    return pd.DataFrame(models)


def pred_lr(df_models, df_preds, bias=None):
    """
    Generate predictions using trained LR models for each brand/model combination.

    Parameters
    ----------
    df_models : pd.DataFrame
        Output from train_models() containing fitted Pipeline objects.
    df_preds : pd.DataFrame
        Prediction data with 'brand', 'model', 'spend', and features.
    bias : float or dict, optional
        Bias correction as a percentage. If float, applies same adjustment to all.
        If dict, keys should be (brand, channel, estimator) tuples, or
        (brand, channel) tuples to apply the same value to every estimator.
        A bias of -10 means the model under-predicts by 10%, so predictions
        are adjusted UP by 10% (multiplied by 1.10).

    Returns
    -------
    pd.DataFrame
        Predictions with original values (mean, ci_lo, ci_hi), bias column showing
        the bias % applied, and when bias is provided, adjusted values 
        (mean_adj, ci_lo_adj, ci_hi_adj).
        When multiple model variants exist, includes 'model' column to distinguish.
    """

    df_preds_final = pd.DataFrame()

    # get the models we'll want to use
    breaks = list(
        df_preds[['brand', 'model']]
        .drop_duplicates()
        .itertuples(index=False, name=None))

    for brand, model_key in breaks:

        # get preds
        df = df_preds[
            (df_preds['brand'] == brand) & 
            (df_preds['model'] == model_key)
            ].copy()
        
        # get all matching models (may have multiple variants like OLS, RLM, PCA)
        model_matches = df_models[
            (df_models['brand'] == brand) & 
            (df_models['model'] == model_key) & 
            (df_models['response'] == RESPONSE_VAR)
            ]
        
        if len(model_matches) == 0:
            print(f"WARNING: No model found for {brand}/{model_key}/{RESPONSE_VAR}, skipping")
            continue
        
        # Iterate over all model variants
        for model_idx, model_row in model_matches.iterrows():
            pipe = model_row['model_obj']
            estimator = model_row.get('estimator', 'unknown')
            base_estimator = model_row.get('base_estimator', None)
            params = model_row.get('params', None)
            
            # print((brand, model_key, RESPONSE_VAR, estimator), type(pipe)) # QA

            # Determine bias value for this brand/channel/estimator
            if bias is not None:
                if isinstance(bias, dict):
                    bias_pct = bias.get((brand, model_key, estimator),
                                        bias.get((brand, model_key), 0))
                else:
                    bias_pct = bias
            else:
                bias_pct = 0
            
            # table to append preds - reset index for proper alignment
            df_results = df[['brand', 'model', 'weekstart', 'spend']].reset_index(drop=True).copy()
            df_results['response'] = RESPONSE_VAR
            df_results['estimator'] = estimator
            df_results['base_estimator'] = base_estimator
            df_results['params'] = params
            df_results['bias'] = bias_pct

            # drop non-model columns and all response variables for prediction
            _meta_cols = ['brand', 'model', 'weekstart'] + RESPONSE_VARS
            df_features = df.drop(columns=[c for c in _meta_cols if c in df.columns]).reset_index(drop=True)

            # mean prediction with intervals from statsmodels
            # Handle pipelines with or without preprocessing steps
            if len(pipe.steps) > 1:
                # Pipeline has preprocessing steps (e.g., pipeline_ols_std)
                df_trans = pipe[:-1].transform(df_features)
            else:
                # Pipeline has only the model (e.g., pipeline_wls)
                df_trans = df_features
            df_ints = pipe.named_steps['model'].get_intervals(df_trans)
            
            # Reset index on intervals to ensure alignment
            df_ints = df_ints.reset_index(drop=True)
            
            df_results = pd.concat([df_results, df_ints], axis=1)

            # Apply bias correction if provided (keep originals, add _adj columns)
            if bias_pct != 0:
                # Apply correction: if bias (MPE) = -10%, predictions are 90% of actual
                # so we divide by 0.90 to correct upward
                # Formula: corrected = pred / (1 + bias/100)
                # Note: get_intervals returns mean_ci_lower/mean_ci_upper, renamed after
                adj_cols = [('mean', 'mean_adj'), 
                           ('mean_ci_lower', 'ci_lo_adj'), 
                           ('mean_ci_upper', 'ci_hi_adj')]
                correction_factor = 1 / (1 + bias_pct / 100)
                for src_col, adj_col in adj_cols:
                    if src_col in df_results.columns:
                        df_results[adj_col] = df_results[src_col] * correction_factor
            elif bias is not None:
                # Bias provided but zero - adjusted equals original
                adj_cols = [('mean', 'mean_adj'), 
                           ('mean_ci_lower', 'ci_lo_adj'), 
                           ('mean_ci_upper', 'ci_hi_adj')]
                for src_col, adj_col in adj_cols:
                    if src_col in df_results.columns:
                        df_results[adj_col] = df_results[src_col]

            # append
            df_preds_final = pd.concat([df_preds_final, df_results])

    return df_preds_final.reset_index(drop=True)



def analyze_lr(results):
    """ 
    Regression table and residual analysis for a results_ attribute for a fitted statsmodels object. Required scipy.stats, matplotlib.pyplot and seaborn.

    Usage: analyze_lr(model.results_)
    """

    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(results.summary())
    fig, axes = plt.subplots(2, 2, figsize=(15,8))
    
    # fitted vs residuals
    sns.scatterplot(x=results.fittedvalues, y=results.resid, ax=axes[0,0])
    axes[0,0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0,0].set_title('Residuals vs Fitted')
    
    # residuals qqplot
    stats.probplot(results.resid, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot')
    
    # residuals histogram
    sns.histplot(results.resid, kde=True, ax=axes[1,0])
    axes[1,0].set_title('Residuals Distribution')

    # cooks distance (may not be available for WLS)
    try:
        influence = results.get_influence()
        cooks_d = influence.cooks_distance[0]
        threshold = 4 / len(cooks_d)
        axes[1,1].stem(range(len(cooks_d)), cooks_d, markerfmt=',')
        axes[1,1].axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
        axes[1,1].set_title("Cook's Distance")

        influential_mask = cooks_d > threshold
        influential_indices = np.where(influential_mask)[0]
        for idx in influential_indices:
            axes[1,1].annotate(f'{idx}', 
                        xy=(idx, cooks_d[idx]), 
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=9,
                        alpha=0.8)
    except (AttributeError, ValueError, np.linalg.LinAlgError):
        axes[1,1].text(0.5, 0.5, "Cook's Distance\nnot available\nfor this model type", 
                      ha='center', va='center', transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].set_title("Cook's Distance (N/A)")

    plt.tight_layout()
    plt.show()


def inspect_models(df_models):
    columns = ('brand', 'model', 'response') 
    for row in df_models.itertuples():
        print(row[1:4])
        analyze_lr(row[9].named_steps['model'].results_)
        user_input = input("\nPress Enter for next (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break


def estimate_variance_structure(X, y, weight_col='spend', add_constant=True):
    """
    Estimate k in Var(ε) = σ² * weight_col^k using log-log regression of OLS residuals.
    
    Standalone utility function for exploratory analysis. The StatsmodelsWLS class
    can auto-estimate this during fit when weight_power=None.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix including the weight column.
    y : array-like
        Response variable.
    weight_col : str
        Column to use as variance proxy. Default 'spend'.
    add_constant : bool
        Whether OLS includes an intercept. Default True.
    
    Returns
    -------
    dict
        {
            'k_estimate': estimated power,
            'k_stderr': standard error of estimate,
            'k_pvalue': p-value (is k significantly != 0?),
            'var_r2': R² of variance model,
            'suggested_weights': human-readable suggestion
        }
    """
    X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    weight_vals = X_df[weight_col].values
    n = len(y)
    
    if n < 10:
        return {
            'k_estimate': np.nan, 'k_stderr': np.nan, 'k_pvalue': np.nan,
            'var_r2': np.nan, 'suggested_weights': 'Insufficient data'
        }
    
    # Stage 1: Fit OLS to get residuals
    X_ols = sm.add_constant(X_df) if add_constant else X_df
    ols_fit = sm.OLS(y, X_ols).fit()
    residuals = ols_fit.resid
    
    # Stage 2: Regress log(residuals²) on log(weight_col)
    mask = weight_vals > 0
    log_resid_sq = np.log(residuals[mask] ** 2)
    log_weight = np.log(weight_vals[mask])
    
    X_var = sm.add_constant(log_weight)
    var_fit = sm.OLS(log_resid_sq, X_var).fit()
    
    k = var_fit.params[1]
    k_se = var_fit.bse[1]
    k_pval = var_fit.pvalues[1]
    var_r2 = var_fit.rsquared
    
    # Suggest weight structure
    if k_pval > 0.1:
        suggestion = 'OLS (no significant heteroskedasticity)'
    elif abs(k - 1) < 0.5:
        suggestion = 'WLS: 1/spend'
    elif abs(k - 2) < 0.5:
        suggestion = 'WLS: 1/spend²'
    else:
        suggestion = f'WLS: 1/spend^{k:.1f}'
    
    return {
        'k_estimate': k,
        'k_stderr': k_se,
        'k_pvalue': k_pval,
        'var_r2': var_r2,
        'suggested_weights': suggestion
    }


def explore_pca(X, standardize=True, plot=True):
    """
    Exploratory PCA analysis to determine optimal number of components.
    
    Run this before fitting PCA regression to understand variance structure
    and choose n_components.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (without response variable).
    standardize : bool
        Whether to standardize features before PCA. Default True.
    plot : bool
        Whether to display scree plot. Default True.
    
    Returns
    -------
    dict
        {
            'n_components_95': int - components for 95% variance,
            'n_components_90': int - components for 90% variance,
            'variance_explained': array - variance per component,
            'cumulative_variance': array - cumulative variance,
            'loadings': DataFrame - feature × component loadings,
            'scree_data': DataFrame - data for custom plotting
        }
    
    Examples
    --------
    >>> # Explore promo variables
    >>> promo_cols = [c for c in df_train.columns if c.startswith('promo_')]
    >>> pca_info = lr.explore_pca(df_train[promo_cols])
    >>> print(f"Need {pca_info['n_components_95']} components for 95% variance")
    """
    X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
    feature_names = X_df.columns.tolist()
    
    # Standardize
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
    else:
        X_scaled = X_df.values
    
    # Fit PCA with all components
    pca = PCA()
    pca.fit(X_scaled)
    
    # Calculate statistics
    var_explained = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(var_explained)
    
    # Find components for variance thresholds
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    n_90 = np.argmax(cumulative_var >= 0.90) + 1
    
    # Loadings matrix
    component_names = [f'PC{i+1}' for i in range(len(var_explained))]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=component_names
    )
    
    # Scree data
    scree_df = pd.DataFrame({
        'component': component_names,
        'component_num': range(1, len(var_explained) + 1),
        'variance_explained': var_explained,
        'cumulative_variance': cumulative_var
    })
    
    # Plot if requested
    if plot:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Scree plot
        ax1 = axes[0]
        ax1.bar(range(1, len(var_explained) + 1), var_explained, alpha=0.7, label='Individual')
        ax1.plot(range(1, len(var_explained) + 1), cumulative_var, 'ro-', label='Cumulative')
        ax1.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95% threshold')
        ax1.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained')
        ax1.set_title('Scree Plot')
        ax1.legend(loc='center right')
        
        # Top loadings for first 3 components
        ax2 = axes[1]
        top_n = min(10, len(feature_names))
        for i, pc in enumerate(['PC1', 'PC2', 'PC3'][:min(3, len(component_names))]):
            top_features = loadings[pc].abs().nlargest(top_n)
            ax2.barh([f"{feat} ({pc})" for feat in top_features.index], 
                    loadings.loc[top_features.index, pc].values,
                    alpha=0.7, label=pc)
        ax2.set_xlabel('Loading')
        ax2.set_title(f'Top {top_n} Feature Loadings (First 3 PCs)')
        ax2.axvline(x=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nSummary:")
        print(f"  Total features: {len(feature_names)}")
        print(f"  Components for 90% variance: {n_90}")
        print(f"  Components for 95% variance: {n_95}")
        print(f"  First 3 PCs explain: {cumulative_var[2]*100:.1f}% variance")
    
    return {
        'n_components_95': n_95,
        'n_components_90': n_90,
        'variance_explained': var_explained,
        'cumulative_variance': cumulative_var,
        'loadings': loadings,
        'scree_data': scree_df
    }


def z_outliers(df, x='spend', y='nd', threshold=3):
    """ 
    Calculate ratio (spend/response) z-scores and drop outliers over SD threshold. 
    Run this outside the pipeline on the training data only.
    """
    df = df.copy()
    n_obs = len(df)
    df['ratio'] = df[y] / df[x]
    df['ratio'] = df['ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    z_scores = ((df['ratio'] - df['ratio'].mean()) / df['ratio'].std()).fillna(0)
    df = df[z_scores.abs() <= threshold]
    df = df.drop(columns='ratio')
    n_outliers = n_obs - len(df)
    return df, n_obs, n_outliers


def calc_shapiro_wilk(results, max_sample=5000, resid=None):
    """
    Calculate Shapiro-Wilk test for normality of residuals.
    
    Parameters
    ----------
    results : statsmodels RegressionResults
        Fitted statsmodels OLS results object.
    max_sample : int
        Maximum sample size for test (samples if n > max_sample).
    resid : array-like, optional
        Residuals array. If None, uses results.resid.
    
    Returns
    -------
    dict
        {'statistic': float, 'pvalue': float}
    """
    if resid is None:
        try:
            resid = results.resid
        except AttributeError:
            resid = None
    if resid is None:
        return {'statistic': np.nan, 'pvalue': np.nan}
    sample_resid = resid[:max_sample] if len(resid) > max_sample else resid
    stat, pvalue = stats.shapiro(sample_resid)
    return {'statistic': stat, 'pvalue': pvalue}


def calc_breusch_pagan(results, resid=None):
    """
    Calculate Breusch-Pagan test for heteroscedasticity.
    
    Parameters
    ----------
    results : statsmodels RegressionResults
        Fitted statsmodels OLS or WLS results object.
    resid : array-like, optional
        Residuals array. If None, uses results.resid.
    
    Returns
    -------
    dict
        {'statistic': float, 'pvalue': float}
    
    Notes
    -----
    For WLS models, this tests heteroscedasticity in the weighted residuals.
    If the WLS weights are correct, this should show no heteroscedasticity.
    """
    try:
        if resid is None:
            try:
                resid = results.resid
            except AttributeError:
                resid = None
        if resid is None:
            return {'statistic': np.nan, 'pvalue': np.nan}
        # Check for model.exog safely
        try:
            exog = results.model.exog
        except AttributeError:
            return {'statistic': np.nan, 'pvalue': np.nan}
        stat, pvalue, _, _ = het_breuschpagan(resid, exog)
    except Exception:
        stat, pvalue = np.nan, np.nan
    return {'statistic': stat, 'pvalue': pvalue}


def calc_durbin_watson(results, resid=None):
    """
    Calculate Durbin-Watson test for autocorrelation.
    
    Parameters
    ----------
    results : statsmodels RegressionResults
        Fitted statsmodels OLS results object.
    resid : array-like, optional
        Residuals array. If None, uses results.resid.
    
    Returns
    -------
    dict
        {'statistic': float}
    """
    if resid is None:
        try:
            resid = results.resid
        except AttributeError:
            resid = None
    if resid is None:
        return {'statistic': np.nan}
    stat = durbin_watson(resid)
    return {'statistic': stat}


def calc_cooks_distance(results, threshold_factor=4):
    """
    Calculate Cook's distance for each observation.
    
    Parameters
    ----------
    results : statsmodels RegressionResults
        Fitted statsmodels OLS or WLS results object.
    threshold_factor : int
        Factor for threshold calculation (threshold = factor / n).
    
    Returns
    -------
    dict
        {'cooks_d': np.array, 'threshold': float, 'n_influential': int, 'influential_indices': np.array}
        Returns NaN values if influence diagnostics are unavailable (e.g., some WLS cases).
    """
    try:
        influence = results.get_influence()
        cooks_d = influence.cooks_distance[0]
        threshold = threshold_factor / len(cooks_d)
        influential_indices = np.where(cooks_d > threshold)[0]
        return {
            'cooks_d': cooks_d,
            'threshold': threshold,
            'n_influential': len(influential_indices),
            'influential_indices': influential_indices
        }
    except (AttributeError, ValueError, np.linalg.LinAlgError) as e:
        # WLS or RegularizedOLS or other models may not support get_influence()
        try:
            n = len(results.resid)
        except AttributeError:
            # RegularizedResults doesn't have resid
            try:
                n = int(results.model.nobs)
            except (AttributeError, TypeError):
                n = 1
        return {
            'cooks_d': np.full(n, np.nan),
            'threshold': threshold_factor / n if n > 0 else np.nan,
            'n_influential': np.nan,
            'influential_indices': np.array([])
        }


def calc_vif(X):
    """
    Calculate maximum Variance Inflation Factor (VIF) across features.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (without constant).
    
    Returns
    -------
    float
        Maximum VIF value across all features. Returns np.nan if calculation fails.
    """
    try:
        X_with_const = sm.add_constant(X)
        vif_values = [variance_inflation_factor(X_with_const.values, i) 
                      for i in range(1, X_with_const.shape[1])]  # skip constant
        return round(max(vif_values), 2)
    except Exception:
        return np.nan


# ==============================================================================
# Module-level Constants (initialized after all functions are defined)
# ==============================================================================

PIPELINE_MAP = {
    'OLS': pipeline_ols_std,
    'WLS': pipeline_wls,
    'RLM': pipeline_rlm,
    'PCA': pipeline_pca_ols,
    'PLS': pipeline_pls
}