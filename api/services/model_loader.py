"""
Model loading and caching service.

Loads trained sklearn pipelines from .joblib files in data/output/ and provides
lookup by brand/channel combination.
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd
from joblib import load

# Default to <repo>/api/data/models, resolved relative to this file so the
# path works regardless of CWD (fastapi dev from repo root, uvicorn from
# api/, Docker WORKDIR, etc.). Override with MODELS_DIR env var.
_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


class ModelLoader:
    """Loads and caches trained models from joblib files."""

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(
            models_dir or os.environ.get("MODELS_DIR") or _DEFAULT_MODELS_DIR
        )
        self._models_df: Optional[pd.DataFrame] = None
        self._loaded_files: list[str] = []
    
    def load_all_models(self) -> int:
        """
        Load all models_*.joblib files from the models directory.
        
        Returns the number of brand/channel combinations loaded.
        """
        dfs = []
        self._loaded_files = []
        
        for joblib_file in self.models_dir.glob("models_*.joblib"):
            try:
                df = load(joblib_file)
                if isinstance(df, pd.DataFrame) and 'model_obj' in df.columns:
                    dfs.append(df)
                    self._loaded_files.append(joblib_file.name)
                    print(f"Loaded {len(df)} models from {joblib_file.name}")
            except Exception as e:
                print(f"Warning: Failed to load {joblib_file}: {e}")
        
        if dfs:
            self._models_df = pd.concat(dfs, ignore_index=True)
            # Deduplicate: keep first occurrence of each brand/model/estimator combo
            self._models_df = self._models_df.drop_duplicates(
                subset=['brand', 'model', 'estimator'], keep='first'
            )
            return len(self._models_df)
        
        self._models_df = pd.DataFrame()
        return 0
    
    def get_model(self, brand: str, model: str) -> Optional[pd.Series]:
        """
        Get the model for a specific brand/model combination.
        
        Returns the first matching model row, or None if not found.
        """
        if self._models_df is None or self._models_df.empty:
            return None
        
        matches = self._models_df[
            (self._models_df['brand'] == brand) & 
            (self._models_df['model'] == model)
        ]
        
        if matches.empty:
            return None
        
        # Return first match (could extend to select by model name)
        return matches.iloc[0]
    
    def list_models(self) -> list[dict]:
        """List all available brand/model/estimator combinations."""
        if self._models_df is None or self._models_df.empty:
            return []

        return [
            {
                "brand": str(row.get('brand') or ''),
                "model": str(row.get('model') or ''),
                "estimator": str(row.get('estimator') or ''),
                "base_estimator": str(row.get('base_estimator') or ''),
                "params": str(row.get('params') or ''),
                "score_r2": float(row.get('R2_CV') or 0.0),
            }
            for row in self._models_df.to_dict('records')
        ]
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._models_df is not None and not self._models_df.empty
    
    @property
    def model_count(self) -> int:
        """Get number of loaded models."""
        if self._models_df is None:
            return 0
        return len(self._models_df)
    
    @property
    def loaded_files(self) -> list[str]:
        """Get list of loaded joblib files."""
        return self._loaded_files


# Global singleton instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader
