'''Simple, transparent baselines for FPL point prediction.'''

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class Tier0FormModel(BaseEstimator, RegressorMixin):
    '''Predict points directly from recent form and fixture difficulty.

    The model learns no parameters. Its rule is intentionally transparent:

    - FDR 1 or 2 (easy): ``form + adjustment``
    - FDR 3 (neutral): ``form``
    - FDR 4 or 5 (hard): ``form - adjustment``

    ``form_last_5`` is the default because it is shifted and therefore safe
    for historical evaluation. Pass ``form_col='form'`` only when that field
    is known to represent information available before the predicted match.
    '''

    def __init__(self, form_col: str = 'form_last_5',
                 difficulty_col: str = 'opponent_difficulty',
                 adjustment: float = 1.0,
                 missing_form_value: float = 0.0,
                 minimum_prediction: Optional[float] = None):
        self.form_col = form_col
        self.difficulty_col = difficulty_col
        self.adjustment = adjustment
        self.missing_form_value = missing_form_value
        self.minimum_prediction = minimum_prediction

    def fit(self, X: pd.DataFrame, y=None):
        '''Validate columns; no parameters are estimated.'''
        self._validate_input(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        '''Apply the fixed Tier-0 rule and return one prediction per row.'''
        self._validate_input(X)
        form = pd.to_numeric(X[self.form_col], errors='coerce').fillna(
            self.missing_form_value)
        difficulty = pd.to_numeric(
            X[self.difficulty_col], errors='coerce')

        fixture_adjustment = np.select(
            [difficulty <= 2, difficulty >= 4],
            [self.adjustment, -self.adjustment],
            default=0.0,
        )
        predictions = form.to_numpy(dtype=float) + fixture_adjustment
        if self.minimum_prediction is not None:
            predictions = np.maximum(predictions, self.minimum_prediction)
        return predictions

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Tier0FormModel expects a pandas DataFrame')
        missing = [column for column in [
            self.form_col, self.difficulty_col,
        ] if column not in X.columns]
        if missing:
            raise ValueError(f'Missing Tier-0 columns: {missing}')
