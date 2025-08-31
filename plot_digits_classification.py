"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Import utility functions
from utils import run_digit_classification_pipeline

if __name__ == "__main__":
    # Run the complete digit classification pipeline
    clf, X_test, y_test, predicted = run_digit_classification_pipeline()
