import warnings

SEED = 808

def get_model():
    """
    Returns XGBClassifier if available.
    Raises ImportError or warns if not installed (depending on usage).
    """
    try:
        import xgboost as xgb
        return xgb.XGBClassifier(
            random_state=SEED,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    except ImportError:
        print("XGBoost library not found. Please install with `pip install xgboost`.")
        warnings.warn("XGBoost not installed. Returning None or falling back logic required.")
        return None

if __name__ == "__main__":
    model = get_model()
    if model:
        print(f"Initialized {model}")
