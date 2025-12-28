from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 808

def get_model():
    """
    Returns Scikit-Learn's Histogram Gradient Boosting Classifier.
    This is an efficient implementation inspired by LightGBM.
    """
    return HistGradientBoostingClassifier(random_state=SEED)

if __name__ == "__main__":
    model = get_model()
    print(f"Initialized {model}")
