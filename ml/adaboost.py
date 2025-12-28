from sklearn.ensemble import AdaBoostClassifier

SEED = 808

def get_model():
    """Returns an AdaBoost Classifier."""
    return AdaBoostClassifier(random_state=SEED)

if __name__ == "__main__":
    model = get_model()
    print(f"Initialized {model}")
