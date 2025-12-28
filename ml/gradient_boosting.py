from sklearn.ensemble import GradientBoostingClassifier

SEED = 808

def get_model():
    """Returns a standard Gradient Boosting Classifier."""
    return GradientBoostingClassifier(random_state=SEED)

if __name__ == "__main__":
    model = get_model()
    print(f"Initialized {model}")
