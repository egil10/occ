from sklearn.ensemble import RandomForestClassifier

SEED = 808

def get_model(n_estimators=100):
    """Returns a configured Random Forest model."""
    return RandomForestClassifier(random_state=SEED, n_estimators=n_estimators)

if __name__ == "__main__":
    model = get_model()
    print(f"Initialized {model}")
