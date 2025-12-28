from sklearn.tree import DecisionTreeClassifier

SEED = 808

def get_model(max_depth=10):
    """Returns a configured Decision Tree model."""
    return DecisionTreeClassifier(random_state=SEED, max_depth=max_depth)

if __name__ == "__main__":
    model = get_model()
    print(f"Initialized {model}")
