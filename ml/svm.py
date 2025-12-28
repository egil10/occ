from sklearn.svm import LinearSVC

SEED = 808

def get_model():
    """Returns a configured Linear SVM model."""
    # dual="auto" handles the choice between primal/dual optimization
    return LinearSVC(random_state=SEED, dual="auto")

if __name__ == "__main__":
    model = get_model()
    print(f"Initialized {model}")
