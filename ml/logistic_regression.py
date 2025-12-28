from sklearn.linear_model import LogisticRegression

SEED = 808

def get_model():
    """Returns a configured Logistic Regression model."""
    return LogisticRegression(random_state=SEED, max_iter=1000)

if __name__ == "__main__":
    model = get_model()
    print(f"Initialized {model}")
