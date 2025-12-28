from sklearn.neural_network import MLPClassifier

SEED = 808

def get_model():
    """
    Returns a Multi-layer Perceptron (Neural Network) Classifier.
    """
    return MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=SEED,
        early_stopping=True
    )

if __name__ == "__main__":
    model = get_model()
    print(f"Initialized {model}")
