from sklearn.neighbors import KNeighborsClassifier

def get_model(n_neighbors=7):
    """Returns a configured KNN model."""
    return KNeighborsClassifier(n_neighbors=n_neighbors)

if __name__ == "__main__":
    model = get_model()
    print(f"Initialized {model}")
