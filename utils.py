"""
Utility functions for digit classification and visualization.
"""

import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


def load_digits_dataset():
    """Load the digits dataset from scikit-learn."""
    return datasets.load_digits()


def visualize_training_samples(digits):
    """Visualize the first 4 training samples from the digits dataset."""
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    return axes


def prepare_data(digits):
    """Flatten the images and prepare data for classification."""
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data


def create_classifier():
    """Create and return a support vector classifier."""
    return svm.SVC(gamma=0.001)


def split_data(data, target, test_size=0.5):
    """Split data into train and test subsets."""
    return train_test_split(data, target, test_size=test_size, shuffle=False)


def train_classifier(clf, X_train, y_train):
    """Train the classifier on the training data."""
    clf.fit(X_train, y_train)
    return clf


def predict_digits(clf, X_test):
    """Predict digit values for test samples."""
    return clf.predict(X_test)


def visualize_predictions(X_test, predicted):
    """Visualize the first 4 test samples with their predicted values."""
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    return axes


def print_classification_report(clf, y_test, predicted):
    """Print the classification report."""
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )


def plot_confusion_matrix(y_test, predicted):
    """Plot and display the confusion matrix."""
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    return disp


def rebuild_classification_report_from_cm(disp):
    """Rebuild classification report from confusion matrix."""
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )


def run_digit_classification_pipeline():
    """Run the complete digit classification pipeline."""
    # Load dataset
    digits = load_digits_dataset()
    
    # Visualize training samples
    visualize_training_samples(digits)
    
    # Prepare data
    data = prepare_data(digits)
    
    # Create and train classifier
    clf = create_classifier()
    X_train, X_test, y_train, y_test = split_data(data, digits.target)
    train_classifier(clf, X_train, y_train)
    
    # Make predictions
    predicted = predict_digits(clf, X_test)
    
    # Visualize predictions
    visualize_predictions(X_test, predicted)
    
    # Print classification report
    print_classification_report(clf, y_test, predicted)
    
    # Plot confusion matrix
    disp = plot_confusion_matrix(y_test, predicted)
    
    # Show all plots
    plt.show()
    
    # Rebuild classification report from confusion matrix
    rebuild_classification_report_from_cm(disp)
    
    return clf, X_test, y_test, predicted 