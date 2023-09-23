"""
mnist_svm
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

# Libraries
# My libraries
import mnist_loader
import joblib

# Third-party libraries
from sklearn import svm


def svm_baseline(path_to_model='./src/models/svm/svm_model.joblib'):
    training_data, validation_data, test_data = mnist_loader.load_data()
    # train

    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])

    # save the model
    with open(path_to_model, 'wb') as model_file:
        joblib.dump(clf, model_file)

    # test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print("Baseline classifier using an SVM.")
    print(f"{num_correct} of {len(test_data[1])} values correct.")


if __name__ == "__main__":
    svm_baseline()
