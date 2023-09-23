import joblib
import numpy as np
import mnist_loader


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)


def evaluate_model(path_model, test_data):
    # Cargar el modelo
    with open(path_model, 'rb') as model_file:
        clf = joblib.load(model_file)

    # Adapta el input
    n = len(test_data)

    test_data_adapted = []
    for i in range(n):
        image = test_data[i][0].reshape(-1)
        test_data_adapted.append(np.array([image]))

    predictions = []
    for i in range(n):
        predictions.append(int(clf.predict(test_data_adapted[i])[0]))

    evaluation_array = np.zeros(n)

    for i in range(n):
        if predictions[i] == test_data[i][1]:
            evaluation_array[i] = 1

    metric = np.mean(evaluation_array)

    print(metric)
    return metric


evaluate_model('./src/models/svm/svm_model.joblib', test_data)
