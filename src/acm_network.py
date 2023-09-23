import mnist_loader
import network
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)


def train_and_save_model(path_model, sizes=[784, 30, 10]):
    """
    Función que entrena el modelo (los parámetros escogidos son los
    indicados por el autor en el capítulo 1)
    """
    net = network.Network(sizes)
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    net.save_model(path_model)


def evaluate_model(path_model, test_data):
    """
    Función que retorna el % de aciertos en el test_data set
    """
    net = network.Network.load_model(path_model)
    n = len(test_data)
    evaluation_array = np.zeros(n)

    for i in range(n):
        output = np.argmax(net.feedforward(test_data[i][0]))
        if output == test_data[i][1]:
            evaluation_array[i] = 1

    metric = np.mean(evaluation_array)

    print(metric)
    return metric





# train_and_save_model('./src/models/20230924_784_30_10_model.pkl')
# train_and_save_model('./src/models/20230923_784_10_model.pkl', sizes=[784, 10])
# evaluate_model('./src/models/20230922_784_30_10_model.pkl', test_data)
# evaluate_model('./src/models/20230923_784_30_10_model.pkl', test_data)
evaluate_model('./src/models/20230923_784_10_model.pkl', test_data)

