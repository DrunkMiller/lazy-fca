import pandas as pd
import preparing_data
import classifiers
from sklearn.model_selection import train_test_split
from geneticalgorithm import geneticalgorithm as ga


def preparing_data_before_testing(features, data):
    data_tmp = data.copy()
    preparing_data.binarize_target_feature(data_tmp, False)
    features_to_delete = list(preparing_data.features_binarize_dict.keys())
    for feature in features:
        features_to_delete.remove(feature)
        preparing_data.features_binarize_dict[feature](data_tmp, False)
    for feature in features_to_delete:
        data_tmp.drop(feature, axis='columns', inplace=True)
    return data_tmp

def test_features(features, data, classification_alg):
    data_to_testing = preparing_data_before_testing(features, data)
    train, test = train_test_split(data_to_testing, test_size=0.2, random_state=45, shuffle=True)
    plus, minus, x_test, y_test = classifiers.prepare_data(test, train)
    y_pred = classification_alg(plus, minus, x_test)
    return classifiers.calc_accuracy(y_test, y_pred)

def fitness(X):
    features = []
    for i in range(len(X)):
        if X[i]:
            features.append(features_list[i])
    return 1 / test_features(features, data, classifiers.simple_classifer)

pd.options.mode.chained_assignment = None
data = pd.read_csv('D:\credit_min.csv', delimiter=';')
features_list = list(data)
features_list.remove('default')
print(features_list)

if __name__ == "__main__":
    algorithm_param = {'max_num_iteration': 100,
                       'population_size': 50,
                       'mutation_probability': 0.2,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': 100}
    model = ga(function=fitness, dimension=len(features_list), variable_type='bool', algorithm_parameters=algorithm_param)
    model.run()