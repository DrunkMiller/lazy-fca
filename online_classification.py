
class OnlineClassifier(object):

    def __init__(self, plus_data, minus_data, algorithm):
        self.plus_data = plus_data
        self.minus_data = minus_data
        self.algorithm = algorithm

    def predict(self, x_pred):
        y_pred = []
        for x in x_pred:
            y = self.algorithm(self.plus_data, self.minus_data, [x])
            y_pred.append(y[0])
            if y[0] == 1:
                self.plus_data.append(x)
            else:
                self.minus_data.append(x)
        return y_pred

