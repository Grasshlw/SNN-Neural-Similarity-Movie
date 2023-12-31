import numpy as np


class SimilarityMetric:
    def __init__(self, seed=2023):
        self.seed = seed
    
    def _pearson_correlation_coefficient(self, x, y, mode="normal"):
        if mode == "cross":
            x_mean = np.mean(x, axis=1, keepdims=True)
            y_mean = np.mean(y, axis=1, keepdims=True)

            x_center = x - x_mean
            y_center = y - y_mean

            x_diag = np.diagonal(np.dot(x_center, x_center.T)).reshape((-1, 1))
            y_diag = np.diagonal(np.dot(y_center, y_center.T)).reshape((1, -1))
            r = np.dot(x_center, y_center.T) / np.sqrt(np.tile(x_diag, (1, x.shape[0])) * np.tile(y_diag, (y.shape[0], 1)))
        elif mode == "parallel":
            x_mean = np.mean(x, axis=0, keepdims=True)
            y_mean = np.mean(y, axis=0, keepdims=True)

            x_center = x - x_mean
            y_center = y - y_mean
            
            r = np.sum(x_center * y_center, axis=0) / np.sqrt(np.sum(x_center * x_center, axis=0) * np.sum(y_center * y_center, axis=0))
        elif mode == "normal":
            x_mean = np.mean(x)
            y_mean = np.mean(y)

            x_center = x - x_mean
            y_center = y - y_mean
        
            r = np.sum(x_center * y_center) / np.sqrt(np.sum(x_center * x_center) * np.sum(y_center * y_center))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return r

    def _spearman_correlation_coefficient(self, x, y):
        x_rank = np.argsort(np.argsort(x)).astype("float64")
        y_rank = np.argsort(np.argsort(y)).astype("float64")
        n = x.shape[0]
        r = 1 - 6 * np.sum((x_rank - y_rank) ** 2) / (n ** 3 - n)

        return r

    def score(self, model_data, neural_data):
        pass


class TSRSAMetric(SimilarityMetric): 
    def score(self, model_data, neural_data):
        num_classes = model_data.shape[0]

        model_RDM = 1 - self._pearson_correlation_coefficient(model_data, model_data, mode="cross")
        neural_RDM = 1 - self._pearson_correlation_coefficient(neural_data, neural_data, mode="cross")

        model_RDM = model_RDM[np.triu_indices(num_classes, 1)]
        neural_RDM = neural_RDM[np.triu_indices(num_classes, 1)]

        return self._spearman_correlation_coefficient(model_RDM, neural_RDM)
