import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA


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

    def _coefficient_of_determination(self, x, x_pred):
        if x.ndim == 1:
            return 1 - np.sum((x - x_pred) ** 2) / np.sum((x - np.mean(x)) ** 2)
        elif x.ndim == 2:
            return np.mean(1 - np.sum((x - x_pred) ** 2, axis=0) / np.sum((x - np.mean(x, axis=0, keepdims=True)) ** 2, axis=0))
        else:
            raise ValueError(f"Expected 1D or 2D array, got {x.ndim}D array instead")

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


class RegMetric(SimilarityMetric):
    def __init__(self, dims=40, seed=2023):
        super().__init__(seed)
        self.dims = dims
    
    def score(self, model_data, neural_data):
        num_stimuli = model_data.shape[0]

        red_model = PCA(n_components=self.dims, random_state=self.seed)
        if self.dims < model_data.shape[1]:
            red_model.fit(model_data)
            model_lowd = red_model.transform(model_data)
        else:
            model_lowd = model_data.copy()

        reg = Ridge(alpha=1.0)
        reg.fit(model_lowd, neural_data)
        neural_pred = reg.predict(model_lowd)
        r = self._coefficient_of_determination(neural_data, neural_pred)
        
        return r
