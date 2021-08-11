from sklearn.decomposition import PCA
import numpy as np
from ._ae import AutoEncoder

class DimRed():
    def __init__(self, X, experiment_name):
        self.X = X
        self.experiment_name = experiment_name
    def pca(self, output_dim, treshold):
        print("starting pca transformation")
        print("n_comp:", output_dim)
        print("x-shape", self.X.shape)
        print("type", type(self.X))
        pca = PCA(n_components = output_dim)
        components = pca.fit_transform(self.X)
        variances = pca.explained_variance_ratio_
        cumsum_variances = np.cumsum(variances)
        if cumsum_variances[-1] <= treshold:
            message = ["Output dimension is not enough"]
            pca = PCA(n_components = int(0.9 * self.X.shape[1]))
            components = pca.fit_transform(self.X)
            variances = pca.explained_variance_ratio_
            cumsum_variances = np.cumsum(variances)
        if cumsum_variances[-1] <= treshold:
            message = ["PCA cannot reduce Dimension of data efficiently"]
        else:
            t_index = np.where(cumsum_variances > treshold)
            print(t_index)
            t_index = t_index[0]
            print("----")
            # print(cumsum_variances)
            idx = max(output_dim, t_index[0])
            print(idx, "idx")
            print(components[:, :idx].shape)
            print("----")
        return components[:, :idx], variances, message
    
    def auto_encoder(self, output_dim):
        print("- - - auto-encoder called - - - ")
        print(self.X.shape, "<< Shape of X")
        dim_reducer = AutoEncoder(input_shape = self.X.shape[1],
                                 layers = [int(0.5 * self.X.shape[1]), int(0.25 * self.X.shape[1]), int(0.125 * self.X.shape[1]), output_dim])
        print("log-checkpoint")
        history = dim_reducer.train(self.X, self.X)
        dim_reducer.save(history.history, self.experiment_name + "_ae_")
        dim_reducer.visualize(history.history, self.experiment_name + "_ae_")
        low_dim = dim_reducer.get_low_dim(self.X)
        return low_dim, None, "AE"