from sklearn.decomposition import PCA
import numpy as np

class DimRed():
    def __init__(self, X, output_dim):
        self.output_dim = output_dim
        self.X = X
    def pca(self, treshold):
        print("starting pca transformation")
        print("n_comp:", self.output_dim)
        print("x-shape", self.X.shape)
        print("type", type(self.X))
        pca = PCA(n_components = self.output_dim)
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
            idx = max(self.output_dim, t_index[0])
            print(idx, "idx")
            print(components[:, :idx].shape)
            print("----")
        return components[:, :idx], variances, message