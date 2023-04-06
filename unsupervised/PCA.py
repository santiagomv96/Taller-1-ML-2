import numpy as np
from unsupervised import unsupervised_class

class PCA:

    def __init__(self,n_components):  
        self.n_components=n_components

    def fit(self,x):
        '''Creates the params for PCA transformation'''   
        self.x=x 
        self.mean_x=np.mean(self.x, axis=0)
        x_centered=self.x - self.mean_x
        cov_matrix=(1/len(self.x))*(x_centered.T @ x_centered)
        cov_matrix=np.cov(self.x, rowvar=False)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(cov_matrix)
        self.sorted_indexes = np.argsort(self.eigenvalues)[::-1]
        self.sorted_eigenvalues = self.eigenvalues[self.sorted_indexes]
        self.sorted_eigenvectors = self.eigenvectors[:,self.sorted_indexes]
        self.selected_eigenvectors = self.sorted_eigenvectors[:, :self.n_components]

    def transform(self,x):
        return (x-self.mean_x)@self.selected_eigenvectors
    
    def fit_transform(self,x):
        self.x=x
        self.fit(self.x) 
        return self.transform(self.x)
        
    

        