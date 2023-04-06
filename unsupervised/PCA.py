import numpy as np
import unsupervised_class

class PCA(unsupervised_class):

    def __init__(self,x,n_components):    
        self.x=x
        self.n_components=n_components

    def fit(self,x,n_components):
        '''Creates the params for PCA transformation'''    
        self.x=x
        self.n_components=n_components  
        self.mean_x=np.mean(self.x)
        x_centered=self.x - self.mean_x
        cov_matrix=(1/len(self.x))*(x_centered.T @ x_centered)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(cov_matrix)
        self.sorted_indexes = np.argsort(self.eigenvalues)[::-1]
        self.sorted_eigenvalues = self.eigenvalues[self.sorted_indexes]
        self.sorted_eigenvectors = self.eigenvectors[:,self.sorted_indexes]
        self.selected_eigenvectors = self.sorted_eigenvectors[:, 0:self.n_components]

    def transform(self,x):
        return (x-self.mean_x)@self.selected_eigenvectors
    
    def fit_transform(self,x,n_components):
        self.x=x
        self.n_components=n_components 
        self.fit(self.x,self.n_components) 
        return self.transform(self.x)
        
    

        