import numpy as np
import unsupervised_class

class PCA(unsupervised_class):

    def __init__(self,n_vectors):   
        self.n_vectors=n_vectors 

    def fit(self,x):
        '''Creates the matrtixes for SVD transformation and generates the truncate matrix, which allows
        to reduce new features using
        params used:
        x: Data to train
        n_vectors: How many vectors you will use
        ''' 
        self.x=x  
        self.U, self.s, self.Vt = np.linalg.svd(self.x) 
        Uk = self.U[:, :self.n_vectors]
        sk = np.diag(self.s[:self.n_vectors])
        Vk = self.Vt[:self.n_vectors, :]        
        self.mu = np.mean(self.x, axis=0)
        self.sigma = np.std(self.x, axis=0)
        self.truncate_svd = np.dot(Uk, np.dot(sk, Vk))

    def transform(self,x):   
        X_new_centered = x - self.mu
        X_new_scaled = X_new_centered / self.sigma
        return np.dot(X_new_scaled, self.truncate_svd)
    
    def fit_transform(self,x):
        self.x=x  
        self.fit(x,self.n_vectors)
        return self.fit(x,self.n_vectors)
    