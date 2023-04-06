import numpy as np

class T_SNE:

    def __init__(self,x):   
        self.x=x 
        self.scalized_X = (self.x - np.mean(self.x, axis=0)) / np.std(self.x, axis=0)

    def similitary_matrix(self):
        '''compute similitary matrix taking an euclidian distance between the points'''
        distances = np.linalg.norm(X[:, None] - X[None, :], axis=-1)

