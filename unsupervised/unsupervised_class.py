class unsupervised_class:
    def __init___(self):
        pass

    def fit(self, x):
        raise NotImplementedError
    
    def transform(self, x):
        raise NotImplementedError
    
    def fit_transform(self, x):        
        return self.transform(self.fit(x))