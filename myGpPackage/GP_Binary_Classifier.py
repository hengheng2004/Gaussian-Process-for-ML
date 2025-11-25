import numpy as np

class GP_Binary_Classifier:
    def __init__(self):
        pass
    
    def se(self,x1,x2):
        return np.exp(-0.5 * np.linalg.norm(x1-x2) ** 2)
    
    def sigma(self,z):
        return 1. / (1 + np.exp(-z))

    def likelihood_y_f(self,y,f):
        return np.log(self.sigma(y * f))

    def grad_likelihood_y_f(self,y,f):
        t = (y+1.) / 2
        pi = self.sigma(f)
        return t - pi
    
    def hessian_likelihood_y_f(self,y,f):
        pi = self.sigma(f)
        return - np.diag(pi * (1 - pi))

    def sqrtm(self, A):
        eigvals, eigvecs = np.linalg.eigh(A)
        A_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        return A_sqrt
    
    def generate_cov(self):
        return np.array([[self.se(x1,x2) for x2 in self.X] for x1 in self.X])

    def load_data(self,X,y):
        assert len(X) == len(y), "数据不匹配"
        self.X = np.array(X)
        self.y = np.array(y)
        self.K = self.generate_cov()


    def fit(self,max_iter=1000,tol=1e-6):
        X = self.X
        y = self.y
        assert not (X is None or y is None), "尚未导入数据"
        K = self.K
        N = len(X)
        f = np.zeros(N)
        I = np.eye(N)
        
        for _ in range(max_iter):
            W = -self.hessian_likelihood_y_f(y,f)
            sqrtW = self.sqrtm(W)
            L = np.linalg.cholesky(I + sqrtW @ K @ sqrtW)
            b = W @ f + self.grad_likelihood_y_f(y,f)
            a = b - sqrtW @ np.linalg.solve(L.T, np.linalg.solve(L, sqrtW @ K @ b))
            f_new = K @ a
            if (np.linalg.norm(f-f_new) < tol):
                f = f_new
                break
            f = f_new
        
        self.f_hat = f
        self.max_log_q = -0.5 * a.T @ f + self.likelihood_y_f(y,f) - np.sum(np.log(np.diag(L)))
        return self.f_hat, self.max_log_q
    
    def predict_prob(self,x_new):
        x_new = np.array(x_new)
        k_star = [[self.se(x1,x2) for x2 in self.X] for x1 in x_new]
        E_f = k_star @ np.linalg.solve(self.K, self.f_hat)
        pi = self.sigma(E_f)
        return pi
    
    def predict_class(self,x_new,threhold=0.5):
        pi = self.predict_prob(x_new)
        pred = [1 if prob > threhold else -1 for prob in pi]
        return pred
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def test():
        np.random.seed(42)
        N = 100
        m = 2
        X = np.random.randn(N * m).reshape((N,m))
        X[50:,:] += 2
        y = np.ones(N)
        y[50:] = -1

        classifier = GP_Binary_Classifier()
        classifier.load_data(X,y)
        f_hat, mle = classifier.fit()

        n = 20
        x_new = np.random.randn(n,m)
        x_new[5:,:] += 1
        prob = classifier.predict_prob(x_new)
        print(prob)
        pred = classifier.predict_class(x_new)
        print(pred)
        pred = np.array(pred)
        fig = plt.figure()
        x_pos = x_new[pred == 1]
        x_neg = x_new[pred == -1]
        print(x_pos)
        print(x_neg)
        plt.scatter(x_pos[:,0],x_pos[:,1],c='blue',label='1')
        plt.scatter(x_neg[:,0],x_neg[:,1],c='red',label='-1')
        plt.legend()
        plt.show()
            
    test()


