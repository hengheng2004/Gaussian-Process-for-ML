import numpy as np

class GP:
    kernel = None
    xs = None

    def __init__(self,kernel,input_dim=1):
        self.kernel = kernel

    
    def load_xs(self, xs):
        self.xs = xs
    
    def generate_cov(self):
        xs = self.xs
        return np.array([[self.kernel(x1,x2) for x2 in xs] for x1 in xs])
    
    def simulate(self, times=1):
        assert not (self.xs is None), "Havn't load xs!"

        cov = self.generate_cov()
        n = len(self.xs)

        res = []

        for _ in range(times):
            res.append(np.random.multivariate_normal(np.zeros(n), cov))
        
        return res


if __name__ == "__main__":
    def se(x1,x2):
        return np.exp(-0.5 * np.linalg.norm(x1 - x2) ** 2)
    
    gp1 = GP(kernel=se)
    xs = np.linspace(0,20,100)
    gp1.load_xs(xs)
    y = gp1.simulate(1)
    y = y[0]
    import matplotlib.pyplot as plt
    plt.plot(xs,y)
    plt.show()
