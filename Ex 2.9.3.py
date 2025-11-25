import numpy as np
import matplotlib.pyplot as plt
from myGpPackage.generate_GP import GP

def wiener_kernel(x1,x2):
    return min(x1,x2) - x1 * x2

gp = GP(wiener_kernel)
xs = np.linspace(0,1,100)[1:-1]

gp.load_xs(xs)

ys = gp.simulate(3)

for y in ys:
    fullx = np.concatenate([[0],xs,[1]])
    fully = np.concatenate([[0],y,[0]])
    plt.plot(fullx, fully)

plt.show()


