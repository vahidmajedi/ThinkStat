import numpy as np
from MyClass import PMF, histogram
import matplotlib.pyplot as plt

x = np.random.normal(0,10,100000)
x = np.round(x)

pmf = PMF(x)
pmf = pmf.normalize()
m = pmf.percentile(60)