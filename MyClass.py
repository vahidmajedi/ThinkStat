import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

EPSILON = 1e-6

class PMF:
    def __init__(self, *args):
        if len(args) == 0:
            self.pop = []
            self.freq = []
        elif len(args) == 1:
            if isinstance(args[0], (np.ndarray, list)):
                dict_ = Counter(args[0])
                self.pop = np.asanyarray([k for k in dict_])
                self.freq = np.asanyarray([k for k in dict_.values()])
            elif isinstance(args[0], dict):
                self.pop = np.asanyarray([k for k in args[0]])
                self.freq = np.asanyarray([k for k in args[0].values()])
            elif isinstance(args[0], PMF):
                self.pop = args[0].pop
                self.freq = args[0].freq
        elif len(args) == 2:
            self.pop = args[0]
            self.freq = args[1]
        else:
            raise ValueError("Arguments is not Compatible")

        if not all(a<=b for a,b in zip(self.pop, self.pop[1:])):
            idx = np.argsort(self.pop)
            self.pop = self.pop[idx]
            self.freq = self.freq[idx]

    def __len__(self):
        return len(self.pop)

    def mean(self):
        return sum(np.multiply(self.pop, self.freq)) / self.total()

    def variance(self):
        mu = self.mean()
        return sum(np.multiply(np.square(self.pop - mu),
                               self.freq)) / self.total()

    def std(self):
        return np.sqrt(self.variance())

    def all_modes(self, n=1):
        idx = np.flip(np.argsort(self.freq), axis=0)[:n-1]
        return [(a, b) for a, b in zip(self.pop[idx], self.freq[idx])]

    def mode(self):
        return self.all_modes()[0][0]

    def max_like(self):
        return self.all_modes()[0][1]

    def total(self):
        return sum(self.freq)

    def normalize(self):
        res = PMF(self)
        res.freq = res.freq / res.total()
        return res

    def isnormal(self):
        return np.abs(self.total() - 1) < EPSILON

    def p(self, val):
        if not isinstance(val, list):
            val = [val]
        res = []
        for x in val:
            idx = np.where(self.pop==x)
            tmp = 0
            if idx != []:
                tmp = self.freq[idx] if self.isnormal() else self.normalize().freq[idx]
            res.append(tmp)
        return res

    def fr(self, val):
        idx = np.where(self.pop==val)
        return self.freq[idx]

    def trim(self, low_band = None, up_band = None):
        if low_band is None:
            low_band = min(self.pop)
        if up_band is None:
            up_band = max(self.pop)
        idx = np.where((self.pop >= low_band) & (self.pop <= up_band))
        res = PMF(self.pop[idx], self.freq[idx])
        return res.normalize()

    def condition(self, cond, sample = None):
        idx = np.where(cond)
        res = PMF(self.pop[idx], self.prob[idx])
        res = res.normalize()
        if sample is None:
            return res
        return (res, res.p(sample))

    def makeCdf(self):
        res = PMF(self)
        res.freq = np.cumsum(self.freq)
        return res


class ConDist():
    def exponent(self, landa, size):
        F = 1 - np.random.uniform(size = size)
        return -(1/landa)*np.log(F)

    def pareto(self, alfa, xmin, size):
        F = 1 - np.random.uniform(size = size)
        return np.exp(np.log(xmin) - np.log(F)/alfa)

    def weibul(self, k, landa, size):
        F = 1 - np.random.uniform(size = size)
        return np.exp(np.log(landa) + (1/k)*np.log(-np.log(F)))

    def fit2exponent(self, data):
        st = PMF(data)
        cdf = st.cdf()
        pop = cdf.pop[:-1]
        prob = cdf.prob[:-1]
        Y = np.log(1-prob)
        A = np.ones((len(pop),2))
        A[:,0] = pop
        landa = np.matmul(np.linalg.inv(np.matmul(A.T, A)),
                          np.matmul(A.T,Y))

        ye = landa[0]*pop
        err = np.sqrt((1/len(Y))*sum((Y-ye)**2))*100

        plt.plot(pop, Y, 'ro')
        plt.plot(pop, ye)
        return -landa[0], err

    def fit2pareto(self, data):
        st = PMF(data)
        cdf = st.cdf()
        pop = cdf.pop[:-1]
        prob = cdf.prob[:-1]
        Y = np.log(1-prob)
        A = np.ones((len(pop),2))
        A[:,0] = np.log(pop)
        factors = np.matmul(np.linalg.inv(np.matmul(A.T, A)),
                          np.matmul(A.T,Y))
        k = -factors[0]
        xm = np.exp(factors[1]/k)

        ye = factors[0]*np.log(pop) + factors[1]
        err = np.sqrt((1/len(Y))*sum((Y-ye)**2))*100

        plt.plot(np.log(pop), Y, 'ro')
        plt.plot(np.log(pop), ye)
        return k, xm, err


def histogram(pdf_list):
    colors = ('b','r','g','k','c')
    if type(pdf_list) != list:
        pdf_list = [pdf_list]
    n = len(pdf_list)
    width = 0.8/n
    for st, i in zip(pdf_list, range(n)):
        plt.bar(st.pop+i*width, st.freq, width=width,
                color=colors[i%5], alpha=0.5)


