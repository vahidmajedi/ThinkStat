import numpy as np
import matplotlib.pyplot as plt

class PDF:
    def __init__(self, pop=[], prob=[]):
        self.pop =  pop
        self.prob = prob

    def probability(self, val):
        idx = np.where(self.pop==val)
        return self.prob[idx]
    
    def normalize(self):
        self.prob /= sum(self.prob)
        return self
        
    def mean(self):
        return sum(np.multiply(self.pop, self.prob))
    
    def variance(self):
        mu = self.mean()
        return sum(np.multiply(np.square(self.pop - mu), self.prob))
    
    def trim(self, low_band, up_band):
        idx = np.where((self.pop >= low_band) & (self.pop <= up_band))
        res = PDF(self.pop[idx], self.prob[idx])
        return res.normalize()
        
    def condition(self, sample, cond):
        idx = np.where(cond)
        res = PDF(self.pop[idx], self.prob[idx])
        res = res.normalize()
        return (res, res.probability(sample))


class Stat:
    def __init__(self, t):
        if type(t) == np.ndarray:
            self.t = t
        else:
            self.t = np.array(t)
        self.r = len(t)
        
    def mean(self):
        return sum(self.t)/self.r

    def variance(self):
        mu = self.mean()
        return sum(np.square(self.t-mu))/self.r

    def freq(self, bin_size= [], bin_start = []):
        if not bin_size:
            seq = self.t
        else:
            if not bin_start:
                bin_start = min(self.t)
            seq = [(l//bin_size)*bin_size+bin_size/2 for l in self.t]
        tmp = {}
        for x in seq:
            tmp[x] = tmp.get(x, 0) + 1
        tmp = dict(sorted(tmp.items()))
        res = PDF(np.array(list(tmp.keys())), np.array(list(tmp.values())))
        return res

    def pdf(self, bin_size =[], bin_start = []):
        res = self.freq(bin_size, bin_start)
        res.prob = res.prob / self.r
        return res
    
    def cdf(self, bin_size = [], bin_start = []):
        res = self.pdf(bin_size, bin_start)
        res.prob = np.cumsum(res.prob)
        return res

    def all_modes(self, bin_size = [], bin_start = []):
        res = []
        tmp = self.pdf(bin_size, bin_start)
        idx = np.flip(np.argsort(tmp.prob))
        for a,b in zip(tmp.pop[idx], tmp.prob[idx]):
            res.append((a, b))
        return res

    def mode(self):
        return self.all_modes()[0][0]

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
        st = Stat(data)
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
        st = Stat(data)
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
        plt.bar(st.pop+i*width, st.prob, width=width,
                color=colors[i%5], alpha=0.5)


class StatList:
    def __init__(self, t):
        if type(t) == list:
            if type(t[0]) == list:
                self.t = [np.array(l) for l in t]
            elif type(t[0]) == np.ndarray:
                self.t = t
            else:
                self.t = [np.array(t)]
        elif type(t) == np.ndarray:
            if np.ndim(t) == 1:
                self.t = [t]
            else:
                _,c = np.shape(self.t)
                self.t = [t[:,i] for i in range(c)]
        else:
            self.t = [t]

        self.c = len(self.t)
        self.r = [len(l) for l in self.t]
        
    def mean(self):
        return [sum(self.t[i])/self.r[i] for i in range(self.c)]

    def variance(self):
        mu = self.mean()
        return [sum(np.square(self.t[i]-mu[i])/self.r[i])
                for i in range(self.c)]

    def freq(self):
        res = []
        for i in range(self.c):
            tmp = {}
            for x in self.t[i]:
                tmp[x] = tmp.get(x, 0) + 1
            tmp = dict(sorted(tmp.items()))
            res.append(PDF())
            res[i].pop = np.array(list(tmp.keys()))
            res[i].prob = np.array(list(tmp.values()))
        return res

    def pdf(self):
        res = self.freq()
        for i in range(self.c):
            res[i].prob = res[i].prob / self.r[i]
        if len(res) == 1:
            return res[0]
        return res
    
    def cdf(self):
        res = self.pdf()
        for i in range(self.c):
            res[i].prob = np.cumsum(res[i].prob)
        if len(res) == 1:
            return res[0]
        return res

    def all_modes(self):
        res = []
        tmp = self.pdf()
        for i in range(self.c):
            tmp1 = []
            idx = np.flip(np.argsort(tmp[i].prob))
            for a,b in zip(tmp[i].pop[idx], tmp[i].prob[idx]):
                tmp1.append((a, b))
            res.append(tmp1)
        return res

    def mode(self):
        return [self.all_modes()[i][0][0] for i in range(self.c)]

    def histogram(self, action):
        colors = ('b','r','g','k','c')
        width = 0.8/self.c
        dist = action()
        for i in range(self.c):
            plt.bar(dist[i].pop+i*width, dist[i].prob, width=width,
                    color=colors[i], alpha=0.5)

