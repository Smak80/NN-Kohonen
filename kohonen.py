import math
import numpy as np
import matplotlib.pyplot as plt

#markers = ('o', '^', 's', 'p', 'h', 'x', 'D')

# нормировка исходных данных
def norm(X):
    NX = X / 255
    NX[NX < 0] = 0
    NX[NX > 1] = 1
    return NX

def plotColors(colors):
    sz = len(colors)
    qsz = math.ceil(math.sqrt(sz))
    lnc = math.ceil(sz / qsz)
    xpoints = np.array([1+int(i / qsz) for i in range(sz)])
    ypoints = np.array([1+i % qsz for i in range(sz)])
    plt.scatter(xpoints, ypoints, c=colors)
    plt.show()

dist = lambda a, b: np.sqrt(np.sum((a - b) ** 2))
nearest = lambda inp, nrn: np.array([dist(inp, w) for w in nrn]).argmin()

def learn(inp, nrn):
    sz = len(inp)
    vlen = len(inp[0])
    la = 0.3
    dla = 0.05
    while la > 0:
        for e in range(epoches):
            for c in inp:
                ivin = nearest(c, nrn)
                nrn[ivin] = nrn[ivin] + la * (c - nrn[ivin])
        la -= dla

def sort(inp, nrn):
    K = len(nrn)
    cluster = [[] for i in range(K)]
    for c in inp:
        cluster[nearest(c, nrn)].append(c)
    return cluster

def show_clusters(clusters):
    for cluster in clusters:
        plotColors(cluster)

#число классов
K = 5
#число эпох обучения
epoches = 100
#нейроны сети
W = np.random.random((K, 3))*0.3

colors = np.asarray(np.loadtxt("colors.txt"), dtype=int)
ncolors = norm(colors)
plotColors(ncolors)
learn(ncolors, W)
clusters = sort(ncolors, W)
show_clusters(clusters)