import math
import numpy as np
import matplotlib.pyplot as plt

normcoeff = []

# нормировка исходных данных
def norm(X):
    TX = np.asarray(X, dtype=float).T
    for i in range(len(TX)):
        k = 3 if (i+1 == len(TX)) else 1
        normcoeff.append((1 / np.std(TX[i]), np.median(TX[i])))
        TX[i] = k * (TX[i] * normcoeff[-1][0] - normcoeff[-1][1] * normcoeff[-1][0])
    NX = TX.T
    return NX

def denorm(X):
    TX = np.asarray(X, dtype=float).T
    for i in range(len(TX)):
        k = 3 if (i + 1 == len(TX)) else 1
        TX[i] = (TX[i] / k + normcoeff[i][1] * normcoeff[i][0]) / normcoeff[i][0]
    NX = TX.T
    return NX


def getcolors(X):
    if (len(X)>0):
        CX = X[:, 0:3]
        CX = CX / 255
        CX[CX < 0] = 0
        CX[CX > 1] = 1
    else:
        CX = np.array([])
    return CX

def plotData(data):
    sz = len(data)
    if sz == 0 :
        return
    cdata = getcolors(data)
    qsz = math.ceil(math.sqrt(sz))
    #lnc = math.ceil(sz / qsz)
    xpoints = np.array([1+int(i / qsz) for i in range(sz)])
    ypoints = np.array([1+i % qsz for i in range(sz)])
    #cmrk = np.empty(sz, dtype=str)
    plt.scatter(xpoints, ypoints, c=cdata, s=data[:,-1])
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
                iwin = nearest(c, nrn)
                nrn[iwin] = nrn[iwin] + la * (c - nrn[iwin])
        la -= dla

def sort(inp, nrn):
    K = len(nrn)
    cluster = [[] for i in range(K)]
    ninp = norm(inp)
    for i in range(len(inp)):
        cluster[nearest(ninp[i], nrn)].append(inp[i])
    for i in range(len(cluster)):
        cluster[i] = np.array(cluster[i])
    return cluster

def show_clusters(clusters):
    for cluster in clusters:
        plotData(cluster)

#число классов
K = 12

#число эпох обучения
epoches = 50

#исходные данные
data = np.asarray(np.loadtxt("data.txt"), dtype=int)

#нейроны сети
W = np.random.random((K, len(data[0])))*0.3
# W = [[255, 0, 0, 74 ],
#      [255, 0, 0, 221],
#      [255, 0, 0, 466],
#      [255, 0, 0, 809],
#      [0, 255, 0, 74 ],
#      [0, 255, 0, 221],
#      [0, 255, 0, 466],
#      [0, 255, 0, 809],
#      [0, 0, 255, 74 ],
#      [0, 0, 255, 221],
#      [0, 0, 255, 466],
#      [0, 0, 255, 809]]
# W = norm(W)

plotData(data)
ndata = norm(data)
learn(ndata, W)
clusters = sort(data, W)
#dclusters = []
#for i in range(len(clusters)):
#    dclusters.append(denorm(clusters[i]))
show_clusters(clusters)

