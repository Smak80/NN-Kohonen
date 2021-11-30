import numpy as np

#markers = ('o', '^', 's', 'p', 'h', 'x', 'D')

colors = np.asarray(np.random.random((89, 3))*255, dtype=int)
#colors[:, -2] = (colors[:, -2] % 10 + 3)**2
#colors[:, -1] %= len(markers)
print(colors)
np.savetxt("colors.txt", colors)
