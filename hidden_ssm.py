import matplotlib.pyplot as plt
import numpy as np
import pickle

ssm = pickle.load(open("test2.p", "rb"))
f = np.vectorize(lambda x : x >= 0.95)
plt.imshow(f(ssm), cmap='gray')
plt.show()

