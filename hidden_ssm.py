import matplotlib.pyplot as plt
import numpy as np
import pickle

ssm = pickle.load(open("test1.p", "rb"))
plt.imshow(ssm, cmap='gray')
plt.show()

