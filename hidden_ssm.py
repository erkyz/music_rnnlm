import matplotlib.pyplot as plt
import pickle

ssm = pickle.load(open("test2.p", "rb"))
plt.imshow(ssm, cmap='gray')
plt.show()

