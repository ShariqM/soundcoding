
import numpy as np
import matplotlib.pyplot as plt


N = 40
x = np.array(range(1,N+1))

plt.subplot(221)
plt.xlabel("x")
plt.ylabel("exp(x)")
plt.plot(x, np.exp(x))

plt.subplot(222)
plt.xlabel("log(x)")
plt.ylabel("exp(x)")
plt.plot(np.log(x), np.exp(x))

plt.subplot(223)
plt.xlabel("x")
plt.ylabel("log(exp(x))")
plt.plot(x, np.log(np.exp(x)))

plt.subplot(224)
plt.xlabel("log(x)")
plt.ylabel("log(exp(x))")
plt.plot(np.log(x), np.log(np.exp(x)))

plt.show()
