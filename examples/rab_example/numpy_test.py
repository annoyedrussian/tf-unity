import numpy as np

a = [1, 2, 3, 4, 5, 6]
b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 

a = np.expand_dims(np.array(a, dtype=np.int32), axis=1)
b = np.array(b, dtype=np.int32).reshape(6, 2)
print(a)
print(b)

o = np.hstack([a, b])
print(o)