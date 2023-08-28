import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('hu.jpg')

plt.imshow(img)
plt.show()