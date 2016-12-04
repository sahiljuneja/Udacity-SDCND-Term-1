import numpy as np
import cv2
import matplotlib.pylab as plt
import matplotlib.image as mpimg
from itertools import product

bicycle = mpimg.imread('bicycle.png')
bicycle = (bicycle * 255).astype(np.uint8)
blurcycle = cv2.GaussianBlur(bicycle, (5, 5), 0)

edges = cv2.Canny(blurcycle, 100, 200)


def make_hough_lines(image, rho=1, theta=np.pi/180):
    lines = cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=100)
    lines = lines.reshape((lines.shape[0], lines.shape[2]))
    for x1, y1, x2, y2 in lines:
        cv2.line(image, (x1, y1), (x2, y2), [255, 0, 0], 3)

    return image





rhos = [1, 2, 3]
thetas = [np.pi/180, 10*np.pi/180, 30*np.pi/180]
r = 1
fig = plt.figure()
for rho, theta in product(rhos, thetas):
    print(rho, theta)
    edges = cv2.Canny(blurcycle, 100, 200)
    image = make_hough_lines(edges, rho, theta)

    a = fig.add_subplot(3, 3, r)

    plt.imshow(image, cmap='gray')

    fig.suptitle('Rho: %s, Theta: %s' % (rho, theta))
    r += 1
plt.show()