""" Print out robot's orientation """

import numpy as np
from matplotlib import pyplot as plt

roll = np.load("roll.npy")
pitch = np.load("pitch.npy")
yaw = np.load("yaw.npy")
names = ["Roll", "Pitch", "Yaw"]

plt.plot(roll)
plt.plot(pitch)
plt.plot(yaw)
plt.legend(names)
plt.title("Angular orientations [radians]")
plt.show()

plt.plot(np.degrees(roll))
plt.plot(np.degrees(pitch))
plt.plot(np.degrees(yaw))
plt.legend(names)
plt.title("Angular orientations [degrees]")
plt.show()