import numpy as np
from math import sin, pi, cos
from banti.glyph import Glyph

halfsize = 40
size = 2*halfsize + 1
picture = np.zeros((size, size))
for t in range(-135, 135):
    x = round(halfsize + halfsize * cos(pi * t / 180))
    y = round(halfsize + halfsize * sin(pi * t / 180))
    picture[x][y] = 1

zoomsz = 1 * halfsize
b = Glyph(['O', 0, 0, size, size, 0, 0, 0, 0, None])
b.set_pix(picture)
c = Glyph()
for t in range(0, 360, 15):
    x = round(zoomsz + zoomsz * cos(pi * t / 180))
    y = round(zoomsz + zoomsz * sin(pi * t / 180))
    b.set_xy_wh((x, y, size, size))
    c = c + b

print(b)
print(c)