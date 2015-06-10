
if __name__ == "__main__":
    from math import sin, pi, cos
    HALFSIZE = 40
    SIZE = 2*HALFSIZE + 1
    s = [[0 for i in range(SIZE)] for j in range(SIZE)]
    for t in range(-135, 135):
        x = round(HALFSIZE + HALFSIZE * cos(pi * t / 180))
        y = round(HALFSIZE + HALFSIZE * sin(pi * t / 180))
        s[x][y] = 1

    ZOOMSZ = 1 * HALFSIZE
    b = Box(['O', 0, 0, SIZE, SIZE, 0, 0, 0, 0, ''])
    b.pack2pic(s)
    c = Box()
    for t in range(0, 360, 15):        
        x = round(ZOOMSZ + ZOOMSZ * cos(pi * t / 180))
        y = round(ZOOMSZ + ZOOMSZ * sin(pi * t / 180))
        print(t, x, y)
        b.set_xy_wh((x, y, SIZE, SIZE))
        c = c.__add__(b)

    c.Print()
    with open('/tmp/test.box', 'w') as f:
        f.write(str(c))
        f.write(str(b))
