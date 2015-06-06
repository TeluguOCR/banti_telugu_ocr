from math import ceil

class Box:
    def  __init__(self, args=None):
        if type(args) is str:
            self.FromLine(args)
        elif type(args) is list:
            self.FromList(args)
        elif args is None:
            self.FromList(['', 0, 0, 0, 0, 0, 0, 0, 0, ''])
        self.error = ''

    def FromLine(self, box_str):
        box = box_str.rstrip().split()
        i = 0
        self.text = box[i]; i += 1
        self.x    = int(box[i]); i += 1
        self.y    = int(box[i]); i += 1
        self.wd   = int(box[i]); i += 1
        self.ht   = int(box[i]); i += 1
        self.bl   = int(box[i]); i += 1
        self.tl   = int(box[i]); i += 1
        self.line = int(box[i]); i += 1
        self.word = int(box[i]); i += 1
        self.pic  = box[i]
        self.FixX2Y2()

    def FromList(self, box_list):
        self.text, self.x, self.y, self.wd, self.ht, self.bl, self.tl, \
        self.line, self.word, self.pic = box_list
        self.FixX2Y2()

    def __str__(self):
        return ' '.join(str(i) for i in (self.text, self.x, self.y, \
                                       self.wd, self.ht, self.bl, self.tl, \
                                       self.line, self.word, self.pic)) + '\n'

    # X2 and Y2 are set pythonically. i.e. they are not part of the image
    # They are just outside
    def FixX2Y2(self):
        self.y2 = self.y + self.ht
        self.x2 = self.x + self.wd
    def FixWH(self):
        self.wd = self.x2 - self.x
        self.ht = self.y2 - self.y
    def SetXYWH(self, xywh):
        self.x, self.y, self.wd, self.ht = xywh
        self.FixX2Y2()
    def SetXYXY(self, xyxy):
        self.x, self.y, self.x2, self.y2 = xyxy
        self.FixWH()

    def UnPack(self):
        return [[self.GetPixel(row,col) for col in range(self.ht)] \
                                             for row in range(self.wd)]

    def GetPixel(self, row, col):
        if not (0 <= row < self.ht and 0 <= col < self.wd): 
            raise IndexError
        ipix = row * self.wd + col 
        isix = ipix//6
        if (ord(self.pic[isix]) - ord('0')) & (1 << (5 - (ipix%6))):
            return 1
        else:
            return 0

    def GetPixelAbs(self, abs_row, abs_col):
        if self.y <= abs_row < self.y2 and self.x <= abs_col < self.x2:
            return self.GetPixel(abs_row-self.y, abs_col-self.x)
        else:
            return 0

    def PackToPic(self, un_pic):
        s = [ord('0') for i in range(ceil(self.ht*self.wd/6))]
        for row in range(self.ht):
            for col in range(self.wd):
                ipix = row * self.wd + col 
                isix = ipix//6
                s[isix] += un_pic[row][col] << (5 - (ipix%6));
        self.pic = ''.join(chr(i) for i in s)

    def AddBox(self, addee, txt=None):
        added = Box()
        added.SetXYXY((min(self.x, addee.x), min(self.y, addee.y),            \
                       max(self.x2, addee.x2), max(self.y2, addee.y2))) 
        un_pic = [[ ( self.GetPixelAbs(row, col) or                           \
                            addee.GetPixelAbs(row, col) )                     \
                                for col in range(added.x, added.x2)]          \
                                    for row in range(added.y, added.y2)]
        added.PackToPic(un_pic)
        added.line, added.word = self.line, self.word
        added.bl, added.tl     = self.bl, self.tl
        if txt is None: txt = self.text + addee.text
        return added.SetText(txt)

    def SetText(self, txt, err=''):
        self.text = txt
        self.error = err
        return self

    def GetArea(self):
        return self.wd*self.ht

    def Print(self):
        for r in range(self.ht):
            for c in range(self.wd):
                if self.GetPixel(r, c): print('#', end='')
                else:                   print(' ', end='')
            print()


def FindBigSmallBoxes(boxes):
    imax = 0
    imin = 0
    for i in range(1, len(boxes)):
        if boxes[imax].GetArea()  < boxes[i].GetArea():
            imax = i
        if boxes[imin].GetArea() >= boxes[i].GetArea():
            imin = i
    return (imax, imin)

def GetBoxesForLine(file_name):
    with open(file_name) as box_file:
        i, l = 0, []
        for line in box_file:
            box = Box(line)
            if box.line != i:
                yield l
                i, l = box.line, []
            l.append(box)
        yield l

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
    b.PackToPic(s)
    c = Box()
    for t in range(0, 360, 15):        
        x = round(ZOOMSZ + ZOOMSZ * cos(pi * t / 180))
        y = round(ZOOMSZ + ZOOMSZ * sin(pi * t / 180))
        print(t, x, y)
        b.SetXYWH((x, y, SIZE, SIZE))
        c = c.AddBox(b)

    c.Print()
    with open('/tmp/test.box', 'w') as f:
        f.write(str(c))
        f.write(str(b))
