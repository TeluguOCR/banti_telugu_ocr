#!/usr/bin/python

# Top left of the main eight bounding boxes
bboxes_tl = [
(0, 3), (2, 3),
(0, 5), (2, 5),
(0, 7), (2, 7),
(0, 9), (2, 9),
]

def get_bboxes((x, y)):
    return (x, y, x+2, y+2)

bboxes = [get_bboxes(tl) for tl in bboxes_tl]

def bbox(i):
    if i in range(0,4):   return bboxes[0]
    if i in range(4,8):   return bboxes[1]
    if i in range(8,12):  return bboxes[2]
    if i in range(12,16): return bboxes[3]

    if i in range(43,47): return bboxes[4]
    if i in range(47,51): return bboxes[5]
    if i in range(51,55): return bboxes[6]
    if i in range(55,59): return bboxes[7]

    if i in (40, 41) : return get_bboxes((4,3))
    if i == 42:        return get_bboxes((4,5))
    if i == 38:        return get_bboxes((6,5))
    raise IndexError

# Start & extent of arc
def start(i):
    def family(j):
        return (j, j+4, j+8, j+12, j+43, j+47, j+51, j+55)

    value = [180, 270, 0, 90]
    for k in range(4):
        if i in family(k):
            return value[k]

    if i == 38: return 180
    if i == 40: return 90
    if i == 41: return 270
    if i == 42: return 180

    raise IndexError

def extent(i):
    if i in (38, 42): return 270
    if i in (40, 41): return 180
    else: return 90

lines = {
#TOP AROUND
    16:( 1, 3, 2, 3),
    17:( 0, 4, 0, 6),
    18:( 1, 7, 3, 7),
    19:( 4, 4, 4, 6),
    20:( 2, 3, 3, 3),
#TOP MID
    21:( 2, 4, 2, 6),
    23:( 1, 5, 3, 5),
    24:( 0, 5, 1, 5),
#VATTU
    25:( 2, 6, 2, 8),
#TOP TA, E
    26:( 2, 0, 3, 3),
    27:( 1, 2, 1, 3),
    28:( 0, 1, 2, 1),
#TOP GUDI
    29:( 1, 2, 2, 3),
    30:( 2, 3, 3, 2),
    31:( 2, 1, 3, 2),
    32:( 1, 2, 2, 1),
#TOP DHEERGAM, POLLU
    33:( 2, 1, 4, 1),
    34:( 3, 0, 3, 2),
#TOP 'O'THER
    36:( 3, 2, 4, 1),
#TOP AA
    39:( 3, 3, 5, 3),
#BOT AROUND
    59:( 0, 8, 0,10),
    60:( 4, 8, 4,10),
#BOT MID
    61:( 2, 8, 2,10),
    62:( 1, 9, 3, 9),
#BOT VATTU
    63:( 2,10, 2,12),
#TRANSITION
    64:( 4, 6, 4, 8),
    66:( 0, 7, 2, 7),
}

combos = {
    #n:[('line', (x1, y1, x2, y2)), ('arc', (bx1,by1, bx2,by2), st, ext)]
    22:[('line', ( 3, 5, 5, 5)), ('arc', (4,5,6,7), 90, 90)],
    35:[('line', (4,1,5,2)),
        ('line', (5,2,4,3)),
        ('line', (4,3,3,2)),],
    37:[('line', (5,3,  7,3)), ('arc', (6,3,8,5), 0, 359)],
    65:[('line', (3,7,4.5,7)), ('arc', (4,7,5,8), 180, 270)],
}

MAG = 10
def mul(tpl):
    return tuple((1 + i)*MAG for i in tpl)

def combo_draw(w, i, b):
    for seg in combos[i]:
        if   seg[0] == 'line':  lin_wrp(w, seg[1], b)
        elif seg[0] == 'arc':   arc_wrp(w, seg[1], seg[2], seg[3], b)

def draw_piece(w, i, b):
    if i in combos: 
        return combo_draw(w, i, b)
    try:
        arc_wrp(w, bbox(i), start(i), extent(i), b)
    except IndexError:
        lin_wrp(w, lines[i], b)

def draw_letter(w, l):
    sortee = sorted(enumerate(l), key=lambda x: x[1])
    for i, b in sortee:
        draw_piece(w, i, b)


import Image, ImageFont, aggdraw
font= aggdraw.Font("red", "archive/Nandini3.ttf", MAG*3)

def lin_wrp(w, co, b):           
    w.line(mul(co), pen(b))

def arc_wrp(w, bb, st, ext, b):  
    w.arc(mul(bb), st, st+ext, pen(b))

def pen(b): 
    b = int(255 * (1-b))
    return aggdraw.Pen((b, b, b), MAG/10)

def draw(l, title, save=False, path='', show=False):
    im = Image.new('RGB', mul((9,13)), 'white')
    w = aggdraw.Draw(im)
    
    draw_letter(w, l)
    w.text(mul((4, 6)), title, font)
    w.flush()
    
    if show: im.show()
    if save: im.save(path+'.jpg')

def draw_show(l, title):    
    draw(l, title, show=False)

def draw_save(l, title, path='imgs/img_'):    
    draw(l, title, save=True, path=path)

if __name__ == '__main__':
    from mallicodes import mallicodes, unicodes
    for i, l in enumerate(mallicodes):
        l = [float(ll)/2 for ll in l]
        print l
        draw_save(l, unicodes[i])