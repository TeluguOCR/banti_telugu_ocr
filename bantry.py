from glyph import Glyph


class Bantry(Glyph):
    """Class used to process a space seperated line and store the probable
    characters and the respective liklihoods for one glyph.
    """
    scaler = lambda *_: None
    classifier = lambda *_: None

    def __init__(self, line_info):
        super().__init__(line_info)
        self.scaled = self.scaler(self)
        self.likelies = self.classifier(self.scaled.pix)

    def strength(self):
        return max(self.likelies)

    def combine(self, other):
        docombine, combined = False, None

        # Do some stuff here

        if docombine:
            combined = self + other
            combined.scaled = self.scaler(self)
            combined.likelies = self.classifier(combined.scaled.pix)

        return docombine, combined


class BantryFile():
    def __init__(self, name):
        in_file = open(name)
        bantries = []

        iline = 0
        iline_bantries = []

        for line in in_file:
            e = Bantry(line)
            if e.linenum == iline:
                iline_bantries.append(e)

            elif e.linenum > iline:
                bantries.append(iline_bantries)
                iline += 1
                while iline < e.linenum:
                    bantries.append([])
                    iline += 1
                iline_bantries = [e]

            else:
                raise ValueError("Line number can not go down.")

        bantries.append(iline_bantries)
        self.bantries = bantries
        self.num_lines = bantries[-1][-1].linenum
        in_file.close()

    def get_line_bantires(self, i):
        return self.bantries[i]

if __name__ == "__main__":
    import ast
    import sys
    from scaler import ScalerFactory

    banti_file_name = sys.argv[1]
    scaler_prms_file = sys.argv[2]

    with open(scaler_prms_file) as scaler_fp:
        scaler_params = ast.literal_eval(scaler_fp.read())
    Bantry.scaler = ScalerFactory(scaler_params)

    bf = BantryFile(banti_file_name)

    iimg = 0
    linenum = 0
    for linenum in range(bf.num_lines):
        print('*' * 60)
        line_bantries = bf.get_line_bantires(linenum)
        for bantry in line_bantries:
            print(bantry.scaled)