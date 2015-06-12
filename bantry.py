from glyph import Glyph
import logging

logger = logging.getLogger(__name__)
logi = logger.info
logd = logger.debug


class Bantry(Glyph):
    """Class used to process a space seperated line and store the probable
    characters and the respective liklihoods for one glyph.
    """
    scaler = lambda *_: None
    classifier = lambda *_: (None, None)

    def __init__(self, line_info=None):
        super().__init__(line_info)
        if line_info:
            self.scaled = self.scaler(self)
            self.likelies = self.classifier(self.scaled)

    @property
    def best_char(self):
        return max(self.likelies, key=lambda x: x[1])[0]

    def strength(self):
        return max(self.likelies, key=lambda x: x[1])[1]

    def __str__(self):
        return super().__str__() + "\n" + \
               "; ".join("{} {:.3f}".format(char, lik) for char, lik in self.likelies)

    def combine(self, other):
        docombine, combined = False, None
        logd("Checking to combine\n{}\n{}".format(self, other))

        # Put checks here
        if other is Space:
            return False, None

        if self.best_char == '-' and other.best_char == '-':
            docombine = True

        if docombine:
            combined = self + other
            combined.scaled = self.scaler(combined)
            combined.likelies = self.classifier(combined.scaled)
            logi("Combining\n{}\n{}\n{}".format(self, other, combined))

        return docombine, combined


class Space():
    likelies = [(" ", 0)]
    best_char = " "
    strength = 0
    scaled = "---\n| |\n---"

    @classmethod
    def combine(cls, other):
        return False, None

    @classmethod
    def __str__(cls):
        return "_"


class BantryFile():
    def __init__(self, name):
        in_file = open(name)
        self.file_bantries = []

        iword, iline = 0, 0
        line_bantries = []

        for line in in_file:
            e = Bantry(line)
            if e.linenum == iline:
                if e.wordnum > iword:
                    iword = e.wordnum
                    line_bantries.append(Space)
                line_bantries.append(e)

            elif e.linenum > iline:
                self.file_bantries.append(line_bantries)
                iword = 0
                iline += 1
                while iline < e.linenum:
                    self.file_bantries.append([])
                    iline += 1
                line_bantries = [e]

            else:
                raise ValueError("Line number can not go down.")

        self.file_bantries.append(line_bantries)
        self.num_lines = self.file_bantries[-1][-1].linenum

        self.text = ""
        for bantries_inline in self.file_bantries:
            for bantree in bantries_inline:
                self.text += bantree.best_char
            self.text += "\n"

        in_file.close()

    def get_line_bantires(self, i):
        return self.file_bantries[i]

if __name__ == "__main__":
    import sys
    from scaler import ScalerFactory

    banti_file_name = sys.argv[1] if len(sys.argv) > 1 else "sample_images/praasa.box"
    scaler_prms_file = sys.argv[2] if len(sys.argv) > 2 else "scalings/relative48.scl"

    Bantry.scaler = ScalerFactory(scaler_prms_file)
    bf = BantryFile(banti_file_name)

    for linenum in range(bf.num_lines):
        print('*' * 60)
        line_bantries = bf.get_line_bantires(linenum)
        for bantry in line_bantries:
            print(bantry.scaled)