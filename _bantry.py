from glyph import Glyph
from scaler import ScalerFactory


class Bantry(Glyph):
    """Class used to process a space seperated line and store the probable
    characters and the respective liklihoods for one glyph.
    """

    def __init__(self, line, word, liklies):
        self.line = line
        self.word = word
        self.liklies = liklies

    @classmethod
    def fromstr(cls, entry_str):
        parts = entry_str.split()
        line = int(parts[0])
        word = int(parts[1])
        liklies = [(parts[i], int(parts[i + 1]) / 1000000)
                   for i in range(2, len(parts), 2)]

        return Bantry(line, word, liklies)


class BantryFile():
    def __init__(self, name):
        in_file = open(name)
        bantries = []

        iline = 0
        iline_bantries = []

        for line in in_file:
            e = Bantry.fromstr(line)
            if e.line == iline:
                iline_bantries.append(e)

            elif e.line > iline:
                bantries.append(iline_bantries)
                iline += 1
                while iline < e.line:
                    bantries.append([])
                    iline += 1
                iline_bantries = [e]

            else:
                raise ValueError("Line number can not go down.")

        bantries.append(iline_bantries)
        self.bantries = bantries
        self.num_lines = bantries[-1][-1].line
        in_file.close()

    def get_line_bantires(self, i):
        return self.bantries[i]

class BoxFileReader(object):
    def __init__(self, box_file, scaler_params, batch_sz=20):
        with open(box_file) as box_fp:
            self.box_file = box_fp.read()
        self.scaler = ScalerFactory(scaler_params)
        self.batch_sz = batch_sz
        self.params = scaler_params

    def __call__(self):
        """
        Yield a batch pf size self.batch_sz on each call.

        """
        ret_data = []
        ret_meta = []
        n_samples = 0
        for glp_entry in self.box_file.split('\n'):
            bantry = Bantry.fromSixPack(glp_entry)
            if bantry is None:
                print('Malformed entry : ', glp_entry)
                continue

            scaled_glp = self.scaler(bantry)
            ret_data.append(scaled_glp)
            ret_meta.append((bantry.linenum, bantry.wordnum, bantry.dtop, bantry.dbot))
            n_samples += 1
            if n_samples == self.batch_sz:
                yield (n_samples, ret_meta, ret_data)
                n_samples = 0
                ret_meta = []
                ret_data = []

        yield (n_samples, ret_meta, ret_data)


# ##############################################################################


def process_line_bantires(line_bantries):
    mypaths = Paths()
    iword = line_bantries[0].word

    for entry in line_bantries:
        if entry.word > iword:
            iword = entry.word
            mypaths.simple_update(' ')
        mypaths.update(entry.liklies)

    mypaths.simple_update(' ', True)
    #mypaths.print_top(1)
    return mypaths.paths[0].text()+'\n'


def main():
    import sys

    if len(sys.argv) < 3:
        print('Usage: {} trigram_pickle_file matches_file'.format(sys.argv[0]))
        sys.exit()

    _path.priorer.set_trigram(sys.argv[1])

    bf = BantryFile(sys.argv[2])

    for l in range(bf.num_lines):
        line_bantries = bf.get_line_bantires(l)
        process_line_bantires(line_bantries)

if __name__ == '__main__':
    main()
