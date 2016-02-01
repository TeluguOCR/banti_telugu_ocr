from .proglyph import ProGlyph, Space
from .page import Page


class ProcessedPage():
    def __init__(self, filename):
        if filename.endswith(".box"):
            self._init_from_box_file(filename)

        elif filename.lower().endswith(".tif"):
            self._init_from_tif(filename)

    def _init_from_tif(self, filename):
        self.page = Page(filename)

    def _init_from_box_file(self, name):
        in_file = open(name)
        self.file_bantries = []

        iword, iline = 0, 0
        line_bantries = []

        for line in in_file:
            e = ProGlyph(line)
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
        self.num_lines = len(self.file_bantries)

        self.text = ""
        for bantries_inline in self.file_bantries:
            for bantree in bantries_inline:
                self.text += bantree.best_char
            self.text += "\n"

        in_file.close()

    def get_line_bantires(self, i):
        return self.file_bantries[i]