from .helpers import is_file_of_type
from .proglyph import ProGlyph, Space
from .page import Page


class ProcessedPage():
    def __init__(self, filename):
        self.file_glyphs = []
        if is_file_of_type(filename, 'box'):
            self._init_from_box_file(filename)

        elif is_file_of_type(filename, 'image'):
            self._init_from_image(filename)

        self.num_lines = len(self.file_glyphs)

        self.text = ""
        for glyphs_inline in self.file_glyphs:
            for pglyph in glyphs_inline:
                self.text += pglyph.best_char
            self.text += "\n"

    def _init_from_image(self, filename):
        self.page = Page(filename)
        self.page.process()

        for iline, line in enumerate(self.page.lines):
            line_glyphs, iword = [], 0
            for iletter, letter in enumerate(line.letters):
                e = ProGlyph(letter)
                if e.wordnum > iword:
                    iword = e.wordnum
                    line_glyphs.append(Space)
                line_glyphs.append(e)
            self.file_glyphs.append(line_glyphs)

    def _init_from_box_file(self, name):
        in_file = open(name)
        iword, iline, line_glyphs = 0, 0, []

        for box_str in in_file:
            e = ProGlyph(box_str)
            if e.linenum == iline:
                if e.wordnum > iword:
                    iword = e.wordnum
                    line_glyphs.append(Space)
                line_glyphs.append(e)

            elif e.linenum > iline:
                self.file_glyphs.append(line_glyphs)
                iword = 0
                iline += 1
                while iline < e.linenum:
                    self.file_glyphs.append([])
                    iline += 1
                line_glyphs = [e]

            else:
                raise ValueError("Line number can not go down.")

        self.file_glyphs.append(line_glyphs)
        in_file.close()

    def get_line_glyphs(self, i):
        return self.file_glyphs[i]
