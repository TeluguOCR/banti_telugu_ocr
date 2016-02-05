import sys

from banti.proglyph import ProGlyph
from banti.processedpage import ProcessedPage
from banti.scaler import ScalerFactory


input_file_name = sys.argv[1] if len(sys.argv) > 1 else "sample_images/praasa.box"
scaler_prms_file = sys.argv[2] if len(sys.argv) > 2 else "scalings/relative48.scl"

ProGlyph.scaler = ScalerFactory(scaler_prms_file)
pro_page = ProcessedPage(input_file_name)

for linenum in range(pro_page.num_lines):
    print('*' * 60)
    line_glyphs = pro_page.get_line_glyphs(linenum)
    for glyph in line_glyphs:
        print(glyph.scaled)