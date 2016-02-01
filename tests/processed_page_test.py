import sys

from banti.proglyph import ProGlyph
from banti.processedpage import ProcessedPage
from banti.scaler import ScalerFactory


banti_file_name = sys.argv[1] if len(sys.argv) > 1 else "sample_images/praasa.box"
scaler_prms_file = sys.argv[2] if len(sys.argv) > 2 else "scalings/relative48.scl"

ProGlyph.scaler = ScalerFactory(scaler_prms_file)
bf = ProcessedPage(banti_file_name)

for linenum in range(bf.num_lines):
    print('*' * 60)
    line_bantries = bf.get_line_bantires(linenum)
    for bantry in line_bantries:
        print(bantry.scaled)