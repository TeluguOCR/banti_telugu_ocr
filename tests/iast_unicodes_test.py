import sys
from banti.iast_unicodes import LabelToUnicodeConverter

labellings = sys.argv[1] if len(sys.argv) > 1 else "labellings/alphacodes.lbl"
idx2chr = LabelToUnicodeConverter(labellings)

for idx in idx2chr.indices:
    print(idx, idx2chr.iast(idx), idx2chr[idx])