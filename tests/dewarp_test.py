import sys
from PIL import Image
from banti.dewarp import DeWarper
from banti.helpers import get_ext_changer, img_to_bin_arr, bin_arr_to_img

img_name = sys.argv[1]
change_ext = get_ext_changer(img_name)

def save(img, ext):
    fname = change_ext(ext)
    print("Saving", fname)
    img.save(fname)

img = Image.open(img_name)
pix = img_to_bin_arr(img)

wrpr = DeWarper(pix)
wrpr.build_model()
save(wrpr.get_info_image(), "_dots.png")

wrpr.apply_vert_disparity()
wrpr.apply_horz_disparity()

save(bin_arr_to_img(wrpr.target_vert), "_vert.tif")
save(bin_arr_to_img(wrpr.target_horz), "_horz.tif")