import sys
from banti.page import Page

image_name = sys.argv[1]
page = Page(image_name)
page.process()
print(page.get_info())
page.save_image_with_hist_and_lines(100)
page.save_words_image_with_hist_and_lines(100)
page.save_letters_img()