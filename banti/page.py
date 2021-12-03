import os
import numpy as np
import statistics as stats

from scipy.ndimage import interpolation as inter
from scipy import ndimage as nd
from PIL import ImageChops
from PIL import Image as im
from PIL import ImageDraw as imd

import logging
from scipy.stats import itemfreq
from .helpers import img_to_bin_arr, bin_arr_to_rgb_img
from .conncomp import get_conn_comp

logger = logging.getLogger(__name__)
logi = logger.info
logd = logger.debug

coarse_limit, coarse_jump = 6, .75
fine_jump = 1/8
deskew_reduce_width_to = 800


class Page():
    def __init__(self, path):
        self.path = path
        self.orig_img = self.trim(im.open(path))
        self.img = self.orig_img

        self.orig_imgarr = img_to_bin_arr(self.orig_img)
        self.imgarr = self.orig_imgarr
        self.wd, self.ht = self.orig_img.size

        self.angle, self.fft, self.best_harmonic = [None] * 3
        self.hist, self.gauss_hist, self.d_gauss_hist = [None] * 3
        self.wmorpharr = None
        self.words_hist, self.gauss_words_hist, self.d_gauss_words_hist = [None] * 3

        self.base_lines, self.top_lines, self.line_sep = [None] * 3
        self.num_lines = None

    def change_ext(self, ext):
        base, old_ext = os.path.splitext(self.path)
        return os.path.join(base + ext)

    def process(self):
        self._correct_skew()
        self._filter_noise()
        self._calc_hist()
        self._find_baselines()
        self._separate_lines()

    def trim(self, img):
        bg = im.new(img.mode, img.size, img.getpixel((0, 0)))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if bbox:
            trimmed_img = img.crop(bbox)
        else:
            trimmed_img = img
        return trimmed_img

    def _filter_noise(self, ):
        logi("Filtering Noise")
        self.imgarr = nd.median_filter(self.imgarr, size=3)

    def _correct_skew(self, ):
        logi("Correcting Skew")
        coarse_angles = np.arange(-coarse_limit, coarse_limit+coarse_jump, coarse_jump)

        def find_scores(arr, angles):
            scores = np.zeros_like(angles)

            for i, a in enumerate(angles):
                data = inter.rotate(arr, a, reshape=0, order=0)
                hist = np.sum(data, axis=1)
                scores[i] = np.sum((hist[1:] - hist[:-1]) ** 2)

            return scores

        imgarr = self.imgarr
        temp_sizes_str = "Resizing before skew correction: " + str(imgarr.shape)
        while imgarr.shape[-1] > deskew_reduce_width_to:
            imgarr = imgarr[::2, ::2]
            temp_sizes_str += " -> " + str(imgarr.shape)
        logi(temp_sizes_str)

        coarse_scores = find_scores(imgarr, coarse_angles)
        logi("Coarse Angles, Scores: {}".format(list(zip(coarse_angles, coarse_scores))))

        # Find a range to do a fine search over.
        top_a, top_b = sorted(np.argpartition(-coarse_scores, 2)[:2])
        if top_a > 0:
            fine_min = coarse_angles[top_a-1]
        else:
            fine_min = coarse_angles[0] - coarse_jump
        if top_b + 1 < len(coarse_angles):
            fine_max = coarse_angles[top_b+1]
        else:
            fine_max = coarse_angles[-1] + coarse_jump

        fine_angles = np.arange(fine_min, fine_max, fine_jump)
        fine_angles = np.setdiff1d(fine_angles, coarse_angles)
        fine_scores = find_scores(imgarr, fine_angles)
        logi("Fine Angles, Scores: {}".format(list(zip(fine_angles, fine_scores))))

        if coarse_scores.max() >= fine_scores.max():
            self.angle = coarse_angles[np.argmax(coarse_scores)]
        else:
            self.angle = fine_angles[np.argmax(fine_scores)]

        self.imgarr = inter.rotate(self.imgarr, self.angle, reshape=0, order=0)
        self.ht, self.wd = self.imgarr.shape
        self.img = bin_arr_to_rgb_img(self.imgarr)
        logi("Best Angle: {:.3f}".format(self.angle))

    def _calc_hist(self, ):
        logi("In Calc Hist")
        self.hist = np.sum(self.imgarr, axis=1).astype('float')

        logi("Finding Fourier Transform")
        self.fft = abs(np.fft.rfft(self.hist - np.mean(self.hist)))
        max_harm = int(np.argmax(self.fft))
        self.best_harmonic = self.ht // (1 + max_harm)
        assert max_harm > 0
        logi("Best harmonic found at {}".format(self.best_harmonic))

        self.gauss_hist = nd.filters.gaussian_filter1d(
            self.hist, self.best_harmonic / 16,                # Equiv ht / 5
            mode='constant', cval=0, truncate=2.0)
        self.d_gauss_hist = nd.filters.convolve(self.gauss_hist, [-1, 0, 1])

        logi("Closing horizontally with median width and finding its histograms.")
        self.wmorpharr = nd.binary_closing(self.imgarr,
                                          structure=np.ones((3, self.best_harmonic//3)))
        self.words_hist = np.sum(self.wmorpharr, axis=1).astype("float")
        self.gauss_words_hist = nd.filters.gaussian_filter1d(
            self.words_hist, self.best_harmonic / 16,
            mode='constant', cval=0, truncate=2.0)
        self.d_gauss_words_hist = nd.filters.convolve(self.gauss_words_hist, [-1, 0, 1])

    def _find_baselines(self, ):
        logi("Finding baselines.")
        d_hist = self.d_gauss_hist
        gmaxval = np.max(d_hist)
        maxloc = np.argmax(d_hist)
        peakthresh = gmaxval / 10.0
        zerothresh = gmaxval / 50.0
        inpeak = False
        min_dist_in_peak = self.best_harmonic / 2.0
        self.base_lines = []
        logi("Max Hist: {:.2f} Peakthresh: {:.2f} Zerothresh: {:.2f} Min Dist in Peak: {:.2f}"
             "".format(gmaxval, peakthresh, zerothresh, min_dist_in_peak))

        for irow, val in enumerate(d_hist):
            if not inpeak:
                if val > peakthresh:
                    inpeak = True
                    maxval = val
                    maxloc = irow
                    mintosearch = irow + min_dist_in_peak
                    logd('\tTransition to in-peak:{} mintosearch:{} '
                         ''.format(irow, mintosearch))
                        # accept no zeros between i and i+mintosearch

            else:  # in peak, look for max
                if val > maxval:
                    maxval = val
                    maxloc = irow
                    mintosearch = irow + min_dist_in_peak
                    logd('\tMoved mintosearch:{}'.format(mintosearch))

                elif irow > mintosearch and val <= zerothresh:
                    # leave peak and save the last baseline found
                    inpeak = False
                    logd('\nLeaving peak with baseline at:{}'.format(maxloc))
                    self.base_lines.append(maxloc)

        if inpeak:
            self.base_lines.append(maxloc)
            logd('\nLast baseline at:{}'.format(maxloc))

        self.num_lines = len(self.base_lines)
        logi("Number of lines found {}".format(self.num_lines))
        logd("Base_lines:{}".format(self.base_lines))

    def _separate_lines(self, ):
        logi("Finding Toplines and line separations.")
        self.top_lines = []
        try:
            self.line_sep = [np.where(self.gauss_hist[0:self.base_lines[0]] == 0)[0][-1]]
        except IndexError:
            self.line_sep = [0]

        for ibase, base in enumerate(self.base_lines):
            # Find top lines
            frm = 0 if ibase == 0 else self.line_sep[ibase]
            top_at = np.argmin(self.d_gauss_words_hist[frm:base])
            self.top_lines.append(frm + top_at)

            # Find line separation
            to = self.base_lines[ibase + 1] if ibase + 1 < self.num_lines else self.ht
            sep_at = np.argmin(self.gauss_hist[base + 1:to])
            self.line_sep.append(base + 1 + sep_at)
            logd("\t{:2d}) Baselines ({:4d}, {:4d}) Topline:{:4d} Separation:{:4d}"
                 "".format(ibase, frm, base, self.top_lines[-1], self.line_sep[-1]))

        self.lines = [Line(self, iline) for iline in range(self.num_lines)]

    def get_info(self):
        ret = (
            "\nImage: {} "
            "\nHeight, Width: {}, {}"
            "\nShapes: Image Array:{} Word_Morphed:{}"
            "\nRotated by angle: {:.2f}"
            "\nBest Harmonic: {}"
            "\nLengths: hist:{} gauss_hist:{} d_gauss_hist:{} FFT:{}"
            "\nNumber of lines:{} "
            "".format(
                self.path,
                self.ht, self.wd,
                self.imgarr.shape, self.wmorpharr.shape,
                self.angle, self.best_harmonic,
                len(self.hist), len(self.gauss_hist), len(self.d_gauss_hist),
                len(self.fft),
                self.num_lines))

        ret += "\nLine From  Top Base Till Words"
        for line in range(self.num_lines):
            ret += "\n{:3d}: {:4d} {:4d} {:4d} {:4d}  {:3d}".format(line,
                self.line_sep[line], self.top_lines[line],
                self.base_lines[line], self.line_sep[line+1],
                self.lines[line].num_words)

        return ret

    def get_hists_info(self):
        return "Line Hist GHist DGHist" + \
               "\n".join(
            ["{:4d} {:7.2f} {:7.2f} {:7.2f}".format(l, i, j, k)
             for l, i, j, k in zip(
                range(self.ht), self.hist, self.gauss_hist, self.d_gauss_hist)])

    def get_image_with_hist(self, width, orig_img=None, hist=None):
        if hist is None:
            hist = self.gauss_hist
        hist = width * hist / np.max(hist)
        appendage = np.full((self.ht, width), 255, dtype='uint8')
        for row, count in enumerate(hist.astype('int')):
            appendage[row, :count] = 0

        appendage = im.fromarray(appendage)
        appended_img = im.new('RGB', (self.wd + width, self.ht))
        appended_img.paste(appendage, (self.wd, 0))

        if orig_img is None:
            orig_img = self.img

        appended_img.paste(orig_img, (0, 0))
        return appended_img

    def _draw_lines(self, target):
        width = target.size[0]
        draw = imd.Draw(target)

        def draw_lines(locations, col):
            for loc in locations:
                draw.line((0, loc, width, loc), fill=col, width=1)

        draw_lines(self.top_lines, (200, 200, 0))
        draw_lines(self.base_lines, (0, 255, 0))
        draw_lines(self.line_sep, (0, 0, 255))

        return target

    def get_image_with_hist_and_lines(self, width):
        appended_img = self.get_image_with_hist(width)
        return self._draw_lines(appended_img)

    def save_image_with_hist_and_lines(self, width):
        target_name = self.change_ext(".png")
        appended_img = self.get_image_with_hist(width)
        self._draw_lines(appended_img).save(target_name)
        print("Saving:", target_name)

    def save_words_image_with_hist_and_lines(self, width):
        target_name = self.change_ext("_words.png")
        appended_img = self.get_image_with_hist(width,
                                                bin_arr_to_rgb_img(self.wmorpharr),
                                                hist=self.gauss_words_hist)
        self._draw_lines(appended_img).save(target_name)
        print("Saving:", target_name)

    def save_letters_img(self):
        arr = np.zeros_like(self.imgarr, "uint8")
        curr_col = 0

        for l in self.lines:
            arr[l.top:l.bot] = l.labelled_img + curr_col
            arr[l.top:l.bot][l.labelled_img==0] = 0
            curr_col += l.num_letters
            arr[l.top, ::2] = 252
            arr[l.topline, ::2] = 253
            arr[l.baseline, ::2] = 254

            for w in l.word_comps:
                try:
                    top, bot = w.y + l.top, w.y2 + l.top
                    arr[top, w.x:w.x2] = 255
                    arr[bot-1, w.x:w.x2] = 255
                    arr[top:bot, w.x] = 255
                    arr[top:bot, w.x2-1] = 255
                except IndexError:
                    pass
        arr[self.line_sep[-1], ::2] = 253

        target_name = self.change_ext("_letters.png")
        img = im.fromarray(arr, "P")
        palette = np.random.randint(256, size=(256 * 3)).tolist()
        palette[:3] = 255, 255, 255
        palette[-12:] = 0, 0, 255, 128, 128, 128, 128, 128, 128, 180, 180, 180
        img.putpalette(palette)
        img.save(target_name)
        print("Saving: ", target_name)


class Line():
    def __init__(self, page, num):
        logi("Initializing line : {}".format(num))
        self.page = page
        self.linenum = num
        self.top, self.bot = page.line_sep[num:num+2]
        self.ht = self.bot - self.top
        self.topline = page.top_lines[num]
        self.baseline = page.base_lines[num]
        self.arr = page.imgarr[self.top:self.bot]
        self.xht = self.baseline - self.topline
        self.num_words = 0
        self.num_letters = 0

        self.find_letters()
        self.find_words()
        self.align_letters_to_words()
        self.sanity_check()
        logi(str(self))

    def find_words(self):
        logi("Finding words.")
        brick_ht = self.median_ht // 3 + 1
        brick_wd = self.median_wd // 2 + 1
        horz_buffer = np.zeros((self.ht, brick_wd))
        # print("Line ", self.linenum, (self.ht, brick_wd), (self.median_ht, self.median_wd))
        logi("Dialating vertically by {}. Closing Horz by {}".format(brick_ht, brick_wd))

        # Dilate Vertically
        self.word_closed_arr = nd.binary_dilation(self.arr, np.ones((brick_ht, 1)))

        # Close Horizontal Gaps (Slightly involved process)
        self.word_closed_arr = np.hstack((horz_buffer, self.word_closed_arr, horz_buffer))
        nd.binary_closing(self.word_closed_arr,
                          np.ones((1, brick_wd)),
                          output=self.word_closed_arr)
        self.word_closed_arr = self.word_closed_arr[:, brick_wd:-brick_wd]

        self.word_comps, self.word_labelled_img = get_conn_comp(self.word_closed_arr)
        if False:
            self.word_comps = [c for c in self.word_comps if (c.ht > self.xht / 8 and c.wd > self.xht / 8)]

        self.num_words = len(self.word_comps)

    def find_letters(self):
        self.letters, self.labelled_img = get_conn_comp(self.arr)
        hts, wds = zip(*[(c.ht, c.wd) for c in self.letters])
        self.median_ht = int(stats.median(hts))
        self.median_wd = int(stats.median(wds))

    def sanity_check(self):
        self.word_support = np.zeros(self.page.wd).astype("uint")
        for w in self.word_comps:
            self.word_support[w.x:w.x2] += 1
        logi(str(itemfreq(self.word_support)[2:]))

    def __str__(self):
        return "Line:{}" \
               "\nFrom:{} Top:{} Base:{} To:{}" \
               "\nHt:{} Xht:{}" \
               "\nWords:{} Letters:{}".format(
            self.linenum, self.top, self.topline, self.baseline, self.bot,
            self.ht, self.xht, self.num_words, self.num_letters)

    def align_letters_to_words(self):
        for letter in self.letters:
            letter.linenum = self.linenum
            letter.baseline = self.baseline - self.top
            letter.topline = self.topline - self.top
            for wordnum, word in enumerate(self.word_comps):
                if letter in word:
                    letter.wordnum = wordnum
                    break
            else:
                for wordnum, word in enumerate(self.word_comps):
                    if word.has_center_of(letter):
                        letter.wordnum = wordnum
                else:
                    raise ValueError("Could not find word for {}".format(letter))
