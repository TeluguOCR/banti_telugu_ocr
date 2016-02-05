from PIL import Image as im
import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from banti.conncomp import get_conn_comp
logi = print
LINE_HT_THRESH = 10
LINE_WD_THRESH = .6

def fit_quad(x, y):
    X = np.array((np.ones_like(x), x, x ** 2)).T
    return np.linalg.lstsq(X, y)[0]


def predict_quad(beta, x):
    X = np.array((np.ones_like(x), x, x ** 2)).T
    return X.dot(beta.T)


def morph_sequence(pix, *param):
    for oc, wd, ht in param:
        logi(" Performing Morph : ", oc, wd, ht)
        structure = np.ones((ht, wd))
        if oc == "c":
            pix = binary_closing(pix, structure)
        elif oc == "o":
            pix = binary_opening(pix, structure)
    return pix


def get_mean_verticals(c):
    xs = c.x + np.arange(c.wd)
    ys = c.y + np.sum(c.pix * np.arange(c.ht)[:, None], axis=0) / np.sum(c.pix, axis=0)
    return np.array((xs, ys))


class DeWarper():
    def __init__(self, pix, sampling=1, redfactor=1, minlines=6, maxdist=50):
        self.sampling = sampling
        self.redfactor = redfactor
        self.minlines = minlines
        self.maxdist = maxdist

        self.pix = pix
        self.ht, self.wd = pix.shape
        self.nx = (self.wd + 2 * sampling - 2)
        self.ny = (self.ht + 2 * sampling - 2)

        self.grid_xs = np.arange(0, self.nx, self.sampling)  # M
        self.grid_ys = np.arange(0, self.ny, self.sampling)  # Lg

        logi("Pix shape, range ", pix.shape, pix.max(), pix.mean(), pix.min())
        logi("grid_xs", len(self.grid_xs), self.grid_xs[:3], self.grid_xs[-3:])
        logi("grid_ys", len(self.grid_ys), self.grid_ys[:3], self.grid_ys[-3:])

    def build_model(self):
        # Morph to lines
        self.morphpix = morph_sequence(self.pix, ("o", 1, 3), ("c", 15, 1),
                                                 ("o", 15, 1), ("c",30, 1))
        # Get Word/Line components and filter
        comps, _ = get_conn_comp(self.morphpix)  # Use 8 conn
        comps = [c for c in comps if c.ht > LINE_HT_THRESH]
        max_wd = max(c.wd for c in comps)
        comps = [c for c in comps if c.wd > max_wd * LINE_WD_THRESH]

        # Get the mean line
        self.midlines = [get_mean_verticals(c) for c in comps]

        # Remove Short Lines
        # self.midlines = [a for a in midlines if (a[0].max() - a[0].min() >= .4 * max_len)]

        self.find_vert_disparity()
        self.find_horz_disparity()

    def find_vert_disparity(self):
        betas = np.zeros((len(self.midlines), 3))  # L' x 3
        for i, midline in enumerate(self.midlines):
            xs, ys = midline
            betas[i] = fit_quad(xs, ys)

        logi(betas.shape)
        beta2s = betas[:, -1]
        median_beta2 = np.median(beta2s)
        median_var_beta2 = np.median(np.abs(beta2s - median_beta2))
        betas = betas[np.abs(beta2s - median_beta2) < 7 * median_var_beta2]
        print(beta2s, median_beta2, median_var_beta2)
        # L x 3
        self.median_abs_beta2 = np.median(np.abs(betas[:-1]))

        fitted_ys_on_gridx = predict_quad(betas, self.grid_xs).T  # Need L x M
        fitted_ys_on_gx_min = fitted_ys_on_gridx.min(axis=-1)  # L vector
        flat_y_order = fitted_ys_on_gx_min.argsort()
        fitted_ys_on_gx_min = fitted_ys_on_gx_min[flat_y_order]
        fitted_ys_on_gridx = fitted_ys_on_gridx[flat_y_order]
        diff_ys = fitted_ys_on_gridx - fitted_ys_on_gx_min[:, None]

        # ptaa0 - grid_xs, fitted_ys_on_gridx (unfiltered)
        # 1 - grid_xs, fitted_ys_on_gridx
        # 2 - grid_xs, fitted_ys_on_gridx (sorted)
        # 3 - grid_xs, diff_ys
        # 4 - fitted_ys_on_gx_min, diff_ys (L, L x M)
        # 5 - grid_ys, fitted_diff_ys

        # In each column fit diff_ys on fitted_ys_on_gx_min
        gammas = np.zeros((diff_ys.shape[1], 3))  # M x 3
        for i, diff_y_i in enumerate(diff_ys.T):
            gammas[i] = fit_quad(fitted_ys_on_gx_min, diff_y_i)

        fitted_diff_ys = predict_quad(gammas, self.grid_ys)   # Lg x M

        self.samp_vert_dispar = fitted_diff_ys
        self.fitted_ys_on_gridx = fitted_ys_on_gridx
        print("sampled vertical disparity: ", fitted_diff_ys.shape)

    def find_horz_disparity(self):
        left_ends, rigt_ends = [], []
        for midline in self.midlines:  # L
            xs, ys = midline
            left_ends.append((xs[0], ys[0]))
            rigt_ends.append((xs[-1], ys[-1]))

        left_ends = np.array(left_ends)  # L x 2
        rigt_ends = np.array(rigt_ends)
        order = np.argsort(left_ends[:, 1])  # L
        left_ends = left_ends[order]
        rigt_ends = rigt_ends[order]

        half = len(self.midlines) // 2
        lines1_lens = rigt_ends[:half, 0] - left_ends[:half, 0]
        lines2_lens = rigt_ends[half:, 0] - left_ends[half:, 0]
        HZ_DISP_LINE_LEN_THRES = .9
        left_ends = np.vstack((left_ends[:half][lines1_lens > HZ_DISP_LINE_LEN_THRES * lines1_lens.max()],
                               left_ends[half:][lines2_lens > HZ_DISP_LINE_LEN_THRES * lines2_lens.max()]))
        rigt_ends = np.vstack((rigt_ends[:half][lines1_lens > HZ_DISP_LINE_LEN_THRES * lines1_lens.max()],
                               rigt_ends[half:][lines2_lens > HZ_DISP_LINE_LEN_THRES * lines2_lens.max()]))
        self.left_ends, self.rigt_ends = left_ends, rigt_ends

        # Fit quadratics
        left_beta = fit_quad(left_ends[:,1], left_ends[:,0])      # Predict x from y
        rigt_beta = fit_quad(rigt_ends[:,1], rigt_ends[:,0])

        left_fitted_xs = predict_quad(left_beta, self.grid_ys)
        rigt_fitted_xs = predict_quad(rigt_beta, self.grid_ys)
        self.left_fitted_xs, self.rigt_fitted_xs = left_fitted_xs, rigt_fitted_xs

        left_minx, left_maxx = left_fitted_xs.min(), left_fitted_xs.max()
        rigt_minx, rigt_maxx = rigt_fitted_xs.min(), rigt_fitted_xs.max()

        left_ref,  rigt_ref = (left_minx, rigt_minx) if True else (left_maxx, rigt_maxx)

        left_fitted_diff_xs = left_ref - left_fitted_xs
        rigt_fitted_diff_xs = rigt_ref - rigt_fitted_xs

        samp_horz_dispar = left_fitted_diff_xs[:, None] + \
                           ((self.grid_xs-left_minx)/(rigt_minx-left_minx)) * \
                           (rigt_fitted_diff_xs - left_fitted_diff_xs)[:, None]
        #L x M = L x .   +   . x M * L x .

        self.samp_horz_dispar = samp_horz_dispar
        print("Sample Horiz Disparity", samp_horz_dispar.shape)

    def apply_horz_disparity(self):
        print("Applying Horz Disp")
        self.target_horz = np.zeros_like(self.pix)
        dispar = self.samp_horz_dispar.astype(int)

        for x in range(self.wd):
            for y in range(self.ht):
                newx = x + dispar[y, x]
                if 0 <= newx < self.wd:
                    self.target_horz[y, x] = self.target_vert[y, newx]

    def apply_vert_disparity(self):
        print("Applying Vert Disp")
        self.target_vert = np.zeros_like(self.pix)
        dispar = self.samp_vert_dispar.astype(int)

        for x in range(self.wd):
            for y in range(self.ht):
                newy = y + dispar[y, x]
                if 0 <= newy < self.ht:
                    self.target_vert[y, x] = self.pix[newy, x]

    def apply_disparities(self):
        self.apply_vert_disparity()
        self.apply_horz_disparity()

    def get_info_image(self):
        pix = self.pix.copy().astype("uint8")
        pix += 2 * self.morphpix

        for xs, ys in self.midlines:
            pix[ys.astype(int), xs.astype(int)] = 4

        for fitted_y in self.fitted_ys_on_gridx.astype(int):
            pix[fitted_y, self.grid_xs] = 5

        for (xl, yl), (xr, yr), yg, xfl, xfr in zip(self.left_ends, self.rigt_ends, self.grid_ys,
                                                    self.left_fitted_xs, self.rigt_fitted_xs):
            pix[yl-3:yl+3, xl:xl+6] = 6
            pix[yr-3:yr+3, xr-6:xr] = 6
            try:
                pix[yg-3:yg+3, xfl:xfl+6] = 7
                pix[yg-3:yg+3, xfr-6:xfr] = 7
            except:
                pass

        img = im.fromarray(pix, "P")
        palette = np.random.randint(256, size=(256 * 3)).tolist()
        palette[:24] = (255, 255, 255,  128, 128, 255, 127, 127, 0, 0, 0, 0,
                        200, 0, 0, 50, 250, 50,
                        0, 0, 255, 0, 100, 200)
        img.putpalette(palette)
        return img