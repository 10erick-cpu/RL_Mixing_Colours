# coding=utf-8

"""
Align
=====

**Align** aligns images relative to each other, for example, to correct
shifts in the optical path of a microscope in each channel of a
multi-channel set of images.

References
^^^^^^^^^^

-  Lewis JP. (1995) “Fast normalized cross-correlation.” *Vision
   Interface*, 1-7.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as scind
import scipy.sparse


from scipy.fftpack import fft2, ifft2

from utils.helper_functions.img_utils import load_image, apply_CLAHE_with_grayscale_image

M_MUTUAL_INFORMATION = 'Mutual Information'
M_CROSS_CORRELATION = 'Normalized Cross Correlation'
M_ALL = (M_MUTUAL_INFORMATION, M_CROSS_CORRELATION)

A_SIMILARLY = 'Similarly'
A_SEPARATELY = 'Separately'

C_SAME_SIZE = "Keep size"
C_CROP = "Crop to aligned region"
C_PAD = "Pad images"

C_ALIGN = "Align"

MEASUREMENT_FORMAT = C_ALIGN + "_%sshift_%s"


class Alignment(object):
    def __init__(self, method=M_ALL, crop_mode=C_CROP):
        self.alignment_method = method
        self.crop_mode = crop_mode

    def get_aligned_images(self, base, target, others=[], apply_clahe=False):
        skip_load = not isinstance(base, str)

        base_processed = load_and_preprocess_image(base, skip_load=skip_load, apply_clahe=apply_clahe)
        base_processed = load_and_preprocess_image(base_processed, skip_load=True, apply_clahe=apply_clahe)
        target_processed = load_and_preprocess_image(target, skip_load=skip_load, apply_clahe=apply_clahe,
                                                     subtract_bg=False)
        if skip_load:
            other_loaded = others
        else:
            other_loaded = [load_image(path, True, True) for path in others]

        offsets, shapes = self.run(base_processed, target_processed, other_loaded)
        if (np.abs(np.asarray(offsets)) >= 20).any():
            raise ValueError("Suspicious alignment")
        print("offsets: ", offsets)
        print("shapes: ", shapes)
        if skip_load:
            align_targets = [base, target] + others
        else:
            align_targets = [load_image(path, True, True) for path in [base, target] + others]

        # x and y are swapped in offsets
        result = [self.apply_alignment(elem, "", offsets[idx][1], offsets[idx][0], shape=shapes[idx]) for idx, elem in
                  enumerate(align_targets)]

        return result[0], result[1], result[1:]

    def run(self, base, target, others=[], align_others_similar=True):

        off_x, off_y = self.align(base, target)  # , base > base.mean(), target > target.mean())
        names = [
            base,
            target]
        offsets = [(0, 0), (off_y, off_x)]

        for additional in others:
            names.append(additional)
            if align_others_similar:
                a_off_x, a_off_y = off_x, off_y
            else:
                a_off_x, a_off_y = self.align(names[0], additional)
            offsets.append((a_off_y, a_off_x))

        shapes = [x.shape[:2] for x in names]
        offsets, shapes = self.adjust_offsets(offsets, shapes)
        return offsets, shapes

    def align(self, image1, image2, mask_1=None, mask_2=None):
        '''Align the second image with the first

        Calculate the alignment offset that must be added to indexes in the
        first image to arrive at indexes in the second image.

        Returns the x,y (not i,j) offsets.
        '''

        image1_pixels = image1.astype(np.float32)
        image2_pixels = image2.astype(np.float32)

        if self.alignment_method == M_CROSS_CORRELATION:
            return self.align_cross_correlation(image1_pixels, image2_pixels)
        else:
            if mask_1 is None:
                mask_1 = np.ones_like(image1, dtype=np.bool)
            if mask_2 is None:
                mask_2 = np.ones_like(image2, dtype=np.bool)
            return self.align_mutual_information(image1_pixels, image2_pixels,
                                                 mask_1, mask_2)

    def align_cross_correlation(self, pixels1, pixels2):
        '''Align the second image with the first using max cross-correlation

        returns the x,y offsets to add to image1's indexes to align it with
        image2

        Many of the ideas here are based on the paper, "Fast Normalized
        Cross-Correlation" by J.P. Lewis
        (http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html)
        which is frequently cited when addressing this problem.
        '''
        #
        # TODO: Possibly use all 3 dimensions for color some day
        #
        if pixels1.ndim == 3:
            pixels1 = np.mean(pixels1, 2)
        if pixels2.ndim == 3:
            pixels2 = np.mean(pixels2, 2)
        #
        # We double the size of the image to get a field of zeros
        # for the parts of one image that don't overlap the displaced
        # second image.
        #
        # Since we're going into the frequency domain, if the images are of
        # different sizes, we can make the FFT shape large enough to capture
        # the period of the largest image - the smaller just will have zero
        # amplitude at that frequency.
        #
        s = np.maximum(pixels1.shape, pixels2.shape)
        fshape = s * 2
        #
        # Calculate the # of pixels at a particular point
        #
        i, j = np.mgrid[-s[0]:s[0],
               -s[1]:s[1]]
        unit = np.abs(i * j).astype(float)
        unit[unit < 1] = 1  # keeps from dividing by zero in some places
        #
        # Normalize the pixel values around zero which does not affect the
        # correlation, keeps some of the sums of multiplications from
        # losing precision and precomputes t(x-u,y-v) - t_mean
        #
        pixels1 = pixels1 - np.mean(pixels1)
        pixels2 = pixels2 - np.mean(pixels2)
        #
        # Lewis uses an image, f and a template t. He derives a normalized
        # cross correlation, ncc(u,v) =
        # sum((f(x,y)-f_mean(u,v))*(t(x-u,y-v)-t_mean),x,y) /
        # sqrt(sum((f(x,y)-f_mean(u,v))**2,x,y) * (sum((t(x-u,y-v)-t_mean)**2,x,y)
        #
        # From here, he finds that the numerator term, f_mean(u,v)*(t...) is zero
        # leaving f(x,y)*(t(x-u,y-v)-t_mean) which is a convolution of f
        # by t-t_mean.
        #
        fp1 = fft2(pixels1, fshape)
        fp2 = fft2(pixels2, fshape)
        corr12 = ifft2(fp1 * fp2.conj()).real

        #
        # Use the trick of Lewis here - compute the cumulative sums
        # in a fashion that accounts for the parts that are off the
        # edge of the template.
        #
        # We do this in quadrants:
        # q0 q1
        # q2 q3
        # For the first,
        # q0 is the sum over pixels1[i:,j:] - sum i,j backwards
        # q1 is the sum over pixels1[i:,:j] - sum i backwards, j forwards
        # q2 is the sum over pixels1[:i,j:] - sum i forwards, j backwards
        # q3 is the sum over pixels1[:i,:j] - sum i,j forwards
        #
        # The second is done as above but reflected lr and ud
        #
        p1_si = pixels1.shape[0]
        p1_sj = pixels1.shape[1]
        p1_sum = np.zeros(fshape)
        p1_sum[:p1_si, :p1_sj] = cumsum_quadrant(pixels1, False, False)
        p1_sum[:p1_si, -p1_sj:] = cumsum_quadrant(pixels1, False, True)
        p1_sum[-p1_si:, :p1_sj] = cumsum_quadrant(pixels1, True, False)
        p1_sum[-p1_si:, -p1_sj:] = cumsum_quadrant(pixels1, True, True)
        #
        # Divide the sum over the # of elements summed-over
        #
        p1_mean = p1_sum / unit

        p2_si = pixels2.shape[0]
        p2_sj = pixels2.shape[1]
        p2_sum = np.zeros(fshape)
        p2_sum[:p2_si, :p2_sj] = cumsum_quadrant(pixels2, False, False)
        p2_sum[:p2_si, -p2_sj:] = cumsum_quadrant(pixels2, False, True)
        p2_sum[-p2_si:, :p2_sj] = cumsum_quadrant(pixels2, True, False)
        p2_sum[-p2_si:, -p2_sj:] = cumsum_quadrant(pixels2, True, True)
        p2_sum = np.fliplr(np.flipud(p2_sum))
        p2_mean = p2_sum / unit
        #
        # Once we have the means for u,v, we can calculate the
        # variance-like parts of the equation. We have to multiply
        # the mean^2 by the # of elements being summed-over
        # to account for the mean being summed that many times.
        #
        p1sd = np.sum(pixels1 ** 2) - p1_mean ** 2 * np.product(s)
        p2sd = np.sum(pixels2 ** 2) - p2_mean ** 2 * np.product(s)
        #
        # There's always chance of roundoff error for a zero value
        # resulting in a negative sd, so limit the sds here
        #
        sd = np.sqrt(np.maximum(p1sd * p2sd, 1e-10))
        corrnorm = corr12 / sd
        #
        # There's not much information for points where the standard
        # deviation is less than 1/100 of the maximum. We exclude these
        # from consideration.
        #
        corrnorm[(unit < np.product(s) / 2) &
                 (sd < np.mean(sd) / 100)] = 0
        i, j = np.unravel_index(np.argmax(corrnorm), fshape)
        #
        # Reflect values that fall into the second half
        #
        if i > pixels1.shape[0]:
            i = i - fshape[0]
        if j > pixels1.shape[1]:
            j = j - fshape[1]
        return j, i

    def align_mutual_information(self, pixels1, pixels2, mask1, mask2):
        '''Align the second image with the first using mutual information

        returns the x,y offsets to add to image1's indexes to align it with
        image2

        The algorithm computes the mutual information content of the two
        images, offset by one in each direction (including diagonal) and
        then picks the direction in which there is the most mutual information.
        From there, it tries all offsets again and so on until it reaches
        a local maximum.
        '''
        #
        # TODO: Possibly use all 3 dimensions for color some day
        #
        if pixels1.ndim == 3:
            pixels1 = np.mean(pixels1, 2)
        if pixels2.ndim == 3:
            pixels2 = np.mean(pixels2, 2)

        def mutualinf(x, y, maskx, masky):
            if maskx is not None:
                x = x[maskx & masky]
            if masky is not None:
                y = y[maskx & masky]
            return entropy(x) + entropy(y) - entropy2(x, y)

        maxshape = np.maximum(pixels1.shape, pixels2.shape)
        pixels1 = reshape_image(pixels1, maxshape)
        pixels2 = reshape_image(pixels2, maxshape)
        if mask1 is not None:
            mask1 = reshape_image(mask1, maxshape)
        if mask2 is not None:
            mask2 = reshape_image(mask2, maxshape)

        best = mutualinf(pixels1, pixels2, mask1, mask2)
        i = 0
        j = 0
        while True:
            last_i = i
            last_j = j
            for new_i in range(last_i - 1, last_i + 2):
                for new_j in range(last_j - 1, last_j + 2):
                    if new_i == 0 and new_j == 0:
                        continue
                    p2, p1 = offset_slice(pixels2, pixels1, new_i, new_j)
                    m2, m1 = offset_slice(mask2, mask1, new_i, new_j)
                    info = mutualinf(p1, p2, m1, m2)
                    if info > best:
                        best = info
                        i = new_i
                        j = new_j
            if i == last_i and j == last_j:
                return j, i

    def apply_alignment(self, input_img, output_image_name,
                        off_x, off_y, shape):
        '''Apply an alignment to the input image to result in the output image

        workspace - image set's workspace passed to run

        input_image_name - name of the image to be aligned

        output_image_name - name of the resultant image

        off_x, off_y - offset of the resultant image relative to the original

        shape - shape of the resultant image
        '''

        mask = np.ones_like(input_img, dtype=np.bool)
        pixel_data = input_img
        if pixel_data.ndim == 2:
            output_shape = (shape[0], shape[1], 1)
            planes = [pixel_data]
        else:
            output_shape = (shape[0], shape[1], pixel_data.shape[2])
            planes = [pixel_data[:, :, i] for i in range(pixel_data.shape[2])]
        output_pixels = np.zeros(output_shape, pixel_data.dtype)
        for i, plane in enumerate(planes):
            #
            # Copy the input to the output
            #
            p1, p2 = offset_slice(plane, output_pixels[:, :, i], off_y, off_x)
            p2[:, :] = p1[:, :]
        if pixel_data.ndim == 2:
            output_pixels.shape = output_pixels.shape[:2]
        output_mask = np.zeros(shape, bool)
        p1, p2 = offset_slice(mask, output_mask, off_y, off_x)
        p2[:, :] = p1[:, :]
        if np.all(output_mask):
            output_mask = None
        crop_mask = np.zeros(pixel_data.shape, bool)
        p1, p2 = offset_slice(crop_mask, output_pixels, off_y, off_x)
        p1[:, :] = True
        if np.all(crop_mask):
            crop_mask = None

        output = output_pixels

        return output
        # output_image =  # cpi.Image(output_pixels, mask=output_mask,crop_mask=crop_mask, parent_image=image)
        # workspace.image_set.add(output_image_name, output_image)

    def adjust_offsets(self, offsets, shapes):
        '''Adjust the offsets and shapes for output

        workspace - workspace passed to "run"

        offsets - i,j offsets for each image

        shapes - shapes of the input images

        names - pairs of input / output names

        Based on the crop mode, adjust the offsets and shapes to optimize
        the cropping.
        '''
        offsets = np.array(offsets)
        shapes = np.array(shapes)
        if self.crop_mode == C_CROP:
            # modify the offsets so that all are negative
            max_offset = np.max(offsets, 0)
            offsets = offsets - max_offset[np.newaxis, :]
            #
            # Reduce each shape by the amount chopped off
            #
            shapes += offsets
            #
            # Pick the smallest in each of the dimensions and repeat for all
            #
            shape = np.min(shapes, 0)
            shapes = np.tile(shape, len(shapes))
            shapes.shape = offsets.shape
        elif self.crop_mode == C_PAD:
            #
            # modify the offsets so that they are all positive
            #
            min_offset = np.min(offsets, 0)
            offsets = offsets - min_offset[np.newaxis, :]
            #
            # Expand each shape by the top-left padding
            #
            shapes += offsets
            #
            # Pick the largest in each of the dimensions and repeat for all
            #
            shape = np.max(shapes, 0)
            shapes = np.tile(shape, len(shapes))
            shapes.shape = offsets.shape
        return offsets.tolist(), shapes.tolist()


def offset_slice(pixels1, pixels2, i, j):
    '''Return two sliced arrays where the first slice is offset by i,j
    relative to the second slice.

    '''
    if i < 0:
        height = min(pixels1.shape[0] + i, pixels2.shape[0])
        p1_imin = -i
        p2_imin = 0
    else:
        height = min(pixels1.shape[0], pixels2.shape[0] - i)
        p1_imin = 0
        p2_imin = i
    p1_imax = p1_imin + height
    p2_imax = p2_imin + height
    if j < 0:
        width = min(pixels1.shape[1] + j, pixels2.shape[1])
        p1_jmin = -j
        p2_jmin = 0
    else:
        width = min(pixels1.shape[1], pixels2.shape[1] - j)
        p1_jmin = 0
        p2_jmin = j
    p1_jmax = p1_jmin + width
    p2_jmax = p2_jmin + width

    p1 = pixels1[p1_imin:p1_imax, p1_jmin:p1_jmax]
    p2 = pixels2[p2_imin:p2_imax, p2_jmin:p2_jmax]
    return p1, p2


def cumsum_quadrant(x, i_forwards, j_forwards):
    '''Return the cumulative sum going in the i, then j direction

    x - the matrix to be summed
    i_forwards - sum from 0 to end in the i direction if true
    j_forwards - sum from 0 to end in the j direction if true
    '''
    if i_forwards:
        x = x.cumsum(0)
    else:
        x = np.flipud(np.flipud(x).cumsum(0))
    if j_forwards:
        return x.cumsum(1)
    else:
        return np.fliplr(np.fliplr(x).cumsum(1))


def entropy(x):
    '''The entropy of x as if x is a probability distribution'''
    histogram = scind.histogram(x.astype(float), np.min(x), np.max(x), 256)
    n = np.sum(histogram)
    if n > 0 and np.max(histogram) > 0:
        histogram = histogram[histogram != 0]
        return np.log2(n) - np.sum(histogram * np.log2(histogram)) / n
    else:
        return 0


def entropy2(x, y):
    '''Joint entropy of paired samples X and Y'''
    from centrosome.filter import stretch
    #
    # Bin each image into 256 gray levels
    #
    x = (stretch(x) * 255).astype(int)
    y = (stretch(y) * 255).astype(int)
    #
    # create an image where each pixel with the same X & Y gets
    # the same value
    #
    xy = 256 * x + y
    xy = xy.flatten()
    sparse = scipy.sparse.coo_matrix((np.ones(xy.shape, dtype=np.int32),
                                      (xy, np.zeros(xy.shape, dtype=np.int32))))
    histogram = sparse.toarray()
    n = np.sum(histogram)
    if n > 0 and np.max(histogram) > 0:
        histogram = histogram[histogram > 0]
        return np.log2(n) - np.sum(histogram * np.log2(histogram)) / n
    else:
        return 0


def reshape_image(source, new_shape):
    '''Reshape an image to a larger shape, padding with zeros'''
    if tuple(source.shape) == tuple(new_shape):
        return source

    result = np.zeros(new_shape, source.dtype)
    result[:source.shape[0], :source.shape[1]] = source
    return result


def load_and_preprocess_image(path, skip_load=False, apply_clahe=True, subtract_bg=False):
    if skip_load:
        image = path
    else:
        image = load_image(path, force_grayscale=True, force_8bit=True)
    if apply_clahe:
        image = apply_CLAHE_with_grayscale_image(image)

    if subtract_bg:
        image, bg = rolling_ball_subtract_bg(image, ball_radius=50, light_bg=False)
    return image


if __name__ == '__main__':
    test1 = "/mnt/unix_data/datastorage/raw_input_data/1_input_data/190327/10x/20190221_96wellsJIMT1_CelltrackerandHoechst/non-isolated_test-4x_008/A1--W00001--P00001--Z00000--T00000--DIC.tif"
    test2 = "/mnt/unix_data/datastorage/raw_input_data/1_input_data/190327/10x/20190221_96wellsJIMT1_CelltrackerandHoechst/non-isolated_test-4x_008/A1--W00001--P00001--Z00000--T00000--Hoechst.tif"
    align = Alignment(method=M_MUTUAL_INFORMATION)

    img_1 = load_and_preprocess_image(test1)
    img_2 = load_and_preprocess_image(test2, apply_clahe=False)

    offx, offy = align.align(apply_CLAHE_with_grayscale_image(img_1), img_2)
    print(offx, offy)

    out_target = align.apply_alignment(img_2, "", offx, offy, shape=img_2.shape)

    base, target, other = align.get_aligned_images(test1, test2, [test1])

    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)

    alpha = 0.4

    ax[0].imshow(img_1, cmap="gray")
    ax[0].imshow(out_target, alpha=alpha, cmap="jet")

    ax[1].imshow(base, cmap="gray")
    ax[1].imshow(target, alpha=alpha, cmap="jet")
    plt.show()
