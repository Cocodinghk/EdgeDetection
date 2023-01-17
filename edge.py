# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
 
# Prewitt
def getPrewitt(image):
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv.filter2D(image, cv.CV_16S, kernelx)
    y = cv.filter2D(image, cv.CV_16S, kernely)
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    res = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv.bitwise_not(res, res)
    plt.subplot(232), plt.imshow(res, cmap=plt.cm.gray),plt.title('Prewitt'), plt.axis('off')

    
# Sobel
def getSobel(image):
    x = cv.Sobel(image, cv.CV_16S, 1, 0)  
    y = cv.Sobel(image, cv.CV_16S, 0, 1) 
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    res = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv.bitwise_not(res, res)
    plt.subplot(233), plt.imshow(res, cmap=plt.cm.gray),plt.title('Sobel'), plt.axis('off')

# Canny
def getCanny(image):
    res = cv.Canny(image, 40, 80)
    cv.bitwise_not(res, res)
    plt.subplot(234), plt.imshow(res, cmap=plt.cm.gray),plt.title('Canny'), plt.axis('off')

def find_neighbors(x, ksize, s, out_h, out_w):
    """ Get sliding windows using numpy's stride tricks. """
    in_c, in_h, in_w = x.shape
    shape = (out_h, out_w, in_c, ksize, ksize)
    itemsize = x.itemsize
    strides = (
        s * in_w * itemsize,
        s * itemsize,
        in_w * in_h * itemsize,
        in_w * itemsize,
        itemsize
    )
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

def refine_flow(flow, mag, ksize):
    """ 
    Refine edge tangent flow 
    """
    _, h, w = flow.shape

    # do padding
    p = ksize // 2
    flow = np.pad(flow, ((0, 0), (p, p), (p, p)))
    mag = np.pad(mag, ((0, 0), (p, p), (p, p)))

    # neighbors of each tangent vector in each window
    flow_neighbors = find_neighbors(flow, ksize,
                                    s=1, out_h=h, out_w=w)

    # centural tangent vector in each window
    flow_me = flow_neighbors[..., ksize // 2, ksize // 2]
    flow_me = flow_me[..., np.newaxis, np.newaxis]

    # compute dot
    dots = np.sum(flow_neighbors * flow_me, axis=2,
                  keepdims=True)

    # compute phi
    phi = np.where(dots > 0, 1, -1)

    # compute wd, weight of direction
    wd = np.abs(dots)

    # compute wm, weight of magnitude
    mag_neighbors = find_neighbors(mag, ksize,
                                   s=1, out_h=h, out_w=w)
    mag_me = mag_neighbors[..., ksize // 2, ksize // 2]
    mag_me = mag_me[..., np.newaxis, np.newaxis]
    wm = (1 + np.tanh(mag_neighbors - mag_me)) / 2

    # compute ws, spatial weight
    ws = np.ones_like(wm)
    x, y = np.meshgrid(np.arange(ksize), np.arange(ksize))
    cx, cy = ksize // 2, ksize // 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)[np.newaxis, ...]
    ws[:, :, dist >= ksize // 2] = 0

    # update flow
    flow = np.sum(phi * flow_neighbors * ws * wm * wd,
                  axis=(3, 4))
    flow = np.transpose(flow, axes=(2, 0, 1))

    # normalize flow
    norm = np.sqrt(np.sum(flow ** 2, axis=0))
    mask = norm != 0
    flow[:, mask] /= norm[mask]

    return flow

def gauss(x, sigma):
    """ Returns guass filter's weight with respect to
    x and sigma. """
    return np.exp(-(x ** 2) / (2 * sigma ** 2)) / \
           (np.sqrt(2 * np.pi) * sigma)


def make_gauss_filter(sigma, threshold=0.001):
    """ Returns a symetric gauss 1-d filter, max weight
    of output filter is less than threshold."""
    i = 0
    while gauss(i, sigma) >= threshold:
        i = i + 1

    return gauss(np.arange(-i, i + 1),
                 sigma).astype('float32')


def shrink_array(a, center, width):
    """ Shrink an 1-D array with respect to center
    and width. """
    return a[-width + center: width + 1 + center]


def detect_edge(img, flow, thresh, sigma_c, rho,
                sigma_m, tau):
    """ Detect edge on input image based of edge tangent
    flow, the following code is messy ... """
    h, w = img.shape
    # normalize input image
    img = img / 255.0

    # create two gauss filter
    gauss_c = make_gauss_filter(sigma_c, threshold=thresh)
    gauss_s = make_gauss_filter(sigma_c * 1.6,
                                threshold=thresh)

    # shrink filter to the same size
    w_gauss_c, w_gauss_s = len(gauss_c) // 2, len(gauss_s) // 2
    w_fdog = min(w_gauss_c, w_gauss_s)
    gauss_c = shrink_array(gauss_c, w_gauss_c, w_fdog)
    gauss_s = shrink_array(gauss_s, w_gauss_s, w_fdog)

    # do padding because some vectorized operations
    # may accross the boundary of image
    img = np.pad(img,
                 ((w_fdog, w_fdog), (w_fdog, w_fdog)))

    # start coords of each pixel (shifted by width of filter)
    sx, sy = np.meshgrid(np.arange(w), np.arange(h))
    start = np.concatenate((sx[np.newaxis, ...],
                            sy[np.newaxis, ...]), axis=0) + w_fdog

    # steps of each pixel
    steps = np.arange(-w_fdog, w_fdog + 1).reshape(
        -1, 1, 1, 1)
    steps = np.repeat(steps, repeats=2, axis=1)

    # rotate flow to get gradient
    grad = np.empty_like(flow)
    grad[0, ...] = flow[1, ...]
    grad[1, ...] = -flow[0, ...]

    # take steps along the gradient
    xy = start + (steps * grad)
    ixy = np.round(xy).astype('int32')
    ix, iy = np.split(ixy, indices_or_sections=2, axis=1)
    ix = ix.reshape(2 * w_fdog + 1, h, w)
    iy = iy.reshape(2 * w_fdog + 1, h, w)

    # neighbors of each pixel along the gradient
    neighbors = img[iy, ix]

    # apply dog filter in gradient's direction
    gauss_c = gauss_c.reshape(2 * w_fdog + 1, 1, 1)
    img_gauss_c = np.sum(gauss_c * neighbors, axis=0) \
                  / np.sum(gauss_c)

    gauss_s = gauss_s.reshape(2 * w_fdog + 1, 1, 1)
    img_gauss_s = np.sum(gauss_s * neighbors, axis=0) \
                  / np.sum(gauss_s)
    img_fdog = img_gauss_c - rho * img_gauss_s

    # remove those pixels with zero gradient
    zero_grad_mask = np.logical_and(
        grad[0, ...] == 0, grad[1, ...] == 0)
    img_fdog[zero_grad_mask] = np.max(img_fdog)

    # make gauss filter along tangent vector
    gauss_m = make_gauss_filter(sigma_m)
    w_gauss_m = len(gauss_m) // 2

    # initialize with a negative weight for coding's convenience
    edge = -gauss_m[w_gauss_m] * img_fdog
    weight_acc = np.full_like(img_fdog,
                              fill_value=-gauss_m[w_gauss_m])

    # do padding
    img_fdog = np.pad(img_fdog,
                      ((w_gauss_m, w_gauss_m), (w_gauss_m, w_gauss_m)))
    zero_grad_mask = np.pad(zero_grad_mask,
                            ((w_gauss_m, w_gauss_m), (w_gauss_m, w_gauss_m)))
    flow = np.pad(flow, ((0, 0),
                         (w_gauss_m, w_gauss_m), (w_gauss_m, w_gauss_m)))

    # start coords of each pixcel
    sx, sy = np.meshgrid(np.arange(w), np.arange(h))
    sx += w_gauss_m;
    sy += w_gauss_m

    # forward mask, indicate whether a pixel need to keep
    # going along tangent vector or not
    forward_mask = np.full(shape=(h, w), fill_value=True,
                           dtype='bool')

    # convert dtype from integer to float for accumulating
    # steps along tangent vector
    x = sx.astype('float32')
    y = sy.astype('float32')
    ix, iy = np.round(x).astype('int32'), \
             np.round(y).astype('int32')

    # start
    for i in range(w_gauss_m + 1):
        # get neighbors of each pixel w.r.t its coordinate
        neighbors = img_fdog[iy, ix]

        # multiply weight, ignore those pixels who stopped
        weight = gauss_m[w_gauss_m + i]
        edge[forward_mask] += (neighbors * weight)[forward_mask]
        weight_acc[forward_mask] += weight

        # take a step along tangent vector w.r.t coordinate
        x += flow[0, iy, ix]
        y += flow[1, iy, ix]

        # update coordinates
        ix, iy = np.round(x).astype('int32'), \
                 np.round(y).astype('int32')

        # update each pixels' status
        none_zero_mask = np.logical_not(
            zero_grad_mask[iy, ix])
        forward_mask = np.logical_and(
            forward_mask, none_zero_mask)

    # going along the reversed tangent vector
    forward_mask = np.full(shape=(h, w), fill_value=True,
                           dtype='bool')
    x = sx.astype('float32')
    y = sy.astype('float32')
    ix, iy = np.round(x).astype('int32'), \
             np.round(y).astype('int32')
    for i in range(w_gauss_m + 1):
        neighbor = img_fdog[iy, ix]

        weight = gauss_m[w_gauss_m - i]
        edge[forward_mask] += (neighbor * weight)[forward_mask]
        weight_acc[forward_mask] += weight

        # take a step
        x -= flow[0, iy, ix]
        y -= flow[1, iy, ix]
        ix, iy = np.round(x).astype('int32'), \
                 np.round(y).astype('int32')

        none_zero_mask = np.logical_not(
            zero_grad_mask[iy, ix])
        forward_mask = np.logical_and(
            forward_mask, none_zero_mask)

    # postprocess
    edge /= weight_acc
    edge[edge > 0] = 1
    edge[edge <= 0] = 1 + np.tanh(edge[edge <= 0])
    edge = (edge - np.min(edge)) / \
           (np.max(edge) - np.min(edge))

    # binarize
    edge[edge < tau] = 0
    edge[edge >= tau] = 255
    return edge.astype('uint8')


def initialze_flow(img, sobel_size):
    """ Initialize edge tangent flow, contains the
    following steps:
        (1) normalize input image
        (2) compute gradient using sobel operator
        (3) compute gradient magnitude
        (4) normalize gradient and magnitude
        (5) rotate gradient to get tanget vector
    """
    img = cv.normalize(img, dst=None, alpha=0.0, beta=1.0,
                       norm_type=cv.NORM_MINMAX, dtype=cv.CV_32FC1)

    # compute gradient using sobel and magtitude
    grad_x = cv.Sobel(img, cv.CV_32FC1, 1, 0,
                      ksize=sobel_size)
    grad_y = cv.Sobel(img, cv.CV_32FC1, 0, 1,
                      ksize=sobel_size)
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # normalize gradient
    mask = mag != 0
    grad_x[mask] /= mag[mask]
    grad_y[mask] /= mag[mask]

    # normalize magnitude
    mag = cv.normalize(mag, dst=None, alpha=0.0, beta=1.0,
                       norm_type=cv.NORM_MINMAX)

    # rotate gradient and get tangent vector
    flow_x, flow_y = -grad_y, grad_x

    # expand dimension in axis=0 for vectorizing
    flow = np.concatenate((flow_x[np.newaxis, ...],
                           flow_y[np.newaxis, ...]), axis=0)
    mag = mag[np.newaxis, ...]

    return flow, mag

def nms(img, flow):
    res = np.ones(img.shape[0], img.shape[1], dtype=np.float64)
    ksize = 2
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            max_ki = i, max_kj = j
            vn = {flow[0][i][j], flow[1][i][j]}
            if (vn[0] == 0 & vn[1] == 0): 
                continue
            dx = i, dy = j
            for s in range(-ksize, ksize):
                x = dx + vn[0]*s
                y = dy + vn[1]*s
                if (x < 0 | y < 0 | x >= img.shape[0] | y >= img.shape[1]):
                    continue
                xi = min(max(x, 0.0), img.shape[0] - 1.0)
                yi = min(max(y, 0.0), img.shape[1] - 1.0)
                val = img[int(xi)][int(yi)]
                minval = img[max_ki][max_kj]
                if (val < minval):
                    max_ki = xi
                    max_kj = yi
            if (max_ki == i & max_kj == j):
                res[i][j] = img[i][j]
    return res            


def run(img, mode, sobel_size=5, etf_iter=4, etf_size=7,
        fdog_iter=3, thresh=0.001, sigma_c=1.0, rho=0.997,
        sigma_m=3.0, tau=0.99):
    """
    Running coherent line drawing on input image.

    Parameters:
    ----------
    - img : ndarray
        Input image, with shape = (h, w, c).

    - mode : int, values 0 or 1
        mode = 0 : FDoG without NMS
        mode = 1 : FDoG with NMS
    
    - sobel_size : int
        Size of sobel filter, sobel filter will be used to compute
        gradient.

    - etf_iter : int
        Iteration times of refining edge tangent flow.

    - etf_size : int
        Size of etf filter.

    - fdog_iter : int
        Iteration times of applying fdog on input image.

    - thresh : float
        Threshold of guass filter's value, this is not an important
        parameter.

    - sigma_c : float
        Standard variance of one gaussian filter of dog filter,
        another's standard variance will be set to 1.6 * sigma_c.

    - rho : float
        Dog = gauss_c - rho * gauss_s.

    - sigma_m : float
        Standard variance of gaussian filter.

    - tau : float
        Threshold of edge map.

    Returns:
    -------
    - edge : ndarray
        Edge map of input image, data type is float32 and pixel's
        range is clipped to [0, 255].
    """
    # convert to gray image
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # initialize edge tangent flow
    flow, mag = initialze_flow(img, sobel_size)

    # refine edge tangent flow
    for i in range(etf_iter):
        flow = refine_flow(flow, mag, ksize=etf_size)

    # do fdog
    for i in range(fdog_iter):
        edge = detect_edge(img, flow, thresh=thresh,
                           sigma_c=sigma_c, rho=rho, sigma_m=sigma_m, tau=tau)
        img[edge == 0] = 0
        img = cv.GaussianBlur(img, ksize=(3, 3), sigmaX=0, sigmaY=0)

    if (mode == 1):
        img = nms(img, flow)
    
    return detect_edge(img, flow, thresh=thresh,
                       sigma_c=sigma_c, rho=rho, sigma_m=sigma_m, tau=tau)

def getFdog(image):
    res = run(img=image, mode=0, sobel_size=5,
            etf_iter=2, etf_size=7,
            fdog_iter=2, sigma_c=1.0, rho=0.997, sigma_m=3.0,
            tau=0.999
        )
    res = cv.cvtColor(res, cv.COLOR_GRAY2RGB)
    #cv.bitwise_not(res, res)
    plt.subplot(235),plt.imshow(res),plt.title('Fdog'), plt.axis('off')
    
def getFdogwithNMS(image):
    res = run(img=image, mode=1, sobel_size=5,
            etf_iter=2, etf_size=7,
            fdog_iter=2, sigma_c=1.0, rho=0.997, sigma_m=3.0,
            tau=0.85
        )
    res = cv.cvtColor(res, cv.COLOR_GRAY2RGB)
    #cv.bitwise_not(res, res)
    plt.subplot(236),plt.imshow(res),plt.title('Fdog_with_NMS'), plt.axis('off')
    
if __name__ == "__main__":
    # load the image
    tests = ['data/fig1.jpg',
        'data/fig2.jpg',
        'data/fig3.jpg',        
        'data/fig4.jpg',  
    ]
    for test in tests:
        img = cv.imread(os.path.join(test))
        img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.subplot(231),plt.imshow(img_RGB),plt.title('Original'), plt.axis('off')
        grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        getPrewitt(grayImage)
        getSobel(grayImage)
        getCanny(grayImage)
        getFdog(img_RGB)
        # get FdogwithNMS(img_RGB)
        plt.show()
        cv.destroyAllWindows()