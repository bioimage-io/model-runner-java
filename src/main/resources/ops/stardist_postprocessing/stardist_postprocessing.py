###
# #%L
# Use deep learning frameworks from Java in an agnostic and isolated way.
# %%
# Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
# %%
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# #L%
###
from time import time
t = time()
import numpy as np
print("impot numpy star " + str(time() - t))
t = time()
from csbdeep.utils import axes_check_and_normalize, _raise
print("impot csbd.utils star " + str(time() - t))
t = time()
import math
print("impot math star NEW " + str(time() - t))
t = time()
import sys
import re
import os
ENV_DIR = os.path.dirname(sys.executable if os.name == 'nt' else os.path.dirname(sys.executable))
SITE_PACKAGES_DIR = os.path.join(ENV_DIR, "lib")
if os.name != 'nt':
	if os.path.isdir(SITE_PACKAGES_DIR + "/python" + str(sys.version_info[0]) + "." + str(sys.version_info[1])):
		SITE_PACKAGES_DIR = os.path.join(SITE_PACKAGES_DIR, "python" + str(sys.version_info[0]) + "." + str(sys.version_info[1]))
	else:
		PATTERN = r"python3\.\d{1,2}$"
		matching_files = [file for file in os.listdir(SITE_PACKAGES_DIR) if re.match(PATTERN, file)]
		SITE_PACKAGES_DIR = os.path.join(SITE_PACKAGES_DIR, matching_files[0])
STARDIST_DIR = os.path.join(SITE_PACKAGES_DIR, "site-packages", "stardist", "lib")
sys.path.append(os.path.join(STARDIST_DIR))
from stardist2d import c_non_max_suppression_inds
print("impot nms star " + str(time() - t))
t = time()
POLYGON_DIR = os.path.join(SITE_PACKAGES_DIR, "site-packages", "skimage", "draw")
sys.path.append(POLYGON_DIR)
from skimage.draw import polygon
print("impot geometry star " + str(time() - t))



def stardist_postprocessing(raw_output, prob_thresh, nms_thresh, n_classes=None,
                             grid=(1,1), b=2, channel=2, n_rays=32, n_dim=2, axes_net='YXC'):
    prob = raw_output[0, :, :, 0:1]
    dist = raw_output[0, :, :, 1:]
    prob, dist = _prep(prob, dist)
    inds = _ind_prob_thresh(prob, prob_thresh, b=b)
    proba = prob[inds].copy()
    dista = dist[inds].copy()
    _points = np.stack(np.where(inds), axis=1)
    pointsa = (_points * np.array(grid).reshape((1,len(grid))))

    if n_classes is not None:
        p = np.moveaxis(raw_output[2], channel, -1)
        prob_classa = p[inds].copy()

    proba = np.asarray(proba)
    dista = np.asarray(dista).reshape((-1, n_rays))
    pointsa = np.asarray(pointsa).reshape((-1, n_dim))

    grid_dict = dict(zip(axes_net.replace('C', ''), grid))
    resizer = StarDistPadAndCropResizer(grid=grid_dict, pad={"X": (0, 0), "Y": (0, 0), "C": (0, 0)},
                                        padded_shape={"X": prob.shape[1], "Y": prob.shape[0]})

    idx = resizer.filter_points(raw_output[0].ndim, pointsa, axes_net)
    prob = proba[idx]
    dist = dista[idx]
    points = pointsa[idx]

    # last "yield" is the actual output that would have been "return"ed if this was a regular function
    if n_classes is not None:
        prob_classa = np.asarray(prob_classa).reshape((-1, n_classes + 1))
        prob_class = prob_classa[idx]
    else:
        prob_classa = None
        prob_class = None

    nms_kwargs = {'verbose': False}
    res_instances = instances_from_prediction(raw_output[0, :, :, 0].shape, prob, dist,
                                              prob_thresh, nms_thresh, grid,
                                              points=points,
                                              prob_class=prob_class,
                                              scale=None,
                                              return_labels=True,
                                              overlap_label=None,
                                              **nms_kwargs)
    return res_instances[0].astype("float32")


def _prep(prob, dist, channel=2):
    prob = np.take(prob, 0, axis=channel)
    dist = np.moveaxis(dist, channel, -1)
    dist = np.maximum(1e-3, dist)
    return prob, dist


def _ind_prob_thresh(prob, prob_thresh, b=2):
    if b is not None and np.isscalar(b):
        b = ((b, b),) * prob.ndim

    ind_thresh = prob > prob_thresh
    if b is not None:
        _ind_thresh = np.zeros_like(ind_thresh)
        ss = tuple(slice(_bs[0] if _bs[0] > 0 else None,
                         -_bs[1] if _bs[1] > 0 else None) for _bs in b)
        _ind_thresh[ss] = True
        ind_thresh &= _ind_thresh
    return ind_thresh


def instances_from_prediction(img_shape, prob, dist, prob_thresh, nms_thresh, grid, points=None, prob_class=None,
                              overlap_label=None, return_labels=True, scale=None, **nms_kwargs):
    """
    if points is None     -> dense prediction
    if points is not None -> sparse prediction

    if prob_class is None     -> single class prediction
    if prob_class is not None -> multi  class prediction
    """
    if overlap_label is not None: raise NotImplementedError("overlap_label not supported for 2D yet!")

    # sparse prediction
    if points is not None:
        points, probi, disti, indsi = non_maximum_suppression_sparse(dist, prob, points, nms_thresh=nms_thresh,
                                                                     **nms_kwargs)
        if prob_class is not None:
            prob_class = prob_class[indsi]

    # dense prediction
    else:
        points, probi, disti = non_maximum_suppression(dist, prob, grid=grid,
                                                       prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs)
        if prob_class is not None:
            inds = tuple(p // g for p, g in zip(points.T, grid))
            prob_class = prob_class[inds]

    if scale is not None:
        # need to undo the scaling given by the scale dict, e.g. scale = dict(X=0.5,Y=0.5):
        #   1. re-scale points (origins of polygons)
        #   2. re-scale coordinates (computed from distances) of (zero-origin) polygons
        if not (isinstance(scale, dict) and 'X' in scale and 'Y' in scale):
            raise ValueError("scale must be a dictionary with entries for 'X' and 'Y'")
        rescale = (1 / scale['Y'], 1 / scale['X'])
        points = points * np.array(rescale).reshape(1, 2)
    else:
        rescale = (1, 1)

    if return_labels:
        labels = polygons_to_label(disti, points, prob=probi, shape=img_shape, scale_dist=rescale)
    else:
        labels = None

    coord = dist_to_coord(disti, points, scale_dist=rescale)
    res_dict = dict(coord=coord, points=points, prob=probi)

    # multi class prediction
    if prob_class is not None:
        prob_class = np.asarray(prob_class)
        class_id = np.argmax(prob_class, axis=-1)
        res_dict.update(dict(class_prob=prob_class, class_id=class_id))

    return labels, res_dict


def non_maximum_suppression(dist, prob, grid=(1,1), b=2, nms_thresh=0.5, prob_thresh=0.5,
                            use_bbox=True, use_kdtree=True, verbose=False):
    """Non-Maximum-Supression of 2D polygons

    Retains only polygons whose overlap is smaller than nms_thresh

    dist.shape = (Ny,Nx, n_rays)
    prob.shape = (Ny,Nx)

    returns the retained points, probabilities, and distances:

    points, prob, dist = non_maximum_suppression(dist, prob, ....

    """

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    assert prob.ndim == 2 and dist.ndim == 3  and prob.shape == dist.shape[:2]
    dist = np.asarray(dist)
    prob = np.asarray(prob)
    n_rays = dist.shape[-1]

    grid = _normalize_grid(grid,2)

    # mask = prob > prob_thresh
    # if b is not None and b > 0:
    #     _mask = np.zeros_like(mask)
    #     _mask[b:-b,b:-b] = True
    #     mask &= _mask

    mask = _ind_prob_thresh(prob, prob_thresh, b)
    points = np.stack(np.where(mask), axis=1)

    dist   = dist[mask]
    scores = prob[mask]

    # sort scores descendingly
    ind = np.argsort(scores)[::-1]
    dist   = dist[ind]
    scores = scores[ind]
    points = points[ind]

    points = (points * np.array(grid).reshape((1,2)))

    if verbose:
        t = time()

    inds = non_maximum_suppression_inds(dist, points.astype(np.int32, copy=False), scores=scores,
                                        use_bbox=use_bbox, use_kdtree=use_kdtree,
                                        thresh=nms_thresh, verbose=verbose)

    if verbose:
        print("keeping %s/%s polygons" % (np.count_nonzero(inds), len(inds)))
        print("NMS took %.4f s" % (time() - t))

    return points[inds], scores[inds], dist[inds]


def non_maximum_suppression_sparse(dist, prob, points, b=2, nms_thresh=0.5,
                                   use_bbox=True, use_kdtree = True, verbose=False):
    """Non-Maximum-Supression of 2D polygons from a list of dists, probs (scores), and points

    Retains only polyhedra whose overlap is smaller than nms_thresh

    dist.shape = (n_polys, n_rays)
    prob.shape = (n_polys,)
    points.shape = (n_polys,2)

    returns the retained instances

    (pointsi, probi, disti, indsi)

    with
    pointsi = points[indsi] ...

    """

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    dist = np.asarray(dist)
    prob = np.asarray(prob)
    points = np.asarray(points)
    n_rays = dist.shape[-1]

    assert dist.ndim == 2 and prob.ndim == 1 and points.ndim == 2 and \
        points.shape[-1]==2 and len(prob) == len(dist) == len(points)

    verbose and print("predicting instances with nms_thresh = {nms_thresh}".format(nms_thresh=nms_thresh), flush=True)

    inds_original = np.arange(len(prob))
    _sorted = np.argsort(prob)[::-1]
    probi = prob[_sorted]
    disti = dist[_sorted]
    pointsi = points[_sorted]
    inds_original = inds_original[_sorted]

    if verbose:
        print("non-maximum suppression...")
        t = time()

    inds = non_maximum_suppression_inds(disti, pointsi, scores=probi, thresh=nms_thresh, use_kdtree = use_kdtree, verbose=verbose)

    if verbose:
        print("keeping %s/%s polyhedra" % (np.count_nonzero(inds), len(inds)))
        print("NMS took %.4f s" % (time() - t))

    return pointsi[inds], probi[inds], disti[inds], inds_original[inds]


def non_maximum_suppression_inds(dist, points, scores, thresh=0.5, use_bbox=True, use_kdtree = True, verbose=1):
    """
    Applies non maximum supression to ray-convex polygons given by dists and points
    sorted by scores and IoU threshold

    P1 will suppress P2, if IoU(P1,P2) > thresh

    with IoU(P1,P2) = Ainter(P1,P2) / min(A(P1),A(P2))

    i.e. the smaller thresh, the more polygons will be supressed

    dist.shape = (n_poly, n_rays)
    point.shape = (n_poly, 2)
    score.shape = (n_poly,)

    returns indices of selected polygons
    """


    assert dist.ndim == 2
    assert points.ndim == 2

    n_poly = dist.shape[0]

    if scores is None:
        scores = np.ones(n_poly)

    assert len(scores) == n_poly
    assert points.shape[0] == n_poly

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    inds = c_non_max_suppression_inds(_prep(dist,  np.float32),
                                      _prep(points, np.float32),
                                      int(use_kdtree),
                                      int(use_bbox),
                                      int(verbose),
                                      np.float32(thresh))

    return inds


def _ind_prob_thresh(prob, prob_thresh, b=2):
    if b is not None and np.isscalar(b):
        b = ((b,b),)*prob.ndim

    ind_thresh = prob > prob_thresh
    if b is not None:
        _ind_thresh = np.zeros_like(ind_thresh)
        ss = tuple(slice(_bs[0] if _bs[0]>0 else None,
                         -_bs[1] if _bs[1]>0 else None)  for _bs in b)
        _ind_thresh[ss] = True
        ind_thresh &= _ind_thresh
    return ind_thresh


class StarDistPadAndCropResizer():

    # TODO: check correctness
    def __init__(self, grid, mode='reflect', pad=None, padded_shape=None, **kwargs):
        assert isinstance(grid, dict)
        self.mode = mode
        self.grid = grid
        self.pad = pad
        self.padded_shape = padded_shape
        self.kwargs = kwargs

    def before(self, x, axes, axes_div_by):
        assert all(a%g==0 for g,a in zip((self.grid.get(a,1) for a in axes), axes_div_by))
        axes = axes_check_and_normalize(axes,x.ndim)
        def _split(v):
            return 0, v # only pad at the end
        self.pad = {
            a : _split((div_n-s%div_n)%div_n)
            for a, div_n, s in zip(axes, axes_div_by, x.shape)
        }
        x_pad = np.pad(x, tuple(self.pad[a] for a in axes), mode=self.mode, **self.kwargs)
        self.padded_shape = dict(zip(axes,x_pad.shape))
        if 'C' in self.padded_shape: del self.padded_shape['C']
        return x_pad


    def after(self, x, axes):
        # axes can include 'C', which may not have been present in before()
        axes = axes_check_and_normalize(axes,x.ndim)
        assert all(s_pad == s * g for s,s_pad,g in zip(x.shape,
                                                       (self.padded_shape.get(a,_s) for a,_s in zip(axes,x.shape)),
                                                       (self.grid.get(a,1) for a in axes)))
        # print(self.padded_shape)
        # print(self.pad)
        # print(self.grid)
        crop = tuple (
            slice(0, -(math.floor(p[1]/g)) if p[1]>=g else None)
            for p,g in zip((self.pad.get(a,(0,0)) for a in axes),(self.grid.get(a,1) for a in axes))
        )
        # print(crop)
        return x[crop]

    def filter_points(self, ndim, points, axes):
        """ returns indices of points inside crop region """
        assert points.ndim==2
        axes = axes_check_and_normalize(axes,ndim)

        bounds = np.array(tuple(self.padded_shape[a]-self.pad[a][1] for a in axes if a.lower() in ('z','y','x')))
        idx = np.where(np.all(points< bounds, 1))
        return idx


def dist_to_coord(dist, points, scale_dist=(1,1)):
    """convert from polar to cartesian coordinates for a list of distances and center points
    dist.shape   = (n_polys, n_rays)
    points.shape = (n_polys, 2)
    len(scale_dist) = 2
    return coord.shape = (n_polys,2,n_rays)
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    assert dist.ndim==2 and points.ndim==2 and len(dist)==len(points) \
        and points.shape[1]==2 and len(scale_dist)==2
    n_rays = dist.shape[1]
    phis = ray_angles(n_rays)
    coord = (dist[:,np.newaxis]*np.array([np.sin(phis),np.cos(phis)])).astype(np.float32)
    coord *= np.asarray(scale_dist).reshape(1,2,1)    
    coord += points[...,np.newaxis] 
    return coord


def ray_angles(n_rays=32):
    return np.linspace(0,2*np.pi,n_rays,endpoint=False)


def polygons_to_label(dist, points, shape, prob=None, thr=-np.inf, scale_dist=(1,1)):
    """converts distances and center points to label image

    dist.shape   = (n_polys, n_rays)
    points.shape = (n_polys, 2)

    label ids will be consecutive and adhere to the order given
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    prob = np.inf*np.ones(len(points)) if prob is None else np.asarray(prob)

    assert dist.ndim==2 and points.ndim==2 and len(dist)==len(points)
    assert len(points)==len(prob) and points.shape[1]==2 and prob.ndim==1

    n_rays = dist.shape[1]

    ind = prob>thr
    points = points[ind]
    dist = dist[ind]
    prob = prob[ind]

    ind = np.argsort(prob, kind='stable')
    points = points[ind]
    dist = dist[ind]

    coord = dist_to_coord(dist, points, scale_dist=scale_dist)

    return polygons_to_label_coord(coord, shape=shape, labels=ind)


def polygons_to_label_coord(coord, shape, labels=None):
    """renders polygons to image of given shape

    coord.shape   = (n_polys, n_rays)
    """
    coord = np.asarray(coord)
    if labels is None: labels = np.arange(len(coord))

    _check_label_array(labels, "labels")
    assert coord.ndim==3 and coord.shape[1]==2 and len(coord)==len(labels)

    lbl = np.zeros(shape,np.int32)

    for i,c in zip(labels,coord):
        rr,cc = polygon(*c, shape)
        lbl[rr,cc] = i+1

    return lbl

def _check_label_array(y, name=None, check_sequential=False):
    err = ValueError("{label} must be an array of {integers}.".format(
        label = 'labels' if name is None else name,
        integers = ('sequential ' if check_sequential else '') + 'non-negative integers',
    ))
    is_array_of_integers(y) or _raise(err)
    if len(y) == 0:
        return True
    if check_sequential:
        label_are_sequential(y) or _raise(err)
    else:
        y.min() >= 0 or _raise(err)
    return True


def is_array_of_integers(y):
    return isinstance(y,np.ndarray) and np.issubdtype(y.dtype, np.integer)


def label_are_sequential(y):
    """ returns true if y has only sequential labels from 1... """
    labels = np.unique(y)
    return (set(labels)-{0}) == set(range(1,1+labels.max()))


def _normalize_grid(grid,n):
    try:
        grid = tuple(grid)
        (len(grid) == n and
         all(map(np.isscalar,grid)) and
         all(map(_is_power_of_2,grid))) or _raise(TypeError())
        return tuple(int(g) for g in grid)
    except (TypeError, AssertionError):
        raise ValueError("grid = {grid} must be a list/tuple of length {n} with values that are power of 2".format(grid=grid, n=n))
