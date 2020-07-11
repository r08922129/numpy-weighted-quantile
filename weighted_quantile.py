#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections.abc
import functools
import re
import sys
import warnings

import numpy as np
import numpy.core.numeric as _nx
from numpy.core import transpose
from numpy.core.numeric import (
    ones, zeros, arange, concatenate, array, asarray, asanyarray, empty,
    ndarray, around, floor, ceil, take, dot, where, intp,
    integer, isscalar, absolute
    )
from numpy.core.umath import (
    pi, add, arctan2, frompyfunc, cos, less_equal, sqrt, sin,
    mod, exp, not_equal, subtract
    )
from numpy.core.fromnumeric import (
    ravel, nonzero, partition, mean, any, sum
    )
from numpy.core.numerictypes import typecodes
from numpy.core.overrides import set_module
from numpy.core import overrides
from numpy.core.function_base import add_newdoc
from numpy.lib.twodim_base import diag
from numpy.core.multiarray import (
    _insert, add_docstring, bincount, normalize_axis_index, _monotonicity,
    interp as compiled_interp, interp_complex as compiled_interp_complex
    )
from numpy.core.umath import _add_newdoc_ufunc as add_newdoc_ufunc

import builtins
import os


# In[20]:


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')
def _weighted_ureduce(a, func, w, **kwargs):
    """
    Internal Function.
    Call `func` with `a` as first argument swapping the axes to use extended
    axis on functions that don't support it natively.
    Returns result and a.shape with axis dims set to 1.
    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    func : callable
        Reduction function capable of receiving a single axis argument.
        It is called with `a` as first argument followed by `kwargs`.
    w : array_like
        Has the sample shape with a.shape.
    kwargs : keyword arguments
        additional keyword arguments to pass to `func`.
    Returns
    -------
    result : tuple
        Result of func(a, **kwargs) and a.shape with axis dims set to 1
        which can be used to reshape the result to the same shape a ufunc with
        keepdims=True would produce.
    """        
    axis = kwargs.get('axis', None)
    if axis is not None:
        keepdim = list(a.shape)
        nd = a.ndim
        axis = _nx.normalize_axis_tuple(axis, nd)

        for ax in axis:
            keepdim[ax] = 1

        if len(axis) == 1:
            kwargs['axis'] = axis[0]
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            # swap axis that should not be reduced to front
            for i, s in enumerate(sorted(keep)):
                a = a.swapaxes(i, s)
                w = w.swapaxes(i, s)
            # merge reduced axis
            a = a.reshape(a.shape[:nkeep] + (-1,))
            w = w.reshape(a.shape[:nkeep] + (-1,))
            kwargs['axis'] = -1
        keepdim = tuple(keepdim)
    else:
        keepdim = (1,) * a.ndim

    r = func(a, w, **kwargs)
    return r, keepdim
def _weighted_quantile_dispatcher(a, q, w, axis=None, out=None, overwrite_input=None,
                         interpolation=None, keepdims=None):
    return (a, q, out)

@array_function_dispatch(_weighted_quantile_dispatcher)
def weighted_quantile(a, q, w, axis=None, out=None, # new w
             overwrite_input=False, interpolation='linear', keepdims=False):
    """
    Compute the q-th weighted quantile of the data along the specified axis.
    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    w : array_like
        The weights of sample. It must have the same shape with a or be a 1d array for broadcast.
        When it's a 1d array, axis should be an integer and w.size == a.shape[axis].
        If all elements in w are the same, this function works like np.quantile.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The
        default is to compute the quantile(s) along a flattened
        version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input
        `a` after this function completes is undefined.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to
        use when the desired quantile lies between two data points
        ``i < j``:
            * linear: ``i + (j - i) * fraction``, where ``fraction``
              is the fractional part of the index surrounded by ``i``
              and ``j``.
            * lower: ``i``.
            * higher: ``j``.
            * nearest: ``i`` or ``j``, whichever is nearest.
            * midpoint: ``(i + j) / 2``.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.
    Returns
    -------
    quantile : scalar or ndarray
        If `q` is a single quantile and `axis=None`, then the result
        is a scalar. If multiple quantiles are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.
    See Also
    --------
    mean
    percentile : equivalent to quantile, but with q in the range [0, 100].
    median : equivalent to ``quantile(..., 0.5)``
    nanquantile
    Notes
    -----
    Given a vector ``V`` of length ``N``, the q-th quantile of
    ``V`` is the value ``q`` of the way from the minimum to the
    maximum in a sorted copy of ``V``. The values and distances of
    the two nearest neighbors as well as the `interpolation` parameter
    will determine the quantile if the normalized ranking does not
    match the location of ``q`` exactly. This function is the same as
    the median if ``q=0.5``, the same as the minimum if ``q=0.0`` and the
    same as the maximum if ``q=1.0``.
    Examples
    --------
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.quantile(a, 0.5)
    3.5
    >>> np.quantile(a, 0.5, axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.quantile(a, 0.5, axis=1)
    array([7.,  2.])
    >>> np.quantile(a, 0.5, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = np.quantile(a, 0.5, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.quantile(a, 0.5, axis=0, out=out)
    array([6.5, 4.5, 2.5])
    >>> m
    array([6.5, 4.5, 2.5])
    >>> b = a.copy()
    >>> np.quantile(b, 0.5, axis=1, overwrite_input=True)
    array([7.,  2.])
    >>> assert not np.all(a == b)
    """
    a = np.asanyarray(a)
    q = np.asanyarray(q)
    w = np.asanyarray(w)
    
    if w.shape!=a.shape:
        if w.ndim != 1:
            raise TypeError(
                "1D weights expected when shapes of a and weights differ.")
        if w.shape[0] != a.shape[axis]:
            raise ValueError(
                "Length of weights not compatible with specified axis.")
        w = np.broadcast_to(w, (a.ndim-1)*(1,) + w.shape)
        w = w.swapaxes(-1, axis)*np.ones(a.shape)

    if not _quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")
    if not _weight_is_valid(w):
        raise ValueError("All the weights must be > 0")
    return _weighted_quantile_unchecked(
        a, q, w, axis, out, overwrite_input, interpolation, keepdims)


def _weighted_quantile_unchecked(a, q, w, axis=None, out=None, overwrite_input=False,
                        interpolation='linear', keepdims=False):
    """Assumes that q is in [0, 1], and is an ndarray"""
    r, k = _weighted_ureduce(a, func=_weighted_quantile_ureduce_func,w=w, q=q, axis=axis, out=out,
                    overwrite_input=overwrite_input,
                    interpolation=interpolation)
    if keepdims:
        return r.reshape(q.shape + k)
    else:
        return r


def _quantile_is_valid(q):
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if q[i] < 0.0 or q[i] > 1.0:
                return False
    else:
        # faster than any()
        if np.count_nonzero(q < 0.0) or np.count_nonzero(q > 1.0):
            return False
    return True

def _weight_is_valid(w):
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if w.ndim == 1 and w.size < 10:
        for i in range(w.size):
            if w[i] <= 0.0:
                return False
    else:
        # faster than any()
        if np.count_nonzero(w <= 0.0):
            return False
    
    return True


def _weighted_lerp(a, b, sa,sb,qsn, out=None):
    """ Linearly interpolate from a to b by a factor of sk 
    The weighted quantile formulation is [X_k + (X_{k+1}-X_k)*(q*S_n-S_k)/(S_{k+1}-S_k)]
    Parameters
    ----------
    a : X_k
    b : X_{k+1}
    sa : S_k
    sb : S_b
    qsn : q*Sn
    """
    diff_b_a = subtract(b, a)
    # asanyarray is a stop-gap until gh-13105
    t = (qsn-sa)/(sb-sa)
    lerp_interpolation = asanyarray(add(a, diff_b_a*t, out=out))
    subtract(b, diff_b_a * (1 - t), out=lerp_interpolation, where=t>=0.5)
    if lerp_interpolation.ndim == 0 and out is None:
        lerp_interpolation = lerp_interpolation[()]  # unpack 0d arrays
    return lerp_interpolation

def _find_weighted_index(sk,qsn,interpolation='linear'):
    
    dim = sk.shape # (N, d1, d2,..., dk)
    Nx = dim[0]
    _sk = sk.reshape(dim[0],-1) # (N,-1)
    _qsn = qsn.reshape(qsn.shape[0],-1) # (q,-1)
    indices = []
    
    for  j in range(_qsn.shape[1]):
        k = 0
        for i in range(_qsn.shape[0]):
            qsn_j = _qsn[i,j]
            sk_j = _sk[:,j]
            # find Sk
            while(True):
                if qsn_j==sk_j[k]:
                    indices.append(k)
                    break
                elif sk_j[k] < qsn_j < sk_j[k+1]:
                    if interpolation == 'lower':
                        indices.append(k)
                    elif interpolation == 'higher':
                        indices.append(k+1)
                    elif interpolation == 'midpoint':
                        indices.append(0.5*(2*k+1))
                    elif interpolation == 'nearest':
                        # To get the same result with np.quantile(), test if k%2==0 and |q*S_n-S_k| == |q*S_n*S_{k+1}|
                        if qsn_j-sk_j[k] < sk_j[k+1]-qsn_j or (k%2==0 and qsn_j-sk_j[k] ==sk_j[k+1]-qsn_j):
                            indices.append(k)
                        else:
                            indices.append(k+1)
                    elif interpolation == 'linear':
                        # just let the indices to be float temporally
                        indices.append(0.5*(2*k+1))
                    else:
                        raise ValueError(
                            "interpolation can only be 'linear', 'lower' 'higher', "
                            "'midpoint', or 'nearest'")
                    break
                k = k+1
    indices = np.asanyarray(indices).reshape(dim[1:]+(qsn.shape[0],))
    indices = np.moveaxis(indices, -1, 0)
    return indices
    
def _weighted_quantile_ureduce_func(a, w, q, axis=None, out=None, overwrite_input=False,
                           interpolation='linear', keepdims=False):

    # ufuncs cause 0d array results to decay to scalars (see gh-13105), which
    # makes them problematic for __setitem__ and attribute access. As a
    # workaround, we call this on the result of every ufunc on a possibly-0d
    # array.
    not_scalar = np.asanyarray
    # prepare a for partitioning
    if overwrite_input:
        if axis is None:
            ap = a.ravel()
            wp = w.ravel()
        else:
            ap = a
            wp = w
    else:
        if axis is None:
            ap = a.flatten()
            wp = w.flatten()
        else:
            ap = a.copy()
            wp = w.copy()

        
    if axis is None:
        axis = 0
    d = q.ndim
    if d > 2:
        # The code below works fine for nd, but it might not have useful
        # semantics. For now, keep the supported dimensions the same as it was
        # before.
        raise ValueError("q must be a scalar or 1d")
    
    Nx = ap.shape[axis]
    
    # reshape to (Nx, d1,d2,...,dk)
    ap = np.moveaxis(ap, axis, 0)
    wp = np.moveaxis(wp, axis, 0)
    # sort ap and wp to compute Sk
    sorted_index = ap.argsort(axis=0)
    ap = np.take_along_axis(ap, sorted_index, axis=0)
    wp = np.take_along_axis(wp, sorted_index, axis=0) # (N, group)
    
    # compute Sk for k = 1,...,n and q*Sn
    sk = np.asarray([k*wp[k,...]+(Nx-1)*sum(wp[:k,...],axis=0) for k in range(Nx)])
    sn = sk[-1,...]
    qp = np.atleast_1d(q)
    sorted_index_q = qp.argsort(axis=0)
    qp = np.take_along_axis(qp, sorted_index_q, axis=0)
    qsn = qp.reshape((-1,)+(1,)*(sn.ndim))*sn # (q,d1,d2,...,dk)
    # round fractional indices according to interpolation method
    indices = _find_weighted_index(sk,qsn,interpolation)

    if np.issubdtype(indices.dtype, np.integer):
        # take the points along axis
        if np.issubdtype(a.dtype, np.inexact):
            n = np.isnan(ap[-1])
        else:
            # cannot contain nan
            n = np.array(False, dtype=bool)          
        r = np.take_along_axis(ap,indices,0)

    else:
        # weight the points above and below the indices
        indices_below = not_scalar(floor(indices)).astype(intp)
        indices_above = not_scalar(indices_below + 1)
        indices_above[indices_above > Nx - 1] = Nx - 1
        if np.issubdtype(a.dtype, np.inexact):
            # may contain nan, which would sort to the end
            n = np.isnan(ap[-1])
        else:
            # cannot contain nan
            n = np.array(False, dtype=bool)

        # get Xk, Xk+1, Sk, Sk+1 to do interpolation
        x_below = np.take_along_axis(ap,indices_below,0)
        x_above = np.take_along_axis(ap,indices_above,0)

        if interpolation == 'midpoint':
            r = 0.5*(x_below + x_above)
        else:
            s_below = np.take_along_axis(sk,indices_below,0)
            s_above = np.take_along_axis(sk,indices_above,0)
            r = _weighted_lerp(x_below, x_above, s_below,s_above,qsn, out=out)
    # if any slice contained a nan, then all results on that slice are also nan
    inverse_index_q = sorted_index_q.argsort(axis=0)
    if np.any(n):
        if r.ndim == 0 and out is None:
            # can't write to a scalar
            r = a.dtype.type(np.nan)
        else:
            r[..., n] = a.dtype.type(np.nan)

    r = r[inverse_index_q]
    return r


# In[21]:


from  itertools import permutations
import pickle

def add_sample(a,test_sample,out=None,overwrite_input=False,keepdims=False):
    w = np.ones_like(a)
    interpolation_list = ['lower','higher','midpoint','nearest','linear']
    q_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,np.random.rand(10)]
    axis_list = [None,0,(0,1)]
    for d in permutations(tuple(range(a.ndim))):
        axis_list.append(d)
    for interpolation in interpolation_list:
        for q in q_list:
            for axis in axis_list:
                d ={'a':a,
                    'q':q,
                    'w':w,
                    'axis':axis,
                    'out':out,
                    'overwrite_input':overwrite_input,
                    'interpolation':interpolation,
                    'keepdims':keepdims}
                test_sample.append(d)
def check_equal(param_list,error_samples):
    f = True
    for param_dict in param_list:
        a = param_dict['a']
        q = param_dict['q']
        w = param_dict['w']
        axis = param_dict['axis']
        out = param_dict['out']
        overwrite_input = param_dict['overwrite_input']
        interpolation = param_dict['interpolation']
        keepdims = param_dict['keepdims']
        
        result_a = weighted_quantile(a, q, w, axis=axis, out=out, overwrite_input=overwrite_input, interpolation=interpolation, keepdims=keepdims)
        result_b = np.quantile(a, q, axis=axis, out=out, overwrite_input=overwrite_input, interpolation=interpolation, keepdims=keepdims)
        if not np.allclose(result_a,result_b,equal_nan=True):
            error_samples.append(param_dict)
            print("Error occurs!")
            print("result_a",result_a)
            print("result_b",result_b)
            f = False
    if f:
        print("Pass!")


if __name__=="__main__":
    with open("test_sample.pkl",'rb') as f:
        test_sample = pickle.load(f)
    error_samples = []
    check_equal(test_sample,error_samples)


# In[ ]:




