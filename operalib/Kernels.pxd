#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False

cimport numpy

ctypedef numpy.float64_t DTYPE_t

cdef class GaussianKernel( object ):

    cdef long _d
    cdef double _gamma

cdef class GenericGaussianOVKernel( GaussianKernel ):

    cdef long _o


cdef class DecomposableKernel( GenericGaussianOVKernel ):

    cdef double[:,:] _A
