import numpy

from libcpp cimport bool
from cpython cimport bool

cimport numpy
cimport Kernels

ctypedef numpy.float64_t DTYPE_t

cdef class Model( object ):

    cdef Kernels.GenericGaussianOVKernel _kernel
    cdef dict _params
    cdef double[:,:] _data

    # cpdef Jacobian( self, numpy.ndarray[DTYPE_t, ndim=2] x )

    cpdef reset( self )
