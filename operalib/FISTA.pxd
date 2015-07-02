#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False

cimport Model
cimport Risk
cimport numpy

import numpy

ctypedef numpy.float64_t DTYPE_t

cdef class FISTA( object ):

    cdef Risk.Risk _risk
    cdef double _L
    cdef long _T
    cdef long _N
    cdef long _log_journal

    cpdef fit( self, Model.Model model, numpy.ndarray[DTYPE_t, ndim=2] features, numpy.ndarray[DTYPE_t, ndim=2] targets )
