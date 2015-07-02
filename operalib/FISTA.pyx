#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False

cimport Model
cimport Risk
cimport LearningRate

cimport numpy

from libcpp cimport bool
from cpython cimport bool

import numpy
import sklearn
import scipy.optimize
import sys

cdef class FISTA( object ):

    def __init__( self, Risk.Risk risk, double L, long N = 50, long T = 50, long log_journal = 0 ):

        self._risk = risk
        self._L    = L
        self._T    = T
        self._N    = N
        self._log_journal = log_journal


    def get_params( self, bool deep = True ):
        return { "risk": self._risk, "L": self._L, "T": self._T }

    def set_params( self, **parameters ):
        for parameter, value in parameters.items( ):
            self.setattr( parameter, value )

    cpdef fit( self, Model.Model model, numpy.ndarray[DTYPE_t, ndim=2] features, numpy.ndarray[DTYPE_t, ndim=2] targets ):
        cdef long o = model.o
        cdef double const = 1
        cdef double next_const
        cdef numpy.ndarray[DTYPE_t, ndim=2] pred = numpy.zeros( ( targets.shape[ 0 ], targets.shape[ 1 ] ) )
        cdef dict temp_t  = model.params # deepcopy
        cdef dict temp_tp = model.params # deepcopy

        if self._log_journal > 0:
            log_journal_file = open( "log_journal.txt", 'w' )

        cdef long k, i
        for k in xrange( 0, self._N ):
            for param_name in model.params.iterkeys( ):
                for t in xrange( 0, self._T ):

                    pred = model( features )
                    temp_tp[ param_name ] = temp_t[ param_name ]
                    model.params[ param_name ] = model.params[ param_name ]  - 1.0 / self._L * self._risk.gradient( pred, targets, model, param_name )
                    temp_t[ param_name ] = self._risk.prox( pred, targets, model, 1.0 / self._L, param_name )

                    next_const = ( 1.0 + numpy.sqrt( 1.0 + 4.0 * const ** 2 ) ) / 2.0
                    model.params[ param_name ] = temp_t[ param_name ] + ( const - 1.0 ) / next_const * ( temp_t[ param_name ] - temp_tp[ param_name ] )
                    const = next_const

                model.params[ param_name ] = temp_t[ param_name ]
