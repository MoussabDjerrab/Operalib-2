cimport numpy

import numpy

from libcpp cimport bool
from cpython cimport bool

cdef class Model( object ):

    def __init__( self, kernel, x ):
        self._kernel = kernel
        self._data   = x
        self._params = { 'coefs': numpy.zeros( ( x.shape[ 0 ] * self.o ) ), 'bias': numpy.zeros( self.o ), 'A': self.kernel.A }

    def __call__( self, numpy.ndarray[DTYPE_t, ndim=2] x = numpy.empty( ( 0, 0 ) ) ):
            self.kernel.A = self.params[ 'A' ]
            return numpy.asarray( numpy.dot( self.kernel( self.data, x ), self.params[ 'coefs' ] ) ).reshape( ( self.n, self.o ) ) + self.params[ 'bias' ]

    property kernel:
        def __get__( self ):
            return self._kernel

    property data:
        def __get__( self ):
            return numpy.asarray( self._data )

    property n:
        def __get__( self ):
            return self.data.shape[ 0 ]

    property d:
        def __get__( self ):
            return self.kernel.d

    property o:
        def __get__( self ):
            return self.kernel.o

    property Gram:
        def __get__( self ):
            return self.kernel( self.data )

    property params:
        def __get__( self ):
            return self._params

    # def set_params( self, **kwargs ):
    #     self._params = kwargs

    # def set_param( self, name, value ):
    #     if name == 'coefs':
    #         self.coefs = value
    #     if name == 'A':
    #         self.kernel.A = value
    #     if name == 'bias':
    #         self.bias = value
    #     return self

    # property sparsity:
    #     def __get__( self ):
    #         return ( numpy.asarray( self._coefs ) == 0 ).sum( ) / float( self._coefs.size )

    # cpdef Jacobian( self, numpy.ndarray[DTYPE_t, ndim=2] x ):
    #     return numpy.asarray( self._feature_map.Jacobian( x, self.coefs ) )

    cpdef reset( self ):
        self._coefs = numpy.zeros( ( self.n, self.o ) )
        self._bias  = numpy.zeros( ( 1, self.o ) )
