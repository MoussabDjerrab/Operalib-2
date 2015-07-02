#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False

# import cachetools
import scipy.spatial.distance
import numpy

cimport cython
cimport numpy
cimport libc.math

cdef class GaussianKernel( object ):
    """Gaussian kernel"""

    def __init__( self, double gamma, long d ):
        super( GaussianKernel, self ).__init__( )

        assert ( gamma >= 0 ), 'gamma must be positive'
        assert ( d > 0 ), 'd must be positive'

        self._gamma  = gamma
        self._d      = d

    def __call__( self, numpy.ndarray[DTYPE_t, ndim=2] x1 = numpy.empty( ( 0, 0 ) ), numpy.ndarray[DTYPE_t, ndim=2] x2 = numpy.empty( ( 0, 0 ), dtype = numpy.float64 ) ):
        if x2.size == 0:
            return numpy.exp( -self.gamma * scipy.spatial.distance.squareform( scipy.spatial.distance.pdist( x1, 'sqeuclidean' ) ) )
        else:
            return numpy.exp( -self.gamma * scipy.spatial.distance.cdist( x1, x2, 'sqeuclidean' ) )

    property gamma:
        def __get__( self ):
            return self._gamma

    property sigma:
        def __get__( self ):
            return libc.math.sqrt( 1.0 / ( 2.0 * self._gamma ) )

    property d:
        def __get__( self ):
            return self._d

cdef class GenericGaussianOVKernel( GaussianKernel ):

    def __init__( self, double gamma, long d ):
        GaussianKernel.__init__( self, gamma, d )

    def __call__( self, numpy.ndarray[DTYPE_t, ndim=2] x1, numpy.ndarray[DTYPE_t, ndim=2] x2 = numpy.empty( ( 0, 0 ), dtype = numpy.float64 ) ):
        return super( GenericGaussianOVKernel, self ).__call__( x1, x2 )


cdef class DecomposableKernel( GenericGaussianOVKernel ):
    """Decomposable Gaussian kernel"""

    def __init__( self, double gamma, long d, numpy.ndarray[DTYPE_t, ndim=2] A ):
        GenericGaussianOVKernel.__init__( self, gamma, d )

        assert ( A.shape[ 0 ] == A.shape[ 1 ] ), 'A must be a square matrix'
        # assert ( A.shape[ 0 ] > 0 ), 'o must be positive'

        self._o = A.shape[ 1 ]
        self._A = A

    property A:
        def __get__( self ):
            return numpy.asarray( self._A )

        def __set__( self, numpy.ndarray[DTYPE_t, ndim=2] A ):
            self._A = A

    property o:
        def __get__( self ):
            return self._o

    def __call__( self, numpy.ndarray[DTYPE_t, ndim=2] x1, numpy.ndarray[DTYPE_t, ndim=2] x2 = numpy.empty( ( 0, 0 ), dtype = numpy.float64 ) ):
        return numpy.kron( super( DecomposableKernel, self ).__call__( x1, x2 ), self.A )

    # cpdef Jacobian( self, numpy.ndarray[DTYPE_t, ndim=2] x, numpy.ndarray[DTYPE_t, ndim=2] w ):
    #     return numpy.tensordot( self.B, numpy.tensordot( self._grad( x ), numpy.asarray( w ).reshape( ( 2 * self.D, self. o ) ), axes = ( [ 2 ], [ 0 ] ) ), axes = ( [ 0 ], [ 2 ] ) ).transpose( ( 1, 0, 2 ) )


# cdef class TransformableFF( generic_GaussianOVGFF ):
#     """Transformable Gaussian kernel"""

#     def __init__( self, double gamma, long d, long D ):
#         generic_GaussianOVGFF.__init__( self, gamma, 1 )
#         self._o = d

#     property d:
#         def __get__( self ):
#             return self._o

#     property o:
#         def __get__( self ):
#             return self._o

#     cpdef predict( self, numpy.ndarray[DTYPE_t, ndim=2] phi_x, numpy.ndarray[DTYPE_t, ndim=2] w ):
#         return TransformableFF._predictf( self, phi_x, w )

#     cpdef kernel_exact( self, numpy.ndarray[DTYPE_t, ndim=2] x1, double[:,:] x2 = numpy.empty( ( 0, 0 ), dtype = numpy.float64 ) ):
#         if x2.size == 0:
#             x2 = x1
#         return numpy.transpose( numpy.exp( -self.gamma * ( numpy.subtract( x1[ :, None, :, None ], x2[ None, :, None, : ] ) ) ** 2 ), ( 0, 2, 1, 3 ) ).reshape( [ self.o * x1.shape[ 0 ], self.o * x2.shape[ 0 ] ] )

#     # cpdef Jacobian( self, numpy.ndarray[DTYPE_t, ndim=2] x, numpy.ndarray[DTYPE_t, ndim=2] w ):

#     #     J = numpy.zeros( ( x.shape[ 0 ], self.o, self.o ), dtype = numpy.float64 )

#     #     cdef long i
#     #     for i in xrange( 0, self.o ):
#     #         J[ :, i, i ] = numpy.tensordot( self._grad( numpy.asarray( x )[ :, i ].reshape( ( x.shape[ 0 ], 1 ) ) ), w.T, axes = ( [ 2 ], [ 0 ] ) ).reshape( ( x.shape[ 0 ] ) )

#     #     return J
