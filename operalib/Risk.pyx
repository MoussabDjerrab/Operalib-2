#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False

cimport numpy
cimport Model

cimport Kernels
import scipy.stats
import scipy.linalg
import numpy

from libcpp cimport bool
from cpython cimport bool

cdef simplex_map( i ):
    if i > 2:
        return numpy.bmat( [ [ [ [ 1 ] ], numpy.repeat( -1.0 / ( i - 1 ), i - 1 ).reshape( ( 1, i - 1 ) ) ], [ numpy.zeros( ( i - 2, 1 ) ), simplex_map( i - 1 ) * numpy.sqrt( 1.0 - 1.0 / ( i - 1 ) ** 2 ) ] ] )
    elif i == 2:
        return numpy.array( [ [ 1, -1 ] ] )
    else:
        raise "invalid number of classes"

################################### Losses #####################################

cdef class Smooth_func( object ):

    cpdef gradient( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, param  ):
        pass

cdef class Non_smooth_func( object ):

    cpdef prox( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, double lbda, param  ):
        pass

cdef class L22_loss( Smooth_func ):

    def __init__( self, eps = 0 ):
        self._eps = eps

    cpdef gradient( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, param ):
        if param == 'coefs':
            return numpy.dot( h.Gram, ( pred - y ).ravel( ) )
        elif param == 'bias':
            return numpy.mean( ( pred - y ), axis = 0 )
        elif param == 'A':
            return numpy.dot( numpy.dot( numpy.linalg.pinv( h.params[ 'A' ] ), y.T ), pred - y ) / y.shape[ 0 ]
        else:
            raise 'Invalid parameter'


############################### Regularizations ################################

cdef class RKHS2_reg( Smooth_func ):

    cpdef gradient( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, param ):
        if param == 'coefs':
            return pred.ravel( )
        elif param == 'bias':
            return numpy.zeros( pred.shape[ 1 ] )
        elif param == 'A':
            raise 'Not implemented yet'
        else:
            raise 'Invalid parameter'

cdef class L22_reg( Smooth_func ):

    cpdef gradient( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, param ):
        cdef double norm
        if param == 'coefs':
            return h.coefs
        elif param == 'bias':
            return numpy.zeros( pred.shape[ 1 ] )
        elif param == 'A':
            raise 'Not implemented yet'
        else:
            raise 'Invalid parameter'

cdef class L2_reg( Non_smooth_func ):

    cpdef prox( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, double lbda, param ):
        cdef double norm = numpy.linalg.norm( h.coefs )
        return h.coefs / norm * numpy.maximum( numpy.abs( norm - lbda ), 0 )


cdef class L1_reg( Non_smooth_func ):

    cpdef prox( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, double lbda, param ):
        return numpy.sign( h.params[ param ] ) * numpy.maximum( numpy.abs( h.params[ param ] ) - lbda, 0 )

cdef class L1L2_group_reg( Non_smooth_func ):

    def __init__( self, long block_size = 1 ):
        self._block_size = block_size

    cpdef prox( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, double lbda, param ):

        cdef long n_blocks = h.coefs.shape[ 1 ] / self._block_size
        norm = numpy.sqrt( numpy.einsum( 'ij,ij->i', h.coefs.reshape( ( n_blocks, self._block_size ) ), h.coefs.reshape( ( n_blocks, self._block_size ) ) ) )
        return ( h.coefs.reshape( ( n_blocks, self._block_size ) ).T / norm * numpy.maximum( norm - lbda, 0 ) ).T.reshape( ( 1, n_blocks * self._block_size ) )

cdef class projection( Non_smooth_func ):

    def __init__( self ):
        pass

    cpdef prox( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, double lbda, param ):
        return h.coefs * min( 1, lbda / numpy.linalg.norm( h.coefs, 2 ) )

cdef class Risk( object ):

    def __init__( self, dict weights, dict smooth = None, dict non_smooth = None ):
        self._smooth     = smooth # gradient update
        self._non_smooth = non_smooth # proximal update
        self._weights    = weights

    property weights:
        def __get__( self ):
            return self._weights

    cpdef gradient( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] targets, Model.Model h, param ):

        if not self._smooth[ param ]:
            return numpy.zeros( h.params[ param ].shape )

        gradient = numpy.zeros( h.params[ param ].shape )
        for idx, fun in enumerate( self._smooth[ param ] ):
            gradient = gradient + self.weights[ param ][ idx ] * fun.gradient( pred, targets, h, param )

        return gradient

    cpdef prox( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] targets, Model.Model h, double eta, param ):

        if not self._non_smooth[ param ]:
            return h.params[ param ]

        for idx, fun in enumerate( self._non_smooth[ param ] ):
            return fun.prox( pred, targets, h, self.weights[ param ][ len( self._smooth[ param ] ) ] * eta, param )


cdef class Ridge( Risk ):

    def __init__( self, double lbda, double eps = 0 ):
        super( Ridge, self ).__init__( weights = numpy.array( [ 1, lbda ] ), smooth = ( L22_loss( eps ), L22_reg( ) ) )

cdef class Lasso( Risk ):

    def __init__( self, double lbda, double eps = 0 ):
        super( Lasso, self ).__init__( weights = numpy.array( [ 1, lbda ] ), smooth = ( L22_loss( eps ), ), non_smooth = L1_reg( ) )

cdef class GroupLasso( Risk ):

    def __init__( self, double lbda, long block_size, double eps = 0 ):
        super( GroupLasso, self ).__init__( weights = numpy.array( [ 1, lbda ] ), smooth = ( L22_loss( eps ), ), non_smooth = L1L2_group_reg( block_size ) )

cdef class ElasticNet( Risk ):

    def __init__( self, double lbda1, double lbda2, double eps = 0 ):
        super( ElasticNet, self ).__init__( weights = numpy.array( [ 1, lbda1, lbda2 ] ), smooth = ( L22_loss( eps ), L22_reg( ) ), non_smooth = L1_reg( ) )

cdef class GroupElasticNet( Risk ):

    def __init__( self, double lbda1, double lbda2, long block_size, double eps = 0 ):
        super( GroupElasticNet, self ).__init__( weights = numpy.array( [ 1, lbda1, lbda2 ] ), smooth = ( L22_loss( eps ), L22_reg( ) ), non_smooth = L1L2_group_reg( block_size ) )

cdef class OKVAR( Risk ):

    def __init__( self, double lbda1, double lbda2, double lbda3, double mu1 = 1, double mu2 = 1, double mu3 = 1, double eps = 0 ):
        super( OKVAR, self ).__init__( weights = { 'coefs': numpy.array( [ mu1, lbda1, lbda2 ] ), 'bias': numpy.array( [ mu2 ] ), 'A': numpy.array( [ mu3, lbda3 ] ) }, smooth = { 'coefs': [ L22_loss( eps ), RKHS2_reg( ) ], 'A': [ L22_loss( eps ) ], 'bias': [ L22_loss( eps ) ] }, non_smooth = { 'coefs': [ L1_reg( ) ], 'A': [ L1_reg( ) ], 'bias': [] }  )
