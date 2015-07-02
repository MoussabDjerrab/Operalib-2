cimport Model
cimport numpy

ctypedef numpy.float64_t DTYPE_t


cdef class Smooth_func( object ):

    cpdef gradient( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, param  )

cdef class Non_smooth_func( object ):

    cpdef prox( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, double lbda, param  )

cdef class L22_loss( Smooth_func ):

    cdef double _eps

    cpdef gradient( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, param )


cdef class RKHS2_reg( Smooth_func ):

    cpdef gradient( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, param )


cdef class L22_reg( Smooth_func ):

    cpdef gradient( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, param )

    # cpdef prox( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, double lbda, param )

cdef class L2_reg( Non_smooth_func ):

    cpdef prox( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, double lbda, param )


cdef class L1_reg( Non_smooth_func ):

    cpdef prox( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, double lbda, param )


cdef class L1L2_group_reg( Non_smooth_func ):

    cdef long _block_size

    cpdef prox( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, double lbda, param )


cdef class projection( Non_smooth_func ):

    cpdef prox( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] y, Model.Model h, double lbda, param )


cdef class Risk( object ):

    cdef dict _weights
    cdef dict _smooth
    cdef dict _non_smooth

    cpdef gradient( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] targets, Model.Model h, param )

    cpdef prox( self, numpy.ndarray[DTYPE_t, ndim=2] pred, numpy.ndarray[DTYPE_t, ndim=2] targets, Model.Model h, double eta, param )

cdef class Ridge( Risk ):

    pass

cdef class Lasso( Risk ):

    pass

cdef class GroupLasso( Risk ):

    pass

cdef class ElasticNet( Risk ):

    pass

cdef class GroupElasticNet( Risk ):

    pass

cdef class OKVAR( Risk ):

    pass
