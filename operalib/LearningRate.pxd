cdef class generic( object ):

    cpdef eta( self, int t )

cdef class Convex( LearningRate.generic ):

    cdef double _eta0
    cdef double _lbda

    cpdef eta( self, int t )

cdef class SConvex( LearningRate.generic ):

    cdef double _eta0
    cdef double _lbda

    cpdef eta( self, int t )

cdef class Optimal( LearningRate.generic ):

    cdef double _eta0
    cdef double _lbda

    cpdef eta( self, int t )

cdef class InvScaling( LearningRate.generic ):

    cdef double _eta0
    cdef double _power
    cdef double _lbda

    cpdef eta( self, int t )

cdef class Constant( LearningRate.generic ):

    cdef double _eta0

    cpdef eta( self, int t )
