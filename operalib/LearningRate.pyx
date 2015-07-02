#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False

cdef class generic( object ):

    def __init__( object ):
        pass

    cpdef eta( self, int t ):
        pass

cdef class Convex( LearningRate.generic ):

    def __init__( self, double eta0, double lbda ):
        self._eta0 = eta0
        self._lbda = lbda

    property eta0:
        def __get__( self ):
            return self._eta0

        def __set__( self, double eta0 ):
            self._eta0 = eta0

    property lbda:
        def __get__( self ):
            return self._lbda

        def __set__( self, double lbda ):
            self._lbda = lbda

    cpdef eta( self, int t ):
        return self._eta0 / ( 1 + self._eta0 * self._lbda * t * t )

cdef class SConvex( LearningRate.generic ):

    def __init__( self, double eta0, double lbda ):
        self._eta0 = eta0
        self._lbda = lbda

    property eta0:
        def __get__( self ):
            return self._eta0

        def __set__( self, double eta0 ):
            self._eta0 = eta0

    property lbda:
        def __get__( self ):
            return self._lbda

        def __set__( self, double lbda ):
            self._lbda = lbda

    cpdef eta( self, int t ):
        return self._eta0 / ( 1 + self._eta0 * self._lbda * t )

cdef class Optimal( LearningRate.generic ):

    def __init__( self, double eta0, double lbda ):
        self._eta0 = eta0
        self._lbda = lbda

    property eta0:
        def __get__( self ):
            return self._eta0

        def __set__( self, double eta0 ):
            self._eta0 = eta0

    cpdef eta( self, int t ):
        return 1.0 / ( self._lbda * ( self._eta0 + t ) )

cdef class InvScaling( LearningRate.generic ):

    def __init__( self, double eta0, double lbda, double power ):
        self._eta0  = eta0
        self._power = power
        self._lbda  = lbda

    property eta0:
        def __get__( self ):
            return self._eta0

        def __set__( self, double eta0 ):
            self._eta0 = eta0

    property power:
        def __get__( self ):
            return self._power

        def __set__( self, double power ):
            self._power = power

    cpdef eta( self, int t ):
        return self._eta0 / ( 1 + self._eta0 * self._lbda * t ) ** self._power

cdef class Constant( LearningRate.generic ):

    def __init__( self, double eta0 ):
        self._eta0  = eta0

    property eta0:
        def __get__( self ):
            return self._eta0

        def __set__( self, double eta0 ):
            self._eta0 = eta0

    cpdef eta( self, int t ):
        return self._eta0
