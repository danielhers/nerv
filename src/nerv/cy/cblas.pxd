# vim:set ft=cython ts=4 sw=4 sts=4 autoindent:

'''
Declarations for the cblas header.

Version:    2014-03-14
Author:     Pontus Stenetorp    <pontus stenetorp se>
'''

cdef extern from 'cblas.h':
    enum CBLAS_ORDER: CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans
    enum CBLAS_UPLO: CblasUpper, CblasLower
    enum CBLAS_DIAG: CblasNonUnit, CblasUnit

    # TODO: Can we declare arguments as const?
    void cblas_dgemv 'cblas_dgemv'(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
            int M, int N, double alpha, double *A, int lda, double *X,
            int incX, double beta, double *Y, int incY) nogil
    void cblas_dgemm 'cblas_dgemm'(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
            CBLAS_TRANSPOSE TransB, int M, int N, int K, double alpha,
            double *A, int lda, double *B, int ldb, double beta, double *C,
            int ldc) nogil
    void cblas_dger 'cblas_dger'(CBLAS_ORDER order, int M, int N, double alpha,
            double *X, int incX, double *Y, int incY, double *A, int lda) nogil
