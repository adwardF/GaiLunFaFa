#ifndef COMMON_H
#define COMMON_H


#include <Eigen/Core>
#include <Eigen/Dense>

template<int rowtype, int coltype>
using Mat = Eigen::Matrix<double, rowtype, coltype, Eigen::RowMajor>;
const int DynDim = Eigen::Dynamic;

const int TENSOR2D_MAT_IND[3][2] {0,0,1,1,0,1};
const int TENSOR_MAT_IND[6][2] {0,0,1,1,2,2,0,1,0,2,1,2};

// Optimize for A=I or B=I ??

template <typename argtype>
class TensorVector2DFunctor
{
public:
    TensorVector2DFunctor(const argtype& inA)
        : A(inA) {}

    const argtype::Scalar &operator()(Eigen::Index pos) const
    {
        return  A(TENSOR2D_MAT_IND[pos][0],TENSOR2D_MAT_IND[pos][1]);
    }

private:
    const argtype& A;
};

template <typename derv>
auto TensorVector2D(const Eigen::MatrixBase<derv>& A)
{
    return Mat<1,3>::NullaryExpr(1, 3, TensorVector2DFunctor(A.derived()));
}


template <typename typeA, typename typeB>
class TensorDya2DFunctor {
public:
    TensorDya2DFunctor(const typeA& A, const typeB& B)
        : A(A), B(B) {}

    double operator()(Eigen::Index row, Eigen::Index col) const
    {
        const int i = TENSOR2D_MAT_IND[row][0];
        const int j = TENSOR2D_MAT_IND[row][1];
        const int k = TENSOR2D_MAT_IND[col][0];
        const int l = TENSOR2D_MAT_IND[col][1];
        return  0.5*(A(i,j) * B(k,l) \
                    +A(k,l) * B(i,j) );
    }

private:
    const typeA & A;
    const typeB & B;
};

template <typename dervA, typename dervB>
auto TensorDya2D(const Eigen::MatrixBase<dervA>& A,
                 const Eigen::MatrixBase<dervB>& B)
{
    return Mat<3,3>::NullaryExpr(3, 3, TensorDya2DFunctor(A.derived(),
                                                          B.derived()));
}


template <typename typeA, typename typeB>
class TensorOutDya2DFunctor {
public:
    TensorOutDya2DFunctor(const typeA & inA, const typeB & inB)
        : A(inA), B(inB) {}

    double operator()(Eigen::Index row, Eigen::Index col) const {
        const int i = TENSOR2D_MAT_IND[row][0];
        const int j = TENSOR2D_MAT_IND[row][1];
        const int k = TENSOR2D_MAT_IND[col][0];
        const int l = TENSOR2D_MAT_IND[col][1];
        return  0.5*(A(i, k) * B(j, l) \
                    +A(i, l) * B(j, k) );
    }

private:
    const typeA & A;
    const typeB & B;
};

template <typename dervA, typename dervB>
auto TensorOutDya2D(const Eigen::MatrixBase<dervA>& A,
                     const Eigen::MatrixBase<dervB>& B)
{
    return Mat<3,3>::NullaryExpr(3, 3, TensorOutDya2DFunctor(A.derived(),
                                                             B.derived()));
}



template <typename argtype>
class TensorVectorFunctor
{
public:
    TensorVectorFunctor(const argtype& inA)
        : A(inA) {}

    const argtype::Scalar &operator()(Eigen::Index pos) const
    {
        return  A(TENSOR_MAT_IND[pos][0],TENSOR_MAT_IND[pos][1]);
    }

private:
    const argtype& A;
};

template <typename derv>
auto TensorVector(const Eigen::MatrixBase<derv>& A)
{
    return Mat<1,6>::NullaryExpr(1, 6, TensorVectorFunctor(A.derived()));
}


template <typename typeA, typename typeB>
class TensorDyaFunctor {
public:
    TensorDyaFunctor(const typeA& A, const typeB& B)
        : A(A), B(B) {}

    double operator()(Eigen::Index row, Eigen::Index col) const
    {
        const int i = TENSOR_MAT_IND[row][0];
        const int j = TENSOR_MAT_IND[row][1];
        const int k = TENSOR_MAT_IND[col][0];
        const int l = TENSOR_MAT_IND[col][1];
        return  0.5*(A(i,j) * B(k,l) \
                    +A(k,l) * B(i,j) );
    }

private:
    const typeA & A;
    const typeB & B;
};

template <typename dervA, typename dervB>
auto TensorDya(const Eigen::MatrixBase<dervA>& A,
                 const Eigen::MatrixBase<dervB>& B)
{
    return Mat<6,6>::NullaryExpr(6, 6, TensorDyaFunctor(A.derived(),
                                                          B.derived()));
}


template <typename typeA, typename typeB>
class TensorOutDyaFunctor {
public:
    TensorOutDyaFunctor(const typeA & inA, const typeB & inB)
        : A(inA), B(inB) {}

    double operator()(Eigen::Index row, Eigen::Index col) const {
        const int i = TENSOR_MAT_IND[row][0];
        const int j = TENSOR_MAT_IND[row][1];
        const int k = TENSOR_MAT_IND[col][0];
        const int l = TENSOR_MAT_IND[col][1];
        return  0.5*(A(i, k) * B(j, l) \
                    +A(i, l) * B(j, k) );
    }

private:
    const typeA & A;
    const typeB & B;
};

template <typename dervA, typename dervB>
auto TensorOutDya(const Eigen::MatrixBase<dervA>& A,
                     const Eigen::MatrixBase<dervB>& B)
{
    return Mat<6,6>::NullaryExpr(6, 6, TensorOutDyaFunctor(A.derived(),
                                                             B.derived()));
}

#endif // COMMON_H
