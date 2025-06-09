#ifndef FEMESH_H
#define FEMESH_H

#include "common.h"

template <typename elem_type>
class FEMesh
{
public:
    template <typename elemconn_container>
    FEMesh(const Mat<DynDim, elem_type::Ndim>& X,
           const elemconn_container & elemconn )
    {
        this->X = X;
        elemlist.resize(elemconn.size());
        for(int i = 0; i < elemconn.size(); ++i)
        {
            elemlist[i] = new elem_type( X(elemconn[i],Eigen::all) );
        }
    };

    ~FEMesh()
    {
        for(const auto &p : elemlist)
        {
            delete p;
        }
    };

    void build_f_and_K()
    {
        ;
    }

private:
    Mat<DynDim,elem_type::Nn> X;
    std::vector<elem_type*> elemlist;
    Eigen::SparseMatrix<double,Eigen::RowMajor> K;
    std::vector<Eigen::Triplet<double>> spvals;
    Mat<1,DynDim> U;
    Mat<1,DynDim> f_target;
    Mat<1,DynDim> res;
    Mat<1,DynDim> f;
};

#endif // FEMESH_H
