#include <Eigen/Core>

#include <Eigen/Sparse>


namespace SuperLU
{
    extern "C"
    {
    #include "slumt/slu_mt_ddefs.h"
    #include "slumt/supermatrix.h"
    }

};

using EigenMat = Eigen::SparseMatrix<double>;
using EigenVec = Eigen::Matrix<double,1,-1,Eigen::RowMajor>;

struct SLUMTSolver
{
    SLUMTSolver() { factored = false; };
    ~SLUMTSolver()
    {
        if(factored == true)
        {
            SuperLU::Destroy_SuperNode_SCP(&L);
            SuperLU::Destroy_CompCol_NCP(&U);
        }
    }

    EigenVec solve_refact(const EigenMat & eigenA,
                          const EigenVec & B );

    EigenVec solve_reuseLU(const EigenVec &B);

private:
    bool factored;
    SuperLU::SuperMatrix A,L,U;
    SuperLU::int_t *perm_c, *perm_r;
    double      *R, *C;
    double      *ferr, *berr;
    SuperLU::superlumt_options_t superlumt_options;


};
