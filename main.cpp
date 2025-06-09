#include <cstdio>
#include <cstdlib>

#include <set>
#include <map>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <Eigen/UmfPackSupport>
#include <Eigen/CholmodSupport>

#include "UsualElements.h"

#include "omp.h"

template <typename derv>
void print_mat(const Eigen::DenseBase<derv> & mat)
{
    for(int i = 0; i < mat.rows(); ++i)
    {
        for(int j = 0; j < mat.cols(); ++j)
            printf("%lf ",mat(i,j));
        putchar('\n');
    }

}

class HyperElem : public LaElem<QRHex3D>
{
public:
    using geom_type = QRHex3D;
    using base_type = LaElem<QRHex3D>;

    HyperElem(const Mat<geom_type::Nn,geom_type::Ndim> & elemX):
        base_type(elemX)
    {};

    ~HyperElem() {};

    virtual std::tuple<Mat<1,6>,Mat<6,6>>
        make_T_and_TE(const Mat<3,3> &F) const
    {
        auto C = F.transpose() * F;

        auto Cinv = C.inverse();

        double I1 = C(0,0)+C(1,1)+C(2,2);

        double J = F.determinant();

        double Ji23 = pow(J,-2.0/3);

        auto EYE = Mat<3,3>::Identity();

        auto T33 = 2.0*Ji23*(EYE-I1/3*Cinv) + 10.0*2.0* (J-1)*J*Cinv;

        Mat<1,6> T;
        T[0]=T33(0,0);T[1]=T33(1,1);T[2]=T33(2,2);
        T[3]=T33(0,1);T[4]=T33(0,2);T[5]=T33(1,2);

        Mat<6,6> T_E;

        T_E = 4.0/3*Ji23*( I1*TensorOutDya(Cinv,Cinv)-TensorDya(Cinv,EYE)*2+I1/3*TensorDya(Cinv,Cinv) ) \
              +10.0*(2.0*(2.0*J-1)*J*TensorDya(Cinv,Cinv)-4.0*(J-1)*J*TensorOutDya(Cinv,Cinv));

        //print_mat(TensorDya2D(Cinv,Cinv));

        return std::make_tuple(T, T_E);
    }
};

class TrussElem
{
public:
    TrussElem(const Mat<2,2> &X): nodeX(X)
    {
        L0 = ( X(0,Eigen::all) - X(1,Eigen::all) ).norm();
    }
    virtual ~TrussElem() {};

    std::tuple<Mat<1,4>,Mat<4,4>> make_f_and_DfDu(const Mat<1,4> &elemU) const
    {
        double n,n_r;

        Mat<1,4> f;
        Mat<4,4> K;

        Mat<2,2> x = nodeX+elemU.template reshaped<Eigen::AutoOrder>(2,2);

        auto r = x(1, Eigen::all)-x(0, Eigen::all);
        double len = r.norm();
        auto ru = r/len;

        n = len/L0 - 1.0; n_r = 1.0/L0;

        f[2] = n*ru[0];
        f[3] = n*ru[1];

        f[0] = -f[2];
        f[1] = -f[3];

        Mat<2,2> K22 = (n/len) * Mat<2,2>::Identity();
        K22 += (n_r-n/len)* ru.transpose()*ru;

        K({0,1},{0,1}) = K22;
        K({2,3},{2,3}) = K22;
        K({2,3},{0,1}) =-K22;
        K({0,1},{2,3}) =-K22;

        /*
        print_mat(r);

        print_mat(ru);

        print_mat(ru.transpose()*ru);

        print_mat(f);

        print_mat(K);
        */

        return std::make_tuple(f,K);
    }

    virtual void compute_ip_data() {} ;

private:
    Mat<2,2> nodeX;
    double L0;
};

Eigen::Matrix<double,1,-1,Eigen::RowMajor>
    SPLUMT_solve(const Eigen::SparseMatrix<double> &eigen_K,
                 const Eigen::Matrix<double,1,Eigen::Dynamic,Eigen::RowMajor> & eigen_rhs);
//auto m = Mat<1,1>{0.5};
const int NK = 1000;
int N_N = 0;

Mat<DynDim,2> X;

using ElemType = TrussElem;

std::vector<ElemType*> elemlist;
std::vector<std::array<int,2>> elemconn;

Eigen::SparseMatrix<double,Eigen::RowMajor> K;
std::vector<Eigen::Triplet<double>> spvals;
Mat<1,DynDim> U;
Mat<1,DynDim> f_target;
Mat<1,DynDim> res;
Mat<1,DynDim> f;

const double PENALTY = 1e8;

void build()
{
    printf("s time %lf\n", clock()*1.0/CLOCKS_PER_SEC);

    f.setZero();
    K.setZero();
    spvals.clear();

    int THREADS = omp_get_num_threads();
    #pragma omp parallel for
    for(int pid = 0; pid < THREADS; ++pid)
    {
        Mat<1,DynDim> localf(N_N*2);
        localf.setZero();
        std::vector<Eigen::Triplet<double>> localspvals;
        for(int n = pid; n < elemlist.size(); n += THREADS)
        {
            //puts("R1");
            Mat<1,2*2> elemseq;
            //puts("Re");
            for(int i = 0; i < 2; ++i)
            {
                elemseq[i*2+0] = elemconn[n][i]*2;
                elemseq[i*2+1] = elemconn[n][i]*2+1;
            }

            auto elemU = U(elemseq);

            elemlist[n]->compute_ip_data();

            auto elem_pair = elemlist[n]->make_f_and_DfDu( elemU );

            localf( elemseq ) += std::get<0>(elem_pair);
            auto Ke = std::get<1>(elem_pair);

            for(int ei = 0; ei < 2 *2; ++ei) for(int ej = 0; ej < 2 *2; ++ej)
            {
                localspvals.push_back({elemseq[ei],elemseq[ej], Ke(ei,ej)});
            }
        }

        #pragma omp critical
        {
            spvals.insert(spvals.end(),localspvals.begin(), localspvals.end());
            f += localf;
        }
    }

    K.setFromTriplets(spvals.begin(), spvals.end() );

    K.makeCompressed();

    printf("e time %lf\n", clock()*1.0/CLOCKS_PER_SEC);

}

int get_node(Mat<-1,2> & Xtmp,
             std::map<std::tuple<int,int>,int> &nset,
             int i,int j)
{
    int r;
    if(nset.find({i,j})==nset.end())
    {
        int ncnt = nset.size();
        r = ncnt;
        nset.insert({{i,j},ncnt});
        Xtmp(ncnt,0) = 1.0*j/(NK-1);
        Xtmp(ncnt,1) = 1.0*i/(NK-1);
        //++ncnt;
    }
    else
        r = nset[{i,j}];
    return r;
}

void make_edge(Mat<-1,2> & Xtmp,
               std::map<std::tuple<int,int>,int> &nset,
               std::set<std::tuple<int,int>>&eset,
               int i,int j, int k, int l)
{
    int na, nb;

    na = get_node(Xtmp,nset,i,j);
    nb = get_node(Xtmp,nset,k,l);

    if(na>nb) std::swap(na,nb);

    if(eset.find({na,nb})==eset.end())
    {
        eset.insert({na,nb});
        elemconn.push_back({na,nb});
        elemlist.push_back(new ElemType( Xtmp({na,nb},Eigen::all) ) );
    }

}

void buildmesh()
{
    Mat<-1,2> Xtmp(NK*NK,2);

    std::map<std::tuple<int,int>,int> nodeset;
    std::set<std::tuple<int,int>> edgeset;

    for(int i = 0; i < NK-1; ++i)
    {
        for(int j = 0; j < NK-1; ++j)
        {
            double cx = (j+j+1)/2.0/(NK-1);
            double cy = (i+i+1)/2.0/(NK-1);

            if(pow(cx-0.5,2.0)+pow(cy-0.5,2.0)<0.15*0.15)
                continue;

            make_edge(Xtmp,nodeset,edgeset,i,j,i+1,j);
            make_edge(Xtmp,nodeset,edgeset,i,j,i,j+1);
            make_edge(Xtmp,nodeset,edgeset,i+1,j,i+1,j+1);
            make_edge(Xtmp,nodeset,edgeset,i,j+1,i+1,j+1);
            make_edge(Xtmp,nodeset,edgeset,i,j,i+1,j+1);

        }
    }

    N_N = nodeset.size();

    X.resize(N_N,2);
    puts("A");
    for(int i = 0; i < nodeset.size(); ++i)
        X(i,Eigen::all) = Xtmp(i,Eigen::all);
}

#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkLine.h>
#include <vtkQuadraticTetra.h>
#include <vtkQuadraticHexahedron.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDoubleArray.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkPointData.h>
/*
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/spai1.hpp>
#include <amgcl/preconditioner/dummy.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/bicgstabl.hpp>
#include <amgcl/solver/gmres.hpp>
*/

#include "SLUMTsolver.h"
SLUMTSolver SLUsolver;

int main()
{
    omp_set_num_threads(8);

    Eigen::setNbThreads(8);
    printf("%d\n",Eigen::nbThreads());

    buildmesh();

    K.resize(N_N*2,N_N*2);
    U.resize(N_N*2);
    f.resize(N_N*2);
    res.resize(N_N*2);
    f_target.resize(N_N*2);

    f_target.setConstant(0);
    U.setZero();

    vtkNew<vtkUnstructuredGrid> vtkgrid;
    vtkNew<vtkXMLUnstructuredGridWriter> vtkwriter;
    vtkwriter->SetDataModeToBinary();

    vtkNew<vtkPoints> vtk_points;
    for(int i = 0; i < X.rows(); ++i)
        vtk_points->InsertNextPoint(X(i,0),X(i,1),0.0);
    vtkgrid->SetPoints(vtk_points);

    /*
    for(int i = 0; i+2 < NK; i += 2)
    {
        for(int j = 0; j+2 < NK; j += 2)
        {
            int na = i*NK+j;
            int nb = i*NK+j+1;
            int nc = i*NK+j+2;
            int nd = (i+1)*NK+j;
            int ne = (i+1)*NK+j+1;
            int nf = (i+1)*NK+j+2;
            int ng = (i+2)*NK+j;
            int nh = (i+2)*NK+j+1;
            int ni = (i+2)*NK+j+2;

            elemlist.push_back(new ElemType(X({na,nc,ni,nb,nf,ne},Eigen::all) ) );

            //Mat<3,2> R = X({na,nb,nd},Eigen::all);
            //print_mat(R);

            elemconn.push_back({na,nc,ni,nb,nf,ne});
            elemlist.push_back(new ElemType(X({na,ng,ni,nd,nh,ne},Eigen::all) ) );
            elemconn.push_back({na,ng,ni,nd,nh,ne});
        }
    }
    */

    for(int i = 0; i < elemconn.size(); ++i)
    {
        vtkNew<vtkLine> t;
        for(int k = 0; k < 2; ++k)
            t->GetPointIds()->SetId(k,elemconn[i][k]);
        vtkgrid->InsertNextCell(t->GetCellType(), t->GetPointIds());
    }



    printf("%d elements\n", elemlist.size());



    //Eigen::SimplicialLDLT< decltype(K) > solver;
    //Eigen::ConjugateGradient< decltype(K) > solver;
    //Eigen::UmfPackLU<decltype(K)> solver;
    //Eigen::CholmodSupernodalLLT<decltype(K)> solver;
    Eigen::CholmodSupernodalLLT<decltype(K)> solver;

    int solver_reuse = 0;


    double targetU = 0.0;

    for(int additer = 0; additer < 20; ++ additer)
    {
    targetU += 0.05;

    int iternum = 100;

    solver_reuse = 0; // force recompute K at begining of each additive step

    while(iternum--)
    {

        f.setZero();

        double fres = 0.0;

        build();

        res = f_target - f;

        for(int i = 0; i < N_N; ++i)
        {
            if(X.coeff(i,0) < 1e-5 || X.coeff(i,0)>1.0-1e-5)
            {
                K.coeffRef(i*2,i*2) = PENALTY;
                res[i*2] = -PENALTY*(U[i*2]-0.0);
                if(X.coeff(i,0)>1.0-1e-5)
                    res[i*2] = -PENALTY*(U[i*2]-targetU);

                K.coeffRef(i*2+1,i*2+1) = PENALTY;
                res[i*2+1] = -PENALTY*(U[i*2+1]-0.0);
                //K.coeffRef(i*3+2,i*3+2) = 1e10;
                //res[i*3+2] = -1e10*(U[i*3+2]-0.0);
            }
            else
                fres += abs(res(i));
        }

        /*

        std::vector<double> rhs_vec(N_N*2);
        std::copy(res.data(),res.data()+N_N*2,rhs_vec.begin());
        std::vector<double> sol_vec(N_N*2);
        std::fill(sol_vec.begin(),sol_vec.end(),0.0);
        Mat<1,DynDim> sol(N_N*2);
        puts("RRR");



        typedef amgcl::backend::builtin<double> SBackend;
        typedef amgcl::backend::builtin<double> PBackend;

        typedef amgcl::make_solver<
            amgcl::amg<
                PBackend,
                amgcl::coarsening::smoothed_aggregation,
                amgcl::relaxation::ilu0>,

            //amgcl::relaxation::as_preconditioner<PBackend,amgcl::relaxation::ilu0>,

            amgcl::solver::cg<SBackend> > Solver;

        //amgcl::preconditioner::dummy<PBackend>,

        printf("G");
        SBackend::matrix Km(K);
        printf("N");

        Solver::params solverp;
        solverp.solver.maxiter = 200;
        solverp.solver.verbose = true;

        Solver amgclsolve(Km,solverp);

        std::cout << amgclsolve;

        auto [iters, errors] = amgclsolve(Km, rhs_vec, sol_vec);

        printf("iters %d errors %lf \n",iters,errors);

        std::copy(sol_vec.begin(),sol_vec.end(),sol.data());

        */

        //Eigen::SimplicialLDLT solver(K);

        if(solver_reuse == 0)
        {
            solver.compute(K);
            solver_reuse += 1;
        }

        --solver_reuse;

        //solver.compute(K);
        //printf("solver compute ret %d\n",solver.info());
        Eigen::Matrix<double,DynDim,1> sol = solver.solve(res.transpose());

        U += sol.transpose();

        //auto sol = SLUsolver.solve_refact(K,res);
        //U += sol;

        printf("Max Du %lf\n", sol.norm());
        printf("res %lf\n", res.norm());

        vtkNew<vtkDoubleArray> soldata;
        soldata->SetName("Displacement");
        soldata->SetNumberOfComponents(3);
        for(int i = 0; i < X.rows(); ++i)
        {
            soldata->InsertNextTuple3(U[2*i+0],U[2*i+1],0.0);
        }
        vtkgrid->GetPointData()->RemoveArray(soldata->GetName());
        vtkgrid->GetPointData()->AddArray(soldata);

        char framename[50];
        sprintf(framename,"output_%03d.vtu",additer);
        vtkwriter->SetFileName(framename);
        vtkwriter->SetInputData(vtkgrid);
        vtkwriter->Write();

        if(sol.norm() < 1e-6)
            break;
    }

    }


    for(int i = 0; i < 18; ++i)
    {
        for(int j = 0; j < 18; ++j)
            printf("%lf ",K.coeffRef(i,j));
        printf("\n");
    }


    for(int i = 0; i < 18; ++i)
    {
        printf("%lf\n", U[i]);
    }

    //print_mat(X);

    for(int i =0; i<elemlist.size(); ++i)
        delete elemlist[i];

    return 0;
}
