#include <cstdio>
#include <cstdlib>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <unsupported/Eigen/IterativeSolvers>

#include <Eigen/UmfPackSupport>
#include <Eigen/CholmodSupport>
//#include <Eigen/PaStiXSupport>

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
    TrussElem(const Mat<2,2> &X): nodeX(X) { }
    virtual ~TrussElem();

    std::tuple<Mat<1,4>,Mat<4,4>> make_f_and_DfDu(const Mat<1,4> &elemU) const
    {
        double n,n_r;

        Mat<1,4> f;
        Mat<4,4> K;

        Mat<2,2> x = nodeX+elemU.template reshaped<Eigen::AutoOrder>(2,2);


    }

private:
    Mat<2,2> nodeX;
};


//auto m = Mat<1,1>{0.5};
//const int NK = 101;
int N_N = 0;

Mat<DynDim,3> X;

using ElemType = HyperElem;

std::vector<ElemType*> elemlist;
std::vector<std::array<int,20>> elemconn;

Eigen::SparseMatrix<double> K;
std::vector<Eigen::Triplet<double>> spvals;
Mat<1,DynDim> U;
Mat<1,DynDim> f_target;
Mat<1,DynDim> res;
Mat<1,DynDim> f;


void build()
{
    printf("s time %lf\n", clock()*1.0/CLOCKS_PER_SEC);

    f.setZero();
    K.setZero();
    spvals.clear();

    #pragma omp parallel for
    for(int n = 0; n < elemlist.size(); ++n)
    {
        //puts("R1");
        Mat<1,20*3> elemseq;
        //puts("Re");
        for(int i = 0; i < 20; ++i)
        {
            elemseq[i*3+0] = elemconn[n][i]*3;
            elemseq[i*3+1] = elemconn[n][i]*3+1;
            elemseq[i*3+2] = elemconn[n][i]*3+2;
        }

        auto elemU = U(elemseq);

        //puts("R");
        elemlist[n]->compute_ip_data();

        auto elem_pair = elemlist[n]->make_f_and_DfDu( elemU );
        //puts("Rmake");

        f( elemseq ) += std::get<0>(elem_pair);
        auto Ke = std::get<1>(elem_pair);

        //print_mat(Ke);

        #pragma omp critical
        {
        for(int ei = 0; ei < 20 *3; ++ei) for(int ej = 0; ej < 20 *3; ++ej)
        {
            spvals.push_back({elemseq[ei],elemseq[ej],Ke(ei,ej)});
        }
        }
    }

    K.setFromTriplets(spvals.begin(), spvals.end() );

    K.makeCompressed();

    printf("e time %lf\n", clock()*1.0/CLOCKS_PER_SEC);

}

#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkQuadraticTetra.h>
#include <vtkQuadraticHexahedron.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDoubleArray.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkPointData.h>

int main()
{
    omp_set_num_threads(8);

    Eigen::setNbThreads(8);
    printf("%d",Eigen::nbThreads());

    vtkNew<vtkUnstructuredGrid> vtkgrid;
    vtkNew<vtkXMLUnstructuredGridWriter> vtkwriter;
    vtkwriter->SetDataModeToBinary();

    FILE *meshfile = fopen("holemesh","r");

    fscanf(meshfile,"%d", &N_N);

    X.resize(N_N,3);
    K.resize(N_N*3,N_N*3);
    U.resize(N_N*3);
    f.resize(N_N*3);
    res.resize(N_N*3);
    f_target.resize(N_N*3);


    f_target.setConstant(0);
    U.setZero();

    for(int i = 0; i < N_N; ++i)
    {
        int ind;
        double x1,x2,x3;
        fscanf(meshfile,"%d,%lf,%lf,%lf",&ind,&x1,&x2,&x3);
        X(i,0) = x1; X(i,1) = x2; X(i,2) = x3;
    }

    /*
    for(int i = 0; i < NK; ++i)
    {
        for(int j = 0; j < NK; ++j)
        {
            X.coeffRef(i*NK+j,0) = 1.0*j/(NK-1);
            X.coeffRef(i*NK+j,1) = 1.0*i/(NK-1);

        }
    }*/

    vtkNew<vtkPoints> vtk_points;
    for(int i = 0; i < X.rows(); ++i)
        vtk_points->InsertNextPoint(X(i,0),X(i,1),X(i,2));
    vtkgrid->SetPoints(vtk_points);

    int N_E;
    fscanf(meshfile,"%d",&N_E);

    for(int i = 0; i < N_E; ++i)
    {
        int id;
        std::array<int,20> conn;
        fscanf(meshfile,"%d",&id);
        printf("%d\n",id);

        for(int n = 0; n < 20; ++n)
        {
            int x;
            fscanf(meshfile,",%d",&x);
            conn[n] = x-1;

        }
        elemlist.push_back(new ElemType(X(conn,Eigen::all) ) );
        elemconn.push_back(conn);
        //printf("%d %d %d %d %d %d\n",a,b,c,d,e,f);
    }
    puts("see");
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

    fclose(meshfile);

    for(int i = 0; i < elemconn.size(); ++i)
    {
        vtkNew<vtkQuadraticHexahedron> t;
        for(int k = 0; k < 20; ++k)
            t->GetPointIds()->SetId(k,elemconn[i][k]);
        vtkgrid->InsertNextCell(t->GetCellType(), t->GetPointIds());
    }



    printf("%d elements\n", elemlist.size());


    //Eigen::SimplicialLDLT< decltype(K) > solver;
    //Eigen::ConjugateGradient< decltype(K) > solver;
    //Eigen::UmfPackLU<decltype(K)> solver;
    //Eigen::CholmodSupernodalLLT<decltype(K)> solver;
    Eigen::CholmodDecomposition<decltype(K)> solver;


    double targetU = 0.0;

    for(int additer = 0; additer < 8; ++ additer)
    {
    targetU += 1.0/8;

    int iternum = 10;

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
                K.coeffRef(i*3,i*3) = 1e10;
                res[i*3] = -1e10*(U[i*3]-0.0);
                if(X.coeff(i,0)>1.0-1e-5)
                    res[i*3] = -1e10*(U[i*3]-targetU);

                K.coeffRef(i*3+1,i*3+1) = 1e10;
                res[i*3+1] = -1e10*(U[i*3+1]-0.0);
                K.coeffRef(i*3+2,i*3+2) = 1e10;
                res[i*3+2] = -1e10*(U[i*3+2]-0.0);
            }
            else
                fres += abs(res(i));
        }



        Eigen::Matrix<double,DynDim,1> rhs_sol(U.size(),1);
        rhs_sol = res;

        solver.compute(K);
        printf("solver compute ret %d\n",solver.info());
        Eigen::Matrix<double,1,DynDim> sol = solver.solve(rhs_sol);

        U += sol.transpose();

        printf("Max Du %lf\n", sol.norm());
        //printf("f res %lf\n", fres);

        vtkNew<vtkDoubleArray> soldata;
        soldata->SetName("Displacement");
        soldata->SetNumberOfComponents(3);
        for(int i = 0; i < X.rows(); ++i)
        {
            soldata->InsertNextTuple3(U[3*i+0],U[3*i+1],U[3*i+2]);
        }
        vtkgrid->GetPointData()->RemoveArray(soldata->GetName());
        vtkgrid->GetPointData()->AddArray(soldata);

        char framename[50];
        sprintf(framename,"output_%03d.vtu",additer);
        vtkwriter->SetFileName(framename);
        vtkwriter->SetInputData(vtkgrid);
        vtkwriter->Write();

        if(sol.norm() < 1e-4)
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
