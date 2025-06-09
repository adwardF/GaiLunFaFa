#ifndef USUAL_ELEMENTS_H
#define USUAL_ELEMENTS_h

#include "ElementBase.h"

class LTri2D : public GeomElementBase<3,2,1>
{
public:
    LTri2D( const Mat<3,2> &elemX ):
        GeomElementBase<3,2,1>(elemX)
    { ; };

    virtual Mat<DynDim, 3>
        compute_N(const Mat<DynDim, 2> &xi) const
    {
        Mat<DynDim, 3> r(xi.rows(),3);
        for(int n = 0; n < xi.rows(); ++n)
        {
            r.coeffRef(n,0) = xi.coeffRef(n,0);
            r.coeffRef(n,1) = xi.coeffRef(n,1);
            r.coeffRef(n,2) = 1.0-xi.coeffRef(n,0)-xi.coeffRef(n,1);
        }
        return r;
    }

    virtual Mat<DynDim, 2*3>
        compute_Nxi(const Mat<DynDim, 2> &xi) const
    {
        Mat<DynDim, 6> r(xi.rows(),6);
        for(int n = 0; n < xi.rows(); ++n)
        {
            r.coeffRef(n,0) = 1.0;
            r.coeffRef(n,1) = 0.0;
            r.coeffRef(n,2) = 0.0;
            r.coeffRef(n,3) = 1.0;
            r.coeffRef(n,4) =-1.0;
            r.coeffRef(n,5) =-1.0;
        }
        return r;
    }

    virtual Mat<1,2> get_ip_xi() const
    {
        return Mat<1,2>{{1.0/3,1.0/3}};
    };
    virtual Mat<1,1> get_ip_w() const
    {
        return Mat<1,1>{0.5};
    }
    virtual Mat<1,3*2> get_ip_Bxi() const
    {
        return Mat<1,3*2>{{1.0, 0.0, 0.0, 1.0, -1.0, -1.0}};
    };

    ~LTri2D() {};

private:

};


class QTri2D : public GeomElementBase<6,2,3>
{
public:
    QTri2D( const Mat<6,2> &elemX ):
        GeomElementBase<6,2,3>(elemX)
    { ; };

    virtual Mat<DynDim, 6>
        compute_N(const Mat<DynDim, 2> &xi) const
    {
        Mat<DynDim, 6> r(xi.rows(),6);
        for(int n = 0; n < xi.rows(); ++n)
        {
            double xi1 = xi(n,0),xi2=xi(n,1);
            double xi3=1-xi1-xi2;

            r(n,0) = xi1*(2*xi1-1);
            r(n,1) = xi2*(2*xi2-1);
            r(n,2) = xi3*(2*xi3-1);
            r(n,3) = 4*xi1*xi2;
            r(n,4) = 4*xi1*xi3;
            r(n,5) = 4*xi2*xi3;
        }
        return r;
    }

    virtual Mat<DynDim, 6*2>
        compute_Nxi(const Mat<DynDim, 2> &xi) const
    {
        Mat<DynDim, 6*2> r(xi.rows(),6*2);
        r.setZero();
        for(int n = 0; n < xi.rows(); ++n)
        {
            double xi1 = xi(n,0),xi2=xi(n,1);
            double xi3=1-xi1-xi2;
            //!--------------------
            r(n,0) = 4*xi1-1;



            r(n,3) = 4*xi2-1;

            r(n,4) =-(4*xi3-1);
            r(n,5) =-(4*xi3-1);

            r(n,6) = 4*xi2;
            r(n,7) = 4*xi1;

            r(n,8) =-4*xi2;
            r(n,9) = 4*xi3-4*xi2;

            r(n,10) = 4*xi3-4*xi1;
            r(n,11) =-4*xi1;

        }
        return r;
    }

    virtual Mat<3,2> get_ip_xi() const
    {
        return Mat<3,2>{{0.0,0.5},{0.5,0.0},{0.5,0.5}};
    };
    virtual Mat<1,3> get_ip_w() const
    {
        return Mat<1,3>{{0.5/3,0.5/3,0.5/3}};
    }
    virtual Mat<3,6*2> get_ip_Bxi() const
    {
        return Mat<3,12>{{-1.0, 0.0, 0.0, 1.0, -1.0, -1.0,\
                            2.0, 0.0, -2.0, 0.0, 2.0, -0.0},\
                          { 1.0, 0.0, 0.0, -1.0, -1.0, -1.0, \
                            0.0, 2.0, -0.0, 2.0, 0.0, -2.0},
                          { 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, \
                            2.0, 2.0, -2.0, -2.0, -2.0, -2.0} };

    };

    ~QTri2D() {};

private:

};


class QTet3D : public GeomElementBase<10,3,4>
{
public:
    QTet3D( const Mat<10,3> &elemX ):
        GeomElementBase<10,3,4>(elemX)
    { ; };

    virtual Mat<DynDim, 10>
        compute_N(const Mat<DynDim, 3> &xi) const
    {
        Mat<DynDim, 10> N(xi.rows(),10);
        N.setZero();
        for(int n = 0; n < xi.rows(); ++n)
        {
            double r = xi(n,0),s=xi(n,1),t=xi(n,2);
            double u=1.0-r-s-t;

            N(n,0) = (2*r-1)*r;
            N(n,1) = (2*s-1)*s;
            N(n,2) = (2*t-1)*t;
            N(n,3) = (2*u-1)*u;

            N(n,4) = 4*r*s;
            N(n,5) = 4*s*t;
            N(n,6) = 4*r*t;
            N(n,7) = 4*u*r;
            N(n,8) = 4*u*s;
            N(n,9) = 4*u*t;
        }
        return N;
    }

    virtual Mat<DynDim, 10*3>
        compute_Nxi(const Mat<DynDim, 3> &xi) const
    {
        Mat<DynDim, 10*3> R(xi.rows(),10*3);
        R.setZero();
        for(int n = 0; n < xi.rows(); ++n)
        {
            double r = xi(n,0),s=xi(n,1),t=xi(n,2);
            double u=1.0-r-s-t;
            //!--------------------

            R(n,0*3+0) = 4 * r - 1;

            R(n,1*3+1) = 4 * s - 1;

            R(n,2*3+2) = 4 * t - 1;

            R(n,3*3+0) =-(4*u-1);
            R(n,3*3+1) =-(4*u-1);
            R(n,3*3+2) =-(4*u-1);

            R(n,4*3+0) = 4*s;
            R(n,4*3+1) = 4*r;

            R(n,5*3+1) = 4*t;
            R(n,5*3+2) = 4*s;

            R(n,6*3+0) = 4*t;
            R(n,6*3+2) = 4*r;

            R(n,7*3+0) = 4-8*r-4*s-4*t;
            R(n,7*3+1) = -4*r;
            R(n,7*3+2) = -4*r;

            R(n,8*3+0) = -4*s;
            R(n,8*3+1) = 4-8*s-4*r-4*t;
            R(n,8*3+2) = -4*s;

            R(n,9*3+0) = -4*t;
            R(n,9*3+1) = -4*t;
            R(n,9*3+2) = 4-8*t-4*r-4*s;

        }
        return R;
    }

    virtual Mat<4,3> get_ip_xi() const
    {
        const double a = 0.58541020, b = 0.13819660;
        return Mat<4,3>{{a,b,b},{b,a,b},{b,b,a},{b,b,b}};
    };
    virtual Mat<1,4> get_ip_w() const
    {
        return Mat<1,4>{{1.0/24,1.0/24,1.0/24,1.0/24}};
    }
    virtual Mat<4,10*3> get_ip_Bxi() const
    {
        return compute_Nxi( get_ip_xi() );
    };

    ~QTet3D() {};

private:

};



class QRHex3D : public GeomElementBase<20,3,8>
{
public:
    QRHex3D( const Mat<20,3> &elemX ):
        GeomElementBase<20,3,8>(elemX)
    { ; };

    virtual Mat<DynDim, 20>
        compute_N(const Mat<DynDim, 3> &xi) const
    {
        Mat<DynDim, 20> N(xi.rows(), 20);
        const auto xi0 = Mat<8,3>\
        {
            { 1.0, 1.0, 1.0}, {-1.0, 1.0, 1.0}, {-1.0,-1.0, 1.0}, { 1.0,-1.0, 1.0},
            { 1.0, 1.0,-1.0}, {-1.0, 1.0,-1.0}, {-1.0,-1.0,-1.0}, { 1.0,-1.0,-1.0}
        };

        for(int n=0; n<xi.rows(); ++n) {
            const double r = xi(n,0), s = xi(n,1), t = xi(n,2);

            // Corner nodes (1-8)
            for(int i=0; i<8; ++i) {
                const double ri = xi0(i,0), si = xi0(i,1), ti = xi0(i,2);
                N(n,i) = 0.125*(1 + r*ri)*(1 + s*si)*(1 + t*ti)*(r*ri + s*si + t*ti - 2);
            }

            // Edge nodes (9-20)
            N(n,8)  = (1-r*r)*(1+s)*(1+t)/4;
            N(n,9)  = (1-s*s)*(1-r)*(1+t)/4;
            N(n,10) = (1-r*r)*(1-s)*(1+t)/4;
            N(n,11) = (1-s*s)*(1+r)*(1+t)/4;
            N(n,12) = (1-r*r)*(1+s)*(1-t)/4;
            N(n,13) = (1-s*s)*(1-r)*(1-t)/4;
            N(n,14) = (1-r*r)*(1-s)*(1-t)/4;
            N(n,15) = (1-s*s)*(1+r)*(1-t)/4;
            N(n,16) = (1-t*t)*(1+r)*(1+s)/4;
            N(n,17) = (1-t*t)*(1-r)*(1+s)/4;
            N(n,18) = (1-t*t)*(1-r)*(1-s)/4;
            N(n,19) = (1-t*t)*(1+r)*(1-s)/4;
        }
        return N;
    }

    virtual Mat<DynDim, 20*3>
        compute_Nxi(const Mat<DynDim, 3> &xi) const
    {
        Mat<DynDim, 20*3> R(xi.rows(), 20*3);
        const auto xi0 = Mat<8,3>\
        {
            { 1.0, 1.0, 1.0}, {-1.0, 1.0, 1.0}, {-1.0,-1.0, 1.0}, { 1.0,-1.0, 1.0},
            { 1.0, 1.0,-1.0}, {-1.0, 1.0,-1.0}, {-1.0,-1.0,-1.0}, { 1.0,-1.0,-1.0}
        };

        for(int n=0; n<xi.rows(); ++n)
        {
            const double r = xi(n,0), s = xi(n,1), t = xi(n,2);

            // Corner nodes (1-8)
            for(int i=0; i<8; ++i)
            {
                const double ri = xi0(i,0), si = xi0(i,1), ti = xi0(i,2);
                R(n,i*3+0) = 0.125*(1 + s*si)*(1 + t*ti)*(2*r+ri*si*s+ri*ti*t-ri);
                R(n,i*3+1) = 0.125*(1 + r*ri)*(1 + t*ti)*(2*s+si*ri*r+si*ti*t-si);
                R(n,i*3+2) = 0.125*(1 + r*ri)*(1 + s*si)*(2*t+ti*ri*r+ti*si*s-ti);
            }
            R(n,3*8+0)  = -2*r*(1+s)*(1+t)/4;
            R(n,3*8+1)  = (1-r*r)*(1+t)/4;
            R(n,3*8+2)  = (1-r*r)*(1+s)/4;
            R(n,3*10+0) = -2*r*(1-s)*(1+t)/4;
            R(n,3*10+1) =-(1-r*r)*(1+t)/4;
            R(n,3*10+2) = (1-r*r)*(1-s)/4;
            R(n,3*14+0) = -2*r*(1-s)*(1-t)/4;
            R(n,3*14+1) =-(1-r*r)*(1-t)/4;
            R(n,3*14+2) =-(1-r*r)*(1-s)/4;
            R(n,3*12+0) = -2*r*(1+s)*(1-t)/4;
            R(n,3*12+1) = (1-r*r)*(1-t)/4;
            R(n,3*12+2) =-(1-r*r)*(1+s)/4;

            R(n,3*11+0) = (1-s*s)*(1+t)/4;
            R(n,3*11+1) = -2*s*(1+r)*(1+t)/4;
            R(n,3*11+2) = (1-s*s)*(1+r)/4;
            R(n,3*9+0) =-(1-s*s)*(1+t)/4;
            R(n,3*9+1) = -2*s*(1-r)*(1+t)/4;
            R(n,3*9+2) = (1-s*s)*(1-r)/4;
            R(n,3*13+0) =-(1-s*s)*(1-t)/4;
            R(n,3*13+1) = -2*s*(1-r)*(1-t)/4;
            R(n,3*13+2) =-(1-s*s)*(1-r)/4;
            R(n,3*15+0) = (1-s*s)*(1-t)/4;
            R(n,3*15+1) = -2*s*(1+r)*(1-t)/4;
            R(n,3*15+2) =-(1-s*s)*(1+r)/4;

            R(n,3*16+0) = (1-t*t)*(1+s)/4;
            R(n,3*16+1) = (1-t*t)*(1+r)/4;
            R(n,3*16+2) = -2*t*(1+r)*(1+s)/4;
            R(n,3*17+0) =-(1-t*t)*(1+s)/4;
            R(n,3*17+1) = (1-t*t)*(1-r)/4;
            R(n,3*17+2) = -2*t*(1-r)*(1+s)/4;
            R(n,3*18+0) =-(1-t*t)*(1-s)/4;
            R(n,3*18+1) =-(1-t*t)*(1-r)/4;
            R(n,3*18+2) = -2*t*(1-r)*(1-s)/4;
            R(n,3*19+0) = (1-t*t)*(1-s)/4;
            R(n,3*19+1) =-(1-t*t)*(1+r)/4;
            R(n,3*19+2) = -2*t*(1+r)*(1-s)/4;
        }

        return R;
    }

    virtual Mat<8,3> get_ip_xi() const
    {
        const double g = 0.5773502692;
        return Mat<8,3>{{g,g,g},{g,g,-g},{g,-g,g},{g,-g,-g},\
                        {-g,g,g},{-g,g,-g},{-g,-g,g},{-g,-g,-g}};
    };
    virtual Mat<1,8> get_ip_w() const
    {
        return Mat<1,8>{{1.0/8,1.0/8,1.0/8,1.0/8, \
                         1.0/8,1.0/8,1.0/8,1.0/8 }};
    }
    virtual Mat<8,20*3> get_ip_Bxi() const
    {
        return compute_Nxi( get_ip_xi() );
    };

    ~QRHex3D() {};

private:

};




template<typename geom_type>
class PoissonElem : public ElementBase<geom_type,1>
{
public:

    using base_type = ElementBase<geom_type,1>;

    PoissonElem(const Mat<geom_type::Nn,geom_type::Ndim> & elemX):
        base_type(elemX)
    {};

    ~PoissonElem() {};

    virtual std::tuple<Mat<1,base_type::Ndof>,
                       Mat<base_type::Ndof,
                           base_type::Ndof>>
        make_f_and_DfDu(const Mat<1,base_type::Ndof> & nodeU) const
    {
        assert( this->ip_data_computed );

        Mat<1,base_type::Ndof> fe;
        Mat<base_type::Ndof, base_type::Ndof> Ke;

        fe.setZero();
        Ke.setZero();

        for(int np = 0; np < geom_type::Nip; ++np)
        {
            const auto & Bx = this->Bip[np];

            Mat<1,geom_type::Ndim> ip_grad = nodeU * Bx;

            Mat<geom_type::Nn,geom_type::Nn> Kip = Bx * Bx.transpose();

            fe = fe + this->Wip[np]*ip_grad * Bx.transpose();

            Ke = Ke + this->Wip[np]*Kip;
        }
        return std::make_tuple(fe, Ke);
    }
};


template <typename geom_type>
class LaPEElem : public ElementBase<geom_type,2>
{
public:
    using base_type = ElementBase<geom_type,2>;

    LaPEElem(const Mat<geom_type::Nn,geom_type::Ndim> & elemX):
        base_type(elemX)
    {};

    ~LaPEElem() {};

    virtual std::tuple<Mat<1,3>,
                       Mat<3,3>> make_T_and_TE(const Mat<2,2> &F) const;

    virtual std::tuple<Mat<1,base_type::Ndof>,
                       Mat<base_type::Ndof,
                           base_type::Ndof>>
        make_f_and_DfDu(const Mat<1,base_type::Ndof> & nodeU) const
    {

        Mat<1,base_type::Ndof> fe;
        Mat<base_type::Ndof, base_type::Ndof> Ke;

        fe.setZero();
        Ke.setZero();

        Mat<geom_type::Nn, 2> Ut = nodeU.template reshaped<Eigen::AutoOrder>(geom_type::Nn, 2);

        for(int np = 0; np < geom_type::Nip; ++np)
        {
            const auto & Bx = this->Bip[np];

            Mat<2,2> F = Ut.transpose() * Bx;
            F(0,0) += 1.0; F(1,1) += 1.0;

            auto [T,T_E] = this->make_T_and_TE(F);

            Mat<3,geom_type::Nn*2> H;
            H.setZero();
            for(int i = 0; i < geom_type::Nn; ++i)
            {
                H( 0, i*2+0 ) = F(0,0) * Bx(i,0);
                H( 0, i*2+1 ) = F(1,0) * Bx(i,0);

                H( 1, i*2+0 ) = F(0,1) * Bx(i,1);
                H( 1, i*2+1 ) = F(1,1) * Bx(i,1);

                H( 2, i*2+0 ) = F(0,0)*Bx(i,1) + F(0,1)*Bx(i,0);
                H( 2, i*2+1 ) = F(1,0)*Bx(i,1) + F(1,1)*Bx(i,0);
            }

            fe = fe + this->Wip[np] * T * H;

            Mat<base_type::Ndof,base_type::Ndof> Kip = H.transpose() * T_E * H; // part 1

            Mat<base_type::Ndof,base_type::Ndof> K_stress;
            K_stress.setZero();

            for(int n = 0; n < geom_type::Nn; ++n) for(int m = 0; m < geom_type::Nn; ++m)
            {
                K_stress(n*2+0,m*2+0) += Bx(n,0)*Bx(m,0)*T[0];
                K_stress(n*2+0,m*2+0) += Bx(n,1)*Bx(m,1)*T[1];
                //K_stress(n*2+0,m*2+0) += 0.5*(Bx(n,1)*Bx(m,0)+Bx(n,0)*Bx(m,1))*T[2];
                K_stress(n*2+0,m*2+0) += (Bx(n,1)*Bx(m,0)+Bx(n,0)*Bx(m,1))*T[2];

                K_stress(n*2+1,m*2+1) = K_stress(n*2+0,m*2+0);
            }

            Kip += K_stress;

            Ke = Ke + this->Wip[np]*Kip;
        }
        return std::make_tuple(fe, Ke);
    }
};


template <typename geom_type>
class LaElem : public ElementBase<geom_type,3>
{
public:
    using base_type = ElementBase<geom_type,3>;

    LaElem(const Mat<geom_type::Nn,geom_type::Ndim> & elemX):
        base_type(elemX)
    {};

    ~LaElem() {};

    virtual std::tuple<Mat<1,6>,
                       Mat<6,6>> make_T_and_TE(const Mat<3,3> &F) const;

    virtual std::tuple<Mat<1,base_type::Ndof>,
                       Mat<base_type::Ndof,
                           base_type::Ndof>>
        make_f_and_DfDu(const Mat<1,base_type::Ndof> & nodeU) const
    {

        Mat<1,base_type::Ndof> fe;
        Mat<base_type::Ndof, base_type::Ndof> Ke;

        fe.setZero();
        Ke.setZero();

        Mat<geom_type::Nn, 3> Ut = nodeU.template reshaped<Eigen::AutoOrder>(geom_type::Nn, 3);

        for(int np = 0; np < geom_type::Nip; ++np)
        {
            const auto & Bx = this->Bip[np];

            Mat<3,3> F = Ut.transpose() * Bx + Mat<3,3>::Identity();

            auto [T,T_E] = this->make_T_and_TE(F);

            Mat<6,geom_type::Nn*3> H;
            H.setZero();
            for(int i = 0; i < geom_type::Nn; ++i)
            {
                for(int J=0;J<3;++J)
                {
                    H( J, i*3+0 ) = F(0,J) * Bx(i,J);
                    H( J, i*3+1 ) = F(1,J) * Bx(i,J);
                    H( J, i*3+2 ) = F(2,J) * Bx(i,J);
                }
                for(int J=3;J<6;++J)
                {
                    H( J, i*3+0 ) = F(0,TENSOR_MAT_IND[J][0])*Bx(i,TENSOR_MAT_IND[J][1])\
                                  + F(0,TENSOR_MAT_IND[J][1])*Bx(i,TENSOR_MAT_IND[J][0]);
                    H( J, i*3+1 ) = F(1,TENSOR_MAT_IND[J][0])*Bx(i,TENSOR_MAT_IND[J][1])\
                                  + F(1,TENSOR_MAT_IND[J][1])*Bx(i,TENSOR_MAT_IND[J][0]);
                    H( J, i*3+2 ) = F(2,TENSOR_MAT_IND[J][0])*Bx(i,TENSOR_MAT_IND[J][1])\
                                  + F(2,TENSOR_MAT_IND[J][1])*Bx(i,TENSOR_MAT_IND[J][0]);
                }
            }

            fe = fe + this->Wip[np] * T * H;

            Mat<base_type::Ndof,base_type::Ndof> Kip = H.transpose() * T_E * H; // part 1

            Mat<base_type::Ndof,base_type::Ndof> K_stress;
            K_stress.setZero();

            for(int n = 0; n < geom_type::Nn; ++n) for(int m = 0; m < geom_type::Nn; ++m)
            {
                K_stress(n*3+0,m*3+0) += Bx(n,0)*Bx(m,0)*T[0];
                K_stress(n*3+0,m*3+0) += Bx(n,1)*Bx(m,1)*T[1];
                K_stress(n*3+0,m*3+0) += Bx(n,2)*Bx(m,2)*T[2];
                K_stress(n*3+0,m*3+0) += (Bx(n,1)*Bx(m,0)+Bx(n,0)*Bx(m,1))*T[3];
                K_stress(n*3+0,m*3+0) += (Bx(n,2)*Bx(m,0)+Bx(n,0)*Bx(m,2))*T[4];
                K_stress(n*3+0,m*3+0) += (Bx(n,2)*Bx(m,1)+Bx(n,1)*Bx(m,2))*T[5];

                K_stress(n*3+1,m*3+1) = K_stress(n*3+0,m*3+0);
                K_stress(n*3+2,m*3+2) = K_stress(n*3+0,m*3+0);
            }

            Kip += K_stress;

            Ke = Ke + this->Wip[np]*Kip;
        }
        return std::make_tuple(fe, Ke);
    }
};

/*
// UNUSABLE: need F data
template <typename geom_type>
class EuPEElem : public ElementBase<geom_type,2>
{
public:
    using base_type = ElementBase<geom_type,2>;

    EuPEElem(const Mat<geom_type::Nn,geom_type::Ndim> & elemX):
        base_type(elemX)
    {};

    ~EuPEElem() {};

    virtual std::tuple<Mat<1,3>,
                       Mat<3,3>> make_Cauchy_and_Truesdell(const Mat<2,2> &F) const;

    virtual std::tuple<Mat<1,base_type::Ndof>,
                       Mat<base_type::Ndof,
                           base_type::Ndof>>
        make_f_and_DfDu(const Mat<1,base_type::Ndof> & nodeU) const
    {

        Mat<1,base_type::Ndof> fe;
        Mat<base_type::Ndof, base_type::Ndof> Ke;

        fe.setZero();
        Ke.setZero();

        Mat<geom_type::Nn, 2> Ut = nodeU.template reshaped<Eigen::AutoOrder>(geom_type::Nn, 2);

        for(int np = 0; np < geom_type::Nip; ++np)
        {
            const auto & Bx = this->Bip[np];

            Mat<2,2> F = Ut.transpose() * Bx;
            F(0,0) += 1.0; F(1,1) += 1.0;

            auto [S, K_Trues] = this->make_Cauchy_and_Truesdell(F);

            Mat<3,geom_type::Nn*2> H;
            H.setZero();
            for(int i = 0; i < geom_type::Nn; ++i)
            {
                H( 0, i*2+0 ) = Bx(i,0);
                //H( 0, i*2+1 ) = ;

                //H( 1, i*2+0 ) = ;
                H( 1, i*2+1 ) = Bx(i,1);

                H( 2, i*2+0 ) = Bx(i,1);
                H( 2, i*2+1 ) = Bx(i,0);
            }

            fe = fe + this->Wip[np] * S * H;

            Mat<base_type::Ndof,base_type::Ndof> Kip = H.transpose() * K_Trues * H; // part 1

            Mat<base_type::Ndof,base_type::Ndof> K_stress;
            K_stress.setZero();

            for(int n = 0; n < geom_type::Nn; ++n) for(int m = 0; m < geom_type::Nn; ++m)
            {
                K_stress(n*2+0,m*2+0) += Bx(n,0)*Bx(m,0)*S[0];
                K_stress(n*2+0,m*2+0) += Bx(n,1)*Bx(m,1)*S[1];
                K_stress(n*2+0,m*2+0) += (Bx(n,1)*Bx(m,0)+Bx(n,0)*Bx(m,1))*S[2];

                K_stress(n*2+1,m*2+1) = K_stress(n*2+0,m*2+0);
            }

            Kip += K_stress;

            Ke = Ke + this->Wip[np]*Kip;
        }
        return std::make_tuple(fe, Ke);
    }
};
*/

#endif // USUAL_ELEMENTS_H
