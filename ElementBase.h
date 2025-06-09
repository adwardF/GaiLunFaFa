#ifndef ELEMENTBASE_H
#define ELEMENTBASE_H

#include <concepts>

#include "common.h"

template<int Nn_tmpl, int Ndim_tmpl, int Nip_tmpl>
class GeomElementBase
{
public:
    //GeomElementBase() ;
    GeomElementBase( const Mat<Nn_tmpl,Ndim_tmpl> &elemX )
    {
        this->nodeX = elemX;
    };

    void setX( const Mat<Nn_tmpl,Ndim_tmpl> &elemX )
    {
        this->nodeX = elemX;
    }

    virtual ~GeomElementBase() {} ;

    virtual Mat<DynDim,Nn_tmpl> compute_N(const Mat<DynDim,Ndim_tmpl> &xi) const = 0;
    virtual Mat<DynDim,Nn_tmpl*Ndim_tmpl> compute_Nxi(const Mat<DynDim,Ndim_tmpl> &xi) const = 0;

    virtual Mat<Nip_tmpl,Ndim_tmpl> get_ip_xi() const = 0;
    virtual Mat<1,Nip_tmpl> get_ip_w() const = 0;
    virtual Mat<Nip_tmpl,Nn_tmpl*Ndim_tmpl> get_ip_Bxi() const = 0;
    //const XXX get_() {  };

    static const int Nn = Nn_tmpl;
    static const int Ndim = Ndim_tmpl;
    static const int Nip = Nip_tmpl;

    Mat<Nn_tmpl, Ndim_tmpl> nodeX;

    Mat<1,Nip_tmpl> Wip;

    Mat<Nn_tmpl, Ndim_tmpl> Bip[Nip_tmpl];

public:
    virtual void compute_ip_data()
    {
        //auto ip_xi = this->get_ip_xi();
        Mat<1,Nip_tmpl> ip_w = this->get_ip_w();
        Mat<Nip_tmpl,Nn_tmpl*Ndim_tmpl> Nxi = this->get_ip_Bxi();
        for(int np = 0; np < Nip_tmpl; ++np)
        {
            Mat<Nn_tmpl,Ndim_tmpl> curNxi = ( Nxi.row(np) ).template reshaped < Eigen::AutoOrder > (Nn_tmpl,Ndim_tmpl);

            Mat<Ndim_tmpl,Ndim_tmpl> X_xi = this->nodeX.transpose() * curNxi;

            double J = abs( X_xi.determinant() );

            auto xi_X = X_xi.inverse();

            Bip[np] = curNxi * xi_X;
            Wip[np] = ip_w.coeffRef(np)*J;
        }

    }

private:

};

//template<typename gelem_type>
//concept GeomElemConcept = std::derived_from<gelem_type,GeomElemBaseC>;


template<typename geom_elem_type, int Nuf_tmpl>
class ElementBase : public geom_elem_type
{
public:
    ElementBase(const Mat<geom_elem_type::Nn,geom_elem_type::Ndim> & elemX):
        geom_elem_type(elemX)
    {};

    ~ElementBase() {};

    static const unsigned int Ndof = geom_elem_type::Nn*Nuf_tmpl;

    virtual std::tuple<Mat<1,Ndof>,Mat<Ndof,Ndof>>
        make_f_and_DfDu(const Mat<1,Ndof> & nodeU) const;
};





#endif // ELEMENTBASE_H
