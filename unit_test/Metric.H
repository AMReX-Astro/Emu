//singularities not handled yet



#ifndef METRIC_H_
#define METRIC_H_


#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>
#include <AMReX_SPACE.H>
//#include "../Source/FlavoredNeutrinoContainer.H"
#include <cmath>



//FlavoredNeutrinoContainer::ParticleType p;


using FourVec = amrex::GpuArray<amrex::Real, 4>;
using Mat4 = amrex::GpuArray<amrex::GpuArray<amrex::Real, 4>, 4>;
//using Mat8 = amrex::GpuArray<amrex::GpuArray<amrex::Real, 8>, 8>;
using Christoffel = amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, 4>, 4>, 4>;
using GeodesicArray = amrex::GpuArray<amrex::Real, 8>; //for 4-position and then 4-momentum



class DummyParticle
{
public:
    amrex::Real time;
    amrex::Real x1;
    amrex::Real x2;
    amrex::Real x3;
    amrex::Real pupt;
    amrex::Real pupx1;
    amrex::Real pupx2;
    amrex::Real pupx3;
};


//abstract base class- Metric

class Metric
{
public:
    
    virtual ~Metric() = default;
    
    //static amrex::Real detg(const amrex::RealVect& x);
    
    //convert 4-postion and 4-momentum from cartesian to chosen coordinate
    virtual void coord_conv(DummyParticle& p) const=0;   
    
    //convert 4-postion and 4-momentum from chosen coordinate to cartesian
    virtual void coord_conv_inv(DummyParticle& p) const = 0;

    
    //Jacobian needed for the forward conversion
    virtual Mat4 jacobian(const DummyParticle& p) const=0;
    
    //Jacobian for the inverse conversion 
    virtual Mat4 jacobian_inv(const DummyParticle& p) const = 0;

    


    //The metric 
    virtual  Mat4 gmunu(const DummyParticle& p) const=0;

    //det of the metric
    virtual  amrex::Real detg(const DummyParticle& p) const=0;

    //dot product of two 4-vectors
    amrex::Real dot(const FourVec& a,
                            const FourVec& b,
                            const DummyParticle& p) const

    {
        const Mat4 g = gmunu(p);
        amrex::Real s = 0.0;
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                s += g[mu][nu] * a[mu] * b[nu];
            }
        }
        return s;
    }    
    
    //function to get the Christoffel symbols
    virtual Christoffel christoffel(const DummyParticle& p) const = 0;
    

    //rhs of the geodesic equations
    GeodesicArray geodesic_rhs( DummyParticle& p) const
    {  
        

        coord_conv(p); 

        const Christoffel Gamma = christoffel(p);

        GeodesicArray Y{
            p.time,
            p.x1,
            p.x2,
            p.x3,
            p.pupt,
            p.pupx1,
            p.pupx2,
            p.pupx3

        };

        GeodesicArray rhs{};

        for (int mu = 0; mu < 4; mu++) {
            rhs[mu] = Y[4+mu]/ Y[4];
        }

        for (int mu = 0; mu < 4; ++mu) {
            rhs[4 + mu] = 0.0;
            for (int alpha = 0; alpha < 4; ++alpha) {
                for (int beta = 0; beta < 4; ++beta) {
                    rhs[4 + mu] -= Gamma[mu][alpha][beta] * Y[4+alpha] * Y[4+beta] / Y[4];
                }
            }
        }
        
        coord_conv_inv(p);

        return rhs;
    }


    virtual amrex::Real vol(amrex::Real x1_max,amrex::Real x1_min,amrex::Real x2_max,amrex::Real x2_min,amrex::Real x3_max,amrex::Real x3_min) const = 0;
                   
};






// child class- CartesianMetric

class CartesianMetric : public Metric
{
public:

    void coord_conv( DummyParticle& p) const override
    {}

    void coord_conv_inv(DummyParticle& p) const override
    {}

    Mat4 jacobian(const DummyParticle& p) const override
    {
        Mat4 J{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                J[mu][nu] = 0.0;
            }
        }

        J[0][0] = 1.0;
        J[1][1] = 1.0;
        J[2][2] = 1.0;
        J[3][3] = 1.0;


        return J;
    }
    

    Mat4 jacobian_inv(const DummyParticle& p) const override
    {
        Mat4 J{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                J[mu][nu] = 0.0;
            }
        }

        J[0][0] = 1.0;
        J[1][1] = 1.0;
        J[2][2] = 1.0;
        J[3][3] = 1.0;


        return J;
    }
    
    Mat4 gmunu(const DummyParticle& p) const override
    {
        Mat4 g{};
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) g[mu][nu] = 0.0;
        }
        g[0][0] = -1.0;
        g[1][1] =  1.0;
        g[2][2] =  1.0;
        g[3][3] =  1.0;
        return g;
    }

    amrex::Real detg(const DummyParticle& p) const override
    {
        return -1.0;
    }
    
    Christoffel christoffel(const DummyParticle& p) const override
    {
        Christoffel Gamma{};

        for (int alpha = 0; alpha < 4; ++alpha) {
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    Gamma[alpha][mu][nu] = 0.0;
                }
            }
        }

        return Gamma;
    }
    
    amrex::Real vol(amrex::Real x1_max, amrex::Real x1_min, amrex::Real x2_max, amrex::Real x2_min, amrex::Real x3_max, amrex::Real x3_min) const override
    {
        const amrex::Real v = (x1_max - x1_min)*(x2_max-x2_min)*(x3_max-x3_min);

        return v;
        
    }

};






// child class- CylindricalMetric

class CylindricalMetric : public Metric
{
public:

    void coord_conv(DummyParticle& p) const override
    {  
        
        //position coord conversion
        const amrex::Real r = std::sqrt(p.x1*p.x1 + p.x2*p.x2);  // r= sqrt(x^2 + y^2)
        const amrex::Real phi = std::atan2(p.x2,p.x1);           // phi= arctan(y/x)
        
        p.x1 = r;     
        p.x2 = phi;
       
        //momentum coord conversersion
        const Mat4 J = jacobian(p);
        
        FourVec p_old = {p.pupt, p.pupx1, p.pupx2, p.pupx3};  // original 4-momentum in cartesian

        FourVec p_new{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                p_new[mu] += J[mu][nu] * p_old[nu];
            }
        }

        p.pupt = p_new[0];
        p.pupx1= p_new[1];
        p.pupx2= p_new[2];
        p.pupx3= p_new[3];

    }

    void coord_conv_inv(DummyParticle& p) const override
    {
        
        //momentum coord inverse conversersion
        const Mat4 J = jacobian_inv(p);
        
        FourVec p_old = {p.pupt, p.pupx1, p.pupx2, p.pupx3};  // original 4-momentum in cartesian

        FourVec p_new{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                p_new[mu] += J[mu][nu] * p_old[nu];
            }
        }

        p.pupt = p_new[0];
        p.pupx1= p_new[1];
        p.pupx2= p_new[2];
        p.pupx3= p_new[3];

        //position coord inverse conversion
        const amrex::Real x = p.x1*std::cos(p.x2);    //x=r*cos(phi)
        const amrex::Real y = p.x1*std::sin(p.x2);    //y=r*sin(phi)
        
        p.x1 = x;     
        p.x2 = y;
       

    }


    Mat4 jacobian(const DummyParticle& p) const override
    {
        

        Mat4 J{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                J[mu][nu] = 0.0;
            }
        }

        J[0][0] = 1.0;
        J[1][1] = std::cos(p.x2);          //cos(phi)       
        J[1][2] = std::sin(p.x2);          //sin(phi)        
        J[2][1] = -std::sin(p.x2) / p.x1;  //-sin(phi)/r 
        J[2][2] =  std::cos(p.x2) / p.x1;  //cos(phi)/r            
        J[3][3] = 1.0;

        return J;
    }

    Mat4 jacobian_inv(const DummyParticle& p) const override
    {

        Mat4 J{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                J[mu][nu] = 0.0;
            }
        }

        J[0][0] = 1.0;
        J[1][1] = std::cos(p.x2);          //cos(phi)
        J[1][2] = -p.x1*std::sin(p.x2);    //-r*sin(phi)    
        J[2][1] = std::sin(p.x2);          //sin(phi)
        J[2][2] =  p.x1*std::cos(p.x2);    //r*cos(phi)
        J[3][3] = 1.0;

        return J;
    }
 
    
    Mat4 gmunu(const DummyParticle& p) const override
    {   
        const amrex::Real x = p.x1;
        const amrex::Real y = p.x2;
        const amrex::Real r = std::sqrt(x*x + y*y);

        Mat4 g{};
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) g[mu][nu] = 0.0;
        }
        g[0][0] = -1.0;
        g[1][1] =  1.0;
        g[2][2] =  r*r;
        g[3][3] =  1.0;
        return g;
    }

    amrex::Real detg(const DummyParticle& p) const override
    {
        const amrex::Real x = p.x1;
        const amrex::Real y = p.x2;
        const amrex::Real r = std::sqrt(x*x + y*y);
          
        return -(r * r);
    }

    Christoffel christoffel(const DummyParticle& p) const override
    {

        Christoffel Gamma{};

        for (int alpha = 0; alpha < 4; ++alpha) {
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    Gamma[alpha][mu][nu] = 0.0;
                }
            }
        }

        // coordinate order: (t, r, phi, z)
        Gamma[1][2][2] = -p.x1;          // -r   
        Gamma[2][1][2] =  1.0 / p.x1;    // 1/r
        Gamma[2][2][1] =  1.0 / p.x1;    // 1/r

        return Gamma;
    }

    amrex::Real vol(amrex::Real x1_max,amrex::Real x1_min,amrex::Real x2_max,amrex::Real x2_min,amrex::Real x3_max,amrex::Real x3_min) const override
    {
        const amrex::Real v = (x1_max*x1_max - x1_min*x1_min)*(x2_max-x2_min)*(x3_max-x3_min) / 2.0;

        return v;
        
    }

    


};



// child class- SphericalMetric

class SphericalMetric : public Metric
{
public:

    void coord_conv(DummyParticle& p) const override
    {  
        
        //position coord conversion
        const amrex::Real r = std::sqrt(p.x1*p.x1 + p.x2*p.x2 + p.x3*p.x3);  // r= sqrt(x^2 + y^2 + z^2)
        const amrex::Real theta = std::acos(p.x3/r);                         //theta= arccos(z/r)
        const amrex::Real phi = std::atan2(p.x2,p.x1);                       // phi= arctan(y/x)
        
        p.x1 = r;     
        p.x2 = theta;
        p.x3 = phi;
       
        //momentum coord conversersion
        const Mat4 J = jacobian(p);
        
        FourVec p_old = {p.pupt, p.pupx1, p.pupx2, p.pupx3};  // original 4-momentum in cartesian

        FourVec p_new{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                p_new[mu] += J[mu][nu] * p_old[nu];
            }
        }

        p.pupt = p_new[0];
        p.pupx1= p_new[1];
        p.pupx2= p_new[2];
        p.pupx3= p_new[3];

    }

    void coord_conv_inv(DummyParticle& p) const override
    {
        
        //momentum coord inverse conversersion
        const Mat4 J = jacobian_inv(p);
        
        FourVec p_old = {p.pupt, p.pupx1, p.pupx2, p.pupx3};  // original 4-momentum in cartesian

        FourVec p_new{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                p_new[mu] += J[mu][nu] * p_old[nu];
            }
        }

        p.pupt = p_new[0];
        p.pupx1= p_new[1];
        p.pupx2= p_new[2];
        p.pupx3= p_new[3];

        //position coord inverse conversion
        const amrex::Real x = p.x1*std::sin(p.x2)*std::cos(p.x3);    //x=r*sin(theta)*cos(phi)
        const amrex::Real y = p.x1*std::sin(p.x2)*std::sin(p.x3);    //y=r*sin(theta)*sin(phi)
        const amrex::Real z = p.x1*std::cos(p.x2);                   //z=r*cos(theta)


        p.x1 = x;     
        p.x2 = y;
        p.x3 = z;
       

    }


    Mat4 jacobian(const DummyParticle& p) const override
    {
        

        Mat4 J{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                J[mu][nu] = 0.0;
            }
        }

        J[0][0] = 1.0;
        J[1][1] = std::sin(p.x2)*std::cos(p.x3);          //cos(phi)       
        J[1][2] = std::sin(p.x2)*std::sin(p.x3);          //sin(phi)      
        J[1][3] = std::cos(p.x2);                         //cos(theta)
        J[2][1] = std::cos(p.x2)*std::cos(p.x3) / p.x1;   //cos(theta)cos(phi)/r 
        J[2][2] = std::cos(p.x2)*std::sin(p.x3) / p.x1;   //cos(theta)sin(phi)/r
        J[2][3] = -std::sin(p.x2) / p.x1;                 //-sin(theta)/r
        J[3][1] = -std::sin(p.x3)/ (p.x1*std::sin(p.x2)); //-sin(phi)/(r sin(theta))
        J[3][2] = std::cos(p.x3)/ (p.x1*std::sin(p.x2));  //cos(phi)/( r sin(theta))

        return J;
    }

    Mat4 jacobian_inv(const DummyParticle& p) const override
    {

        Mat4 J{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                J[mu][nu] = 0.0;
            }
        }

        J[0][0] = 1.0;
        J[1][1] = std::sin(p.x2)*std::cos(p.x3);            //sin(theta)cos(phi)
        J[1][2] = p.x1*std::cos(p.x2)*std::cos(p.x3);       //r*cos(theta)*cos(phi)  
        J[1][3] = -p.x1*std::sin(p.x2)*std::sin(p.x3);      //r*cos(theta)*cos(phi)   
        J[2][1] = std::sin(p.x2)*std::sin(p.x3);            //siin(theta)sin(phi)
        J[2][2] = p.x1*std::cos(p.x2)*std::sin(p.x3);       //r*cos(theta)*sin(phi)
        J[2][3] = p.x1*std::sin(p.x2)*std::cos(p.x3);       //r*sin(theta)cos(phi)
        J[3][1] = std::cos(p.x2);                           //cos(ttheta)
        J[3][2] = -p.x1*std::sin(p.x2);                     //-r*sin(theta)

        return J;
    }
 
    
    Mat4 gmunu(const DummyParticle& p) const override
    {   
        const amrex::Real x = p.x1;
        const amrex::Real y = p.x2;
        const amrex::Real z = p.x3;
        const amrex::Real r = std::sqrt(x*x + y*y + z*z);
        const amrex::Real theta = std::acos(z/r);

        Mat4 g{};
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) g[mu][nu] = 0.0;
        }
        g[0][0] = -1.0;
        g[1][1] =  1.0;
        g[2][2] =  r*r;
        g[3][3] =  r*r*std::sin(theta)*std::sin(theta);
        return g;
    }

    amrex::Real detg(const DummyParticle& p) const override
    {
        const amrex::Real x = p.x1;
        const amrex::Real y = p.x2;
        const amrex::Real z = p.x3;
        const amrex::Real r = std::sqrt(x*x + y*y + z*z);
        const amrex::Real theta = std::acos(z/r);
          
        return -std::pow(r,4)*std::sin(theta)*std::sin(theta);
    }

    Christoffel christoffel(const DummyParticle& p) const override
    {
        
        const amrex::Real x = p.x1;
        const amrex::Real y = p.x2;
        const amrex::Real z = p.x3;
        const amrex::Real r = std::sqrt(x*x + y*y + z*z);
        const amrex::Real theta = std::acos(z/r);

        Christoffel Gamma{};

        for (int alpha = 0; alpha < 4; ++alpha) {
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    Gamma[alpha][mu][nu] = 0.0;
                }
            }
        }

        // coordinate order: (t, r, theta, phi)
        Gamma[1][2][2] = -r;             
        Gamma[1][3][3] = -r*std::sin(theta)*std::sin(theta);
        Gamma[2][1][2] =  1.0/r;   
        Gamma[2][2][1] =  1.0/r;    
        Gamma[2][3][3] = -std::sin(theta)*std::cos(theta);
        Gamma[3][1][3] = 1.0/r;
        Gamma[3][3][1] = 1.0/r;
        Gamma[3][2][3] = 1.0/std::tan(theta);
        Gamma[3][3][2] = 1.0/std::tan(theta);



        return Gamma;
    }
    
    amrex::Real vol(amrex::Real x1_max,amrex::Real x1_min,amrex::Real x2_max,amrex::Real x2_min,amrex::Real x3_max,amrex::Real x3_min) const override
    {
        const amrex::Real v = (std::pow(x1_max,3.0) - std::pow(x1_min,3.0))*(std::cos(x2_min)-std::cos(x2_max))*(x3_max-x3_min) / 3.0;

        return v;
        
    }
    


};


#endif