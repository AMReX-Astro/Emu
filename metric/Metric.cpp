//Forward Euler Geodesic equations solver test

#include "Metric.H"
#include <iostream>
#include <cmath>

/*
array<double, 8> F( const array<double, 8> Y)  //dY/dt= F(Y)  Y={t, r, phi, z, pupt, pupr, pupphi, pupz}
{
    array<double, 8> dydt
    

    dydt[0] = 1.0;                                  //dt/dt = 1
    dydt[1] = Y[5]/Y[4];                            //dr/dt = pupr/ pupt
    dydt[2] = Y[6]/Y[4];                            //dphi/dt = pupphi/ pupt
    dydt[3] = Y[7]/Y[4];                            //dz/dt = pupz/ pupt 
    dydt[4] = 0;                                    //dpupt/dt = 0
    dydt[5] = Y[1]/Y[4] * std::pow(pupphi, 2.0);    //dpupr/dt = r/pupt * pupphi^2
    dydt[6] = -2/(Y[1] * Y[4]) * Y[5] * Y[6];       //dupphi/dt = -2/(r * pupt) * pupr * pupphi 
    dydt[7] = 0;                                    //dpupz/dt = 0 

    return dydt;
}
*/

void Euler( DummyParticle& p, double dt, int steps)
{   
   

    CylindricalMetric metric;


    for(int i=0; i<steps; i++)
    {
        GeodesicArray dydt= metric.geodesic_rhs(p);
        metric.coord_conv(p);

        GeodesicArray Y= 
    {
        p.time, p.x1, p.x2, p.x3, p.pupt, p.pupx1, p.pupx2, p.pupx3
    };

        for( int j=0; j<8; j++)
        {
            Y[j] += dt * dydt[j];   // Y_(n+1)= Y_n + dt_n*dY/dt_n
        }
        p.time= Y[0];
        p.x1= Y[1];
        p.x2= Y[2];
        p.x3= Y[3];
        p.pupt= Y[4];
        p.pupx1= Y[5];
        p.pupx2= Y[6];
        p.pupx3= Y[7];

        metric.coord_conv_inv(p);
    }

} 


int main()
{
    DummyParticle p;

    p.time= 0.0;
    p.x1= 1.0;
    p.x2= 1.0;
    p.x3= 0.0;
    
    // the folloing is chosen to replicate null particle (p.p=0)
    p.pupt= 3.0;
    p.pupx1= 2.0;
    p.pupx2= 2.0;
    p.pupx3= 1.0;

    double t= 5;
    int steps= 100;
    double dt= t/steps;

    Euler(p,dt, steps);
    
    CylindricalMetric metric;
    //metric.coord_conv(p);

    std::cout << "final values\n";
    std::cout << "time = " << p.time  << "\n";
    std::cout << "x    = " << p.x1    << "\n";
    std::cout << "y    = " << p.x2    << "\n";
    std::cout << "z    = " << p.x3    << "\n";
    std::cout << "pt   = " << p.pupt  << "\n";
    std::cout << "px   = " << p.pupx1 << "\n";
    std::cout << "py   = " << p.pupx2 << "\n";
    std::cout << "pz   = " << p.pupx3 << "\n";

    



}