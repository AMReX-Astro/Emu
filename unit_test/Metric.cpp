//Forward Euler Geodesic equations solver test

#include "../Source/Metric.H"
#include <iostream>
#include <cmath>
#include <string>
#include <cassert>



void Euler( EParticle& p, Metric &metric, double dt, int steps)
{   
    // This function time integrates the geodesic equations using forward Euler method. 
    //EParticle is passed to this function in cartesian and leaves this function in cartesian.




    for(int i=0; i<steps; i++)
    {
        GeodesicArray dydt= metric.geodesic_rhs(p);
        metric.coord_conv(p);

        GeodesicArray Y= 
        {
        p.rdata(PIdx::time),
        p.rdata(PIdx::x),
        p.rdata(PIdx::y),
        p.rdata(PIdx::z),
        p.rdata(PIdx::pupt),
        p.rdata(PIdx::pupx),
        p.rdata(PIdx::pupy),
        p.rdata(PIdx::pupz)
        };

        for( int j=0; j<8; j++)
        {
            Y[j] += dt * dydt[j];   // Y_(n+1)= Y_n + dt_n*dY/dt_n
        }
        p.rdata(PIdx::time)= Y[0];
        p.rdata(PIdx::x)= Y[1];
        p.rdata(PIdx::y)= Y[2];
        p.rdata(PIdx::z)= Y[3];
        p.rdata(PIdx::pupt)= Y[4];
        p.rdata(PIdx::pupx)= Y[5];
        p.rdata(PIdx::pupy)= Y[6];
        p.rdata(PIdx::pupz)= Y[7];

        metric.coord_conv_inv(p);
    }

} 


int main()
{   
    //chosing the coordinate system from user input

    std::string coord;

    std::cout << "Enter coordinate system:";
    std::cin >> coord;

    assert(
    coord == "cartesian" ||
    coord == "cylindrical" ||
    coord == "spherical"
    );

    CartesianMetric cart_metric;
    CylindricalMetric cyl_metric;
    SphericalMetric sph_metric;

    Metric *met = nullptr;
    
    if (coord == "cartesian") {
        met = &cart_metric;
    }
    else if (coord == "cylindrical") {
        met = &cyl_metric;
    }
    else if (coord == "spherical") {
        met = &sph_metric;
    }

    Metric &metric= *met;
    
    //declaring and initializing the particle
    EParticle p;
    
    p.rdata(PIdx::time)= 0.0;
    p.rdata(PIdx::x)= 1.0;
    p.rdata(PIdx::y)= 1.0;
    p.rdata(PIdx::z)= 1.0;
    
    // the folloing is chosen to replicate null particle (p.p=0)
    p.rdata(PIdx::pupt)= 3.0;
    p.rdata(PIdx::pupx)= 2.0;
    p.rdata(PIdx::pupy)= 1.0;
    p.rdata(PIdx::pupz)= 2.0;
    
    //time step setup
    double t= 5;
    int steps= 10000;
    double dt= t/steps;
    double v;
    
    Euler(p, metric, dt, steps); //Euler does the time integration
    
    // calculating volume. Syntax: vol(xmax,xmin,ymax,ymin,zmax,zmin). 
    v=metric.vol(4,2,3,-1,5,1);  
    
    double tol = coord == "cartesian" ? 1e-12 : 1e-2;

    assert(std::abs(p.rdata(PIdx::time)- 5.0) < tol);
    assert(std::abs(p.rdata(PIdx::x) - 13.0 / 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::y) - 8.0 / 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::z) - 13.0 / 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupt) - 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupx) - 2.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupy) - 1.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupz) - 2.0) < tol);
    
    // Printing out final state values and volume
    if( coord=="cartesian" )
    {
        std::cout << "final values in cartesian\n";
        std::cout << "time = " << p.rdata(PIdx::time)  << "\n";
        std::cout << "x    = " << p.rdata(PIdx::x)    << "\n";
        std::cout << "y    = " << p.rdata(PIdx::y)    << "\n";
        std::cout << "z    = " << p.rdata(PIdx::z)    << "\n";
        std::cout << "pt   = " << p.rdata(PIdx::pupt)  << "\n";
        std::cout << "px   = " << p.rdata(PIdx::pupx) << "\n";
        std::cout << "py   = " << p.rdata(PIdx::pupy) << "\n";
        std::cout << "pz   = " << p.rdata(PIdx::pupz) << "\n";
        std::cout << "v   = " << v << "\n";
    }

    if( coord=="cylindrical" || coord=="spherical" )
    {   
        metric.coord_conv(p);
        std::cout << "final values in "<<" "<<coord<<"\n";
        std::cout << "time = " << p.rdata(PIdx::time)  << "\n";
        std::cout << "x    = " << p.rdata(PIdx::x)    << "\n";
        std::cout << "y    = " << p.rdata(PIdx::y)    << "\n";
        std::cout << "z    = " << p.rdata(PIdx::z)    << "\n";
        std::cout << "pt   = " << p.rdata(PIdx::pupt)  << "\n";
        std::cout << "px   = " << p.rdata(PIdx::pupx) << "\n";
        std::cout << "py   = " << p.rdata(PIdx::pupy) << "\n";
        std::cout << "pz   = " << p.rdata(PIdx::pupz) << "\n";
        std::cout << "v   = " << v << "\n";
        
        metric.coord_conv_inv(p);

        std::cout << "final values in cartesian\n";
        std::cout << "time = " << p.rdata(PIdx::time)  << "\n";
        std::cout << "x    = " << p.rdata(PIdx::x)    << "\n";
        std::cout << "y    = " << p.rdata(PIdx::y)    << "\n";
        std::cout << "z    = " << p.rdata(PIdx::z)    << "\n";
        std::cout << "pt   = " << p.rdata(PIdx::pupt)  << "\n";
        std::cout << "px   = " << p.rdata(PIdx::pupx) << "\n";
        std::cout << "py   = " << p.rdata(PIdx::pupy) << "\n";
        std::cout << "pz   = " << p.rdata(PIdx::pupz) << "\n";
        


        

    }
    



}
