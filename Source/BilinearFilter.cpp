/* Copyright 2019 Andrew Myers, Maxence Thevenet, Weiqun Zhang
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL

    WarpX v20.07 Copyright (c) 2018, The Regents of the University of
    California, through Lawrence Berkeley National Laboratory, and Lawrence
    Livermore National Security, LLC, for the operation of Lawrence Livermore
    National Laboratory (subject to receipt of any required approvals from
    the U.S. Dept. of Energy). All rights reserved.


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:


    (1) Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    (2) Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    (3) Neither the name of the University of California, Lawrence Berkeley
    National Laboratory, Lawrence Livermore National Security, LLC, Lawrence
    Livermore National Laboratory, U.S. Dept. of Energy, nor the names of its
    contributors may be used to endorse or promote products derived from this
    software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    You are under no obligation whatsoever to provide any bug fixes, patches,
    or upgrades to the features, functionality or performance of the source
    code ("Enhancements") to anyone; however, if you choose to make your
    Enhancements available either publicly, or directly to Lawrence Berkeley
    National Laboratory, without imposing a separate written license
    agreement for such Enhancements, then you hereby grant the following
    license: a non-exclusive, royalty-free perpetual license to install, use,
    modify, prepare derivative works, incorporate into other computer
    software, distribute, and sublicense such enhancements or derivative
    works thereof, in binary and source code form.

 */
#include "BilinearFilter.H"

#ifdef _OPENMP
#   include <omp.h>
#endif


using namespace amrex;

namespace {
    void compute_stencil(Gpu::ManagedVector<Real> &stencil, int npass)
    {
        Gpu::ManagedVector<Real> old_s(1+npass,0.);
        Gpu::ManagedVector<Real> new_s(1+npass,0.);

        old_s[0] = 1.;
        int jmax = 1;
        amrex::Real loc;
        // Convolve the filter with itself npass times
        for(int ipass=1; ipass<npass+1; ipass++){
            // element 0 has to be treated in its own way
            new_s[0] = 0.5 * old_s[0];
            if (1<jmax) new_s[0] += 0.5 * old_s[1];
            loc = 0.;
            // For each element j, apply the filter to
            // old_s to get new_s[j]. loc stores the tmp
            // filtered value.
            for(int j=1; j<jmax+1; j++){
                loc = 0.5 * old_s[j];
                loc += 0.25 * old_s[j-1];
                if (j<jmax) loc += 0.25 * old_s[j+1];
                new_s[j] = loc;
            }
            // copy new_s into old_s
            old_s = new_s;
            // extend the stencil length for next iteration
            jmax += 1;
        }
        // we use old_s here to make sure the stencil
        // is corrent even when npass = 0
        stencil = old_s;
        stencil[0] *= 0.5; // because we will use it twice
    }
}

void BilinearFilter::ComputeStencils(){
    BL_PROFILE("BilinearFilter::ComputeStencils()");
    stencil_length_each_dir = npass_each_dir;
    stencil_length_each_dir += 1.;
#if (AMREX_SPACEDIM == 3)
    // npass_each_dir = npass_x npass_y npass_z
    stencil_x.resize( 1 + npass_each_dir[0] );
    stencil_y.resize( 1 + npass_each_dir[1] );
    stencil_z.resize( 1 + npass_each_dir[2] );
    compute_stencil(stencil_x, npass_each_dir[0]);
    compute_stencil(stencil_y, npass_each_dir[1]);
    compute_stencil(stencil_z, npass_each_dir[2]);
#elif (AMREX_SPACEDIM == 2)
    // npass_each_dir = npass_x npass_z
    stencil_x.resize( 1 + npass_each_dir[0] );
    stencil_z.resize( 1 + npass_each_dir[1] );
    compute_stencil(stencil_x, npass_each_dir[0]);
    compute_stencil(stencil_z, npass_each_dir[1]);
#endif
    slen = stencil_length_each_dir.dim3();
#if (AMREX_SPACEDIM == 2)
    slen.z = 1;
#endif
}
