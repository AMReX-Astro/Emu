/* Copyright 2019 Maxence Thevenet, Weiqun Zhang
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
#include <AMReX_MultiFab.H>

#ifndef WARPX_FILTER_H_
#define WARPX_FILTER_H_

class Filter
{
public:
    Filter () = default;

    // Apply stencil on MultiFab.
    // Guard cells are handled inside this function
    void ApplyStencil(amrex::MultiFab& dstmf,
                              const amrex::MultiFab& srcmf, int scomp=0,
                              int dcomp=0, int ncomp=10000);

    // Apply stencil on a FabArray.
    void ApplyStencil (amrex::FArrayBox& dstfab,
                       const amrex::FArrayBox& srcfab, const amrex::Box& tbx,
                       int scomp=0, int dcomp=0, int ncomp=10000);

    // public for cuda
    void DoFilter(const amrex::Box& tbx,
                          amrex::Array4<amrex::Real const> const& tmp,
                          amrex::Array4<amrex::Real      > const& dst,
                          int scomp, int dcomp, int ncomp);

    // In 2D, stencil_length_each_dir = {length(stencil_x), length(stencil_z)}
    amrex::IntVect stencil_length_each_dir;

protected:
    // Stencil along each direction.
    // in 2D, stencil_y is not initialized.
    amrex::Gpu::ManagedVector<amrex::Real> stencil_x, stencil_y, stencil_z;
    // Length of each stencil.
    // In 2D, slen = {length(stencil_x), length(stencil_z), 1}
    amrex::Dim3 slen;

private:

};
#endif // #ifndef WARPX_FILTER_H_
