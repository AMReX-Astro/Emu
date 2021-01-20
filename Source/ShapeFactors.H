/* Copyright 2019 Maxence Thevenet, Michael Rowan
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
#ifndef SHAPEFACTORS_H_
#define SHAPEFACTORS_H_

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialized templates are defined below for orders 0 to 3.
 *  Shape factor functors may be evaluated with double arguments
 *  in current deposition to ensure that current deposited by
 *  particles that move only a small distance is still resolved.
 *  Without this safeguard, single and double precision versions
 *  can give disagreeing results in the time evolution for some
 *  problem setups.
 */
template <int depos_order>
struct Compute_shape_factor
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(T* const sx, T xint) const { return 0; }
};

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialization for order 0
 */
template <>
struct Compute_shape_factor< 0 >
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(T* const sx, T xmid) const
    {
        const T x = xmid + T(0.5);
        const auto i = static_cast<int>(x);
	const auto j = std::signbit(x) ? i-1 : i;
        sx[0] = T(1.0);
        return j;
    }
};

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialization for order 1
 */
template <>
struct Compute_shape_factor< 1 >
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(T* const sx, T xmid) const
    {
        const T x = xmid;
        const auto i = static_cast<int>(x);
	const auto j = std::signbit(x) ? i-1 : i;
        const T xint = xmid - T(j);
        sx[0] = T(1.0) - xint;
        sx[1] = xint;
        return j;
    }
};

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialization for order 2
 */
template <>
struct Compute_shape_factor< 2 >
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(T* const sx, T xmid) const
    {
        const T x = xmid + T(0.5);
        const auto i = static_cast<int>(x);
	const auto j = std::signbit(x) ? i-1 : i;
        const T xint = xmid - T(j);
        sx[0] = T(0.5)*(T(0.5) - xint)*(T(0.5) - xint);
        sx[1] = T(0.75) - xint*xint;
        sx[2] = T(0.5)*(T(0.5) + xint)*(T(0.5) + xint);
        // index of the leftmost cell where particle deposits
        return j-1;
    }
};

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialization for order 3
 */
template <>
struct Compute_shape_factor< 3 >
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(T* const sx, T xmid) const
    {
        const T x = xmid;
        const auto i = static_cast<int>(x);
	const auto j = std::signbit(x) ? i-1 : i;
        const T xint = xmid - T(j);
        sx[0] = (T(1.0))/(T(6.0))*(T(1.0) - xint)*(T(1.0) - xint)*(T(1.0) - xint);
        sx[1] = (T(2.0))/(T(3.0)) - xint*xint*(T(1.0) - xint/(T(2.0)));
        sx[2] = (T(2.0))/(T(3.0)) - (T(1.0) - xint)*(T(1.0) - xint)*(T(1.0) - T(0.5)*(T(1.0) - xint));
        sx[3] = (T(1.0))/(T(6.0))*xint*xint*xint;
        // index of the leftmost cell where particle deposits
        return j-1;
    }
};


#endif // SHAPEFACTORS_H_
