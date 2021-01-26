EMU_HOME ?= ../
AMREX_HOME ?= ../amrex

SHAPE_FACTOR_ORDER ?= 2
NUM_FLAVORS ?= 3

COMP = gnu

DEBUG = FALSE

USE_MPI       = TRUE
USE_OMP       = FALSE
USE_ACC       = FALSE
USE_CUDA      = FALSE
USE_HDF5      = FALSE

TINY_PROFILE  = TRUE
USE_PARTICLES = TRUE

PRECISION     = DOUBLE

Bpack   :=
Blocs   := .

ifeq ($(USE_HDF5), TRUE)
HDF5_HOME =
DEFINES += -DAMREX_USE_HDF5
INCLUDE_LOCATIONS += $(HDF5_HOME)/include
LIBRARIES         += -L$(HDF5_HOME)/lib -lhdf5 -lz -ldl
endif

include $(EMU_HOME)/Make.Emu
