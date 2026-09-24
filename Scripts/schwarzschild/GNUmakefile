AMREX_HOME ?= ../../submodules/amrex

DEBUG	= FALSE

DIM	= 3

COMP    = gnu

USE_MPI   = TRUE
USE_OMP   = FALSE
USE_PARTICLES = TRUE

DEFINES += -DNUM_FLAVORS=2

INCLUDE_LOCATIONS += ../../Source
INCLUDE_LOCATIONS += .
VPATH_LOCATIONS   += ../../Source

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Particle/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
