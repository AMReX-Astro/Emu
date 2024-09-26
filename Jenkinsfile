pipeline {
    triggers { pollSCM('') }  // Run tests whenever a new commit is detected.
    agent { dockerfile {args '--gpus all -v /mnt/scratch/EOS:/EOS:ro /mnt/scratch/NuLib:/NuLib:ro'}} // Use the Dockerfile defined in the root Flash-X directory
    environment {
		// Get rid of Read -1, expected <someNumber>, errno =1 error
    	// See https://github.com/open-mpi/ompi/issues/4948
        OMPI_MCA_btl_vader_single_copy_mechanism = 'none'
    }
    stages {

        //=============================//
    	// Set up submodules and amrex //
        //=============================//
    	stage('Prerequisites'){ steps{
	    sh 'mpicc -v'
	    sh 'nvidia-smi'
	    sh 'nvcc -V'
	    sh 'git submodule update --init'
	    sh 'cp makefiles/GNUmakefile_jenkins_HDF5_CUDA Exec/GNUmakefile'
	    dir('Exec'){
	        sh 'make generate; make -j'
	    }
	}}

	//=======//
	// Tests //
	//=======//
	stage('MSW'){ steps{
	    dir('Exec'){
		sh 'python ../Scripts/initial_conditions/st0_msw_test.py'
	        sh 'mpirun -np 4 ./main3d.gnu.TPROF.MPI.CUDA.ex ../sample_inputs/inputs_msw_test'
	        sh 'python ../Scripts/tests/msw_test.py'
			sh 'rm -rf plt*'
	    }
	}}

	stage('Bipolar'){ steps{
	    dir('Exec'){
			sh 'python ../Scripts/initial_conditions/st1_bipolar_test.py'
	        sh 'mpirun -np 4 ./main3d.gnu.TPROF.MPI.CUDA.ex ../sample_inputs/inputs_bipolar_test'
			sh 'rm -rf plt*'
	    }
	}}

	stage('Fast Flavor'){ steps{
	    dir('Exec'){
		sh 'python ../Scripts/initial_conditions/st2_2beam_fast_flavor.py'
	        sh 'mpirun -np 4 ./main3d.gnu.TPROF.MPI.CUDA.ex ../sample_inputs/inputs_fast_flavor'
	        sh 'python ../Scripts/tests/fast_flavor_test.py'
			sh 'rm -rf plt*'
	    }
	}}

	stage('Fast Flavor k'){ steps{
	    dir('Exec'){
		sh 'python ../Scripts/initial_conditions/st3_2beam_fast_flavor_nonzerok.py'
	        sh 'mpirun -np 4 ./main3d.gnu.TPROF.MPI.CUDA.ex ../sample_inputs/inputs_fast_flavor_nonzerok'
	        sh 'python ../Scripts/tests/fast_flavor_k_test.py'
			sh 'rm -rf plt*'
	    }
	}}

	stage('Fiducial 2F GPU Binary'){ steps{
		dir('Exec'){
			sh 'python ../Scripts/initial_conditions/st4_linear_moment_ffi.py'
			sh 'mpirun -np 4 ./main3d.gnu.TPROF.MPI.CUDA.ex ../sample_inputs/inputs_1d_fiducial'
			sh 'python ../Scripts/data_reduction/reduce_data_fft.py'
			sh 'python ../Scripts/data_reduction/reduce_data.py'
			sh 'python ../Scripts/data_reduction/combine_files.py plt _reduced_data.h5'
			sh 'python ../Scripts/data_reduction/combine_files.py plt _reduced_data_fft_power.h5'
			sh 'python ../Scripts/babysitting/avgfee.py'
			sh 'python ../Scripts/babysitting/power_spectrum.py'
			sh 'python ../Scripts/data_reduction/convertToHDF5.py'
			sh 'gnuplot ../Scripts/babysitting/avgfee_gnuplot.plt'
			archiveArtifacts artifacts: '*.pdf'
			sh 'rm -rf plt*'
		}
	}}

	stage('Fiducial 3F CPU HDF5'){ steps{
		dir('Exec'){
	    	sh 'cp ../makefiles/GNUmakefile_jenkins_HDF5 GNUmakefile'
	        sh 'make realclean; make generate; make -j'
			sh 'python ../Scripts/initial_conditions/st4_linear_moment_ffi_3F.py'
			sh 'mpirun -np 4 ./main3d.gnu.TPROF.MPI.ex ../sample_inputs/inputs_1d_fiducial'
			/*sh 'python3 ../Scripts/babysitting/avgfee_HDF5.py'*/
			sh 'rm -rf plt*'
		}
	}}

	stage('Collisions flavor instability'){ steps{
		dir('Exec'){
			sh 'cp ../makefiles/GNUmakefile_jenkins_HDF5_CUDA GNUmakefile'
	        sh 'make realclean; make generate NUM_FLAVORS=2; make -j NUM_FLAVORS=2'
			sh 'python ../Scripts/initial_conditions/st8_coll_inst_test.py'
			sh 'mpirun -np 4 ./main3d.gnu.TPROF.MPI.CUDA.ex ../sample_inputs/inputs_collisional_instability_test'
			sh 'python ../Scripts/data_reduction/reduce_data.py'
	        sh 'python ../Scripts/tests/coll_inst_test.py'
			sh 'rm -rf plt* *pdf'
		}
	}}

	stage('Collisions to equilibrium'){ steps{
		dir('Exec'){
			sh 'cp ../makefiles/GNUmakefile_jenkins_HDF5_CUDA GNUmakefile'
	        sh 'make realclean; make generate NUM_FLAVORS=3; make -j NUM_FLAVORS=3'
			sh 'python ../Scripts/initial_conditions/st7_empty_particles.py'
			sh 'mpirun -np 4 ./main3d.gnu.TPROF.MPI.CUDA.ex ../sample_inputs/inputs_coll_equi_test'
	        sh 'python ../Scripts/tests/coll_equi_test.py'
			sh 'rm -rf plt* *pdf'
		}
	}}

	stage('Fermi-Dirac test'){ steps{
		dir('Exec'){
			sh 'python ../Scripts/initial_conditions/st9_empty_particles_multi_energy.py'
			sh 'python ../Scripts/collisions/nsm_constant_background_rho_Ye_T__writer.py'
			sh 'mpirun -np 4 ./main3d.gnu.TPROF.MPI.CUDA.ex ../sample_inputs/inputs_fermi_dirac_test'
	        sh 'python ../Scripts/collisions/writeparticleinfohdf5.py'
	        sh '../Scripts/tests/fermi_dirac_test.py'
			sh 'rm -rf plt* *pdf rho_Ye_T.hdf5'
		}
	}}

    } // stages{

    post {
        always {
	    cleanWs(
	        cleanWhenNotBuilt: true,
		deleteDirs: true,
		disableDeferredWipeout: false,
		notFailBuild: true,
		patterns: [[pattern: 'submodules', type: 'EXCLUDE']] ) // allow submodules to be cached
	}
    }

} // pipeline{
