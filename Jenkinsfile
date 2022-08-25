pipeline {
    triggers { pollSCM('') }  // Run tests whenever a new commit is detected.
    agent { dockerfile {args '--gpus all'}} // Use the Dockerfile defined in the root Flash-X directory
    stages {

        //=============================//
    	// Set up submodules and amrex //
        //=============================//
    	stage('Prerequisites'){ steps{
	    sh 'mpicc -v'
	    sh 'nvidia-smi'
	    sh 'nvcc -V'
	    sh 'git submodule update --init'
	    sh 'cp makefiles/GNUmakefile_jenkins Exec/GNUmakefile'
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
	    }
	}}

	stage('Bipolar'){ steps{
	    dir('Exec'){
		sh 'python ../Scripts/initial_conditions/st1_bipolar_test.py'
	        sh 'mpirun -np 4 ./main3d.gnu.TPROF.MPI.CUDA.ex ../sample_inputs/inputs_bipolar_test'
	    }
	}}

	stage('Fast Flavor'){ steps{
	    dir('Exec'){
		sh 'python ../Scripts/initial_conditions/st2_2beam_fast_flavor.py'
	        sh 'mpirun -np 4 ./main3d.gnu.TPROF.MPI.CUDA.ex ../sample_inputs/inputs_fast_flavor'
	        sh 'python ../Scripts/tests/fast_flavor_test.py'
	    }
	}}

	stage('Fast Flavor k'){ steps{
	    dir('Exec'){
	        sh 'mpirun -np 4 ./main3d.gnu.TPROF.MPI.CUDA.ex ../sample_inputs/inputs_fast_flavor_nonzerok'
	        sh 'python ../Scripts/tests/fast_flavor_k_test.py'
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
		patterns: [[pattern: 'amrex', type: 'EXCLUDE']] ) // allow amrex to be cached
	}
    }

} // pipeline{
