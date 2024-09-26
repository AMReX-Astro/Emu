pipeline {
    triggers { pollSCM('') }  // Run tests whenever a new commit is detected.
    agent { dockerfile {args '--gpus all -v /mnt/scratch/tables/EOS:/EOS:ro /mnt/scratch/tables/NuLib:/NuLib:ro'}} // Use the Dockerfile defined in the root Flash-X directory
    environment {
		// Get rid of Read -1, expected <someNumber>, errno =1 error
    	// See https://github.com/open-mpi/ompi/issues/4948
        OMPI_MCA_btl_vader_single_copy_mechanism = 'none'
    }
    stages {

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
