import os
import shutil
import glob

n_digits_rundir = 3
rundir_base = "RUN"

# write an input file in a particular directory with a set of inputs
def write_inputfile(directory, inputs_dict, filename="inputs"):
    path = directory+"/"+filename
    f = open(path,"w")
    for key in inputs_dict:
        f.write(key+" = ")
        value = inputs_dict[key]
        wrapper = "\"" if type(value)==str else ""
        f.write(wrapper)
        if type(value)==list:
            first = True
            for elem in value:
                if first: f.write("(")
                else: f.write(",")
                first = False
                f.write(str(elem))
            f.write(")")
        else:
            f.write(str(inputs_dict[key]))
        f.write(wrapper)
        f.write("\n")
    f.close()

def clean_string(s):
    for character in ["\"","\'"," ","\n"]:
        s = s.replace(character,"")
    return s
    
# given a run directory and an optional inputs filename,
# return a dictionary with key-value pairs for the parameters
# keys and values are both returned as strings
def read_inputfile(directory, filename="inputs"):
    path = directory + "/"+filename
    f = open(path,"r")
    inputs_dict = {}
    for line in f.readlines():
        # remove comments
        line = line.split("#")[0]

        # skip empty lines
        if "=" not in line:
            continue

        # split line into key/value
        key,value = line.split("=")

        # get the format
        is_string = any((c in value) for c in ['\"','\''])
        is_list = "(" in value
        is_float = (not is_string) and (("." in value) or ("e" in value) )
        is_int = (not is_float) and (not is_string)
        
        if(is_int): cast_func = int
        if(is_float): cast_func = float
        if(is_string): cast_func = clean_string

        if is_list:
            value = value.replace("(","")
            value = value.replace(")","")
            value = value.split(",")
            value = [cast_func(v) for v in value]
        else:
            value = cast_func(value)
            
        key = key.strip()

        # append to the dictionary
        inputs_dict[key] = value

    f.close()
    return inputs_dict

# set up a new simulation in the specified location with an inputs dictionary and executable
# Refuse to overwrite unless overwrite==True
# note that the simulation still lacks a particle file! Must be created separately.
def create_new_simulation(directory, inputs_dict, executable_path, overwrite=False):
    
    # check whether the directory already exiss
    if os.path.exists(directory):
        if not overwrite:
            raise(directory+" already exists. Cannot create ")
        else:
            os.system('rm -rf '+directory)
    print("   Creating",directory)

    # create the new directories
    os.mkdir(directory)
    subdir = directory+"/"+rundir_base+str(1).zfill(n_digits_rundir)
    os.mkdir(subdir)

    # copy in the executable
    os.system("cp "+executable_path+" "+subdir)

    # write the inputs file
    write_inputfile(subdir, inputs_dict)

    return subdir

def out_step_list(directory,subdir=None):
    search_string = directory+"/plt?????"
    if subdir==None:
        dirlist = glob.glob(search_string)
        steplist = sorted([int(d.split("/")[-1][3:]) for d in dirlist])
    else:
        search_string = search_string+"/"+subdir
        dirlist = glob.glob(search_string)
        steplist = sorted([int(d.split("/")[-2][3:]) for d in dirlist])
    return steplist

# create a new run subdirectory based on the previous one
# replace the parameters and executable as necessary
def create_restart_simulation(directory, replace_inputs_dict=None, executable_path=None):
    # determine new run dir
    rundirlist = sorted(glob.glob(directory+"/"+rundir_base+"*"))
    assert(len(rundirlist)>0)
    lastrun = int(rundirlist[-1].split("/")[-1].replace(rundir_base,"").strip("0"))
    nextrundir = directory+"/"+rundir_base+str(lastrun+1).zfill(n_digits_rundir)
    os.mkdir(nextrundir)

    # print some diagnostics
    print("   Restarting",directory,"at",nextrundir)
    last_step_list = out_step_list(rundirlist[-1])
    print("      "+rundirlist[-1]+" outputs grid data from",str(last_step_list[0]),"to",str(last_step_list[-1]))
    particle_step_list = out_step_list(rundirlist[-1],"neutrinos")
    print("      "+rundirlist[-1]+" outputs particle data from",str(particle_step_list[0]),"to",str(particle_step_list[-1]))
    assert(len(particle_step_list)>0)

    # get inputs dict from previous simulation
    inputs_dict = read_inputfile(rundirlist[-1])
    
    # activate restarting
    inputs_dict["do_restart"] = 1

    # set restart directory
    inputs_dict["restart_dir"] = "../"+rundirlist[-1].split("/")[-1]+"/plt"+str(particle_step_list[-1]).zfill(5)

    # use replace_inputs_dict last to override anything else
    if replace_inputs_dict != None:
        print("replacing dict")
        for key in replace_inputs_dict:
            print(key in inputs_dict)
            inputs_dict[key] = replace_inputs_dict[key]

    # write the inputs file
    write_inputfile(nextrundir,inputs_dict)

    # copy in executable
    if executable_path==None:
        executable_path = rundirlist[-1]+"/*.ex"
    os.system("cp "+executable_path+" "+nextrundir)

    return nextrundir
