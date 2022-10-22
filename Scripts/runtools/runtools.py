import os
import shutil

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
        for character in ["\""," ","\n"]:
            value = value.replace(character,"")

        # append to the dictionary
        inputs_dict[key] = value

    f.close()
    return inputs_dict

# set up a new simulation in the specified location with an inputs dictionary and executable
# Refuse to overwrite unless overwrite==True
# note that the simulation still lacks a particle file! Must be created separately.
def new_simulation(directory, inputs_dict, executable_path, overwrite=False):

    # check whether the directory already exiss
    if os.path.exists(directory):
        if not overwrite:
            raise(directory+" already exists. Cannot create ")
        else:
            os.system('rm -rf '+directory)

    # create the new directories
    os.mkdir(directory)
    subdir = directory+"/run0"
    os.mkdir(subdir)

    # copy in the executable
    os.system("cp "+executable_path+" "+subdir)

    # write the inputs file
    write_inputfile(subdir, inputs_dict)


    
