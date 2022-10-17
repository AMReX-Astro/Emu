# write an input file in a particular directory with a set of inputs
def print_inputfile(par_dict, directory):
    filename = directory+"/inputs"
    f = open(filename,"w")
    for key in par_dict:
        f.write(key+" = ")
        value = par_dict[key]
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
            f.write(str(par_dict[key]))
        f.write(wrapper)
        f.write("\n")
    f.close()

