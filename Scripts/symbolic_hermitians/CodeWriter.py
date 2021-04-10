import os
import shutil

class CodeWriter(object):
    def __init__(self, emu_home):
        self.emu_home = emu_home

    def write(self, code, output_file, template=None):
        ## Puts the output file by default into emu_home/Source/generated_files
        ## If output_file is instead a file path, then interpret it as an absolute path
        if os.path.basename(output_file) == output_file:
            output_file = os.path.join(self.emu_home, "Source", "generated_files", output_file)

        try:
            fo = open(output_file, 'w')
        except:
            print(f"could not open output file {output_file} for writing")
            raise

        # Write generated code
        for line in code:
            fo.write(f"{line}\n")

        fo.close()

    def delete_generated_files(self):
        try:
            shutil.rmtree(os.path.join(self.emu_home, "Source", "generated_files"))
        except FileNotFoundError:
            pass