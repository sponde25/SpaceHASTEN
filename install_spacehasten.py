# SpaceHASTEN: installer
#
# Copyright (c) 2024-2025 Orion Corporation
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, 
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, 
# this list of conditions and the following disclaimer in the documentation 
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software 
# without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.
#
import os
import shutil
from cfg import SpaceHASTENConfiguration

def ask_for_file(default_exe,desc=None):
    if desc is None:
        desc = default_exe.split("/")[-1]
    asked_exe = input("Please enter the path to "+desc+" executable [default:"+default_exe+"]: ")
    if asked_exe == "":
        asked_exe = default_exe
    if not os.path.exists(asked_exe):
        print("The specified path does not exist.")
        exit()
    return asked_exe

def ask_for_dir(default_dir,desc,exist=True):
    asked_dir = input("Please enter the path to "+desc+" executable [default:"+default_dir+"]: ")
    if asked_dir == "":
        asked_dir = default_dir
    if exist and not os.path.isdir(asked_dir):
        print("The specified path is not a directory.")
        exit()
    return asked_dir

print()
print(" ___                      _ _  ___  ___  ___  ___  _ _ ")
print("/ __> ___  ___  ___  ___ | | || . |/ __>|_ _|| __>| \ |")
print("\__ \| . \<_> |/ | '/ ._>|   ||   |\__ \ | | | _> |   |")
print("<___/|  _/<___|\_|_.\___.|_|_||_|_|<___/ |_| |___>|_\_|")
print("     |_|                                               ")
print()
print("SpaceHASTEN installer " + str(SpaceHASTENConfiguration.SPACEHASTEN_VERSION)+"\n")
print("This script will install SpaceHASTEN on your system.")

print("NOTE: THIS MUST BE A NFS DIRECTORY VISIBLE TO ALL COMPUTING NODES.")
print("ADDITIONAL NOTE: slurm is used by default, SGE users need to manually reconfigure spacehasten.ini\n")
path = ask_for_dir("/data/programs/spacehasten-"+str(SpaceHASTENConfiguration.SPACEHASTEN_VERSION),"Installation directory",exist=False)
if os.path.exists(path):
    print("The specified path already exists.")
    print("Installation aborted.")
    exit()
os.makedirs(path)
spacelight_exe = ask_for_file("/data/programs/BiosolveIT/spacelight-2.0.0-Linux-x64/spacelight")
ftrees_exe = ask_for_file("/data/programs/BiosolveIT/ftrees-7.0.0-Linux-x64/ftrees")
spaces_dir = ask_for_dir("/data/programs/BiosolveIT/spaces_new","BiosolveIT spaces directory")
default_space = ask_for_file("/data/programs/BiosolveIT/spaces_new/REALSpace_83bn_2025-09.space","default space")
scratch_dir = ask_for_dir("/wrk","scratch directory")
enaminereal_seeds = ask_for_file("/data/work/db/Enamine_Diverse_REAL_drug-like_48.2M_cxsmiles.cxsmiles.bz2","Enamine REAL seeds")
prepare_anaconda = input("Please enter the anaconda3 activation command [default:source /data/programs/oce/actoce]: ")
if prepare_anaconda == "":
    prepare_anaconda = "source /data/programs/oce/actoce"
activate_chemprop = input("Please enter the anaconda3 chemprop activation command [default:conda activate chemprop-2.1.2]: ")
if activate_chemprop == "":
    activate_chemprop = "conda activate chemprop-2.1.2"
gpu_exclusive = input("Please type 1 here if you want GPU exlusive run, type 0 otherwise [default:1]: ")
if gpu_exclusive == "":
    gpu_exclusive = "1"
slurm_queue = input("Please enter the SLURM partition name [default:jobs]: ")
if slurm_queue == "":
    slurm_queue = "jobs"
slurm_gpu_parameter = input("Please enter the SLURM GPU parameter [default:--gpus=1]: ")
if slurm_gpu_parameter == "":
    slurm_gpu_parameter = "--gpus=1"

print("Copying files...")
w = open(path+"/spacehasten.ini","wt")
w.write("[General]\n")
w.write("SCHEDULER = slurm\n")
w.write("PREPARE_ANACONDA = "+prepare_anaconda+"\n")
w.write("ACTIVATE_CHEMPROP = "+activate_chemprop+"\n")
w.write("GPU_EXCLUSIVE = "+gpu_exclusive+"\n")
w.write("CPU_COUNT_SEARCH = 2\n")
w.write("CPU_COUNT_DOCK = 1\n")
w.write("CPU_COUNT_PREDICT = 1\n")
w.write("CPU_COUNT_CONTROL = 1\n")
w.write("\n")
w.write("[Paths]\n")
w.write("EXE_SPACELIGHT_DEFAULT = "+spacelight_exe+"\n")
w.write("EXE_FTREES_DEFAULT = "+ftrees_exe+"\n")
w.write("SPACES_DIR_DEFAULT = "+spaces_dir+"\n")
w.write("SPACES_FILE_DEFAULT = "+default_space+"\n")
w.write("SCRATCH_DEFAULT = "+scratch_dir+"\n")
w.write("ENAMINEREAL_SEEDS = "+enaminereal_seeds+"\n")
w.write("\n")
w.write("[Slurm]\n")
w.write("SLURM_PARTITION = "+slurm_queue+"\n")
w.write("SLURM_GPU_PARAMETER = "+slurm_gpu_parameter+"\n")
w.write("\n")
w.write("[SGE]\n")
w.write("SGE_QUEUE = jobs\n")
w.write("SGE_PE = smp\n")
w.write("SGE_GPU_PARAMETER = -l gpu=1\n")
w.write("\n")
w.write("[Properties]\n")
w.write("MW_MIN = 0.0\n")
w.write("MW_MAX = 500.0\n")
w.write("SLOGP_MIN = -10.0\n")
w.write("SLOGP_MAX = 5.0\n")
w.write("HBA_MIN = 0\n")
w.write("HBA_MAX = 10\n")
w.write("HBD_MIN = 0\n")
w.write("HBD_MAX = 5\n")
w.write("ROTBONDS_MIN = 0\n")
w.write("ROTBONDS_MAX = 10\n")
w.write("TPSA_MIN = 0.0\n")
w.write("TPSA_MAX = 140.0\n")
w.close()

files_to_copy = ["verify","verify_spacehasten.py","cfg.py","control.py","chunkpredict.py",
                 "export_poses.py","grid-test_dock.zip","spacehasten_logo.png","test_dock.in","examples.smi","example.csv",
                 "control.py","docking_functions.py","export_functions.py","export_poses.py","functions.py","gui.py",
                 "importseeds_functions.py","prediction_functions.py","simsearch_functions.py","scheduler_functions.py","spacehasten",
                 "spacehasten.py","training_functions.py","example.smi","archive_functions.py"]
for file_to_copy in files_to_copy:
    if not os.path.exists(file_to_copy):
        print("Error: file '"+file_to_copy+"' not found.")
        exit()
    shutil.copy(file_to_copy,path)

print("SpaceHASTEN has been installed successfully.")
print("Please verify that everything is OK by running '"+path+"/verify' before starting the actual virtual screening process.")
print("The test should take around 15-30 minutes to run.")
os.system("chmod +x "+path+"/verify "+path+"/spacehasten")

