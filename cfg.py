# SpaceHASTEN: configuration
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
import sys
import configparser
import shutil
import os

class SpaceHASTENConfiguration:
      
    SPACEHASTEN_DIRECTORY = "/".join(os.path.abspath(sys.argv[0]).split("/")[0:-1])
    CONTROL_EXE = SPACEHASTEN_DIRECTORY+"/control.py"
    CHUNKPREDICT_EXE = SPACEHASTEN_DIRECTORY+"/chunkpredict.py"
    EXPORTPOSES_EXE = "$SCHRODINGER/run " + SPACEHASTEN_DIRECTORY + "/export_poses.py"
    SPACEHASTEN_VERSION = 0.7
    MAX_CORES = 250
    MAX_JOBNAME_LEN = 15
    EXE_SPACELIGHT_DEFAULT = "/data/programs/BiosolveIT/spacelight-1.5.0-Linux-x64/spacelight"
    EXE_FTREES_DEFAULT = "/data/programs/BiosolveIT/ftrees-6.13.0-Linux-x64/ftrees"
    SPACES_DIR_DEFAULT = "/data/programs/BiosolveIT/spaces"
    SPACES_FILE_DEFAULT = "/data/programs/BiosolveIT/spaces/REALSpace_70bn_2024-09.space"
    QUERIES_DEFAULT = 1000
    DOCKING_DEFAULT = 1000000
    DOCKING_CHUNK = 1000
    RDKIT_CHUNK_DEFAULT = 12345
    RDKIT_CPU_DEFAULT = 0
    PROP_MW_MIN_DEFAULT = 0.0
    PROP_MW_MAX_DEFAULT = 500.0
    PROP_SLOGP_MIN_DEFAULT = -10.0
    PROP_SLOGP_MAX_DEFAULT = 5.0
    PROP_HBA_MIN_DEFAULT = 0
    PROP_HBA_MAX_DEFAULT = 10
    PROP_HBD_MIN_DEFAULT = 0
    PROP_HBD_MAX_DEFAULT = 5
    PROP_ROTBONDS_MIN_DEFAULT = 0
    PROP_ROTBONDS_MAX_DEFAULT = 10
    PROP_TPSA_MIN_DEFAULT = 0.0
    PROP_TPSA_MAX_DEFAULT = 140.0
    CHEMPROP_CHUNK_DEFAULT = 12345
    CHEMPROP_CPU_DEFAULT = 250
    SCRATCH_DEFAULT = "/wrk"
    SIM_SPACELIGHT_DEFAULT = 0.5
    SIM_FTREES_DEFAULT = 0.9
    NNN_DEFAULT = 10000
    TRAIN_DOCKING_CUTOFF = 10.0
    FIELD_SMILES_DEFAULT = "SMILES"
    FIELD_TITLE_DEFAULT = "title"
    FIELD_SCORE_DEFAULT = "r_i_docking_score"
    FIELD_SIMILARITY_SPACELIGHT = "fingerprint-similarity"
    FIELD_SIMILARITY_FTREES = "pharmacophore-similarity"

    SCHEDULER = "slurm"
    PREPARE_ANACONDA = None
    ACTIVATE_CHEMPROP = None
    GPU_EXCLUSIVE = "1"
    CPU_COUNT_SEARCH = "2"
    CPU_COUNT_DOCK = "1"
    CPU_COUNT_PREDICT = "1"
    CPU_COUNT_CONTROL = "1"

    SCHEDULER_SUBMIT = None
    SCHEDULER_JOBNAME = None
    SCHEDULER_OUTPUT_LOG = None
    SCHEDULER_OUTPUT_ERR = None
    SCHEDULER_PARTITION = None
    SCHEDULER_CPU_PER_TASK = None
    SCHEDULER_ARRAY_JOB = None
    SCHEDULER_ARRAY_ID = None
    SCHEDULER_GPU = None
    SCHEDULER_GPU_EXLUSIVE = None
    SCHEDULER_PARTITION = None

    SLURM_PARTITION = None
    SLURM_GPU_PARAMETER = None

    SGE_QUEUE = None
    SGE_PE = None
    SGE_GPU_PARAMETER = None
    
    ENAMINEREAL_SEEDS = "/data/tuomo/PROJECTS/SPACEHASTEN/Enamine_Diverse_REAL_drug-like_48.2M_cxsmiles.cxsmiles.bz2"
    ENAMINEREAL_SEEDS_COUNT = 1000000
    ENAMINEREAL_SEEDS_CPU = 4
    
    def __init__(self):
        cparser = configparser.ConfigParser()
        cparser.read(self.SPACEHASTEN_DIRECTORY+"/spacehasten.ini")

        for setting in cparser["General"]:
            if setting == "scheduler":
                self.SCHEDULER = cparser["General"][setting]
            elif setting == "prepare_anaconda":
                self.PREPARE_ANACONDA = cparser["General"][setting]
            elif setting == "activate_chemprop":
                self.ACTIVATE_CHEMPROP = cparser["General"][setting]
            elif setting == "gpu_exclusive":
                self.GPU_EXCLUSIVE = cparser["General"][setting]
            elif setting == "cpu_count_search":
                self.CPU_COUNT_SEARCH = cparser["General"][setting]
            elif setting == "cpu_count_dock":
                self.CPU_COUNT_DOCK = cparser["General"][setting]
            elif setting == "cpu_count_predict":
                self.CPU_COUNT_PREDICT = cparser["General"][setting]
            elif setting == "cpu_count_control":
                self.CPU_COUNT_CONTROL = cparser["General"][setting]
            else:
                print("Error: Unknown setting in spacehasten.ini:",setting)
                sys.exit(1)

        for setting in cparser["Paths"]:
            if setting == "exe_spacelight_default":
                self.EXE_SPACELIGHT_DEFAULT = cparser["Paths"][setting]
            elif setting == "exe_ftrees_default":
                self.EXE_FTREES_DEFAULT = cparser["Paths"][setting]
            elif setting == "scratch_default":
                self.SCRATCH_DEFAULT = cparser["Paths"][setting]
            elif setting == "spaces_dir_default":
                self.SPACES_DIR_DEFAULT = cparser["Paths"][setting]
            elif setting ==  "spaces_file_default":
                self.SPACES_FILE_DEFAULT = cparser["Paths"][setting]
            elif setting ==  "enaminereal_seeds":
                self.ENAMINEREAL_SEEDS = cparser["Paths"][setting]
            else:
                print("Error: Unknown setting in spacehasten.ini:",setting)
                sys.exit(1)

        for setting in cparser["Slurm"]:
            if setting == "slurm_partition":
                self.SLURM_PARTITION = cparser["Slurm"][setting]
            elif setting == "slurm_prepare_anaconda":
                self.SLURM_PREPARE_ANACONDA = cparser["Slurm"][setting]
            elif setting == "slurm_activate_chemprop":
                self.SLURM_ACTIVATE_CHEMPROP = cparser["Slurm"][setting]
            elif setting == "slurm_gpu_parameter":
                self.SLURM_GPU_PARAMETER = cparser["Slurm"][setting]
            else:
                print("Error: Unknown setting in spacehasten.ini:",setting)
                sys.exit(1)

        for setting in cparser["SGE"]:
            if setting == "sge_queue":
                self.SGE_QUEUE = cparser["SGE"][setting]
            elif setting == "sge_pe":
                self.SGE_PE = cparser["SGE"][setting]
            elif setting == "sge_prepare_anaconda":
                self.SGE_PREPARE_ANACONDA = cparser["SGE"][setting]
            elif setting == "sge_activate_chemprop":
                self.SGE_ACTIVATE_CHEMPROP = cparser["SGE"][setting]
            elif setting == "sge_gpu_parameter":
                self.SGE_GPU_PARAMETER = cparser["SGE"][setting]
            else:
                print("Error: Unknown setting in spacehasten.ini:",setting)
                sys.exit(1)

        for setting in cparser["Properties"]:
            if setting ==  "mw_min":
                self.PROP_MW_MIN_DEFAULT = float(cparser["Properties"][setting])
            elif setting == "mw_max":
                self.PROP_MW_MAX_DEFAULT = float(cparser["Properties"][setting])
            elif setting ==  "slogp_min":
                self.PROP_SLOGP_MIN_DEFAULT = float(cparser["Properties"][setting])
            elif setting == "slogp_max":
                self.PROP_SLOGP_MAX_DEFAULT = float(cparser["Properties"][setting])
            elif setting == "hba_min":
                self.PROP_HBA_MIN_DEFAULT = int(cparser["Properties"][setting])
            elif setting == "hba_max":
                self.PROP_HBA_MAX_DEFAULT = int(cparser["Properties"][setting])
            elif setting == "hbd_min":
                self.PROP_HBD_MIN_DEFAULT = int(cparser["Properties"][setting])
            elif setting == "hbd_max":
                self.PROP_HBD_MAX_DEFAULT = int(cparser["Properties"][setting])
            elif setting == "rotbonds_min":
                self.PROP_ROTBONDS_MIN_DEFAULT = int(cparser["Properties"][setting])
            elif setting == "rotbonds_max":
                self.PROP_ROTBONDS_MAX_DEFAULT = int(cparser["Properties"][setting])
            elif setting == "tpsa_min":
                self.PROP_TPSA_MIN_DEFAULT = float(cparser["Properties"][setting])
            elif setting == "tpsa_max":
                self.PROP_TPSA_MAX_DEFAULT = float(cparser["Properties"][setting])
            else:
                print("Error: Unknown setting in spacehasten.ini:",setting)
                sys.exit(1)

        if shutil.which("chemprop") is None:
            sys.exit("Error: chemprop version 2.x not available, please active chemprop 2 environment before SpaceHASTEN!")
        if shutil.which("bzcat") is None:
            sys.exit("Error: bzcat not available, please install bzip2 package!")
        if not os.path.exists(self.EXE_SPACELIGHT_DEFAULT):
            sys.exit("Error: "+self.EXE_SPACELIGHT_DEFAULT+" not available!")                      
        if not os.path.exists(self.EXE_FTREES_DEFAULT):
            sys.exit("Error: "+self.EXE_FTREES_DEFAULT+" not available!")                      
        if not os.path.exists(os.getenv("SCHRODINGER")+"/run"):
            sys.exit("Error: $SCHRODINGER/run not available.")
        if not os.path.exists(self.SPACEHASTEN_DIRECTORY+"/control.py"):
            sys.exit("Error: invalid SpaceHASTEN installation, control.py missing!")
        if not os.path.exists(self.SPACEHASTEN_DIRECTORY+"/chunkpredict.py"):
            sys.exit("Error: invalid SpaceHASTEN installation, chunkpredict.py missing!")
        if not os.path.exists(self.SPACEHASTEN_DIRECTORY+"/export_poses.py"):
            sys.exit("Error: invalid SpaceHASTEN installation, export_poses.py missing!")
        if not os.path.exists(self.SPACEHASTEN_DIRECTORY+"/spacehasten_logo.png"):
            sys.exit("Error: invalid SpaceHASTEN installation, spacehasten_logo missing!")
        if self.SCHEDULER == "slurm" :
            self.SCHEDULER_SUBMIT = "sbatch"
            self.SCHEDULER_JOBNAME = "#SBATCH -J"
            self.SCHEDULER_OUTPUT_LOG = "#SBATCH -o /dev/null"
            self.SCHEDULER_OUTPUT_ERR = "#SBATCH -e /dev/null"
            self.SCHEDULER_PARTITION = "#SBATCH -p "+self.SLURM_PARTITION
            self.SCHEDULER_CPU_PER_TASK = "#SBATCH --cpus-per-task="
            self.SCHEDULER_ARRAY_JOB = "#SBATCH --array=1-"
            self.SCHEDULER_ARRAY_ID="SLURM_ARRAY_TASK_ID"
            self.SCHEDULER_GPU = "#SBATCH "+self.SLURM_GPU_PARAMETER
            if self.GPU_EXCLUSIVE == "1":
                self.SCHEDULER_GPU_EXCLUSIVE = "#SBATCH --exclusive"
            else:
                self.SCHEDULER_GPU_EXCLUSIVE = "#nop"
        elif self.SCHEDULER != "SGE":
            self.SCHEDULER_SUBMIT = "qsub"
            self.SCHEDULER_JOBNAME = "#$ -N"
            self.SCHEDULER_OUTPUT_LOG = "#$ -o /dev/null"
            self.SCHEDULER_OUTPUT_ERR = "#$ -e /dev/null"
            self.SCHEDULER_PARTITION = "#$ -q "+self.SGE_QUEUE
            self.SCHEDULER_CPU_PER_TASK = "#$ -pe "+self.SGE_PE+" "
            self.SCHEDULER_ARRAY_JOB = "#$ -t 1-"
            self.SCHEDULER_ARRAY_ID="SGE_TASK_ID"
            self.SCHEDULER_GPU = "#$ "+self.SGE_GPU_PARAMETER
            if self.GPU_EXCLUSIVE == "1":
                self.SCHEDULER_GPU_EXCLUSIVE = "#$ -l exclusive"
            else:
                self.SCHEDULER_GPU_EXCLUSIVE = "#nop"
        else:
            sys.exit("Error: Unknown scheduler ('"+self.SCHEDULER+"') defined in spacehasten.ini (only slurm and SGE supported).")
