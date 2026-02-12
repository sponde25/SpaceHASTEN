# SpaceHASTEN: scheduler functions (used to be slurm_functions.py)
#
# Copyright (c) 2024-2026 Orion Corporation
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

import functions
import os
import tqdm
import glob
import time

def write_header(w,args,jobname):
    w.write("#!/bin/bash\n")
    w.write(args.c.SCHEDULER_JOBNAME+" "+jobname+"\n")
    w.write(args.c.SCHEDULER_OUTPUT_LOG+"\n")
    w.write(args.c.SCHEDULER_OUTPUT_ERR+"\n")
    w.write(args.c.SCHEDULER_PARTITION+"\n")

def close_job(w,args,personal_scratch):
    w.write("cd $curdir\n")
    w.write("rm -fr "+personal_scratch+"\n")
    w.write("touch jobdone-"+args.name+"-CPU$"+args.c.SCHEDULER_ARRAY_ID+"\n")
    w.close()

def write_search_scheduler(cycle_dir,args):
    """
    Write scheduler script for Spacelight/FTrees searching

    :param cycle_dir: directory where this cycle is
    :param args: the args
    """
    jobname="queries_"+args.name+"_cycle"+cycle_dir.split("_cycle")[-1]
    cpuname=jobname+"_cpu$"+args.c.SCHEDULER_ARRAY_ID
    personal_scratch=args.scratch+"/"+os.getenv("USER")+"/"+cpuname
    w = open(cycle_dir+"/submit_queries_"+args.name+".sh","wt")
    write_header(w,args,jobname)
    w.write(args.c.SCHEDULER_CPU_PER_TASK+str(args.c.CPU_COUNT_SEARCH)+"\n")
    w.write(args.c.SCHEDULER_ARRAY_JOB+str(args.top)+"%"+str(args.cpu)+"\n")
    w.write("smiles=$(sed -n \"$"+args.c.SCHEDULER_ARRAY_ID+"\"\"p\" queries_"+args.name+".smi)\n")
    w.write("curdir=$(pwd)\n")
    w.write("rm -fr "+personal_scratch+"\n")
    w.write("mkdir -p "+personal_scratch+"\n")
    w.write("cd "+personal_scratch+"\n")
    # note: thread-count 1 is needed, but still takes two threads. This is a bug in spacelight. 0 means that use all cores.
    w.write(args.c.EXE_SPACELIGHT_DEFAULT + " -i \"$smiles\" -s "+args.space+" -o spacelightresult_"+args.name+"_$"+args.c.SCHEDULER_ARRAY_ID+".csv --max-nof-results "+str(args.nnn)+" --min-similarity-threshold "+str(args.sim_spacelight)+" --thread-count 1\n")
    w.write(args.c.EXE_FTREES_DEFAULT +  " -i \"$smiles\" -s "+args.space+" -o ftreesresult_"+args.name+"_$"+args.c.SCHEDULER_ARRAY_ID+".csv --max-nof-results "+str(args.nnn)+" --min-similarity-threshold "+str(args.sim_ftrees)+" --thread-count 1\n")
    w.write("cp *.csv $curdir/\n")
    close_job(w,args,personal_scratch)

def write_dock_scheduler(dock_dir,args,chunk_counter):
    """
    Write scheduler script for Phase/LigPrep + Docking

    :param dock_dir: Work directory
    :param args: the args
    :param chunk_counter: number of docking chunks
    """
    jobname="dockinput_"+args.name+"_iter"+dock_dir.split("_iter")[-1]
    cpuname=jobname+"_cpu$"+args.c.SCHEDULER_ARRAY_ID
    personal_scratch=args.scratch+"/"+os.getenv("USER")+"/"+cpuname
    w = open(dock_dir+"/submit_"+jobname+".sh","wt")
    write_header(w,args,jobname)
    w.write(args.c.SCHEDULER_CPU_PER_TASK+str(args.c.CPU_COUNT_DOCK)+"\n")
    w.write(args.c.SCHEDULER_ARRAY_JOB+str(chunk_counter)+"%"+str(args.cpu)+"\n")
    w.write("curdir=$(pwd)\n")
    w.write("rm -fr "+personal_scratch+"\n")
    w.write("mkdir -p "+personal_scratch+"\n")
    w.write("cp "+cpuname+".smi "+personal_scratch+"/\n")
    w.write("cp "+cpuname+".inp "+personal_scratch+"/\n")
    w.write("cp glide_"+cpuname+".in "+personal_scratch+"/\n")
    w.write("cp glide_grid.zip "+personal_scratch+"/\n")
    w.write("cd "+personal_scratch+"\n")
    if args.c.SCHRODINGER_FEATURE_FLAGS is not None:
        w.write("export SCHRODINGER_FEATURE_FLAGS=\""+args.c.SCHRODINGER_FEATURE_FLAGS+"\"\n")
    w.write("$SCHRODINGER/pipeline -prog phase_db "+cpuname+".inp -OVERWRITE -WAIT -HOST localhost:1 -NJOBS 1\n")
    w.write("$SCHRODINGER/phase_database $(pwd)/"+cpuname+".phdb export -omae $(pwd)/"+cpuname+" -get 1 -limit 99999999 -WAIT\n")
    w.write("rm -fr $(pwd)/"+cpuname+".phdb\n")
    w.write("$SCHRODINGER/glide -new -OVERWRITE -WAIT -NJOBS 1 -HOST localhost:1 glide_"+cpuname+".in\n")
    w.write("rm -f glide_grid.zip\n")
    w.write("tar --exclude=results-"+cpuname+".tar.gz -czf results-"+cpuname+".tar.gz .\n")
    w.write("mv results-"+cpuname+".tar.gz $curdir\n")
    close_job(w,args,personal_scratch)

def write_predict_scheduler(control_dir,args):
    """
    Write scheduler script for predicting

    :param control_dir: Work directory
    :param args: the args
    """

    simsearch_cycle = functions.get_latest_cycle(args.name)
    if simsearch_cycle == 0:
        simsearch_cycle = 1

    jobname="predict_"+args.name+"_cycle"+str(simsearch_cycle)
    cpuname="predict_"+args.name+"_cpu$"+args.c.SCHEDULER_ARRAY_ID
    modelname="model_"+str(args.name)+"_ver"+str(functions.get_latest_model(args.name))
    personal_scratch=args.scratch+"/"+os.getenv("USER")+"/"+cpuname
    w = open(control_dir+"/submit_"+jobname+".sh","wt")
    w.write("#!/bin/bash\n")
    write_header(w,args,jobname)
    w.write(args.c.SCHEDULER_CPU_PER_TASK+str(args.c.CPU_COUNT_PREDICT)+"\n")
    w.write(args.c.SCHEDULER_ARRAY_JOB+str(args.cpu)+"\n")
    w.write("curdir=$(pwd)\n")
    w.write("rm -fr "+personal_scratch+"\n")
    w.write("mkdir -p "+personal_scratch+"\n")
    w.write("cp "+cpuname+".csv "+personal_scratch+"/\n")
    w.write("cp -r "+modelname+" "+personal_scratch+"/\n")
    w.write("cp control.param "+personal_scratch+"/\n")
    w.write("cd "+personal_scratch+"\n")
    w.write(args.c.PREPARE_ANACONDA + "\n")
    w.write(args.c.ACTIVATE_CHEMPROP + "\n")
    w.write("python3 "+args.c.CHUNKPREDICT_EXE+" "+str(args.chemprop_chunk)+" "+cpuname+".csv "+modelname+"\n")
    w.write("mv predicted_"+cpuname+".csv $curdir\n")
    close_job(w,args,personal_scratch)

def write_train_scheduler(control_dir,args):
    """
    Write scheduler script for training

    :param control_dir: Work directory
    :param args: the args
    """
    model_version=functions.get_latest_model(args.name)+1
    jobname="train_"+args.name+"_ver"+str(model_version)
    data_filename=jobname+".csv"
    w = open(control_dir+"/submit_"+jobname+".sh","wt")
    write_header(w,args,jobname)
    w.write(args.c.SCHEDULER_GPU+"\n")
    w.write(args.c.SCHEDULER_GPU_EXCLUSIVE+"\n")
    w.write("cd "+control_dir+"\n")
    w.write(args.c.PREPARE_ANACONDA + "\n")
    w.write(args.c.ACTIVATE_CHEMPROP + "\n")
    w.write("chemprop train --devices 1 --num-workers 0 --task-type regression --target-columns docking_score --data-path "+data_filename+" --save-dir model_"+args.name+"_ver"+str(model_version)+" --batch-size 250 --no-cache --epochs 30\n")
    w.write("touch jobdone-train_"+args.name+"-CPU1\n")
    w.close()

def write_control_scheduler(control_dir,args):
    """
    Write scheduler script for prop control + prediction

    :param control_dir: Work directory
    :param args: the args
    """
    simsearch_cycle = functions.get_latest_cycle(args.name)
    if simsearch_cycle == 0:
        simsearch_cycle = 1
    jobname="ctrl_"+args.name+"_cycle"+str(simsearch_cycle)
    cpuname="control_"+args.name+"_cpu$"+args.c.SCHEDULER_ARRAY_ID
    modelname="model_"+str(args.name)+"_ver"+str(functions.get_latest_model(args.name))
    personal_scratch=args.scratch+"/"+os.getenv("USER")+"/"+cpuname
    w = open(control_dir+"/submit_"+jobname+".sh","wt")
    write_header(w,args,jobname)
    w.write(args.c.SCHEDULER_CPU_PER_TASK+str(args.c.CPU_COUNT_CONTROL)+"\n")
    w.write(args.c.SCHEDULER_ARRAY_JOB+str(args.cpu)+"\n")
    w.write("curdir=$(pwd)\n")
    w.write("rm -fr "+personal_scratch+"\n")
    w.write("mkdir -p "+personal_scratch+"\n")
    w.write("cp "+cpuname+".smi.gz "+personal_scratch+"/\n")
    w.write("cp -r "+modelname+" "+personal_scratch+"/\n")
    w.write("cp control.param "+personal_scratch+"/\n")
    w.write("cd "+personal_scratch+"\n")
    w.write(args.c.PREPARE_ANACONDA + "\n")
    w.write(args.c.ACTIVATE_CHEMPROP + "\n")
    w.write("python3 "+args.c.CONTROL_EXE+" "+cpuname+".smi.gz control.param\n")
    w.write("python3 "+args.c.CHUNKPREDICT_EXE+" "+str(args.chemprop_chunk)+" propoutput_"+cpuname+".csv.gz "+modelname+"\n")
    w.write("mv predicted_propoutput_"+cpuname+".csv $curdir\n")
    close_job(w,args,personal_scratch)

def write_cluster_scheduler(cluster_dir,args):
    """
    Write scheduler script for clustering

    :param cluster_dir: directory where the clustering input data is
    :param args: the args
    """
    jobname="cluster_"+args.name+"_tmp"
    cpuname=jobname+"_cpu$"+args.c.SCHEDULER_ARRAY_ID
    personal_scratch=args.c.SCRATCH_DEFAULT+"/"+os.getenv("USER")+"/"+cpuname
    w = open(cluster_dir+"/submit_cluster_"+args.name+".sh","wt")
    write_header(w,args,jobname)
    w.write(args.c.SCHEDULER_CPU_PER_TASK+str(args.c.CPU_COUNT_CLUSTERING)+"\n")
    # a bit misleading name, but the exclusivity switch is the same for both CPU and GPU
    w.write(args.c.SCHEDULER_GPU_EXCLUSIVE+"\n")
    w.write("curdir=$(pwd)\n")
    w.write("rm -fr "+personal_scratch+"\n")
    w.write("mkdir -p "+personal_scratch+"\n")
    w.write("gunzip -c "+cluster_dir+"/clustering_input.smi.gz >"+personal_scratch+"/clustering_input.smi\n")
    w.write("cd "+personal_scratch+"\n")
    w.write(args.c.PREPARE_ANACONDA + "\n")
    w.write(args.c.ACTIVATE_CLUSTERING + "\n")
    w.write(args.c.EXE_CLUSTERING_DEFAULT + " clustering_input.smi\n")
    w.write("mv clustering.csv $curdir/\n")
    close_job(w,args,personal_scratch)

def wait_until_jobs_done(directory_to_monitor,name,num_jobs):
    """
    Wait until jobs done and update progress bar when jobs are done

    :param directory_to_monitor: directory where the job writes completition files
    :param name: job name
    :param num_jobs: number of jobs to be done
    """
    pbar = tqdm.tqdm(total=num_jobs)
    jobs_done = len(glob.glob(directory_to_monitor+"/jobdone-"+name+"-CPU*"))
    while jobs_done<num_jobs:
        time.sleep(5)
        jobs_done = len(glob.glob(directory_to_monitor+"/jobdone-"+name+"-CPU*"))
        pbar.n = jobs_done
        pbar.refresh()
    pbar.close()
