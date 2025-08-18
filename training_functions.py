# SpaceHASTEN: functions to do the training
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
 
import os
import functions
import sqlite3
import pandas as pd
import scheduler_functions
import time
import glob

def train_new_model(args):
    """
    Train new model

    :param args: the args
    """
    print("Training new model:")
    dbname = args.name + ".dbsh"
    
    model_version = functions.get_latest_model(args.name)+1
    print("Model version",model_version)

    modeldir=os.getenv("HOME")+"/SPACEHASTEN/TRAIN_"+args.name+"_ver"+str(model_version)
    os.system("rm -fr "+modeldir)
    os.system("mkdir -p "+modeldir)

    print("Extracting data for training...")
    data_filename=modeldir+"/train_"+args.name+"_ver"+str(model_version)+".csv"
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    # exclude failed dockings with arbitrary score as they screw up training
    training_data = pd.read_sql_query("SELECT smiles,dock_score FROM data WHERE dock_score IS NOT NULL AND dock_score < "+str(args.c.TRAIN_DOCKING_CUTOFF),conn)
    training_data["docking_score"] = training_data["dock_score"]
    training_data.drop(["dock_score"],axis=1,inplace=True)
    training_data.to_csv(data_filename,index=False)
    scheduler_functions.write_train_scheduler(modeldir,args)

    curdir=os.getcwd()
    os.chdir(modeldir)
    print("Running training via scheduler at "+modeldir+" ...")
    os.system("sbatch submit_train_"+args.name+"_ver"+str(model_version)+".sh")
    jobs_left = 1 - len(glob.glob(modeldir+"/jobdone-train_"+args.name+"*"))
    while jobs_left>0:
        time.sleep(5)
        jobs_left = 1 - len(glob.glob(modeldir+"/jobdone-train_"+args.name+"*"))
    os.system("rm "+data_filename)
    tarfile = "model_"+args.name+"_ver"+str(model_version)+".tar.gz"
    os.system("tar -czf "+tarfile+" model_"+args.name+"_ver"+str(model_version))
    with open(tarfile,"rb") as f:
        blob = f.read()
    c.execute("INSERT INTO models VALUES("+str(model_version)+",?)",[memoryview(blob)])
    conn.commit()
    conn.close()
    os.system("rm -r model_"+args.name+"_ver"+str(model_version)+" model_"+args.name+"_ver"+str(model_version)+".tar.gz")
    os.chdir(curdir)
    print("Training complete!")
