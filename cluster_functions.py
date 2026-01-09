# SpaceHASTEN: functions to do clustering
#
# Copyright (c) 2025-2026 Orion Corporation
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
import sqlite3
import gzip
import os
import scheduler_functions
import time
import pandas as pd
import tqdm

def process_cluster_results(args,cluster_dir):
    print("Importing cluster data...")
    dbname = args.name + ".dbsh"
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS clusters")
    c.execute("CREATE TABLE clusters(spacehastenid INTEGER PRIMARY KEY,clusterid INTEGER)")
    clusters = pd.read_csv(cluster_dir + "/clustering.csv")
    clusters.to_sql("clusters",conn,if_exists="replace",index=False)
    conn.commit()
    conn.close()
    print("Done!")

def cluster_dbsh(args):
    print("Clustering compounds in .dbsh")
    os.system("date")
    dbname = args.name + ".dbsh"
    if not os.path.exists(os.getenv("HOME")+"/SPACEHASTEN"):
        os.mkdir(os.getenv("HOME")+"/SPACEHASTEN")
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    cluster_dir = os.getenv("HOME")+"/SPACEHASTEN/CLUSTERING_"+args.name+"_tmp"
    os.system("rm -fr "+cluster_dir)
    os.system("mkdir -p "+cluster_dir)
    # as dbsh is on local disk, we need to copy data to NFS
    w = gzip.open(cluster_dir+"/clustering_input.smi.gz","wt")
    print("Writing compounds for clustering....")
    num_compounds = c.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    for row in tqdm.tqdm(c.execute("SELECT smiles,spacehastenid FROM data"),mininterval=1,total=num_compounds):
        w.write(row[0]+" "+str(row[1])+"\n")
    w.close()
    conn.commit()
    conn.close()
    scheduler_functions.write_cluster_scheduler(cluster_dir,args)
    curdir = os.getcwd()
    os.chdir(cluster_dir)
    os.system("rm -f jobdone-"+args.name+"-CPU*")
    print("Running clustering via scheduler...")
    os.system(args.c.SCHEDULER_SUBMIT + " submit_cluster_"+args.name+".sh")
    os.chdir(curdir)

    scheduler_functions.wait_until_jobs_done(cluster_dir,args.name,1)
    
    process_cluster_results(args,cluster_dir)
    os.system("rm -fr "+cluster_dir)
    os.system("date")
    print("Clustering done!")
