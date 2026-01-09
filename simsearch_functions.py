# SpaceHASTEN: functions to do the similarity searching
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
#
import os
import functions
import sqlite3
import scheduler_functions
import prediction_functions
import glob
import time
import pandas as pd
import tqdm
import gzip
import multiprocessing as mp
import cluster_functions

def remove_existing(filtered_mols,args):
    """
    Remove existing compounds from Spacelight/FTrees results

    :param filtered_mols: Searching results from Spacelight/FTrees
    :param args: the args
    :return: List of molecules
    """
    dbname = args.name + ".dbsh"
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    to_db = []
    for line in filtered_mols:
        l = line.split("§")
        reghash = l[0]
        smiles = l[1]
        title = l[2]
        existing_spacehastenid = c.execute("SELECT spacehastenid FROM data WHERE reghash = '"+reghash+"'").fetchall()
        if len(existing_spacehastenid)==0:
            to_db.append((reghash,smiles,title))
    return to_db

def process_sim_results(cycle_dir,args):
    """
    Process results from spacelight and ftrees

    :param cycle_dir: cycle dir
    :param args: the args
    """
    sim_methods = ["spacelight","ftrees"]
    raw_mols = {}
    sims = {}
    for sim_method in sim_methods:
        sims[sim_method] = {}
        print("Reading in "+sim_method+" results...")
        duplicates = 0
        if sim_method == "spacelight":
            sim_field = args.field_similarity_spacelight
        elif sim_method == "ftrees":
            sim_field = args.field_similarity_ftrees
        # was until 0.2, group by title here
        # from 0.3, groupby SMILES here instead of title as in some spaces titles are not unique
        for resfile in tqdm.tqdm(glob.glob(cycle_dir+"/"+sim_method+"result_"+args.name+"_*_1.csv"),mininterval=1):
            resdata = pd.read_csv(resfile)
            for smiles,title,similarity in resdata[["#result-smiles","result-name",sim_field]].values.tolist():
                if smiles not in sims[sim_method] or sims[sim_method][smiles] <= similarity:
                    sims[sim_method][smiles] = similarity
                if smiles not in raw_mols:
                    raw_mols[smiles] = smiles + "§" + title
                else:
                    duplicates += 1
        print(len(raw_mols),"before property control added from",sim_method)
    # split data into smaller chunks 
    smiles_per_cpu = round(float(len(raw_mols))/float(args.cpu))
    print("Splitting molecules, SMILES per CPU:",smiles_per_cpu)

    cpu_counter = 0
    
    simsearch_cycle = functions.get_latest_cycle(args.name)
    if simsearch_cycle == 0:
        simsearch_cycle = 1

    os.system("mkdir -p "+cycle_dir+"/CONTROL")
    os.system("rm -f "+cycle_dir+"/CONTROL/control_*.smi.gz")
    model_version = functions.get_latest_model(args.name)
    conn = sqlite3.connect(args.name + ".dbsh")
    c = conn.cursor()
    model_blob = c.execute("SELECT model_tar FROM models WHERE model_version = "+str(model_version)).fetchall()[0][0]
    model_name = "model_"+args.name+"_ver"+str(model_version)
    with open(model_name+".tar.gz","wb") as f:
        f.write(model_blob)
    os.system("tar -xzf "+model_name+".tar.gz -C "+cycle_dir+"/CONTROL/")
    os.remove(model_name+".tar.gz")
    conn.close()

    prop_args = functions.get_dbsh_properties(args.name + ".dbsh")

    new_file = True
    chunk_size = 0
    for smiles in tqdm.tqdm(raw_mols,mininterval=1):
        if new_file:
            cpu_counter += 1
            csv_filename = cycle_dir + "/CONTROL/control_"+args.name+"_cpu"+str(cpu_counter)+".smi.gz"
            w = gzip.open(csv_filename,"wt")
            new_file = False
        w.write(raw_mols[smiles]+"\n")
        chunk_size += 1
        if chunk_size >= smiles_per_cpu:
            chunk_size = 0
            w.close()
            new_file = True
    if not new_file:
        w.close()
    
    scheduler_functions.write_control_scheduler(cycle_dir+"/CONTROL",args)
    w = open(cycle_dir+"/CONTROL/control.param","wt")
    w.write(prop_args.prop_mw_min+"\n")
    w.write(prop_args.prop_mw_max+"\n")
    w.write(prop_args.prop_slogp_min+"\n")
    w.write(prop_args.prop_slogp_max+"\n")
    w.write(prop_args.prop_hba_min+"\n")
    w.write(prop_args.prop_hba_max+"\n")
    w.write(prop_args.prop_hbd_min+"\n")
    w.write(prop_args.prop_hbd_max+"\n")
    w.write(prop_args.prop_rotbonds_min+"\n")
    w.write(prop_args.prop_rotbonds_max+"\n")
    w.write(prop_args.prop_tpsa_min+"\n")
    w.write(prop_args.prop_tpsa_max+"\n")
    w.close()

    curdir = os.getcwd()
    os.chdir(cycle_dir+"/CONTROL")
    os.system("rm -f jobdone-"+args.name+"-CPU*")
    print("Controlling properties and predicting docking_score via scheduler at "+cycle_dir+"/CONTROL ...")
    os.system("sbatch submit_ctrl_"+args.name+"_cycle"+str(simsearch_cycle)+".sh")
    os.chdir(curdir)

    scheduler_functions.wait_until_jobs_done(cycle_dir+"/CONTROL",args.name,args.cpu)

    print("Reading in predictions...")
    prop_inputs = []
    for prop_input in glob.glob(cycle_dir+"/CONTROL/predicted_propoutput_control_*.csv"):
        prop_inputs.append(prop_input)
    pool = mp.Pool(mp.cpu_count())
    filtered_mol_runs_with_predictions = list(filter(None,list(tqdm.tqdm(pool.imap_unordered(functions.get_rdkit_properties,prop_inputs),mininterval=1,total=len(prop_inputs)))))
    docking_scores = {}
    filtered_mols = []
    for cpu in range(len(filtered_mol_runs_with_predictions)):
        rawmol_list,docking_score_list = filtered_mol_runs_with_predictions[cpu]
        for comp_index in range(len(rawmol_list)):
            # upto 0.2, this was title (index 2)
            # from 0.3, we use reghash (index 0) => check for collisions
            reghash = rawmol_list[comp_index].split("§")[0]
            docking_score = float(docking_score_list[comp_index])
            if reghash not in docking_scores:
                docking_scores[reghash] = docking_score
                filtered_mols.append(rawmol_list[comp_index])
            if docking_scores[reghash] > docking_score:
                docking_scores[reghash] = docking_score

    filtered_sims = {}
    for sim_method in sim_methods:
        filtered_sims[sim_method] = {}
    for filtered_mol in filtered_mols:
        # upto 0.2, was title (index 2)
        # from 0.3, we use reghash (index 0)
        # note that sims is mapped via smiles (index 1) not reghash, so we use that here to switch to reghash
        reghash = filtered_mol.split("§")[0]
        smiles = filtered_mol.split("§")[1]
        for sim_method in sim_methods:
            if smiles in sims[sim_method]:
                filtered_sims[sim_method][reghash] = sims[sim_method][smiles]

    return (filtered_mols,filtered_sims,docking_scores)

def simsearch(args,do_not_update_gui=False):
    """
    Do the similarity searching

    :param args: the args
    :param do_not_update_gui: do not update GUI
    """
    print("Similarity searching:")
    os.system("date")
    dbname = args.name + ".dbsh"
    
    if not os.path.exists(os.getenv("HOME")+"/SPACEHASTEN"):
        os.mkdir(os.getenv("HOME")+"/SPACEHASTEN")

    conn = sqlite3.connect(dbname)
    c = conn.cursor()

    cycle_number = functions.get_latest_cycle(args.name)+1
    cycle_dir = os.getenv("HOME")+"/SPACEHASTEN/SIMSEARCH_"+args.name+"_cycle"+str(cycle_number)
    os.system("mkdir -p "+cycle_dir)
    print("Starting searching cycle",cycle_number)
    print("Picking",args.top,"top docked/predicted compounds for similarity searching queries...")
    if not args.use_predicted:
        print("Using docking_score")
        print("Acquisition method:")
        print(args.acquisition_method)
        if args.acquisition_method == "greedy":
            to_search = c.execute("SELECT smiles,spacehastenid FROM data WHERE query IS NULL AND dock_score IS NOT NULL ORDER BY dock_score LIMIT ?",[args.top]).fetchall()
        elif args.acquisition_method == "clustering":
            to_search = c.execute("SELECT smiles,data.spacehastenid FROM data,clusters WHERE data.spacehastenid = clusters.spacehastenid AND query IS NULL AND dock_score IS NOT NULL GROUP BY clusterid ORDER BY MIN(dock_score) LIMIT ?",[args.top]).fetchall()
        else:
            print("INTERNAL ERROR IN SIMSEARCH!!! UNKNOWN ACQUISITION METHOD")
            exit()
    else:
        print("Using predicted score")
        print("Acquisition method:")
        print(args.acquisition_method)
        if args.acquisition_method == "greedy":
            to_search = c.execute("SELECT smiles,spacehastenid FROM data WHERE query IS NULL AND pred_score IS NOT NULL AND dock_score IS NULL ORDER by pred_score LIMIT ?",[args.top]).fetchall()
        elif args.acquisition_method == "clustering":
            to_search = c.execute("SELECT smiles,data.spacehastenid FROM data,clusters WHERE data.spacehastenid = clusters.spacehastenid AND pred_score IS NOT NULL AND dock_score IS NULL AND query IS NULL GROUP BY clusterid ORDER BY MIN(pred_score) LIMIT ?",[args.top]).fetchall()
        else:
            print("INTERNAL ERROR IN SIMSEARCH!!! UNKNOWN ACQUISITION METHOD")
            exit()
    for smiles,spacehastenid in to_search: c.execute("UPDATE data SET query = "+str(cycle_number)+" WHERE spacehastenid = "+str(spacehastenid))
    conn.commit()
    conn.close()
    w = open(cycle_dir+"/queries_"+args.name+".smi","wt")
    for smiles,spacehastenid in to_search: w.write(smiles.strip() + " " + str(spacehastenid) + "\n")
    w.close()
    scheduler_functions.write_search_scheduler(cycle_dir,args)

    curdir = os.getcwd()
    os.chdir(cycle_dir)
    os.system("rm -f jobdone-"+args.name+"-CPU*")
    print("Running similarity searching via scheduler...")
    os.system("sbatch submit_queries_"+args.name+".sh")
    os.chdir(curdir)

    scheduler_functions.wait_until_jobs_done(cycle_dir,args.name,args.top)

    filtered_mols,filtered_sims,predicted_scores = process_sim_results(cycle_dir,args)
    print(len(filtered_mols),"property-matched mols")
    sim_methods = ["spacelight","ftrees"]
    for sim_method in sim_methods:
        print(sim_method,":",len(filtered_sims[sim_method]))
    only_in_spacelight = 0
    only_in_ftrees = 0
    # this was title before 0.3, now reghash is used
    for reghash in filtered_sims["spacelight"]:
        if reghash not in filtered_sims["ftrees"]:
            only_in_spacelight += 1
    for reghash in filtered_sims["ftrees"]:
        if reghash not in filtered_sims["spacelight"]:
            only_in_ftrees += 1
    print("Only in SpaceLight",only_in_spacelight,", Only in FTrees:",only_in_ftrees)
    if len(filtered_mols) > 0:
        print("Making sure that new compounds have unique RegistrationHashes...")
        df_filtered_mols = pd.DataFrame(filtered_mols,columns=["rawmol"])
        df_filtered_mols[["RegHash","SMILES","SMILES_ID"]] = df_filtered_mols["rawmol"].str.split("§",expand=True)
        reghash_unique_filtered_mols = list(df_filtered_mols.groupby("RegHash").first()["rawmol"])
        print(df_filtered_mols.shape[0],"before unique reghash check")
        print(len(reghash_unique_filtered_mols),"after unique reghash check")

        print("Checking which are already discovered...")
        new_filtered_mols = remove_existing(reghash_unique_filtered_mols,args)
        
        print("Adding new compounds to SQLite3...")
        conn = sqlite3.connect(dbname)
        c = conn.cursor()
        to_db = []
        for reghash,smiles,smilesid in new_filtered_mols:
            spacelight_similarity = None
            ftrees_similarity = None
            # 0.2 used here smilesid, 0.3+ reghash
            if reghash in filtered_sims["spacelight"]:
                spacelight_similarity = filtered_sims["spacelight"][reghash]
            if reghash in filtered_sims["ftrees"]:
                ftrees_similarity = filtered_sims["ftrees"][reghash]
            to_db.append((reghash,smiles.strip(),smilesid,spacelight_similarity,ftrees_similarity,predicted_scores[reghash],cycle_number))
        c.executemany("INSERT INTO data(reghash,smiles,smilesid,spacelight,ftrees,pred_score,simsearch_cycle) VALUES (?,?,?,?,?,?,?)",to_db)
        conn.commit()
        conn.close()
    print("\nSimilarity seaching done!")
    cluster_functions.cluster_dbsh(args)
    os.system("date")
    if not do_not_update_gui:
        args.q.put("DoneSimsearch")
