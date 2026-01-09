# SpaceHASTEN: functions used by various parts of the program
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
from rdkit import Chem
from rdkit.Chem import RegistrationHash
import argparse
import shutil
import os
import glob
import pandas as pd
import sqlite3
import cfg

import sys
import subprocess
from types import SimpleNamespace

def get_dbsh_properties(dbname):
    """
    Get property ranges from .dbsh
    
    :params dbname: contains .dbsh filename
    :return args Namespace containg values
    """
    job_args = SimpleNamespace()
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    try:
        job_args.prop_mw_min,job_args.prop_mw_max = c.execute("SELECT min_limit,max_limit FROM properties WHERE property LIKE 'mw'").fetchall()[0]
        job_args.prop_slogp_min,job_args.prop_slogp_max = c.execute("SELECT min_limit,max_limit FROM properties WHERE property LIKE 'slogp'").fetchall()[0]
        job_args.prop_hba_min,job_args.prop_hba_max = c.execute("SELECT min_limit,max_limit FROM properties WHERE property LIKE 'hba'").fetchall()[0]
        job_args.prop_hbd_min,job_args.prop_hbd_max = c.execute("SELECT min_limit,max_limit FROM properties WHERE property LIKE 'hbd'").fetchall()[0]
        job_args.prop_rotbonds_min,job_args.prop_rotbonds_max = c.execute("SELECT min_limit,max_limit FROM properties WHERE property LIKE 'rotbonds'").fetchall()[0]
        job_args.prop_tpsa_min,job_args.prop_tpsa_max = c.execute("SELECT min_limit,max_limit FROM properties WHERE property LIKE 'tpsa'").fetchall()[0]
    except:
        print("NOTE: Most likely old .dbsh file and No properties found in "+dbname+", using defaults from spacehasten.ini....")
        def_config = cfg.SpaceHASTENConfiguration()
        job_args.prop_mw_min = def_config.PROP_MW_MIN_DEFAULT
        job_args.prop_mw_max = def_config.PROP_MW_MAX_DEFAULT
        job_args.prop_slogp_min = def_config.PROP_SLOGP_MIN_DEFAULT
        job_args.prop_slogp_max = def_config.PROP_SLOGP_MAX_DEFAULT
        job_args.prop_hba_min = def_config.PROP_HBA_MIN_DEFAULT
        job_args.prop_hba_max = def_config.PROP_HBA_MAX_DEFAULT
        job_args.prop_hbd_min = def_config.PROP_HBD_MIN_DEFAULT
        job_args.prop_hbd_max = def_config.PROP_HBD_MAX_DEFAULT
        job_args.prop_rotbonds_min = def_config.PROP_ROTBONDS_MIN_DEFAULT
        job_args.prop_rotbonds_max = def_config.PROP_ROTBONDS_MAX_DEFAULT
        job_args.prop_tpsa_min = def_config.PROP_TPSA_MIN_DEFAULT
        job_args.prop_tpsa_max = def_config.PROP_TPSA_MAX_DEFAULT
    conn.close()
    return job_args


def update_dbsh_properties(dbname,args):
    """ 
    Update property ranges in .dbshfile using args

    :params dbname: contains .dbsh filename
    :param args: contains all property ranges
    """
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS properties")
    c.execute("CREATE TABLE properties (property TEXT,is_double INTEGER,min_limit TEXT,max_limit TEXT)")
    c.execute("INSERT INTO properties (property,is_double,min_limit,max_limit) VALUES ('mw',1,"+args.prop_mw_min+","+args.prop_mw_max+")")
    c.execute("INSERT INTO properties (property,is_double,min_limit,max_limit) VALUES ('slogp',1,"+args.prop_slogp_min+","+args.prop_slogp_max+")")
    c.execute("INSERT INTO properties (property,is_double,min_limit,max_limit) VALUES ('hba',0,"+args.prop_hba_min+","+args.prop_hba_max+")")
    c.execute("INSERT INTO properties (property,is_double,min_limit,max_limit) VALUES ('hbd',0,"+args.prop_hbd_min+","+args.prop_hbd_max+")")
    c.execute("INSERT INTO properties (property,is_double,min_limit,max_limit) VALUES ('rotbonds',0,"+args.prop_rotbonds_min+","+args.prop_rotbonds_max+")")
    c.execute("INSERT INTO properties (property,is_double,min_limit,max_limit) VALUES ('tpsa',1,"+args.prop_tpsa_min+","+args.prop_tpsa_max+")")
    conn.commit()
    conn.close()

def check_glide_gridgen_input(glideinfile):
    """ 
    Check against glide grid generation input instead of docking input

    :return True if the file does look like grid generation file
    """
    warning_signs = ["GRID_CENTER", "INNERBOX", "OUTERBOX", "RECEP_FILE"]
    for line in open(glideinfile):
        for warning_sign in warning_signs:
            if warning_sign in line:
                return True
    return False
        
def check_nfs(filename):
    """ 
    Check if filename is on NFS drive

    :return True if the file is on NFS
    """
    cmd = "stat -f -c %T "+filename
    process = subprocess.Popen(cmd,shell=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    result = process.stdout.readlines()
    for resline in result:
        if "nfs" in resline:
            return True
    return False

def get_rdkit_properties(csv_filename):
    """
    Read rdkit output

    :return molecules that passed the properties criteria
    """
    passed_rows = pd.read_csv(csv_filename)
    return (passed_rows["smilesid"].tolist(),passed_rows["docking_score"].tolist())

def get_latest_model(name):
    """
    Get the version number of latest model
    
    :param name: args.name usually
    :return int of the latest version, 0 if no models exist
    """
    dbname = name + ".dbsh"
    if not os.path.exists(dbname):
        raise SystemExit("Internal Error: SpaceHASTEN database ("+dbname+") missing when called get_latest_model()!!!")
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    model_version = c.execute("SELECT COUNT(*) FROM models").fetchall()[0][0]
    if model_version == None:
        model_version = 0
    conn.close()
    return model_version

def get_latest_cycle(name):
    """
    Get the latest searching cycle that has been done
    
    :param name: usually args.name
    :return int of the latest cycle, 0 if no exists
    """

    dbname = name + ".dbsh"
    if not os.path.exists(dbname):
        raise SystemExit("Internal Error: SpaceHASTEN database ("+dbname+") missing when called get_latest_cycle()!!!")
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    cycle = c.execute("SELECT MAX(simsearch_cycle) FROM data").fetchall()[0][0]
    if cycle == None:
        cycle = 0
    conn.close()
    return cycle

def get_latest_iteration(name):
    """
    Get the latest docking iteration that has been done
    
    :param name: usually args.name
    :return int of the latest iteration, 0 if no exists
    """
    dbname = name + ".dbsh"
    if not os.path.exists(dbname):
        raise SystemExit("Internal Error: SpaceHASTEN database ("+dbname+") missing when called get_latest_iteration()!!!")
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    iteration = c.execute("SELECT MAX(dock_iteration) FROM data").fetchall()[0][0]
    if iteration == None:
        iteration = 0
    conn.close()
    return iteration

def mol2hash(line):
    """
    Take a § limited line and generate RegistrationHash

    :param line: The line to be parsed
    :return: RegistrationHash + input-line (§ limited)
    """
    l = line.split("§")
    smiles = l[0]
    mol_id = l[1]
    docking_score = l[2]

    mol = Chem.MolFromSmiles(smiles)
    if mol == None:
        return None

    return RegistrationHash.GetMolLayers(mol)[RegistrationHash.HashLayer.TAUTOMER_HASH] + "§" + line

def cxsmi2smi(cxsmiles_with_id):
    """
    Take a cxsmiles and id (sep §) and return smiles plus linefeed

    :param cxsmiles_with_id: The cxsmiles and id (sep §) to be parsed
    :return: SMILES with linefeed added
    """

    s = cxsmiles_with_id.split("§") 
    mol = Chem.MolFromSmiles(s[0])
    if mol == None:
        return None
    return Chem.MolToSmiles(mol) + " " + s[1] + "\n"
