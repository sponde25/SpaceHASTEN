#!/bin/bash
cat >sec_clustering.py <<'EOF'
# SpaceHASTEN: clustering script
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
# Reference:
# https://rdkit.blogspot.com/2020/11/sphere-exclusion-clustering-with-rdkit.html
#
#
from rdkit.SimDivFilters import rdSimDivPickers
import pickle
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
import tqdm
import sys

# check for memory
from FPSim2 import FPSim2Engine
import glob
import pickle

if len(sys.argv) == 1:
    fp = []
    mols = []
    print("Loading fingerprints and mols...")
    for fpname in tqdm.tqdm(glob.glob("fp_*.p"),mininterval=1):
        chunk = pickle.load(open(fpname,"rb"))
        mols += open(fpname.replace("fp_","").replace(".p","")).readlines()
        fp += chunk
    smiles = {}
    for mol in mols:
        smiles[int(mol.split(" ")[1])] = mol.split(" ")[0]
    lp = rdSimDivPickers.LeaderPicker()
    # similarity is given, convert to distance
    distance_thresh = 1.0 - 0.3
    print(f"Identifying cluster centroids with distance of {distance_thresh}...")
    cluster_centroids = lp.LazyBitVectorPick(fp,len(fp),distance_thresh)
    num_clusters = len(cluster_centroids)
    print("Cluster centroids identified:",num_clusters)

    w = open("cluster_centroids.smi","w")
    for cluster_centroid in cluster_centroids:
        w.write(mols[cluster_centroid])
    w.close()
else:
    if sys.argv[1] == "make_fp":
        fps = []
        morgan2_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
        for line in open(sys.argv[2]):
            mol = Chem.MolFromSmiles(line.split(" ")[0])
            fps.append(morgan2_generator.GetFingerprint(mol))
        pickle.dump(fps,open("fp_"+sys.argv[2]+".p","wb"))
    elif sys.argv[1] == "search_fp":
        fpe = FPSim2Engine("fp.h5")
        results = []
        for line in open(sys.argv[2]):
            l = line.strip().split()
            res = fpe.similarity(l[0],threshold=0.30, metric="tanimoto", n_workers=1)
            for comp_id,sim in res:
                results.append((int(l[1]),comp_id,sim))
        pickle.dump(results,open("comps_"+sys.argv[2]+".p","wb"))
    elif sys.argv[1] == "compile":
        mols = set()
        print("Compiling results, reading input molecules...")
        for line in open(sys.argv[2]):
            l = line.strip().split(" ")
            mols.add(int(l[1]))

        duplicates = 0
        clusters = {}
        sims = {}
        print("Assign clusters...")
        for filename in tqdm.tqdm(glob.glob("comps_x*.p")):
            for clusterid,compid,sim in pickle.load(open(filename,"rb")):
                if compid not in clusters or sims[compid] < sim:
                    clusters[compid] = clusterid
                    sims[compid] = sim

        print("Writing final output to clustering.csv...")
        w = open("clustering.csv","w")
        w.write("spacehastenid,clusterid\n")
        for mol in mols:
            w.write(str(mol)+","+str(clusters[mol])+"\n")
        w.close()
    else:
        print("ERROR! INVALID MODE CALLED IN SEC_CLUSTERING.PY!")
        exit(1) 
EOF
rm -f x* temp_chunk_* comps_x*
split --lines=10000 $1
ls -1 x* | gawk '{ print "python3 sec_clustering.py make_fp",$1 }' | parallel --bar
echo Preparing FPSim2...
fpsim2-create-db $1 fp.h5 --fp_type Morgan --fp_params '{"radius": 2, "fpSize": 1024}' --processes $(nproc)
echo Running clustering...
python3 sec_clustering.py
rm -f x* temp_chunk_*
echo Similarity searching...
split --lines=10 cluster_centroids.smi
ls -1 x* | gawk '{ print "python3 sec_clustering.py search_fp",$1 }' | parallel --bar
echo Assigning compounds to clusters...
python3 sec_clustering.py compile $1
echo "Done!"
