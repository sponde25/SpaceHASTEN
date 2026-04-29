# SpaceHASTEN: command line interface
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
import argparse

spacehastentutorial="""

Importing SMILES into SpaceHASTEN database:
spacehasten --database db.dbsh --action importsmiles --smiles seeds.smi --dock_params glide.in --dock_grid grid.zip --dock_cpus 250

Virtual Screening with SpaceHASTEN:
spacehasten --database db.dbsh --action screen --mode greedy --space mols.space --simsearch_queries 100 --simsearch_cpus 100 --dock_mols 100000 --dock_cpus 250

Exporting docking results from SpaceHASTEN database to CSV file:
spacehasten --database db.dbsh --action exportcsv --cutoff -10.0 --export_file results.csv

Cluster compounds in SpaceHASTEN database:
spacehasten --database db.dbsh --action cluster

"""

def parse_cmdline():
    """
    Parse command line using ArgumentParser

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description=spacehastentutorial)
    parser.add_argument("--database",required=False,type=str,help="SpaceHASTEN database")
    parser.add_argument("--action",required=False,type=str,choices=["importsmiles","screen","exportcsv","cluster"],help="Action to perform")
    parser.add_argument("--smiles",required=False,type=str,help="Input seeds SMILES file for importing")
    parser.add_argument("--dock_params",required=False,type=str,help="Docking parameters file")
    parser.add_argument("--dock_grid",required=False,type=str,help="Docking grid file")
    parser.add_argument("--mode",required=False,type=str,choices=["greedy","clustering"],help="Acquisition method")
    parser.add_argument("--space",required=False,type=str,help="Chemical space to screen")
    parser.add_argument("--simsearch_queries",required=False,type=int,help="Number of SimSearch queries")
    parser.add_argument("--simsearch_cpus",required=False,type=int,help="Number of parallel CPUs for SimSearch")
    parser.add_argument("--dock_mols",required=False,type=int,help="Number of compounds to dock")
    parser.add_argument("--dock_cpus",required=False,type=int,help="Number of parallel CPUs for docking")
    parser.add_argument("--cutoff",required=False,type=float,help="Score cutoff for exporting docking results")
    parser.add_argument("--export_file",required=False,type=str,help="Output file for exporting docking results")
    cmdline_args = parser.parse_args()
    return cmdline_args