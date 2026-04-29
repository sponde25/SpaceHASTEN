# SpaceHASTEN: archiving functionality
#
# Copyright (c) 2025 Orion Corporation
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
import tqdm

def archive(args):
    print("Archiving files...")
    resdir = args.scratch + "/" + os.getenv("USER") + "/ARCHIVE_" + args.name
    os.system("rm -fr " + resdir)
    os.system("mkdir -p " + resdir)
    os.system("pigz -c " + args.dbname + ">" + resdir + "/" + args.name + ".dbsh.gz")
    curdir = os.getcwd()
    os.chdir(resdir)
    dirs_to_copy = []
    for cycle in range(1,functions.get_latest_cycle(args.name)+1):
        dirs_to_copy.append(os.getenv("HOME") + "/SPACEHASTEN/SIMSEARCH_" + args.name + "_cycle" + str(cycle) + " " + resdir)
    for iteration in range(1,functions.get_latest_iteration(args.name)+1):
        dirs_to_copy.append(os.getenv("HOME") + "/SPACEHASTEN/DOCKING_" + args.name + "_iter" + str(iteration) + " " + resdir)
    for dir_to_copy in dirs_to_copy:
        os.system("ln -s "+dir_to_copy)
    archive_name = args.dbname.replace(".dbsh",".archived-spacehasten")
    os.system("tar chf " + archive_name + " .")
    os.chdir(curdir)
    os.system("rm -fr " + resdir)
    print("Created",archive_name)
    print("NOTE: remember to clean up the data after you have put the archive in the safe place.")

def restore(args):
    print("Restoring archive to " + os.getenv("HOME") + "/SPACEHASTEN ...")
    resdir = args.scratch + "/" + os.getenv("USER") + "/ARCHIVE_" + args.name
    commands = ["rm -fr " + resdir, "mkdir -p " + resdir, "tar xf " + args.dbname + " -C " + resdir, 
                "gunzip -c " + resdir + "/" + args.name + ".dbsh.gz >" + os.getcwd() + "/" + args.name + ".dbsh",
                "mv " + resdir + "/SIMSEARCH_* "+ os.getenv("HOME") + "/SPACEHASTEN/",
                "mv " + resdir + "/DOCKING_* "+ os.getenv("HOME") + "/SPACEHASTEN/",
                "rm -fr "+resdir]
    for command in tqdm.tqdm(commands): os.system(command)
    print("All done! You may now open " + args.dbname.replace(".archived-spacehasten",".dbsh"))

def clean(args):
    print("Cleaning up job from working space at " + os.getenv("HOME") + "/SPACEHASTEN ...")
    remove_dbsh = "rm -f " + args.name + ".dbsh"
    os.system(remove_dbsh)
    dirs_to_remove = []
    for cycle in range(1,functions.get_latest_cycle(args.name)+1):
        dirs_to_remove.append("rm -fr " + os.getenv("HOME") + "/SPACEHASTEN/SIMSEARCH_" + args.name + "_cycle" + str(cycle))
    for iteration in range(1,functions.get_latest_iteration(args.name)+1):
        dirs_to_remove.append("rm -fr " + os.getenv("HOME") + "/SPACEHASTEN/DOCKING_" + args.name + "_iter" + str(iteration))
    for dir_to_remove in tqdm.tqdm(dirs_to_remove):
        os.system(dir_to_remove)
    print("All cleaned up. You may return the job to workspace by restoring " + args.name + ".archived-spacehasten")
