SpaceHASTEN version 0.10, Developed by Tuomo Kalliokoski, Orion Pharma <tuomo.kalliokoski at orionpharma.com>, 2026-02-11

# Introduction

This is a tool that allows you screen BiosolveIT's .space-files using molecular docking tool Glide.
You can get large number of good scoring compounds from those vast chemical spaces of billions of compounds just by docking few million structures.
Hardware requirements: few hundred CPU cores, one reasonable GPU (in year 2024) and few hundred gigabytes of disk space.
Only Linux is supported (tested on Ubuntu 22.04.4 and Rocky Linux 8.8).

SpaceHASTEN requires following commercial software:

* Schrödinger Suite (version 2026-1): phase/ligprep, glide, python API (run)[https://www.schrodinger.com/release-download/]
* SpaceLight (version 2.0.0)[https://www.biosolveit.de/download/?product=spacelight]
* FTrees (version 7.0.0)[https://www.biosolveit.de/download/?product=ftrees]

Depending where you work, anaconda3 [may or may not be free for you][https://www.anaconda.com/pricing]:
* miniconda3 [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh]

In addition, these free tools must be installed:

* slurm workload manager (21.08.5 and 23.02.4 tested)[https://slurm.schedmd.com/documentation.html]
* chemprop (version 2.1.2)[https://github.com/chemprop/chemprop/archive/refs/tags/v2.1.2.tar.gz]

You need some clustering algorithm in addition. Out of the box sphere exclusion clustering with RDKit and FPSim2 is supported. You need to install fpsim2 manually:
* fpsim2 (version 0.7.3)[https://github.com/chembl/FPSim2/archive/refs/tags/0.7.3.tar.gz]

Note that current version of SpaceHASTEN is using chemprop 2.x (older versions of the software used chemprop 1.x).
Following commands in May 2025 can be used to install chemprop v2. This assumes that you have miniconda/anaconda,
modify the included conda_activation_example.sh`(the same script is required when installing SpaceHASTEN as well):

```
source conda_activation_example.sh
tar xzf v2.1.2.tar.gz
cd chemprop-2.1.2
conda create -n chemprop-2.1.2 python=3.11 -y
conda activate chemprop-2.1.2
pip install -e .
```

Installing fpsim2 was like this in Nov 2025:

```
source conda_activation_example.sh
conda create -n fpsim2-0.7.3 python=3.10 -y
conda activate fpsim2-0.7.3
tar xzf v0.7.3.tar.gz
export FPSIM2_MARCH_NATIVE=1
pip install FPSim2-0.7.3/
```

In addition to software, you'll need some chemical spaces to search.
Download them from BiosolveIT:[https://www.biosolveit.de/chemical-spaces/]

Diverse sets of enumerated compounds can be downloaded from:
 
* Enamine: https://enamine.net/compound-collections/real-compounds/real-database-subsets
* SYNPLE: https://www.synplechem.com/solutions/public-library-download
* FreedomSpace: ask from ChemSpace customer service

For more information, please see the publication in JCIM describing on how the method works. [https://doi.org/10.1021/acs.jcim.4c01790]

# Installation and useage

Please check the requirements once more that make sure that you have all needed software and hardware.
If you still think that you have all pieces in place, follow these instructions:

* Check out this repository to some temporary directory: `git clone https://github.com/TuomoKalliokoski/SpaceHASTEN`
* Go to that directory and run the installer: `cd SpaceHASTEN ; python3 install_spacehasten.py`
* Start verify script as suggested by the installer to see that everything is running smoothly.

# Command line interface
You can also run some tasks using the command line interface.

* Importing seeds with 250 CPUs:
```
spacehasten --database db.dbsh --action importsmiles --smiles seeds.smi --dock_params glide.in --dock_grid grid.zip --dock_cpus 250
```

* Run a virtual screening round with greedy acquisition strategy with 100 similarity search queries with 100 cores and docking of 10000 with 250 cores:
```
spacehasten --database db.dbsh --action screen --mode greedy --space chemical.space --simsearch_queries 100 --simsearch_-cpus 100 -dock_mols 100000 --dock_cpus 250
```

* Exporting docking results from SpaceHASTEN database to CSV file:
```
spacehasten --database db.dbsh --action exportcsv --cutoff -10.0 --export_file results.csv
```

* Clustering database:
```
spacehasten --database db.dbsh --action cluster
```

# Note about the clustering
The clustering feature is at really alpha-stage and is currently probably not that useful as an alternative to the default acquisition strategy.


# Version history

* 0.1: initial version
* 0.2: updated the code for new versions of BioSolveIT tools (change in the output similarity-field).
* 0.3: list of changes:
    * can handle identical compound identifiers for different compounds.
    * output now includes the unique SpaceHASTEN_ID, added to smilesid with "§" as separator.
    * checks if started on NFS and warns the user. Insists that .dbsh if saved onto the starting directory.
    * Added literature reference to main screen
* 0.4: chemprop 2.x used instead of chemprop 1.x.
* 0.5:
    * Iteration number now added automatically when exporting poses (issue #5, gui.py)
    * Default exporting poses directory is the current workind girectory (issue #11, gui.py)
    * Proper checking for existing searches implemented to avoid messed up screens (issue #4, gui.py)
    * Job names have been now limited to MAX_JOBNAME_LEN (15) characters to avoid issues with other tools (issue #13, gui.py and cfg.py)
    * When picking Enamine seeds, original compound names are now kept instead of renaming them (issue #12, gui.py and functions.py)
    * Glide .in file is now checked for the common mistake of picking grid generation .in file instead (issue #7, gui.py and functions.py)
    * various smaller changes to installer and post-install verification tool (install_spacehasten.py and verify_spacehasten.py)
* 0.6:
    * properties are now adjustable for each screen separately and not anymore globally controlled via spacehasten.ini (issue #14, gui.py, cfg.py, simsearch_functions.py, importseeds_functions.py, functions.py)
    * similarity searching doesn't crash anymore if no compounds are retrieved (simsearch_functions.py)
    * alpha-version of SGE support (issue #8, cfg.py, slurm_functions.py renamed to scheduler_functions.py, docking_functions.py, prediction_functions.py, training_functions.py, simsearch_functions.py, spacehasten, gui.py, install_spacehasten.py, verify_spacehasten.py, verify)
* 0.7: docking chunk size can be now less than 1000 if more CPUs are available than there are compounds to dock (issue #18, cfg.py, docking_functions.py)
* 0.8:
    * similarity searching results are now compressed to save disk space (issue #21, cfg.py, scheduler_functions.py, simsearch_functions.py)
    * added archiving functionality (issue #16, gui.py, install_spacehasten.py, archive_functions.py)
    * current working directory is now used as the default directory in file requesters (issue #23, gui.py)
* 0.9:
    * experimental alternative acquisition method clustering implemented (issue #24, cfg.py, install_spacehasten.py, gui.py, cluster_functions.py, scheduler_functions.py, simsearch_functions.py, docking_functions.py, importseeds_functions.py, sec_clustering.sh)
    * exporting results now supports clustering (export_functions.py)
    * functions.get_latest_cycle() and functions.get_latest_iteration() now changed to get their information instead from .dbsh instead of SPACEHASTEN working directory, GUI adjusted as this is slightly slower than looking at files (functions.py, gui.py).
    * write access for .dbsh now checked on runtime, user is warned if it is readonly (gui.py)
    * empty training directory cleaned up from SPACEHASTEN working directory (training_functions.py)
    * progress bars now added for scheduler jobs (issue #27, scheduler_functions.py, training_functions.py, simsearch_functions.py, prediction_functions.py, docking_functions.py, cluster_functions.py)
    * command line interface implemented (issue #28, spacehasten.py, cmdline.py, spacehasten, gui.py)
    * installation verification check that clustering works (verify_spacehasten.py)
    * support for seeds from ChemSpace's FreedomSpace added (gui.py, cfg.py, install_spacehasten.py)
* 0.10:
    * New, faster version of Glide now used for docking (scheduler_functions.py)
    * Support the changed job control in Schrodinger Suite 2026-1 (scheduler_functions.py, install_spacehasten.py, cfg.py, verify_spacehasten.py)
    * ASCII-art now correctly printed in all Python versions (install_spacehasten.py, verify_spacehasten.py)
    * version numbering switched from float to string (cfg.py)
    * installer asked the location of seeds twice (install_spacehasten.py)
