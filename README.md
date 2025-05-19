SpaceHASTEN version 0.4, Developed by Tuomo Kalliokoski, Orion Pharma <tuomo.kalliokoski at orionpharma.com>, 2025-05-16

# Introduction

This is a tool that allows you screen BiosolveIT's .space-files using molecular docking tool Glide.
You can get large number of good scoring compounds from those vast chemical spaces of billions of compounds just by docking few million structures.
Hardware requirements: few hundred CPU cores, 1 reasonable GPU (2024) and few hundred gigabytes of disk space.
Only Linux is supported (tested on Ubuntu 22.04.4 and Rocky Linux 8.8).

SpaceHASTEN requires following commercial software:

* Schrödinger Suite (version 2024-4): phase/ligprep, glide, python API (run)[https://www.schrodinger.com/release-download/]
* SpaceLight (version 1.5.0)[https://www.biosolveit.de/download/?product=spacelight]
* FTrees (version 6.13.0)[https://www.biosolveit.de/download/?product=ftrees]

Depending where you work, anaconda3 [may or may not be free for you][https://www.anaconda.com/pricing]:
* miniconda3[https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh]

In addition, these free tools must be installed:

* slurm workload manager (21.08.5 and 23.02.4 tested)[https://https://slurm.schedmd.com/documentation.html]
* chemprop (version 2.1.2)[https://github.com/chemprop/chemprop/archive/refs/tags/v2.1.2.tar.gz]

Note that current version of SpaceHASTEN is using chemprop 2.x (older versions of the software used chemprop 1.x).
Following commands in May 2025 can be used to install chemprop v2. This assumes that you have miniconda/anacond installed with mamba,
modify the included conda_activation_example.sh`(the same script required when installing SpaceHASTEN as well):

```
source conda_activation_example.sh
tar xzf v2.1.2.tar.gz
cd chemprop-2.1.2
conda create -n chemprop-2.1.2 python=3.11 -y
conda activate chemprop-2.1.2
pip install -e .
```

In addition to software, you'll need some chemical spaces to search.
Download them from BiosolveIT:[https://www.biosolveit.de/chemical-spaces/]

Enamine provides diverse sets of Enamine REAL compounds that are good sources of seed molecules for Enamine REALSpace searches.
Download Enamine REAL lead-like subset:[https://enamine.net/compound-collections/real-compounds/real-database-subsets]

For more information, please see the publication describing on how the method works. [https://doi.org/10.1021/acs.jcim.4c01790]

# Installation and useage

Please check the requirements once more that make sure that you have all needed software and hardware.
If you still think that you have all pieces in place, follow these instructions:

* Check out this repository to some temporary directory: `git clone https://github.com/TuomoKalliokoski/SpaceHASTEN`
* Go to that directory and run the installer: `cd SpaceHASTEN ; python3 install_spacehasten.py`
* Start verify script as suggested by the installer to see that everything is running smoothly.

# Version history

* 0.1: initial version
* 0.2: updated the code for new versions of BioSolveIT tools (change in the output similarity-field).
* 0.3: list of changes:
    * can handle identical compound identifiers for different compounds.
    * output now includes the unique SpaceHASTEN_ID, added to smilesid with "§" as separator.
    * checks if started on NFS and warns the user. Insists that .dbsh if saved onto the starting directory.
    * Added literature reference to main screen
* 0.4: chemprop 2.x used instead of chemprop 1.x.
