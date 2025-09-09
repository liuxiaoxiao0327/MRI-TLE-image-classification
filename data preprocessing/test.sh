#!/bin/bash
# Description: This script performs batch preprocessing for multiple subjects
# using FreeSurfer's recon-all command.

# Set the FreeSurfer home and subjects directory paths.
# Note: These paths should be adjusted to your specific server configuration.
export FREESURFER_HOME=$HOME/gaoyu/freesurfer [cite: 25]
export SUBJECTS_DIR=$HOME/gaoyu/recon [cite: 26]

# Source the FreeSurfer setup script to configure the environment.
source $FREESURFER_HOME/SetUpFreeSurfer.sh [cite: 27]

# Loop through each subject directory within the specified path.
# The 'ls' command is used to list all subject folders.
# The path should be changed to your specific data directory.
for subj in $(ls /users/Gaoyu/gaoyu/AE_10) [cite: 41]
do
    # Run the recon-all command for the current subject.
    # The '-I' and '-all' flags are used for processing.
    recon-all -s $subj -I -all [cite: 43]
done