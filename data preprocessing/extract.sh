#!/bin/bash
# Description: This script extracts statistical data from FreeSurfer's output
# for various features, parcellations, and hemispheres.

# Set the FreeSurfer home and subjects directory paths.
export FREESURFER_HOME=$HOME/gaoyu/freesurfer [cite: 70]
export SUBJECTS_DIR=$HOME/gaoyu/recon_AE [cite: 71]

# Source the FreeSurfer setup script.
source $FREESURFER_HOME/SetUpFreeSurfer.sh [cite: 72]

# Define arrays for the different parameters to be extracted.
label=(area volume thickness thicknessstd thickness.T1 meancurv gauscurv foldind curvind) [cite: 74, 75]
hemia=(lh rh) [cite: 76]
parc=(aparc.a2009s aparc.DKTatlas aparc.pial aparc.BA_exvivo BA_exvivo.thresh) [cite: 77]
b="_" [cite: 78]

# Outer loop for iterating through different parcellation schemes.
for ((i=0; i<=4; i++)) [cite: 79]
do
    aparc=${parc[i]} [cite: 81]

    # Middle loop for iterating through the hemispheres (left and right).
    for ((k=0; k<=1; k++)) [cite: 83]
    do
        hem=${hemia[k]} [cite: 85]

        # Inner loop for iterating through the different statistical labels.
        for ((j=0; j<=8; j++)) [cite: 86]
        do
            lbl=${label[j]} [cite: 88]

            # Construct the output filename.
            name=$hem$b$aparc$b$lbl [cite: 89]
            
            # Run the aparcstats2table command to generate the data table.
            # The list of subjects should be adjusted to your specific dataset.
            aparcstats2table -hemi $hem \
            -meas $lbl \
            -parc $aparc \
            -tablefile $name.txt \
            --subjects CAIJIA CHENPANYU SHENMINGLI ZHAOMINGHUI YANGHANYUE LIXIAOHUA LIUCHANGZHENG CHENZHEN [cite: 90, 91, 92, 93]
        done
    done
done