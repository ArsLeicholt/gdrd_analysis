#!/bin/bash

# Get current date for directory naming
CURRENT_DATE=$(date +%Y%m%d)
ANALYSIS_DIR="md_analysis_${CURRENT_DATE}"
BASE_DIR=$(pwd)
LOG_FILE="${BASE_DIR}/${ANALYSIS_DIR}/output.dat"

# Create analysis directory at the same level as gdrd_D_ directories
mkdir -p "${ANALYSIS_DIR}"

# Function to process one run of a species
process_run() {
    local species_dir=$1
    local run=$2
    local species=$(basename ${species_dir} | sed 's/gdrd_D_//')
    
    {
        echo "Processing species: ${species}, run: ${run}"
        
        cd "${BASE_DIR}/${species_dir}/run_${run}" || return
        
        # Rename files if needed
        if [ -f "md_0_1_noPBC.xtc" ]; then
            mv "md_0_1_noPBC.xtc" "${species}${run}.xtc"
        fi
        if [ -f "md_0_1.tpr" ]; then
            mv "md_0_1.tpr" "${species}${run}.tpr"
        fi
        
        # Check if files exist
        if [ ! -f "${species}${run}.xtc" ] || [ ! -f "${species}${run}.tpr" ]; then
            echo "Required files for ${species}${run} not found, skipping..."
            return
        fi
        
        # Run RMSD analysis
        echo "Running RMSD analysis for ${species}${run}..."
        gmx rms -s "${species}${run}.tpr" -f "${species}${run}.xtc" -o "rmsd_${species}${run}.xvg" -tu ns <<EOF
4
4
EOF
        
        # Run RMSF analysis
        echo "Running RMSF analysis for ${species}${run}..."
        gmx rmsf -f "${species}${run}.xtc" -s "${species}${run}.tpr" -o "rmsf_${species}${run}.xvg" -res <<EOF
3
EOF
        
        # Run gyration analysis
        echo "Running gyration analysis for ${species}${run}..."
        gmx gyrate -s "${species}${run}.tpr" -f "${species}${run}.xtc" -o "gyrate_${species}${run}.xvg" <<EOF
1
EOF
        
        # Convert XVG to DAT files
        for type in rmsd rmsf gyrate; do
            if [ -f "${type}_${species}${run}.xvg" ]; then
                grep -v [#@] "${type}_${species}${run}.xvg" > "${type}_${species}${run}.dat"
                echo "Converted ${type}_${species}${run}.xvg to .dat"
            fi
        done
        
        # Create 250ns PDB snapshot
        echo "Creating 250ns PDB snapshot for ${species}${run}..."
        gmx trjconv -f "${species}${run}.xtc" -s "${species}${run}.tpr" -o "${species}${run}_250ns.pdb" -b 200000 -e 200001 <<EOF
1
EOF
        
        # Copy files to analysis directory
        echo "Copying files to analysis directory for ${species}${run}..."
        cp "${species}${run}.xtc" "${species}${run}.tpr" \
           "rmsd_${species}${run}.xvg" "rmsd_${species}${run}.dat" \
           "rmsf_${species}${run}.xvg" "rmsf_${species}${run}.dat" \
           "gyrate_${species}${run}.xvg" "gyrate_${species}${run}.dat" \
           "${species}${run}_250ns.pdb" \
           "${BASE_DIR}/${ANALYSIS_DIR}/"
           
        echo "Completed processing ${species}${run}"
    } >> "${LOG_FILE}" 2>&1
}

export -f process_run
export BASE_DIR ANALYSIS_DIR LOG_FILE

# Create a list of all jobs to run
echo "Preparing job list..." > "${LOG_FILE}"
jobs_list=()
for species_dir in gdrd_D_*; do
    if [ -d "$species_dir" ]; then
        for run in {1..3}; do
            jobs_list+=("$species_dir $run")
        done
    fi
done

# Run jobs in parallel using 20 cores
echo "Starting parallel processing with 20 cores..." >> "${LOG_FILE}"
printf "%s\n" "${jobs_list[@]}" | xargs -P 20 -I {} bash -c 'process_run {}' &

echo "Analysis started in background. Check ${LOG_FILE} for progress."
