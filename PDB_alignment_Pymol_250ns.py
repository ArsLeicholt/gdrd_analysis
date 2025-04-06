#!/usr/bin/env python3
from pymol import cmd
import os
import sys

def merge_triplicates(species_prefix):
    """
    Load and visualize triplicate PDB structures from 250ns snapshots.
    Structures will be aligned and displayed with different colors matching plot_md.py.
    
    Parameters:
    species_prefix (str): Species prefix (e.g., 'ana', 'mel', etc.)
    """
    # Color scheme matching plot_md.py
    colors = ['lightblue', 'deepsalmon', 'palegreen']
    
    # Load each structure
    for i in range(1, 4):
        pdb_file = f"{species_prefix}{i}_250ns.pdb"
        structure_name = f"{species_prefix}{i}"
        
        # Check if file exists
        if not os.path.exists(pdb_file):
            print(f"Error: {pdb_file} not found")
            continue
            
        # Load structure
        cmd.load(pdb_file, structure_name)
        
        # Color the structure
        cmd.color(colors[i-1], structure_name)
        
        # Show as cartoon representation
        cmd.show_as("cartoon", structure_name)
        
        # Hide waters and ions
        cmd.remove(f"resn HOH or resn NA or resn CL and {structure_name}")
        
    # Align all structures to the first one
    ref = f"{species_prefix}1"
    for i in range(2, 4):
        mobile = f"{species_prefix}{i}"
        cmd.align(mobile, ref)
    
    # Center view
    cmd.center()
    cmd.zoom()
    
    # Set nice visualization
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_transparency", 0.2)
    cmd.bg_color("white")
    
    # Save the session
    cmd.save(f"{species_prefix}_merged.pse")
    
    # Save as image
    cmd.ray(1024, 1024)
    cmd.png(f"{species_prefix}_merged.png", dpi=300)

def main():
    if len(sys.argv) != 2:
        print("Usage: ./merge_pdb.py <species_prefix>")
        print("Example: ./merge_pdb.py ana")
        sys.exit(1)
        
    species_prefix = sys.argv[1]
    merge_triplicates(species_prefix)

if __name__ == "__main__":
    main()
