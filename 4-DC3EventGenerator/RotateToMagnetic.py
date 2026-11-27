
"""
RotateToMagnetic.py

Reads a CSV file containing True North (CS_X_North_m) and True East (CS_Y_East_m)
coordinates, and rotates them to align with Magnetic North based on a given 
magnetic declination angle. 

It outputs a new CSV containing the original data (and preserves original comments)
plus the new fields:
- CS_X_MagNorth_m
- CS_Y_MagWest_m

Usage:
  python RotateToMagnetic.py input_positions.csv <declination_degrees> -o output_positions.csv
"""

import sys
import os
import argparse
import csv
import math

def rotate_to_magnetic(x_north, y_east, declination_deg):
    """
    Rotates True North/East coordinates into Magnetic North/West coordinates.
    
    A positive declination means Magnetic North is East of True North.
    """
        
    delta = math.radians(declination_deg)
    
    """
    Rotates True North/East coordinates into Magnetic North/West coordinates.
    
    A positive declination means Magnetic North is East of True North.
    
    Mathematical Rotation of Axes:
    N_mag = N_true * cos(delta) + E_true * sin(delta)
    E_mag = -N_true * sin(delta) + E_true * cos(delta)
    
    Since we want Magnetic WEST, we invert the Magnetic East axis:
    W_mag = -E_mag = N_true * sin(delta) - E_true * cos(delta)
    """
    
    
    
    x_mag_north = (x_north * math.cos(delta)) + (y_east * math.sin(delta))
    y_mag_west  = (x_north * math.sin(delta)) - (y_east * math.cos(delta))
    
    return x_mag_north, y_mag_west

def main():
    parser = argparse.ArgumentParser(description="Rotates CSV coordinates to Magnetic North/West")
    parser.add_argument("input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("declination", type=float, help="Magnetic declination in degrees (Positive = East of True North)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV file name")
    
    args = parser.parse_args()
    
    input_file = args.input_csv
    output_file = args.output if args.output else input_file.replace(".csv", "_MagRotated.csv")
    declination = args.declination
    
    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    print(f"Applying Magnetic Declination of {declination}° to coordinates...")

    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        
        # 1. Write the new rotation comment at the very top
        outfile.write(f"# Coordinates rotated to Magnetic North. Declination used: {declination} degrees\n")
        
        # 2. Extract and copy all original comments, advancing the file pointer
        last_pos = infile.tell()
        line = infile.readline()
        while line.startswith('#'):
            outfile.write(line)
            last_pos = infile.tell()
            line = infile.readline()
            
        # 3. We read one line too far (the actual header). We need to rewind the file pointer!
        infile.seek(last_pos)
        
        # 4. Now read the CSV safely using standard reader (ignores trailing comma issues)
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Process Header
        header = next(reader)
        try:
            idx_x = header.index('CS_X_North_m')
            idx_y = header.index('CS_Y_East_m')
        except ValueError:
            print("Error: Required columns 'CS_X_North_m' and/or 'CS_Y_East_m' missing from header!")
            print(f"Header found: {header}")
            sys.exit(1)
            
        # Append new headers
        header.append('CS_X_MagNorth_m')
        header.append('CS_Y_MagWest_m')
        writer.writerow(header)
        
        # 5. Process Data Rows
        rows_processed = 0
        for row in reader:
            # Skip empty lines
            if not row or not row[0].strip():
                continue
                
            x_north = float(row[idx_x])
            y_east = float(row[idx_y])
            
            x_mag_north, y_mag_west = rotate_to_magnetic(x_north, y_east, declination)
            
            row.append(f"{x_mag_north:.4f}")
            row.append(f"{y_mag_west:.4f}")
            
            writer.writerow(row)
            rows_processed += 1
            
    print(f"Success! Processed {rows_processed} points.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
