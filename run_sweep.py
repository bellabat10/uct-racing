import subprocess

# Paths to input files
geometry = r"C:\uct-racing\configs\uct_front_geometry_generated.yaml"
sweep = r"C:\uct-racing\configs\uct_front_sweep.yaml"

# Paths to output files
results_csv = r"C:\uct-racing\outputs\results.csv"
plot_image = r"C:\uct-racing\outputs\plot.png"

# 1. Generate the geometry plot
subprocess.run(["kinematics", "visualize", "--geometry", geometry, "--output", plot_image])

# 2. Run the sweep and export results
subprocess.run(["kinematics", "sweep", "--geometry", geometry, "--sweep", sweep, "--out", results_csv])

print("Done! Check the outputs folder.")