# Design Optimization of High-Rise Buildings: An Aerodynamic Study using Computational Fluid Dynamics (CFD)

## Subtitle: A Two-Stage Python Workflow for Twist Angle Optimization and Visualization
### 1. Introduction and Project Goal
Introduction and Project GoalThis document describes a two-stage computational workflow developed in Python to investigate the impact of façade twisting on the aerodynamic performance of tall buildings, measured by their resistance to wind flow (drag reduction and wake minimization). The methodology utilizes a simplified Lattice-Boltzmann based CFD model implemented within the $\text{COMPAS}$ framework for rapid iterative analysis.

$\text{The primary goal of this project is twofold}$
1.  Optimization: To determine the single optimal twist angle that minimizes the wind resistance score across 360 geometric variations of a conceptual building mass.
2.  Visualization: To apply the optimal angle to a customized, scaled-up DNA-style building design and visualize the resulting complex flow patterns using $\text{CFD}$ stream-lines.

### 2. Stage I: Optimization and Analysis (building_generator.ipynb)
The first stage is dedicated to systematically exploring the design space and identifying the best performing geometry. This process is handled by the building_generator.ipynb script (which aligns with the wind_analysis.ipynb discussion).
2.1  Geometric Generation and Parameter Sweep
The program generates 360 distinct building masses, where the only variable is the total façade twist angle, ranging from $0^\circ$ to $359^\circ$ in $1^\circ$ increments. All other parameters, such as the building's base radius ($R$), pitch, and height, are kept constant to ensure a controlled comparison.
2.2  CFD Simulation and Velocity Field Calculation
For each of the 360 models, the script performs a simplified two-dimensional $\text{CFD}$ simulation. The wind field is modeled as a combination of uniform freestream flow ($U_{\infty}$) and a calculated swirl/vortex component that interacts with the building mass. The velocity vector at any point $P(x, y, z)$ is determined by the total vector sum, often integrating the flow using a method like Runge-Kutta (RK2).
2.3  Aerodynamic Scoring and Ranking
To quantify performance, a custom scoring metric is used, derived from the concept of momentum deficit in the building's wake:$$\text{Score} = \left( 1 - \frac{\sum \rho (U_{\infty}^2 - U_i^2)}{\sum \rho U_{\infty}^2} \right) \times 100$$Where:$\rho$ is the fluid density (constant).$U_{\infty}$ is the freestream wind speed.$U_i$ is the actual wind speed at point $i$ in the flow field.A higher score (closer to 100) indicates less wind energy loss (lower momentum deficit), meaning the building is more aerodynamically efficient. The script computes the score for all 360 angles and sorts the results.
2.4  Output of Optimal Angle
The process concludes by extracting the total twist angle that yielded the highest aerodynamic score. This single value is then exported to a standardized file format (e.g., best_angle_data.json) for use in the next stage.

### 3.  Stage II: Design Application and Visualization (dna_building_CFD_generator.ipynb)
The second stage, handled by the dna_building_CFD_generator.ipynb script, uses the optimized result to finalize a specific architectural design and perform a final validation.
3.1  Customized Design Input
The script allows the user to define the actual architectural parameters for the final design, such as CUSTOM_RADIUS, CUSTOM_HEIGHT, and CUSTOM_FLOORS. This is where the optimized angle is applied to a realistic, scaled-up building.
3.2  Single-Run CFD Analysis (Non-Scored)
The imported optimal angle is used to generate the final building geometry (a DNA-style mesh). Crucially, a single $\text{CFD}$ simulation is run on this new, custom-sized building.
The purpose of this run is purely for data generation and visualization, not for re-optimization. The simulation calculates the new velocity field based on the significantly altered geometry and larger domain, generating the streamline segments (segments list) and corresponding velocity magnitudes.
3.3  Flow Visualization
The final step is to combine the refined geometry and the $\text{CFD}$ data for visualization within the $\text{COMPAS}$ Viewer:
3.3.1  Building Model: The custom DNA-style building mesh is rendered.
3.3.2  Streamlines: The calculated streamlines are drawn as a series of segments/arrows.
3.3.3  Color Mapping: The streamlines are color-coded based on local velocity: Blue typically represents slow/stagnant flow, and Red/Orange represents fast/accelerated flow. This visually confirms how the optimized twist angle manages wind acceleration and wake formation around the tall structure.

### 4.  Conclusion

This two-stage workflow demonstrates an effective coupling of computational geometry and fluid dynamics analysis. By systematically optimizing a single geometric variable (twist angle) in Stage I and then applying and visualizing the result on a detailed, scaled-up model in Stage II, the project provides both a quantitative validation (through the score) and a qualitative understanding (through the flow visualization) of how architectural massing can be leveraged to mitigate adverse wind effects.
