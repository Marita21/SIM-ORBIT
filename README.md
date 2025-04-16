🛰️ SIM-ORBIT
This project simulates a satellite orbit around Earth using Keplerian elements. It numerically integrates the trajectory in ECI coordinates and converts it to latitude and longitude to visualize the ground track on a global map using Matplotlib and Cartopy.

✨ Features
✅ Keplerian to Cartesian conversion

✅ Orbital integration using solve_ivp from SciPy

✅ Ground track generation with Cartopy

✅ Simple, modular, and readable Python code

🌍 Orbit Visualization
The figure below shows the satellite’s ground track projection over the Earth during one or more orbital periods. The simulation provides insights into coverage patterns and is useful for mission analysis and ground station planning.
  

  <p align="center">
  <img src="Orbit.jpeg" alt="Orbit" width="500">
</p>


   🚀 How to Run
   
1. Install dependencies:

   pip install numpy scipy matplotlib cartopy
   
2. Run the main script:

   python SIM-ORBIT.py



