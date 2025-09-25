
"""
Optimized Ising Model Simulation with Simulated Annealing

This script simulates a 2D Ising model with random couplings and implements
simulated annealing to find low-energy spin configurations.

Author: Clara
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rd
import winsound
from typing import Tuple, List

class IsingModel:
    """A class to simulate the 2D Ising model with simulated annealing."""
    
    def __init__(self, N: int = 50, h: float = 0.01, k: float = 1.0, 
                 t_f: int = 300000, tau: float = 70000.0):
        """
        Initialize the Ising model.
        
        Parameters:
        -----------
        N : int
            Grid size (N x N)
        h : float
            Magnetic field strength
        k : float
            Boltzmann constant
        t_f : int
            Total number of time steps
        tau : float
            Characteristic time for temperature decay
        """
        self.N = N
        self.h = h
        self.k = k
        self.t_f = t_f
        self.tau = tau
        
        # Initialize random couplings and spins
        self.interactions_vertical = np.random.uniform(0, 1, (N, N))
        self.interactions_horizontal = np.random.uniform(0, 1, (N, N))
        self.spin = np.random.choice([-1, 1], (N, N))
        
        # Energy tracking
        self.energy = 0.0
        self.energy_history = []
        
        # Temperature schedules for comparison
        self.temperature_schedules = {
            'linear': lambda t: 1 - (t / t_f),
            'exponential': lambda t: np.exp(-t / tau),
            'quadratic': lambda t: (1 - (t / t_f) ** 2)
        }
        
    def calculate_energy_change(self, i: int, j: int) -> float:
        """
        Calculate energy change for flipping spin at position (i, j).
        
        Uses periodic boundary conditions via np.roll.
        """
        s0 = self.spin[i, j]
        
        # Get neighbor spins with periodic boundaries
        neighbors = [
            (np.roll(self.spin, 1, 0)[i, j], self.interactions_vertical[i, j]),           # Up
            (np.roll(self.spin, -1, 0)[i, j], np.roll(self.interactions_vertical, -1, 0)[i, j]),  # Down
            (np.roll(self.spin, 1, 1)[i, j], self.interactions_horizontal[i, j]),         # Left
            (np.roll(self.spin, -1, 1)[i, j], np.roll(self.interactions_horizontal, -1, 0)[i, j]) # Right
        ]
        
        # Calculate energy change: ΔE = 2*s0*(Σ J_ij*s_j + h)
        neighbor_energy = sum(J * s for s, J in neighbors)
        return 2 * s0 * (neighbor_energy + self.h)
    
    def metropolis_step(self, temperature: float) -> None:
        """Perform one Metropolis-Hastings Monte Carlo step."""
        # Randomly select a spin
        i, j = rd.randrange(self.N), rd.randrange(self.N)
        
        # Calculate energy change if we flip this spin
        delta_E = self.calculate_energy_change(i, j)
        
        # Metropolis criterion
        if delta_E < 0 or rd.random() < np.exp(-delta_E / (self.k * temperature)):
            self.spin[i, j] *= -1  # Flip the spin
            self.energy += delta_E
    
    def simulate(self, schedule_type: str = 'quadratic', 
                 visualization_freq: int = 1000) -> np.ndarray:
        """
        Run the simulated annealing simulation.
        
        Parameters:
        -----------
        schedule_type : str
            Type of temperature schedule ('linear', 'exponential', 'quadratic')
        visualization_freq : int
            How often to update visualization (0 for no visualization)
            
        Returns:
        --------
        energy_history : np.ndarray
            Array of energy values at each time step
        """
        if schedule_type not in self.temperature_schedules:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        temperature_func = self.temperature_schedules[schedule_type]
        
        # Pre-allocate array for better performance
        self.energy_history = np.zeros(self.t_f + 1)
        self.energy_history[0] = self.energy
        
        # Track temperatures for analysis
        temperatures = []
        
        print("Starting simulation...")
        for t in range(1, self.t_f + 1):
            # Update temperature according to schedule
            T = temperature_func(t)
            temperatures.append(T)
            
            # Perform Monte Carlo step
            self.metropolis_step(T)
            
            # Record energy
            self.energy_history[t] = self.energy
            
            # Optional visualization
            if visualization_freq > 0 and t % visualization_freq == 0:
                self._update_visualization(t)
                
            # Progress indicator
            if t % (self.t_f // 10) == 0:
                print(f"Progress: {100 * t / self.t_f:.1f}%")
        
        return self.energy_history, temperatures
    
    def _update_visualization(self, t: int) -> None:
        """Update the spin configuration visualization."""
        plt.clf()
        plt.imshow(self.spin, cmap='coolwarm', interpolation='nearest')
        plt.title(f'Spin Configuration (t={t})')
        plt.colorbar(label='Spin (+1 / -1)')
        plt.pause(0.001)
    
    def analyze_results(self) -> None:
        """Analyze and plot the simulation results."""
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Final spin configuration
        im = ax1.imshow(self.spin, cmap='coolwarm', interpolation='nearest')
        ax1.set_title('Final Spin Configuration')
        ax1.set_xlabel('X position')
        ax1.set_ylabel('Y position')
        plt.colorbar(im, ax=ax1)
        
        # Energy evolution
        ax2.plot(self.energy_history, 'b-', alpha=0.7, linewidth=1)
        ax2.set_title('Energy Evolution')
        ax2.set_xlabel('Time (iteration)')
        ax2.set_ylabel('Energy (arb. units)')
        ax2.grid(True, alpha=0.3)
        
        # Energy histogram
        ax3.hist(self.energy_history[1000:], bins=50, alpha=0.7, color='green')
        ax3.set_title('Energy Distribution')
        ax3.set_xlabel('Energy')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Magnetization (sum of all spins)
        magnetization = np.sum(self.spin) / (self.N * self.N)
        ax4.bar(['Magnetization'], [magnetization], color='orange', alpha=0.7)
        ax4.set_title(f'Final Magnetization: {magnetization:.3f}')
        ax4.set_ylabel('Average Spin')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Final energy: {self.energy_history[-1]:.2f}")
        print(f"Final magnetization: {magnetization:.3f}")
        print(f"Energy range: {np.min(self.energy_history):.2f} to {np.max(self.energy_history):.2f}")

def main():
    """Main function to run the simulation."""
    # Simulation parameters
    params = {
        'N': 50,           # Grid size
        'h': 0.01,         # Magnetic field
        'k': 1.0,          # Boltzmann constant
        't_f': 100000,     # Time steps (reduced for demo)
        'tau': 70000,      # Temperature decay time
    }
    
    # Create and run model
    model = IsingModel(**params)
    
    # Choose temperature schedule: 'linear', 'exponential', or 'quadratic'
    energy_history, temperatures = model.simulate(
        schedule_type='quadratic', 
        visualization_freq=5000  # Update every 5000 steps
    )
    
    # Analyze results
    model.analyze_results()
    
    # Notification sound when done
    frequency = 500
    duration = 1000
    winsound.Beep(frequency, duration)

if __name__ == "__main__":
    main()