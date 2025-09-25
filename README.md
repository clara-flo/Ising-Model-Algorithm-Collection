# Ising Model Algorithms Collection

A comprehensive collection of algorithms for solving and simulating the Ising model, implemented in Python. This repository serves as both an educational resource and a practical toolkit for studying statistical physics and optimization problems.

## 📖 Table of Contents
- [What is the Ising Model?](#what-is-the-ising-model)
- [Project Overview](#project-overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Results and Visualizations](#results-and-visualizations)
- [Contributing](#contributing)
- [References](#references)

## 🔬 What is the Ising Model?

The Ising model is a mathematical model of ferromagnetism in statistical physics. It provides a simplified representation of magnetic materials where atomic spins interact with their neighbors.

### Key Concepts:

- **Spins**: Each lattice site has a discrete spin variable σᵢ ∈ {-1, +1} ("down" or "up")
- **Interactions**: Spins interact with their nearest neighbors through exchange coupling J
- **Magnetic Field**: External magnetic field h can influence spin alignment
- **Hamiltonian**: The energy function governing the system:

  <div align="center">
  
  `H = -J ∑⟨ij⟩ σᵢσⱼ - h ∑ᵢ σᵢ`
  
  </div>

### Physical Significance:
- **Phase Transitions**: Exhibits spontaneous magnetization below critical temperature T꜀
- **Universality**: Belongs to same universality class as many real physical systems
- **Computational Paradigm**: Foundation for understanding Monte Carlo methods and optimization

### Applications Beyond Physics:
- Neuroscience (neural networks)
- Computer vision (image segmentation)
- Optimization problems (traveling salesman)
- Machine learning (Boltzmann machines)
- Social sciences (opinion dynamics)

## 🚀 Project Overview

This project implements various computational algorithms to study the Ising model, focusing on:

1. **Finding ground states** (lowest energy configurations)
2. **Studying phase transitions** near critical temperature
3. **Comparing algorithm efficiency** for different scenarios
4. **Visualizing spin dynamics** and energy landscapes

### Key Features:
- Multiple algorithm implementations with consistent interfaces
- Real-time visualization of spin configurations
- Performance benchmarking and analysis
- Educational explanations and comments
- Modular design for easy extension

## 📊 Algorithms Implemented

### 1. **Monte Carlo Methods**
- **Metropolis-Hastings**: Classic local update algorithm
- **Heat Bath Algorithm**: Alternative acceptance criterion
- **Simulated Annealing**: Temperature scheduling for optimization

### 2. **Cluster Algorithms**
- **Wolff Algorithm**: Reduces critical slowing down
- **Swendsen-Wang**: Multiple cluster updates per step

### 3. **Advanced Methods**
- **Parallel Tempering**: Replica exchange for rough landscapes
- **Mean-Field Approximation**: Analytical approach
- **Belief Propagation**: Message passing algorithm

### 4. **Machine Learning Approaches**
- **Boltzmann Machines**: Neural network implementation
- **Variational Methods**: Optimization-based approaches

## 💻 Installation

### Prerequisites
- Python 3.8+
- NumPy
- Matplotlib
- SciPy

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/ising-model-algorithms.git
cd ising-model-algorithms

# Install dependencies
pip install -r requirements.txt
