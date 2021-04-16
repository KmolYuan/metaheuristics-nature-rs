# metaheuristics-nature

A collection of nature-inspired metaheuristic algorithms. Provides objective function trait,
well-known methods, and tool functions let you implement your own searching method.

This crate implemented following algorithms:

+ Real-coded Genetic Algorithm (RGA)
+ Differential Evolution (DE)
+ Particle Swarm Optimization (PSO)
+ Firefly Algorithm (FA)
+ Teaching-Learning Based Optimization (TLBO)

Each algorithm gives same API and default parameters to help you test different implementation.
For example, you can test another algorithm by simply replacing `RGA` and `RGASettings` to `DE` and `DESetting`.
