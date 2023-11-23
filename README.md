# LBG from scratch

Implementing Linde–Buzo–Gray algorithm for clustering from scratch in Python.

## Structure
	├── lbg.py              # implemented function for LBG algorithm
	├── lbg_example.ipynb   # examples on applying to datasets
	├── README.md 

## LBG algorithm
### Introduction
Linde–Buzo–Gray algorithm is a clustering algorithm similar to K-means. The most notable difference is that LBG split code-vector by adding or subtracting epsilon from already-existed code-vectors to increase the number of code-vectors. 

### Steps
1. Initialize codebook $C$ as $c_0$ where $c_0$ is mean of all input data vectors.
2. While length of $C$ < expected size of codebook: 
	 - Split code vector.
	 - While $D'- D < err$
	   - Find closest code vector for all input data vectors.
	   - Update code vectors by the centroid of each clusters.
	   - Compute Distortion $D$
3. End

## References

1. [Linde–Buzo–Gray algorithm](https://en.wikipedia.org/wiki/Linde%E2%80%93Buzo%E2%80%93Gray_algorithm)
2. [An Algorithm for Vector Quantizer Design](https://ieeexplore.ieee.org/document/1094577)
3. [Python and Java Implementations for Linde-Buzo-Gray](https://mkonrad.net/projects/gen_lloyd.html) 



