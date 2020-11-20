Provided is a small program demonstrating unusual behavior with the Xilinx solver library function gtsv. Running the gtsv solver on diagonals of length 3136 randomly geneated doubles, and comparing to reference lapack implementation we get that the sum over the solutions is similar, but the found solution in the Xilinx solver library is not actually a valid solution to the system. Considering that the sum of the solutions is the same across both solvers, I propose there is some ordering issue in the xf::solver implementation. 

 Output from running the host program is: 

Total error on solution /w xf::solver: 883143
sum of solution /w xf::solver: 4965.55
Total error on solution /w lapack: 2.81301e-11
sum of solution /w lapack: 4965.55

I propose there is some kind of ordering problem.  
