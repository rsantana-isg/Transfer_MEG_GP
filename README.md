# Transfer_MEG_GP

Contains the main implementations of programs for the paper:  GP-based methods for domain adaptation: Using brain decoding across subjects as a test-case, by   R. Santana, L. Marti, and M. Zhang (Submitted for publication).

This project implements different approaches to the creation of classifiers for MEG data in brain-decoding experiments.
The description of the main steps for data processing and the explanation of how to run  the different programs is givenven in the file Steps_For_Problem_Solution.pdf.

 The evolutionary algorithms are based on the DEAP library that implements evolutionary algorithms (https://github.com/deap/deap). The importance weighting crossvalidation schemes which are used with the classical classifiers are implemented using the \texttt{libtlda} Python library \footnote{ \url{https://github.com/wmkouw/libTLDA}}, which is a library of transfer learners and domain-adaptive classifiers.  
