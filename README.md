This repository contains the code for the article by Keskin et al., titled "Adaptive Cross-Country Optimization Strategies in Thermal Soaring Birds."

The soaring_strategy_update.py file requires CSV files containing raw GPS points (latitude, longitude, and altitude). The specific files used in this study can be accessed at https://doi.org/10.5281/zenodo.12607007.
This Python code is used to create glide polars and calculate the daily climb rate and horizontal speed during thermalling, as explained in detail in the related article. 

The prune_tree.r file is used to prune the full phylogeny obtained from BirdTree.org.

The pgls_figure3_stats.r and pgls_fig5_stats.r files are used to perform Phylogenetic Generalized Least Squares (PGLS) regression analysis. The difference between the two files is that the analysis for Figure 3 uses a no-intercept approach, while the analysis for Figure 5 includes an intercept.
