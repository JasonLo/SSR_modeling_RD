# About 

This repository (repo) contains the supplementray materials for a manuscript titled ***Using computational modeling to understand the interaction between risk and protective factors in reading disability*** (hereinafter refered to as "paper") submitted to [*Scientific Studies of Reading*](https://www.tandfonline.com/toc/hssr20/current)

For reproduceability, the codes and data in this repo can reproduce all the reported results in our paper.

# Getting Started

Option 1 (Recommanded): You can use [Codeocean](https://codeocean.com/capsule/1821081/tree) to run all python codes in the cloud. Just push **Reproducible Run** button on the upper right corner to run all codes. The results will be stored in results folder after running. I have trouble in configuring an environment that can smoothly run both python and r inside a single codeocean capsule, you may have to run the r codes locally.      

Option 2: More advanced users can clone this repo on [Github](https://github.com/JasonLo/SSR_modeling_RD.git)

## Folder structure

- data: simulations raw data
- code: source code for running analysis
    - python : [python](https://www.python.org/) scripts mainly for visualization
    - r : [R](https://cran.r-project.org/) scripts mainly for statistical analysis
- results: analysis results
- prerun_results: a snapshots of results that I ran, including both r and python parts.

## Codes organization

- For the main analysis, the codes are stored in a jupyter notebook or a R Markdown document with ample annotations.  
- To make the code easier to follow, all basic codes that are not directly relevant to the analysis logics are inside helper.py or helper.r


## Prerequisites
- If you are running on Codeocean, you can ignore this part
- Required python libraries is listed under [code/python/requirements.txt](code/python/requirements.txt)
- Required R libraries is listed under [code/r/requirements.txt](code/r/requirements.txt)


# License

Distributed under the MIT License. See `LICENSE` for more information.

# Contact

Jason Lo - lcmjlo@gmail.com

Project Link: [https://github.com/JasonLo/SSR_modeling_RD](https://github.com/JasonLo/SSR_modeling_RD)