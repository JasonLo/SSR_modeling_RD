# Readme

## About

This repository (repo) contains the supplementray materials for a manuscript titled ***Using computational modeling to understand the interaction between risk and protective factors in reading disability*** (hereinafter refereed to as "paper") submitted to [*Scientific Studies of Reading*](https://www.tandfonline.com/toc/hssr20/current). For reproducibility, the codes and data in this repo can reproduce all the reported results in our paper.

## Getting Started

### Running options

1. CodeOcean (Recommended): You can use [CodeOcean](https://codeocean.com/capsule/1821081/tree) to run all python codes in the cloud. Just push *Reproducible Run* button on the upper right corner to run all codes. The results will be stored in results folder after running. I have trouble in configuring an environment that can run both python and r inside a single CodeOcean capsule, you may have to run the r codes locally or just read my outputs inside *prerun_results* folder.

2. Locally: More advanced users can clone this repo on [Github](https://github.com/JasonLo/SSR_modeling_RD.git) and run locally

### Folder structure

- data: simulations raw data
- code: source code for running analysis
  - python : [python](https://www.python.org/) scripts mainly for visualization
  - r : [R](https://cran.r-project.org/) scripts mainly for statistical analysis
- results: analysis results
- prerun_results: a snapshots of results that I ran, including both r and python parts.

### Codes organization

- For the main analysis, the codes are stored in a jupyter notebook or a R Markdown document with ample annotations.  
- To make the code easier to follow, all basic codes that are not directly relevant to the analysis logics are inside helper.py or helper.r

### Prerequisites

- If you are running on CodeOcean, you can ignore this part
- Required python libraries is listed under [code/python/requirements.txt](code/python/requirements.txt)
- Required R libraries is listed under [code/r/requirements.txt](code/r/requirements.txt)

## License

- Distributed under the MIT License. See `LICENSE` for more information.

## Contact

- Jason Lo - lcmjlo@gmail.com

- Project Link: [https://github.com/JasonLo/SSR_modeling_RD](https://github.com/JasonLo/SSR_modeling_RD)