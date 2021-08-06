Rscript -e "rmarkdown::render(input = 'r/figure3_and_lme.rmd', output_dir = '../results', clean = TRUE)"
jupyter nbconvert --to html --execute python/figure1.ipynb    