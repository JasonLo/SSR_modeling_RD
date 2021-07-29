library(effectsize)
library(insight)
library(lme4)
library(lmerTest)
library(sjPlot)
library(rlang)


num_format <- function(num, digits = 2, leading_zero = T) {
    # Readable number formatting
    fnum <- round(num, digits) %>% format(nsmall = digits)
    if (leading_zero == F)
        fnum <- sub('0.', '.', fnum)
    return(fnum)
}

my_format_summary <- function(summary) {
    # For formatting final table for viewing
    formatted_summary <- summary %>%
        mutate(
            p_eta2 = num_format(partial_eta2),
            beta = num_format(Std_Coefficient),
            t = num_format(!!sym("t value")),
            df = num_format(df, 0),
            ci = paste0('[', num_format(CI_low),
                        ', ', num_format(CI_high),
                        ']'),
            p = num_format(!!sym("Pr(>|t|)"), 3, F)
        ) %>%
        select(Parameter, p_eta2, beta, t, df, ci, p, messages, model)
    return(formatted_summary)
}

my_lmer <- function(f, data, summarize_last_row = F, boxplot = F, save_file = F) {
    if (boxplot)
        boxplot(as.formula(f), data = data)
    
    # Create lme formula with random effect from model ID
    model <- lmer(as.formula(paste(f, "+ (1 | ID)")), data)
    
    # Check whether interaction term exist to print appropriate plot
    ifelse(grepl("\\*", f),
           plot_model(model, type = "int") %>% print(),
           plot_model(model, type = "pred") %>% print())

    # Make summary
    model_summary <- summary(model)
    
    # Extract results
    pars <- model_summary$coefficients
    spars <- standardize_parameters(model, method = "basic")
    
    # Create low level summary table
    low_level_summary <- cbind(pars, spars)
    
    ## Extract partial eta-square from anova
    model_anova <- anova(model)
    eta2s <- F_to_eta2(model_anova$`F value`, 
                       model_anova$NumDF, 
                       model_anova$DenDF)
    low_level_summary["partial_eta2"] <- c(NA, eta2s$Eta2_partial)
    
    ## Export model level label and warnings
    low_level_summary["data_name"] <- deparse(substitute(data))
    low_level_summary["model"] <- f
    msg <- model_summary$optinfo$conv$lme4$messages
    low_level_summary["messages"] <- ifelse(is.null(msg), 'good', msg)
    
    if (summarize_last_row) low_level_summary <- low_level_summary[nrow(pars),]
    
    fsum <- my_format_summary(low_level_summary)
    print(fsum)
    
    if (save_file != F){
        write.csv(fsum, save_file)
    }
    
    
    list(model = model, 
         summary = low_level_summary,
         formatted_summary = fsum)
}
