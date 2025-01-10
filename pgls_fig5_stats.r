library(ape)
library(caper)

listofmodels  <-c(b_observed~x_best_glide_mean, b_observed~a_observed)

for ( mymodel in listofmodels ){
  
  output_file <- paste("Penny",all.vars(mymodel)[1], "_", all.vars(mymodel)[2], "_intercept.csv", sep="")
  df <- read.csv("updated_results_30secs.csv")
  listoftrees <- ape::read.tree("pruned.tre", keep.multi=TRUE)
  ###############################################################################
  
  results <- data.frame()
  
  for (i in 1:length(listoftrees)) {
    #currenttree <- listoftrees[[i]]
    # Rescale branch lengths using Grafen's method
    currenttree <- compute.brlen(listoftrees[[i]], method = "Grafen")
    
    dataTree <- comparative.data(
      phy=currenttree,             # the phylogeny
      data=df,             # the data frame
      names.col="Species", # names in the data frame that
      vcv=FALSE,           # whether to include a variance covariance array
      warn.dropped=TRUE    # whether to warn when data 
      # or tips are dropped
    )
    # PGLS function
    pgls1 <- pgls(
      mymodel, # model formula
      data=dataTree,             # the "comparative.data" object
      lambda='ML'
      # the branch length transformations are 
      # optimised using maximum likelihood
    )
    
    
    #plot(pgls1)
    coef_pgls <- coef(pgls1)
    
    summary = summary(pgls1)
    # Extracting R-squared
    r_squared <- summary(pgls1)$r.squared
    
    Std_error = summary(pgls1)$coefficient[2]
    t_value = summary(pgls1)$coefficient[3]
    p_of_t = summary(pgls1)$coefficient[4]
    
    # Extracting F-statistic and p-value
    f_stat <- summary(pgls1)$fstatistic
    #p_value <- summary(pgls1)$f.p.value
    residual_std_error <-  summary(pgls1)$sigma
    f_value <- f_stat[1]
    #p_value_fstat <- 1 - pf(f_value, df1 = f_stat[2], df2 = f_stat[3])  # Calculating p-value
    p_value <- 1 - pf(f_stat[1], f_stat[2], f_stat[3])
    
    lambda_value <- summary(pgls1)$param["lambda"]
    # Combine all metrics
    result_row <- c(
      Tree = i,               # Tree index
      intercept = coef_pgls[1],
      Slope = coef_pgls[2],
      R_squared = r_squared,
      F_statistic = f_value,
      p_va = p_value,
      Lambda = lambda_value
      #Slope = coef_pgls[2]
    )
    
    # Add to results
    results <- rbind(results, result_row)
    print(results)
  }
  
  write.csv(results, output_file)
  
}
