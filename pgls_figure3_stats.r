library(ape)
library(caper)

listofmodels  <-c(xminsink~SQRTWL-1, xbestglide~SQRTWL-1, MeanRadius~WL-1, MeanRadius~MeanHorizontalSpeed-1)

for ( mymodel in listofmodels ){
  
  output_file <- paste(all.vars(mymodel)[1], "_", all.vars(mymodel)[2], "_no_intercept_with_lamda.csv", sep="")
  df <- read.csv("species_traits_V2.csv")
  
  listoftrees <- ape::read.tree("pruned.tre", keep.multi=TRUE)
  ###############################################################################
  
  results <- data.frame()
  
  for (i in 1:length(listoftrees)) {
    print(i)
    
    #currenttree <- listoftrees[[i]]
    # Rescale branch lengths using Grafen's method
    currenttree <- compute.brlen(listoftrees[[i]], method = "Grafen")
    
    print(listoftrees[[i]]$edge.length)
    
    dataTree <- comparative.data(
      phy=currenttree,             # the phylogeny
      data=df,             # the data frame
      names.col="Species", # names in the data frame that
      vcv=FALSE,           # whether to include a variance covariance array
      warn.dropped=TRUE   # whether to warn when data 
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
    p_value <- summary(pgls1)$f.p.value
    residual_std_error <-  summary(pgls1)$sigma
    f_value <- f_stat[1]
    #p_value_fstat <- 1 - pf(f_value, df1 = f_stat[2], df2 = f_stat[3])  # Calculating p-value
    lambda_value <- summary(pgls1)$param["lambda"]
    
    # Combine all metrics
    result_row <- c(
      Tree = i,               # Tree index
      Slope = coef_pgls[1],
      Std_error_w = Std_error ,
      t_value_w = t_value,
      p_of_t_w = p_of_t,
      rse = residual_std_error,
      R_squared = r_squared,
      Lambda = lambda_value
      #F_statistic = f_stat,
      #Slope = coef_pgls[2]
    )
    
    # Add to results
    results <- rbind(results, result_row)
    #print(results)
  }
  
  write.csv(results, output_file)
  
}
