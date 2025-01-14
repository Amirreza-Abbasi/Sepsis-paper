getwd()

setwd("D:/Papers/Abedi/Sepsis/WGCNA")

library(WGCNA)
# The following setting is important, do not omit.
options(stringsAsFactors = FALSE);



lnames1 = load(file = "Normal.RData");
lnames1


lnames2 = load(file = "Sepsis.RData");
lnames2




lnamesnormal=load(file = "Normalmodules.RData")
lnamesnormal





setLabels = c("NormalN", "TomurN");
multiExpr = list( NormalN = list(data = datExprNormal),TumorN = list(data = datExprTumor));
multiColor = list(NormalN = moduleColors1);




system.time( {
  mp = modulePreservation(multiExpr, multiColor,
                          referenceNetworks = 1,
                          nPermutations =50,
                          randomSeed = 1,
                          quickCor = 0,
                          verbose = 3)
} );
# Save the results
save(mp, file = "modulePreservation.RData");

