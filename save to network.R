getwd();
setwd("D:/Papers/Abedi/Sepsis/WGCNA")

# Load the WGCNA package
library(WGCNA)
# The following setting is important, do not omit.
options(stringsAsFactors = FALSE);
enableWGCNAThreads()


# Load the expression and trait data saved in the first part
lnames = load(file = "Normal.RData");
#The variable lnames contains the names of loaded variables.
lnames
# Load network data saved in the second part.
lnames = load(file = "Normalmodules.RData");
lnames

datExprNormal = as.data.frame(datExprNormal)

###############################################
###########Exporting to Cytoscape##############
###############################################
# Recalculate topological overlap if needed
TOM = TOMsimilarityFromExpr(datExprNormal, power = 6);
colnames(TOM) <- colnames(datExprNormal)
rownames(TOM) <- colnames(datExprNormal)
TOM1= as.data.frame(TOM)
TOM = as.matrix(TOM1)
# Select modulesTOM# Select modules

#modules = c("darkmagenta","darkorange","floralwhite","greenyellow","lightcyan","magenta", "orange","plum1","sienna3","yellowgreen","gold","paleturquoise","skyblue3");

modules = "orangered4"
# Select module probes
probes = names(datExprNormal)
inModule = is.finite(match(moduleColors1, modules));
modProbes = probes[inModule];


# Select the corresponding Topological Overlap
modTOM = TOM[inModule, inModule];



dimnames(modTOM) = list(modProbes, modProbes)
# Export the network into edge and node list files Cytoscape can read
cyt = exportNetworkToCytoscape(modTOM,
                               edgeFile = paste("CytoscapeInput-edges-", paste(modules, collapse="-"), ".txt", sep=""),
                               nodeFile = paste("CytoscapeInput-nodes-", paste(modules, collapse="-"), ".txt", sep=""),
                               weighted = TRUE,
                               threshold = 0.02,
                               nodeAttr = moduleColors1[inModule]);

