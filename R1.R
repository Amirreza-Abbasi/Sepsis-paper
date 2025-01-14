getwd()
setwd("D:/Papers/Abedi/Sepsis/WGCNA")

library(readxl)
library(WGCNA)
options(stringsAsFactors = FALSE)
femData <- read_excel("Controlw.xlsx")

dim(femData)
names(femData)


datExpr1 = as.data.frame(t(femData[, -c(1:1)]))
View(datExpr1)
dim(datExpr1)
datExpr2= as.matrix(datExpr1)
datExpr2=as.numeric(datExpr2)
dim(datExpr1)
datExpr2= matrix(datExpr2, 12, 2156)


colnames(datExpr2) = femData$Genes

View(datExpr2)
rownames(datExpr2) = names(femData)[-c(1:1)]
dim(datExpr2)

View(datExpr2)


gsg = goodSamplesGenes(datExpr2, verbose = 3);
gsg$allOK

if (!gsg$allOK)
{
  # Optionally, print the gene and sample names that were removed:
  if (sum(!gsg$goodGenes)>0)
    printFlush(paste("Removing genes:", paste(names(datExpr2)[!gsg$goodGenes], collapse = ", ")));
  if (sum(!gsg$goodSamples)>0)
    printFlush(paste("Removing samples:", paste(rownames(datExpr2)[!gsg$goodSamples], collapse = ", ")));
  # Remove the offending genes and samples from the data:
  datExpr2 = datExpr2[gsg$goodSamples, gsg$goodGenes]
}





library(flashClust)

sampleTree = flashClust(dist(datExpr2), method = "average");
sizeGrWindow(12,9)
pdf(file = "NormalsampleClusteringwithoutouts.pdf", width = 12, height = 9);
par(cex = 0.9);
par(mar = c(0,6,2,0))
plot(sampleTree, main = "Sample clustering to detect outliers", sub="", xlab="", cex.lab = 1.5,
     cex.axis = 1.5, cex.main = 2)

dev.off()
datExprNormal=datExpr2;



save(datExprNormal, file = "Normal.RData")



################################################   Tumor tissue  ###########################

library(WGCNA)
options(stringsAsFactors = FALSE)
femData <- read_excel("Sepsis.xlsx")

dim(femData)
names(femData)


datExpr1 = as.data.frame(t(femData[, -c(1:1)]))

datExpr2= as.matrix(datExpr1)
datExpr2=as.numeric(datExpr2)
dim(datExpr1)
datExpr2= matrix(datExpr2, 10, 2156)


colnames(datExpr2) = femData$Genes

View(datExpr2)
rownames(datExpr2) = names(femData)[-c(1:1)]
dim(datExpr2)

View(datExpr2)



gsg = goodSamplesGenes(datExpr2, verbose = 3);
gsg$allOK

if (!gsg$allOK)
{
  # Optionally, print the gene and sample names that were removed:
  if (sum(!gsg$goodGenes)>0)
    printFlush(paste("Removing genes:", paste(names(datExpr2)[!gsg$goodGenes], collapse = ", ")));
  if (sum(!gsg$goodSamples)>0)
    printFlush(paste("Removing samples:", paste(rownames(datExpr2)[!gsg$goodSamples], collapse = ", ")));
  # Remove the offending genes and samples from the data:
  datExpr2 = datExpr2[gsg$goodSamples, gsg$goodGenes]
}





library(flashClust)

sampleTree = flashClust(dist(datExpr1), method = "average");
sizeGrWindow(12,9)
pdf(file = "SepsissampleClustering.pdf", width = 12, height = 9);
par(cex = 0.9);
par(mar = c(0,6,2,0))
plot(sampleTree, main = "Sample clustering to detect outliers", sub="", xlab="", cex.lab = 1.5,
     cex.axis = 1.5, cex.main = 2)

dev.off()
datExprTumor=datExpr2;



save(datExprTumor, file = "Sepsis.RData")
