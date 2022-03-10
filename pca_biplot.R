# ilham - 馬希迪
# M10118033
data <- read.csv('Boston.csv')
head(data)
summary(data)

# see if there is any na value in our dataset
colSums(is.na(data))

# make a pca model
data.pca <- prcomp(data, center = TRUE, scale. =TRUE)
## note 
# - we can use prcomp() function to make pca model in r language
# - I think this function automaticaly pick the numper of pc and
#   do scaling to the data.

# we can see the explained variance by this line of code 
summary(data.pca)

# Screeplot
layout(matrix(1:2, ncol=2))
screeplot(data.pca)
screeplot(data.pca, type="lines")
## note
# - we can see actually we can keep the 1,2,3 PCs
#   because the variance or the eigenvalues more than 1.0

# library for plotting biplot
library(ggbiplot)

# for plotting biplot in r
ggbiplot(data.pca)
