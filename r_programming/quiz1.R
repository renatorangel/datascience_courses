file <- "rprog-data-quiz1_data.zip"

zipFileInfo <- unzip(file, list=TRUE)

data <- read.csv(unz(file, as.character(zipFileInfo$Name)))

x <- c(4, "a", TRUE)
class(x)
x


x <- c(1,3, 5) 
y <- c(3, 2, 10)
class(cbind(x, y))

x <- 1:4 
y <- 2
x + y
colnames(data)
head(data, n = 2)

summary(data)

nrow(data)
tail(x = data, n = 2)

data[47,1]

sum(is.na(data$Ozone))

mean(data$Ozone, na.rm = TRUE)



mean(data$Solar.R[data$Ozone > 31 & data$Temp > 90], na.rm = TRUE)

mean(data$Temp[data$Month == 6], na.rm = TRUE)

max(data$Ozone[data$Month == 5], na.rm = TRUE)
