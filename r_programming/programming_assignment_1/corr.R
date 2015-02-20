
getCorrelation <- function(directory1, file_number) {
    ## 'directory' is a character vector of length 1 indicating
    ## the location of the CSV files

    ## 'id' is an integer vector indicating the monitor ID numbers
    ## to be used

    ## Return a data frame of the pollutant
    ## (ignoring NA values)

    file_list <- list.files(directory1)

    #create the data frame
    for (file in file_list){
        file_name <- substr(file, 1, nchar(file)-4) # take off the last 4 digits

        if(is.element(as.integer(file_name), file_number)){
            dataset <- read.csv(file.path(directory1, file), header=TRUE)
        }
    }
    newDataset <- dataset[rowSums(is.na(dataset)) == 0,]

    return(cor(x = newDataset$nitrate, y = newDataset$sulfate ))
}

corr <- function(directory, threshold = 0) {
    ## 'directory' is a character vector of length 1 indicating
    ## the location of the CSV files

    ## 'threshold' is a numeric vector of length 1 indicating the
    ## number of completely observed observations (on all
    ## variables) required to compute the correlation between
    ## nitrate and sulfate; the default is 0

    ## Return a numeric vector of correlations

    totalComplete <- complete(directory)

    thresholdComplete <- totalComplete[totalComplete$nobs > threshold,1]


    correlation = lapply(thresholdComplete, getCorrelation,
              directory1 = directory)

    return(rapply(correlation, c))

}