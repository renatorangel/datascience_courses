pollutantmean <- function(directory, pollutant, id = 1:332) {
    ## 'directory' is a character vector of length 1 indicating
    ## the location of the CSV files

    ## 'pollutant' is a character vector of length 1 indicating
    ## the name of the pollutant for which we will calculate the
    ## mean; either "sulfate" or "nitrate".

    ## 'id' is an integer vector indicating the monitor ID numbers
    ## to be used

    ## Return the mean of the pollutant across all monitors list
    ## in the 'id' vector (ignoring NA values)

    file_list <- list.files(directory)

    #create the data frame
    for (file in file_list){
        file_name <- substr(file, 1, nchar(file)-4) # take off the last 4 digits

        if(is.element(as.integer(file_name), id)){
            # if the merged dataset doesn't exist, create it
            if (!exists("dataset")){
                dataset <- read.csv(file.path(directory, file), header=TRUE)
            }
            else{
                temp_dataset <- read.csv(file.path(directory, file), header=TRUE)
                dataset<-rbind(dataset, temp_dataset)
                rm(temp_dataset)
            }
        }

    }

    #
    if(pollutant == "sulfate"){
        return(mean(dataset$sulfate, na.rm = TRUE))
    }
    else{
        return(mean(dataset$nitrate, na.rm = TRUE))
    }
}
