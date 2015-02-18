complete <- function(directory, id = 1:332) {
    ## 'directory' is a character vector of length 1 indicating
    ## the location of the CSV files

    ## 'id' is an integer vector indicating the monitor ID numbers
    ## to be used

    ## Return a data frame of the form:
    ## id nobs
    ## 1  117
    ## 2  1041
    ## ...
    ## where 'id' is the monitor ID number and 'nobs' is the
    ## number of complete cases

    file_list <- list.files(directory)
#     df <- data.frame()
#     colnames(df) <- c("id", "nobs")
    #create the data frame
    for (file in file_list){
        file_name <- substr(file, 1, nchar(file)-4) # take off the last 4 digits

        if(is.element(as.integer(file_name), id)){
            dataset <- read.csv(file.path(directory, file), header=TRUE)

            if (!exists("df1")){

                asdf <- matrix(nrow = 1,ncol = 2)
                asdf[1,1] <-  as.integer(file_name)
                asdf[1,2] <- nrow(dataset[rowSums(is.na(dataset)) == 0,])
                df1 <- as.data.frame(asdf)

                col_headings <- c("id","nobs")
                names(df1) <- col_headings
            }
            else{
                row <- c(as.integer(file_name), nrow(dataset[rowSums(is.na(dataset)) == 0,]))
                df1 <- rbind(df1, row)
                rm(row)
            }
        }
    }
    if(head(id,n=1) > tail(id, n=1)){
        df1 <- df1[nrow(df1):1, ]
    }
    return(df1)
}