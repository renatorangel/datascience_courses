file <- 'activity.zip'

zipFileInfo <- unzip(file, list=TRUE)

data <- read.csv(unz(file, as.character(zipFileInfo$Name)))
data$date <- as.Date(as.character(data$date) , format = "%Y-%m-%d")

# aggregate

data.sum <- aggregate(x = data[c("steps")],
                      FUN = sum,
                      by = list(date = data$date))


data.average <- aggregate(x = data[c("steps")],
                          FUN = mean,
                          by = list(interval = data$interval),
                          na.rm = TRUE)

hist(data.sum$steps, breaks = 61)
summary(data.average)

mean(data.sum$steps, na.rm = TRUE)

median(data.sum$steps, na.rm = TRUE)

plot(x = data.average$interval, y = data.average$steps,type = "l")

max(data.average$steps)


data.average$interval[which.max( data.average$steps )]

sum(is.na(data$steps))


newData <- data


missing <- is.na(newData$steps)
newData$steps[missing] <- as.numeric(data.average$steps)
sum(is.na(newData$steps))


newData.sum <- aggregate(x = newData[c("steps")],
                         FUN = sum,
                         by = list(date = newData$date))


hist(newData.sum$steps, breaks = 61)


mean(newData.sum$steps)

median(newData.sum$steps)

newData.sum$steps - data.sum$steps

data.sum$steps[is.na(data.sum$steps)] <- 0


#create factor weekday and weekend
newData$day <- weekdays(newData$date)

newData$day[newData$day == "Saturday" | newData$day ==  "Sunday"] <- "weekend"
newData$day[newData$day != "weekend"] <- "weekday"
newData$day <- as.factor(newData$day)

levels(newData$day)


# split data frame
list_week <- split(newData, newData$day)

head(list_week[[2]])

# aggregate the data frame by average of number steps taken
newData.weekday.average <- aggregate(x = list_week[[1]][c("steps")],
                          FUN = mean,
                          by = list(interval = list_week[[1]]$interval))

newData.weekend.average <- aggregate(x = list_week[[2]][c("steps")],
                                     FUN = mean,
                                     by = list(interval = list_week[[2]]$interval))

# plot the average steps X interval
par(mfrow=c(1,2))

plot(x = newData.weekday.average$interval, y = newData.weekday.average$steps, type = "l")

plot(x = newData.weekend.average$interval, y = newData.weekend.average$steps, type = "l")
