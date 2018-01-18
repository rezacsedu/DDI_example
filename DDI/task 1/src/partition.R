#Author: Sachin Joshi
#Task 1
#Subtask 6
#Partitioning into Training(80%) and Testing(20%) Data Set

#Reading the Sentence.csv file into the sen data frame in the R environment
sen <- read.csv("Sentence.csv")
View(sen) #Viewing the sen data frame 

Training <- data.frame(Sentence=numeric(0)) #Creating a training data frame with Sentence as s single column 
Testing <- data.frame(Sentence=numeric(0)) #Creating a testing data frame with Sentence as s single column

k <- 1 #Counter for the Training data frame
p <- 1 #Counter for the Testing data frame

#Random partitioning of sentences into the Training and Testing data set: Training(80%), Testing(20%) 
for(i in 1:94) #Training
{
  Training[k,1] <- c(toString(sen$Sentence[i]))
  k <- k+1
}
for(i in 95:495) #Testing
{
  Testing[p,1] <- c(toString(sen$Sentence[i]))
  p <- p+1
}
for(i in 496:1199) #Training
{
  Training[k,1] <- c(toString(sen$Sentence[i]))
  k <- k+1
}
for(i in 1200:1750) #Testing
{
  Testing[p,1] <- c(toString(sen$Sentence[i]))
  p <- p+1
}
for(i in 1751:2299) #Training
{
  Training[k,1] <- c(toString(sen$Sentence[i]))
  k <- k+1
}
for(i in 2300:2354) #Testing
{
  Testing[p,1] <- c(toString(sen$Sentence[i]))
  p <- p+1
}
for(i in 2355:3169) #Training
{
  Training[k,1] <- c(toString(sen$Sentence[i]))
  k <- k+1 
}
for(i in 3170:3269) #Testing
{
  Testing[p,1] <- c(toString(sen$Sentence[i]))
  p <- p+1 
}
for(i in 3270:5532) #Training
{
  Training[k,1] <- c(toString(sen$Sentence[i]))
  k <- k+1 
}


View(Training) #Viewing the Training data frame
View(Testing) #Viewing the Testing data frame

#Writing the Training and Testing data frame into csv files
library(foreign)
write.csv(Training, "D:/Lehigh/SEM 2/Text Mining/Project 2/semeval_task9_train/Train/Training.csv")#Writing the Training data frame into csv files
write.csv(Testing, "D:/Lehigh/SEM 2/Text Mining/Project 2/semeval_task9_train/Train/Testing.csv")#Writing the Testing data frame into csv files

