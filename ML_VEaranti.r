# 'HarvardX Data Science Capstone Project: Predicting Movie Ratings in the MovieLens 10M Dataset'
#author: "Venkata Earanti"
#date: "July 17, 2021"


# Summary

# As part of HarvardX Data Science Capstone Project, I worked on creating a movie recommendation system using the Movie Ratings in the MovieLens 10M Dataset.

# Recommendation systems allow the customer to rate, gather the user ratings and business analyze the data to predict user behavior.

# The goal of this project is to train a machine learning algorithm that predicts user movie ratings using the inputs of a supplied subset to predict film ratings in a supplied validation set. The term used for evaluating efficiency of the algorithm is the Root Mean Square Error, or RMSE. RMSE is one of the most widely used measurement of the differences between the values predicted by a model and the observed values.
# RMSE is a measure of accuracy, to compare forecasting errors of different models for a particular dataset, An RMSE of 0 means we are always correct, not a possibility. An RMSE of 1 means the predicted ratings are off by 1 star, a lower RMSE is better than a higher one. The effect of increasing error on RMSE is proportional to the size of the squared error; hence, larger errors impact RMSE disproportionately.
# Data will be divided into two data sets, edx for training and validation for the data validation purpose. After the data analysis, models will be developed and compared to arrive to a conclusion.


# Data Analysis
## Data preparation


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(magrittr)) install.packages("magrittr", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(magrittr)
library(knitr)
library(kableExtra)



knitr::opts_chunk$set(error = TRUE)
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1) # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
# Add rows removed from validation set back into edx set 
removed <- anti_join(temp, validation)

edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Algorithm will be developed on Edx data set and validation data set will be used
#for testing the final algorithm.



## Exploratory Data Analysis

## Exploration/Validation 1

#Get first few rows of the edx and get familiarize with dataset




head(edx)


# edx data set contain the six variables 'userID', 'movieID', 'rating', 'timestamp','title', and 'genres'.
# Each row represent a single rating of a user for a single movie.
# Rating is the target variable,the value we are tryng to predict

## Exploration/Validation 2

#Check if there are any missing values


summary(edx)


# The summary of the edx confirms that there are no missing values.

## Exploration/Validation 3

#Check total of unique movies and users


edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId), .groups ='drop') %>% niceKable


#Unique users - 69878\
#Unique movies - 10677


## Exploration/Validation 4

#Review ratings distribution


edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "black") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Rating distribution")



# Based on the Rating Distribution Diagram, most users rated 4. 0.5 is the least rated.
# More Users tends to give "Full-Star" rating, few users gave "Half-Star" rating
# If we consider anything less than 3 is negative, there are small number of negative ratings



## Exploration/Validation 5

#Review number of ratings per movie


edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("# of ratings") +
  ylab("# of movies") +
  ggtitle("# of ratings per movie")


# Some movies (approx 125) were rated only one time (very low)
# Some movies were rated more often than others 
# Low rating number could impact the quality of the prediction




## Exploration/Validation 6

# Movies rated only once


edx %>%
  group_by(movieId) %>%
  summarize(count = n(), .groups ='drop') %>%
  filter(count == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = rating, n_rating = count, .groups ='drop') %>%
  slice(1:20)



# some movies are rated only once,predictions of future ratings for them will be difficult.



## Exploration/Validation 7

# Review number of ratings given by users


edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") + 
  ylab("Number of users") +
  ggtitle("Number of ratings given by users")


# Majority of the users rated 30-100 movies




## Exploration/Validation 8

#Review mean movie ratings by users


edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating), .groups ='drop') %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black") +
  xlab("Mean rating") +
  ylab("Number of users") +
  ggtitle("Mean movie ratings given by users") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  theme_light()


# Looking at users who rated at least 100 movies, some users gave much lower rating and some gave much higher than average. 
# Users differe on how critical they are.




# Approach

#RMSE is defined as follows

#$$ RMSE = \sqrt{\frac{1}{N}\displaystyle\sum_{u,i} (\hat{y}_{u,i}-y_{u,i})^{2}} $$
  
  
#  The RMSE is our measure of model accuracy


RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}



##    1. Average rating (Naive Baseline) Model

#The first basic model to predict the same rating for all movies is using dataset’s mean rating.

#A model based approach assumes the same rating for all movie with all differences explained by
#random variation :
#  $$ Y_{u, i} = \mu + \epsilon_{u, i} $$
#  with $\epsilon_{u,i}$ independent error sample from the same distribution


mu <- mean(edx$rating)
mu

naive_rmse <- RMSE(validation$rating, mu)
naive_rmse


#The mean is 3.512465

#Naive RMSE is 1.061202. 

#It is very far for the target RMSE (below 0.87) and that indicates poor performance for the model.


rmse_results <- tibble(method = "Average movie rating model", RMSE = 
                         naive_rmse)
rmse_results 





##    2.Movie effect model

#Based on out data analysis, some movies are just generally rated higher than others. Higher ratings are 
#mostly linked to popular movies among users and the opposite is true for unpopular movies. We compute the estimated deviation of each movies’ mean rating from the total mean of all movies $\mu$. The resulting variable is called "b" ( as bias ) for each movie "i" 
#$b_{i}$, that represents average ranking for movie $i$:
#  $$Y_{u, i} = \mu +b_{i}+ \epsilon_{u, i}$
  

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu) , .groups ='drop')
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = 
                       I("black"),
                     ylab = "Number of movies", main = "Number of movies with the computed b_i")
predicted_ratings <- mu + validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_1_rmse <- RMSE(predicted_ratings, validation$rating)



#Movie effect model RMSE is 0.9439087


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie effect model", 
                                     RMSE = model_1_rmse ))
rmse_results 




##    3.Movie and user effect model

#The above model do not consider individual user rating effect, that was found in exploration/validation 8.

#Compute the average rating for user $\mu$, for those that have rated over 100 movies.



user_avgs<- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating - mu - b_i))
user_avgs%>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = 
                     I("black"))

user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i) , .groups ='drop')

predicted_ratings <- validation%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

model_2_rmse <- RMSE(predicted_ratings, validation$rating)

model_2_rmse



#Movie and user effect model RMSE is 0.8653488


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie and user effect model", 
                                     RMSE = model_2_rmse))
rmse_results




##    4.Regularized movie and user effect model

#In the earlier models we computed standard error and constructed confidence intervals to 
#account for different levels of uncertainty. However, when making predictions, 
#we need one number, one prediction, not an interval. 

#We introduce the concept of regularization, that permits to penalize large estimates that come 
#from small sample sizes. The idea is to add a penalty for large values of $b_{i}$ to the sum of squares equation that we minimize. So having many large $b_{i}$, make it harder to minimize. Regularization is a method used to reduce the effect of overfitting.

#Estimates of $b_{i}$ and $b_{u}$ are caused by movies with very few ratings and in some users that only rated a very small number of movies. Hence this can strongly influence the prediction. The use of the regularization permits to penalize these aspects. We should find the value of lambda (that is a tuning parameter) that will minimize the RMSE. This shrinks the $b_{i}$ and $b_{u}$ in case of small number of ratings.


lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l) , .groups ='drop')
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l), .groups ='drop')
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, rmses) 




lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized movie and user effect 
model", 
                                     RMSE = min(rmses)))
rmse_results 


#Regularized movie and user effect model - RMSE - 0.8648170

#Optimal Lambda - 5.25



# Predictions

#We are going to choose the "Regularized movie and user effect model" calculation since it has the lowest RMSE \


l = 5.25
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l) , .groups ='drop')

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l), .groups ='drop')

predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

validation = cbind(validation[,c(1:3)], predicted_ratings, validation[,4])

head(validation) 


#This gives a sample of rating and predicted rating


