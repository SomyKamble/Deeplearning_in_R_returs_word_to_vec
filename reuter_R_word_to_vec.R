library(keras)
library(tensorflow)
library(tidyverse)

c(c(train_data,train_labels),c(test_data,test_labels)) %<-% dataset_reuters(num_words = 10000)

length(train_data)


length(test_data)


train_data[[1]]

dataset_reuters_word_index() %>%
  unlist()%>%
  sort%>%
  names()->word_index

library(purrr)

train_data[[1]]%>%
  map(~ ifelse(.x>=3,word_index[.x-3],"?"))%>%
  as_vector()%>%
  cat()

#onehotencoding

Vectorize_sequences <- function(sequences,dimension=10000){
  
  results <- matrix(0,nrow = length(sequences),ncol = dimension)

  for (i in 1:length(sequences)) {
    results[i,sequences[[i]]]<- 1
  }
    results
  }



train_data_vec<-Vectorize_sequences(train_data)
test_data_vec<-Vectorize_sequences(test_data)

train_example<-sort(unique(train_data[[1]]))
train_example

train_data_vec[1,1:100]

str(train_labels)

sort(unique(train_labels))

train_data_vec[1,1:100]


library(ggplot2)
train_labels %>%
  plyr::count() %>%
  ggplot(aes(x,freq))+
  geom_col()


#the real one hot encoding 


train_labels_vec <- to_categorical(train_labels)


test_labels_vec <- to_categorical(test_labels)

str(train_labels_vec)

str(test_labels_vec)


#creation of the deep learning architecture 

network <- keras_model_sequential()%>%
  layer_dense(units = 64,activation = 'relu',input_shape = c(10000))%>%
  layer_dense(units = 64,activation = 'relu')%>%
  layer_dense(units = 46,activation = 'softmax')
  
summary(network)

#complie
network%>%compile(
  optimizer='rmsprop',
  loss='categorical_crossentropy',
  metric=c('accuracy')
)

#validate

index<- 1:1000
val_data_vec<-train_data_vec[index,]
train_data_vec<- train_data_vec[-index,]

val_labels_vec<-train_labels_vec[index,]
train_labels_vec<- train_labels_vec[-index,]


#training our model for 20 epochs

history <- network %>% fit(
  train_data_vec,
  train_labels_vec,
  epochs=20,
  batch_size=512,
  validation_data= list(val_data_vec,val_labels_vec)
)

plot(history)



#recreating our model for 9 epochs since val_acc is going into overfitting 
network <- keras_model_sequential()%>%
  layer_dense(units = 64,activation = 'relu',input_shape = c(10000))%>%
  layer_dense(units = 64,activation = 'relu')%>%
  layer_dense(units = 46,activation = 'softmax')

summary(network)

#complie
network%>%compile(
  optimizer='rmsprop',
  loss='categorical_crossentropy',
  metric=c('accuracy')
)


history <- network %>% fit(
  train_data_vec,
  train_labels_vec,
  epochs=9,
  batch_size=512,
  validation_data= list(val_data_vec,val_labels_vec)
)

plot(history)


#evaluat it on test data_set

metric<- network %>% evaluate(test_data_vec,test_labels_vec)


metric$accuracy
metric$loss

#predict 

network %>% predict_classes(test_data_vec[1:10,])

#prediction for all 
predictions <- network %>% predict_classes(test_data_vec)
actual<- unlist(test_labels)
total_misses<-sum(prediction!=actual)

#confusion matrix

suppressPackageStartupMessages(library(tidyverse))

library(dplyr)

data.frame(target=actual,
           prediction=predictions)%>%
  filter(target !=prediction) %>%
  group_by(target,prediction) %>%
  count() %>%
  ungroup() %>%
  mutate(perc=n/nrow(.)*100) %>%
  filter(n > 1)%>%
  ggplot(aes(target,prediction,size=n))+
  geom_point(shape= 15,col= "#9F92C6")+
  scale_x_continuous("acutal target", breaks = 0:45)+
  scale_y_continuous("prediction", breaks = 0:45)+
  scale_size_area(breaks=c(2,5,10,15),max_size = 5)+
  coord_fixed()+
  ggtitle(paste(total_misses,"mismatches"))+
  theme_classic()+
  theme(rect=element_blank(),
           axis.line = element_blank(),
           axis.text = element_text(colour = "black"))