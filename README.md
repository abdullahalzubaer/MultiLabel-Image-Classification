# Multi label image classification.



## Data preprocessing

In this project I am goint to use movie poster dataset for multi label image classificaiton by adopting deep neural network. The dataset comes in two folder, in one folder we have
the movie posters and in another folder we have the metadata for each poster. The metadata in the poster has Id (correspnding to the movie poster) and other attributes including Genres.

To label the movie posters we have to use the ID in the metadata file and load the corresponding movie poster into a pandas dataframe.
In this way we will make sure that the labels of the corresponding movie posters are matching with the metadata.
The reason is, the metadata sequence and the movie poster sequence are not same.


## Data properties

We have 8052 movie poster, with Id as the name of the images. We have metadata for 7255 movie
posters. Which means we wont be able to use all the movie posters. For this project can use
7255 movie posters along with their metadata. The metadata has 27 columns. For computational
purpose we are not going to use 7255 movie poster, we will use frst 3000 movie poster.

## Training and validation

Train on 2400 samples, validate on 600 samples. 

## Testing

By using a random movie poster (after preprocessing) and predicting the top ten labels using the trained model.


Model prediction below:
```
Comedy (0.728)
Drama (0.504)
Romance (0.29)
Adventure (0.106)
Crime (0.105)
Family (0.0754)
Action (0.0734)
Fantasy (0.0488)
Music (0.0404)
Thriller (0.0301)
```




---

## TODO
* [ ] Add plots for training and validation accuracy and loss
* [ ] Detail description of loss function the last layer's activation function
* [ ] Experiment with the network architecture 
* [ ] Increase training samples and epochs (post the results)

---

Dataset: https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/




Reference:
-  "Python for Microscopists": https://github.com/bnsreenu/python_for_microscopists/blob/master/142-multi_label_classification.py
- Wei-Ta Chu and Hung-Jui Guo, “Movie Genre Classification based on Poster Images with Deep Neural Networks,” Proceedings of International Workshop on Multimodal Understanding of Social, Affective and Subjective Attributes, pp. 39-45, 2017. (in conjunction with ACM Multimedia 2017)
