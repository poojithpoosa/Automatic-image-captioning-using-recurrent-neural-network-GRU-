# Automatic-image-captioning-using-recurrent-neural-network

From the internet, news stories, document diagrams, and commercials that we see every day, we gather a great number of pictures. It is up to the viewer to make sense of the visuals in these sources. Despite the lack of a description, people can make sense of most visuals without them. However, if people want automated picture captions, robots must be able to decipher some type of image caption. Image captioning is a major AI research field that deals with the interpretation of images and the description of those images in a foreign language. Understanding an image involves more than just finding and identifying items; it also includes figuring out the scene, the location, the attributes of the objects, and how they interact. Both syntactic and semantic knowledge are necessary to produce well-formed sentences. For the project, I have implemented an image captioning model using spatial adaptive attention, which is focused on refining the image features and helps the model understand the semantics of the captions. The model was trained on the flicker 8k dataset and had BLEU1 scores of.81 and 0.73 for blue2 when tested on the testing set and 0.8145 and 0.775 on the training set.

##	Aim of the project
It is possible to automatically generate a caption for a picture via the use of image captioning. It's becoming more and more popular as a new field of study. So that we may fulfil the aim of image captioning, semantic information from pictures must be gathered and articulated in natural languages. Image captioning is a difficult job since it connects the CV and NLP fields. The solution to this issue has been presented in a number of ways. I this project I am going to use GRU with spatial adaptive attention mechanism.

## Research question
Does using attention mechanism using GRU improve performance of automatic image captioning?

# working of the project

![image](https://user-images.githubusercontent.com/61981756/199013083-f259d439-1753-4a88-8538-1a374eabe01b.png)

For the project I have used flicker 8k dataset. The flicker 8k dataset consists of 8092 images with 4 to 5 captions for each caption. The dataset is almost of 2GB and contains images and its captions. We can download the dataset from Kaggle:  
https://www.kaggle.com/datasets/ming666/flicker8k-dataset 

# Proposed network:

![image](https://user-images.githubusercontent.com/61981756/199013306-4ad30a3c-7e2a-440e-993d-b2981b1baf04.png)

# Libraries used:
![image](https://user-images.githubusercontent.com/61981756/199013471-bd0225ff-1f11-4d2a-90c1-124aae4454bf.png)

# Project Design

![image](https://user-images.githubusercontent.com/61981756/199013529-4bc3d999-14cd-444a-b2ae-ffae638c7af4.png)


Results:

![image](https://user-images.githubusercontent.com/61981756/199013706-54ce9685-3ff8-4e13-b64b-ea1dd4c063bb.png)

![image](https://user-images.githubusercontent.com/61981756/199013724-1adeb3c7-925b-4c31-b7d2-6d51776ccb0f.png)
