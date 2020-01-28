# Gender_Classification

![image](https://user-images.githubusercontent.com/33619392/73286651-94acb080-4200-11ea-818b-015c14ef7398.png)
# Introduction:
In this project we took a data set and analyst it to know the gender of the speaker voice  
# The Dataset:
The efficiency of the all self-labeled algorithms was evaluated using the Voice gender dataset and the Deterring dataset.
Voice gender dataset took from https://www.kaggle.com. This database
was created to identify a voice as male or female, based upon acoustic properties of the voice and speech.
It consists of 3168 recorded voice samples, collected from male and female speakers. 
Each record from the dataset has contains 20 features that every feature represent the voice data 


![image](https://user-images.githubusercontent.com/33619392/73287294-bbb7b200-4201-11ea-8aca-180241ac34d8.png)

And the label represents the gender of that data.

We convert the data from the csv file to class that contains the features and the label.
After that we build our first model.

![image](https://user-images.githubusercontent.com/33619392/73287443-f3265e80-4201-11ea-8ba0-5503644c1ec9.png)

# First attempt

For our first attempt we used logistic regression in order to classify between male and female and it showed good results after 6000 iteration. 

![image](https://user-images.githubusercontent.com/33619392/73286009-94f87c00-41ff-11ea-8961-07e12be0edc2.png)
      ![image](https://user-images.githubusercontent.com/33619392/73286058-a5a8f200-41ff-11ea-8afa-c45802c6a095.png)
# Multilayer perceptron (MLP)

We tried to add multilayer perceptron that we used two Adding Hidden Layer by adding another weight and biases and multipy the two weight and biases, for 2500 iterations.

![image](https://user-images.githubusercontent.com/33619392/73287619-2ff25580-4202-11ea-94cd-739691ef1a48.png)


![image](https://user-images.githubusercontent.com/33619392/73287656-3d0f4480-4202-11ea-8b56-819d7a326992.png)
# Recurrent Neural Networks (RNN)  Long Short Term Memory (LSTM)
Reasons we chose this model: RNN can benefit sequential data like frequencies. 
We used RNN LSTM because of it’s important when dealing with time series data. 
Remembering those factors over time is crucial for to continuation of the learning and weights correctness and prevent vanishing gradient that happen with big batches of data like voice recording.

###### Results
After using the neural network RNN and LSTM with 50 iteration we got 98%+ accuracy

![image](https://user-images.githubusercontent.com/33619392/73287776-71830080-4202-11ea-86b4-2fced4afeb07.png)





