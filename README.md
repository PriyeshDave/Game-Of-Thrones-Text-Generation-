# Game-Of-Thrones-Text-Generation
![Python](https://img.shields.io/badge/Python-3.9-blueviolet)
![Machine Learning](https://img.shields.io/badge/-Machine%20Learning-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)

This is a classic Text Generation problem, using scriptures of famous books, Game of Thrones in this case. 

![image](https://user-images.githubusercontent.com/81012989/170167145-c95b1dad-4bdb-4ff5-90b7-acf5db8aeaa7.png)
## Introduction
Developed an intelligent application that predicts the next n_words given a sequence of words based on a Markov's N-Grams assumption (predicting next word given preceeding 50 words in this case).

## 🧾 Dataset: 
The dataset is based on scriptures of famous books, **Game of Thrones**. The book has around 572 pages, however the data is trained on first 210 pages. 

### :bar_chart: Exploratory Data Analysis:
* Exploratory Data Analysis is the first step of understanding your data and acquiring domain knowledge. 

### :hourglass: Data Preprocessing:
#### 1.) Dataset preparation:
* The dataset is prepared from this raw text data by using **Markov's assumption** method where the entire text data is converted to lines of length 51 where the first 50 words acts as the independent features while the last word is kept as dependent feature.
* Below is the code sinippet for data preparation where the words highlighted in blue at [15] and [16] are the input sequences while the output word is highlighted in yellow. 

<img width="926" alt="image" src="https://user-images.githubusercontent.com/81012989/170168276-2b3aea79-fe3a-45da-8037-2eff6206e052.png">

#### 1.) Data Cleaning:
* The given data was cleaned & preprocessed by first removing the unwanted special characters and numbers from text.
* The stop words in this case are not removed as it would change the contextual meaning of the text.
* The preprocessed data is saved in **Cleaned-Text.txt** file.

#### 2.) Vectorization:
* I have used **Keras Tokenizer library** to tokenize the input sequences of length 50 each and assigned each word in the sequence with their respective index based on the vocabulary size of 7479.
* The output words are one hot encoded using **to_categorical** method from **tensorflow.keras.utils**. 

### ⚙ Model Training:
The model is trained using **Sequential Model** with a 2 **LSTM layers** with 100 units each.
* First I added the embedding layer which embedds the input sequences by representing each word with a vector of length 50 based on our vocabulary.
* Next, I used Keras LSTM layer.
* For the first LSTM layer, I have added 100 units that represent the dimensionality of outer space. The return_sequences parameter is set to true for returning the last output in output. 
* Similarly, I added the other **LSTM** layer with 100 units again.
* The next step is to add the dense layer with 100 hidden neurons followed by other dense layer with number of neurons equal our vocabulary size. I have used softmax activation function. This will give an output vector with length of our vocabulary size where each vector unit represents the probability of occuring of the respective word.
* At last, my model is compiled with the help of **adam optimizer**. 
* The error is computed using **categorical_crossentropy** and metric used is **accuracy**.
* Finally, the model is fit using 450 epochs with a batch size of 512.
![model](https://user-images.githubusercontent.com/81012989/170167203-5c361f7d-4498-469e-af4f-32ec2fd4af98.png)


* My trained model predicts the next words with an accuracy of 77.23%
* As per the problem statement I used **accuracy** as the evaluation metric for my model.
* Model: **'Game-Of-Thrones-Text-Generation-\Model\Word2Vec-Model-450-512.h5'** <br>
  https://drive.google.com/file/d/1-DN4FK0GcI-0RTFspWMe5U09ZMeMS_-5/view?usp=sharing
  
  ## Web Application :computer: :earth_americas: :
Built a web application using **Streamlit** and deployed on **streamlit-cloud**.

<img width="960" alt="image" src="https://user-images.githubusercontent.com/81012989/170180431-f5e35b97-b1d8-458d-ba65-3a27f913e70d.png">

https://share.streamlit.io/priyeshdave/game-of-thrones-text-generation-/main/app.py
