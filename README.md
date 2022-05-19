# Sentiment_Analysis_Emails

BERT stands for Bidirectional Encoder Representations from Transformers. It's intended to use both left and right context conditioning to pre-train deep bidirectional representations from unlabeled text. As a result, with just one extra output layer, the pre-trained BERT model may be fine-tuned to generate state-of-the-art models for a wide range of NLP applications.

In other words, the BERT scans the entire sequence of words at once, unlike directional models that read the text input sequentially (left-to-right). As a result, it is classified as bidirectional. This feature enables the model to learn the context of a word from its surroundings (to the left and right of the word).

Fistly, since we are about to use tensorflow, we have to import some libraries ---
  - %tensorflow_version 2.6
  - import tensorflow as tf
  - import tensorflow_hub as hub
  - !pip install tensorflow-text
  - import tensorflow_text as text

**NOTE** - *The versions are very important while imorting tensorflow*

**Please refer to documentation https://www.tensorflow.org/text/tutorials/classify_text_with_bert and stackoverflow for issues**

*Here, I have used version-2.6* 

To find about the emails , we first extrxated the mails dataset from https://www.kaggle.com/code/sid321axn/sms-spam-classifier-naive-bayes-ml-algo/data
After the dataset was extracted we find it very uncertain and unbalanced.
We can see the spam and ham valued datasets respectively

![unbal](https://user-images.githubusercontent.com/76419241/169275987-209bc3a2-33ff-4370-b97a-1239cd8948c4.png)


So, we balance it by taking sample of ham upto size of spam

Now they are balanced
![bal](https://user-images.githubusercontent.com/76419241/169276405-b80624a4-96e0-47c1-ad37-363eb43fe84b.png)


After balancing, we can just make this categorical dataset by changing ham values to 0 and spam value as 1

 After all the dataset work, we'll just divide our dataset in training and testing dataset using *train_test_split*
 
 Now, we are ready to import the BERT MODEL
 We have to import the-
  - preprocessor from "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
  - encoder from "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

To get function embeedings, we define a function

![func](https://user-images.githubusercontent.com/76419241/169283017-28c37369-f005-4f6c-be09-fe1ecc206426.png)

   
Now. we are ready to apply our training data on this function to get embeedings of sentences
It even includes applying *DROPOUT AND DENSE* **NEURAL NETWORK LAYERS**
More about them from here **https://towardsdatascience.com/machine-learning-part-20-dropout-keras-layers-explained-8c9f6dc4c9ab**

Finally, we will create model as **model = tf.keras.Model(inputs=[text_input], outputs=[l])**

The model summary looks like-
![modlesum](https://user-images.githubusercontent.com/76419241/169275893-d5fcb361-b5ea-41fb-8d19-59784a324b40.png)

After compiling model, we'll fit it with respective epochs on training data, **epochs can be adjusted accordingly**
Then, we perfom evaluation on test data.

After evaluation and taking confusion matrix, we get the heatmap as

![visua](https://user-images.githubusercontent.com/76419241/169283977-d5dd9822-fddd-432a-a384-9a5b79002282.png)

   
We can further get our model's report based on confusion matrix

![accur](https://user-images.githubusercontent.com/76419241/169284215-e9617563-b9d5-471f-ac53-6c8befdd7681.png)

And, finally we can check the perfomace of our model on random datasets

![probab](https://user-images.githubusercontent.com/76419241/169284366-b431eb25-3f3a-4991-a271-20e438bd2b0a.png)

**To be noted, here we can see the probabilities of our new sentences of how close are they to be considered as spam.
Further we can refine this model, to exavctly predict it in as 1 or 0(spam or not) by simply applying (>0.5) condition.**


