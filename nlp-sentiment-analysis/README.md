# Sentiment Analysis, NLP
Project to recognize hate in tweets, natural language processing.

# Collaborators
- Sebastián Volti
- Maria Eugenia Miranda
- Agustina Sierra 
- Agustina Salmanton

# OverView
Starting from a dataset of tweets, we proceed to preprocess them.<br/>
Then, using word-embeddings and different models of neural network, we can predict if a single tweet, contains "hate".

## Execution 
For the execution you have to install the requirements by means of the command:

`pip3 install -r requirements.txt`

For the execution of the main file, the requested format is respected:

`python3 es_odio.py <data_path> test_file1.csv ... test_fileN.csv`

For the execution of the model with the file *test.csv* received as a resource:

`python3 es_odio.py ./resources`

To execute the cross validation, the following command must be executed:

`python3 crossValidation.py`


## Files
 The following files are included:
 - **es_odio.py**: main file, where a model is created, trained and evaluated
 - **model.py**: file where the implemented models are located
 - **data.py**: file where data is loaded and cleaned
 - **constants.py**: file where you have general constants
 - **crossValidation.py**: execution of the configuration of the hyperparameters by means of cross validation.
 
## Data cleaning
Libraries:
- pandas v1.0.3
- nltk v3.5

The **data.py** file contains the different functions that were used to clean the data.
Initially, all the data was loaded into a DataFrame with two columns, one for the tweets and the other for their classification.
Each of these tweets was then subjected to a pipeline of preprocessing functions:

- **removen(text)**: remove chars '\n'
- **removeUrl(text)**: Remove change the url for the string URL
- **removeHashtags(text)**: Remove hashtag symbols (#)
- **toLowerCase(text)**: Convert each character from uppercase to lowercase
- **removeUsers(text)**: Remove the user mentions (@user) and change them to the string USER
- **removeRepetitions(text)**: Remove repeating letters from words with repeating letters and leave a occurrence (**i.e**: "holaaa" => "hola")
- **removePunctuation(text)**: Remove any non-alphanumeric characters
- **removeLaughter(text)**: Modify some meanings of laughter for  "jaja" (**i.e**: "jajajajajaja"=> "jaja", "jeje" => "jaja", "jajsj" => "jaja")

## Selected models:
Libraries:
- numpy v1.18.2
- keras v2.4.3
- tensorflow v2.3.1 

In the **model.py** file, the Model class is defined, which creates the model and contains the different functions that are of interest to apply to the models. The models have training and evaluation methods, these are **eval() **and** train() ** respectively.
To initialize the models we use the provided embbeding. Initially we touch the tweets and generate an embedings matrix, where for each word of the volcabulary found we place its vector of embedings as a column in the matrix. Then that matrix is ​​used as the first layer in our neural networks.

### Generated models:
Simple neural network model
Neural network LSTM1.
Neural network LSTM2.
Bidireccional neural network .
Convolucional neural network.
In the lab **CrossValidation.ipynb** there is a detail of them.

## Cross Validation
To find the best model with its best parameters, cross validation was implemented, it is found in the **CrossValidation.py** file and in the **CrossValidation.ipynb** file it is found in detail how this procedure was performed.
The parameters that were adjusted here are:
- epochs
- neurons
- dropout
- batchs
- model_type.

It was concluded that the best model is the Convolutional Model.


In turn, the best parameters were:
|parameter|value|
|---|---|
|dropout|0.1|
|epochs|10|
|neurons|64|
|batches|64|

