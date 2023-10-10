# In this machine learning project, we will develop a Language Translator using a many-to-many encoder-decoder sequence model. We will train our model using LSTM which will convert English text to hinglish where English will be input text and Hinglish will be the target text. For this, we will be using English-Hindi dataset and  the dataset provided as the test data set.

1. Import Libraries and initialize variables.
Firstly we will import all libraries which have been shared in the prerequisites section. Also we need to initialize variables globally which can be used throughout our functions.

2. Parse the dataset file
We will traverse the dataset file and extract all the input and target texts. For this, we will be using the first 25,000 rows of our dataset for the training and testing part. It can be changed as per requirements. We will lower case all characters ,remove quotes remove all numbers from the text and remove extra spaces.

3. One Hot Encoding (Vectorization)
Models cannot work directly on the categorical data. For this, we require one hot encoding process. One-hot encoding deals with the data in binary format so we encode the categorical data in binary format.
One-hot means that we can only make an index of data 1 (true) if it is present in the vector or else 0 (false). So every data has its unique representation in vector format.
For example, if we have an array of data like : [“python”,”java”,”c++”] then the one hot encoding representation of this array will be :
[ [ 1 , 0 , 0 ]
[ 0 , 1 , 0 ]
[ 0 , 0 , 1 ] ]
So in our project after separating characters from input and target text we will use a one-hot encoding process. We will fit characters and transform the texts accordingly. So if the character from input text is present in the sets of characters then it will put 1 and 0 otherwise. Our encoder input data, decoder input data, and decoder target data will be a 3D array where encoder input data will have shape (number of pairs, max length of English text, number of English text characters), decoder input data will have shape (number of pairs, max length of hindi texts, number of hindi  characters). Decoder target data will be same as decoder input data but it will be one timestep ahead as it will not include the start character i.e. ‘\t’ of our target sentence.


4. Build the training model
In this language translation project, we will be using LSTM to train our machine learning model to translater language, so let’s see what is LSTM:

LSTM (Long Short Term Memory) network: LSTM is a type of RNN (Recurrent Neural Network) that solves scenarios where RNN is failed.
Long-Term Dependency: In RNN, networks have the data of previous output in memory for a short period of time because of this they are unaware about the actual context of the sentence over a long period of time. This raised the issue of long-term dependency.
Vanishing Gradient: While training our models, in order to get the best output we have to minimize the loss i.e. errors after every time step. This can be achieved by propagating backward and calculating the gradients, that is loss with respect to weights applied to every vector at different time steps. We repeat this process until we get an optimal set of weights for which the error is minimum. After reaching at some time step gradient value becomes so less that it approximates to zero or gradient vanishes. After reaching that limit, the network stops training. This leads to the problem of vanishing gradient.
These are some issues which are resolved by the LSTM networks. Instead of a single neural network layer, LSTM has three gates along with hidden and cell states. We will use following example to understand the basic functionality of LSTM:
“John wants to know how Language Translators work so he started studying a technique known as Deep Learning. His friend Jim, on the other hand, is interested in Self-Driving Cars and is learning about a technique known as Reinforcement Learning.”
Cell Memory state ( ct ): Cell state is actually what makes LSTM a unique network. Cell state holds the memory for over a long period of time. Data can be removed or added in cell state depending upon the layer requirements.
Hidden state ( ht ): hidden state is basically output of the previous block. We decide what to do with the memory looking at the previous hidden state output and current input. And also we don’t want output after every timestep until we reach the last input of our sequence.
Forget Gate ( ft ): Forget Gate is used to check what data we want to neglect away from the cell state. This is done using a sigmoid layer. This Gate looks at hidden output from previous time steps and current input, after that it outputs number 0 which means neglect the data, or 1 means keep the data.
So from our example, in the first half of the sentence, we say “John” is interested in “Language Translator”. So in order to answer “Deep Learning”, we can frame a question like “Which technique is used to develop a Language Translator?” For this, we require “Language Translator” throughout every time step. So what about “John”? In this sentence the context is about technique, so we don’t require “John” therefore we will forget or remove the data of “John” using forget Gate ( ft ).
Input Gate: We want to check what new information we are going to store in the cell memory state ( ct ). So data will pass through the sigmoid function which decides which values to update ( it ). Next a tanh function creates a vector of new candidates ( čt ) for our state.
Now our new data is ready and we want to update cell state with our new data. In order to do that we will multiply the old value of the cell state with the forget gate which will remove the data and then add the combined value of the tanh function which contains the candidate value ( čt ) and the input vector ( it ) for the new data.
From our example we know “Language translator” is required so we will store that in the cell state which can be accessible for the layers after every time step. And likewise we also want “technique” so we will store it in cell state.
Now our cell state contains “Language Translator” as well as “technique”.
Output Gate: We want to pass the output to the next layer. As the output is dependent upon the cell state, we will be using sigmoid which decides what parts of the cell state we are going to output. Then we will apply tanh and multiply them with sigmoid value of output ( ot ) so that we only output values that are required.
So we will pass cell state and hidden state value output to another block of the LSTM layer as an input.
For the second part of our example now his friend “Jim” is interested in “Self-Driving Cars” we can frame a question like “Which technique is used for Self-Driving Cars?”.
As you can see the context remains same i.e. “technique” but now we require a technique for “Self-Driving Car” so we will forget the previous stored data i.e. “Language Translator” from cell state using the forget gate and we will add “Self-Driving Cars” in the cell state.
And this cell and hidden output are passed to the next block of LSTM layers where the same steps repeat till we reach the right prediction
Encoder: For the language translation machine learning model we will be creating keras Input objects whose shape will be equal to the total number of input characters. We will use RNN’s LSTM model and only the return state will be True because we only want value from hidden and cell state so we will discard encoder output and only keep the states.

Decoder: In decoder, our Input object shape will be equal to the total number of target characters. The LSTM model with the return state and return sequence will be True as we need a model to return full output sequence(text) as well as states. We will be using a softmax activation function and also a Dense layer for our output.



5. Train the model
To train the model we will fit ‘(encoder input and decoder input)’ which will turn into (‘decoder target data’) using ‘Adam’ optimizer with a validation split of 0.2 and provide an epoch of 200 in a batch size of 64. 
Also, we can see the summary or visualize our model. The summary contains 1) Layers that we have used for our model 2) Output Shape which shows dimensions or shapes of our layers 3) The number of parameters for every layer is the total number of output size, i.e. number of neurons associated with the total number of input weights and one weight of connection with bias so basically
N (number of parameters) = (number of neurons) * (number of inputs + 1)
So the number of parameters for our dense layer (output layer) will be number of decoder characters present in our dataset i.e 67 associated with the number of input weights i.e. 256 and one weight of connection with bias, therefore our N will be
N = 67 * ( 256 + 1) = 17219
After plotting our model we can see that our first input layer has a shape of the total number of input characters (English characters) i.e 47, which we will then pass to the first LSTM layer that has the input shape same as the previous layer and the latent(hidden) dimension of 256. Model initializes the second LSTM layer with the output of the first LSTM layer along with input from the decoder layer which has a shape of total number of target characters(hindi characters) i.e 67. Finally we will pass the output of the second LSTM layer to the dense layer which is our final output layer that has the shape of target characters.
