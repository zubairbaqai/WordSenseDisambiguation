from lxml import etree
import ast
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.engine import Layer
import keras.layers as layers
from keras import backend as K
import numpy as np
from keras.models import load_model
import tensorflow_hub as hub
import tensorflow as tf
import pickle



#elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True )


max_len=40
batch_size = 16


elmo_model = hub.Module("./elmo_embeddings/9bb74bc86f9caffc8c47dd7b33ec4bb354d9602d")

#REPLACE THIS BEFORE TRAINING
mappings = open('/home/baqai/Desktop/public_homework_3/WSD_Training_Corpora/SemCor/semcor.gold.key.txt')
Dataset="/home/baqai/Desktop/public_homework_3/WSD_Training_Corpora/SemCor/semcor.data.xml"



##The Lemmas is the key , and values are lemma+ All the instances of it 
SimilarLemmas={}

#The Instance is mapped to Wordnet ID's
InstancetoLabel={}


Dictionary={}
InvertseDictionary={}

Input_tokens=[]
Output_tokens=[]



def maincode(input_path):

    #### Labels of Input Sentences  STEP1
    for x in mappings:
        x=x.replace("   "," ")
        x=x.replace("\t"," ")
        x=x.replace("\n","")
        x = x.split(" ")
        InstancetoLabel[x[0]]=x[1]


   #Step 2
    context=etree.iterparse(input_path, tag="text") #Parsing the Dataset , based on <text> tag and getting its child
    Dictionary["<PAD>"]=0
    Dictionary["<UNK>"]=1
    for action, text in context:

        #elem.tag prints <texts>
        for sentence in text:
            #Prints <sentence>
            InputSentence=[]
            OutputSentence=[]
            for childs in sentence:

                attributes=ast.literal_eval(str(childs.attrib)) # The Dictionary is made from all the Attributes of the child of <text> . 
                lema=attributes['lemma']

                InputSentence.append(lema) #InputSentence list , that contains all tokens of 1 sentence, and Input_tokens is list of list, which has list of Sentence's Tokens
                if(Dictionary.get(lema)==None):#Adding the Lemmas to dictionary, eg Dictionary["Fox"]=10 , where 10 is the ID of the word Fox
                    Dictionary[lema]=len(Dictionary)




                if(childs.tag=="wf"):
                	#for Each Input to model, the output could be a worditself or Sense(Instance) , <wf> are the once which are just pure lemmasZ
                    OutputSentence.append(lema)

                    #SimilarLemmas is a dictionary , where the key is the Pure lemma "Stick" and the values are list of its lemma and senses ["Stick","Stick%56:67:00",etc ,etc]
                    #We Store SimilarLemmas for prediction , because for each Instance input we must have a Sense not a lemma, So even if network gives lemma as most probly sense, We need to Substitude to next Highest probability Instance
                    if(SimilarLemmas.get(lema)==None):
                        SimilarLemmas[lema]=[lema]
                    else:
                        if(lema not in SimilarLemmas[lema]):
                            SimilarLemmas[lema].append(lema)
                    

                if(childs.tag=="instance"):
                    instancceID=attributes["id"]

                    #We get the InstanceID and we Substitude this with the Sense key which is provided to us for mapping
                    TLabel=InstancetoLabel[instancceID]
                    
                    OutputSentence.append(TLabel)
                    #We Add Instances also in Dictionary . as they need to have a ID when they are being output by Model
                    if(Dictionary.get(TLabel)==None):
                    
                        Dictionary[TLabel]=len(Dictionary)


                    if(SimilarLemmas.get(lema)==None):
                        SimilarLemmas[lema]=[TLabel]
                    else:
                        if(TLabel not in SimilarLemmas[lema]):
                            SimilarLemmas[lema].append(TLabel)
                    
            Input_tokens.append(InputSentence)
            Output_tokens.append(OutputSentence)
        text.clear()


    #We Need to have Inverse Dict as well. to get the Words back from the ID's that are output from the Network
    InvertseDictionary = {v: k for k, v in Dictionary.items()}



    #We Are Saving these three Strctures which will be used in Prediction as well
    with open('../resources/Dictionary.pkl', 'wb') as f:
                            pickle.dump(Dictionary, f, pickle.HIGHEST_PROTOCOL)


    with open('../resources/InvertseDictionary.pkl', 'wb') as f:
                            pickle.dump(InvertseDictionary, f, pickle.HIGHEST_PROTOCOL)


    with open('../resources/SimilarLemmas.pkl', 'wb') as f:
                            pickle.dump(SimilarLemmas, f, pickle.HIGHEST_PROTOCOL)



     ############################## We have a dictionary, Inverse Dictionary , Input Sentences as words , output sentences as words , We will convert the words to Integers ##########


    for sentences in range(len(Input_tokens)):
        for words in range(len(Input_tokens[sentences])):
            Input_tokens[sentences][words]=Dictionary[Input_tokens[sentences][words]]
            Output_tokens[sentences][words]=Dictionary[Output_tokens[sentences][words]]



    #Getting length of dictionary which will be needed to specify Number of Softmax Neurons
    n_words=len(Dictionary)
    X = pad_sequences(maxlen=max_len, sequences=Input_tokens, padding="post")
    Y = pad_sequences(maxlen=max_len, sequences=Output_tokens, padding="post")


    #We Used Elmo Embeddings rather than normal embeddings, as elmo embeddings give embedding based on context
    input_text = layers.Input(shape=(max_len,), dtype="string")
    embedding = layers.Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
    model = layers.Dropout(0.1)(embedding)
    model = layers.Bidirectional(layers.LSTM(units=512, return_sequences=True, recurrent_dropout=0.1))(model)
    out = layers.TimeDistributed(layers.Dense(n_words, activation="softmax"))(model)  # softmax output layer


    model = Model(input_text, out)


    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics = [custom_sparse_categorical_accuracy])



    #After We have passed the data through pad_Sequnces and have the paddings and truncations, we convert the IDs back to string, as ELMO takes the Strings directly rather than ID's
    X=X.tolist()
    Y=Y.tolist()
    for sentences in range(len(X)):
        for words in range(len(X[sentences])):
            X[sentences][words]=InvertseDictionary[(X[sentences][words])]
            #Output_tokens[sentences][words]=InvertseDictionary[Output_tokens[sentences][words]]





    #### MAKING the inputs to be a multiple of batch size . As Elmo's Documentation asks that we specify what is the size of batch we will be inputting.

    if(len(X)%batch_size != 0):
        lastitemX=X[0]
        lsatitemY=Y[0]
        while(len(X)%batch_size != 0):
   
            X.append(lastitemX)
            Y.append(lsatitemY)






    Y=np.asarray(Y)
    X=np.asarray(X)


    Y=np.expand_dims(Y, axis=2)

    history = model.fit(X, Y, batch_size=batch_size, epochs=10, validation_split=0, verbose=1)


    #Unlike HW2, We have to Store the Weights of model rather than whole Model . as the Embeddings of Elmo, Will not be Saved along with Save_model as they are custom once.
    model.save_weights('my_model_weights.h5')






#We Mention that Input will be string, and the Length of Each sequence will be 40, and there will be 16 sequences(batch) . based on our settings.
def ElmoEmbedding(x):   
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, "string")),
                            "sequence_len": tf.constant(batch_size*[max_len])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]



#Since we are using loss="sparse_categorical_crossentropy" , when trying with metrics=[accuracy] we get error , and after resarching they mentioned that we have to use a custom metrics , as define below
def custom_sparse_categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())




maincode(Dataset)


