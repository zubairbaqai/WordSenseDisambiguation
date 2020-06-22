from lxml import etree
import ast
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.engine import Layer
import keras.layers as layers
from keras import backend as K
from nltk.corpus import wordnet as wn
from difflib import SequenceMatcher
import numpy as np
from keras.models import load_model
import tensorflow_hub as hub
import tensorflow as tf
import pickle
from collections import OrderedDict




max_len=40
elmo_model = hub.Module("./elmo_embeddings/9bb74bc86f9caffc8c47dd7b33ec4bb354d9602d", trainable=True )
batch_size = 16


SimilarLemmas={}
Dictionary={}
InvertseDictionary={}
instances=[]
wordnet2babelnet={}
babelnet2others={}

def predict_babelnet(input_path : str, output_path : str, resources_path : str) -> None:

    Backend(input_path,output_path,resources_path,"babelnet")

  
    pass


def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:

    Backend(input_path,output_path,resources_path,"domains")

    pass


def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    Backend(input_path,output_path,resources_path,"lexicographer")
    pass



def Backend(input_path, output_path, resources_path ,OutputType):

    mappings = open(resources_path+'/babelnet2wordnet.tsv')
    for x in mappings:
        x=x.replace("   "," ")
        x=x.replace("\t"," ")
        x=x.replace("\n","")
        x = x.split(" ")
        wordnet2babelnet[x[1]]=x[0]

    if(OutputType=="domains"):
        mappings = open(resources_path+'/babelnet2wndomains.tsv')
        for x in mappings:
            x=x.replace("   "," ")
            x=x.replace("\t"," ")
            x=x.replace("\n","")
            x = x.split(" ")
            babelnet2others[x[0]]=x[1]


    if(OutputType=="lexicographer"):
        mappings = open(resources_path+'/babelnet2lexnames.tsv')
        for x in mappings:
            x=x.replace("   "," ")
            x=x.replace("\t"," ")
            x=x.replace("\n","")
            x = x.split(" ")
            babelnet2others[x[0]]=x[1]








    

    #LOADING The 3 resources we need for this project.
    with open(resources_path+'/Dictionary.pkl', 'rb') as f:
                    Dictionary = pickle.load(f)

    with open(resources_path+'/SimilarLemmas.pkl', 'rb') as f:
                    SimilarLemmas = pickle.load(f)


    with open(resources_path+'/InvertseDictionary.pkl', 'rb') as f:
                    InvertseDictionary = pickle.load(f)

    outfile=open(output_path,"w")



    
    context=etree.iterparse(input_path, tag="text")

    Input_tokens=[]
    for action, text in context:

        #elem.tag prints <texts>
        for sentence in text:
            #Prints <sentence>


            InputSentence=[]  #We keep this List to store Words of single Sentences .
            index=0 #In Sentences there are words which have senses and they are called Instances, We Get the locations of Instances in sentence we use index iterator ,to know where instance lie in the sentence
            newInstances={}#Once we have the index of instance , we store Each Sentences in List of list.


            counter=0 #After My Implementation , I saw that due to the fact that we have set max len to 35, if we have sentences greater than 35 we lost lemmas and instances, so we broke down the sentences for the max len

            for childs in sentence:
                counter=counter+1

                attributes=ast.literal_eval(str(childs.attrib))
                lema=attributes['lemma']

                InputSentence.append(lema)

                if(childs.tag=="instance"):
                    instancceID=attributes["id"]
                    newInstances[index]=instancceID

                index=index+1

                if(counter==max_len):  #If a sentence had more than maxlength words, then we are storing the words upto 35 as a diffenent sentence, and add the remaining as the second sentence.
                    Input_tokens.append(InputSentence)
                    instances.append(newInstances)

                    InputSentence=[]
                    index=0
                    counter=0
                    newInstances={}


            Input_tokens.append(InputSentence)
            instances.append(newInstances)




    ############################## We have a dictionary, Inverse Dictionary , Input Sentences as words , output sentences as words , We will convert the words to Integers ##########



    OriginalWord={} #If Our Dictionary dosnt have the a new word , and we replace with <UNK> , then at last stage when we are assigning a Sense ID as our requirement, we need to know the original Word.

    for sentences in range(len(Input_tokens)):
        for words in range(len(Input_tokens[sentences])):
            if(Dictionary.get(Input_tokens[sentences][words])!=None):
                Input_tokens[sentences][words]=Dictionary[Input_tokens[sentences][words]]   #We Are converting Words to integers from our dictionary we generated in Training step, because we need to pass Number form List into pad_sequence
            else:
                if(OriginalWord.get(sentences)==None): # Since OriginalWord is A Dictionary of Dictionary , so we can store the sentence number and word number as keys and the Original Word which is being replaced by UNK as the value
                    OriginalWord[sentences]={}

                OriginalWord[sentences][words]=Input_tokens[sentences][words]

                Input_tokens[sentences][words]=1 #After we have stored the Original value of the word , we replace by 1, which is <UNK> so we can feed into the network


    n_words=len(Dictionary) #To Remake our Model , We must specify the Number of neurons on Softmax layer , So n_words are the number of words in our dictionary , Includes both ,<wf> and <instance>



    X = pad_sequences(maxlen=max_len, sequences=Input_tokens, padding="post")


    input_text = layers.Input(shape=(max_len,), dtype="string")
    embedding = layers.Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
    model = layers.Dropout(0.1)(embedding)
    model = layers.Bidirectional(layers.LSTM(units=512, return_sequences=True, recurrent_dropout=0.1))(model)
    out = layers.TimeDistributed(layers.Dense(n_words, activation="softmax"))(model)  # softmax output layer


    model = Model(input_text, out)


    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    model.load_weights(resources_path+'/my_model_weights.h5')


    X=X.tolist()
    for sentences in range(len(X)):
        for words in range(len(X[sentences])):
            X[sentences][words]=InvertseDictionary[(X[sentences][words])]

    #Now That we have Adding Paddings , and Truncated oversized sentences , we bring back to Word form , Because I applied Elmo Embedding , And they require The inputs to be in form of sentences or tokens
  




    #Above All , now that we are using Elmo , the Architecture it has set itself for is , that it must take input of length 35 and of the batch size we mentioned . So we Add Dummy Entries of Sentence, while keeping original length so we can remove self added Entries
    originalLength=len(X)
    if(len(X)%batch_size != 0):
        lastitemX=X[0]
        while(len(X)%batch_size != 0):
            X.append(lastitemX)
    input=np.asarray(X)







    SentenceCounter=0



    for i in range(int(len(input)/batch_size)):

        print("Progress: "+str(SentenceCounter)+"/"+str(originalLength))
        #This May look Weird to not Feed the network at once , But it ran out of memory pretty soon , as the Embeddings along with huge Softmax probability take a lot of memory ... So we Feed the network in batches , do calculations on what to give output on file .
        p=model.predict(input[i*batch_size:(i+1)*batch_size],batch_size=batch_size)
        #We take 32 sentences, and get 32 outputs in a batch

        for i in range(len(p)):

            if(SentenceCounter<originalLength):
                #Remember the Dummy Data we added to make it compatible with Batch size requirement , We terminate the Inputs after original Sentence length, as rest are few of the dummy oncce . 

                for WordIndex in range(len(input[SentenceCounter])):

                    WordDictIndex=[]
                    #For instance output , We could Get output as general Lemma , rather than the instance, so we Use SimilarLemma dictionary, which we made in training . It has all possible senses of a lemma , that we saw during training
                    #For the output We want the Senses of just that specific lemma , so we restrict the argmax to the senses of the lemma we feeded .

                    #CurrentWord give us the Lemma that we have on the current index, and on this CurrentWord we get all associated Instancces(Senses) as explained above
                    CurentWord=input[SentenceCounter][WordIndex]

                    if(CurentWord=="<PAD>"):
                        continue
                    elif(CurentWord=="<UNK>"):
                        WordDictIndex.append(1)
                    else:
                        AllLemmas=SimilarLemmas[CurentWord] ###this will be a List of Possible lemmas and instances of the input words
                        for Lem in AllLemmas:
                            WordDictIndex.append(Dictionary[Lem])

                    InstanceID=instances[SentenceCounter].get(WordIndex)
                    if(InstanceID!=None):


                        if(len(WordDictIndex)>1):

                            for wording in WordDictIndex:

                                wordingInv=InvertseDictionary[wording]

                                #Since the current Index which we are outputting is An instance, so we must assign a Sense rather than the lemma , even if its top prediction is the lemma , so we remove the lemma from our Allemas dictionary of that word
                                if len(wordingInv.split("%"))==1:
                                    WordDictIndex.remove(wording)

                        #Here We Have The Integer of the Most probable Sense of the Instance that was feeded on current index , We Get the Associated Word to this Integer which is the Key to get the word
                        FinalDictID=np.argmax(p[i][WordIndex][WordDictIndex])


                        #IF during our training , We didnt train at all for this Instance, we will get <UNK> , We then get Original Word from our OriginalWord Dict which we created at starting , And from the Original Word we Find the MFS of it
                        if(InvertseDictionary[WordDictIndex[FinalDictID]] =="<UNK>"):
                            OrgWord=OriginalWord[SentenceCounter][WordIndex]
                            for syn in wn.synsets(OrgWord):
                                synsets=syn.name().split('.')[0]
                                if(OrgWord==synsets):
                                    NewFoundID=syn.lemmas()[0]._key
                                    NewFoundID=synsetfromsensekey(NewFoundID)
                                    BabelnetID=wordnet2babelnet[NewFoundID]

                                    if(OutputType!="babelnet"): 
                                        NewFoundID=babelnet2others.get(BabelnetID)
                                        if(NewFoundID==None):
                                            print("NOT FOUND :"+BabelnetID+" Assigning bn:00000001n as Default")
                                            NewFoundID=babelnet2others["bn:00000001n"]
                                    

                            

                            outfile.write(InstanceID+" "+NewFoundID+"\n")

                        #Above Case was for , if there was No lemma neither an instance Trained from training data, Below is When There is no Instance but there is a lemma trained and <UNK> is not given . Input=Instance, Output=Lemma , We convert lemma into the Wordnet MFS lemma
                        else:
                            if(len((InvertseDictionary[WordDictIndex[FinalDictID]]).split("%")) ==1):
                                NoInstanceWord=InvertseDictionary[WordDictIndex[FinalDictID]]
                                for syn in wn.synsets(NoInstanceWord):
                                    synsets=syn.name().split('.')[0]
                                    if(NoInstanceWord==synsets):
                                        NewFoundID=syn.lemmas()[0]._key
                                        NewFoundID=synsetfromsensekey(NewFoundID)
                                        BabelnetID=wordnet2babelnet[NewFoundID]

                                if(OutputType!="babelnet"): 
                                    NewFoundID=babelnet2others.get(BabelnetID)
                                    if(NewFoundID==None):
                                        print("NOT FOUND :"+BabelnetID+" Assigning bn:00000001n as Default")
                                        NewFoundID=babelnet2others["bn:00000001n"]


                                outfile.write(InstanceID+" "+NewFoundID+"\n")

                            else:
                                NewFoundID=InvertseDictionary[WordDictIndex[FinalDictID]]
                                NewFoundID=synsetfromsensekey(NewFoundID)
                                BabelnetID=wordnet2babelnet[NewFoundID]

                                if(OutputType!="babelnet"): 
                                        NewFoundID=babelnet2others.get(BabelnetID)
                                        if(NewFoundID==None):
                                            print("NOT FOUND :"+BabelnetID+" Assigning bn:00000001n as Default")
                                            NewFoundID=babelnet2others["bn:00000001n"]
                                                    

                                outfile.write(InstanceID+" "+NewFoundID+"\n")
                SentenceCounter=SentenceCounter+1
    outfile.close()





#This is How we Initiate the ElmoEmbedding, We Have Set that it will be accepting Tokens, specified its input will be String , sequence_len has to be mentioned , how many tokens will it take , and what will be the size of each batch.
def ElmoEmbedding(x):   
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, "string")),
                            "sequence_len": tf.constant(batch_size*[max_len])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]


def synsetfromsensekey(SenseKey):
    synset = wn.lemma_from_key(SenseKey).synset()
    synset_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos()
    return synset_id






predict_lexicographer("/home/baqai/Desktop/public_homework_3 (copy)/Evaluation_Datasets/ALL/ALL.data.xml","../outputFile.txt","/home/baqai/Desktop/public_homework_3/resources")


