import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.io import read_file
from tensorflow.image import decode_jpeg,resize
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.applications import InceptionV3
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense,Embedding,GRU
from tensorflow import expand_dims,reduce_sum,concat,reshape,cast,reduce_mean
from tensorflow.nn import tanh,softmax,relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.math import logical_not,equal
image_path='dataset/Images/'
caption_path = "dataset/captions.txt"


with open(caption_path,'r') as f:
    captions=f.read()
    f.close()

captions_dict = collections.defaultdict(list)

for i in captions.split('\n'):
    try:
        image_name,cap=i.split(',')
        cap='seqstart '+cap+' seqend'
        img_path=image_path+image_name
        if img_path.endswith('.jpg'):
            captions_dict[img_path].append(cap)
    except:
        pass


img_path=list(captions_dict.keys())

total_cap=[]
total_image=[]

for paths in img_path:
    capt=captions_dict[paths]
    total_cap.extend(capt)
    total_image.extend([paths]*len(capt))

def data_loading(path):
    img=read_file(path)
    img=decode_jpeg(img,channels=3)
    img=resize(img,(299,299))
    img=inception_v3.preprocess_input(img)
    return img,path

cnn_model=InceptionV3(include_top=False,weights='imagenet')
feature_model=Model(cnn_model.input,cnn_model.layers[-1].output)
feature_model.summary()


from tensorflow.data import Dataset

img_dataset=Dataset.from_tensor_slices(sorted(set(total_image)))
img_dataset=img_dataset.map(data_loading,num_parallel_calls=tf.data.AUTOTUNE).batch(16)

for image,path in tqdm(img_dataset):
    batch_feat=feature_model(image)
    #print(batch_feat.shape)
    batch_feat=tf.reshape(batch_feat,(batch_feat.shape[0],-1,batch_feat.shape[3]))
    #print(batch_feat.shape)
    for batch_f, pth in zip(batch_feat,path):
        path_features=pth.numpy().decode('utf-8')
        np.save(path_features,batch_f.numpy())

tokenizer = Tokenizer(oov_token='<oov_>',filters='!"#$%+*.,/')
tokenizer.fit_on_texts(total_cap)
tokenizer.word_index['<pad>']=0
tokenizer.index_word[0]='<pad>'

seq=tokenizer.texts_to_sequences(total_cap)
seq_vect=tf.keras.preprocessing.sequence.pad_sequences(seq,padding='post')

max_len = max(len(cap.split()) for cap in total_cap)-1


#pipeline
img_capt_dict=collections.defaultdict(list)

for img,captions in zip(total_image,seq_vect):
    img_capt_dict[img].append(captions)
keys=list(img_capt_dict.keys())

train,test=keys[:7000],keys[7000:]

train_image=[]
train_cap=[]
for i in train:
    cap_len=len(img_capt_dict[i])
    train_image.extend([i]*cap_len)
    train_cap.extend(img_capt_dict[i])

test_image=[]
test_cap=[]
for i in test:
    cap_len=len(img_capt_dict[i])
    test_image.extend([i]*cap_len)
    test_cap.extend(img_capt_dict[i])


size_vocab = len(tokenizer.word_index) + 1

def load_cache(img_name,caption):
    img_ten=np.load(img_name.decode('utf-8')+'.npy')
    return img_ten,caption

data=Dataset.from_tensor_slices((train_image,train_cap))

data=data.map(lambda value1,value2:tf.numpy_function(load_cache,[value1,value2],[tf.float32,tf.int32]))

data=data.shuffle(500).batch(16)
data=data.prefetch(buffer_size=tf.data.AUTOTUNE)


class Attention(Model):
    def __init__(self,num_attentions):
        super(Attention,self).__init__()
        self.dense1=Dense(num_attentions)
        self.dense2=Dense(num_attentions)
        self.dense3=Dense(num_attentions)
    def call(self,feat,hidden_state):
        hidden_time=expand_dims(hidden_state,1)
        attention_hidden=(tanh(self.dense1(feat)+self.dense2(hidden_time)))
        sc=self.dense3(attention_hidden)
        weights_of_attention=softmax(sc,axis=1)
        context_vec=weights_of_attention*feat
        context_vec=reduce_sum(context_vec,axis=1)
        return context_vec,weights_of_attention

class encoder(Model):
    def __init__(self,embed_dim):
        super(encoder,self).__init__()
        self.fully_connected_layer=Dense(embed_dim)
    def call(self,vec):
        vec=self.fully_connected_layer(vec)
        vec=relu(vec)
        return vec

class decoder(Model):
    def __init__(self,embed_dim,total_units,vocab_size):
        super(decoder,self).__init__()
        self.total_units=total_units
        self.embed_layer=Embedding(vocab_size,embed_dim)
        self.LSTM_layer=GRU(self.total_units,return_sequences=True,recurrent_initializer='glorot_uniform',return_state=True)
        self.fully_connect1=Dense(self.total_units)
        self.fully_connect2=Dense(vocab_size)
        self.attention_layer=Attention(self.total_units)
    
    def call(self,x,feat,hidden_state):
        context_vec,weights_of_attention=self.attention_layer(feat,hidden_state)
        x=self.embed_layer(x)
        x=concat([expand_dims(context_vec,1),x],axis=-1)
        out,hidden_states=self.LSTM_layer(x)
        x=self.fully_connect1(out)
        x=reshape(x,(-1,x.shape[2]))
        x=self.fully_connect2(x)
        return x,hidden_states,weights_of_attention
    def state_reset(self,batch_size):
        return tf.zeros(batch_size,self.total_units)

encoder=encoder(256)
decoder=decoder(256,16,size_vocab)


optimizer=Adam()
loss=SparseCategoricalCrossentropy(from_logits=True,reduction='none')

def loss_fuc(real_caption,generated_caption):
    m=logical_not(equal(real_caption,0))
    losss=loss(real_caption,generated_caption)
    m=cast(m,dtype=losss.dtype)
    losss=losss*m
    return reduce_mean(losss)

plot_loss=[]
@tf.function
def train_step(img_data,output):
    loss=0
    hidden_st=decoder.state_reset(batch_size=output.shape[0])
    decoder_input=expand_dims([tokenizer.word_index['seqstart']]*output.shape[0],1)
    with tf.GradientTape() as tape:
        feat=encoder(img_data)
        for i in range(1,output.shape[1]):
            pred,hidden_state,_=decoder(decoder_input,feat,hidden_st)
            loss= loss+loss_fuc(output[:i],pred)
            decoder_input=expand_dims(output[:i],1)
    final_loss=loss/int(output.shape[1])
    train_variables=encoder.train_variables+decoder.train_variables
    grad=tape.gradient(loss, train_variables)
    optimizer.apply_gradients(zip(grad,train_variables))
    return loss,final_loss

epoch=20
for i in range(0,epoch):
    final_loss=0
    for(b,(img_feature,caption)) in enumerate(data):
        batch_loss,total_loss=train_step(img_feature,caption)
        final_loss=final_loss+total_loss
        plot_loss.append(final_loss/len(train))
    print(i,final_loss/len(train))

def tests(img):
    attent_plot=np.zeros((max_len,14))#attenttion_feature_size
    hidden_state=decoder.state_reset(1)
    input_temp=expand_dims(load_cache(img)[0],0)
    img_input=cnn_model(input_temp)
    img_input=reshape(img_input,(img_input.shape[0],-1,img_input.shape[3]))
    feat=encoder((img_input))
    decoder_input=expand_dims([tokenizer.word_index['seqstart']],0)
    cap=[]
    for i in range(max_len):
        pred,hidden_state,weights_attention=decoder(decoder_input,feat,hidden_state)
        attent_plot[i]=reshape(weights_attention,(-1,)).numpy()
        pred_id=tf.argmax(pred[0]).numpy()
        cap.append(tokenizer.index_word[pred_id])
        
        if tokenizer.index_word[pred_id]=='seqend':
            return cap,attent_plot
        decoder_input=expand_dims([pred_id],0)
    attent_plot=attent_plot[:len(cap),:]
    return cap,attent_plot

def image_plot(img,cap,attent_plot):
    temp_img=np.array(Image.open(img))
    figure=plt.figure(figsize=(12,12))
    len_cap=len(cap)
    for i in range(len_cap):
        temp_attention=np.resize(attent_plot[i],(10,10))
        g_size=max(int(np.cell(len_cap/2)),2)
        axis=figure.add_subplot(g_size,g_size,i+1)
        axis.set_title(cap[i])
        img=axis.imshow(temp_img)
        axis.imshow(temp_attention,cmap='gray',alpha=.5,extent=img.get_extent())
    plt.tight_layout()
    plt.plot()
        
img=image_path+test[0]

original_caption=' '.join(tokenizer.index_word[i] for i in test_cap[0] if i not in [0])
predicted,attent_plot=tests(img)
print('original caption:',original_caption)
print('predicted caption',' '.join(predicted))
image_plot(img,predicted,attent_plot)

Image.open(img)