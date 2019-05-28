#!/usr/bin/env python
# coding: utf-8

# In[51]:


from keras.models import Model
from keras.layers import *
from keras.datasets import mnist
from matplotlib import pyplot as plt

#prepare data
from keras.utils import np_utils 
import numpy as np
import os
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train =x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_test/=255
x_train/=255
y_train=np_utils.to_categorical(y_train,10)
y_test= np_utils.to_categorical(y_test,10)

#num of choice per parameter
optimize=7
activate=4
layer=3
conv=3

#num of population,generation
pop=12
gen=10

#create first parameters
fake_opti=np.random.randint(0,optimize,pop)
fake_acti=np.random.randint(0,activate,pop)
fake_laye=np.random.randint(0,layer,pop)
fake_conv=np.random.randint(0,conv,pop)


# In[52]:


print(fake_acti)
print(fake_conv)
print(fake_opti)
print(fake_laye)


# In[53]:


def mutate():
    for j in range(pop-4):
        
        #make 3 clones each
        i=j//4
        for h in range(4):
            
            new_pop[h][j+4]=new_pop[h][i]
        
        #mutate
        if np.random.rand(1)<0.25:
            new_pop[0][j+4]=np.random.randint(0,activation)
        if np.random.rand(1)<0.25:
            rand1=np.random.randint(0,4)
            rand2=np.random.randint(0,4)
            rand3=np.random.randint(0,4)
            new_pop[1][j+4]=new_pop[1][rand1]+0.75*(new_pop[1][rand2]-new_pop[1][rand3])
        if np.random.rand(1)<0.25:
            new_pop[2][j+4]=np.random.randint(0,optimizer)
        if np.random.rand(1)<0.25:
            rand1=np.random.randint(0,4)
            rand2=np.random.randint(0,4)
            rand3=np.random.randint(0,4)
            new_pop[3][j+4]=new_pop[3][rand1]+0.75*(new_pop[3][rand2]-new_pop[3][rand3])
      
    return


# In[54]:


def keep(m):
    new_opti.append(fake_opti[m])
    new_conv.append(fake_conv[m])
    new_acti.append(fake_acti[m])
    new_laye.append(fake_laye[m])
    return


# In[ ]:


#determine best combinantion of parameters
for i in range(gen):
    print(fake_acti)
    print(fake_conv)
    print(fake_opti)
    print(fake_laye)
    #change parameter to useful values
    fake_conv=fake_conv-fake_conv%1
    real_conv=32*2**fake_conv
    real_acti=[]
    for a in range(pop):
        b=fake_acti[a]
        if b//activate==0:
            c='elu'
        if b//activate==1:
            c='relu'
        if b//activate==2:
            c='tanh'
        if b//activate==3:
            c='sigmoid'
        real_acti.insert(a,c)
    real_opti=[]
    for d in range(pop):
        e=fake_opti[d]
        if e//optimize==0:
            f='rmsprop'
        if e//optimize==1:
            f='adam'
        if e//optimize==2:
            f='sgd'
        if e//optimize==3:
            f='adagrad'
        if e//optimize==4:
            f='adadelta'
        if e//optimize==5:
            f='adamax'
        if e//optimize==6:
            f='nadam'
        real_opti.insert(d,f)
    real_laye=fake_laye-fake_laye%1
  
    
    #evaluate each model in generation
    score_list=[]
    print('\ngen.',i+1,'\n')
    for j in range(pop):
        input_layer= Input(shape=(28,28,1))
        conv=Conv2D(real_conv[j],(3,3),activation=real_acti[j])(input_layer)
        
        #add extra layers
        for k in range(real_laye[j]):
            conv=Conv2D(real_conv[j],(3,3),activation=real_acti[j])(conv)
        maxpool1= MaxPooling2D(pool_size=(2,2))(conv)
        flat1=Flatten()(maxpool1)
        dense1= Dense(128,activation=real_acti[j])(flat1)
        output=Dense(10,activation='softmax')(dense1)
        model=Model(inputs=input_layer,outputs=output)
        model.compile(loss='categorical_crossentropy',optimizer=real_opti[j],metrics=["accuracy"])
        print('\nnum.',j+1,'\n')
        print(model.summary())
        print(model.output_shape)
        model.fit(x_train,y_train,batch_size=32,epochs=1,verbose=1)
        scores=model.evaluate(x_test,y_test,verbose=1)
        print("loss",scores[0])
        print("accuracy",scores[1])
        score_list.append(scores[1])
        
    #sort scores    
    A=np.array(score_list)
    B=A.argsort()[::-1]
    print(B[0])
    print(max(score_list))
    
    #no more need for creating new parameters
    if i==gen-1:
        break
        
   #keep selected parameters
    new_opti=[]
    new_conv=[]
    new_acti=[]
    new_laye=[]
    
    #keep 3 top 
    for l in range(3):
        m=B[l]
        keep(m)
    
    #keep 1 random    
    p=np.random.randint(pop-3)
    m=B[p+2]
    keep(m)
    
    #new population arranged for mutation
    new_pop=[[0 for h in range(pop)]for e in range(4)]
    for g in range(4):
        new_pop[0][g]=new_acti[g]
        new_pop[1][g]=new_conv[g]
        new_pop[2][g]=new_opti[g]
        new_pop[3][g]=new_laye[g]
    
    #mutate
    mutate()
    
    #introduce new parameters
    for f in range(pop):
        fake_acti[g]=new_pop[0][g]
        fake_conv[g]=new_pop[1][g]
        fake_opti[g]=new_pop[2][g]
        fake_laye[g]=new_pop[3][g]
    
    
        


# In[49]:


print(B[0])
print('final accuracy',max(score_list))
    

