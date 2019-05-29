#!/usr/bin/env python
# coding: utf-8

# In[87]:


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


# In[88]:


print(fake_acti)
print(fake_conv)
print(fake_opti)
print(fake_laye)


# In[89]:


def mutate():
    for j in range(pop-4):
        
        #make 3 clones each
        i=j//4
        for h in range(4):
            new_pop[h][j+4]=new_pop[h][i]
        
        #mutate
        if np.random.rand(1)<0.25:
            new_pop[0][j+4]=np.random.randint(0,activate)
        if np.random.rand(1)<0.25:
            rand1=np.random.randint(0,4)
            rand2=np.random.randint(0,4)
            rand3=np.random.randint(0,4)
            new_pop[1][j+4]=new_pop[1][rand1]+0.75*(new_pop[1][rand2]-new_pop[1][rand3])
        if np.random.rand(1)<0.25:
            new_pop[2][j+4]=np.random.randint(0,optimize)
        if np.random.rand(1)<0.25:
            rand1=np.random.randint(0,4)
            rand2=np.random.randint(0,4)
            rand3=np.random.randint(0,4)
            new_pop[3][j+4]=new_pop[3][rand1]+0.75*(new_pop[3][rand2]-new_pop[3][rand3])
    return


# In[90]:


def keep(m):
    new_opti.append(fake_opti[m])
    new_conv.append(fake_conv[m])
    new_acti.append(fake_acti[m])
    new_laye.append(fake_laye[m])
    return


# In[91]:


#determine best combinantion of parameters
for i in range(gen):
    print(fake_acti)
    print(fake_conv)
    print(fake_opti)
    print(fake_laye)
    
    #change parameter to useful values
    real_acti=[]
    for a in range(pop):
        b=fake_acti[a]
        if b==0:
            c='elu'
        if b==1:
            c='relu'
        if b==2:
            c='tanh'
        if b==3:
            c='sigmoid'
        real_acti.insert(a,c)
        
    fake_conv=fake_conv-fake_conv%1
    real_conv=16*2**fake_conv
    
    real_opti=[]
    for d in range(pop):
        e=fake_opti[d]
        if e==0:
            f='rmsprop'
        if e==1:
            f='adam'
        if e==2:
            f='sgd'
        if e==3:
            f='adagrad'
        if e==4:
            f='adadelta'
        if e==5:
            f='adamax'
        if e==6:
            
            f='nadam'
        real_opti.insert(d,f)
    
    real_laye=fake_laye-fake_laye%1

    print(real_acti)
    print(real_opti)
    
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
    print(score_list[B[0]])
    
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
    m=B[p+3]
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
        fake_acti[f]=new_pop[0][f]
        fake_conv[f]=new_pop[1][f]
        fake_opti[f]=new_pop[2][f]
        fake_laye[f]=new_pop[3][f]
    
    
        


# In[92]:


print(score_list,B)


print('final accuracy=',score_list[B[0]])
    


# In[93]:



print(fake_acti)
print(real_acti)
print(fake_conv)
print(fake_opti)
print(fake_laye)


# In[85]:


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
m=B[p+3]
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
    fake_acti[f]=new_pop[0][f]
    fake_conv[f]=new_pop[1][f]
    fake_opti[f]=new_pop[2][f]
    fake_laye[f]=new_pop[3][f]


    


# In[94]:


print(new_pop)


# In[ ]:




