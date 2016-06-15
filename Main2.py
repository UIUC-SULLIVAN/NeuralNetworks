
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import genfromtxt
import tensorflow as tf
import time
import os


# In[2]:

TestData = np.load('ANSI_Test_Set_1E3_V2.npy')
TestData_key = np.load('ANSI_Test_Set_key_1E3_V2.npy')


# In[3]:

TrnData = np.load('ANSI_Trn_Set_1E5_V3.npy')
TrnData_key = np.load('ANSI_Trn_Set_key_1E5_V3.npy')


# In[ ]:

LLD=30


# In[ ]:

tmpp = TrnData[5]
tmpp[0:LLD] = 0

plt.plot(tmpp)

print np.sum(TrnData[0])

print TrnData_key[0]


# In[7]:

plt.plot(TrnData[0])


# In[8]:


for i in range(TrnData.shape[0]):
    
    TrnData[i,0:LLD]=0
    TrnData[i,:] = TrnData[i,:]/np.max(TrnData[i,:])
    

for i in range(TestData.shape[0]):
    TrnData[i,0:LLD]=0
    TestData[i,:] = TestData[i,:]/np.max(TestData[i,:])
    


# In[9]:

plt.plot(TrnData[7])
TrnData_key[7]


# In[10]:

def results(res):
    print 'Am241: %s' %res[0] 
    print 'Ba133: %s' %res[1] 
    print 'Co60: %s'  %res[2] 
    print 'Cs137: %s'  %res[3] 
    print 'Eu152: %s'  %res[4] 
    print 'K40: %s' %res[5] 
    print 'Ra226: %s' %res[6] 
  


# In[11]:

isotopes = ['Am241:'
    , 'Ba133:'
    , 'Co57:'
    , 'Co60:'
    , 'Cr51:'
    , 'Cs137:'
    , 'Eu152:'
    , 'Ga67:'
    , 'I123:'
    , 'I125:'
    
    , 'I131:'
    , 'In111:'
    , 'Ir192:'
    , 'K40:'
    , 'Lu177m:' 
    , 'Mo99:'
    , 'Np237:'
    , 'Pd103:'
    , 'Pu239:'
    , 'Pu240:'
    
    , 'Ra226:'
    , 'Se75:'
    , 'Sm153:'
    , 'Sr89:'
    , 'Tc99m:'
    , 'Th232:'
    , 'Tl201:'
    , 'Tl204:'
    , 'U233:'
    , 'U235:'

    , 'U238:'
    , 'Xe133:' ]


# In[ ]:




# In[12]:

def results(res):
    print 'Am241: %s' %res[0] 
    print 'Ba133: %s' %res[1] 
    print 'Co57: %s'  %res[2] 
    print 'Co60: %s'  %res[3] 
    print 'Cr51: %s'  %res[4] 
    print 'Cs137: %s' %res[5] 
    print 'Eu152: %s' %res[6] 
    print 'Ga67: %s' %res[7] 
    print 'I123: %s' %res[8] 
    print 'I125: %s'  %res[9]
    
    print 'I131: %s' %res[10] 
    print 'In111: %s' %res[11] 
    print 'Ir192: %s'  %res[12] 
    print 'K40: %s'  %res[13] 
    print 'Lu177m: %s'  %res[14] 
    print 'Mo99: %s' %res[15] 
    print 'Np237: %s' %res[16] 
    print 'Pd103: %s' %res[17] 
    print 'Pu239: %s' %res[18] 
    print 'Pu240: %s'  %res[19]
    
    print 'Ra226: %s' %res[20] 
    print 'Se75: %s' %res[21] 
    print 'Sm153: %s'  %res[22] 
    print 'Sr89: %s'  %res[23] 
    print 'Tc99m: %s'  %res[24] 
    print 'Th232: %s' %res[25] 
    print 'Tl201: %s' %res[26] 
    print 'Tl204: %s' %res[27] 
    print 'U233: %s' %res[28] 
    print 'U235: %s'  %res[29]

    print 'U238: %s' %res[30] 
    print 'Xe133: %s'  %res[31]
    




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[13]:

def results2(res):
    
    index = [i[0] for i in sorted(enumerate(res), key=lambda x:x[1])]
    index = list(reversed(index))
    for i in range(len(res)):
        print isotopes[index[i]], res[index[i]]



# In[ ]:




# In[ ]:




# In[ ]:

sess.close()


# In[ ]:

# execfile('ANN_V5_1.py')  ## for part of ANSI
execfile('ANN_V5_2.py') ## For full ANSI 


# In[15]:

plt.plot(loss_train[0:i])
plt.plot(loss_test[0:i],'--r')
plt.title('Train Error')
#plt.savefig('test_%s_%s/Train_Error.png' %(I ,J))


# In[16]:

print loss_train[i-1]
print loss_test[i-1]


# In[17]:

L2_all = sess.run(L2_Reg)


# In[18]:

print L2_all


# In[ ]:




# In[19]:

spec_CoCs1 = np.empty(1024)

file_name = "CoCs1.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        spec_CoCs1[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int


# In[22]:

0.0505896/0.776605


# In[21]:

spec_CoCs1[0:LLD]=0
now = time.time()
CoCs1_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (spec_CoCs1/np.max(spec_CoCs1)).reshape(1,1024),keep_prob: 1.0 })[0]
later = time.time()
results2(CoCs1_results)

plt.plot(spec_CoCs1)


# In[202]:

import time 


# In[203]:

later-now


# In[ ]:




# In[ ]:




# In[ ]:




# In[23]:

spec_temp3 = np.empty(1024)

spec_temp2 = np.empty(1024)

spec_UNatK40 = np.empty(1024)

spec_Back = np.empty(1024)



# In[24]:


file_name = "Background_long.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        spec_Back[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int




file_name = "UNat_1ft_K-40_1ft.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        spec_UNatK40[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int



file_name = "Co60_500s000.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        spec_temp2[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int



file_name = "Cs137_500s001.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        spec_temp3[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int


# In[25]:

spec_Back[0:LLD]=0
spec_UNatK40[0:LLD]=0
spec_temp2[0:LLD]=0
spec_temp3[0:LLD]=0


# In[26]:

Co1_test = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (spec_temp2/np.max(spec_temp2)).reshape(1,1024),keep_prob: 1.0 })[0]
Cs1_test = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (spec_temp3/np.max(spec_temp3)).reshape(1,1024),keep_prob: 1.0 })[0]
UK40 = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (spec_UNatK40/np.max(spec_UNatK40)).reshape(1,1024),keep_prob: 1.0 })[0]
Back = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (spec_Back/np.max(spec_Back)).reshape(1,1024),keep_prob: 1.0 })[0]


# In[27]:

results2(Co1_test)
plt.plot(spec_temp2/np.max(spec_temp2))
plt.xlabel('channel')
plt.ylabel('Intensity')


# In[28]:

results2(UK40)
plt.plot(spec_UNatK40/np.max(spec_UNatK40))
plt.xlabel('channel')
plt.ylabel('Intensity')


# In[29]:

results2(Back)
plt.plot(spec_Back/np.max(spec_Back))
plt.xlabel('channel')
plt.ylabel('Intensity')


# In[30]:

# 662 -> 199
# 1173 -> 348
# 1332 -> 396


# In[31]:

results2(Cs1_test)
plt.plot(spec_temp3/np.max(spec_temp3))
plt.xlabel('channel')
plt.ylabel('Intensity')


# In[ ]:




# In[32]:

Test_Ra500s = np.empty(1024)

file_name = "Ra_500s000.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        Test_Ra500s[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int

Test_Eu500s = np.empty(1024)

file_name = "Eu152_500s000.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        Test_Eu500s[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int


Test_Ba500s = np.empty(1024)

file_name = "Ba133_500s000.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        Test_Ba500s[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int
        
        
        

Test_BaCo60s = np.empty(1024)

file_name = "60sDECAYTEST000_BaCo.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        Test_BaCo60s[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int
                
Test_CoCs200s = np.empty(1024)

file_name = "200sDECAYTEST000_CoCs.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        Test_CoCs200s[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int
                
        
        

Test_CoCs60s = np.empty(1024)

file_name = "60sDECAYTEST000_CoCs.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        Test_CoCs60s[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int
                
        
        

Test_CoCs10s = np.empty(1024)

file_name = "10sDECAYTEST000_CoCs.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        Test_CoCs10s[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int
                
        
        
Test_CoCs5s = np.empty(1024)

file_name = "5sDECAYTEST000_CoCs.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        Test_CoCs5s[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int
         

            
            
            
            
Test_Ba15s = np.empty(1024)

file_name = "Ba_15sDECAYTEST000.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        Test_Ba15s[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int
                
 

            
Test_Ba5s = np.empty(1024)

file_name = "Ba_5sDECAYTEST000.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        Test_Ba5s[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int
                
 


# In[33]:



Test_BaCo10s = np.empty(1024)

file_name = "BaCo_10sDECAYTEST000.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        Test_BaCo10s[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int
             


# In[34]:

Test_Eu10s = np.empty(1024)

file_name = "Eu_10sDECAYTEST000.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        Test_Eu10s[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int
             
            
            
Test_Eu10s_1 = np.empty(1024)

file_name = "Eu_10sDECAYTEST001.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        Test_Eu10s_1[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int
             
            
            
Test_Eu60s = np.empty(1024)

file_name = "Eu_60sDECAYTEST000.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        Test_Eu60s[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int
             
            
                


# In[35]:

Test_CoCs5s[0:LLD]=0

CoCs5s_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (Test_CoCs5s/np.max(Test_CoCs5s)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(CoCs5s_results)
plt.plot(Test_CoCs5s)


# In[ ]:




# In[ ]:




# In[36]:

Test_CoCs10s[0:LLD]=0


CoCs10s_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (Test_CoCs10s/np.max(Test_CoCs10s)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(CoCs10s_results)
plt.plot(Test_CoCs10s)


# In[37]:

Test_CoCs60s[0:LLD]=0

CoCs60s_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (Test_CoCs60s/np.max(Test_CoCs60s)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(CoCs60s_results)
plt.plot(Test_CoCs60s)


# In[38]:

np.linspace(0,5,6)


# In[39]:

E_from_C = 3.533488*np.linspace(0,1023,1024)-49.137384


# In[40]:

plt.semilogy(E_from_C,Test_CoCs200s)
plt.xlim([1300,1800])
#plt.ylim([0,200])
print np.argmax(Test_CoCs200s[370:550])


# In[41]:

plt.semilogy(E_from_C,Test_Eu500s)
plt.xlim([1300,1800])
#plt.ylim([0,200])


# In[42]:

plt.semilogy(E_from_C,Test_Eu500s)
plt.xlim([1300,1800])
#plt.ylim([0,200])


# In[43]:

plt.semilogy(E_from_C,spec_UNatK40)
plt.xlim([1300,1800])
#plt.ylim([0,200])


# In[44]:

plt.semilogy(E_from_C,spec_Back)
#plt.xlim([1300,1800])
#plt.ylim([0,200])


# In[ ]:




# In[ ]:




# In[68]:

Test_CoCs200s[0:LLD]=0



plt.rcParams.update({'font.size': 30})


CoCs200s_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (Test_CoCs200s/np.max(Test_CoCs200s)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(CoCs200s_results)


plt.figure(figsize=(20,10))
plt.semilogy(E_from_C,Test_CoCs200s)
plt.xlim([0,3000])
#plt.title('Co-60 and Cs-137')

plt.ylabel('Counts')
plt.xlabel('Energy [keV]')


# In[46]:

Test_Ra500s[0:LLD]=0

Ra500_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (Test_Ra500s/np.max(Test_Ra500s)).reshape(1,1024), keep_prob: 1.0})[0]
results2(Ra500_results)

plt.plot(Test_Ra500s)


# In[47]:

Test_Eu500s[0:LLD]=0

Eu152_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (Test_Eu500s/np.max(Test_Eu500s)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(Eu152_results)
plt.plot(Test_Eu500s)


# In[48]:

Test_Eu60s[0:LLD]=0

Eu152_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (Test_Eu60s/np.max(Test_Eu60s)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(Eu152_results)
plt.plot(Test_Eu60s)



# In[49]:

Test_Eu10s[0:LLD]=0

Eu152_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (Test_Eu10s/np.max(Test_Eu10s)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(Eu152_results)
plt.plot(Test_Eu10s)


# In[50]:

Test_Eu10s_1[0:LLD]=0

Eu152_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (Test_Eu10s_1/np.max(Test_Eu10s_1)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(Eu152_results)
plt.plot(Test_Eu10s_1)


# In[51]:

Test_Ba5s[0:LLD]=0

Ba133_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (Test_Ba5s/np.max(Test_Ba5s)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(Ba133_results)

plt.plot(Test_Ba5s)


# In[52]:

Test_Ba15s[0:LLD]=0

Ba133_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (Test_Ba15s/np.max(Test_Ba15s)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(Ba133_results)

plt.plot(Test_Ba15s)


# In[53]:

Test_Ba500s[0:LLD]=0

Ba133_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (Test_Ba500s/np.max(Test_Ba500s)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(Ba133_results)

plt.plot(Test_Ba500s)


# In[ ]:




# In[82]:


Test_BaCo10s[0:LLD]=0

plt.rcParams.update({'font.size': 30})

BaCo10s_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (Test_BaCo10s/np.max(Test_BaCo10s)).reshape(1,1024),keep_prob: 1.0 })[0]



plt.figure(figsize=(20,10))
plt.semilogy(E_from_C,Test_BaCo10s)
plt.xlim([0,3000])
#plt.title('Co-60 and Ba-133')
plt.ylabel('Counts')
plt.xlabel('Energy [keV]')


results2(BaCo10s_results)


# In[55]:

Test_BaCo60s[0:LLD]=0


BaCo60s_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (Test_BaCo60s/np.max(Test_BaCo60s)).reshape(1,1024),keep_prob: 1.0 })[0]

plt.plot(Test_BaCo60s)

results2(BaCo60s_results)


# In[56]:

TEST_1 = np.empty(1024)

file_name = "CoCsBaEu2.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        TEST_1[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int

        

TEST_1[0:LLD]=0
plt.semilogy(TEST_1)


Test1_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (TEST_1/np.max(TEST_1)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(Test1_results)


# In[ ]:




# In[ ]:




# In[ ]:




# In[71]:

TEST_1 = np.empty(1024)


plt.rcParams.update({'font.size': 30})


file_name = "2400sDECAYTEST000.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        TEST_1[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int

        


TEST_1[0:LLD]=0
plt.figure(figsize=(20,10))

plt.semilogy(E_from_C,TEST_1/2400.0)
plt.xlim([0,3000])
#plt.title('HEU, Rocky Flats Shells')
plt.ylabel('Counts')
plt.xlabel('Energy [keV]')


BaCo60s_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (TEST_1/np.max(TEST_1)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(BaCo60s_results)


# In[ ]:




# In[ ]:




# In[81]:

TEST_1 = np.empty(1024)





file_name = "60sDECAYTEST000_RF.Spe"
with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        TEST_1[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int

        


TEST_1[0:LLD]=0
plt.figure(figsize=(20,10))

plt.semilogy(E_from_C,TEST_1)
plt.xlim([0,3000])
#plt.title('HEU, Rocky Flats Shells')
plt.ylabel('Counts')
plt.xlabel('Energy [keV]')


BaCo60s_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (TEST_1/np.max(TEST_1)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(BaCo60s_results)


# In[80]:

TEST_1 = np.empty(1024)

plt.rcParams.update({'font.size': 30})

#file_name = "BERP_halfinchlead_119cm_300s.Spe"
file_name = "60sDECAYTEST000_BERP.Spe"

with open(file_name) as f:

    # read each spectra into a temp file, total of 1024 channels in this spectra
    content = f.readlines() # read all of the .Spe file into contnet 
    for i in range(1024):
        TEST_1[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int

plt.figure(figsize=(20,10))        
plt.semilogy(E_from_C,TEST_1)
plt.xlim([0,3000])
#plt.title('BERP Ball')
plt.ylabel('Counts')
plt.xlabel('Energy [keV]')
TEST_1[0:LLD]=0


BaCo60s_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (TEST_1/np.max(TEST_1)).reshape(1,1024),keep_prob: 1.0 })[0]
results2(BaCo60s_results)


# In[169]:

plt.plot(np.linspace(-5,5,100),np.tanh(np.linspace(-5,5,100)))
plt.xlim([-2.5,2.5])
plt.ylim([-1.5,1.5])
plt.title('tanh')


# In[170]:

saver.save(sess, "mymodel_SORMA.ckpt")


# In[162]:

np.linspace(-5,5,100)


# In[ ]:



