
# coding: utf-8

# In[156]:


#https://higgsml.lal.in2p3.fr/software/starting-kit/
import pandas as pd
import numpy as np
import os
import sklearn
import random
import tensorflow as tf
from sklearn.metrics import f1_score,accuracy_score
import xgboost as xgbc


# In[157]:


train = pd.read_csv('/kaggle/higgs-boson/data/training.csv')
test = pd.read_csv('/kaggle/higgs-boson/data/test.csv')


# In[158]:


train.shape


# In[159]:


test.shape


# In[160]:


train.columns.values


# In[161]:


test.columns.values


# In[162]:


train.drop('Weight',axis=1,inplace=True)


# In[163]:


train.shape


# In[164]:


#replace s with 1 and p with 0
labels = [int(i == 's') for i in train.Label.values]
train['Label'] = labels


# In[165]:


cols_der = ['DER_mass_MMC', 'DER_mass_transverse_met_lep',
       'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet',
       'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep',
       'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau',
       'DER_met_phi_centrality', 'DER_lep_eta_centrality']
cols_raw = ['PRI_tau_pt',
       'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta',
       'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet',
       'PRI_jet_num', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta',
       'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi',
       'PRI_jet_all_pt']


# In[166]:


elim_cols = []
for i in cols_der:
    for j in cols_raw:
        if j!=i: 
            corr = np.corrcoef(train[i],train[j])[0][1]
            if corr < -0.8 or corr > 0.85:
                print i,j,"corr coef:",corr
                elim_cols.append(j)


# In[167]:


final_cols = list(set(list(train.columns.values))-set(elim_cols))


# In[168]:


def countsnine(z):
    return sum([1 if i==-999.0 else 0 for i in z])        


# In[169]:


rem_cols = []
for i in final_cols:
    if i!='Label':
        if countsnine(train[i])!=0 or countsnine(test[i])!=0:
            print i,countsnine(train[i]),countsnine(test[i])
            rem_cols.append(i)


# In[170]:


rem_cols = rem_cols + ['EventId','Label']


# In[171]:


final_cols_updated = list(set(final_cols)-set(rem_cols))


# In[172]:


len(final_cols_updated)


# In[173]:


train_x = train[final_cols_updated]
train_y = train['Label'].values
test_final = test[final_cols_updated]


# In[174]:


train_y = np.array([[0,1] if i==1 else [1,0] for i in train_y])


# In[175]:


nrm = sklearn.preprocessing.StandardScaler()


# In[176]:


train_id = random.sample(range(train_x.shape[0]),int(0.8*train_x.shape[0]))
valid_id = list(set(range(train_x.shape[0]))-set(list(train_id)))


# In[177]:


xtr_train = train_x.iloc[train_id]
y_train = train_y[train_id]
xtr_valid = train_x.iloc[valid_id]
y_valid = train_y[valid_id]


# In[178]:


xtr_train = nrm.fit_transform(xtr_train)
xtr_valid = nrm.transform(xtr_valid)
xt = nrm.transform(test_final)


# In[179]:


zero_ids = np.argwhere(y_train[:,0]==1).reshape([-1])
one_ids = np.argwhere(y_train[:,0]==0).reshape([-1])


# In[180]:


#y_train = np.reshape(y_train,newshape=[y_train.shape[0],1])
print y_train.shape


# In[181]:


len(zero_ids)


# In[182]:


len(one_ids)


# In[183]:


xtr_train.shape


# In[184]:


y_train.shape


# In[185]:


xt.shape


# In[188]:


xgb = xgbc.XGBClassifier(max_depth=40,n_estimators=201)

ids = random.sample(zero_ids,100000)+random.sample(one_ids,60000)
xgb.fit(xtr_train[ids],y_train[ids,1])

print xgb.score(xtr_train,y_train[:,1])
print xgb.score(xtr_valid,y_valid[:,1])
train_pred = xgb.predict(xtr_train)
valid_pred = xgb.predict(xtr_valid)
print f1_score(train_pred,y_train[:,1])
print f1_score(valid_pred,y_valid[:,1])


# In[189]:


outs = [[i,abs(k-j)] for i,j,k in zip(ids,xgb.predict_proba(xtr_train[ids])[:,1],y_train[ids,1])]


# In[190]:


outs_f = [i for i,j in outs if j>0.49]


# In[191]:


new_ids = list(set(ids)-set(outs_f))


# In[192]:


xgb.fit(xtr_train[new_ids],y_train[new_ids,1])

print xgb.score(xtr_train,y_train[:,1])
print xgb.score(xtr_valid,y_valid[:,1])
train_pred = xgb.predict(xtr_train)
valid_pred = xgb.predict(xtr_valid)
print f1_score(train_pred,y_train[:,1])
print f1_score(valid_pred,y_valid[:,1])


# In[202]:


for d in ['/gpu:0','/gpu:1','/gpu:2']:
    with tf.device(d):
        tf.reset_default_graph()

        xs = tf.placeholder(dtype=tf.float32,shape=[None,xtr_train.shape[1]])
        ys = tf.placeholder(dtype=tf.float32,shape=[None,2])
        keep_prob = tf.placeholder(tf.float32)

        w1 = tf.Variable(tf.truncated_normal(shape=[xtr_train.shape[1],64],mean=0,stddev=1),dtype=tf.float32)
        w1 = tf.multiply(w1,tf.sqrt(2.0/xtr_train.shape[1]))
        w2 = tf.Variable(tf.truncated_normal(shape=[64,64],mean=0,stddev=1),dtype=tf.float32)
        w2 = tf.multiply(w2,tf.sqrt(2.0/64))
        w3 = tf.Variable(tf.truncated_normal(shape=[64,64],mean=0,stddev=1),dtype=tf.float32)
        w3 = tf.multiply(w3,tf.sqrt(2.0/64))
        w4 = tf.Variable(tf.truncated_normal(shape=[64,y_train.shape[1]],mean=0,stddev=1),dtype=tf.float32)
        w4 = tf.multiply(w4,tf.sqrt(2.0/64))
        b1 = tf.Variable(tf.zeros([64]),dtype=tf.float32)
        b2 = tf.Variable(tf.zeros([64]),dtype=tf.float32)
        b3 = tf.Variable(tf.zeros([64]),dtype=tf.float32)
        b4 = tf.Variable(tf.zeros([y_train.shape[1]]),dtype=tf.float32)

        l1 = tf.matmul(xs,w1)+b1
        o1 = tf.nn.relu(l1)
        o1 = tf.nn.dropout(o1,keep_prob)
        l2 = tf.matmul(o1,w2)+b2
        o2 = tf.nn.relu(l2)
        o2 = tf.nn.dropout(o2,keep_prob)
        l3 = tf.matmul(o2,w3)+b3
        o3 = tf.nn.relu(l3)
        o3 = tf.nn.dropout(o3,keep_prob)
        l4 = tf.matmul(o3,w4)+b4
        o4 = tf.nn.softmax(l4)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l4,labels=ys))
        train = tf.train.AdamOptimizer(5*1e-5).minimize(cross_entropy)
        correct_pred = tf.cast(tf.equal(tf.argmax(o4,1),tf.argmax(ys,1)),dtype=tf.float32)
        y_pred = tf.argmax(o4,1)
        accuracy = tf.reduce_mean(correct_pred)
        init = tf.global_variables_initializer()


# In[203]:


steps = 1000000
batch_size = 200
'''def pred_val(x,t=0.5):
        return np.array([0 if i<t else 1 for i in x])
'''
tr_acc = []
val_acc = []
tr_f1score = []
val_f1score = []
tr_loss = []


# In[204]:


#config = tf.ConfigProto()
#config.gpu_options.allocator_type = 'BFC'
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    for i in range(steps):
        batch_ids = random.sample(zero_ids,150)+random.sample(one_ids,150)
        np.random.shuffle(batch_ids)
        batch_x = xtr_train[batch_ids]
        batch_y = y_train[batch_ids]
        #print np.unique(batch_y,return_counts=True)
        _,loss = sess.run([train,cross_entropy],feed_dict={xs:batch_x,ys:batch_y,keep_prob:0.6}) 
        if i%2000==0:
            tracc,t_pred = sess.run([accuracy,y_pred],feed_dict={xs:batch_x,ys:batch_y,keep_prob:1.0})
            trfone =  f1_score(t_pred,np.argmax(batch_y,axis=1))
            valacc,v_pred = sess.run([accuracy,y_pred],feed_dict={xs:xtr_valid,ys:y_valid,keep_prob:1.0})
            valfone = f1_score(v_pred,np.argmax(y_valid,axis=1))
            tr_acc.append(tracc)
            val_acc.append(valacc)
            tr_f1score.append(trfone)
            val_f1score.append(valfone)
            tr_loss.append(loss)
            print "TrLoss: {1}, TrF1-score: {2}, TrAcc: {3}, ValF1-score: {4}, ValAcc: {5}".format(i+1,loss,trfone,tracc,valfone,valacc)
            #print logit2
            #break
    saver = tf.train.Saver()
    saver.save(sess,'/home/sayon/higg_model.ckpt')
    np.save("/home/sayon/higgs_boson_train_accuracy.npy",tr_acc)
    np.save("/home/sayon/higgs_boson_validation_accuracy.npy",val_acc)
    np.save("/home/sayon/higgs_boson_f1score,npy",tr_f1score)
    np.save("/home/sayon/higgs_boson_valid_f1score.npy",val_f1score)
    np.save("/home/sayon/higgs_boson_train_loss.npy",tr_loss)
    #test_pred = sess.run(o4,feed_dict={xs:xt,keep_prob:1.0})
    #np.save("/home/sayon/higgs_boson_pred.npy",test_pred)


# In[ ]:




