# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 18:57:10 2017

@author: nxk176630
"""
from __future__ import division

from Bring_Data_In_Matrix_Form import create_pickle_file
from Test_Performance import lifted_relational_neural_network_test, get_weights_for_test_matrix

import tensorflow as tf
import numpy
import pickle
import sparse
import subprocess
import sys

learning_rate = 0.05
batch_size = 1   # Batch size remains 1 for this implementation
training_epochs = 1
test_interval = 2
patience_threshold = 1
#t = tf.cast(tf.constant([[[1, 1, 1, 1, 1],[2, 2, 2, 2, 2],[3, 3, 3, 3, 3]],[[4, 4, 4, 4, 4],[5, 5, 5, 5, 5],[6, 6, 6, 6, 6]], [[7, 7, 7, 7, 7],[8, 8, 8, 8, 8],[9, 9, 9, 9, 9]], [[10, 10, 10, 10, 10],[11,11,11,11, 11],[12, 12, 12, 12, 12]]]), dtype=tf.float32)
MyList=[]


def define_shared_weights(lifted_rw,input_size):
   #print(input_size, lifted_rw)
   encoder = []
   with tf.variable_scope('SharedVariable',initializer = tf.random_uniform([1,lifted_rw], dtype=tf.float32)):
      W = tf.get_variable('W',trainable=True)

   for i in range(input_size):
       with tf.variable_scope('SharedVariable', reuse=True):
           W = tf.get_variable('W',trainable=True)
           encoder.append(W)
    
   W_encoder = tf.convert_to_tensor(encoder, dtype=tf.float32,name='MyWeights')
   
   #print(W_encoder.shape)
   W1        = tf.transpose(W_encoder,perm = [2, 0 ,1], name = 'WeightsMatrix')
  # print(W1.shape)
   return (W1)

def define_input_hidden_products(input_layer, hidden_layer):
    return (tf.matmul(input_layer, hidden_layer)) 

#def perform_sigmoid(tensor_input):
 #   return tf.nn.sigmoid(tensor_input)

def perform_combining_rules(tensor_input):
   #return tf.reduce_sum(tensor_input, 1, keepdims=True)    
   return tf.reduce_mean(tensor_input, 1, keepdims=True)    


def lifted_relational_neural_network(x, tf_weights, input_size,lifted_rw,Maximum_Groundings,Output_Layer):
    
   # W_encoder = define_shared_weights(lifted_rw,input_size)
    
    W_hidden = define_input_hidden_products(x, tf_weights)
    
    #W_hidden_after_sigmoid = tf.nn.sigmoid( W_hidden)
    W_hidden_after_sigmoid = W_hidden # lifted_RW x GROUNDING_rw x 1
    
    W_lifted_RW_layer = perform_combining_rules(W_hidden_after_sigmoid)
    
    w_output = tf.Variable(tf.random_uniform([lifted_rw, Output_Layer], dtype=tf.float32),name='FullyConnectedLayer')
    
    Input_provided_to_Output_layer = tf.tensordot(W_lifted_RW_layer, w_output, axes = [[0], [0]])
    
    Input_provided_to_Output_layer1 = tf.reshape(tf.reshape(Input_provided_to_Output_layer,[-1],name = "ReshapeBeforeSoftmax"), [1,2]) # shape of Input_provided_to_Output_layer1 (2,)
    
    #output = tf.nn.softmax(Input_provided_to_Output_layer1)
    
    tf.summary.histogram("activations", Input_provided_to_Output_layer1)    
    return Input_provided_to_Output_layer1
    #return output

def read_input_data(pickle_path):
    
    with open(pickle_path,"rb") as f:
        training_data_X,training_data_Y,lifted_rw_to_number = pickle.load(f)
        
    return training_data_X, training_data_Y, lifted_rw_to_number

def read_next_input(training_data_X,training_data_Y,example_number,number_of_lifted_rw,number_of_grounding_per_rw,number_of_facts):
    column= numpy.array(training_data_X.coords[0,:] == example_number)
    Data = training_data_X.data[column]
    #print(type(Data)) # 'numpy.ndarray'
    if(Data.size == 0):
        return None, None
    else:
        Y=  training_data_X.coords[:,column]
        
        Z = Y[1:4,:]
       
        indices = numpy.array(Z, dtype=numpy.int64)
        values  = numpy.array(Data,dtype=numpy.float32)
    
        label = numpy.array(training_data_Y[example_number],dtype=numpy.float32)
    
        #print(indices.shape, values.shape, label.shape, number_of_lifted_rw, number_of_grounding_per_rw, number_of_facts)
        training_data_New = sparse.COO(indices, values,shape=(number_of_lifted_rw, number_of_grounding_per_rw, number_of_facts))
        # Sparse.COO is available at http://sparse.pydata.org/en/latest/construct.html
  
        return training_data_New, label


def perform_test_phase(Test_data_X,Test_data_Y,lifted_rw_to_number_test,lifted_rw_to_number_training,train_path,AUC_path,validation):
    
    number_of_test_examples          = int(Test_data_X.shape[0])
    number_of_lifted_rw_test         = int(Test_data_X.shape[1])
    number_of_test_grounding_per_rw  = int(Test_data_X.shape[2])
    number_of_test_facts             = int(Test_data_X.shape[3])
    
   
    x1 = tf.placeholder("float", [number_of_lifted_rw_test, number_of_test_grounding_per_rw, number_of_test_facts], name= "XValueTest")
    Y1 = tf.placeholder("float", [1, 2], name = "YValueTest")
    W_encoder,W_fully_connected = get_weights_for_test_matrix(lifted_rw_to_number_training,lifted_rw_to_number_test,number_of_test_facts,number_of_lifted_rw_test,train_path)
    
    prediction = lifted_relational_neural_network_test(x1, W_encoder,W_fully_connected)
  
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(Y1,1))
    accuracy  = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    total_batch = int(number_of_test_examples)

    count = 0;
    test_accuracy = 0.
    init = tf.global_variables_initializer()
    AUCROC =[]
    
    with tf.Session() as sess:
        sess.run(init)
        for j in range(total_batch):
            training_data_new1, label1 = read_next_input(Test_data_X,Test_data_Y,j,number_of_lifted_rw_test,number_of_test_grounding_per_rw,number_of_test_facts)
           
            if training_data_new1 is None:
                continue
            else:
                myx1 = training_data_new1.todense()
                label2 = numpy.reshape(label1, (1,2))
                accur = sess.run(accuracy, feed_dict={x1: myx1, Y1: label2})  
                test_accuracy += accur
                if(validation != True):
                    pred  = sess.run(prediction, feed_dict={x1: myx1, Y1: label2})
                    #print(j," prediction ",pred, label1, pred[0], label1[0])
                    x_arg0 = "{:.9f}".format(pred[0][0])
                    x_arg1 = "{:.9f}".format(pred[0][1])
                    y_arg0 = int(label2[0][0])
                    y_arg1 = int(label2[0][1])
                    z1 = str(x_arg0)+"\t"+str(x_arg1)+"\t"+ str(y_arg0)+" \t"+str(y_arg1)
                    print(z1+"**")
                    z = str(x_arg1)+" \t"+str(y_arg1)
                    AUCROC.append(z)
                del(myx1)
                count = count +1
        test_accuracy = test_accuracy/count
    
    if(validation !=True):
        path = AUC_path+"/AUCROC.txt";
        f= open(path,'w')
        for i in range(len(AUCROC)):
            f.write(AUCROC[i]+"\n")
        f.close() 
           
    return test_accuracy


# MAIN CODE STARTS HERE ######################################################################################
#basic_path              ="Old_DataSets/CoraSameVenue/5Folds/Fold1"
basic_path              = "Old_DataSets/imdb_small/5Folds/Fold1"
#basic_path              = sys.argv[1]
train_path              = basic_path +"/Training"    
pickle_training_path    = train_path +"/test/training.pickle"
test_path               = basic_path +"/Test"
pickle_test_path        = test_path+"/test/test.pickle"
model_path              = train_path+"/best_model"

#target_predicate  = sys.argv[2]
target_predicate = "workedUnder"
create_pickle_file(train_path,None,target_predicate)    
training_data_X, training_data_Y,lifted_rw_to_number_training = read_input_data(pickle_training_path) 

create_pickle_file(test_path,lifted_rw_to_number_training,target_predicate)  
Test_data_X,Test_data_Y,lifted_rw_to_number_test = read_input_data(pickle_test_path)   

number_of_training_examples   = int(training_data_X.shape[0])
number_of_lifted_rw           = int(training_data_X.shape[1])
number_of_grounding_per_rw    = int(training_data_X.shape[2])
number_of_facts               = int(training_data_X.shape[3]) 


x = tf.placeholder("float", [number_of_lifted_rw, number_of_grounding_per_rw, number_of_facts],name="XValue")
Y = tf.placeholder("float", [1, 2],name="YValue")

W_encoder1 =  define_shared_weights(number_of_lifted_rw,number_of_facts)
logits    =  lifted_relational_neural_network(x, W_encoder1, number_of_facts,number_of_lifted_rw,number_of_grounding_per_rw, 2)

#define the loss function
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

weights   = tf.trainable_variables() 
l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.001, scope=None)
regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

#lossL2 = tf.add_n([ tf.nn.l1_loss(v) for v in vars1 ]) * 0.001

#define training step
#train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)
train_step = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss_op + regularization_penalty)

#define accuracy
prediction = tf.nn.softmax(logits)
#prediction = logits

#correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y,1)) # this throws error
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
accuracy  = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#create a saver
saver = tf.train.Saver()
#train
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver.save(sess,model_path)
        
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/nn_logs",sess.graph)
    print("# of Training Examples ",number_of_training_examples,"# lifted RW: ",number_of_lifted_rw," #grounding: ",number_of_grounding_per_rw,"# facts: ",number_of_facts)
 
    for epoch in range(training_epochs):
        total_batch = int(number_of_training_examples)
        # Loop over all batches
        for i in range(total_batch):
            training_data_new, label = read_next_input(training_data_X,training_data_Y,i,number_of_lifted_rw,number_of_grounding_per_rw,number_of_facts)
            if training_data_new is None:
                continue
            label1 = numpy.reshape(label, (1,2))
            myx = training_data_new.todense()
            _  = sess.run([train_step], feed_dict = {x: myx, Y: label1})
            pred = sess.run(prediction, feed_dict = {x: myx, Y: label1})
           # for v in weights:
            #    if v.name == "SharedVariable/W:0":
             #       print("SharedWeights ")
            #        print(sess.run(v))
             #   if v.name == "FullyConnectedLayer:0":
              #      print("Fully Connected Layer: ")
            #        print(sess.run(v))
            #print(sess.run(weights))
            print("Epoch: ",epoch,"i: ",i," prediction: ", pred," Y: ",label1)
            if epoch == 0:
             #print( " cost:   ",c)
             summary, _ = sess.run([merged, train_step], feed_dict={x: myx, Y: label1})
             writer.add_summary(summary, i)
            #if i%10 == 0:
            saver.save(sess,model_path)
sess.close()      
mystring = perform_test_phase(Test_data_X,Test_data_Y,lifted_rw_to_number_test,lifted_rw_to_number_training,train_path,test_path,False)
print("Final test accuracy"+"={:.9f}".format(mystring))   
auctestpath = test_path+"/AUCROC.txt"
print(auctestpath)
#subprocess.call(['java', '-jar', 'auc.jar',auctestpath,'list'])
process = subprocess.Popen(['java', '-jar', 'auc.jar',auctestpath,'list'],stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
output, error = process.communicate()
lines = output.split(b'\n')
for s in lines:
    if b'Area Under the Curve for ROC' in s:
        c = str(s)
        print(c[2:-3])
    if b'Precision - Recall' in s:
        d = str(s)
        print(d[2:-3])