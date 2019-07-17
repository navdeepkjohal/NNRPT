# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:38:06 2017

@author: nxk176630
"""


import tensorflow as tf
import numpy as np
import pickle



MyList=[]


def write_to_file(directory, facts):
    with open(directory, 'w') as pos_file:
        for f in facts:
            pos_file.write(f+"\n")

def read_test_data(test_path):
    pickle_path  = test_path+"/test/test.pickle"
    with open(pickle_path,"rb") as f:
        Test_data_X,Test_data_Y,lifted_rw_to_number = pickle.load(f)
        
    return Test_data_X,Test_data_Y,lifted_rw_to_number

def invert_rw_mapping(lifted_rw_to_number):
    number_to_lifted_rw = dict((v, k) for k, v in lifted_rw_to_number.items())
    
    return number_to_lifted_rw

def perform_combining_rules(tensor_input):
   #return tf.reduce_sum(tensor_input, 1, keepdims=True)
      return tf.reduce_mean(tensor_input, 1, keepdims=True) 


def define_input_hidden_products(input_layer, hidden_layer):
    return (tf.matmul(input_layer, hidden_layer)) 

def get_weights_for_lifted_rw(lifted_rw_to_number,train_path):
    
    rw_to_weight_mapping = dict()   
    number_to_lifted_rw = invert_rw_mapping(lifted_rw_to_number)
    
    new_graph = tf.Graph()
    with tf.Session(graph=new_graph) as sess1:
        saver = tf.train.import_meta_graph(train_path+'/best_model.meta')
        saver.restore(sess1, tf.train.latest_checkpoint(train_path))   
        W1 = new_graph.get_tensor_by_name('WeightsMatrix:0')
        W_fully_connected = new_graph.get_tensor_by_name('FullyConnectedLayer:0')
        
        Dim0 = W1.get_shape()[0]
        
        B = sess1.run(W_fully_connected)
    
        for i in range(Dim0):
            rw                       = number_to_lifted_rw[i]
            weight                   = sess1.run(W1[i][0][0])
         #   weight1                  = sess.run(W1[i][1][0])
        #    weight2                  = sess.run(W1[i][2][0])
          #  print("RW No: ",i," Random Walk: ",rw, "Weight: ",weight)
        #    print("weights ",weight," ",weight1," ",weight2)
            rw_to_weight_mapping[rw] = weight
    sess1.close()
    tensor_weights = tf.convert_to_tensor(B, dtype=tf.float32) 
#    pickle_path                            = "./weights_for_rw.pickle"
#    with open(pickle_path, "wb") as f:
#         pickle.dump((rw_to_weight_mapping), f)  
         
    return tensor_weights, rw_to_weight_mapping

def get_weights_for_test_matrix(lifted_rw_to_number_training,lifted_rw_to_number_test,number_of_test_facts,number_of_lifted_rw,train_path):
    
    W_fully_connected1, rw_to_weight_mapping = get_weights_for_lifted_rw(lifted_rw_to_number_training,train_path)
    
    #with open("./weights_for_rw.pickle","rb") as f:
    #    rw_to_weight_mapping  = pickle.load(f)
    number_to_lifted_rw_test = dict((v, k) for k, v in lifted_rw_to_number_test.items())
       
    Weights_test = np.array(np.zeros((1,number_of_lifted_rw)))
    
    for i in range(len(number_to_lifted_rw_test)):
        one_test_rw = number_to_lifted_rw_test.get(i)
        
        if rw_to_weight_mapping.get(one_test_rw) is None:
            print("Training and Test Lifted RW do not match")
        else:
           j = lifted_rw_to_number_training.get(one_test_rw)
           Weights_test[0,j] = rw_to_weight_mapping.get(one_test_rw)
           #print("Test RW id: ",i," Train RW id: ",j, " RW : ",one_test_rw, " Weight ",Weights_test[0,j])
           tensor_weights = tf.convert_to_tensor(Weights_test, dtype=tf.float32)    
    
    encoder_test = []
    for i in range(number_of_test_facts):
        encoder_test.append(tensor_weights)
    
    W_encoder_test = tf.convert_to_tensor(encoder_test, dtype=tf.float32,name='MyTestWeights')
    W1_test        = tf.transpose(W_encoder_test,perm = [2, 0 ,1], name = 'TestWeightsMatrix')
    
    """Dim0 = W1_test.get_shape()[0]
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)   
        for i in range(Dim0):
            weight                   = sess.run(W1_test[i][0][0])
            print("i :",i," ",weight)
    
    new_graph = tf.Graph()
    with tf.Session(graph=new_graph) as sess1:
        saver = tf.train.import_meta_graph('best_model.meta')
        saver.restore(sess1, tf.train.latest_checkpoint('./'))   
        W1 = new_graph.get_tensor_by_name('WeightsMatrix:0')
              
        Dim1 = W1.get_shape()[0]
        
        for j in range(Dim1):
            weight1                   = sess1.run(W1[j][0][0])
            print("i :",j," ",weight1)
    
    """

    return (W1_test, W_fully_connected1)

def lifted_relational_neural_network_test(x1,W_encoder1,W_fully_connected1):
    
   # W_encoder,W_fully_connected = get_weights_for_test_matrix(lifted_rw_to_number_training,lifted_rw_to_number_test,number_of_facts,lifted_rw)
    
   #W_hidden = define_input_hidden_products(x, W_encoder, lifted_rw, Maximum_Groundings, number_of_facts)
   
   # print("encoder ***",W_encoder.shape)
   # print("data ***", x.shape)
    W_hidden = define_input_hidden_products(x1, W_encoder1)
    
  #  W_hidden_after_sigmoid = tf.nn.sigmoid(W_hidden)
    W_hidden_after_sigmoid = W_hidden
    
    W_lifted_RW_layer = perform_combining_rules(W_hidden_after_sigmoid)
    
    #w_output = tf.Variable(tf.random_uniform([lifted_rw, Output_Layer], dtype=tf.float32))
    
    Input_provided_to_Output_layer1 = tf.tensordot(W_lifted_RW_layer,W_fully_connected1, axes = [[0], [0]])
    
    #Input_provided_to_Output_layer1 = tf.reshape(Input_provided_to_Output_layer,[-1])
    Input_provided_to_Output_layer2 = tf.reshape(tf.reshape(Input_provided_to_Output_layer1,[-1]), [1,2]) 
    
    output = tf.nn.softmax(Input_provided_to_Output_layer2)
    
    return output



###########################Main code starts here ####################################################################



########################################################################################################            
            
   #fc7= graph.get_tensor_by_name('WeightsMatrix:0')
   #for op in tf.get_default_graph().get_operations():
   #   mylist.append(str(op.name)) 
   #write_to_file("./temp.txt",mylist)
   
   #print(tf.trainable_variables())
   
 #    print(sess.run('WeightsMatrix:0'))
 
 #fc7= graph.get_tensor_by_name('SharedVariable/W:0')
 ######################################################################################
 #W1 = new_graph.get_tensor_by_name('WeightsMatrix:0')
 #  fc7= W1.get_shape().as_list()
 # print(fc7)
 
#   Dim0 = W1.get_shape()[0]
#   print(Dim0)      #2149
 
 #####################################################################################
# print(sess.run('WeightsMatrix:0'))
 ######################################################################################
# W1 = new_graph.get_tensor_by_name('WeightsMatrix:0')
#   print(W1)
 ##################################################################################
 # count=0
#   for i in range(Dim0):
 #      count = count+1
  #     print(count, sess.run(W1[i][0][0]), sess.run(W1[i][1][0]), sess.run(W1[i][2][0]))
######################################################################################
#  Mystr = "i: "+str(i)+str(rw)+str(weight)
#            MyList.append(Mystr)
#    write_to_file("./weights_for_rw.txt",MyList)
#####################################################################################
        # ky =  rw_to_weight_mapping.keys()
        
    #    for k in ky:
     #       print(k," : ",rw_to_weight_mapping[k])
###################################################################################
#print(Weights_test.shape)

#with tf.Session() as sess:
#     print(sess.run(tensor_weights.get_shape()))