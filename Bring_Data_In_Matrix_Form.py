# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 19:20:02 2017

@author: nxk176630
"""

from File_Handling_Lifted_RW import Read_All_the_Data,extract_predicates_in_order_from_random_walks
import random
import math
import numpy
import pickle
import sparse

training_to_validation_ratio = 0.8
number_of_classes  = 2
grw_to_facts_ratio = 0.1


def partition_training_validation_set(positive_examples,negative_examples):
    AllExamples = positive_examples + negative_examples
    random.shuffle(AllExamples)

    number_of_total_examples = len(AllExamples)
    number_of_training_examples = int(math.ceil(number_of_total_examples*training_to_validation_ratio))

    training_examples = AllExamples[0:number_of_training_examples-1]
    validation_examples = AllExamples[number_of_training_examples: number_of_total_examples]
    
    return training_examples, validation_examples

def obtain_examples(positive_examples, negative_examples):
    AllExamples = positive_examples + negative_examples
    random.shuffle(AllExamples)
    
    return AllExamples

def get_facts_index_from_grounded_rw(facts_to_number,one_grw):
    
    facts_index = []
    each_fact = one_grw.split("), ")
    
    last = each_fact[-1]
    each_fact = [x+ ")" for x in each_fact]
    each_fact[-1] = last
    
    
    #for x in each_fact:
    #    print("*"+x)
    facts_index = [facts_to_number[x] for x in each_fact]
      
    return facts_index
 
"""        
def get_the_next_example_in_matrix_form(X, example,example_number, grounded_rw,fact_to_number,lifted_rw_to_number):
    
    facts_rw_index =[]
   # number_of_grounding_per_rw =int(math.ceil(len(fact_to_number)*grw_to_facts_ratio)) 
    grounded_rw_for_current_example = grounded_rw.get(example)
    
   # print(number_of_grounding_per_rw,"*" ,len(grounded_rw_for_current_example))
#    if(len(grounded_rw_for_current_example) > number_of_grounding_per_rw):
    # if number of groundings are greater than maximum allowed groundings, then sample
#    grounded_rw_for_current_example =  random.sample(grounded_rw_for_current_example, number_of_grounding_per_rw)
     
    count = 0; #stores the grounded random walk index (dimension 1)
    for one_grw in grounded_rw_for_current_example:
       lifted_rw       = extract_predicates_in_order_from_random_walks(one_grw)
       lifted_rw_index = lifted_rw_to_number[lifted_rw]
       facts_rw_index  = get_facts_index_from_grounded_rw(fact_to_number,one_grw)
       for fact_index in facts_rw_index:
           temp = [example_number,lifted_rw_index,count,fact_index]
           X.append(temp)
       count = count+1

    return X
"""

       
def get_the_next_example_in_matrix_form(X, example,example_number, grounded_rw,fact_to_number,lifted_rw_to_number):
    
    facts_rw_index =[]
   # number_of_grounding_per_rw =int(math.ceil(len(fact_to_number)*grw_to_facts_ratio)) 
    grounded_rw_for_current_example = grounded_rw.get(example)
    
    max_grounded_RW_each_example = dict.fromkeys(lifted_rw_to_number)
    
    for k in max_grounded_RW_each_example:
            max_grounded_RW_each_example[k] = 0
    
   # print(number_of_grounding_per_rw,"*" ,len(grounded_rw_for_current_example))
#    if(len(grounded_rw_for_current_example) > number_of_grounding_per_rw):
    # if number of groundings are greater than maximum allowed groundings, then sample
#    grounded_rw_for_current_example =  random.sample(grounded_rw_for_current_example, number_of_grounding_per_rw)
     
    #count = 0; #stores the grounded random walk index (dimension 1)
    for one_grw in grounded_rw_for_current_example:
       lifted_rw       = extract_predicates_in_order_from_random_walks(one_grw)
       count           = max_grounded_RW_each_example[lifted_rw]
       
       lifted_rw_index = lifted_rw_to_number[lifted_rw]
       facts_rw_index  = get_facts_index_from_grounded_rw(fact_to_number,one_grw)
       for fact_index in facts_rw_index:
           temp = [example_number,lifted_rw_index,count,fact_index]
           X.append(temp)
       count = count+1
       max_grounded_RW_each_example[lifted_rw] = count
    return X

def count_examples_with_non_zero_groundings(training_examples, grounded_rw):
    example_number = 0;
    for example  in training_examples:
         gr = grounded_rw.get(example)
         if gr is None:
            print(example,"**")
         else:
            example_number = example_number+1
    return example_number

"""
def find_grounded_rw_dimension_size(training_examples, grounded_rw):
    number_of_grounding_per_rw = -1;
    
    for example in training_examples:
        grounded_rw_for_current_example = grounded_rw.get(example)
        
        if grounded_rw_for_current_example is None:
            continue
        else:
            if(len(grounded_rw_for_current_example) > number_of_grounding_per_rw):
                number_of_grounding_per_rw = len(grounded_rw_for_current_example) #dim 2
    print("$$", number_of_grounding_per_rw)         
    return number_of_grounding_per_rw

"""
def find_grounded_rw_dimension_size(training_examples, grounded_rw, lifted_rw_to_number):
   
    number_of_grounding_per_rw = -1;
    
    for example in training_examples:
        
        max_grounded_RW_each_example = dict.fromkeys(lifted_rw_to_number)
    
        for k in max_grounded_RW_each_example:
            max_grounded_RW_each_example[k] = 0
           
        grounded_rw_for_current_example = grounded_rw.get(example)
        
        if grounded_rw_for_current_example is None:
            continue
        else:
            for one_grw in grounded_rw_for_current_example:
                lifted_rw       = extract_predicates_in_order_from_random_walks(one_grw)
                count           = max_grounded_RW_each_example[lifted_rw]
                max_grounded_RW_each_example[lifted_rw] = count + 1
            temp = max_grounded_RW_each_example[max(max_grounded_RW_each_example, key = max_grounded_RW_each_example.get)]
           
            if(temp > number_of_grounding_per_rw):
                number_of_grounding_per_rw = temp #dim 2
                
    #print("$$",number_of_grounding_per_rw)         
    return number_of_grounding_per_rw

def create_dataset(training_examples, positive_examples, negative_examples, grounded_rw, fact_to_number,lifted_rw_to_number):
    
    number_of_training_examples = len(training_examples)        #dim 0 for training data    
    #number_of_validation_examples = len(validation_examples)    #dim 0 for validation data
    number_of_lifted_rw        =len(lifted_rw_to_number)         #dim 1
    #number_of_grounding_per_rw = math.ceil(len(fact_to_number)*grw_to_facts_ratio)  #dim 2
    number_of_facts            =len(fact_to_number)             #dim 3

    training_data_X_index=[]
    training_data_Y = numpy.zeros(shape=(number_of_training_examples, number_of_classes))
#    training_data_X = numpy.zeros(shape=(number_of_training_examples, number_of_lifted_rw, number_of_grounding_per_rw, number_of_facts))
    # dim = number_of_examples X number_of_lifted_rw X number_of_grounding_per_rw X number_of_facts
    
  #  number_of_grounding_per_rw = find_grounded_rw_dimension_size(training_examples, grounded_rw)
    number_of_grounding_per_rw = find_grounded_rw_dimension_size(training_examples, grounded_rw, lifted_rw_to_number)      
    example_number = 0;
    for example  in training_examples:
        gr = grounded_rw.get(example)
        if gr is None:
            continue
        else:
            training_data_X_index = get_the_next_example_in_matrix_form(training_data_X_index, example,example_number ,grounded_rw, fact_to_number, lifted_rw_to_number)
            #print(example,"* ",len(training_data_X_index))
            if example in positive_examples:
                training_data_Y[example_number]  = [0,1]
              #  print("**",training_data_Y[example_number])
            else:
                
                training_data_Y[example_number]  = [1,0]
             #   print("&&",training_data_Y[example_number])
                
        example_number = example_number + 1
    
    data_size = len(training_data_X_index)
    data      = numpy.ones(data_size)
    
    B = numpy.asarray(training_data_X_index,dtype=numpy.int32)
    C = B.transpose()
   # print(C.shape,data.shape)
    
    training_data_X = sparse.COO(C,data,shape=(number_of_training_examples, number_of_lifted_rw, number_of_grounding_per_rw, number_of_facts))
    # Sparse.COO is available at http://sparse.pydata.org/en/latest/construct.html
 #   print("number_of_training_examples*",number_of_training_examples)
  #  print(type(training_data_X))
   # print(training_data_Y.shape)
    
    return training_data_X, training_data_Y


########## The Main Function ##########################

def create_pickle_file(filepath,lifted_rw_to_number_training, target_predicate): 
   # training_data_path =        "UWCSE_ILP2016/5Folds/Fold1/Test/test/"
    training_data_path =        filepath+"/test/"
    model_path_for_train_data = filepath+"/train/models/"
    #model_path_for_train_data = "UWCSE_ILP2016/5Folds/Fold1/Test/train/models/"
   # target_predicate = "advisedBy"
    
    training_flag = False
    if "Training" in training_data_path:
        training_flag = True
    else:
        training_flag = False

    grounded_rw, fact_to_number, lifted_rw_to_number1, positive_examples, negative_examples = Read_All_the_Data(training_data_path,model_path_for_train_data,target_predicate)
    
    if lifted_rw_to_number_training is None:
        lifted_rw_to_number = lifted_rw_to_number1
    else:
        lifted_rw_to_number = lifted_rw_to_number_training
#print(fact_to_number)
    if(training_flag):
        training_examples                      = obtain_examples(positive_examples, negative_examples)
        training_data_X, training_data_Y       = create_dataset(training_examples, positive_examples, negative_examples, grounded_rw, fact_to_number, lifted_rw_to_number)
       #storing the dataset as pickle
        
        pickle_path                            = training_data_path+"training.pickle"
        with open(pickle_path, "wb") as f:
            pickle.dump((training_data_X,training_data_Y,lifted_rw_to_number), f)
    else:
        test_examples = positive_examples + negative_examples
        random.shuffle(test_examples)
        Test_data_X, Test_data_Y = create_dataset(test_examples, positive_examples, negative_examples, grounded_rw, fact_to_number,lifted_rw_to_number)
              
        pickle_path                            = training_data_path+"test.pickle"
        with open(pickle_path, "wb") as f:
            pickle.dump((Test_data_X,Test_data_Y,lifted_rw_to_number), f)
        
#create_pickle_file("New_DataSets/UWCSE/20Folds/Fold20/Training")
