# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 20:33:11 2017

@author: nxk176630
"""
import subprocess
from os.path import dirname

def obtain_grounded_random_walks(testpath, modelpath, target):
        return subprocess.call(['java', '-jar', 'GroundedRandomWalks.jar','-grw','-mln','-trees', str(1), '-i', '-test', testpath, '-target', str(target),'-model',modelpath,'-aucJarPath','./'])

def write_to_file(directory, facts):
    with open(directory, 'w') as pos_file:
        for f in facts:
            pos_file.write(f+"\n")

def remove_wrapper_around_grounded_random_walks(filename,numcharacters):
    new_groundings = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.rstrip()
            new_groundings.append(line[1:numcharacters])
    file.close()
    #print(new_groundings,"\n")
    return new_groundings

def read_my_file(filename):
    file_content = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        last = lines[-1]
        for line in lines:
            if line.startswith("//"):
                continue
            else:
                if line is last:
                    if "\n" in line:
                        file_content.append(line[:-1])
                    else:
                        file_content.append(line)
                else:
                    file_content.append(line[:-1])
    file.close()
    return file_content

def create_multimap_equivalent_in_python(Mylist, splitcharacter):
    target_and_body_seperated = dict()
       
    for line in Mylist:
        split_line = line.split(splitcharacter)
        if split_line[0] in target_and_body_seperated:
            target_and_body_seperated[split_line[0]].append(split_line[1])
        else:
        # create a new array in this slot
           target_and_body_seperated[split_line[0]] = [split_line[1]]
  #  for k in target_and_body_seperated.keys():
  #      print("keys: ", k, " value: ",target_and_body_seperated[k],"\n")
    return target_and_body_seperated
 
def create_multimap_for_facts(Mylist, splitcharacter):
    predicate_and_body_seperated = dict()
       
    for line in Mylist:
        split_line = line.split(splitcharacter)
        if split_line[0] in predicate_and_body_seperated:
            predicate_and_body_seperated[split_line[0]].append(line[:-1])
        else:
        # create a new array in this slot
           predicate_and_body_seperated[split_line[0]] = [line[:-1]]
 #   for k in predicate_and_body_seperated.keys():
      # print("keys: ", k, " value: ",predicate_and_body_seperated[k])
    return predicate_and_body_seperated

def extract_predicates_in_order_from_random_walks(randomwalk):
    if (randomwalk.find(":-") !=-1):
        g = randomwalk.find(":-")
        randomwalk = randomwalk[g+4:]
       # print("*",randomwalk)
    split_rw = randomwalk.split("), ")
    resulting_rw =""
    for pred in split_rw:
        split_pred = pred.split("(")
        resulting_rw = resulting_rw+","+split_pred[0]
        #print(split_pred[0])
    return resulting_rw[1:] 
 
def reading_grounded_random_walks(filename):
    RW_without_wrapper = remove_wrapper_around_grounded_random_walks(filename,-1)
    #print(RW_without_wrapper)
    groundedRW = create_multimap_equivalent_in_python( RW_without_wrapper, " :- ")
    return groundedRW

def reading_lifted_random_walks(filename):
    RW_without_wrapper = remove_wrapper_around_grounded_random_walks(filename,-6)
   # print(RW_without_wrapper)
    RW_predicates=[]
    one_RW=""
    for line in RW_without_wrapper:
        one_RW = extract_predicates_in_order_from_random_walks(line)
        RW_predicates.append(one_RW)
        
    return RW_predicates

def invert_one_predicate(facts_to_be_inverted):
    inverted_facts =[]
    
    for line in facts_to_be_inverted:
       # print(line[:-1])
        predicatename = line.split("(")
        constantsname = predicatename[1].split(",")
        inverted_predicate = "_"+ predicatename[0]+"("+constantsname[1][:-1]+","+constantsname[0]+")."
     #   if ")," in inverted_predicate: # last line of the facts file
     #       inverted_predicate = "_"+ predicatename[0]+"("+constantsname[1][:-2]+","+constantsname[0]+")."
        inverted_facts.append(inverted_predicate)
        
    return inverted_facts

def create_inverted_facts(factsfile, backgroundfile, target_predicate):
    
    factsfile =  read_my_file(factsfile)
    facts_dictionary = create_multimap_for_facts(factsfile, "(")
  #  print(facts_dictionary)
    
    with open(backgroundfile,'r') as file:
        for line in file:
            if "import" in line:
                f =backgroundfile.rfind("/")
                g = line.find('"')
                h = line.find('t".')
                backgroundfile = backgroundfile[:f+1]+line[g+1:h+1]
                #print("**"+backgroundfile)
                b = True
                while(b):
                    if "../" in backgroundfile:
                        f = backgroundfile.find("../")
                        h = backgroundfile[f+3:]
                        backgroundfile =  backgroundfile[:f-1]
                        backgroundfile = dirname(backgroundfile)+"/"+h
                    else:
                        b = False
              
            break
    facts_to_be_inverted = [] 
    final_facts =[]
    predicates_under_mode_keyword=[]
    predicates_under_randomwalksconstraint =[]
    facts_inverted_per_predicate=[]
    
    with open(backgroundfile, 'r') as file:
        for line in file:
           if "randomwalkconstraint" in line:
                rw_constraint_index= line.find("randomwalkconstraint")+len("randomwalkconstraint")+1
                h = line[rw_constraint_index:]
                split_line = h.split("=")
                ky = split_line[0][1:]
                predicates_under_randomwalksconstraint.append(ky)
                
                if "NoTwin" in h:
                    continue
                else:
                     facts_to_be_inverted = facts_dictionary.get(ky)
                     if facts_to_be_inverted is None:
                         continue
                     else:
                         facts_inverted_per_predicate = invert_one_predicate(facts_to_be_inverted)
                         final_facts.extend(facts_inverted_per_predicate)
                    
           if "mode" in line:
                rw_constraint_index= line.find("mode")+len("mode")+1
                h = line[rw_constraint_index:]
                split_line = h.split("(")
                ky = split_line[0][1:]
                predicates_under_mode_keyword.append(ky)
    
   # a = set(predicates_under_randomwalksconstraint)    
   # b = set(predicates_under_mode_keyword)
    
   # for line in (b-a):
   #     if line == target_predicate:
   #         continue
   #     else:
   #         facts_to_be_inverted = facts_dictionary.get(line)
   #         if facts_to_be_inverted is None:
   #             continue
   #         else:
   #             facts_inverted_per_predicate = invert_one_predicate(facts_to_be_inverted)
   #             final_facts.extend(facts_inverted_per_predicate)
                                           
    final_facts.extend(factsfile)
    

    return final_facts      

def  bring_atoms_to_grounded_rw_format(All_facts_including_inverted):

    # remove trailing decimal and give space after comma i.e. yearsInProgram(person6, year_5)
    final_facts =[]
    for line in All_facts_including_inverted:
        #print(line)
        #split_line = line.split(",")
        #new_line = split_line[0]+", "+split_line[1][:-1]
        if "," in line:
            line = line.replace(",",", ")
        new_line = line[:-1]
        #print(new_line)
        final_facts.append(new_line)
    return final_facts
        

def hashing_fact_to_number(facts):
    fact_to_number = dict()
    #print(facts)
    count = 0
    for onefact in facts:
                
        fact_to_number[onefact] = count
        count = count + 1
    return fact_to_number
    

def hashing_lifted_rw_to_number(lifted_rw):
    lifted_rw_to_number = dict()
    
    count = 0
    for one_rw in lifted_rw:
        lifted_rw_to_number[one_rw] = count
        count = count + 1
   
    return lifted_rw_to_number 
        
    
# The Main Program's code starts here

def Read_All_the_Data(training_data_path,model_path_for_train_data,target_predicate):

    outputRW = training_data_path+"OutputRW.txt"
    facts_path = training_data_path+"test_facts.txt"
    backgroundfile = training_data_path+"test_bk.txt"
    inverted_facts_file = training_data_path+"test_facts_inv.txt"
    lifted_RW_path = model_path_for_train_data+"bRDNs/Trees/"+target_predicate+"Tree0.tree"
    positive_example_path = training_data_path+"test_pos.txt"
    negative_example_path = training_data_path+"test_neg.txt"
    
    obtain_grounded_random_walks(training_data_path,model_path_for_train_data,target_predicate)
 
    grounded_rw                 =   reading_grounded_random_walks(outputRW)
    
    lifted_rw                   =   reading_lifted_random_walks(lifted_RW_path)
    All_facts_including_inverted=   create_inverted_facts(facts_path, backgroundfile,target_predicate)
    positive_examples           =   read_my_file(positive_example_path)
    negative_examples           =   read_my_file(negative_example_path)
    
    write_to_file(inverted_facts_file,All_facts_including_inverted)
   
    All_facts_including_inverted   =   bring_atoms_to_grounded_rw_format(All_facts_including_inverted)
    positive_examples              =   bring_atoms_to_grounded_rw_format(positive_examples)
    negative_examples              =   bring_atoms_to_grounded_rw_format(negative_examples)

    fact_to_number =      hashing_fact_to_number(All_facts_including_inverted)

    lifted_rw_to_number = hashing_lifted_rw_to_number(lifted_rw)
    
    #print("Lifted RW to Number: ",lifted_rw_to_number)
    #print("Facts To Number ", fact_to_number)
    #print("Positive Examples ", positive_examples)
    #print("Negative Examples ", negative_examples)
    #print("Grounded RW ", grounded_rw)
    
    return grounded_rw, fact_to_number, lifted_rw_to_number, positive_examples, negative_examples

#Read_All_the_Data("Old_DataSets/imdb_small/5Folds/Fold1/Training/test/","Old_DataSets/imdb_small/5Folds/Fold1/Training/train/models/","workedUnder")
