In order to run this code, you first need to generate the lifted random walks and followed by
grounding them during roll out of proposed NNRPT network: 

1. Please generate lifted random walks using the following code: https://starling.utdallas.edu/software/boostsrl/wiki/lifted-relational-random-walks/

2. Bring data into the format described in the grounding random walks (https://starling.utdallas.edu/software/boostsrl/wiki/grounded-relational-random-walks/) so that the proposed neural network can generate the groundings of clauses during the rollout of the network.
Please note that you do not need to call the grounded random walk code, NNRPT code will do that internally. This link is a guideline to bring data to specific format accepted by NNRPT.

3. Run the main file Lifted_R_NN_Network.py as
  python Lifted_R_NN_Network.py "path/to/input/file" "targetpredicate"
  
  please contact Navdeep.Kaur@utdallas.edu
