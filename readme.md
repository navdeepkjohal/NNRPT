## Software Requirements:

(a) Tensorflow 

(b) Sparse package available [here](http://sparse.pydata.org/en/latest/construct.html)


This is the NNRPT model proposed in 'Neural Network for Relational data' paper which is available [here](https://arxiv.org/abs/1909.04723)


The implementation is according to the convolutional neural network conversion as explained in the discussion section of the paper.


In order to run this code, you first need to generate the lifted random walks and followed by
grounding them during roll out of proposed NNRPT network: 

1. Please generate lifted random walks using the following code: https://starling.utdallas.edu/software/boostsrl/wiki/lifted-relational-random-walks/

2. Bring data into the format described in the grounding random walks (https://starling.utdallas.edu/software/boostsrl/wiki/grounded-relational-random-walks/) so that the proposed neural network can generate the groundings of clauses during the rollout of the network.
Please note that you do not need to call the grounded random walk code, NNRPT code will do that internally. This link is a guideline to bring data to specific format accepted by NNRPT.

3. Run the main file Lifted_R_NN_Network.py as
  python Lifted_R_NN_Network.py "path/to/input/file" "targetpredicate"
  
  please contact me for further questions at navdeepkjohal@gmail.com

## Citation
Please cite our paper as:
```bash
@inproceedings{yang-etal-2023-coupling,
    title = "Coupling Large Language Models with Logic Programming for Robust and General Reasoning from Text",
    author = "Yang, Zhun  and
      Ishay, Adam  and
      Lee, Joohyung",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.321",
    doi = "10.18653/v1/2023.findings-acl.321",
    pages = "5186--5219"
}
```
