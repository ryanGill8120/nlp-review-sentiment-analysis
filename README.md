## Sentiment Analysis on IMDb Movies Reviews
![alt text](https://www.theinformationlab.co.uk/wp-content/uploads/2018/12/Sentiment.jpg)

Different NLP techniques have been applied to different datasets for sentiment analysis of movies reviews, online product reviews, etc. These techniques have shown different levels of accuracy on different datasets. Using different datasets, however, makes it difficult to juxtapose these techniques. This presents a need to compare these techniques on the same dataset to better identify which technique is better for the sentiment analysis of movies reviews.
##### In this study, we will compare five NLP techniques for sentiment analysis:
1. XLNet
2. GraphStar
3. BlackSparse
4. Bert Large
5. Modified LMU

------------

### XLNet

XLNet seeks to combine the best of Auto-Regressive (AR) and Auto-Encoded (AE) models by implementing a high-level permutation technique during pre-training which trains it into a state of capable of higher accuracy on many NLP tasks, famously being one of the first models to outcompete BERT.
The topics discussed on XLNet are:
- The general idea of the permuatation approach
- Model Architecture challenges and solutions
- Research and Third-Party results on the IMDB dataset
- Novelty and Cost

### Permuting Factorizations in XLNet:
![XLNet Permutation Example] (/Images/)

### GraphStar
GraphStar is able to map the global state effectively without incurring system overhead and heavy computation costs.
GraphStar trains the model to: 
- Perform inductive tasks on previously unseen graph data 
- Aggregate both local and long-range information, making the model globally aware, extracting high-level abstraction typically not represented in individual node features
- The relays serve as a hierarchical representation of the graphs and can be used directly for graph classification tasks.

GraphStar can be used in three general graph tasks like:
1. *Node classification*: predict a property of a node.
2.	*Graph classification*: categorize different graphs. 
3.	*Link prediction*: predict whether there are missing links between two nodes.

### GraphStar Model Architecture:
- Step 1: Initial Representation of the Star
- Step 2: Real Node Update
- Step 3: Star Update

![GraphStar Model Archotecture](/Images/graphstar_arch.jpg "GraphStar Model Architecture")

### Running GraphStar on IMDb:
GraphStar propose a new method to tackle sentiment analysis based on node classification and use IMDB-binary dataset as an example. This dataset was originally 4 not a graph task; it is usually treated as a natural language processing task (NLP). 

GraphStar turns the pure NLP task into a graph data task based on document embedding. First, for IMBD, the model uses a pre-trained large Bert model to document encoding, and treats every film review as a node in a graph. Then it links the nodes (film reviews) which belong to the same topic and create a graph dataset. This approach is highly generalizable to most topic-specific classification tasks.

|  # Nodes  | # Features  | #Classes  |  # Training Nodes   | #Validation Nodes  | # Test Nodes |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| 50,000  |  1024 |  2 |  23,003 (12 graphs ) | 1997 (1 graph)  | 25,000 (13 graphs) |  |  

### GraphStar Accuracy:
![GraphStar Accuracy](/Images/graphstar_acc.jpg "GraphStar Accuracy")













