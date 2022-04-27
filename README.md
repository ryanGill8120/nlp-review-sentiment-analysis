## Sentiment Analysis on IMDb Movies Reviews
![alt text](https://www.theinformationlab.co.uk/wp-content/uploads/2018/12/Sentiment.jpg)

Different NLP techniques have been applied to different datasets for sentiment analysis of movies reviews, online product reviews, etc. These techniques have shown different levels of accuracy on different datasets. Using different datasets, however, makes it difficult to juxtapose these techniques. This presents a need to compare these techniques on the same dataset to better identify which technique is better for the sentiment analysis of movies reviews.
##### In this study, we will compare five NLP techniques for sentiment analysis:
1. XLNet
2. GraphStar
3. BlockSparse
4. Bert Large
5. Modified LMU

------------

### XLNet

XLNet seeks to combine the best of Auto-Regressive (AR) and Auto-Encoded (AE) models by implementing a high-level permutation technique during pre-training which makes it capable of higher accuracy on many NLP tasks, famously being one of the first models to outcompete BERT.

The topics discussed on XLNet are:
- The general idea of the permuatation approach
- Model Architecture challenges and solutions
- Research and Third-Party results on the IMDB dataset
- Novelty and Cost

### Permuting Factorizations in XLNet:

![XLNet Permutation Example](/Images/XLNet_Factorization.png "XLNet Permutation Example")

During the tokenization of pre-training inputs using AR techniques, the model is typically allowed to view previous tokens as it makes the prediction on its current factorization. Conversely, AE techniques mask out a small percentage of tokens during the same phase of the process as the model is being trained. XLNet seeks to combine these strategies by considering *all permuations* of a given factorization.

For instance using the image above, the model is allowed access to the data from the previous factorization. So when trying to predict x<sub>3</sub> a normal AR factorization would simply be 1 -> 2 -> 3 -> 4. However XLNet will consider all permutations for a factorization (we don't even have all of them in the image) as the prediction is passed through its layers before making a prediction for that particular node. Naturally, making a prediction given the permutation in the top-left of the image will be difficult, however, the model is allowed more data in other permutations, thereby fine-tuning its ability to make an accurate prediction.  

### XLNet Model Architecture:

In the research, almost all general architectural choices were made with hopes of using BERT models as a standard of comparison. Thus, XLNet chose not only to base their augmentation on existing BERT and AR strategies, but even implemented their model with the same number of layers as BERT.

The main challenges presented by XLNet:
- Level of permutation is costly
- Need to make a prediction on a token while using previous factorization predictions in *only one pass through the model*

Though the costliness of XLNet can really only be solved by hardware and time, the research team employed an engineering approach to solve the second problem. They call it a *Masked Two-Stream Attention* wherein they implement two hidden layers in their transformers instead of the traditional one. The difference being on (h) is initialized with the embedding of tokens at and before its position, and the second (g) is not initialized immediatly and is only allowed to look at tokens from previous layers/factorizations. 

### XLNet Results with IMDB

Because of the permutative nature of the model, training the full XLNet requires significant resources (discussed in next section). So in regards to IMDB sentiment analysis, the research repository supplies a python notebook that reports the following:

- *Accuracy: 92.416%*
- *eval_loss .31708*

After training with code developed in notebook setting for IMDB by Euguene Siow, the following results were achieved:

- *Accuracy: 92.156%*
- *eval_loss* .38349*

The full model was tested on multiple datasets, but they reported their state-of-the-art error on IMDB as follows:

![XLNet IMDB Error](/Images/XLNet_err.png "XLNet IMDB Error")

### Cost and Novelty of XLNet

XLNet is a recent collaboration between researchers from Carnegie Mellon University and the Google AI Brain team only being published in January of 2020, this leads to a few possible pitfalls in the research:
- *Actual Impact of the Permuatation Technique:* The model combines many different techniques from different models and even has a section in it's paper exploring how much of an impact the novel technique has
- *Cost to Train:* As tweeted by Eugene Siow, some estimates put the cost of training the full XLNet model at a quarter of a million dollars
- *Difficulty in Reproduction:* Because of cost and time constraints, the paper and Third-party training involve incomplete models that do not achieve full accuracy described in the research

Hopefully with more time and research, XLNet can be proven as an effective and accessible NLP model.

![XLNet Cost](/Images/XLNet_Cost.png "XLNet Cost")

------------

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

------------

### BlockSparse

------------

### BERT Large

------------

### Modified LMU

------------

### Citations

### XLNet 

Yang, Zhilin, et al. "Xlnet: Generalized autoregressive pretraining for language understanding." Advances in neural information processing systems 32 (2019).

Turner, Elliot [@eturner303]. “Holy crap: It costs $245,000 to train the XLNet model (the one that's beating BERT on NLP tasks..512 TPU v3 chips * 2.5 days * $8 a TPU) -”. Twitter, Jun 24 2019, https://twitter.com/eturner303/status/1143174828804857856?lang=en.

Sentiment_Analysis_Movie_Reviews.ipynb. @eugenesiow, commit 0ae04e7. Independent, Dec 22, 2020. GitHub

### GraphStar

### BlockSparse

### BERT Large

### Modified LMU













