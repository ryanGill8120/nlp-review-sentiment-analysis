## Sentiment Analysis on IMDb Movies Reviews
![alt text](https://www.theinformationlab.co.uk/wp-content/uploads/2018/12/Sentiment.jpg)

Different NLP techniques have been applied to different datasets for sentiment analysis of movies reviews, online product reviews, etc. These techniques have shown different levels of accuracy on different datasets. Using different datasets, however, makes it difficult to juxtapose these techniques. This presents a need to compare these techniques on the same dataset to better identify which technique is better for the sentiment analysis of movies reviews.
##### In this study, we will compare five NLP techniques for sentiment analysis on IMDb Movies Reviews Dataset:
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

### BERT LARGE
Bidirectional Encoder Representations from Transformers (BERT) is a pre-trained transformer-based machine learning technique for natural language processing (NLP) developed by Google.
BERT was pre-trained on two tasks: language modelling and next sentence prediction.

#### Unsupervised Data Augmentation for Consistency Training Summary
The Unsupervised Data Augmentation (UDA) uses a semi-supervised learning (SSL) method to help address a fundamental weakness in deep learning which is that it typically requires a lot of labeled data to work. Works in SSL based on consistency training have shown to work well on many benchmarks and in order to improve consistency training, high quality data augmentation methods were applied.

#### Architectural Design
![Architectual Design for UDA](/Images/UDA_Architectual_Design.png "Architectual Design for UDA")

##### Main Idea
The predictions of the unlabeled data should align with the predictions of the same unlabeled data after going through data augmentations

#### Augmentationf or Text Classification 
One augmentation method used for text classification was back translation which refers to the procedure of translating an existing example x in language A into another language B and then translating it back into A to obtain an augmented example xˆ. The paraphrases below generated by back-translation sentence are diverse and have similar semantic meanings.
The sentence was translated from English to French and back to English.
![Back Translation](/Images/Back_Translation.png "Back Translation")
For the evaluation on text classification datasets, UDA was combined with BERT, considering four unsipervised shemas: random Transformer, BERT Base, BERT Large, and BERT Finetune. Performances were compared under each of these schemas with and without UDA. 
![UDA IMDB](/Images/UDA_IMDB.png "UDA IMDB")
##### Observations
- On binary sentiment analysis tasks, with only 20 supervised examples, UDA outperforms the previous SOTA trained with full supervised data on IMDb
- When initialized with BERT and further finetuned on in-domain data, UDA can still significantly reduce the error rate from 6.50 to 4.20 on IMDb.

#### Fine-tuning BERT on in-domain unsupervised data
The BERT model was fine-tuned on in-domain unsupervised data using the code released by BERT.
- learning rate of 2e-5, 5e-5 and 1e-4
- batch size of 32, 64 and 128
- number of training steps of 30k, 100k and 300k

Accuracy on IMDb with different number of labeled examples. In the large-data regime, with the full training set of IMDb, UDA also provides robust gains
#####Accuracy with 20 labeled examples: 95.8%
![Accuracy](/Images/UDA_and_BERT_Accuracy.png "Accuracy")

#### Key points of UDA model
- Uses semi-supervised learning method
- Uses SSL based on consistency training
- Substitute traditional noise injection methods with data augmentation methods
- Evaluated on a variety of language and vision tasks

------------

### Modified LMU

Prallelizing legendre memory unit training leverage the linear time-invariant (LTI) memory component of the LMU to construct a simplified variant that can be parallelized during training (and yet executed as an RNN during inference), thus overcoming a well known limitation of training RNNs on GPUs.

#### Main Idea

- Simplifying the LMU such that recurrence exists only in the linear system. 

Inspired by the success of self-attention. Self-attention based architectures have come to replace RNN based approaches for problems such as language modelling, machine translation, and a slew of other NLP tasks (Radford et al., 2018; Raffel et al., 2019). Three properties that make self-attention desirable over RNNs are:
1. it is better at handling the challenging problem of long-range dependencies
2. it is purely feedforward
3. when the sequence length is smaller than the dimension of representation

#### Model

Implement a general affine transformation followed by an element-wise nonlinearity

![Model](https://github.com/ryanGill8120/nlp-review-sentiment-analysis/blob/cb7a78bdfc6dcd47cdb74a1c16ec56f7499719a6/Images/Modified%20LMU%20-%20Model%20General.png "Model")

1. Parallel Training:
 - One of the motivations for the above mentioned architectural changes is that the model now has only one recurrent connection: mt’s dependence on itself from the past. But because this is an LTI system, standard control theory (Astrom & Murray , 2010) gives a non-iterative way of evaluating this equation as shown below

![Parallel Training](https://github.com/ryanGill8120/nlp-review-sentiment-analysis/blob/0e0e9aab9a0f527daf114241c2d62e29f7242fa6/Images/Modified%20LMU%20-%20Model%20Parallel%20Training.png "Parallel Training")

 - It is also evident from the structure of the U matrix that although this reformulation turns the DN into a feedforward layer, it still respects causality. In other words, the state mt depends only on the inputs seen until that point of time

![Parallel Training Updated](https://github.com/ryanGill8120/nlp-review-sentiment-analysis/blob/0e0e9aab9a0f527daf114241c2d62e29f7242fa6/Images/Modified%20LMU%20-%20Model%20Parallel%20Training%202.png "Parallel Training Updated")

2. Complexity: 
 - This can be made more efficient by employing the convolution theorem which gives an equivalent way of evaluating the convolution in the Fourier space as
 
![Complexity](https://github.com/ryanGill8120/nlp-review-sentiment-analysis/blob/0e0e9aab9a0f527daf114241c2d62e29f7242fa6/Images/Modified%20LMU%20-%20Model%20Complexity.png "Complexity")
 
 - It was argued in Vaswani et al. (2017) that a self-attention layer is cheaper than an RNN when the representation dimension of the input, dx, is much greater than the length of the sequence, n, which is seen in NLP applications.

3. Recurrent Inference:


Machine learning algorithms are usually optimized for training rather than deployment (Crankshaw, 2019), and because of that models need to be modified, sometimes non-trivially, to be more suitable for inference.
While this model can be trained in parallel, it can also be run in an iterative manner during inference, and hence can process data in an online or streaming fashion during inference.

#### Experiments

In the following experiments, comparing the model against the LMU, LSTMs and transformers. 

psMNIST: as the name suggests, is constructed by permuting and then flattening the (28 × 28) MNIST images. The permutation is chosen randomly and is fixed for the duration of the task. It uses the standard 50k/10k/10k split.

1. Architecture:
Model uses 165k parameters, original LMU model, which uses 102k parameters, and the HiPPO-LegS model, which is reported to use 512 hidden dimensions

2. Results & Discussion:
Test scores of various models on this dataset are reported in the table below. Model not only surpasses the LSTM model, but also beats the current stateof-the result of 98.3% set by HiPPO-LegS (Gu et al., 2020) recently. Thus, Model sets a new state-of-the art result for RNNs of 98.49% on psMNIST. It is interesting that the model, despite being simpler than the original LMU, outperforms it on this dataset.

![Results](https://github.com/ryanGill8120/nlp-review-sentiment-analysis/blob/0e0e9aab9a0f527daf114241c2d62e29f7242fa6/Images/Modified%20LMU%20-%20Experiment.png "Results")

3. Additional Experiment:
PyTorch implementations for Parallelizing Legendre Memory Unit Training on psMNIST

![Additional Experiment](https://github.com/ryanGill8120/nlp-review-sentiment-analysis/blob/0e0e9aab9a0f527daf114241c2d62e29f7242fa6/Images/Modified%20LMU%20-%20Additional%20Experiment.png "Additional Experiment")

#### Supplementary Materials

![Supplementary Materials](https://github.com/ryanGill8120/nlp-review-sentiment-analysis/blob/0e0e9aab9a0f527daf114241c2d62e29f7242fa6/Images/Modified%20LMU%20-%20Supplementary%20Materials.png "Supplementary Materials")

------------

### Citations

### XLNet 

Yang, Zhilin, et al. "Xlnet: Generalized autoregressive pretraining for language understanding." Advances in neural information processing systems 32 (2019).

Turner, Elliot [@eturner303]. “Holy crap: It costs $245,000 to train the XLNet model (the one that's beating BERT on NLP tasks..512 TPU v3 chips * 2.5 days * $8 a TPU) -”. Twitter, Jun 24 2019, https://twitter.com/eturner303/status/1143174828804857856?lang=en.

Sentiment_Analysis_Movie_Reviews.ipynb. @eugenesiow, commit 0ae04e7. Independent, Dec 22, 2020. GitHub

### GraphStar
Haonan, Lu, et al. "Graph star net for generalized multi-task learning." arXiv preprint arXiv:1906.12330 (2019).
GitHub: https://github.com/graph-star-team/graph_star

### BlockSparse

### BERT Large

[Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/pdf/1904.12848.pdf)
[UDA GitHub](https://github.com/google-research/uda)
[Video presentation: Unsupervised Data Augmentation for Consistency Training](https://papertalk.org/papertalks/8414)
[YouTube video: Unsupervised Data Augmentation](https://www.youtube.com/watch?v=-u8Mi57BDIY)
[Open Sourcing BERT: State-of-the-Art Pre-training for Natural Language Processing](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

### Modified LMU













