# Stanford CS224n Natural Language Processing with Deep Learning

The course notes about Stanford CS224n Winter 2019 (using PyTorch)

> Some general notes I'll write in [my Deep Learning Practice repository](https://github.com/daviddwlee84/DeepLearningPractice)

Course Related Links

* [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
* [Lecture Videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)
* [Course contents backup](https://github.com/zhanlaoban/CS224N-Stanford-Winter-2019)
* Others' answer
  * [Luvata/CS224N-2019: My completed implementation solutions for CS224N 2019](https://github.com/Luvata/CS224N-2019)
  * [Observerspy/CS224n](https://github.com/caijie12138/CS224n-2019) (not fully 2019)
  * [ZacBi/CS224n-2019-solutions](https://github.com/ZacBi/CS224n-2019-solutions) (didn't finish the written part)
  * [caijie12138/CS224n-2019](https://github.com/caijie12138/CS224n-2019) (not quite the assignment)
  * [ZeyadZanaty/cs224n-assignments](https://github.com/ZeyadZanaty/cs224n-assignments) (just coding part assignment 2, 3)

## Schedule

| Week             | Lectures                                                                                                                                               | Assignments                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| 2019/7/1~7/7     | [Introduction and Word Vectors](#Lecture-1-Introduction-and-Word-Vectors), [Word Vectors 2 and Word Senses](#Lecture-2-Word-Vectors-2-and-Word-Senses) | [Assignment 1](#Assignment-1-Exploring-Word-Vectors) |
| 2019/7/8~7/14    | [Word Window Classification, Neural Networks, and Matrix Calculus](#Lecture-3-Word-Window-Classification-Neural-Networks-and-Matrix-Calculus)          | -                                                    |
| 2019/7/15~7/21   | [Backpropagation and Computation Graphs](#Lecture-4-Backpropagation-and-Computation-Graphs)                                                            | [Assignment 2](#Assignment-2-word2vec)               |
| 2019/10/21~10/27 | [Linguistic Structure: Dependency Parsing](#lecture-5-linguistic-structure-dependency-parsing)                                                         | -                                                    |
| 2019/10/28~11/3  | -                                                                                                                                                      | [Assignment 3](#Assignment-3-Dependency-Parsing)     |

Lecture

1. [X] Introduction and Word Vectors
2. [X] Word Vectors 2 and Word Senses
3. [X] Word Window Classification, Neural Networks, and Matrix Calculus
4. [X] Backpropagation and Computation Graphs
5. [ ] Linguistic Structure: Dependency Parsing
6. [ ] The probability of a sentence? Recurrent Neural Networks and Language Models
7. [ ] Vanishing Gradients and Fancy RNNs
8. [ ] Machine Translation, Seq2Seq and Attention
9. [ ] Practical Tips for Final Projects
10. [ ] Question Answering and the Default Final Project
11. [ ] ConvNets for NLP
12. [ ] Information from parts of words: Subword Models
13. [ ] Modeling contexts of use: Contextual Representations and Pretraining
14. [ ] Transformers and Self-Attention For Generative Models
15. [ ] Natural Language Generation
16. [ ] Reference in Language and Coreference Resolution
17. [ ] Multitask Learning: A general model for NLP?
18. [ ] Constituency Parsing and Tree Recursive Neural Networks
19. [ ] Safety, Bias, and Fairness
20. [ ] Future of NLP + Deep Learning

Assignment

1. [X] Exploring Word Vectors
2. [X] word2vec
   1. [X] code
   2. [X] written

Paper reading

* [ ] word2vec
* [ ] negative sampling
* [ ] GloVe
* [ ] improveing distrubutional similarity
* [ ] embedding evaluation methods

Derivation

* [ ] backprop

### Lectures

#### Lecture 1: Introduction and Word Vectors

* [slides](CourseMaterials/slides/cs224n-2019-lecture01-wordvecs1.pdf)
* [notes](CourseMaterials/notes/cs224n-2019-notes01-wordvecs1.pdf)
* readings
  * [ ] [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
  * [ ] [Efficient Estimation of Word Representations in Vector Space (original word2vec paper)](CourseMaterials/readings/word2vec.pdf)
  * [ ] [Distributed Representations of Words and Phrases and their Compositionality (negative sampling paper)](CourseMaterials/readings/NegativeSampling.pdf)
* [Gensim example](CourseMaterials/GensimWordVectorVisualization.ipynb)
  * preparing embedding: download [this](https://nlp.stanford.edu/data/glove.6B.zip) zip file and unzip the `glove.6B.*d.txt` files into `embedding/GloVe` directory

Outline

* Introduction to Word2vec
  * objective function
  * prediction function
  * how to train it
* Optimization: Gradient Descent & Chain Rule

#### Lecture 2: Word Vectors 2 and Word Senses

* [slides](CourseMaterials/slides/cs224n-2019-lecture02-wordvecs2.pdf)
* [notes](CourseMaterials/notes/cs224n-2019-notes02-wordvecs2.pdf)
* readings
  * [ ] [GloVe: Global Vectors for Word Representation (original GloVe paper)](CourseMaterials/readings/glove.pdf)
  * [ ] [Improving Distributional Similarity with Lessons Learned from Word Embeddings](CourseMaterials/readings/ImprovingDistributionalSimilarity.pdf)
  * [ ] [Evaluation methods for unsupervised word embeddings](CourseMaterials/readings/EmbeddingEvaluationMethods.pdf)
* additional readings
  * [ ] [A Latent Variable Model Approach to PMI-based Word Embeddings](CourseMaterials/additional/PMI-basedWordEmbeddings.pdf)
  * [ ] [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](CourseMaterials/additional/LinearAlgebraicStructureOfWordSenses.pdf)
  * [ ] [On the Dimensionality of Word Embedding](CourseMaterials/additional/DimensionalityOfWordEmbedding.pdf)

Outline

* More detail to Word2vec
  * Skip-grams (SG)
  * Continuous Bag of Words (CBOW)
* Similarity visualization
* Co-occurrence matrix + SVD (LSA) vs. Embedding
* Evaluation on word vectors
  * Intrinsic
  * Extrinsic

#### Lecture 3: Word Window Classification, Neural Networks, and Matrix Calculus

* [slides](CourseMaterials/slides/cs224n-2019-lecture03-neuralnets.pdf)
  * [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/)
* [matrix calculus](CourseMaterials/other/gradient-notes.pdf)
* [notes](CourseMaterials/notes/cs224n-2019-notes03-neuralnets.pdf)
* readings
  * [ ] [CS231n notes on backprop](https://cs231n.github.io/optimization-2/)
  * [ ] [Review of differential calculus](CourseMaterials/other/review-differential-calculus.pdf)
* additional readings
  * [ ] [Natural Language Processing (Almost) from Scratch](CourseMaterials/other/NLPfromScratch.pdf)

Outline

* Some basic idea of NLP tasks
* Matrix Calculus
  * Jacobian Matrix
  * Shape convention
* Loss
  * Softmax
  * Cross-entropy

#### Lecture 4: Backpropagation and Computation Graphs

* [slides](CourseMaterials/slides/cs224n-2019-lecture04-backprop.pdf)
* [notes](CourseMaterials/notes/cs224n-2019-notes03-neuralnets.pdf) - same as lecture 3
* readings
  * [ ] [CS231n notes on network architectures](https://cs231n.github.io/neural-networks-1/)
  * [ ] [Learning Representations by Backpropagating Errors](CourseMaterials/other/backprop_old.pdf)
  * [ ] [Derivatives, Backpropagation, and Vectorization](CourseMaterials/other/derivatives.pdf)
  * [ ] [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)

Outline

* Computational Graph
* Backprop & Forwardprop
* Introducing regularization to prevent overfitting
* Non-linearity: activation functions
* Practical Tips
  * Parameter Initialization
  * Optimizers
    * plain SGD
    * more sophisticated adaptive optimizers
  * Learing Rates

#### Lecture 5: Linguistic Structure: Dependency Parsing

* [slides](CourseMaterials/slides/cs224n-2019-lecture05-dep-parsing.pdf)
* [notes](CourseMaterials/notes/cs224n-2019-notes04-dependencyparsing.pdf)
  * readings
    * [ ] Incrementality in Deterministic Dependency Parsing
    * [ ] A Fast and Accurate Dependency Parser using Neural Networks
    * [ ] Dependency Parsing
    * [ ] Globally Normalized Transition-Based Neural Networks
    * [ ] Universal Stanford Dependencies: A cross-linguistic typology

> mentioned CS103, CS228

### Assignments

#### Assignment 1: Exploring Word Vectors

* [code](Assignments/a1/exploring_word_vectors.ipynb)
* [directory](Assignments/a1)

#### Assignment 2: word2vec

* [handout](Assignments/a2/a2.pdf)
* [directory](Assignments/a2)
  * [written](Assignments/a2/written/assignment2.pdf)
  * [code](Assignments/a2/code)
    * `python3 word2vec.py` check the correctness of word2vec
    * `python3 sgd.py` check the correctness of SGD
    * `./get_datasets.sh; python3 run.py` - training took 9480 seconds

Related

* [Data processing in cs224n assignment 2 word2vec (2019)](https://medium.com/@ilyarudyak/data-processing-in-cs224n-assignment-2-word2vec-2019-288bdc8d4cb6)

Others' Answer

* [ZacBi/CS224n-2019-solutions/.../word2vec.py](https://github.com/ZacBi/CS224n-2019-solutions/blob/master/Assignments/a2/word2vec.py)
* [ZeyadZanaty/cs224n-assignments/.../word2vec.py](https://github.com/ZeyadZanaty/cs224n-assignments/blob/master/Assignment-2/word2vec.py)

#### Assignment 3: Dependency Parsing

* [handout](Assignments/a3/a3.pdf)
* [directory](Assignments/a3)
  * [written](Assignments/a3/written/assignment3.pdf)
  * [code](Assignments/a3/code)

Others' Answer

* [CS224N-2019/Assignment/a3 at master · Luvata/CS224N-2019](https://github.com/Luvata/CS224N-2019/tree/master/Assignment/a3)
