# Stanford CS224n Natural Language Processing with Deep Learning

The course notes about Stanford CS224n Winter 2019 (using PyTorch)

> Some general notes I'll write in [my Deep Learning Practice repository](https://github.com/daviddwlee84/DeepLearningPractice)

Course Related Links

* [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
* [Lecture Videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)
* [Course contents backup](https://github.com/zhanlaoban/CS224N-Stanford-Winter-2019)
* Others' answer
  * [Observerspy/CS224n](https://github.com/caijie12138/CS224n-2019) (not fully 2019)
  * [ZacBi/CS224n-2019-solutions](https://github.com/ZacBi/CS224n-2019-solutions) (didn't finish the written part)
  * [caijie12138/CS224n-2019](https://github.com/caijie12138/CS224n-2019) (not quite the assignment)

## Schedule

| Week           | Lectures                                                                                                                                               | Assignments                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| 2019/7/1~7/7   | [Introduction and Word Vectors](#Lecture-1-Introduction-and-Word-Vectors), [Word Vectors 2 and Word Senses](#Lecture-2-Word-Vectors-2-and-Word-Senses) | [Assignment 1](#Assignment-1-Exploring-Word-Vectors) |
| 2019/7/8~7/14  | [Word Window Classification, Neural Networks, and Matrix Calculus](#Lecture-3-Word-Window-Classification-Neural-Networks-and-Matrix-Calculus)          | -                                                    |
| 2019/7/15~7/21 | [Backpropagation and Computation Graphs](#Lecture-4-Backpropagation-and-Computation-Graphs)                                                            | [Assignment 2](#Assignment-2-word2vec)               |

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

### Assignments

#### Assignment 1: Exploring Word Vectors

* [code](Assignments/a1/exploring_word_vectors.ipynb)
* [directory](Assignments/a1)

#### Assignment 2: word2vec

* [handout](Assignments/a2/a2.pdf)
* [directory](Assignments/a2)
  * [written](Assignments/a2/written)
  * [code](Assignments/a2/code)
