# Text Classification and Machine Translation
  
  - This is a practice exercise in CS221 - Natural Language Processing (University of Information Technology - VNUHCM)
  - We build a standard process in building models to solve NLP problems (Text Classification and Machine Translation):
    1. Preprocessing
      - Vietnamese: Normalize unicode, Standardize Vietnamese punctuation, Separating Vietnamese words, Convert lowercase letters, Sentence normalization
      - English: Punctuation Standardization, Convert lowercase letters
    2. Prepare data: Divide the dataset into train, validation, and test sets.
    3. Word embedding: a technique in natural language processing (NLP) that represents words in a mathematical form, typically as vectors, which can be easily processed by machine learning algorithms. 
      - There are several methods of word embedding: One-Hot Encoding, Count-based Embedding (LSA), Prediction-based Embedding (Word2Vec and GloVe), Contextual Embedding (BERT)
      - In this project, we use 2 popular methods: CountVectorizer and TfIdfVectorizer.
    4. Model selection and training:
      - Text Classification: Support Vector Machine (SVM), Naivie Bayes (NB), Logistic Regression (LR)
      - Machine Translation: Encoder-Decoder LSTM (because of resource and time limitations I only train on 100 epochs and a small portion of data)
    5. Evaluation:
      - Text Classification: Compare the performance of the models on 4 metrics: accuracy, precision, recall, f1_score (specific results are detailed in the notebook)
  - For these two problems, we both use the [Domain_specific_EVCorpus_Done](https://drive.google.com/file/d/1iYGB705res6ENM6HnN-_DypLe9wmNE2f/view) dataset
  - Additionally, we use libraries ([underthesea](https://github.com/undertheseanlp/underthesea.git), [gensim](https://github.com/RaRe-Technologies/gensim.git), unicodedata) for preprocessing. To handle Vietnamese problems, we can also use [VnCoreNLP](https://github.com/KhaLee2307/VnCoreNLP.git) and [pyvi](https://github.com/KhaLee2307/pyvi.git).
