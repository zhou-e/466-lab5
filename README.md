# Lab5-CSC466
### Authors
Edward Zhou: ekzhou@calpoly.edu
Eric Inman: eainman@calpoly.edu

### Comments
#### knnAuthorship.py
> Takes in `<vectorized_filename> <k neighbors> <cos or okapi>`, where `<cos or okapi>` is 0 for cosine similarity and 1 for Okapi similiarity. \
> Please note that the vectorized files are in the `\knn_vectors` folder.

> If cosine similarity is chosen, the command line arguments are just `<vectorized_filename> <k neighbors> 0`. \
> If Okapi similarity is chosen, the command line arguments are `<vectorized_filename> <k neighbors> 1 <k_1 parameter> <k_2 parameter> <b parameter>`. Where k_1 is a float from 1.2 to 2, k_2 is a float from 1 to 1000, and b is a float between 0 and 1 (generally 0.75).

#### classifierEvaluation.py
> Takes in `<output_filename>` and prints out the precision, recall, f-1 score for each author, along with the overall accuracy, # correct, # incorrect and confusion matrix. \
> This script also writes out the confusion matrix to `<output_filename>_confusion_matrix.tsv`.

Instructions on Running Random Forest:

    1. Run "python3 oneVWorldEval.py [stem] [stopword]" in the command line
    
    2. If you want stemming or stopword removal, you have to input 'T' for yes and 
       any other letter for no in the [stem] and [stopword] spots above.
       
    3. Note: step 2 only works if you have the associated pickle files that go along
       with it, so I don't believe you'll be able to run it with those parameters.