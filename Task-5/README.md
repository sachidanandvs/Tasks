Sentiment analysis using DistilBERT on Yelp reviews - Full and analysing the performance after applying the EDA.
=======

Link for the [Colab Notebook](https://drive.google.com/file/d/1qFFptZkOHtPtj5rIJ5Cq_BTnq7uQgzrn/view?usp=sharing) and the [Dataset](https://drive.google.com/drive/folders/16igV70hjWDon5BeMVc0gfGbdUc0uOGad?usp=sharing) used along with the augmented data.

Preprocessing Data
-----------
To tokenize the sentences and to get Id of eack token we are using DistilBertTokenizer from Transformers library.

---
Model
-------
We are using DistilBERT transformer model and loading pretrained weights of bert-base-uncased.

---
EDA Implementation
--------
Data Augmentation used are:

    * Random Deletion
    * Random Swap
    * Random Insertion
    * Synonym Replacement
Example:

Original sentence:

*dr goldberg offers everything i look for in a general practitioner

Augmented Sentence:


*dr goldberg and everything i look for in a general practitioner

*dr goldberg offers everything i look his surgery a general practitioner


Code for data augmentation is used from [here](https://github.com/jasonwei20/eda_nlp).Which is an implementation of this [paper](https://arxiv.org/pdf/1901.11196.pdf)

---

Comparing Results befor and after EDA:
---
---

Due to Computational restriction we are using only 6500 reviews from Yelp Review -Full and trained using 20 % ,40 % and 100 % of this dataset.

From the graph it is clear that accuracy is increased after EDA but increase in accuracy is slowly dies out when dataset is huge.





![Image](plot.png)

