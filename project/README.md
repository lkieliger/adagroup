# Amazon Reviews Analysis for Focused Online Shopping

# [Project Notebook](https://github.com/lkieliger/adagroup/blob/master/project/Natural_Language_Project.ipynb)

## Abstract

When shopping online, we often rely on the feedback provided by other internet users to decide wheter the merchandise we are considering buying is worth the money. Most electronic commerce platform allow their users to review or rate their purchases, usually with a system of stars or points. In this project, we aim to complement those means of providing feedback by automatically analysing and aggregating the reviews of multiple users. We experiment with opinion mining through the use of natural language processing techniques. The goal is to extract from the reviews common opinion on product characteristics such as quality or performance. This way, subsequent buyers will be able to grasp at a glance the strong and weak points of what want to acquire.

## Research questions

1. What are the recurring opinions on a given product ?
2. Can they be used to characterize it and offer to the reader a quick overview of its pros and cons?

## Additional research questions:
1. Do certain brands tend to be associated with specific characteristics ?

## Datasets

[**Amazon product data**](http://jmcauley.ucsd.edu/data/amazon/) contains
> Electronic product reviews and metadata from Amazon, spanning May 1996 - July 2014.
* Use reviews for characteristics extraction
* Ratings for sentiment analysis training
* Metadata to associate common product characteristics to brands

[**Wiktionary page titles**](https://dumps.wikimedia.org/enwiktionary/latest/)
> Page titles from Wiktionary, a free and open dictionnary project from the Wikimedia foundation. Over 5 millions english words.
* Use wiktionnary entries to filter out bigram characteristics that corresponds to compound words

## External libraries
* Natural Language Tool Kit
* Numpy
* Pandas

## Work distribution
* **Leandro Kieliger**: Updated research questions for milestone 2, studied NLP techniques and implemented bigram collocations extraction with PMI and likelihood ratios. Implemented first approach for sentiment analysis with SentiWordNet. Made structure and first draft of the notebook, heavy rewriting on report.
* **Quentin Bacuet**: Performed exploratory data analysis. Improved sentiment analysis with multilayer perceptron and collocation counting. Worked on collocation aggregation per brand, scoring mechanism for products and brands. Implemented notebook visualisations. Made structure and first draft of the report.
* **Olesya Altunina**: wrote initial research questions for milestone 1..
