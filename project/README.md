# Amazon Reviews Analysis for Focused Online Shopping

## Abstract

When shopping online, we often rely on feedback provided by other internet users to decide wheter the product we are considering buying is worth the money. Most electronic commerce platform allow their users to review or rate their purchases, usually with a system of stars or points. In this project, we aim to complement those means of providing feedback by automatically analysing and aggregating the reviews of multiple users. We experiment with opinion mining through the use of natural language processing techniques. The goal is to extract from the reviews common opinion on product characteristics such as quality or performance. This way, subsequent buyers will be able to grasp at a glance the strong and weak points of the products.

## Research questions

1. What are the recurring opinions on a given product ? Can they be used to characterize it and offer to the reader a quick overview of the pros and cons of the product ?

## Additional research questions:
1. Do certain brands tend to be associated with specific characteristics ?
2. Do that perception vary over time ? 

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
