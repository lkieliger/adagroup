# Amazon Reviews Analysis for Focused Online Shopping

## Abstract

<Other people's opinions tend to make us more **open-minded and critical thinking** when they are more personal and detailed. It is easier for us to **relate to and question** specific stories, which describe someone's opinion formation, rather than general statements. In this project, we aim to illuminate this observation from different perspectives analysing Amazon product reviews dataset using **Natural Language Processing** (NLP) techniques. We assume, first, that **longer** reviews tend to be **more informative**, thus more detailed and personal, and second, that users **find helpful** those reviews that helped them make **informed undisturbed decisions** and **left them satisfied** with their choices. This angle of view on opinion formation and decision making might be helpful towards reframing online shopping experience from quite a **distracting and exhaustive** one to more **thought-out and unbiased** by (at least) increasing people's awareness about the **value of focused life**.



## Research questions

1. What are the recurring opinions on a given product ? Can they be used to characterize it and offer to the reader a quick overview ?
2. Do certain brands tend to be associated with specific qualifiers ?
3. Do that perception vary over time ? 

<!--
1. What are the **positive/critical reviews percentages** for **short vs long** reviews (using ratings)?
2. How do **bags-of-words** differ for **short vs long** Amazon reviews?
3. How do **bags-of-words/n-grams** differ for **long** Amazon reviews (long reviews allow n-gram analysis)?
4. Are there **critical** reviews with **high** ratings?
5. ... or the **positive** ones with **low** ratings?
6. To what extent do **long** reviews tend to be found **more helpful**?
7. Is there a **length threshold** that separates unhelpful reviews from the helpful ones?
8. What product **categories** tend to contain **more long reviews** (e.g. cheap vs expensive)?
9. What product **categories** tend to contain **long reviews** that are found **more helpful** (e.g. cheap vs expensive)?
10. How does the **tendency** towards writing long reviews depend on **age/gender** (using reviewer IDs)?
11. What **attitudes** might be reflected in writing **long detailed** reviews?
12. How does the percentage of **long** reviews **evolve** throughout the timespan of the dataset?
-->


## Dataset

[**Amazon product data**](http://jmcauley.ucsd.edu/data/amazon/) contains
> product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014.

What we are going to use:
- Reviews (for NLP),
- Ratings,
- Helpfulness,
- Reviewer IDs (to access age/gender),
- Product categories.

The dataset is **easily parsed** using Python and the size of it (max. 25 Gb) allows to store it on a personal computer.

We will need to perform some **web-scraping** for information about reviewers (by their IDs). Depending on how much **time and size** it takes and how **accessible the old data** is, we will probably only scrape for relatively recent reviewers.


## Internal milestones up to Milestone 2

1. Download the dataset.
2. Perform **missing values** check.
3. Bring the dataset into most **readable and easy-to-manipulate** form (preferably a single dataset).
4. Answer simple **statistical questions** on the dataset (including length distributions).
5. Assess the first approximation of short/long reviews **threshold**.
6. **Web-scrape** some information on reviewers.
7. **Integrate** the scraped information into the original dataset.
8. Train a simple **sentiment analysis classifier** on long reviews (rating prediction).
9. Train an **unsupervised classifier** to look at what's going on inside short/long reviews.
10. Try to perform **aspect based sentiment analysis** to dive into more details.


## Questions to TAs

Not yet :)
