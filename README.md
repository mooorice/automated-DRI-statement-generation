# Automated Deliberative Reason Index (DRI) Statement Generation

This repository contains the code for an automated approach to generating Deliberative Reason Index (DRI) statements using advanced natural language processing (NLP) and large language models (LLMs). The project aims to streamline the process of assessing group deliberation quality by automating the traditionally labor-intensive task of DRI statement generation.

## Overview

The DRI is a sophisticated metric for evaluating group reasoning and deliberation quality. Traditionally, generating DRI statements has been a complex and time-consuming process requiring extensive human labor. This project introduces an innovative automated approach that significantly reduces the human effort involved in survey preparation while maintaining the quality and validity of the assessment.

## Project Structure

The repository is organized into several key components:

- `01a_get_articles.ipynb`: Notebook for retrieving and processing news articles from the swissdox API
- `01b_content_anchor_generation.ipynb`: Notebook for generating content anchors with an LLM
- `02_chunk_into_paragraphs.ipynb`: Notebook for text chunking and preprocessing for the newspaper articles from swissdox API
- `03_generate_embeddings.py`: Script for generating embeddings for content anchors and newspaper paragraphs
- `04_dimensionality_reduction.py`: Script for reducing embedding dimensions
- `05_categorization.py`: Script for categorizing content by cosine similarity scoring
- `06_similarity_score_analysis.ipynb`: Notebook for analyzing the distribution of similarity scores
- `07_generate_dri_statements.ipynb`: Notebook for generating final DRI statements with an LLM
- `Validation/`: Directory containing the notebooks, scripts and data for the validation process
- `data/`: Directory for storing processed data and intermediate results

## Methodology

The project implements a systematic framework that:

1. Retrieves and processes news articles to approximate the public sphere
2. Generates embeddings from the content using multilingual models
3. Categorizes content using cosine similarity
4. Leverages LLMs to generate DRI statements
5. Validates the generated statements against a traditional DRI survey

## Setup

1. Clone the repository
2. Install required dependencies
3. Set up your environment variables in `.env` and `Validation/.env`

## Usage

The workflow follows a sequential process through the numbered notebooks and scripts. Each step builds upon the previous one:

1. Start with `01a_get_articles.ipynb` to retrieve your source articles
2. Generate content anchors using `01b_content_anchor_generation.ipynb`
3. Process the articles usin `02_chunk_into_paragraphs.ipynb`
4. Generate embeddings using `03_generate_embeddings.py`
5. Apply dimensionality reduction with `04_dimensionality_reduction.py`
6. Categorize content using `05_categorization.py`
7. Analyze similarity scores in `06_similarity_score_analysis.ipynb`
8. Generate final DRI statements using `07_generate_dri_statements.ipynb`

## Results

The validation results and analysis can be found in the `Validation/` directory. The final DRI statements are generated in `07_generate_dri_statements.ipynb`.

## Acknowledgments

This project is part of a Master's thesis in Political Science, focusing on the application of AI and ML techniques to automatically generate DRI statements.

## License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
