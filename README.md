# LFQA-HP-1M: A Large-Scale Human Preference Dataset for Long-Form Question Answering

The repository provides a research-oriented framework for constructing, analyzing,
and evaluating long-form question answering (LFQA) datasets. The
repository supports human preference modeling, pairwise answer
evaluation, fine-grained rubric extraction, and LLM-as-a-Judge
experimentation.


## Paper

📄 LFQA-HP-1M: A Large-Scale Human Preference Dataset for Long-Form Question Answering  
Proceedings of the 14th International Conference on Language Resources and Evaluation (LREC 2026)

Paper link:  


---

## Dataset Access


The dataset associated with this framework are publicly available on Hugging Face:

LFQA-HP-1M (Full Dataset):
https://huggingface.co/datasets/nlpatunt/LFQA-HP-1M


This toolkit is released for academic and research purposes and is
designed to facilitate reproducible experimentation in evaluation
methodology research.

------------------------------------------------------------------------

## Overview

Long-form question answering requires evaluation beyond simple
correctness. The repository provides tools to:

-   Data collection, format, and cost-effective LFQA filter methodology 
-   Extract fine-grained evaluation rubrics
-	Simple Logistic Regression Model
-   Run LLM-as-a-judge evaluator

The framework emphasizes modularity, transparency, and reproducibility.

------------------------------------------------------------------------

## Repository Structure

    config/                     Environment configuration files
    dataset_creation/           Pairwise dataset formatting and randomization
    dataset_analysis_sampling/  Dataset inspection and sampling utilities
    rubric_extraction/          Fine-grained rubric extraction modules
    prompt/                     Evaluation prompt templates
    Main.py                     Main Model Selection

------------------------------------------------------------------------

## Installation

Clone the repository:

    git clone https://github.com/nlpatunt/lfqa-eval-lrec-2026.git
    cd lfqa-eval-lrec-2026

Create a virtual environment:

    python3 -m venv venv
    source venv/bin/activate

Install dependencies:

    pip install torch transformers tqdm datasets

Additional dependencies may be required depending on experiment
configuration.

------------------------------------------------------------------------

## Environment Configuration

Update the `.env` file inside the `config/` directory:

    OPENROUTER_API_KEY=your_api_key
    HF_TOKEN=your_huggingface_token



## Model selection from Open Router

	Main.py

        router = OpenRouter(
            model_name= <Replace with the model>,
            key=api_key
            
        )

------------------------------------------------------------------------

## Usage

### 1. Dataset Construction

    `dataset_creation/`

The repository contains script files that has been to collect, filtered, and formate for LFQA research.

### 2. Rubric Extraction

    python rubric_extraction/<script_name>.py

Extracts fine-grained evaluation dimensions including:

-   Specificity
-   Grammar
-   Fluency
-   Completeness
-   Coherence
-   Relevance
-   Conciseness
-   Use of Examples
-   Factuality

### 3. Logistic Regression Model Weights

    python rubric_extraction/RegressionPreferenceModel.py
	
Contain codes to extract logistic regression model feature weights.


### 3. Prompts

    `prompt/`

Contains all Prompts


### 3. Evaluation
	python rubric_extraction/LogisticValidate.py
	
Rubric based LR model evaluation

	python rubric_extraction/LLM_judgement.py

LLM-as-a-judge evaluation



------------------------------------------------------------------------

## Adversarial Perturbations Victime Model and Dataset

Model: https://huggingface.co/nlpatunt/modernbert-base-lfqa

## Datasets
Two adversarially perturbed LFQA datasets are provided:

https://huggingface.co/datasets/nlpatunt/lfqa-textbugger

https://huggingface.co/datasets/nlpatunt/lfqa-deepwordbug

------------------------------------------------------------------------

## Reproducibility Statement

-   Deterministic randomization supported via fixed seeds\
-   Modular pipeline design for controlled experimentation\
-   Compatible with large-scale datasets

Researchers are encouraged to report model versions, seeds, and hardware
configurations when publishing results.

------------------------------------------------------------------------

## Data Statement

This repository provides tooling for dataset construction and
evaluation. Datasets processed using this framework may contain
user-generated content.

Researchers should ensure:

-   Appropriate licensing and redistribution rights\
-   Removal of sensitive or personally identifiable information\
-   Compliance with ethical research standards

Any dataset created using this pipeline should document:

-   Data sources\
-   Annotation methodology\
-   Sampling strategy\
-   Potential biases

------------------------------------------------------------------------

## Ethics and Limitations

The evaluation of long-form responses may reflect biases present in:

-   Training data
-   Human annotation processes
-   Model-generated reasoning

Rubric-based scoring does not fully capture nuanced human judgment.
LLM-as-a-Judge evaluations may exhibit positional, verbosity, or
stylistic biases.

Researchers should interpret evaluation metrics cautiously and report
limitations transparently.




------------------------------------------------------------------------

## License



------------------------------------------------------------------------

## Contact

For questions, issues, or collaboration inquiries, please open an issue
in this repository.


## Citation

If you use LFQA-HP-1M in your research, please cite:

```bibtex
@inproceedings{jahan-etal-2026-lfqahp1m,
  title     = {LFQA-HP-1M: A Large-Scale Human Preference Dataset for Long-Form Question Answering},
  author    = {Jahan, Rafid Ishrak and Iqbal, Fahmid Shahriar and Ray Choudhury, Sagnik},
  booktitle = {Proceedings of the 14th International Conference on Language Resources and Evaluation (LREC 2026)},
  year      = {2026},
  publisher = {European Language Resources Association (ELRA)},
  address   = {Marseille, France}
}






