# Aramis Image Argument Search

This repository represents the attempt for the [Touché Task 3: Image Retrival for Arguments](https://webis.de/events/touche-22/) by the group Aramis. 

## Setup
Clone this repository and add the [data](https://files.webis.de/data-in-progress/data-research/arguana/touche/touche22/2022-task3/) into the directory `data/`. The entrypoint to start the application is `startup.py`
After the virtualenv installation with ``python -m pip install -r requirements.txt`` you need to run `python -m spacy download en_core_web_sm`

### Image-detection
To run the image-detection you have to download the tesseract5-installer [directly](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.0.0-rc1.20211030.exe)
or from the [Uni-Mannheim-page](https://github.com/UB-Mannheim/tesseract/wiki) and install it. Please do not use tesseract 4 or a version below because i don't know what will happend. Maybe the whole project makes boom.

You can install the application directly into ``\properties`` or install it anywhere and copy the ``tesseract`` folder into ``\properties``.
Check, if the ``\properties\tesseract\tesseract.exe`` exists, but if you have done everything correctly this file should be there.

At least please run the ``run_once.py`` just once. After that everything is set up to have fun :D

## Evaluation

| Models used      | topic_weight | arg_weight | stance_weight | Topic filtered | strong@20 | strong@50 | both@20 | both@50 | topics     |
|------------------|--------------|------------|---------------|----------------|-----------|-----------|---------|---------|------------|
| NNArg & NNStance | 0            | 1          | 0             | False          | 0.0923    | ?         | 0.158   | ?       | all evaled |
| NNArg & NNStance | 0            | 1          | 0             | True           | 0.1337    | 0.1316    | 0.2324  | 0.2309  | all evaled |
| NNArg & NNStance | 1            | 0          | 0             | False          | 0.0736    | ?         | 0.131   | ?       | all evaled |