# Hindi-English Neural Machine Translation

This repository contains an implementation of a Neural Machine Translation (NMT) model using an Encoder-Decoder architecture with LSTM layers to translate English sentences into Hindi. The dataset used for training is a TED corpus containing pairs of English-Hindi sentences.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)

## Project Overview
This project aims to build a machine translation model to translate English sentences into Hindi using the sequence-to-sequence model architecture. It leverages LSTMs for both encoding and decoding processes.

The project involves:
1. Preprocessing the text data (removing special characters, normalizing, tokenizing).
2. Building and training an Encoder-Decoder model using LSTM layers.
3. Evaluating the model's performance on a test set.

## Dataset
The dataset used is the **Hindi-English Truncated Corpus** obtained from TED talks. It includes over 120,000 sentence pairs. We sample 25,000 rows and preprocess the data by cleaning and adding start (`START_`) and end (`_END`) tokens to the target Hindi sentences.

### Dataset Information
- Source: TED talks
- Language Pairs: English → Hindi
- Sentence Count: 25,000 after sampling
- File Name: `Hindi_English_Truncated_Corpus.csv`

## Model Architecture
The model follows the **Encoder-Decoder** architecture using LSTM layers for both encoder and decoder components.

### Encoder
- Input: English sentences
- Layers: Embedding → LSTM
- Latent Dimension: 300

### Decoder
- Input: Hindi sentences with special tokens (`START_`, `_END`)
- Layers: Embedding → LSTM → Dense layer with softmax activation
- Latent Dimension: 300

### Training
The model is trained using the following parameters:
- Loss: Categorical Crossentropy
- Optimizer: RMSprop
- Epochs: 30
- Batch Size: 128

## Dependencies
Ensure you have the following libraries installed:

```bash
numpy
pandas
matplotlib
seaborn
tensorflow (Keras included)
scikit-learn
```

## Usage

1. Clone the repository and navigate to the project directory:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Load the dataset and preprocess it:
```python
import pandas as pd
lines = pd.read_csv('Hindi_English_Truncated_Corpus.csv', encoding='utf-8')
```

3. Split the data into training and testing sets:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

4. Train the model using the provided code in the notebook:
```python
model.fit_generator(generator=generate_batch(X_train, y_train),
                    steps_per_epoch=train_samples // batch_size,
                    epochs=30,
                    validation_data=generate_batch(X_test, y_test),
                    validation_steps=val_samples // batch_size)
```

5. Evaluate the model performance on test data:
```python
loss = model.evaluate_generator(generator=generate_batch(X_test, y_test))
```



## Results
After training the model for 30 epochs, the following results were obtained:
- Training Loss: ~3.5
- Validation Loss: ~5.3

Further tuning of hyperparameters and adding more data may improve these results.

## Future Work
- Attention Mechanism: Implement an attention layer to enhance translation accuracy.
- Data Augmentation: Use additional datasets to improve model performance.
- Beam Search: Use beam search during decoding for better translation results.
- Evaluation Metrics: Introduce the BLEU score for a more robust evaluation of translation quality.



