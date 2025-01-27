# Image Caption Generator using Deep Learning on Flickr8K Dataset

This project involves generating captions for images using a deep learning model built on the Flickr8K dataset. The model utilizes a combination of a pre-trained VGG16 model for image feature extraction and an LSTM-based model with attention mechanisms for generating image captions.

## Project Structure

1. **Data**: The project uses the Flickr8K dataset, which contains 8,000 images along with multiple captions for each image.
   
2. **Model**: The model consists of an encoder-decoder architecture. The encoder extracts features from images using the VGG16 model, while the decoder generates captions based on the image features using an LSTM network with an attention mechanism.

3. **Captions Preprocessing**: Captions are cleaned, tokenized, and padded to create input-output pairs for training the model.

4. **Training**: The model is trained on the Flickr8K dataset to predict captions for unseen images.

5. **Evaluation**: The model's performance is evaluated using the BLEU score, which measures the quality of the generated captions.

---

## Requirements

- **Python** 3.x
- **TensorFlow** 2.x
- **Keras**
- **NumPy**
- **Matplotlib**
- **PIL** (Pillow)
- **NLTK**
- **tqdm** (for progress bar)

---

## Installation

You can clone this repository and install the required dependencies using `pip`:

```bash
git clone https://github.com/NiyatiP10/Image-Caption-Generator-using-Deep-Learning-on-Flickr8K-dataset.git
cd Image-Caption-Generator-using-Deep-Learning-on-Flickr8K-dataset
pip install -r requirements.txt
```

---

## Data

The dataset used in this project is the [Flickr8K dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k).

1. Download the dataset and place it in the `Images` directory.
2. The captions file (`captions.txt`) contains the descriptions for each image, and should be placed alongside the image directory.

---

## Model Architecture

### Encoder

The encoder is built using the VGG16 model, which is pre-trained on the ImageNet dataset. We remove the final classification layers and use the penultimate layer's output as the image features. These features are then processed by a Bidirectional LSTM to create a context vector.

### Decoder

The decoder uses an LSTM network to generate captions. The input sequence is the previously generated word, and the LSTM produces the next word. Attention mechanism is applied to focus on relevant parts of the image while generating the caption.

---

## Steps

1. **Feature Extraction**:  
   We use the VGG16 model to extract features from the images. The features are stored in a dictionary where the image ID is the key.

2. **Caption Preprocessing**:  
   The captions are preprocessed by converting them to lowercase, removing non-alphabetical characters, and adding "startseq" and "endseq" tokens. This allows the model to learn the start and end of a caption.

3. **Tokenization**:  
   The preprocessed captions are tokenized using Keras' `Tokenizer`, and sequences are padded to a maximum caption length. The vocabulary size is determined from the tokenized captions.

4. **Model Training**:  
   The model is trained using the `data_generator` function, which yields batches of image features and caption sequences. The model is trained using categorical cross-entropy loss and the Adam optimizer.

5. **Caption Prediction**:  
   After training, the model can be used to predict captions for new images. The `predict_caption` function generates captions word by word using the trained model.

6. **Evaluation**:  
   BLEU scores (BLEU-1 and BLEU-2) are computed to evaluate the quality of the generated captions compared to the ground truth captions.

---

## Example Usage

To generate a caption for a new image, use the `generate_caption` function:

```python
generate_caption("101669240_b2d3e7f17b.jpg")
```

This will display the image along with the predicted caption and the actual captions.

---

## BLEU Score Evaluation

The model's performance is evaluated using BLEU scores, which measure the overlap between the predicted and actual captions.

- **BLEU-1**: Measures unigram precision.
- **BLEU-2**: Measures bigram precision.

Example BLEU scores:

```
BLEU-1: 0.028607
BLEU-2: 0.000000
```

---

## Saving and Loading the Model

The model can be saved using the following code:

```python
model.save('path/to/save/mymodel.keras')
```

You can load the saved model as follows:

```python
from tensorflow.keras.models import load_model
model = load_model('path/to/save/mymodel.keras')
```

---

## Future Improvements

- **Fine-tuning the VGG16 model**: Fine-tune the pre-trained VGG16 layers on the dataset to improve performance.
- **Using more advanced architectures**: Experiment with transformer-based architectures like the Vision Transformer (ViT) for better image captioning performance.
- **Improving BLEU score**: Try various caption generation techniques like beam search to improve the quality of the generated captions.

---

## Thank You!
