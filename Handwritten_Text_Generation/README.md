# Task 3 – Handwritten Text Generation ✍️  

## 📌 Overview  
This project implements a **Recurrent Neural Network (RNN)** to generate handwritten-style text using the **IAM Handwriting Dataset**.  
The model learns handwriting patterns and can generate new handwritten sequences from text input.  

## 📂 Project Structure  
HANDWRITING_TEXT_GENERATION/
│── app.py # Main training & evaluation script
│── data_loader.py # Custom dataset loader
│── model.py # RNN model definition
│── handwriting_rnn_epoch1.pth # Saved checkpoint after epoch 1
│── handwriting_rnn_epoch2.pth
│── handwriting_rnn_epoch3.pth
│── handwriting_rnn_epoch4.pth
│── handwriting_rnn_epoch5.pth
│── handwriting_rnn.pth # Final trained model


## 📂 Dataset  
- **IAM Handwriting Database** → [Teklia IAM-Line Dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)  
- Dataset includes: handwritten English text lines.  
- Preprocessing steps:  
  - Convert to grayscale  
  - Resize all images to **128 × 512**  
  - Normalize & convert to tensor  

```python
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 512)),
    transforms.ToTensor()
])
⚙️ Steps Performed
Data Loading → Custom dataloader for IAM dataset.

Preprocessing → Grayscale, resize, tensor conversion.

Model Definition → RNN-based handwriting generation model (model.py).

Training → Trained for 5+ epochs with saved checkpoints.

Evaluation → Generated handwriting-like sequences.

📊 Results
The model successfully learns handwriting strokes.

Generated samples improve as epochs increase.

Example checkpoints:

handwriting_rnn_epoch1.pth → Early stage handwriting

handwriting_rnn_epoch5.pth → More refined handwriting

🚀 How to Run
🔹 Training the Model
python app.py
🔹 Using Saved Models
Replace handwriting_rnn.pth with any checkpoint (e.g., handwriting_rnn_epoch3.pth) to test intermediate results.

📌 Future Improvements
Train for more epochs with GPU support.

Use Transformer-based architectures for smoother strokes.

Expand dataset for multi-style handwriting generation.

📌 Conclusion
This project demonstrates how RNNs can learn handwriting styles from real-world datasets and generate new handwriting from text.
The trained model can be further improved with advanced deep learning architectures
