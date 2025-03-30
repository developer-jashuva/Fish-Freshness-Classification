# 🐟 Fish Freshness Classification

This project is a **Deep Learning-based Fish Freshness Classification** system. It classifies fish images into four categories: **Fresh Eyes, Fresh Gills, Nonfresh Eyes, and Nonfresh Gills**. The model is built using **CNN (Convolutional Neural Networks)** and deployed using **Streamlit**.

## 📌 Features
- **Deep Learning-based Image Classification**
- **Pre-trained Model Support**
- **Streamlit Web App for Image Upload & Prediction**
- **User-friendly UI**

## 📂 Project Structure
```
fish-classification/
│── scripts/                     # Scripts for training and testing the model
│   ├── train_model.py           # Train the model
│   ├── test_model.py            # Evaluate the model
│   ├── predict.py               # CLI-based prediction
│── dataset/                     # Dataset folder (if needed)
│── models/                      # Trained models storage
│   ├── model.pth                # Trained model file
│── streamlit_app/               # Streamlit deployment folder
│   ├── app.py                   # Streamlit app file
│── requirements.txt             # Dependencies
│── README.md                    # Project documentation
```

## ⚙️ Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/fish-classification.git
cd fish-classification
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Download Pre-trained Model (Optional)
If you want to use a pre-trained model, place it inside the `models/` folder:
```
models/
│── model.pth  # Pre-trained model file
```

## 🚀 Training the Model
Run the `train_model.py` script to train the model:
```bash
python scripts/train_model.py
```

## 📊 Evaluating the Model
After training, you can evaluate the model performance using:
```bash
python scripts/test_model.py
```

## 🔍 Running Predictions (CLI)
To make predictions from the command line:
```bash
python scripts/predict.py --image path/to/image.jpg
```

## 🌍 Deploying the Web App (Streamlit)
Run the Streamlit app using:
```bash
streamlit run streamlit_app/app.py
```
This will launch a web UI where you can upload fish images and get freshness classification results.

## 📌 Example Predictions
| Sample Image | Predicted Class |
|-------------|----------------|
| ![Fresh Eyes]<img src="https://github.com/developer-jashuva/Fish-Freshness-Classification/blob/main/fresh_eyes.JPG" width="150" height="150" /> | Fresh_Eyes |
| ![Fresh Gills]<img src="https://github.com/developer-jashuva/Fish-Freshness-Classification/blob/main/fresh_Gills.JPG" width="150" height="150" /> | Fresh_Gills |

## 🔥 Future Enhancements
- Improve model accuracy with **more training data**
- Add **real-time webcam support** for classification
- Deploy online using **Streamlit Cloud**

## 🤝 Contributing
Feel free to contribute! Open an issue or submit a PR.

## 📜 License
This project is open-source under the **MIT License**.

---

🚀 **Happy Coding!** 🐟

