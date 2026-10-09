# Mushroom CNN and RNN Classification + Hugging Face NLP Tokenization

Deep learning classification project using TensorFlow/Keras to build, train, evaluate, and compare CNN and RNN models on the Mushroom/Fungi Classification dataset, with Hugging Face Transformer tokenization for an NLP task.

## Project Description

The Mushroom dataset describes mushrooms using only **categorical** attributes (cap shape, odor, gill color, habitat, ...). The task is to predict each mushroom's class, edible or poisonous. The project:

1. Explores and preprocesses the categorical data (one-hot encoding, 70/30 stratified split, no data leakage).
2. Trains a **1D Convolutional Neural Network (CNN)** and a **Recurrent Neural Network (SimpleRNN)** with TensorFlow/Keras.
3. Evaluates both models (test accuracy, predictions, confusion matrices, classification reports) and recommends the better model based on measured results.
4. Completes a **Generative AI / NLP task** by tokenizing text with a pretrained Hugging Face tokenizer (`bert-base-uncased`).

## Objectives

- Convert categorical features into a numerical representation suitable for neural networks.
- Build, train and evaluate a CNN and an RNN on the same train/test split.
- Visualize training/validation accuracy and loss, confusion matrices and a model comparison.
- Recommend a model using the measured test accuracy.
- Show how text is converted into tokens, input IDs and attention masks for Large Language Models.

## Dataset Information

| File | Source | Description |
| --- | --- | --- |
| `data/mushrooms.csv` | <https://lms.digiskills.pk/Courses/AIP301/Downloads/mushrooms.csv> | Mushroom dataset. All columns are categorical letter codes. The notebook detects the target column automatically (`class`: `e` = edible, `p` = poisonous). |
| `data/exercise_text.json` | <https://lms.digiskills.pk/Courses/AIP301/Downloads/exercise_text.json> | JSON file containing the text used for the tokenization task. |

In the mushroom data, a `?` value marks an unknown attribute. It is kept as its own category. Any feature column with only one value in the training data is dropped, because it carries no information.

## Model Configurations

### CNN

| Setting | Value |
| --- | --- |
| Model | CNN (Conv1D) |
| Batch size | 30 |
| Epochs | 10 |
| Optimizer | Adam |
| Input shape | `(samples, number_of_features, 1)` |
| Architecture | Input -> Conv1D(32, ReLU) -> MaxPool -> Conv1D(64, ReLU) -> MaxPool -> Flatten -> Dense(64, ReLU) -> Dropout(0.3) -> Output |

### RNN

| Setting | Value |
| --- | --- |
| Model | RNN (SimpleRNN) |
| Batch size | 40 |
| Epochs | 20 |
| Optimizer | RMSprop |
| Input shape | `(samples, number_of_features, 1)` |
| Architecture | Input -> SimpleRNN(64) -> Dropout(0.3) -> Dense(32, ReLU) -> Output |

The output layer is chosen from the number of classes found in the data:

- 2 classes: `Dense(1, sigmoid)` with binary cross-entropy
- more than 2 classes: `Dense(n, softmax)` with sparse categorical cross-entropy

Both models use `validation_split=0.2` taken from the training set. The 30% test set stays unseen until evaluation.

## NLP Task

The text in `exercise_text.json` is extracted after inspecting the JSON structure. It is then tokenized with `AutoTokenizer.from_pretrained("bert-base-uncased")`. The notebook shows the original text, the tokenizer output, the tokens, input IDs, attention mask and the number of tokens. No language model is trained.

## Technologies Used

- Python 3.10 – 3.13
- TensorFlow / Keras (CNN and RNN)
- Hugging Face Transformers (tokenization)
- scikit-learn (encoding, splitting, metrics)
- pandas, NumPy
- matplotlib, seaborn (visualization)
- Jupyter Notebook

`torch` is listed in `requirements.txt` as requested, but it is **not** used. All neural networks are implemented with TensorFlow/Keras.

## Project Structure

```text
mushroom-cnn-rnn-classification/
│
├── data/
│   ├── mushrooms.csv
│   └── exercise_text.json
│
├── notebooks/
│   └── mushroom_cnn_rnn_assignment.ipynb
│
├── outputs/
│   ├── figures/      # PNG plots
│   ├── reports/      # classification reports, comparison table, predictions, tokens
│   └── models/       # trained Keras models (.keras)
│
├── requirements.txt
├── README.md
└── download_data.py
```

## Installation

Use a Python version supported by TensorFlow (3.10 – 3.13 at the time of writing).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install the dependencies
pip install -r requirements.txt
```

## Download / Prepare the Data

```bash
python download_data.py
```

The script:

- creates `data/` if it does not exist
- downloads both files and checks that they are not empty and are valid CSV/JSON
- skips files that already exist and are valid (use `--force` to re-download)
- handles HTTP errors, timeouts and connection failures

The LMS server does not send its full HTTPS certificate chain. If Python cannot verify the certificate, the script retries using the operating system's certificate store (`truststore` package) and then the system `curl` command. Certificate verification is never turned off.

If the download still fails, download the files manually from the URLs above and save them as `data/mushrooms.csv` and `data/exercise_text.json`.

The notebook also checks for the files. It runs `download_data.py` automatically if they are missing.

## How to Run the Notebook

```bash
jupyter notebook notebooks/mushroom_cnn_rnn_assignment.ipynb
```

Then choose **Kernel -> Restart & Run All** to run the whole assignment from start to finish.

To run it non-interactively from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/mushroom_cnn_rnn_assignment.ipynb
```

The first run downloads the `bert-base-uncased` tokenizer from the Hugging Face Hub, so it needs an internet connection.

## Expected Outputs

**Figures** (`outputs/figures/`):

- `class_distribution.png`: target class distribution
- `cnn_training_history.png`: CNN training vs validation accuracy and loss
- `cnn_confusion_matrix.png`: CNN confusion matrix
- `rnn_training_history.png`: RNN training vs validation accuracy and loss
- `rnn_confusion_matrix.png`: RNN confusion matrix
- `cnn_vs_rnn_accuracy.png`: test accuracy comparison bar chart

**Reports** (`outputs/reports/`):

- `cnn_classification_report.txt` / `.csv`
- `rnn_classification_report.txt` / `.csv`
- `cnn_vs_rnn_comparison.csv`: Model, Optimizer, Batch Size, Epochs, Test Accuracy, and more
- `test_predictions_cnn_rnn.csv`: actual class vs CNN and RNN predictions for every test sample
- `tokenization_output.csv`: token, input ID and attention mask for every token

**Models** (`outputs/models/`):

- `cnn_model.keras`, `rnn_model.keras`

## Academic Note

All accuracy values, reports and figures are produced by running the notebook. None are written by hand, and this README does not list any accuracy values. Random seeds are fixed (`random_state=42` and `keras.utils.set_random_seed(42)`). Neural-network training can still vary slightly depending on hardware (CPU/GPU), operating system and the installed TensorFlow version, so your numbers may differ slightly from another run.
