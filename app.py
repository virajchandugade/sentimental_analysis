
import joblib
import re
from flask import Flask, request, render_template
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Load the saved model pipeline
pipeline_filename = 'sentiment_model.pkl'
try:
    model_pipeline = joblib.load(pipeline_filename)
    print("Pipeline Steps and Parameters:")
    for step_name, step_process in model_pipeline.named_steps.items():
        print(f"Step Name: {step_name}")
        print(f"Step Process: {step_process.__class__.__name__}")
        print(f"Parameters: {step_process.get_params()}")
        print('-' * 40)


except FileNotFoundError:
    raise FileNotFoundError(f"Model file '{pipeline_filename}' not found.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

def preprocess_text(text):
    try:
        # Remove mentions, special characters, and digits
        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d', ' ', text)
        # Lowercase and strip whitespace
        text = text.lower().strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in stopwords.words('english')]
        
        # Join tokens back into a single string
        cleaned_text = ' '.join(tokens)
        return cleaned_text
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return text

def predict_sentiment(text):
    try:
        # Preprocess the text
        cleaned_text = preprocess_text(text)
        print(f"Cleaned text: {cleaned_text}")
        
        # Predict using the loaded model pipeline
        # Make sure to pass the text as a 2D array (list of lists)
        prediction = model_pipeline.predict([cleaned_text])
        
        return prediction[0]
    except Exception as e:
        print(f"Error predicting sentiment: {e}")
        return "Error"


@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        try:
            comment = request.form['text']
            sentiment = predict_sentiment(comment)
        except Exception as e:
            print(f"Error in POST request: {e}")
            sentiment = "Error"
    return render_template('sentiment_analysis.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
