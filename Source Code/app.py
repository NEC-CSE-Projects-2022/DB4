import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import io
import os
import base64

from lime import lime_image
from skimage.segmentation import mark_boundaries

# --- CONFIG ---
MODEL_PATHS = {
    "EfficientNetV2S": "models/EfficientNetV2S_model.h5",
    "DenseNet121": "models/DenseNet121_model.h5",
    "InceptionV3": "models/InceptionV3_model.h5"  # Used for LIME
}
ENSEMBLE_WEIGHTS = {
    "EfficientNetV2S": 0.2,
    "DenseNet121": 0.4,
    "InceptionV3": 0.4
}
CLASS_LABELS = ['Monkeypox', 'Normal']
IMAGE_SIZE = (224, 224)

CUSTOM_OBJECTS = {
    "GlobalAveragePooling2D": tf.keras.layers.GlobalAveragePooling2D,
    "Dense": tf.keras.layers.Dense
}

models = {}
explainer = None  # Global LIME explainer

# --- LOAD MODELS ---
def load_all_models():
    global models, explainer
    print("* Loading models and LIME explainer...")
    explainer = lime_image.LimeImageExplainer()
    
    for name, path in MODEL_PATHS.items():
        full_path = os.path.join(os.path.dirname(__file__), path)
        try:
            models[name] = tf.keras.models.load_model(full_path, compile=False)
            print(f"  > Loaded {name}")
        except Exception as e:
            print(f"  > FAILED {name}: {e}")
    print("* All models loaded.")


# --- LIME EXPLANATION ---
def get_lime_explanation(img_array):
    inception_model = models['InceptionV3']
    classifier_fn = lambda x: inception_model.predict(x)
    
    explanation = explainer.explain_instance(
        image=img_array[0],
        classifier_fn=classifier_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    
    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        hide_rest=False,
        num_features=10
    )
    lime_img = mark_boundaries(temp, mask)
    
    img = Image.fromarray((lime_img * 255).astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# --- FLASK APP ---

app = Flask(__name__,
      template_folder='../../frontend/src/templates',
    static_folder='../../frontend/src/static'
    )

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/diagnose')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contactus.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not models:
        return jsonify({'error': 'Models not loaded'}), 500
    
    file = request.files['file']
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Ensemble prediction
        total_preds = np.zeros((1, len(CLASS_LABELS)))
        individual_scores = {}
        for name, model in models.items():
            pred = model.predict(img_array, verbose=0)
            total_preds += pred * ENSEMBLE_WEIGHTS[name]
            individual_scores[name] = float(pred[0][0])

        final_class_idx = np.argmax(total_preds, axis=1)[0]
        final_label = CLASS_LABELS[final_class_idx]
        ensemble_conf = float(np.max(total_preds))

        lime_b64 = get_lime_explanation(img_array)

        return jsonify({
            'status': 'success',
            'final_diagnosis': final_label,
            'confidence_scores': individual_scores,
            'ensemble_confidence': ensemble_conf,
            'lime_image_b64': lime_b64
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500


if __name__ == '__main__':
    load_all_models()
    app.run(debug=True)
