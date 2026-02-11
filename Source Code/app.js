document.getElementById('imageUpload').addEventListener('change', previewImage);
document.getElementById('uploadForm').addEventListener('submit', handleUpload);

function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function(){
        const output = document.getElementById('imagePreview');
        output.src = reader.result;
        output.style.display = 'block';
        document.getElementById('results').innerHTML = ''; // Clear old results
        document.getElementById('limeExplanation').style.display = 'none'; // Hide old LIME
        output.scrollIntoView({ behavior: 'smooth' });
    };
    reader.readAsDataURL(event.target.files[0]);
}

async function handleUpload(e) {
    e.preventDefault(); 

    const fileInput = document.getElementById('imageUpload');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const limeDiv = document.getElementById('limeExplanation');

    if (!fileInput.files.length) {
        resultsDiv.innerHTML = '<p class="error">Please select an image file.</p>';
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]); 

    loadingDiv.style.display = 'block';
    resultsDiv.innerHTML = ''; 
    limeDiv.style.display = 'none'; 

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.status === 'success') {
            const scores = data.confidence_scores;
            const finalDiagnosisText = data.final_diagnosis;
            const confidencePercent = (data.ensemble_confidence * 100).toFixed(2);
            
            const diagnosisColor = finalDiagnosisText === 'Monkeypox' ? 'red' : 'green';

            let html = `
                <h2 style="color: ${diagnosisColor};">
                  ${finalDiagnosisText === 'Monkeypox' 
                    ? '⚠️ Diagnosis: Monkeypox Detected' 
                    : '✅ Diagnosis: Normal Skin Condition'}
                </h2>

                <p>Threshold: <strong>0.5</strong> — Above means Monkeypox, below means Normal</p>
                
                <div class="score-card">
                    <h3>Individual Model Contributions (Confidence in Monkeypox)</h3>
                    <p>EfficientNetV2-S (0.2x): <span>${scores.EfficientNetV2S.toFixed(4)}</span></p>
                    <p>DenseNet121 (0.4x): <span>${scores.DenseNet121.toFixed(4)}</span></p>
                    <p>InceptionV3 (0.4x): <span>${scores.InceptionV3.toFixed(4)}</span></p>
                </div>
            `;
            resultsDiv.innerHTML = html;
            
            // --- LIME Display Logic ---
            if (data.lime_image_b64) {
                const imgTag = document.getElementById('limeImage');
                // The Base64 string is embedded directly into the src attribute
                imgTag.src = `data:image/png;base64,${data.lime_image_b64}`;
                limeDiv.style.display = 'block';
            }

        } else {
            resultsDiv.innerHTML = `<p class="error">Prediction Error: ${data.error}</p>`;
        }
    } catch (error) {
        console.error('Fetch Error:', error);
        resultsDiv.innerHTML = '<p class="error">A network error occurred. Is the server running?</p>';
    } finally {
        loadingDiv.style.display = 'none';
        resultsDiv.scrollIntoView({ behavior: 'smooth' });
    }
}