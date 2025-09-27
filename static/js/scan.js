// Face Scan JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const scanForm = document.getElementById('scanForm');
    const imageInput = document.getElementById('imageInput');
    const resultsContainer = document.getElementById('results');
    
    // Handle form submission
    scanForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        const file = imageInput.files[0];
        const scanType = document.querySelector('input[name="scanType"]:checked').value;
        
        if (!file) {
            FaceScan.showNotification('Please select an image file', 'error');
            return;
        }
        
        if (!FaceScan.validateImageFile(file)) {
            return;
        }
        
        formData.append('image', file);
        
        // Show loading state
        resultsContainer.innerHTML = '<div class="loading">Processing image...</div>';
        
        try {
            const endpoint = scanType === 'detect' ? '/api/v1/scan/detect' : '/api/v1/scan/recognize';
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                displayResults(result, scanType);
            } else {
                FaceScan.showNotification(result.error.message, 'error');
                resultsContainer.innerHTML = '<p class="error">Scan failed</p>';
            }
            
        } catch (error) {
            console.error('Scan error:', error);
            FaceScan.showNotification('Scan failed: ' + error.message, 'error');
            resultsContainer.innerHTML = '<p class="error">Scan failed</p>';
        }
    });
    
    // Handle file selection
    imageInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            // Show image preview
            const reader = new FileReader();
            reader.onload = function(e) {
                const preview = document.createElement('div');
                preview.className = 'image-preview';
                preview.innerHTML = `
                    <img src="${e.target.result}" alt="Preview" style="max-width: 100%; height: auto;">
                    <p>File: ${file.name} (${FaceScan.formatFileSize(file.size)})</p>
                `;
                
                // Remove existing preview
                const existingPreview = document.querySelector('.image-preview');
                if (existingPreview) {
                    existingPreview.remove();
                }
                
                // Add new preview
                resultsContainer.innerHTML = '';
                resultsContainer.appendChild(preview);
            };
            reader.readAsDataURL(file);
        }
    });
});

function displayResults(result, scanType) {
    const resultsContainer = document.getElementById('results');
    
    let html = `
        <div class="scan-results">
            <h3>Scan Results</h3>
            <div class="result-summary">
                <p><strong>Faces ${scanType === 'detect' ? 'Detected' : 'Recognized'}:</strong> ${result.faces_detected || result.faces_recognized}</p>
                <p><strong>Processing Time:</strong> ${result.processing_time}s</p>
            </div>
    `;
    
    if (result.faces && result.faces.length > 0) {
        html += '<div class="faces-list">';
        result.faces.forEach((face, index) => {
            html += `
                <div class="face-item">
                    <h4>Face ${face.id}</h4>
                    <p><strong>Location:</strong> (${face.bounding_box.x}, ${face.bounding_box.y}) - ${face.bounding_box.width}x${face.bounding_box.height}</p>
            `;
            
            if (scanType === 'recognize' && face.name) {
                html += `<p><strong>Name:</strong> ${face.name}</p>`;
                html += `<p><strong>Confidence:</strong> ${(face.confidence * 100).toFixed(1)}%</p>`;
            }
            
            html += '</div>';
        });
        html += '</div>';
    } else {
        html += '<p class="no-faces">No faces found in the image.</p>';
    }
    
    html += '</div>';
    
    resultsContainer.innerHTML = html;
}
