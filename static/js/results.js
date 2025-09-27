// Results page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Load results from URL parameters or localStorage
    loadResults();
});

function loadResults() {
    // Try to get results from URL parameters first
    const urlParams = new URLSearchParams(window.location.search);
    const resultsData = urlParams.get('results');
    
    if (resultsData) {
        try {
            const results = JSON.parse(decodeURIComponent(resultsData));
            displayResults(results);
        } catch (error) {
            console.error('Error parsing results:', error);
            FaceScan.showNotification('Error loading results', 'error');
        }
    } else {
        // Try to get from localStorage
        const savedResults = localStorage.getItem('scanResults');
        if (savedResults) {
            try {
                const results = JSON.parse(savedResults);
                displayResults(results);
            } catch (error) {
                console.error('Error loading saved results:', error);
            }
        } else {
            // Show placeholder
            document.getElementById('facesCount').textContent = '0';
            document.getElementById('processingTime').textContent = '0ms';
            document.getElementById('scanType').textContent = 'Detection';
        }
    }
}

function displayResults(results) {
    // Update summary stats
    document.getElementById('facesCount').textContent = results.faces_detected || results.faces_recognized || 0;
    document.getElementById('processingTime').textContent = `${(results.processing_time * 1000).toFixed(0)}ms`;
    document.getElementById('scanType').textContent = results.faces_recognized ? 'Recognition' : 'Detection';
    
    // Display face details
    const faceDetails = document.getElementById('faceDetails');
    
    if (results.faces && results.faces.length > 0) {
        let html = '';
        results.faces.forEach((face, index) => {
            html += `
                <div class="face-detail">
                    <h4>Face ${face.id}</h4>
                    <p><strong>Location:</strong> (${face.bounding_box.x}, ${face.bounding_box.y})</p>
                    <p><strong>Size:</strong> ${face.bounding_box.width} x ${face.bounding_box.height} pixels</p>
            `;
            
            if (face.name) {
                html += `<p><strong>Name:</strong> ${face.name}</p>`;
            }
            
            if (face.confidence !== undefined) {
                html += `<p><strong>Confidence:</strong> ${(face.confidence * 100).toFixed(1)}%</p>`;
            }
            
            html += '</div>';
        });
        faceDetails.innerHTML = html;
    } else {
        faceDetails.innerHTML = '<p class="placeholder">No faces detected</p>';
    }
}

function downloadResults() {
    const resultsData = localStorage.getItem('scanResults');
    if (resultsData) {
        const blob = new Blob([resultsData], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `scan_results_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        FaceScan.showNotification('Results downloaded successfully', 'success');
    } else {
        FaceScan.showNotification('No results to download', 'warning');
    }
}

function drawFaceBoxesOnImage(imageSrc, faces) {
    const image = new Image();
    image.onload = function() {
        const canvas = document.getElementById('faceCanvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = image.width;
        canvas.height = image.height;
        
        // Draw the image
        ctx.drawImage(image, 0, 0);
        
        // Draw face boxes
        faces.forEach((face, index) => {
            const { x, y, width, height } = face.bounding_box;
            
            // Draw rectangle
            ctx.strokeStyle = '#e74c3c';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, width, height);
            
            // Draw label
            ctx.fillStyle = '#e74c3c';
            ctx.font = '16px Arial';
            ctx.fillText(`Face ${index + 1}`, x, y - 10);
            
            // Draw name if available
            if (face.name && face.name !== 'Unknown') {
                ctx.fillText(face.name, x, y + height + 20);
            }
        });
        
        canvas.style.display = 'block';
    };
    image.src = imageSrc;
}
