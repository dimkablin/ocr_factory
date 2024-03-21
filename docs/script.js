const btnSendPhoto = document.querySelector('#btnSendPhoto')
const inpImg = document.querySelector('#inpImg')
const divContentFiles = document.querySelector('#contentFiles')

const addElementOcr = (key, value) => `
    <div>
        <h3>${key}</h3>
        <p>${value}</p>
    </div>
`
const applyTextBlockStyles = (textBlock) => {
    textBlock.style.display = 'flex';
    textBlock.style.justifyContent = 'center';
    textBlock.style.alignItems = 'center';
    textBlock.style.textAlign = 'center';
    textBlock.style.backgroundColor = 'rgba(255, 255, 255, 0.5)';
    textBlock.style.borderRadius = '12px'; 
    textBlock.style.fontWeight = 'bold';
};

function result2show(image, result) {
    // Create canvas elements
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const canvas2 = document.createElement('canvas');
    const ctx2 = canvas2.getContext('2d');
    const div = document.createElement('div');
    div.style.display = 'flex';
    div.style.width = '100%';

    // Set canvas sizes
    canvas.width = image.width;
    canvas.height = image.height;
    canvas2.width = image.width;
    canvas2.height = image.height;

    // Draw original image on the first canvas
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    // Draw empty white image on the second canvas
    ctx2.fillStyle = '#FFFFFF'; // White color
    ctx2.fillRect(0, 0, canvas2.width, canvas2.height);

    // Create container for canvas elements
    const container = document.getElementById('imageContainer');
    container.appendChild(canvas);
    container.appendChild(canvas2);

    // Iterate through the result dictionary
    for (let i = 0; i < result.rec_texts.length; i++) {
        const rec_texts = result.rec_texts[i];
        const rec_scores = result.rec_scores[i];
        const det_polygons = result.det_polygons[i];
        const det_scores = result.det_scores[i];

        const x_coord = det_polygons.filter((_, index) => index % 2 === 0);
        const y_coord = det_polygons.filter((_, index) => index % 2 !== 0);

        // Draw detection polygons on the first canvas
        ctx.beginPath();
        ctx.moveTo(x_coord[0], y_coord[0]);
        for (let j = 1; j < x_coord.length; j++) {
            ctx.lineTo(x_coord[j], y_coord[j]);
        }
        ctx.closePath();
        ctx.strokeStyle = `rgba(0, ${Math.round(255 * (1 - det_scores))}, 0, 1)`; // Green color for edge
        ctx.lineWidth = 1;
        ctx.stroke();

        // Draw recognition polygons and text on the second canvas
        ctx2.beginPath();
        ctx2.moveTo(x_coord[0], y_coord[0]);
        for (let j = 1; j < x_coord.length; j++) {
            ctx2.lineTo(x_coord[j], y_coord[j]);
        }
        ctx2.closePath();
        ctx2.fillStyle = `rgba(0, ${Math.round(255 * (1 - rec_scores / 2))}, 0, 0.5)`; // Green color for fill
        ctx2.fill();
        ctx2.strokeStyle = 'green'; // Green color for edge
        ctx2.lineWidth = 1;
        ctx2.stroke();

        // Draw text on the second canvas
        ctx2.fillStyle = 'black'; // Black color for text
        ctx2.font = '12px Arial';
        ctx2.textBaseline = 'middle';
        ctx2.textAlign = 'left';
        ctx2.fillText(rec_texts.replace('$', '\\$'), x_coord[0], y_coord.reduce((acc, val) => acc + val, 0) / y_coord.length);
    }
}


btnSendPhoto.addEventListener('click', e => {
    e.preventDefault();
    const formData = new FormData();
    Array.from(inpImg.files).forEach(file => {
        formData.append('image', file);
    });
    fetch(window.origin + '/ocr/', {
        method: 'POST',
        body: formData
    })
        .then(resp => resp.json())
        .then(ocr => {
            const imageContainer = document.querySelector('#imageContainer');
            const textBlocks = document.querySelector('#textBlocks');
        
            // Clear previous text blocks
            textBlocks.innerHTML = '';
        
            // Insert uploaded image
            const uploadedImage = document.querySelector('#uploadedImage');
            uploadedImage.src = URL.createObjectURL(inpImg.files[0]);
            uploadedImage.onload = () => {   

                result2show(uploadedImage, ocr);
            };
        })
        .catch(error => {
            console.error('Error:', error);
        });
});


// Function to fetch model names
const fetchModelNames = () => {
    fetch(window.origin +'/get-model-names')
        .then(response => response.json())
        .then(data => {
            const modelSelect = document.getElementById('modelSelect');
            data.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
            // Add event listener to handle model change
            modelSelect.addEventListener('change', () => {
                const selectedModel = modelSelect.value;
                changeModel(selectedModel);
            });
        })
        .catch(error => {
            console.error('Error fetching model names:', error);
        });
};

// Function to change the model
const changeModel = (modelName) => {
    fetch(window.origin + `/change-model?model_name=${encodeURIComponent(modelName)}`)
        .then(response => {
            if (response.ok) {
                console.log(`Model changed to ${modelName}`);
            } else {
                console.error('Failed to change model:', response.statusText);
            }
        })
        .catch(error => {
            console.error('Error changing model:', error);
        });
};

// Call fetchModelNames to populate the model selection block
fetchModelNames();
