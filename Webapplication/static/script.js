
document.getElementById('file').addEventListener('change', function(event) {
    var filePath = event.target.value;
    console.log('Selected PDF path:', filePath);
});

function predict() {
    var text = document.getElementById('text').value;
    var modelSelect = document.getElementById('model-select');
    var model = modelSelect.value;
    var Accuracy = 0;

    if(model === 'random_forest') {
        Accuracy = 72;
    }
    else if(model === 'gbm') {
        Accuracy = 70;
    }
    else if(model === 'svm') {
        Accuracy = 73;
    }


    if (text === '') {
        document.getElementById('prediction').innerHTML = 'Please enter some text.';
        return;
    }

    fetch('/text-classification', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            "text": text,
            "model": model
        })
    })
    .then(function(response) {
        if (!response.ok) {
            throw new Error('Network response was not ok: ' + response.status);
        }
        return response.json();
    })
    .then(function(data) {
        var category = data.category;
        console.log(category);
        if(model === 'random_forest') {
            document.getElementById('prediction').innerHTML = category.toString().toUpperCase() + " - ACCURACY " + Accuracy + " %";
        }else{
            document.getElementById('prediction').innerHTML = category.toUpperCase() + " - ACCURACY " + Accuracy + " %";
        }
    })
    .catch(function(error) {
        if (error.message.includes('500')) {
            document.getElementById('prediction').innerHTML = 'Error: Internal Server Error';
        } else {
            document.getElementById('prediction').innerHTML = 'Error: ' + error.message;
        }
    });
}



