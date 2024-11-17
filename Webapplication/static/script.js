function predict() {
    var text = document.getElementById('text').value;

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
            "text": text
        })
    })
    .then(function(response) {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    })
    .then(function(data) {
        var category = data.category;
        console.log(category);
        document.getElementById('prediction').innerHTML = category[0].toUpperCase() + category.slice(1) +" - Accurecy 72 %";
    })
    .catch(function(error) {
        document.getElementById('prediction').innerHTML = 'Error: ' + error.message;
    });
}