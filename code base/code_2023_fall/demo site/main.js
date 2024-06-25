let fileEl = null;


addEventListeners();




// Functions below
function addEventListeners() {
    document.getElementById('fileInput').addEventListener('change', openFileDialogue);
    document.getElementById('makeTriples').addEventListener('click', createTriples);
}


async function createTriples() {
    if (fileEl) {
        // console.log(fileEl);
        try {
            console.log("Creating triples...");

            const response = await fetch('/runPyScript', { method: 'POST' });
            const result = await response.text();
            console.log(result);
        } catch {
            console.error('Error:', error);
        }

    } else {
        document.getElementById('output').innerHTML = 'Please select a file first.';
    }
}


function openFileDialogue() {
    document.getElementById('fileInput').click();
    fileEl = document.getElementById('fileInput').files[0];
}
