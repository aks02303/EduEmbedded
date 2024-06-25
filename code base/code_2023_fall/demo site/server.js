const { exec } = require('child_process');
const express = require('express');
const app = express();
const http = require('http');
const port = 3000;

app.use(express.static('/public'));

app.post('/runPyScript', (req, res) => {
    const server = http.createServer((req, res) => {
        res.writeHead(200, {'Content-Type': 'text/plain'});
    
        // Replace 'your_script.sh' with the path to your shell script
        const shellScript = 'runPyScript.sh';
    
        exec(`sh ${shellScript}`, (error, stdout, stderr) => {
            if (error) {
                res.end(`Error: ${error.message}`);
                return;
            }
            if (stderr) {
                res.end(`stderr: ${stderr}`);
                return;
            }
            res.end(`stdout: ${stdout}`);
        });
    });
});


app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}/`);
});