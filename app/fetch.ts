import fs from 'fs';

async function run() {
    try {
        const res = await fetch('https://huggingface.co/hexgrad/Kokoro-82M/raw/main/voices.json');
        if (!res.ok) {
            console.log("Failed", res.status);
            return;
        }
        const data = await res.json();
        const keys = Object.keys(data);
        console.log(keys.join(", "));
    } catch (e) {
        console.log("Error", e);
    }
}
run();
