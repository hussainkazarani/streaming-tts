// ============================================================
// client.js — review-3
// ============================================================

const SAMPLE_RATE = 24000;

let audioContext  = null;
let nextStartTime = 0;

const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl     = `${protocol}//${window.location.host}/ws`;
let ws          = new WebSocket(wsUrl);
ws.binaryType   = 'arraybuffer';

// DOM
const statusDot  = document.getElementById('statusDot');
const statusSpan = document.getElementById('status');
const sendBtn    = document.getElementById('sendBtn');
const textInput  = document.getElementById('textInput');

// ----------------------------------------------------------------
// Auto-expanding textarea
// ----------------------------------------------------------------
textInput.addEventListener('input', () => {
    textInput.style.height = 'auto';
    textInput.style.height = textInput.scrollHeight + 'px';
});

// ----------------------------------------------------------------
// WebSocket lifecycle
// ----------------------------------------------------------------
ws.onopen = () => {
    setStatus('ready', 'Ready');
    sendBtn.disabled = false;
};
ws.onclose = () => setStatus('error', 'Disconnected');
ws.onerror = (err) => { console.error(err); setStatus('error', 'Error'); };

ws.onmessage = (event) => {
    const data = event.data;

    if (typeof data === 'string') {
        if (data === 'END_OF_AUDIO') {
            console.log('Audio generation complete.');
            // Keep button disabled until the queued audio finishes playing
            waitForPlaybackEnd();
        }
    } else if (data instanceof ArrayBuffer) {
        playAudioChunk(data);
    }
};

window.addEventListener("beforeunload", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close(1000, "User refreshed/closed tab");
    }
});

// ----------------------------------------------------------------
// Wait for all scheduled audio to finish, then re-enable button
// ----------------------------------------------------------------
function waitForPlaybackEnd() {
    if (!audioContext) {
        setStatus('ready', 'Ready');
        sendBtn.disabled = false;
        return;
    }

    const remaining = nextStartTime - audioContext.currentTime;
    const delay     = Math.max(0, remaining) * 1000 + 100; // +100ms buffer

    setTimeout(() => {
        setStatus('ready', 'Ready');
        sendBtn.disabled = false;
    }, delay);
}

// ----------------------------------------------------------------
// AudioContext
// ----------------------------------------------------------------
function initAudio() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: SAMPLE_RATE,
        });
    }
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
}

// ----------------------------------------------------------------
// Gapless PCM playback
// ----------------------------------------------------------------
function playAudioChunk(arrayBuffer) {
    const float32Data = new Float32Array(arrayBuffer);
    const audioBuffer = audioContext.createBuffer(1, float32Data.length, SAMPLE_RATE);
    audioBuffer.getChannelData(0).set(float32Data);

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    const now = audioContext.currentTime;
    if (nextStartTime < now) nextStartTime = now;

    source.start(nextStartTime);
    nextStartTime += audioBuffer.duration;
}

// ----------------------------------------------------------------
// Send
// ----------------------------------------------------------------
sendBtn.onclick = () => {
    const text = textInput.value.trim();
    if (!text || sendBtn.disabled) return;

    initAudio();
    nextStartTime = audioContext.currentTime;

    setStatus('busy', 'Generating...');
    sendBtn.disabled = true;

    ws.send(text);
};

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------
function setStatus(dot, text) {
    statusDot.className  = `status-dot ${dot}`;
    statusSpan.textContent = text;
}