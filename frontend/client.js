const SAMPLE_RATE = 24000;
let audioContext  = null;
let nextStartTime = 0;
let ws = null;
let selectedVoice = null;

// DOM Elements
const statusSpan = document.getElementById('status');
const statusDot = document.getElementById('statusDot');
const sendBtn = document.getElementById('sendBtn');
const textInput = document.getElementById('textInput');
const voiceList = document.getElementById('voiceList');

// ==============================================================================
// STATUS HELPER
// ==============================================================================
function setStatus(text, state) {
    // state: 'ready' | 'busy' | 'error' | ''
    statusSpan.innerText = text;
    statusDot.className  = 'status-dot' + (state ? ' ' + state : '');
}

// ==============================================================================
// 1. LOAD VOICES
// Fetches /api/voices and populates the voice button list.
// ==============================================================================
async function loadVoices() {
    try {
        const res = await fetch('/api/voices');
        const voices = await res.json();

        voiceList.innerHTML = '';

        if (!voices || voices.length === 0) {
            voiceList.innerHTML = '<span class="voice-loading">No voices found — add a .wav + .txt to the voices/ folder.</span>';
            return;
        }

        voices.forEach((v, i) => {
            const btn        = document.createElement('button');
            btn.className    = 'voice-btn';
            btn.textContent  = v.name;
            btn.dataset.name = v.name;
            btn.onclick      = () => selectVoice(v.name, btn);
            voiceList.appendChild(btn);

            // Auto-select first voice on load
            if (i === 0) selectVoice(v.name, btn);
        });

        addPlusButton();

    } catch (e) {
        console.error('Failed to load voices:', e);
        voiceList.innerHTML = '<span class="voice-loading" style="color:#ef4444">Failed to load voices from server.</span>';
        setStatus('Error loading voices', 'error');
    }
}

// ==============================================================================
// 2. VOICE SELECTION
// Closes old websocket, opens new one to /api/{voice_name}
// ==============================================================================
function openVoiceSocket(name) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws             = new WebSocket(`${protocol}//${window.location.host}/api/${name}`);
    ws.binaryType  = 'arraybuffer';

    ws.onopen = () => {
        setStatus(`Ready (${name})`, 'ready');
        sendBtn.disabled = false;
    };

    ws.onclose = () => {
        if (pendingVoice) {
            const next = pendingVoice;
            pendingVoice = null;
            if (audioContext) { audioContext.close(); audioContext = null; }
            nextStartTime = 0;
            openVoiceSocket(next);
        } else {
            setStatus('Disconnected', '');
            sendBtn.disabled = true;
            if (audioContext) { audioContext.close(); audioContext = null; }
            nextStartTime = 0;
        }
    };

    ws.onerror = (err) => {
        console.error(err);
        setStatus('Connection error', 'error');
    };

    ws.onmessage = async (event) => {
        if (typeof event.data === 'string') {
            if (event.data === 'END_OF_AUDIO') {
                const now = audioContext ? audioContext.currentTime : 0;
                const timeRemaining = Math.max(0, nextStartTime - now);
                if (timeRemaining > 0) {
                    setStatus(`Playing (${name})...`, 'busy');
                    setTimeout(() => {
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            setStatus(`Ready (${name})`, 'ready');
                            sendBtn.disabled = false;
                        }
                    }, timeRemaining * 1000);
                } else {
                    setStatus(`Ready (${name})`, 'ready');
                    sendBtn.disabled = false;
                }
            }
        } else if (event.data instanceof ArrayBuffer) {
            playAudioChunk(event.data);
        }
    };
}

let pendingVoice = null; // voice name waiting to connect after old WS closes

function selectVoice(name, btnEl) {
    document.querySelectorAll('.voice-btn').forEach(b => b.classList.remove('active'));
    btnEl.classList.add('active');

    selectedVoice    = name;
    sendBtn.disabled = true;
    setStatus(`Connecting to ${name}...`, '');

    if (ws && ws.readyState !== WebSocket.CLOSED) {
        // Mark which voice to open once this socket finishes closing
        pendingVoice = name;
        ws.close();
        // onclose will call openVoiceSocket(name) once the server releases the lock
    } else {
        openVoiceSocket(name);
    }
}

// ==============================================================================
// 3. AUDIO PLAYBACK
// ==============================================================================
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

function playAudioChunk(arrayBuffer) {
    const float32Data = new Float32Array(arrayBuffer);
    const audioBuffer = audioContext.createBuffer(1, float32Data.length, SAMPLE_RATE);
    audioBuffer.getChannelData(0).set(float32Data);

    const source  = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    const now = audioContext.currentTime;
    if (nextStartTime < now) nextStartTime = now;

    source.start(nextStartTime);
    nextStartTime += audioBuffer.duration;
}

// ==============================================================================
// 4. SEND
// ==============================================================================
sendBtn.onclick = () => {
    const text = textInput.value.trim();
    if (!text) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        setStatus('Not connected — select a voice first', 'error');
        return;
    }

    initAudio();
    nextStartTime    = audioContext.currentTime;
    sendBtn.disabled = true;
    setStatus(`Streaming (${selectedVoice})...`, 'busy');
    ws.send(text);
};

// Auto-expand textarea as user types
textInput.addEventListener('input', function() {
    this.style.height = 'auto'; // Reset height briefly to recalculate
    this.style.height = (this.scrollHeight + 2) + 'px'; // Set to scroll height (+2 for borders)
});

// ==============================================================================
// 5. ADD VOICE MODAL
// ==============================================================================
function addPlusButton() {
    const btn         = document.createElement('button');
    btn.className     = 'voice-btn-add';
    btn.title         = 'Add voice';
    btn.textContent   = '+';
    btn.onclick       = openModal;
    voiceList.appendChild(btn);
}

function openModal() {
    document.getElementById('modalOverlay').classList.add('open');
    document.getElementById('voiceName').focus();
    document.getElementById('modalStatus').textContent  = '';
    document.getElementById('modalUpload').disabled     = false;
}

function closeModal() {
    document.getElementById('modalOverlay').classList.remove('open');
    document.getElementById('voiceName').value          = '';
    document.getElementById('voiceTranscript').value    = '';
    document.getElementById('voiceFile').value          = '';
    document.getElementById('modalStatus').textContent  = '';
}

document.getElementById('modalCancel').onclick = closeModal;

document.getElementById('modalOverlay').onclick = (e) => {
    if (e.target === document.getElementById('modalOverlay')) closeModal();
};

document.getElementById('modalUpload').onclick = async () => {
    const name        = document.getElementById('voiceName').value.trim();
    const transcript  = document.getElementById('voiceTranscript').value.trim();
    const file        = document.getElementById('voiceFile').files[0];
    const modalStatus = document.getElementById('modalStatus');
    const modalUpload = document.getElementById('modalUpload');

    if (!name || !transcript || !file) {
        modalStatus.textContent = 'Please fill in all fields and select a .wav file.';
        modalStatus.style.color = '#ef4444';
        return;
    }

    modalStatus.textContent = 'Uploading...';
    modalStatus.style.color = '#666';
    modalUpload.disabled    = true;

    const form = new FormData();
    form.append('name', name);
    form.append('transcript', transcript);
    form.append('file', file);

    try {
        const res  = await fetch('/api/voices/upload', { method: 'POST', body: form });
        const data = await res.json();

        if (data.success) {
            // Insert new pill before the + button
            const plusBtn    = voiceList.querySelector('.voice-btn-add');
            const btn        = document.createElement('button');
            btn.className    = 'voice-btn';
            btn.textContent  = name;
            btn.dataset.name = name;
            btn.onclick      = () => selectVoice(name, btn);
            voiceList.insertBefore(btn, plusBtn);
            closeModal();
            selectVoice(name, btn);
        } else {
            modalStatus.textContent = `Error: ${data.error}`;
            modalStatus.style.color = '#ef4444';
            modalUpload.disabled    = false;
        }
    } catch (e) {
        modalStatus.textContent = 'Upload failed — server error.';
        modalStatus.style.color = '#ef4444';
        modalUpload.disabled    = false;
    }
};

// ==============================================================================
// INIT
// ==============================================================================
loadVoices();