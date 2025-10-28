# ShivX UI/UX Quick Start Guide

## Installation

### 1. Backend Setup

```bash
cd /home/user/shivx

# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies for voice
sudo apt-get install -y ffmpeg espeak libespeak-dev

# Set environment variables
export SHIVX_ENV=development
export SHIVX_JWT_SECRET_KEY=your-secret-key-here
```

### 2. Frontend Setup

```bash
cd /home/user/shivx/frontend

# Install npm dependencies (already done)
npm install

# For development
npm run dev

# For production build
npm run build
```

## Running the Application

### Development Mode

**Terminal 1 - Backend:**
```bash
cd /home/user/shivx
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd /home/user/shivx/frontend
npm run dev
# Opens at http://localhost:3000
```

### Production Mode

**Backend:**
```bash
cd /home/user/shivx
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Frontend:**
```bash
cd /home/user/shivx/frontend
npm run build
npx serve -s dist -p 3000
```

## Testing the Features

### 1. Test Dashboard

1. Open browser to `http://localhost:3000`
2. Login (default: demo@shivx.ai / password)
3. Dashboard should show:
   - Agent count
   - Task statistics
   - System health metrics
   - Real-time updates

### 2. Test WebSocket

**Open browser console:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws?token=YOUR_JWT_TOKEN');
ws.onmessage = (event) => console.log('Message:', event.data);
```

**Verify:**
- Connection established
- Ping/pong messages every 30s
- Real-time updates appear

### 3. Test Voice Input

1. Navigate to Settings page
2. Click "Test Voice Input" button
3. Allow microphone access
4. Speak: "Show me the trading dashboard"
5. Verify transcription appears
6. Check audio playback of response

### 4. Test Emotion Detection

**Using API:**
```bash
curl -X POST http://localhost:8000/api/voice/emotion \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so frustrated with this error"}'
```

**Expected Response:**
```json
{
  "primary_emotion": "frustration",
  "sentiment": "negative",
  "confidence": 0.85,
  "emotions": {
    "frustration": 0.85,
    "anger": 0.65,
    "neutral": 0.15
  },
  "indicators": ["frustrated", "error"]
}
```

### 5. Test Keyboard Shortcuts

1. Press `g` then `d` → Navigate to Dashboard
2. Press `g` then `a` → Navigate to Agents
3. Press `Ctrl+K` (or `Cmd+K`) → Open command palette
4. Press `Ctrl+/` → Show shortcuts help
5. Press `Tab` → Navigate through focusable elements

## Troubleshooting

### Frontend won't build

```bash
cd /home/user/shivx/frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### WebSocket connection fails

1. Check backend is running on port 8000
2. Verify JWT token is valid
3. Check CORS settings in `main.py`
4. Inspect browser console for errors

### Voice transcription fails

1. Verify Whisper model downloaded:
   ```python
   import whisper
   whisper.load_model("base")  # Downloads on first run
   ```

2. Check audio file format (WAV recommended)
3. Verify ffmpeg is installed: `ffmpeg -version`

### Emotion detection returns "neutral" always

1. Check text has emotional content
2. Verify VADER sentiment installed:
   ```bash
   pip install vaderSentiment
   ```
3. Test with strong emotional text:
   - "I am extremely happy!" → joy
   - "This is terrible!" → sadness
   - "I am very frustrated!" → frustration

## Development Tips

### Hot Reload

Frontend hot reload works out of the box with Vite. Backend hot reload requires `--reload` flag:

```bash
uvicorn main:app --reload
```

### Debugging

**Frontend:**
- React DevTools extension
- Redux DevTools for Zustand
- Network tab for API calls
- Console for WebSocket messages

**Backend:**
- Add `import pdb; pdb.set_trace()` for breakpoints
- Check logs in terminal
- Use `/api/docs` for Swagger UI (dev mode only)

### Testing

**Backend:**
```bash
pytest tests/test_websocket.py -v
pytest tests/test_voice.py -v
pytest tests/test_emotion.py -v
```

**Frontend:**
```bash
cd frontend
npm test
npm run test:coverage
```

## Environment Variables

### Backend (.env)

```env
SHIVX_ENV=development
SHIVX_VERSION=2.0.0
SHIVX_HOST=0.0.0.0
SHIVX_PORT=8000
SHIVX_RELOAD=true

# JWT
SHIVX_JWT_SECRET_KEY=your-256-bit-secret
SHIVX_JWT_ALGORITHM=HS256
SHIVX_JWT_EXPIRE_MINUTES=1440

# CORS
SHIVX_CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Trading
SHIVX_TRADING_MODE=paper

# Privacy
SHIVX_OFFLINE_MODE=false
SHIVX_AIRGAP_MODE=false
SHIVX_GDPR_MODE=true
```

### Frontend (.env)

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_ENV=development
```

## Production Checklist

- [ ] Set `SHIVX_ENV=production`
- [ ] Use strong `JWT_SECRET_KEY` (openssl rand -hex 32)
- [ ] Configure proper `CORS_ORIGINS` (no wildcard)
- [ ] Enable HTTPS/WSS
- [ ] Set up reverse proxy (nginx)
- [ ] Configure rate limiting
- [ ] Enable monitoring (Prometheus)
- [ ] Set up error tracking (Sentry)
- [ ] Configure backup strategy
- [ ] Test disaster recovery
- [ ] Document deployment process

## Useful Commands

```bash
# Backend
uvicorn main:app --reload                # Development
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker  # Production
pytest tests/ -v --cov                   # Run tests with coverage
python -m pip list --outdated            # Check outdated packages

# Frontend
npm run dev                              # Development server
npm run build                            # Production build
npm run preview                          # Preview production build
npm test                                 # Run tests
npm run lint                             # Run linter
npm run type-check                       # Check TypeScript types

# Docker (optional)
docker-compose up -d                     # Start all services
docker-compose logs -f backend           # View backend logs
docker-compose down                      # Stop all services
```

## Resources

- **Full Documentation**: `/home/user/shivx/docs/UI_UX_COMPLETION_REPORT.md`
- **API Documentation**: http://localhost:8000/api/docs (dev mode)
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000
- **WebSocket**: ws://localhost:8000/ws

## Support

For issues or questions:
1. Check the full documentation
2. Review error logs
3. Test with sample data
4. Check WebSocket connection
5. Verify dependencies installed

---

**Last Updated:** 2025-10-28
**Version:** 2.0.0
