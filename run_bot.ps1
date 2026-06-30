# Set your xAI API key here (only needed when LLM analysis is re-enabled)
# $env:XAI_API_KEY = "your-xai-api-key-here"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# Start dashboard server in background
Start-Process python -ArgumentList "C:\Users\joelt\server.py" -WindowStyle Minimized

Write-Host "[*] Dashboard: http://localhost:5000"
Start-Sleep -Seconds 2

# Start the bot
python C:\Users\joelt\zoom_transcribe_bot_llm.py
pause
