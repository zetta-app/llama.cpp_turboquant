curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.5-9B","messages":[{"role":"user","content":"Hey"}],"max_tokens":256}' | python3 -m json.tool
