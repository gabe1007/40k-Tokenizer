from flask import Flask, render_template, request, jsonify
from bpe import BPE
import tiktoken

app = Flask(__name__)

# Initialize the BPE tokenizer
bpe = BPE()

# Initialize the GPT-2 and GPT-3.5-Turbo tokenizers
gpt4_o_encoding = tiktoken.encoding_for_model("gpt-4o")
gpt2_encoding = tiktoken.encoding_for_model("gpt2")
gpt3_5_encoding = tiktoken.get_encoding("cl100k_base")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tokenize', methods=['POST'])
def tokenize():
    text = request.json.get('text', '')
    tokenizer = request.json.get('tokenizer', 'custom')

    if tokenizer == 'gpt-4o':
        encoded_tokens = gpt4_o_encoding.encode(text)
        tokens = []
        current_position = 0
        for token_id in encoded_tokens:
            token_text = gpt4_o_encoding.decode([token_id])
            tokens.append({
                "text": token_text,
                "length": len(token_text),
                "position": current_position
            })
            current_position += len(token_text)
    elif tokenizer == 'gpt2':
        encoded_tokens = gpt2_encoding.encode(text)
        tokens = []
        current_position = 0
        for token_id in encoded_tokens:
            token_text = gpt2_encoding.decode([token_id])
            tokens.append({
                "text": token_text,
                "length": len(token_text),
                "position": current_position
            })
            current_position += len(token_text)
    elif tokenizer == 'gpt-3.5-Turbo':
        encoded_tokens = gpt3_5_encoding.encode(text)
        tokens = []
        current_position = 0
        for token_id in encoded_tokens:
            token_text = gpt3_5_encoding.decode([token_id])
            tokens.append({
                "text": token_text,
                "length": len(token_text),
                "position": current_position
            })
            current_position += len(token_text)
    else:
        encoded_tokens = bpe.encode(text)
        tokens = []
        current_position = 0
        for token_id in encoded_tokens:
            token_text = bpe.decode([token_id])
            tokens.append({
                "text": token_text,
                "length": len(token_text),
                "position": current_position
            })
            current_position += len(token_text)

    token_count = len(encoded_tokens)

    return jsonify({
        'tokens': tokens,
        'encoded_tokens': encoded_tokens,
        'token_count': token_count
    })

if __name__ == '__main__':
    app.run(debug=True)