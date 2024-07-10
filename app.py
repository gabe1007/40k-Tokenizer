from flask import Flask, render_template, request, jsonify
from bpe import BPE
import tiktoken

app = Flask(__name__)

# Initialize the BPE tokenizer
bpe = BPE()

# Initialize the GPT-2 and GPT-4 tokenizers
gpt4_o_encoding = tiktoken.encoding_for_model("gpt-4o")
gpt2_encoding = tiktoken.encoding_for_model("gpt2")
gpt3_5_encoding = tiktoken.get_encoding("cl100k_base")

@app.route('/')
def index():
    """
    Render the index.html template when accessing the root URL.
    """
    return render_template('index.html')

@app.route('/tokenize', methods=['POST'])
def tokenize():
    """
    Tokenize the provided text using the specified tokenizer.
    The tokenizer can be 'gpt-4o', 'gpt2', 'gpt-3.5-Turbo', or 'custom'.
    Returns the tokens, encoded tokens, and token count in JSON format.
    """
    text = request.json.get('text', '')
    tokenizer = request.json.get('tokenizer', 'custom')

    tokens = []
    current_position = 0

    # Tokenization process based on the selected tokenizer
    if tokenizer == 'gpt-4o':
        encoded_tokens = gpt4_o_encoding.encode(text)
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
