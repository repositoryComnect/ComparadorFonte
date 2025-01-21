import os
import difflib
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import utils  # Importa o arquivo utils.py
from gpt4all import GPT4All  # Importa o modelo Llama (via GPT-4 ou outro modelo similar)

# Inicializar o modelo CodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")

# Carregar o modelo Llama (ou GPT-4)
llama_model_path = r"C:\Users\lolegario\Desktop\Projetos\BotComnect\model\Llama-3.2-1B-Instruct-Q4_0.gguf"  
llama_model = GPT4All(llama_model_path)

# Criar a aplicação Flask
app = Flask(__name__)

def gerar_resposta_llama(codigo):
    """Função para gerar uma resposta detalhada usando o modelo Llama (GPT-4)"""
    prompt = f"Analisando o código abaixo, forneça uma explicação detalhada e sugira melhorias onde possível.\n\nCódigo:\n{codigo}\n\nExplicação:"
    with llama_model.chat_session() as session:
        response = session.generate(prompt, max_tokens=512)
        return response

@app.route('/compare', methods=['POST'])
def compare():
    try:
        data = request.get_json()
        codigo1 = data.get("codigo1")
        codigo2 = data.get("codigo2")

        if not codigo1 or not codigo2:
            return jsonify({"error": "Ambos os códigos devem ser fornecidos."}), 400

        # Calcular similaridade para os códigos originais
        inputs_original = tokenizer(codigo1, codigo2, return_tensors="pt", padding=True, truncation=True)
        outputs_original = model(**inputs_original)
        logits_original = outputs_original.logits
        similarity_original = torch.softmax(logits_original, dim=1)[0][1].item()

        codigo1_normalizado = utils.normalizar_codigo(codigo1)
        codigo2_normalizado = utils.normalizar_codigo(codigo2)

        # Calcular similaridade para os códigos normalizados
        inputs_normalizado = tokenizer(codigo1_normalizado, codigo2_normalizado, return_tensors="pt", padding=True, truncation=True)
        outputs_normalizado = model(**inputs_normalizado)
        logits_normalizado = outputs_normalizado.logits
        similarity_normalizado = torch.softmax(logits_normalizado, dim=1)[0][1].item()

        semantica_similarity = utils.comparar_semantica(codigo1_normalizado, codigo2_normalizado)  # Basear na versão normalizada
        diferencas_codigo_original = utils.comparar_diferencas(codigo1, codigo2)

        # Comparar as diferenças entre os códigos normalizados
        diferencas_codigo_normalizado = utils.comparar_diferencas(codigo1_normalizado, codigo2_normalizado)

        # Ajuste na lógica de similaridade final
        if diferencas_codigo_original == "Sem diferencas." and diferencas_codigo_normalizado == "Sem diferencas.":
            similarity_final = 1.0  # Similaridade máxima caso não haja diferenças
        else:
            # Média ponderada das similaridades com penalização adicional
            similarity_final = (0.5 * similarity_original) + (0.3 * similarity_normalizado) + (0.2 * semantica_similarity)

        # Verificar se há uma diferença clara no tipo de operação (como soma vs concatenação)
        if "str" in codigo1 and "str" in codigo2:
            similarity_final -= 0.4  # Penalização adicional para concatenação de strings

        # Garantir que a similaridade final esteja no intervalo [0, 1]
        similarity_final = max(0, min(similarity_final, 1))

        # Gerar uma explicação enriquecida usando o modelo Llama
        explicacao_llama = gerar_resposta_llama(codigo1)

        # Sugerir melhorias caso a similaridade seja baixa
        melhorias = utils.sugerir_melhorias(similarity_final)

        return jsonify({
            "similarity_original": similarity_original,
            "similarity_normalizado": similarity_normalizado,
            "semantica_similarity": f'{semantica_similarity * 100} %',
            "similarity_final": similarity_final,
            "diferencas_codigo_original": diferencas_codigo_original,
            "sugestoes_melhorias": melhorias,
            "explicacao_llama": explicacao_llama  # Resposta enriquecida
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
