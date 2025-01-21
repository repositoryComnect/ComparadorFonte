import re

# Função para normalizar o código
def normalizar_codigo(codigo):
    codigo = re.sub(r'public\s+class\s+\w+', 'public class CLASSNAME', codigo)
    codigo = re.sub(r'\b(int|String|double|float|char|boolean)\s+\w+', '\1 VAR', codigo)  # Substitui nomes de variáveis por placeholders
    codigo = re.sub(r'(\b\w+\s+)\w+\s*\(', '\1FUNCNAME(', codigo)  # Substitui nomes de métodos
    codigo = re.sub(r'\s+', ' ', codigo)  # Remove espaços extras
    codigo = re.sub(r'//.*|/\*.*?\*/', '', codigo, flags=re.DOTALL)  # Remove comentários
    codigo = re.sub(r'\b\d+\b', 'NUMBER', codigo)  # Substitui números por um placeholder
    return codigo

# Função para extrair operações chave
def extrair_operacoes(codigo):
    operacoes = []
    operacoes += re.findall(r'\b(for|while|if|else|switch|case)\b', codigo)
    operacoes += re.findall(r'\b\w+\s*\(', codigo)
    operacoes += re.findall(r'[+\-*/><]', codigo)  # Captura operadores matemáticos
    return set(operacoes)

# Função para comparar a semântica do código, levando em consideração tipos de dados
def comparar_semantica(codigo1, codigo2):
    operacoes1 = extrair_operacoes(codigo1)
    operacoes2 = extrair_operacoes(codigo2)

    # Calcular a similaridade semântica com base nas operações encontradas
    similaridade_operacoes = len(operacoes1.intersection(operacoes2)) / max(len(operacoes1), len(operacoes2)) if operacoes1 and operacoes2 else 1.0

    # Verificar se há diferença nos tipos de dados
    tipos1 = re.findall(r'\b(int|float|str)\b', codigo1)
    tipos2 = re.findall(r'\b(int|float|str)\b', codigo2)
    if set(tipos1) != set(tipos2):
        # Penalizar fortemente se os tipos de dados forem diferentes
        similaridade_operacoes -= 0.5

    # Penalizar diferenças nos operadores matemáticos
    operadores1 = re.findall(r'[+\-*/]', codigo1)
    operadores2 = re.findall(r'[+\-*/]', codigo2)
    diferenca_operadores = len(set(operadores1).symmetric_difference(set(operadores2)))

    if diferenca_operadores > 0:
        # A penalização agora é mais forte para diferenças de operadores
        similaridade_operacoes -= 0.4 * diferenca_operadores / max(len(operadores1), len(operadores2), 1)

    return max(similaridade_operacoes, 0)  # Garantir que não seja negativo

# Função para sugerir melhorias no código original quando a similaridade for baixa
def sugerir_melhorias(similarity_score):
    if similarity_score < 0.45:
        return "A similaridade entre os codigos esta baixa. Verifique os seguintes pontos no codigo original para melhorar a clareza e legibilidade:\n" \
               "- Nomes de variaveis e funcoes: Considere usar nomes mais descritivos e que reflitam claramente o proposito de cada variavel ou funcao. Evite nomes genericos como temp, x, data, e prefira algo mais informativo como userInput ou orderList.\n" \
               "- Modularizacao e reutilizacao: Tente dividir o codigo em funcoes menores e mais reutilizaveis. Funcoes grandes e complexas devem ser divididas em partes menores, cada uma com uma responsabilidade unica. Isso facilita a manutencao e melhora a legibilidade.\n" \
               "- Estrutura de codigo e formacao: Verifique se a formacao do codigo segue um estilo consistente (ex: indentacao, espaco entre funcoes e blocos de codigo). Utilize ferramentas de formacao automatica como black para Python, prettier para JavaScript, etc. Alem disso, evite linhas de codigo muito longas (tente manter com menos de 80-100 caracteres por linha).\n" \
               "- Uso de tipos de dados adequados: Certifique-se de que os tipos de dados utilizados sao os mais apropriados para o contexto. Por exemplo, evite usar int quando float seria mais apropriado. Para strings, utilize a concatenacao de maneira eficiente e evite operacoes caras em loops. Utilize colecoes imutaveis (como tuples ou frozenset) quando possivel para evitar alteracoes indesejadas.\n" \
               "- Controle de fluxo e complexidade: Analise o uso de estruturas de controle (como if, for, while) e veja se ha oportunidades para simplificacao. Evite aninhamentos muito profundos, pois isso torna o codigo dificil de ler e entender. Prefira a clareza em vez de tentar otimizar prematuramente.\n" \
               "- Comentarios e documentacao: Certifique-se de que o codigo tenha comentarios explicativos sempre que necessario, mas sem exagerar. Use docstrings para documentar funcoes e classes, explicando o que elas fazem e quais sao seus parametros e valores de retorno. Isso e util tanto para quem le o codigo quanto para quem da manutencao posteriormente.\n" \
               "- Evitar duplicacao de codigo: Se voce perceber que esta repetindo um bloco de codigo em varias partes, considere refatora-lo em uma funcao. Isso diminui a duplicacao, tornando o codigo mais facil de manter e reduzindo a chance de introduzir erros.\n" \
               "- Excecoes e erros: Verifique o tratamento de excecoes e erros. Em vez de deixar falhas passarem sem controle, implemente tratamento adequado de excecoes para tornar o sistema mais robusto, evitando falhas inesperadas.\n" \
    
    else:
        return "A similaridade esta boa. O codigo parece estar em conformidade com boas praticas. No entanto, aqui estao algumas sugestoes adicionais para aprimorar a legibilidade e a manutencao:\n" \
               "- Documentacao: Embora o codigo pareca bem estruturado, sempre busque manter uma documentacao clara, principalmente se o codigo for parte de um projeto maior. Isso pode incluir explicacoes de alto nivel sobre como as diferentes partes do sistema interagem.\n" \
               "- Refatoracao continua: Mesmo quando o codigo esta bom, sempre pense na refatoracao. As vezes, mesmo pequenas melhorias podem levar a grandes ganhos em legibilidade e desempenho.\n" \
               "- Verificacao de performance: Considere revisar o desempenho do codigo, especialmente se o sistema estiver lidando com grandes volumes de dados. Isso pode envolver o uso de algoritmos mais eficientes ou a adocao de praticas de otimizacao, como o uso de cache e paralelismo.\n" \
               "- Boas praticas de seguranca: Para garantir a seguranca do codigo, revise praticas como a validacao de entradas e o uso de criptografia, caso o codigo manipule dados sensiveis. Isso pode ajudar a evitar vulnerabilidades comuns como injecoes de codigo e vazamento de dados."

# Função para comparar as diferenças de código e retornar apenas os trechos com diferença
def comparar_diferencas(codigo1, codigo2):
    import difflib
    diff = difflib.ndiff(codigo1.splitlines(), codigo2.splitlines())
    diferencas = [linha[2:] for linha in diff if linha.startswith('+ ') or linha.startswith('- ')]
    return '\n'.join(diferencas) if diferencas else "Sem diferencas."
