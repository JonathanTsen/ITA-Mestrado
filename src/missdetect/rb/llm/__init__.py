"""LLM como classificador de mecanismos (passos 42 e 43 do plano; pré-registro §4, H4; emendas P20–P23).

Módulos:
- `render`: renderização cega (A1, tabela em inteiros de 0 a 999) e dos descritores (A2);
- `cache`: caderno de respostas em disco, que nunca é apagado;
- `client`: chamadas ao Gemini (`google-genai`), com chave gratuita opcional, trava de autorização e modo de lote;
- `panel`: painel de LLM (famílias, contextos, instâncias, repetição e ordem);
- `budget`: contagem de tokens, projeção de custo e trava do teto.
"""
