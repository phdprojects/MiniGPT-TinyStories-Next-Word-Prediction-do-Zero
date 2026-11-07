# MiniGPT-TinyStories

**Next-word prediction com um Transformer (mini-GPT) construído do zero em PyTorch — sem bibliotecas de transformers — treinado no dataset [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (Hugging Face), inteiramente em Databricks (CPU serverless).**

## Índice

1. [Visão geral](#visão-geral)
2. [Arquitetura do modelo](#arquitetura-do-modelo)
3. [Pipeline de dados](#pipeline-de-dados)
4. [Treino](#treino)
5. [Resultados](#resultados)
6. [Investigação — 5 perguntas, 5 respostas](#investigação--5-perguntas-5-respostas)
7. [Geração de texto](#geração-de-texto)
8. [Como reproduzir](#como-reproduzir)
9. [Lições de engenharia](#lições-de-engenharia)

## Visão geral

Objetivo: perceber e implementar de raiz o mecanismo por trás dos modelos tipo GPT — prever a próxima palavra a partir de um contexto — com um Transformer pequeno mas completo, do ingente dataset à geração de texto autoregressiva.

* **Dataset**: TinyStories (2.141.709 histórias sintéticas em inglês simples, geradas por GPT-3.5/4).
* **Vocabulário**: 27.025 tokens word-level (palavras com freq ≥ 5 + `<pad>`, `<unk>`, `<eos>`).
* **Dados de treino**: 3.747.482 pares (contexto de 16 tokens → palavra seguinte) de 20 mil histórias.
* **Modelo**: 4.254.592 parâmetros, treinado em CPU com orçamento de tempo (350 passos).
* **Resultado**: perda de validação 10,20 → 5,01; perplexidade 27.025 → 150; precisão top-1 15,8%.

## Arquitetura do modelo

Transformer **decoder-only** clássico (estilo GPT-2 em miniatura):

* Embeddings de token (27.025 × 128) + embedding posicional aprendido (16 × 128)
* 4 blocos Transformer **pre-LN**:
  * Atenção multi-cabeça causal (4 cabeças, d_model=128) com máscara causal
  * MLP (128 → 512 → 128) com GELU e dropout
* LayerNorm final + cabeça de saída com **weight tying** (partilha os pesos do embedding — 0 parâmetros extra)
* Inicialização estilo GPT-2 (std = 0,02) — verifica-se pela perda inicial ≈ ln(V) = 10,2

Total: **4,25M parâmetros**. Previsão na última posição do contexto (16 tokens).

## Pipeline de dados

| Etapa | Descrição | Output |
| --- | --- | --- |
| Ingestão | Download do TinyStories (HF) → Delta bronze | `tinystories_raw` (2.141.709 histórias) |
| EDA | Comprimentos, palavras frequentes (média 175 palavras/história) | — |
| Tokenização | Word-level, minúsculas + pontuação `[.,!?]`, freq ≥ 5, `<eos>` no fim | `tinystories_vocab`, `tinystories_tokens` |
| Pares | Janela deslizante de 16 tokens → alvo seguinte (sem padding) | `tinystories_pairs_train` (3,7M), `..._val` (370k) |

## Treino

* AdamW (lr 1e-3, weight decay 0,01), lotes de 512, subamostra de 200k pares de treino / 5k de validação
* 350 passos (~179k exemplos vistos) em CPU com orçamento de tempo — orçamento respeitado
* Métricas registadas a cada 50 passos (perda treino/validação, perplexidade, P(palavra correta) num probe fixo)
* Tracking em **MLflow**; modelo persistido em Unity Catalog Volume; métricas em tabela Delta

## Resultados

| Métrica | Antes do treino | Depois de 350 passos |
| --- | --- | --- |
| Perda de validação | 10,20 (aleatório ≈ ln 27.025) | **5,01** |
| Perplexidade | 27.025 | **150** |
| Precisão top-1 | ~0% | **15,8%** |
| P(palavra correta) média | 0,004% | **5,3%** |
| Ranking mediano da palavra correta | — | **14º** de 27.025 |

## Investigação — 5 perguntas, 5 respostas

1. **Mais contexto ajuda?** Sim: precisão top-1 sobe de 4,5% (contexto de 1 palavra) para 15,8% (16 palavras).
2. **P(palavra correta) evolui com o treino?** Sim: 0,0004 → 0,0015 no probe fixo (37× acima do aleatório), sempre a subir enquanto a perplexidade cai 413 → 159.
3. **Que palavras são difíceis?** Correlação 0,84 entre frequência e facilidade. Mais fáceis: pontuação e stopwords («.» P=25%). Mais difíceis: palavras frequentes mas dependentes do contexto («all», «fun», «back») — o modelo ainda não distingue situações.
4. **Gramática ou coocorrência?** Gramática: baralhar a ordem das palavras do contexto (mesmas palavras!) derruba P(correta) em **−63%** e a precisão de 15,8% para 5,0% — a atenção captura estrutura, não é um saco-de-palavras.
5. **Mais dados = melhor?** Só com mais computação: com os mesmos 120 passos, 25k pares (5,62) bateram 200k (5,72) — a repetição consolida; com 350 passos sobre 200k, 5,01.

## Geração de texto

Geração autoregressiva clássica: tokenizar o prompt → modelo olha para os últimos 16 tokens → softmax sobre todo o vocabulário (27.025) → escolher a palavra (greedy ou amostragem com temperatura) → acrescentar ao contexto → repetir até ao limite de palavras ou ao token de paragem `<eos>`.

```
PROMPT: once upon a time there was a little girl named lily . she loved to play
GERADO: . she was so happy. she was so so so ...
```

O texto é repetitivo — comportamento esperado de um modelo subtreinado (350 passos em CPU) — mas o mecanismo é exatamente o dos GPT reais: a qualidade escala com o treino.

## Como reproduzir

1. Workspace Databricks com compute CPU serverless (sem GPU necessário).
2. Executar as células por ordem (STEP 0 → STEP 15): o STEP 0 instala o PyTorch CPU (`%pip install torch --index-url https://download.pytorch.org/whl/cpu`) e reinicia o Python.
3. Cada STEP é autónomo e documentado: input, output, decisões e operações no cabeçalho da célula.
4. Dependências entre células: investigação e geração requerem STEP 0 + STEP 5 (classes) + STEP 7 (modelo treinado carregado do volume).

Artefactos produzidos: tabelas Delta (`tinystories_raw`, `tinystories_vocab`, `tinystories_tokens`, `tinystories_pairs_train/val`, `tinystories_treino_metricas`), pesos em UC Volume (`mini_gpt.pt`) e runs MLflow (treino + investigação Q5).

## Lições de engenharia

* A inicialização por defeito do `nn.Embedding` (std 1) dá perda inicial ~91; com std 0,02 (GPT-2) a perda inicial ≈ ln(V) — comportamento esperado e verificável.
* Máscaras causais fixas (16×16) rebentam com sequências mais curtas: fatiar a máscara (`mask[:T, :T]`) torna o modelo robusto a qualquer comprimento.
* Em CPU, aplicar a cabeça de saída apenas na última posição poupa ~16× FLOPs sem alterar a arquitetura.
* Orçamento de tempo no loop de treino garante que a célula termina de forma previsível em compute partilhado.

## Limitações e próximos passos

* Treinar até convergência em GPU (milhões de passos sobre os 3,7M pares) — o modelo atual está subtreinado por design (orçamento CPU).
* Learning-rate scheduler (warmup + cosine), weight decay afinado, early stopping pela perda de validação.
* Tokenização BPE/subword em vez de word-level; beam search e top-k sampling na geração.

---

*Projeto de portfólio — construído fase a fase, com validação de cada etapa: ingestão → EDA → tokenização → pares → modelo → treino → investigação → geração → conclusão.*