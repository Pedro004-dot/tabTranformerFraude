# TabTransformer para Detecção de Fraude em Transações de Cartão de Crédito

## 📋 Descrição

Este projeto implementa um modelo de **Transformer** customizado para detecção de fraude em transações de cartão de crédito. O modelo utiliza arquitetura baseada em atenção (attention mechanism) para analisar sequências temporais de transações e identificar padrões suspeitos.

## 🎯 Objetivo

Desenvolver um sistema de detecção de fraude que:
- Analisa sequências temporais de transações por cliente
- Identifica padrões anômalos usando arquitetura Transformer
- Classifica transações como legítimas ou fraudulentas com alta precisão

## 📊 Dataset

- **Arquivo**: https://www.kaggle.com/datasets/kartik2112/fraud-detection
- **Tamanho**: ~1.3 milhões de transações
- **Colunas**: 24 features incluindo data/hora, valor, categoria, localização, etc.
- **Target**: `is_fraud` (binário: 0 = legítimo, 1 = fraude)

## 🏗️ Arquitetura do Modelo

### FraudTransformer

O modelo implementa uma arquitetura Transformer customizada com os seguintes componentes:

1. **Camada de Entrada Híbrida (Feature Fusion)**
   - Embeddings para categorias de estabelecimentos
   - Embeddings para features temporais (hora, dia da semana)
   - Projeções lineares para features numéricas (valor, delta de tempo)

2. **Positional Encoding**
   - Learnable positional embeddings para capturar ordem temporal

3. **Transformer Encoder**
   - Múltiplas camadas de atenção multi-head
   - Feedforward networks com ativação GELU
   - Dropout para regularização

4. **Classification Head**
   - Camadas densas para classificação binária final

### Hiperparâmetros

- `d_model`: 64 (dimensão do modelo)
- `nhead`: 4 (número de heads de atenção)
- `num_layers`: 2 (camadas do encoder)
- `dim_feedforward`: 128
- `dropout`: 0.1
- `seq_len`: 20 (tamanho da janela de sequência)
- `batch_size`: 64
- `learning_rate`: 0.001

## 🔧 Funcionalidades Principais

### 1. Pré-processamento de Dados

- **Limpeza**: Remove colunas não relevantes (nome, endereço, coordenadas geográficas, etc.)
- **Features Temporais**: Extrai hora do dia e dia da semana da transação
- **Time Delta**: Calcula diferença de tempo entre transações consecutivas do mesmo cliente
- **Normalização**: Aplica log transform em valores monetários e deltas de tempo

### 2. Tokenização de Categorias

- Implementa `CategoryTokenizer` customizado
- Converte categorias de estabelecimentos em índices numéricos
- Suporta tokens desconhecidos (`<UNK>`) para categorias não vistas no treino
- Salva vocabulário em JSON para uso em produção

### 3. Criação de Sequências Temporais

- **Sliding Windows**: Cria janelas deslizantes de transações por cliente
- Agrupa transações por cartão de crédito
- Gera sequências de tamanho fixo (padrão: 20 transações)
- Target é a última transação da sequência

### 4. Divisão Temporal

- **Split 80/20**: Divisão temporal (não aleatória)
- Transações mais antigas → Treino
- Transações mais recentes → Validação
- Garante que o modelo seja avaliado em dados futuros

## 📈 Resultados

### Métricas de Performance (Validação)

- **Acurácia**: 99.99%
- **Precisão (Fraude)**: 97%
- **Recall (Fraude)**: 86%
- **F1-Score (Fraude)**: 91%
- **AUC-ROC**: Disponível nos logs de treinamento

### Distribuição de Dados

- **Treino**: ~1.04 milhões de janelas
- **Validação**: ~242 mil janelas
- **Classes**: Altamente desbalanceadas (fraude é minoria)

## 🚀 Como Usar

### Pré-requisitos

```bash
pip install pandas scikit-learn matplotlib seaborn tqdm torch
```

### Execução

1. **Preparar os dados**:
   - Coloque o arquivo `credit_card_transactions.csv` no diretório do projeto

2. **Executar o notebook**:
   - Abra `tabTrasnformerFraudeV1ipynb.ipynb`
   - Execute todas as células em ordem

3. **Arquivos gerados**:
   - `category_vocab.json`: Vocabulário de categorias
   - `best_fraud_transformer.pth`: Modelo treinado (melhor checkpoint)

## 📁 Estrutura do Projeto

```
tabTransformerFraude/
├── tabTrasnformerFraudeV1ipynb.ipynb  # Notebook principal
├── credit_card_transactions.csv       # Dataset (não incluído)
├── category_vocab.json                # Vocabulário gerado
├── best_fraud_transformer.pth         # Modelo treinado
└── README.md                          # Este arquivo
```

## 🔍 Features Utilizadas

O modelo utiliza as seguintes features após pré-processamento:

1. **category_idx**: Índice da categoria do estabelecimento (tokenizado)
2. **amt_log**: Valor da transação (log transformado)
3. **hour**: Hora do dia (0-23)
4. **day_of_week**: Dia da semana (0-6)
5. **time_delta_log**: Tempo decorrido desde última transação (log transformado)

## 🧠 Decisões de Design

### Features Excluídas

- **Dados Pessoais**: Nome, data de nascimento, profissão (não relevantes para fraude)
- **Localização Geográfica**: Coordenadas lat/long (complexidade adicional sem ganho significativo)
- **Endereços**: Street, city, state, zip (redundantes)
- **Merchant ID**: Identificador específico do comerciante

### Features Incluídas

- **Temporais**: Hora e dia da semana capturam padrões comportamentais
- **Categoria**: Tipo de estabelecimento pode indicar padrões de fraude
- **Valor**: Transações anômalas em valor
- **Time Delta**: Intervalos suspeitos entre transações

## 🛠️ Tecnologias Utilizadas

- **Python 3.9+**
- **PyTorch**: Framework de deep learning
- **Pandas**: Manipulação de dados
- **Scikit-learn**: Métricas e avaliação
- **Matplotlib/Seaborn**: Visualizações
- **NumPy**: Operações numéricas

## 📝 Notas Importantes

- O modelo foi treinado por **5 épocas**
- Usa **BCEWithLogitsLoss** para classificação binária
- Otimizador **AdamW** (similar ao usado em GPT)
- Suporta **CUDA**, **MPS** (Apple Silicon) e **CPU**
- O modelo salva automaticamente o melhor checkpoint baseado em loss de validação

## 🔮 Melhorias Futuras

- Implementar RoPE (Rotary Position Embedding) para melhor captura de ordem temporal
- Adicionar features geográficas (distância entre transações)
- Experimentar com diferentes tamanhos de janela
- Adicionar visualizações de atenção para interpretabilidade

## 📄 Licença

Este projeto é para fins educacionais e de pesquisa.

## 👤 Autor

Desenvolvido por Pedro torrezani

---

**Nota**: Este modelo foi desenvolvido para demonstrar a aplicação de Transformers em problemas de detecção de fraude em dados tabulares com sequências temporais.

