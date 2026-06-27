# Manga Translator — Manual do Usuário

Tradutor local de mangás e quadrinhos. Captura uma região da tela, detecta balões de fala, executa OCR e exibe a tradução diretamente sobre a imagem.

---

## Requisitos do Sistema

### Obrigatórios

| Item | Mínimo |
|---|---|
| Sistema operacional | Windows 10 ou 11 (64-bit) |
| Python | 3.10 ou superior |
| RAM | 4 GB (sem LLM) / 10 GB (com LLM Qwen 3B) |
| Espaço em disco | 2 GB (sem LLM) / 10 GB (com LLM) |
| Internet | Necessária apenas no primeiro uso (download dos modelos) |

### Opcionais

| Item | Finalidade |
|---|---|
| GPU NVIDIA com CUDA 13.0+ | Acelera a tradução LLM de ~60s para ~5s por página |
| 6 GB de VRAM | Para rodar o modelo Qwen2.5-3B inteiramente na GPU |

---

## Instalação

### 1. Instalar Python

Baixe e instale o Python 3.10 ou superior em https://python.org/downloads  
Durante a instalação, marque a opção **"Add Python to PATH"**.

### 2. Instalar dependências obrigatórias

Abra o terminal (Prompt de Comando ou PowerShell) na pasta do programa e execute:

```bash
pip install pillow keyboard
pip install onnxruntime opencv-python
pip install manga-ocr easyocr
pip install transformers sentencepiece
```

### 3. (Opcional) Aceleração por GPU

Substitua o PyTorch padrão pela versão com suporte CUDA:

```bash
pip install --force-reinstall --no-deps torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
```

> Necessário somente se possuir placa NVIDIA com drivers atualizados.

---

## Primeiro Uso

### Executar o programa

```bash
python manga_translator.py
```

Na primeira execução, os modelos de OCR e tradução serão baixados automaticamente para a pasta `models/` (sem necessidade de configuração manual). O download pode levar alguns minutos dependendo da conexão.

**Modelos baixados automaticamente:**

| Modelo | Tamanho | Finalidade |
|---|---|---|
| `manga-ocr-base` | ~400 MB | OCR para japonês |
| EasyOCR (latin/english) | ~110 MB | OCR para demais idiomas |
| `opus-mt-ja-en` | ~300 MB | Tradução japonês → inglês (Marian) |
| `opus-mt-en-ROMANCE` | ~300 MB | Tradução inglês → português (Marian) |
| `Qwen2.5-3B-Instruct` | ~6 GB | Tradução com contexto (LLM) — opcional |

> O modelo Qwen é baixado somente se o checkbox **"LLM com contexto"** estiver marcado durante o uso.

---

## Interface

### Configurações principais

**Ler idioma** — idioma do texto original na imagem (ex: japonês, inglês, vietnamita).  
**Traduzir para** — idioma de saída da tradução.

**Marian (offline, rápido)** — usa modelos locais dedicados por par de idiomas. Mais veloz, sem necessidade de internet após o download. Recomendado para uso diário.

**LLM com contexto (mais lento)** — usa o modelo Qwen2.5-3B para tradução com memória de diálogo. Melhor qualidade para mangás com personagens e contexto narrativo, porém consome mais RAM e é mais lento.

> Se ambos estiverem marcados, o Marian é tentado primeiro. Caso falhe, o LLM assume. Se nenhum estiver marcado, a tradução é feita via Google Translate (requer internet).

**Substituir texto nos balões** — exibe a tradução sobrepostas diretamente no balão, no lugar do texto original.

---

## Passo a Passo para Traduzir

1. Abra o programa com `python manga_translator.py`
2. Clique em **"Selecionar região"** e arraste sobre a área da tela que contém o mangá
3. Escolha o idioma de leitura e o idioma de saída
4. Clique em **"Traduzir 1x"** para uma tradução única, ou **"Iniciar"** para modo contínuo
5. As traduções aparecem no painel lateral e, se ativado, sobrepostas nos balões

### Atalhos de teclado

| Tecla | Ação |
|---|---|
| F9 | Traduzir uma vez |
| F10 | Parar modo contínuo |

---

## Glossário de Personagens

O glossário garante que nomes de personagens sejam sempre traduzidos da mesma forma pelo LLM, evitando inconsistências.

### Adicionar manualmente

1. Clique no botão **"Glossário"**
2. Preencha o campo **Original** com o nome na língua fonte (ex: `悟空`)
3. Preencha o campo **Tradução** com o nome desejado (ex: `Goku`)
4. Clique em **"Adicionar"**

### Importar via arquivo TXT

Crie um arquivo `.txt` com um par por linha, usando qualquer um dos separadores:

```
# linhas com # são comentários e serão ignoradas
悟空 → Goku
光 -> Hikaru
主人公 = Protagonist
キャラ	Chara
```

No editor de glossário, clique em **"Importar TXT"** e selecione o arquivo.  
As entradas são mescladas ao glossário existente (nada é apagado).

O glossário é salvo automaticamente em `glossary.json` na pasta do programa.

---

## Idiomas Suportados

### Leitura (OCR)

| Idioma | Engine |
|---|---|
| Japonês | manga-ocr (especializado) |
| Chinês, Coreano, Inglês, Português, Espanhol, Francês, Alemão, Italiano, Vietnamita, Russo, Árabe | EasyOCR |

### Tradução (Marian — offline)

| Par | Modelo |
|---|---|
| Japonês → Inglês | opus-mt-ja-en |
| Japonês → Português | opus-mt-ja-en + opus-mt-en-ROMANCE (encadeado) |
| Inglês → Português | opus-mt-en-ROMANCE |
| Demais pares | LLM (Qwen) ou Google Translate |

---

## Solução de Problemas

**"Nenhum texto detectado"**  
Verifique se a região selecionada contém balões visíveis e se o idioma de leitura está correto.

**Tradução lenta**  
O modelo LLM (Qwen) roda em CPU por padrão, o que pode levar 30–90 segundos por página. Para acelerar, instale o PyTorch com CUDA (veja seção de instalação) ou desative o checkbox **"LLM com contexto"** para usar o Marian ou Google Translate.

**Erro ao baixar modelos**  
Verifique sua conexão com a internet. Os modelos são baixados do HuggingFace e do repositório EasyOCR na primeira execução.

**Overlay não aparece sobre o mangá**  
Certifique-se de que o checkbox **"Mostrar overlay"** está marcado e que a janela do mangá não está em modo exclusivo de tela cheia.

---

## Estrutura de Arquivos

```
MangaTranslator/
├── manga_translator.py       # script principal
├── glossary.json             # glossário de nomes (editável)
├── MANUAL.md                 # este manual
└── models/
    ├── comic-text-detector.onnx   # detector de balões (incluído)
    ├── easyocr/                   # OCR multilíngue (baixado automaticamente)
    └── huggingface/               # modelos de OCR e tradução (baixados automaticamente)
```

---

## Licença e Créditos

- Detecção de balões: [comic-text-detector](https://github.com/zyddnys/manga-image-translator)
- OCR japonês: [manga-ocr](https://github.com/kha-white/manga-ocr)
- OCR multilíngue: [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- Tradução local: [Helsinki-NLP/opus-mt](https://github.com/Helsinki-NLP/Opus-MT) via MarianMT
- Tradução com contexto: [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
