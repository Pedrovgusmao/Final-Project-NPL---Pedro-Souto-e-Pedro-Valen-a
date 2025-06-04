# ChatInsight AI - Busca Inteligente para Suas Imagens Exportadas

## 🚀 Resumo do Projeto (MVP)

O ChatInsight AI é uma ferramenta web desenvolvida para simplificar a forma como você encontra imagens específicas em sua coleção pessoal. **Neste MVP (Minimum Viable Product), o foco é permitir que usuários façam upload manual de imagens previamente exportadas (do WhatsApp ou qualquer outra fonte)**. Uma vez enviadas, a ferramenta utiliza um modelo (CLIP) para indexar essas imagens, possibilitando buscas futuras através de descrições em linguagem natural.

**Objetivo principal do MVP:** "Facilitar a busca inteligente e rápida por imagens utilizando descrições textuais naturais, reduzindo o tempo médio de busca manual (cerca de 1 minuto por imagem) para perto de 5 segundos por busca na ferramenta, com pelo menos 60% de precisão nos resultados relevantes."

Por exemplo, após exportar suas fotos e fazer o upload, você pode digitar "bolo de aniversário com amigos" ou "documento importante escaneado" e o ChatInsight AI trará as imagens correspondentes da sua coleção carregada.

## 🤔 O Problema: Localizar Imagens em Coleções Exportadas

Muitas vezes, acumulamos um grande volume de imagens exportadas de aplicativos como o WhatsApp, ou de outras fontes (fotos de câmera, scans, etc.), guardadas em pastas no computador. Encontrar uma imagem específica nesse mar de arquivos pode ser uma tarefa demorada e frustrante, baseada em tentar lembrar nomes de arquivos ou datas, o que frequentemente leva mais de um minuto por imagem. O problema que este MVP busca resolver é justamente essa **dificuldade e demora em localizar imagens específicas que já foram exportadas manualmente para o computador do usuário.**

## ✨ Solução Proposta (MVP): Busca Inteligente em Imagens Carregadas Manualmente

O ChatInsight AI oferece uma solução prática para este desafio:

1.  **Exportação Manual pelo Usuário:** Primeiro, o usuário exporta as imagens desejadas (de conversas do WhatsApp, galeria do celular, etc.) para uma pasta em seu computador.
2.  **Upload para a Ferramenta:** Em seguida, o usuário acessa a interface web do ChatInsight AI e faz o upload dessas imagens para a plataforma.
3.  **Indexação Inteligente com IA:** A ferramenta utiliza o modelo CLIP (Contrastive Language-Image Pre-Training) da OpenAI para processar cada imagem carregada. O CLIP cria uma representação vetorial (embedding) que capta o conteúdo semântico da imagem.
4.  **Busca por Linguagem Natural:** O usuário pode então digitar uma descrição textual do que procura (ex: "cachorro brincando no parque", "recibo de restaurante"). O sistema converte essa descrição em um vetor e o compara com os vetores das imagens indexadas, retornando as mais similares.

O elemento inovador é o uso da IA (CLIP) para compreender o conteúdo visual das imagens e o significado das descrições textuais, permitindo uma busca muito mais intuitiva e eficiente do que métodos tradicionais.

## ⭐ Principais Funcionalidades (MVP)

*   **Upload de Múltiplas Imagens:** Permite que o usuário selecione e envie múltiplas imagens de uma vez, desde que estas **já tenham sido exportadas manualmente** para o seu computador.
*   **Busca por Descrição Textual:** Após o upload e indexação, o usuário pode buscar imagens na sua coleção carregada usando linguagem natural (ex: "pôr do sol na praia", "gato dormindo no sofá", "gráfico de pizza colorido").
*   **Visualização de Resultados:** As imagens encontradas são exibidas de forma clara na interface.
*   **Visualização em Tela Cheia:** Permite clicar em uma imagem do resultado para visualizá-la em tamanho maior.
*   **Interface Web Simples:** Interface projetada para ser intuitiva e fácil de usar.

## 🎯 Metas de Sucesso e Impacto Esperado (MVP)

Este MVP será considerado bem-sucedido ao atingir as seguintes metas quantitativas e qualitativas:

*   **Precisão da Busca:** Pelo menos 60% das buscas devem retornar imagens relevantes que correspondam à descrição fornecida pelo usuário, considerando o conjunto de imagens carregadas.
*   **Redução do Tempo de Busca:** O tempo para encontrar uma imagem específica usando a ferramenta (após o upload inicial) deve ser perto de 5 segundos, em contraste com o tempo médio de busca manual estimado em aproximadamente 1 minuto por imagem em uma pasta de arquivos.
*   **Usabilidade da Interface:** A interface do usuário deve ser considerada intuitiva e fácil de usar, validada através de feedback de usuários reais (mesmo que em testes informais nesta fase de MVP).
*   **Benefício Principal:** Proporcionar uma maneira significativamente mais rápida e eficiente para os usuários localizarem imagens dentro de suas próprias coleções exportadas, economizando tempo e reduzindo a frustração.

### Verificar instalações:
```bash
python --version
docker --version
git --version
```

## 🚀 Como Rodar o Projeto

### Passo 1: Clonar o Repositório
```bash
git clone <url-do-repositorio>
cd projdoscaras
```

### Passo 2: Configurar o Ambiente Python
```bash
# Criar ambiente virtual
python -m venv projnlp

# Ativar ambiente virtual
# No macOS/Linux:
source projnlp/bin/activate
# No Windows:
# projnlp\\Scripts\\activate

# Instalar dependências
pip uninstall pillow-heif -y
pip install -r requirements.txt
```

### Passo 3: Iniciar o Qdrant (Banco de Dados Vetorial)

**⚠️ IMPORTANTE: Este passo é obrigatório antes de rodar o backend!**

```bash
# Iniciar container do Qdrant em modo detached (background)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Para verificar se o Qdrant está funcionando, acesse: http://localhost:6333/dashboard

### Passo 4: Iniciar o Backend (Flask)

**Em um novo terminal (ou na mesma aba, após o Qdrant iniciar em background):**

```bash
# Navegar para o diretório do projeto (se não estiver lá)
# cd projdoscaras

# Ativar ambiente virtual (se não estiver ativo)
source projnlp/bin/activate

# Rodar o servidor Flask
python multimodal_search_backend.py
```

O backend estará disponível em: http://localhost:5001

### Passo 5: Abrir a Interface Web

Abra o arquivo `index.html` no seu navegador.

Como alternativa, se preferir usar um servidor local simples para o frontend:
```bash
# Opção: Usar servidor Python (em um novo terminal, na pasta do projeto)
python -m http.server 8000
# Depois acesse: http://localhost:8000
```

### Guia Rápido de Uso (MVP)

1.  **Exporte Suas Imagens:** Primeiro, certifique-se de que as imagens que você deseja pesquisar foram **exportadas de sua origem** (WhatsApp, galeria do celular, etc.) para uma pasta em seu computador.
2.  **Adicione as Imagens à Ferramenta:**
    *   Na interface do ChatInsight AI, use a seção de upload para selecionar as imagens da pasta do seu computador. Você pode selecionar múltiplas imagens.
    *   Clique em "Processar Fotos". As imagens serão enviadas e indexadas.
3.  **Busque Suas Imagens:**
    *   Use a barra de busca para digitar uma descrição da imagem que procura (ex: "foto de um gato laranja", "documento com assinatura").
    *   Clique em "Buscar Fotos". Os resultados da sua coleção carregada serão exibidos.
4.  **Visualize os Resultados:**
    *   Clique em qualquer imagem para vê-la maior. Feche com 'Esc' ou clicando fora.

## 📁 Estrutura do Projeto

```
projdoscaras/
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências Python
├── multimodal_search_backend.py       # API Flask
├── index.html                         # Interface web
├── uploads/                           # Evidências processadas (local de armazenamento padrão)
└── projnlp/                           # Ambiente virtual Python
```

## ⚙️ Como Funciona (Tecnologias Envolvidas - MVP)

*   **Modelo de IA (CLIP):** Utiliza o modelo CLIP da OpenAI para gerar representações vetoriais (embeddings) de imagens e textos, permitindo a comparação semântica entre eles.
*   **Banco de Dados Vetorial (Qdrant):** Armazena os embeddings das imagens carregadas e realiza buscas por similaridade de forma eficiente.
*   **Backend (Python/Flask):** Gerencia o upload, processamento de imagens com CLIP, comunicação com Qdrant e as requisições de busca.
*   **Frontend (HTML, CSS, JavaScript):** Interface web para interação do usuário.

## 🔮 Futuras Melhorias (Pós-MVP)

Embora o MVP atual se concentre na funcionalidade principal de busca em imagens carregadas manualmente, futuras versões do ChatInsight AI poderiam explorar:

*   **Integração Direta com WhatsApp:** Desenvolver uma forma de acessar e indexar mídias diretamente do WhatsApp (respeitando a privacidade e permissões do usuário), eliminando a necessidade de exportação manual.
*   **Funcionalidades Avançadas de Gerenciamento:** Melhorar as opções de gerenciamento de imagens, como exclusão em lote, criação de álbuns virtuais, etc.
*   **Suporte a Outros Tipos de Mídia:** Expandir para incluir busca em vídeos ou outros formatos de arquivo.
*   **Melhorias na Precisão e Performance:** Continuar refinando os modelos e a infraestrutura para buscas ainda mais rápidas e precisas.

---
**Desenvolvido como um projeto prático e demonstrativo.** 