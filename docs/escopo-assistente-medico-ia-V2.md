# Escopo do Projeto — Assistente Médico IA

**Versão:** 2.0  
**Data:** Maio/2026  
**Classificação:** Uso Interno  

---

## Visão Geral

O projeto consiste no desenvolvimento de um **Assistente Médico inteligente** para auxiliar médicos no atendimento de emergência hospitalar. O sistema combina Fine-Tuning local com QLoRA (TinyLlama-1.1B), RAG com busca vetorial FAISS, e orquestração de fluxos decisórios via LangGraph, com foco em precisão clínica, rastreabilidade e segurança.

O assistente atua como **suporte à decisão clínica** — não substitui o médico, mas responde dúvidas, sugere condutas e emite alertas com base nos protocolos internos do hospital, sempre com supervisão humana mandatória.

### Usuário Principal

- **Perfil:** Médico de atendimento da emergência hospitalar, sem especialidade definida
- **Contexto de uso:** Atendimento ao paciente na emergência
- **Interação:** O médico informa sintomas do paciente ou submete laudos laboratoriais ao assistente e recebe sugestões de conduta clínica baseadas nos protocolos internos

### Papel do Sistema

O assistente médico tem papel exclusivo de **suporte à decisão clínica**:

- Responder dúvidas do médico sobre condutas clínicas
- Sugerir procedimentos com base nos protocolos internos do hospital
- Analisar exames laboratoriais e classificar alterações via escala CTCAE
- Emitir alertas para a equipe médica quando necessário
- Registrar todas as interações para auditoria completa
- **Não** tomar decisões clínicas de forma autônoma — supervisão humana mandatória

---

## 1. Objetivos do Projeto

O projeto atende a quatro pilares fundamentais da **IA responsável em saúde**:

### 1.1 Personalização do Modelo

Adaptação do LLM **TinyLlama-1.1B** por meio de fine-tuning com **QLoRA**, utilizando protocolos hospitalares sintéticos. O modelo é especializado em terminologia médica e nas diretrizes internas do hospital, garantindo respostas contextualmente adequadas ao ambiente clínico.

### 1.2 Precisão Clínica

Implementação de classificadores de toxicidade segundo o **Common Terminology Criteria for Adverse Events (CTCAE)**, com análise avançada de exames laboratoriais via GPT-4o (API). O sistema classifica alterações em graus G1 a G5, com acionamento automático de alertas para graus críticos (G3+).

### 1.3 Segurança e Governança

Aplicação de controles para evitar recomendações geradas por IA sem revisão, assegurando **supervisão humana mandatória** em toda sugestão clínica. Inclui trava ética para bloquear diagnósticos indevidos, prescrições não autorizadas e recomendações perigosas, além de mecanismos de anonimização em conformidade com a LGPD.

### 1.4 Auditabilidade

Registro estruturado de todas as operações e consultas em **arquivos CSV de log**, permitindo rastreabilidade completa do fluxo de decisão, com timestamp, tipo de solicitação, intenção detectada e resposta gerada.

---

## 2. Descrição do Projeto

- Criação de Assistente Médico virtual treinado com dados internos do hospital
- Fine-tuning do modelo LLM TinyLlama-1.1B com PEFT/QLoRA, utilizando protocolos, perguntas frequentes, laudos e receitas
- Dados preparados com técnicas de **preprocessing, anonimização e curadoria**
- Integração do modelo com **LangChain** para consultas a bases estruturadas e contexto do paciente
- Fluxos automatizados de decisão: verificar exames, sugerir tratamentos e emitir alertas
- Limites de atuação para evitar sugestões impróprias, com **validação humana obrigatória**
- Implementação de **logging detalhado** e explicabilidade das respostas
- Código modular em Python, com instruções completas no README

---

## 3. Datasets — Protocolos Internos do Hospital

Os datasets constituem a base de conhecimento do assistente, utilizados tanto no fine-tuning do modelo quanto na indexação vetorial para RAG.

| Dataset | Descrição |
|---|---|
| `dataset-faq-medica.json` | Perguntas e respostas frequentes do contexto médico da emergência |
| `dataset-laudos.json` | Laudos laboratoriais e de imagem anonimizados |
| `dataset-pacientes.json` | Perfis de pacientes sintéticos para contextualização clínica |
| `dataset-pedidos-medicos.json` | Pedidos médicos e solicitações de exames |
| `dataset-perguntas-respostas-medicos.json` | Pares Q&A clínicos para fine-tuning supervisionado |
| `dataset-procedimentos.json` | Procedimentos clínicos e cirúrgicos com indicações |
| `dataset-protocolos-medicos.json` | Protocolos internos do hospital por condição clínica |
| `dataset-receitas.json` | Prescrições e receitas médicas anonimizadas |

> **Nota:** Todos os datasets utilizam dados **sintéticos**, gerados para fins de desenvolvimento e teste. Nenhum dado real de pacientes é utilizado nesta fase.

---

## 4. Arquitetura Técnica

A solução adota uma **arquitetura modular e não-linear**, baseada em grafos de estado (LangGraph), permitindo controle explícito do fluxo de raciocínio. É uma arquitetura híbrida de IA médica que combina:

- Classificação de intenção
- Encaminhamento inteligente
- LLM local especializada
- RAG com busca vetorial FAISS
- Camadas éticas e de privacidade
- Logging para auditoria clínica

### 4.1 Componentes Principais

| Componente | Tecnologia | Função |
|---|---|---|
| **LLM Local** | TinyLlama-1.1B | NLP médico especializado |
| **Fine-Tuning** | PEFT / QLoRA | Especialização clínica com dados do hospital |
| **Vector Store** | FAISS | RAG — recuperação semântica nos protocolos |
| **Expert LLM** | GPT-4o (API) | Análise clínica avançada e classificação CTCAE |
| **Orquestrador** | LangGraph | Controle de fluxo baseado em estados |
| **Pipeline de Agentes** | LangChain | Integração entre modelo, dados e ferramentas |
| **Logging** | CSV / Python | Auditoria e rastreabilidade de todas as operações |

### 4.2 Camadas da Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│  CAMADA DE INTERFACE — Médico                                   │
│  Chat Clínico · Upload de Laudos · Painel de Alertas            │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  CAMADA DE ORQUESTRAÇÃO — LangGraph                             │
│  Classificador de Intenções · Router · Gerenciador de Estado    │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
┌──────────────▼────────┐   ┌─────────────▼─────────────────────┐
│  RAMO 1               │   │  RAMO 2                            │
│  Expert LLM + CTCAE   │   │  RAG FAISS + TinyLlama-1.1B        │
│  GPT-4o (API)         │   │  Protocolos Internos               │
└──────────────┬────────┘   └─────────────┬─────────────────────┘
               │                          │
┌──────────────▼──────────────────────────▼───────────────────────┐
│  GERAÇÃO DE RESPOSTA CONSOLIDADA                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  FILTRO TRANSVERSAL — Anonimização + Trava Ética                │
│  LGPD · Bloqueio de Diagnósticos · Supervisão Humana            │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  GRAVAÇÃO LOG CSV — Auditoria Completa                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                 Resposta ao Médico
```

---

## 5. Fluxo de Decisão Detalhado

### 5.1 Início

Marca o início da execução do processo. Inicializa o pipeline de atendimento e processamento da solicitação do médico.

### 5.2 Entrada: Pergunta / Dados do Paciente

Recebe os dados que o médico extraiu do paciente. Constrói o contexto necessário para a tomada de decisão do sistema.

**Entradas possíveis:**

- Pergunta clínica
- Sintomas relatados pelo paciente
- Exames e laudos laboratoriais
- Histórico médico do paciente
- Perfil do paciente
- Contexto hospitalar

### 5.3 Nó: Classificar Intenção

Identifica automaticamente o tipo da solicitação recebida e encaminha para o módulo mais adequado. É o componente central do roteamento via LangGraph.

**Tipos de intenção detectados:**

| Intenção | Exemplo | Rota |
|---|---|---|
| Relato de Sintomas | "Paciente com náusea persistente e perda de apetite" | Ramo 2 — RAG FAISS |
| Laudo Laboratorial | Hemograma completo submetido pelo médico | Ramo 1 — GPT-4o + CTCAE |
| Dúvida de Protocolo | "Dose de X para insuficiência renal?" | Ramo 2 — RAG FAISS |
| Exames Pendentes | "Quais exames ainda não retornaram?" | Alerta / Status |

### 5.4 Ramo 1: Análise Especialista GPT-4o + CTCAE

Executa análise especializada de exames e eventos clínicos com:

- Classificação de toxicidade e gravidade conforme escala CTCAE (G1–G5)
- Análise médica estruturada baseada em exames e critérios clínicos
- Acionamento de alertas para alterações de grau G3 ou superior
- Encaminhamento para o nó de geração de resposta final com análise completa

### 5.5 Ramo 2: Busca RAG FAISS

Realiza recuperação de informações na base documental:

- Consulta ao banco vetorial FAISS com os datasets do hospital
- Extração de contextos relevantes dos protocolos médicos
- Fornece contexto confiável e documental para respostas sobre sintomas, protocolos e procedimentos
- Integração com TinyLlama-1.1B (fine-tuned) para geração contextualizada

### 5.6 Nó: Gerar Resposta Final

Consolida as informações provenientes dos módulos anteriores e produz uma resposta:

- Clara e objetiva para o médico
- Contextualizada ao caso do paciente
- Clinicamente coerente com os protocolos do hospital
- Com citação obrigatória da fonte (protocolo de origem)

### 5.7 Filtro: Anonimização e Trava Ética

Camada transversal aplicada antes de toda resposta entregue ao médico.

**Anonimização:**
- Remoção de dados pessoais identificáveis
- Ocultação de informações sensíveis
- Conformidade com LGPD e normas do CFM

**Trava Ética:**
- Impedir diagnósticos indevidos por IA
- Bloquear recomendações perigosas ou fora dos protocolos
- Evitar prescrições não autorizadas
- Aplicar limites regulatórios vigentes
- Garantir supervisão humana mandatória em toda sugestão clínica

### 5.8 Gravação de Log CSV

Registra cada interação em arquivo de log estruturado com os campos:

| Campo | Descrição |
|---|---|
| `timestamp` | Data e hora da interação |
| `tipo_solicitacao` | Categoria da entrada (sintoma, laudo, protocolo etc.) |
| `intencao_detectada` | Intenção classificada pelo router LangGraph |
| `ramo_acionado` | Ramo 1 (GPT-4o/CTCAE) ou Ramo 2 (RAG/FAISS) |
| `resposta_gerada` | Resumo da resposta entregue ao médico |
| `fonte_protocolo` | Protocolo ou dataset de origem da resposta |

### 5.9 Fim: Resposta ao Médico

Entrega da resposta processada ao médico solicitante. O resultado esperado é sempre:

- **Seguro** — filtrado pela trava ética e anonimização
- **Contextualizado** — baseado nos dados do paciente e no protocolo do hospital
- **Auditável** — com log registrado e fonte citada
- **Alinhado** aos protocolos clínicos internos

---

## 6. Análise de Exames — Escala CTCAE

A escala CTCAE (Common Terminology Criteria for Adverse Events) é utilizada para classificar a gravidade das alterações encontradas nos exames laboratoriais, acionada via GPT-4o no Ramo 1 do fluxo.

| Grau | Classificação | Descrição | Ação do Sistema |
|---|---|---|---|
| **G1** | Leve | Assintomático ou sintomas mínimos | Registro + monitoramento |
| **G2** | Moderado | Intervenção mínima indicada | Sugestão de conduta |
| **G3** | Grave | Hospitalização ou intervenção urgente indicada | ⚠ Alerta à equipe médica |
| **G4** | Risco de Vida | Intervenção urgente e imediata necessária | 🚨 Alerta crítico imediato |
| **G5** | Óbito | Relacionado ao evento adverso | Registro crítico em auditoria |

### Exemplo Prático — Hemograma

```
Neutrófilos: 900/mm³      → G3 — Grave (Neutropenia severa)       ⚠ Alerta emitido
Hemoglobina: 8,5 g/dL     → G2 — Moderado (Anemia moderada)        Conduta sugerida
Plaquetas: 25.000/mm³     → G4 — Risco de Vida (Trombocitopenia)  🚨 Alerta crítico
```

---

## 7. Fases do Projeto

### Fase 1 — Fundação: Dados e Modelos

**Entregas:**
- Coleta, curadoria, preprocessing e anonimização dos datasets do hospital
- Indexação dos datasets no banco vetorial FAISS
- Fine-tuning do TinyLlama-1.1B com PEFT/QLoRA nos dados clínicos sintéticos
- Configuração de embeddings especializados em linguagem médica

**Tecnologias:** TinyLlama-1.1B, PEFT, QLoRA, FAISS, Python

---

### Fase 2 — Agentes: LangChain, RAG e CTCAE

**Entregas:**
- Pipeline RAG com recuperação semântica via FAISS nos datasets do hospital
- Agente de Sintomas (LangChain + RAG + TinyLlama)
- Agente de Análise de Exames com classificação CTCAE via GPT-4o
- Testes unitários de cada agente

**Tecnologias:** LangChain, FAISS, GPT-4o API, CTCAE, TinyLlama-1.1B

---

### Fase 3 — Orquestração: LangGraph

**Entregas:**
- Classificador de intenções do médico
- Router de fluxos LangGraph com todos os cenários clínicos mapeados
- Fluxos automatizados: sintomas (Ramo 2), laudos (Ramo 1), alertas, pendências
- Integração de todos os agentes sob orquestração unificada

**Tecnologias:** LangGraph, Classificação de Intenções, Grafos de Estado

---

### Fase 4 — Segurança, Interface e Homologação

**Entregas:**
- Filtro de anonimização e trava ética implementados
- Sistema de logging em CSV com campos auditáveis
- Interface web de chat para o médico
- Autenticação e controle de acesso por perfil
- Testes de homologação com equipe médica em ambiente simulado de emergência
- README completo com instruções de instalação e operação

**Tecnologias:** Python, Auth, LGPD, CSV Logging, Frontend

---

## 8. Premissas e Restrições

### Premissas

- Os protocolos internos do hospital já existem em formato digital e serão convertidos para os datasets listados na Seção 3
- O hospital fornecerá dados clínicos anonimizados para fine-tuning
- O médico é o responsável final por toda e qualquer decisão clínica
- A infraestrutura de servidores será disponibilizada pelo hospital
- Existe contrato vigente de acesso à API do GPT-4o

### Restrições

- O sistema **não toma decisões clínicas autônomas** — papel exclusivo de suporte com supervisão humana mandatória
- Toda sugestão deve citar a fonte (protocolo) de origem
- Dados de pacientes não podem sair do ambiente hospitalar (on-premises ou VPC privada)
- Conformidade obrigatória com **LGPD** e normas do **CFM**
- O sistema não emite prescrições médicas

---

## 9. Limitações Conhecidas

| Limitação | Descrição |
|---|---|
| **Dados Sintéticos** | Uso exclusivo de dados sintéticos nesta fase, sem validação clínica real com pacientes |
| **Alucinações** | Possibilidade de alucinações inerentes a modelos generativos — toda resposta requer revisão do médico |
| **Ambiente Não Assistencial** | Não recomendado para uso em ambientes assistenciais reais sem validação clínica completa |
| **Capacidade do Modelo** | Limitações de contexto e capacidade do TinyLlama-1.1B para casos de alta complexidade — nesses casos, o sistema escalona para o GPT-4o |
| **Especialidades** | O assistente não é configurado para especialidades médicas específicas nesta fase; foco na emergência geral |

---

## 10. Fora do Escopo

Os itens abaixo **não fazem parte desta fase do projeto**:

| Item | Justificativa |
|---|---|
| Especialidades Médicas | Foco exclusivo na emergência geral nesta fase |
| Prescrição Automática | O sistema sugere condutas, não emite documentos clínicos |
| Integração com HIS/Prontuário | Integração com sistemas de prontuário eletrônico é escopo futuro |
| Atendimento ao Paciente | O assistente se comunica exclusivamente com o médico |
| Aplicativo Mobile | Interface web (desktop/tablet) é o canal desta fase |
| Análise de Imagens | Processamento de raio-X, tomografia etc. é escopo futuro |
| Validação Clínica Real | Testes com pacientes reais dependem de aprovação do Comitê de Ética |

---

## 11. Glossário

| Termo | Definição |
|---|---|
| **RAG** | Retrieval-Augmented Generation — técnica que combina busca semântica com geração de texto |
| **Fine-Tuning** | Ajuste fino de um modelo de linguagem com dados específicos do domínio |
| **QLoRA** | Quantized Low-Rank Adaptation — técnica eficiente de fine-tuning que reduz uso de memória GPU |
| **PEFT** | Parameter-Efficient Fine-Tuning — conjunto de técnicas para ajuste fino com menos parâmetros |
| **TinyLlama-1.1B** | Modelo de linguagem compacto (1,1 bilhão de parâmetros) da família LLaMA, otimizado para execução local |
| **LangChain** | Framework para construção de aplicações com modelos de linguagem e agentes |
| **LangGraph** | Extensão do LangChain para orquestração de fluxos stateful e não-lineares baseados em grafos |
| **FAISS** | Facebook AI Similarity Search — biblioteca de busca vetorial por similaridade para RAG |
| **CTCAE** | Common Terminology Criteria for Adverse Events — escala de classificação de eventos adversos (G1–G5) |
| **Embedding** | Representação vetorial de texto para busca semântica |
| **HIS** | Hospital Information System — sistema de informação hospitalar/prontuário eletrônico |
| **CFM** | Conselho Federal de Medicina |
| **LGPD** | Lei Geral de Proteção de Dados — Lei nº 13.709/2018 |
| **Alucinação** | Fenômeno em que modelos de linguagem geram informações plausíveis, mas factualmente incorretas |

---

*Assistente Médico IA — Escopo do Projeto · Versão 2.0 · Maio/2026 · Documento de uso interno*
