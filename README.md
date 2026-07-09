# 📊 Inteligência Preditiva e Engenharia Espacial para Internacionalização do Turismo

Este projeto foi desenvolvido em cooperação entre o ambiente acadêmico e uma operadora de turismo local sediada no Rio de Janeiro. O objetivo principal do sistema é substituir a precificação puramente intuitiva por um ecossistema de inteligência de dados, preparando a operação para atrair e absorver fluxos de turistas internacionais — especificamente dos mercados **russo, francês e inglês**.

Para viabilizar essa expansão de mercado, o pipeline consolida e cruza dados históricos de malha aérea (**ANAC**), infraestrutura urbana (**DataRio**) e microdados de acomodações por temporada (**Inside Airbnb**).

---

## 📑 Perguntas de Negócio Respondidas

O projeto foi desenhado para solucionar quatro incógnitas críticas da diretoria da empresa:
1. **Isolamento do Mercado Padrão:** Como expurgar de forma automática acomodações com preços discrepantes (*outliers*) que inflam as métricas e distorcem a realidade de preços do mercado?
2. **Elasticidade Geográfica:** Qual o impacto exato do distanciamento dos pontos de interesse turístico sobre a depreciação ou valorização do preço da diária?
3. **Modelagem de Demanda:** O volume histórico de engajamento (*reviews*) comporta-se de forma estável para servir como uma variável preditora de tração de mercado?
4. **Custo Marginal da Infraestrutura:** Qual o incremento financeiro médio gerado no faturamento esperado a cada unidade física de leito (`beds`) adicionada ao portfólio?

---

## 🛠️ Ecossistema Tecnológico (Stack)

* **Engenharia de Recursos e ETL:** `Python` e `Pandas` (ingestão e limpeza de dados); `NumPy` (cálculos matemáticos vetorizados).
* **Modelagem Preditiva e Machine Learning:** `Scikit-Learn` (ajuste do regressor Ensemble `RandomForestRegressor`, divisão de matrizes e validação cruzada).
* **Visualização de Dados:** `Seaborn` e `Matplotlib` (análise descritiva e diagnósticos estatísticos).
* **Dashboard Executivo:** `Plotly` integrado ao `Streamlit` / `Dash` (interface interativa para simulação de cenários de preços em tempo real).

---

## 🧮 Metodologia e Pipeline de Dados

O projeto está estruturado em blocos lógicos sequenciais de execução:

### 1. Higienização de Dados (Tratamento de Outliers)
A média aritmética simples é altamente sensível a valores extremos. Para garantir que o modelo aprenda com o mercado real e escalável, aplicamos o Intervalo Interquartil (IQR) para criar um teto de corte estável:

```text
IQR = Q₃ − Q₁
Limite Superior = Q₃ + (1.5 × IQR)
