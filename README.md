# Gnose A - API de Análise de Portfólio e Predições

Sistema backend Django para análise de mercado financeiro, gerenciamento de portfólios de investimentos e predição API Endpoints

###  Sistema de Autenticação

A API utiliza dois mecanismos de segurança:

#### 1. API Key (Para endpoints de autenticação)
- Necessária para: `/auth/register/`, `/auth/login/`, `/auth/refresh/`
- Adicione no **header** de todas as requisições de autenticação:
```
X-Api-Key: sua-api-key-aqui
```

#### 2. JWT Token (Para demais endpoints)
- Necessário para todos os endpoints `/api/*`
- Após fazer login, você recebe dois tokens:
  - **Access Token**: Válido por **30 minutos** - use para requisições
  - **Refresh Token**: Válido por **7 dias** - use para renovar o access token

**Como usar o Access Token:**
Adicione no **header** de todas as requisições protegidas:
```
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

**Fluxo de autenticação:**
1. Faça login com `/auth/login/` (com API Key)
2. Guarde o `access` e `refresh` token recebidos
3. Use o `access` token nas requisições (válido 30 min)
4. Quando expirar, use `/auth/refresh/` para renovar (com API Key + refresh token)
5. Se o refresh token expirar (7 dias), faça login novamente

---

### Autenticação

| Método | Endpoint | Descrição | Requer API Key | Requer Token |
|--------|----------|-----------|----------------|--------------|
| POST | `/auth/register/` | Registra novo usuário | ✅ | ❌ |
| POST | `/auth/login/` | Login e obtenção de tokens JWT | ✅ | ❌ |
| POST | `/auth/logout/` | Logout do usuário | ✅ | ✅ |
| POST | `/auth/refresh/` | Renova access token | ✅ | ❌ |

**Exemplo - Registro:**s utilizando modelos LSTM.

---

## Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Tecnologias](#tecnologias)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Comandos de Gerenciamento](#comandos-de-gerenciamento)
- [API Endpoints](#api-endpoints)
- [Modelos de Dados](#modelos-de-dados)


---

##  Sobre o Projeto

O **Projeto B** é uma API REST desenvolvida em Django que oferece:

- **Coleta e Armazenamento**: Dados históricos de ações e ativos financeiros
- **Predições**: Modelos LSTM para prever próximos 5 dias de preços
- **Gerenciamento de Portfólio**: Criação, acompanhamento e otimização de carteiras
- **Análise de Risco**: Cálculo de métricas como VaR, Vol, Sharpe Ratio
- **Autenticação JWT**: Sistema seguro de autenticação de usuários

---

##  Tecnologias

- **Backend**: Django 5.2.2 + Django REST Framework
- **Machine Learning**: TensorFlow/Keras, scikit-learn, numpy
- **Dados Financeiros**: yfinance, pandas, bacen, FRED, investing.com
- **Banco de Dados**: SQLite 
- **Autenticação**: JWT (Simple JWT)
- **CORS**: django-cors-headers

---

##  Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/matheusff1/back-end-projeto-b.git
cd back-end-projeto-b/server
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv
```


### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

##  Configuração

### 1. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
DJANGO_SECRET_KEY=sua-chave-secreta-aqui
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=sqlite:///db.sqlite3
```

### 2. Execute as migrações

```bash
python manage.py migrate
```

### 3. Crie um superusuário (opcional)

```bash
python manage.py createsuperuser
```

### 4. Execute o servidor

```bash
python manage.py runserver 0.0.0.0:8000
```

Acesse: `http://localhost:8000`

---

##  Comandos de Gerenciamento

O projeto possui comandos customizados para automação de tarefas:

### `collect_data`
Coleta dados históricos iniciais dos ativos.

```bash
python manage.py collect_data
```

**O que faz:**
- Baixa dados históricos de todos os símbolos permitidos
- Popula o banco de dados com informações de mercado

---

### `update_market_data`
Atualiza dados de mercado com informações recentes.

```bash
python manage.py update_market_data
```

**O que faz:**
- Busca novos dados desde a última atualização
- Mantém o banco de dados sincronizado com o mercado
- **Recomendado**: Executar diariamente

---

### `run_and_save_predictions`
Executa modelos de predição e salva resultados.

```bash
python manage.py run_and_save_predictions
```

**O que faz:**
- Treina modelos LSTM para cada ativo
- Gera predições para os próximos 5 dias
- Calcula métricas de performance (MAE, loss)
- Salva resultados no banco de dados

**Parâmetros do modelo:**
- Janela temporal: 60 dias de entrada e 5 dias de saída
- Arquitetura: LSTM de uma camada com 160 neurônios
- Features: Seleção automática por correlação

---

### `update_portfolios_data`
Atualiza valores e distribuições dos portfólios de usuários.

```bash
python manage.py update_portfolios_data
```

**O que faz:**
- Recalcula valor atual dos portfólios
- Atualiza distribuição de ativos
- Registra histórico de rastreamento (tracking)
- **Recomendado**: Executar diariamente após `update_market_data`

---

##  API Endpoints

### Autenticação

| Método | Endpoint | Descrição | Autenticação |
|--------|----------|-----------|--------------|
| POST | `/auth/register/` | Registra novo usuário | Não |
| POST | `/auth/login/` | Login e obtenção de tokens JWT | Não |
| POST | `/auth/logout/` | Logout do usuário | Sim |
| POST | `/auth/refresh/` | Renova access token | Sim |

**Exemplo - Registro:**
```http
POST /auth/register/
X-Api-Key: sua-api-key-aqui
Content-Type: application/json

{
  "email": "joao@example.com",
  "username": "joao",
  "password": "senha123",
  "name": "João Silva",
  "phone": "11999999999"
}
```

**Resposta:**
```json
{
  "id": 1,
  "email": "joao@example.com",
  "username": "joao",
  "name": "João Silva",
  "phone": "11999999999",
  "created_at": "2024-10-16T10:30:00Z"
}
```

---

**Exemplo - Login:**
```http
POST /auth/login/
X-Api-Key: sua-api-key-aqui
Content-Type: application/json

{
  "email": "joao@example.com",
  "password": "senha123"
}
```

**Resposta:**
```json
{
  "refresh": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "access": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "id": 1,
    "email": "joao@example.com",
    "username": "joao",
    "name": "João Silva",
    "phone": "11999999999"
  }
}
```

 **Importante:** 
- `access` token expira em **30 minutos**
- `refresh` token expira em **7 dias**

---

**Exemplo - Renovar Access Token:**
```http
POST /auth/refresh/
X-Api-Key: sua-api-key-aqui
Content-Type: application/json

{
  "refresh": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

**Resposta:**
```json
{
  "access": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

---

**Exemplo - Logout:**
```http
POST /auth/logout/
X-Api-Key: sua-api-key-aqui
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
Content-Type: application/json

{
  "refresh": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

**Resposta:**
```json
{
  "detail": "Logout successful."
}
```

---

### Dados de Mercado

| Método | Endpoint | Descrição | Requer Token |
|--------|----------|-----------|--------------|
| GET | `/api/get_all_symbols/` | Lista todos os símbolos disponíveis | ✅ |
| GET | `/api/get_asset_historical_data/<symbol>/` | Dados históricos de um ativo | ✅ |
| GET | `/api/get_symbols_current_data/` | Dados atuais de múltiplos símbolos | ✅ |
| GET | `/api/get_assets_last_data/` | Últimos dados de múltiplos ativos | ✅ |

**Exemplo - Símbolos disponíveis:**
```http
GET /api/get_all_symbols/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

**Resposta:**
```json
{
  "symbols": [
    "VALE3.SA",
    "PETR4.SA",
    "ITUB4.SA",
    "AAPL",
    "NVDA",
    "MSFT"
  ]
}
```

**Exemplo - Dados históricos:**
```http
GET /api/get_asset_historical_data/PETR4.SA/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

**Resposta:**
```json
{
  "symbol": "PETR4.SA",
  "data": [
    {
      "date": "2024-01-02",
      "close": 38.50,
      "high": 39.00,
      "low": 38.20,
      "open": 38.80,
      "volume": 15000000
    },
    ...
  ]
}
```

**Exemplo - Dados atuais (múltiplos símbolos):**
```http
POST /api/get_symbols_current_data/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
Content-Type: application/json

{
  "symbols": ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
}
```

---

### Análise de Risco

| Método | Endpoint | Descrição | Requer Token |
|--------|----------|-----------|--------------|
| GET | `/api/get_asset_risk_data/<symbol>/` | Métricas de risco de um ativo | ✅ |

**Exemplo:**
```http
GET /api/get_asset_risk_data/PETR4.SA/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

**Resposta:**
```json
{
  "symbol": "PETR4.SA",
  "risk_metrics": {
    "volatility": 0.0234,
    "var_95": -0.0456,
    "cvar_95": -0.0623,
    "sharpe_ratio": 1.45,
    "max_drawdown": -0.1234
  }
}
```

---

### Predições

| Método | Endpoint | Descrição | Requer Token |
|--------|----------|-----------|--------------|
| GET | `/api/get_portfolio_assets_predictions/` | Predições de ativos do portfólio | ✅ |

**Exemplo:**
```http
GET /api/get_portfolio_assets_predictions/?id=1
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

**Resposta:**
```json
{
  "portfolio_id": 1,
  "predictions": [
    {
      "symbol": "PETR4.SA",
      "current_price": 38.50,
      "predictions": [38.75, 39.00, 39.20, 39.10, 39.30],
      "dates": ["2024-10-17", "2024-10-18", "2024-10-19", "2024-10-20", "2024-10-21"],
      "metrics": {
        "mae": 0.0234,
        "confidence": 0.87
      }
    },
    ...
  ]
}
```

---

### Portfólios

| Método | Endpoint | Descrição | Requer Token |
|--------|----------|-----------|--------------|
| POST | `/api/create_portfolio/` | Cria novo portfólio | ✅ |
| GET | `/api/get_user_portfolios/` | Lista portfólios do usuário | ✅ |
| GET | `/api/get_portfolio/` | Detalhes de um portfólio | ✅ |
| POST | `/api/get_and_save_portfolio_pnl/` | Calcula e salva PnL | ✅ |
| GET | `/api/get_portfolio_pnl/` | Histórico de PnL | ✅ |
| GET | `/api/get_portfolio_risk/` | Métricas de risco do portfólio | ✅ |
| GET | `/api/get_optimized_portfolio/` | Otimização de portfólio | ✅ |
| GET | `/api/get_starter_portfolio/` | Portfólio sugerido | ✅ |
| POST | `/api/delete_portfolio/` | Deleta um portfólio | ✅ |
| POST | `/api/add_asset_to_portfolio/` | Adiciona ativo ao portfólio | ✅ |
| POST | `/api/delete_asset_from_portfolio/` | Remove ativo do portfólio | ✅ |

**Exemplo - Criar portfólio:**
```http
POST /api/create_portfolio/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
Content-Type: application/json

{
  "name": "Meu Portfólio Tech",
  "description": "Focado em empresas de tecnologia",
  "assets": [
    {
      "symbol": "AAPL",
      "quantity": 10,
      "price": 180.50
    },
    {
      "symbol": "MSFT",
      "quantity": 15,
      "price": 420.00
    },
    {
      "symbol": "NVDA",
      "quantity": 5,
      "price": 495.20
    }
  ]
}
```

**Resposta:**
```json
{
  "id": 1,
  "name": "Meu Portfólio Tech",
  "description": "Focado em empresas de tecnologia",
  "initial_balance": 10482.50,
  "current_balance": 10482.50,
  "assets": [...],
  "distribution": {
    "AAPL": 0.172,
    "MSFT": 0.601,
    "NVDA": 0.236
  },
  "created_at": "2024-10-16T10:30:00Z"
}
```

**Exemplo - Listar portfólios:**
```http
GET /api/get_user_portfolios/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

**Resposta:**
```json
{
  "portfolios": [
    {
      "id": 1,
      "name": "Meu Portfólio Tech",
      "current_balance": 10682.30,
      "initial_balance": 10482.50,
      "return": 0.0191,
      "return_percentage": "1.91%"
    },
    ...
  ]
}
```

**Exemplo - Obter risco do portfólio:**
```http
GET /api/get_portfolio_risk/?id=1
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

**Resposta:**
```json
{
  "portfolio_id": 1,
  "risk_metrics": {
    "portfolio_volatility": 0.0189,
    "var_95": -0.0312,
    "cvar_95": -0.0445,
    "sharpe_ratio": 1.67,
    "max_drawdown": -0.0823,
    "diversification_ratio": 0.78
  },
  "individual_risks": [
    {
      "symbol": "AAPL",
      "weight": 0.172,
      "contribution_to_risk": 0.0032
    },
    ...
  ]
}
```

**Exemplo - Otimizar portfólio:**
```http
GET /api/get_optimized_portfolio/?portfolio_id=1&min_return=0.0006
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
Content-Type: application/json
```

**Resposta:**
```json
{
  "optimized_weights": {
    "AAPL": 0.25,
    "MSFT": 0.30,
    "NVDA": 0.20,
    "GOOGL": 0.15,
    "META": 0.10
  },
  "expected_return": 0.157,
  "expected_volatility": 0.0234,
  "sharpe_ratio": 4.81
}
```

---

##  Modelos de Dados

### MarketData
Armazena dados históricos de mercado.

```python
{
  "id": 1,
  "symbol": "PETR4.SA",
  "date": "2024-10-16",
  "close": 38.50,
  "high": 39.00,
  "low": 38.20,
  "open": 38.80,
  "volume": 15000000
}
```

### Prediction
Armazena predições geradas por modelos LSTM.

```python
{
  "id": 1,
  "symbol": "PETR4.SA",
  "date": "2024-10-16",
  "prediction": [38.75, 39.00, 39.20, 39.10, 39.30],
  "results": {
    "metrics": {"mae": 0.0234, "loss": 0.00054},
    "history": {...},
    "selected_features": [("VALE3.SA", 0.85), ...]
  }
}
```

### Portfolio
Gerencia portfólios de usuários.

```python
{
  "id": 1,
  "user": 1,
  "name": "Meu Portfólio",
  "description": "Portfólio diversificado",
  "assets": [...],
  "initial_balance": 10000.00,
  "current_balance": 10250.00,
  "initial_distribution": {...},
  "current_distribution": {...},
  "created_at": "2024-10-16T10:30:00Z"
}
```

### PortfolioTracking
Rastreia evolução dos portfólios ao longo do tempo.

```python
{
  "id": 1,
  "portfolio": 1,
  "date": "2024-10-16",
  "balance": 10250.00,
  "distribution": {...},
  "return_value": 250.00,
  "return_percentage": 0.025
}
```

---

##  Rotinas Automáticas

# Atualizar dados de mercado às 01h
0 19 * * 1-5 cd /caminho/do/projeto && python manage.py update_market_data

# Atualizar portfólios às 01h30
30 19 * * 1-5 cd /caminho/do/projeto && python manage.py update_portfolios_data

# Executar predições  às 02h
0 2 * * 0 cd /caminho/do/projeto && python manage.py run_and_save_predictions



##  Deploy

### Usando ngrok (Desenvolvimento/Testes)

```bash
# Rode o servidor
python manage.py runserver 0.0.0.0:8000

# Em outro terminal
ngrok http 8000
```

Compartilhe a URL gerada: `https://abc123.ngrok-free.app`


