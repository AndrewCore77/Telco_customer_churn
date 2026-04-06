# Прогнозирование Оттока Клиентов в Телекоме

End-to-end ML case study на датасете IBM Telco Customer Churn.  
Этот проект собран как portfolio-grade пример того, как пройти путь от сырых клиентских данных до продуктово-ориентированной retention-стратегии: не просто предсказать churn, а перевести выводы модели в конкретные действия для маркетинга.

## Бизнес-проблема

Цель проекта — заранее находить клиентов с повышенным риском оттока и успевать вмешаться до того, как выручка будет потеряна.

В этом проекте churn prediction рассматривается как задача принятия решения:

- кто с высокой вероятностью уйдет;
- насколько уверена модель;
- каких клиентов стоит таргетировать в первую очередь;
- какой trade-off возникает между поимкой большего числа churn-клиентов и перерасходом retention-бюджета.


## Бизнес-эффект

Финальная модель здесь подается не как «просто классификатор», а как инструмент для retention-решений.

Ключевые бизнес-выводы:

- Выбранная модель, **Logistic Regression**, показала лучший общий баланс для proactive retention с `ROC-AUC = 0.842` и `Recall = 0.791` на дефолтном threshold.
- Threshold tuning показывает, что модель поддерживает разные operating modes:
  - **Aggressive retention (`0.30`)** ловит около `92.5%` churn-клиентов, но создает большую нагрузку на CRM.
  - **Balanced (`0.55`)** ловит около `75.4%` churn-клиентов и заметно сокращает число false positives.
  - **Conservative (`0.70`)** уменьшает объем кампании, но пропускает значительно больше реальных churn-клиентов.
- Если использовать `MonthlyCharges` как простой proxy выручки, balanced threshold выглядит разумным компромиссом между расходом retention-бюджета и риском потерянной выручки.

Важно: в датасете нет реальной стоимости кампаний, contribution margin и фактического uplift, поэтому финансовая интерпретация в проекте носит **концептуальный**, а не строго финансовый характер.

## Датасет

**Источник:** IBM Telco Customer Churn  
**Наблюдений:** 7,043 клиента  
**Исходных колонок:** 21  
**Target:** `Churn`

Датасет включает:

- демографию клиентов,
- срок жизни аккаунта,
- подключенные телеком-услуги,
- тип контракта,
- поведение в оплате,
- ежемесячные и накопленные платежи.

## Структура проекта

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       └── telco_churn_clean.csv
├── notebooks/
│   └── churn_prediction.ipynb
└── src/
    ├── data_preprocessing.py
    ├── evaluate.py
    ├── features.py
    ├── train.py
    └── utils.py
```

### 2. Exploratory Data Analysis

- проанализировано распределение churn и class imbalance;
- сравнены churn vs non-churn по ключевым числовым признакам;
- профилированы категориальные драйверы churn, такие как contract type, internet service, payment method и add-on services;
- сформулированы предмодельные бизнес-гипотезы для retention.

### 3. Добавления к данным

Добавлены легкие и интерпретируемые признаки, которые улучшают моделирование без утечки данных:

- `tenure_group`
- `is_new_customer`
- `num_services`
- `avg_monthly_spend_proxy`
- `has_auto_payment`
- 
### 5. Моделирование

- baseline classifier;
- Logistic Regression;
- Random Forest.

## Использованные модели

### Baseline

Dummy classifier, который нужен, чтобы показать, почему одной accuracy недостаточно в churn-задаче с несбалансированным target.

### Logistic Regression

Выбрана как основная модель, потому что сочетает:

- хорошее качество ранжирования;
- высокий recall для proactive retention;
- прозрачную интерпретацию через коэффициенты;
- более простую коммуникацию с продуктовой и маркетинговой командами.

### Random Forest

Используется как нелинейный benchmark, чтобы улавливать interaction effects и сравнивать precision/recall trade-offs с линейной моделью.

## Метрики

Основные метрики проекта:

- **Accuracy**
- **Precision**
- **Recall**
- **ROC-AUC**

### Результаты на test set

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline | 0.7346 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.7324 | 0.4975 | 0.7914 | 0.8420 |
| Random Forest | 0.7729 | 0.5551 | 0.7273 | 0.8405 |

Интерпретация:

- **Baseline** выглядит приемлемо по accuracy, но полностью проваливается в ловле churn-клиентов;
- **Random Forest** сильнее по accuracy и precision;
- **Logistic Regression** сильнее по recall и немного лучше по ROC-AUC, поэтому лучше подходит для proactive retention.

## Ключевые инсайты

Проект показывает консистентную бизнес-историю, которая повторяется в EDA, коэффициентах Logistic Regression и feature importance у Random Forest.

### Самые рискованные клиенты не случайны

High-risk клиенты чаще всего:

- находятся на **month-to-month contracts**;
- имеют **короткий tenure**, особенно в первый год;
- пользуются **fiber optic internet**;
- платят через **electronic check**;
- не используют **Online Security**;
- не используют **Tech Support**.

### Churn выглядит как сочетание ценового давления, слабой привязки и низкой продуктовой вовлеченности

Сильнейшие churn-сигналы говорят о том, что клиенты уходят не из-за одного фактора, а из-за комбинации нескольких условий:

- низкий switching friction;
- нестабильность на раннем жизненном цикле;
- высокие или особенно заметные ежемесячные платежи;
- слабое использование “sticky” сервисных фич.

### Первый год особенно важен

Период onboarding и early-life — наиболее опасное окно по риску оттока. Это значит, что retention должен начинаться раньше, чем клиент явно проявит недовольство.

### Самый сильный high-risk сегмент реально пригоден для бизнеса

Один из самых сильных сегментов в проекте:

`Month-to-month + tenure <= 12 months + fiber optic`

На test sample этот сегмент показал особенно высокий фактический churn и выглядит сильным кандидатом для targeted CRM treatment.

## Бизнес-рекомендации

1. **В первую очередь таргетировать first-year month-to-month fiber customers.** Это самый очевидный high-risk кластер и лучший кандидат для приоритетного вмешательства.
2. **Тестировать contract-migration offers для рискованных month-to-month клиентов.** Ограниченные по времени annual-plan incentives или bundle credits могут работать эффективнее, чем широкие скидки для всех.
3. **Предлагать bundle или trial для Online Security и Tech Support у хрупких internet-клиентов.** Эти услуги стабильно связаны с более низким churn и могут усиливать stickiness.
4. **Запустить payment-friction кампанию для клиентов с electronic check.** Стимулировать переход на autopay через удобство, снижение friction и недорогие incentives.
5. **Использовать score-based retention tiers вместо одной общей кампании.** High-risk клиенты должны получать самые дорогие и сильные вмешательства, medium-risk — более легкие nudges.
6. **Построить early-life retention journey.** Сфокусироваться на первых 90-180 днях: onboarding, first-bill communication, раннее выявление проблем.
7. **Тестировать retention actions по сегментам, а не только глобально.** Правильный save playbook для price-sensitive newcomer не будет тем же самым, что и для billing-friction клиента.

## Как запустить локально

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/churn_prediction.ipynb
```

Если вы хотите использовать reusable Python modules напрямую, основные артефакты проекта такие:

- notebook: `notebooks/churn_prediction.ipynb`
- preprocessing helpers: `src/data_preprocessing.py`
- feature engineering и preprocessing: `src/features.py`
- training logic: `src/train.py`
- evaluation и interpretation helpers: `src/evaluate.py`

## Итог

Этот проект показывает, как превратить стандартный churn dataset в более сильный:

- с чистым ML pipeline;
- с понятным сравнением моделей;
- с интерпретируемыми факторами churn;
- с threshold tuning, привязанным к бизнес trade-offs;
- и с конкретной retention-стратегией, которую реально могут использовать product, CRM и marketing команды.
