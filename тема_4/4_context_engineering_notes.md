# КОНСПЕКТ ЛЕКЦІЇ №4: Контекст-інжиніринг

> **Тема:** як проєктувати контекст для LLM так, щоб відповідь була точною, керованою і безпечнішою.
>
> **Формат:** самодостатній конспект із виправленнями помилок, поясненнями та практичними шаблонами.

---

## ЧАСТИНА 0. ЩО САМЕ МИ ВИВЧАЄМО

Після цієї лекції ви зможете:

- відрізняти `prompt engineering` від `context engineering`;
- пояснювати, як працює in-context learning без оновлення ваг моделі;
- обирати між zero-shot, few-shot, CoT, ReAct, ToT, GoT, PoT, Plan-and-Solve;
- розуміти межі long context і явище Lost in the Middle;
- зменшувати галюцинації через контекст, інструменти і валідацію;
- будувати безпечніші LLM-системи з урахуванням prompt injection ризиків;
- правильно структурувати ролі в OpenAI/Anthropic API.

### Короткі скорочення

| Термін | Що означає |
| --- | --- |
| `ICL` | In-Context Learning |
| `CoT` | Chain-of-Thought |
| `PoT` | Program-of-Thoughts |
| `ToT` | Tree-of-Thoughts |
| `GoT` | Graph-of-Thoughts |
| `RAG` | Retrieval-Augmented Generation |
| `MSJ` | Many-Shot Jailbreaking |
| `VLM` | Vision-Language Model |
| `QA` | Question Answering |
| `Q/K/V` | Query / Key / Value у механізмі уваги |
| `OWASP` | Open Web Application Security Project |

---

## ЧАСТИНА 1. IN-CONTEXT LEARNING: ЯК МОДЕЛЬ "НАВЧАЄТЬСЯ" В ПРОМПТІ

### 1.1 Що таке ICL

**In-Context Learning** — це здатність моделі використовувати приклади та інструкції з поточного запиту без оновлення її параметрів.

- Модель бере патерн із контексту (приклади "вхід -> вихід") і переносить на новий випадок.
- Це не fine-tuning: ваги моделі не змінюються.
- Правила живуть доти, доки є в контексті цього запиту/діалогу.

Класичний прорив показала GPT-3 (2020): few-shot приклади в промпті значно покращили вирішення нових задач.

![Few-shot translation](images/few_shot_translation_ludwig.png)

Джерела:
- [GPT-3 / few-shot і ICL](https://arxiv.org/abs/2005.14165)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Ludwig: In-Context Learning](https://ludwig.ai/latest/user_guide/llms/in_context_learning/)

### 1.2 Якість прикладів важливіша за кількість

- 2-5 сильних прикладів часто дають більший приріст, ніж десятки однотипних.
- Більше прикладів = більше токенів, вища ціна, більший ризик шуму.
- "Плато після 2-5 прикладів" — **корисна евристика**, не універсальний закон.

Практика:
- покривайте різні типи кейсів;
- прибирайте дублікати;
- додавайте приклади тільки якщо вони вимірно покращують якість.

Джерело:
- [The few-shot prompting guide](https://www.prompthub.us/blog/the-few-shot-prompting-guide)

### 1.3 Порядок прикладів має значення (order sensitivity)

Однаковий набір few-shot демонстрацій у різному порядку може дати різну точність.

- Медіана та квартилі на box plot показують розподіл якості, а не "межі правильної відповіді".
- Якщо медіана не по центру прямокутника — це нормально: розподіл просто асиметричний.

Міні-приклад (9 значень): `50, 60, 70, 80, 85, 87, 88, 89, 90`
- нижній квартиль: `70`
- медіана: `85`
- верхній квартиль: `88`

Тут медіана ближче до верхнього краю коробки, бо верхня частина значень щільніша.

### 1.4 Lost in the Middle: моделі "люблять" край контексту

![Lost in the Middle](images/lost_in_the_middle_accuracy.png)

У long-context задачах важливий факт, розміщений у середині, часто використовується гірше, ніж на початку/в кінці.

- Це виявлено у дослідженні 2023 року.
- Новіші моделі частково пом'якшують ефект, але ризик залишається.
- У 2025-2026 роботах показано: впливає не лише позиція, а й відстань між фрагментами, які треба поєднати.

Практика:
- ставте критичні інструкції на початку;
- конкретне питання/критерій відповіді — ближче до кінця;
- для критичних фактів допускається контрольне повторення наприкінці.

Джерела:
- [Lost in the Middle (arXiv)](https://arxiv.org/abs/2307.03172)
- [ACL 2025 findings](https://aclanthology.org/2025.findings-acl.28/)
- [Long context follow-up 2026](https://arxiv.org/abs/2605.23170)
- [Пояснювальний огляд](https://the-decoder.com/large-language-models-and-the-lost-middle-phenomenon/)

### 1.5 Контекстна деградація і що означає `n^2`

Фраза "кожен токен погіршує якість" — надто категорична. Правильніше так:

- із ростом контексту зростають обчислювальні витрати;
- сигнал-шум часто погіршується;
- релевантну інформацію важче "витягти", якщо контекст перевантажений.

Чому згадують `n^2`:
- у повній self-attention для `n` токенів треба обчислити приблизно `n * n` взаємодій;
- це пояснює вартість і latency, але **саме по собі не доводить** причину галюцинацій.

### 1.6 Zero-shot vs Few-shot

![Zero-shot vs Few-shot](images/zero_shot_vs_few_shot_neptune.png)

**Zero-shot**: без прикладів, тільки інструкція.  
**Few-shot**: додаємо приклади, щоб зафіксувати формат/межі.

Коли zero-shot зазвичай достатній:
- типове завдання з чітким форматом;
- сильна instruction-tuned або reasoning-модель;
- низька неоднозначність.

Коли few-shot потрібен:
- нестандартні категорії;
- тонка грань між схожими класами;
- важливий стиль або структура відповіді.

---

## ЧАСТИНА 2. CONTEXT ENGINEERING VS PROMPT ENGINEERING

### 2.1 Ключова різниця

- **Prompt engineering**: як сформулювати конкретну інструкцію.
- **Context engineering**: як побудувати **повний контекстовий простір**: ролі, правила, джерела, пам'ять, інструменти, формат виходу, guardrails.

![Context engineering components](images/context_engineering_components.png)

Тобто prompt engineering — частина context engineering.

Джерело:
- [Context Engineering: beyond prompt engineering and RAG](https://thenewstack.io/context-engineering-going-beyond-prompt-engineering-and-rag/)

### 2.2 Компоненти контексту в продакшн-системі

| Компонент | Що робить |
| --- | --- |
| `System/Developer instructions` | Визначає межі поведінки моделі |
| `User query` | Поточна задача |
| `Short-term memory` | Історія/резюме поточного діалогу |
| `Long-term memory` | Збережені факти між сесіями |
| `RAG retrieval` | Підтягує релевантні документи |
| `Tool results` | Дані з калькулятора, БД, пошуку, коду |
| `Structured output constraints` | JSON schema, required поля, типи |
| `Guardrails` | Політики, фільтри, перевірки |

### 2.3 Що таке guardrails на практиці

Guardrails — це не лише "напис у промпті". Це і програмні перевірки:

- перевірка доступів до інструментів;
- валідація формату JSON;
- фільтрація небезпечних дій;
- human approval для ризикових операцій.

### 2.4 Параметри моделі vs гіперпараметри

| Категорія | Приклади | Де змінюється |
| --- | --- | --- |
| Параметри моделі | ваги, зсуви (`W`, `b`) | під час навчання |
| Гіперпараметри | lr, batch size, кількість шарів | задаються розробником |

---

## ЧАСТИНА 3. ЕВОЛЮЦІЯ ТЕХНІК ПРОМПТИНГУ Й КОНТЕКСТ-ІНЖИНІРИНГУ

### 3.1 2020: Few-shot ICL

- GPT-3 показала, що формат "приклад -> новий кейс" може замінити окреме донавчання в частині задач.
- Це започаткувало масовий практичний prompt-based підхід.

### 3.2 2021: PET (Pattern-Exploiting Training)

![PET template](images/pet_template_mask.png)

Важливе уточнення:
- PET — це не тільки "хитрий промпт".
- PET включає **навчання**: ваги моделі оновлюються на шаблонізованих задачах.

Джерело:
- [PET paper](https://aclanthology.org/2021.eacl-main.20/)

### 3.3 2021-2022: Instruction Tuning / FLAN

- `FLAN` = Finetuned Language Net.
- Модель донавчали на великій множині задач у форматі інструкцій.
- Висновок: zero-shot сильно покращується, якщо модель тренували виконувати інструкції.

Важливе уточнення:
- zero-shot не означає "модель не навчали";
- це означає "в поточному запиті немає прикладів".

Джерело:
- [Finetuned language models are zero-shot learners (FLAN)](https://research.google/pubs/finetuned-language-models-are-zero-shot-learners/)

### 3.4 Січень 2022: Chain-of-Thought (CoT)

![CoT example](images/chain_of_thought_example.jpg)

CoT просить модель явно генерувати проміжні кроки перед фінальною відповіддю.

Виправлення неточності:
- у класичній роботі CoT (січень 2022) використовували few-shot демонстрації з розв'язанням;
- фраза "Let's think step by step" без прикладів — це **Zero-shot CoT** (травень 2022, окрема робота).

Джерела:
- [CoT paper](https://arxiv.org/abs/2201.11903)
- [Zero-shot CoT](https://arxiv.org/abs/2205.11916)

### 3.5 Що таке PaLM і GSM8K

- `PaLM` (Pathways Language Model) — велика модель Google.
- `GSM8K` — набір шкільних математичних задач (~8.5k).

Коректне формулювання результату:
- "state-of-the-art на той момент", а не універсальне твердження "рівень людини назавжди".

Джерела:
- [PaLM](https://arxiv.org/abs/2204.02311)
- [GSM8K](https://github.com/openai/grade-school-math)

### 3.6 Травень 2022: Least-to-Most

Підхід:
1. розкласти задачу на підзадачі;
2. розв'язувати послідовно, передаючи результат кроку далі.

Плюс:
- краще для складних задач, де прямий CoT "ламається".

Джерело:
- [Least-to-Most Prompting](https://arxiv.org/abs/2205.10625)

### 3.7 Березень 2022 / ICLR 2023: Self-Consistency

- Генеруємо кілька CoT-трас для однієї задачі.
- Обираємо найчастішу фінальну відповідь (majority vote).

Уточнення:
- це **голосування**, не усереднення;
- більшість може теж помилятися;
- витрати ростуть, бо запусків кілька.

Джерело:
- [Self-Consistency](https://arxiv.org/abs/2203.11171)

### 3.8 Жовтень 2022: ReAct (Reason + Act)

Цикл ReAct:

| Крок | Сенс |
| --- | --- |
| Thought | міркування про наступну дію |
| Action | виклик інструмента |
| Observation | результат інструмента |
| Answer/Next step | фінальна відповідь або нова дія |

Важлива межа:
- модель не виконує дію сама; це робить зовнішня програма.

Джерело:
- [ReAct](https://arxiv.org/abs/2210.03629)

### 3.9 Листопад 2022: Program-of-Thoughts (PoT)

PoT просить модель писати програму для обчислень.

Приклад:
- правильний синтаксис коду ще не гарантує правильну формулу;
- тобто PoT зменшує арифметичні помилки, але не прибирає логічні.

Джерело:
- [Program of Thoughts Prompting](https://arxiv.org/abs/2211.12588)

### 3.10 2023: Tree-of-Thoughts (ToT) і Graph-of-Thoughts (GoT)

![ToT scheme](images/tree_of_thoughts_scheme.png)
![GoT scheme](images/graph_of_thoughts_scheme.png)
![GoT plan example](images/got_login_test_plan.png)

**ToT**: розгалуження кількох варіантів і відсікання слабких гілок.  
**GoT**: загальніший граф, де можна об'єднувати часткові рішення.

Mermaid-ілюстрація ToT:

```mermaid
flowchart TD
    startTask["ПочатковаЗадача"] --> optionA["ВаріантA"]
    startTask --> optionB["ВаріантB"]
    startTask --> optionC["ВаріантC"]
    optionA --> nextA1["ПродовженняA1"]
    optionA --> nextA2["ПродовженняA2"]
    optionB --> deadEndB["ГлухийКутВідкинути"]
    optionC --> nextC1["ПродовженняC1"]
```

Mermaid-ілюстрація GoT:

```mermaid
flowchart TD
    mainTask["Задача"] --> partA["ЧастковеРішенняA"]
    mainTask --> partB["ЧастковеРішенняB"]
    partA --> mergeAB["ОбєднатиAB"]
    partB --> mergeAB
    mergeAB --> reviewNode["ПеревіритиРезультат"]
    reviewNode -->|"ПотрібніЗміни"| mergeAB
    reviewNode -->|"ПеревіркуПройдено"| finalAnswer["Відповідь"]
```

Навчальний Python-приклад GoT:

```python
from dataclasses import dataclass

@dataclass
class Node:
    text: str
    parents: list[str]

graph = {}
graph["task"] = Node(
    text="Створи план тестування форми входу: email і пароль.",
    parents=[],
)

def create_node(node_id, instruction, parents):
    context = "\n\n".join(
        f"{p}:\n{graph[p].text}" for p in parents
    )
    prompt = f"{instruction}\n\nВхідніМатеріали:\n{context}"
    answer = llm(prompt)  # умовний виклик моделі
    graph[node_id] = Node(text=answer, parents=parents)

create_node("positive", "Запропонуй позитивні перевірки.", ["task"])
create_node("negative", "Запропонуй негативні перевірки.", ["task"])
create_node("merged", "Обєднай перевірки в один план.", ["positive", "negative"])
create_node("review", "Знайди прогалини у плані.", ["task", "merged"])
create_node("final", "Виправ план з урахуванням review.", ["task", "merged", "review"])
```

Джерела:
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- [Graph of Thoughts](https://arxiv.org/abs/2308.09687)

### 3.11 Червень 2023: Function Calling і Structured Output

- Модель повертає структурований виклик інструмента.
- JSON schema значно зменшує форматні помилки.

Уточнення:
- правильний JSON не гарантує правильний зміст;
- валідація формату і валідація факту — різні кроки.

Приклад:

```json
{
  "name": "getWeather",
  "arguments": {
    "location": "Kyiv",
    "unit": "C"
  }
}
```

### 3.12 Травень 2023: Plan-and-Solve

- Спершу модель створює план, потім виконує його кроки.
- Це zero-shot стратегія, що зменшує пропуски кроків у складних задачах.

Уточнення:
- метод з'явився у 2023, не 2024;
- план знижує ризик пропусків, але не гарантує істинність.

Джерело:
- [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091)

### 3.13 2024-2026: reasoning-моделі

- Багато "старих хаків" промптингу стали менш критичними.
- Часто варто починати з чіткого zero-shot + schema + tools.
- Few-shot додаємо, якщо це вимірно піднімає якість.

Уточнення:
- промптинг не "помер";
- просто акцент змістився з трюків на системний дизайн (контекст + перевірки + інструменти).

Джерела:
- [OpenAI reasoning best practices](https://developers.openai.com/api/docs/guides/reasoning-best-practices)
- [OpenAI reasoning guide](https://developers.openai.com/api/docs/guides/reasoning)

---

## ЧАСТИНА 4. ГАЛЮЦИНАЦІЇ: ЩО ЦЕ І ЯК ЗМЕНШУВАТИ

### 4.1 Визначення

Галюцинація — правдоподібна, але непідтверджена або хибна відповідь.

Критичне розділення:
- **Factuality**: чи твердження істинне в реальному світі.
- **Faithfulness**: чи твердження випливає з наданого джерела.

Факт може бути істинним, але непідтвердженим наданим документом.

### 4.2 Типи галюцинацій (практичний список)

1. **Фактичні вигадки** — дати, цифри, "вигадані джерела".  
   Захист: вимагати посилання/цитати, дозволяти "не знаю".

2. **Контекстні суперечності** — відповідь конфліктує з даними в prompt/RAG.  
   Захист: правило "відповідай лише з контексту".

3. **Порушення інструкції** — ігнор формату/мови/обмежень.  
   Захист: JSON schema + поствалідація.

4. **Логічні збої** — самосуперечність, помилки міркування.  
   Захист: self-check, tool verification.

5. **Темпоральні зсуви** — застарілі твердження.  
   Захист: "станом на YYYY-MM", вказувати дату джерела.

6. **Небезпечні поради** — юридичні/фінансові/медичні перекручення.  
   Захист: policy + human review.

7. **Амальгамовані помилки** — змішування умов у нову хибну тезу.  
   Захист: розкладання умов списком перед відповіддю.

8. **Off-topic nonsense** — відхід від теми.  
   Захист: явні критерії релевантності.

9. **Task-specific збої** — код, QA, summary, мультимодальність.  
   Захист: спеціалізовані тести під кожен тип задачі.

### 4.3 Бенчмарки і метрики (що саме вони міряють)

| Набір/метод | Що міряє |
| --- | --- |
| `TruthfulQA` | Опір популярним, але хибним твердженням |
| `FEVER`, `HoVer` | Верифікація тверджень по доказах |
| `FActScore`, `QAFactEval` | Фактична узгодженість генерованого тексту |
| `NarrativeQA`, `LongBench` | Робота з довгим/складним контекстом |
| `Natural Questions`, `TriviaQA` | Фактологічне QA |
| `HumanEval`, `MBPP` | Коректність коду за тестами |
| `DocVQA`, `ChartQA`, `TextCaps` | Мультимодальні задачі документів/графіків/зображень |

Ключовий висновок:
- не всі ці тести прямо дорівнюють "рівню галюцинацій";
- це різні зрізи якості, які треба комбінувати.

### 4.4 Базовий антигалюцинаційний контур

1. retrieval (RAG / tools),
2. чіткі інструкції,
3. структурований output,
4. перевірка фактів/посилань,
5. fallback "недостатньо даних",
6. human-in-the-loop для high-risk.

---

## ЧАСТИНА 5. ПРОМПТ-ІН'ЄКЦІЇ ТА ДЖЕЙЛБРЕЙКИ

### 5.1 Prompt Injection vs Jailbreak

![Injection example](images/prompt_injection_invoice.png)
![Injection vs Jailbreak](images/injection_vs_jailbreak_table.png)

**Prompt injection**: ненадійний контент змушує систему змінити поведінку.  
**Jailbreak**: підклас атак, орієнтований на обхід safety-обмежень.

Важливе уточнення:
- у практиці терміни інколи перетинаються;
- в OWASP jailbreak зазвичай розглядають як різновид prompt injection.

Джерела:
- [OWASP LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
- [Arthur taxonomy overview](https://www.arthur.ai/blog/from-jailbreaks-to-gibberish-understanding-the-different-types-of-prompt-injections)

### 5.2 Чому "без конкатенації немає ін'єкції" — некоректно

Конкатенація рядків — частий механізм, але не єдина умова.

Ін'єкція можлива також через:
- RAG-документи,
- вебсторінки,
- email,
- зображення (multimodal),
- tool outputs.

Критичний критерій: **чи може ненадійний вміст стати керівною інструкцією**.

### 5.3 Хронологія (виправлена)

- 2022: популяризація атак після релізу ChatGPT; публічні розмови про prompt injection з'явилися ще до цього.
- 2023: активна еволюція jailbreak-промптів; adversarial suffix дослідження (літо 2023).
- 2024: many-shot jailbreaking, непрямі ін'єкції у веб/документи.
- 2025+: індустріалізація технік і ширший security фокус.

Важливі посилання:
- [Simon Willison: prompt injection (2022)](https://simonwillison.net/2022/Sep/12/prompt-injection/)
- [Universal adversarial suffix attacks](https://llm-attacks.org/zou2023universal.pdf)
- [Anthropic: Many-shot jailbreaking](https://www.anthropic.com/research/many-shot-jailbreaking)
- [CVE-2024-5184](https://www.cve.org/CVERecord?id=CVE-2024-5184)
- [The Guardian, 2024 case report](https://www.theguardian.com/technology/2024/dec/24/chatgpt-search-tool-vulnerable-to-manipulation-and-deception-tests-show)

### 5.4 Таксономія атак

![Direct injection techniques](images/direct_injection_techniques_table.png)
![ASCII jailbreak](images/artprompt_ascii_jailbreak.png)

1. **Прямі**: attacker пише інструкцію прямо в чат.
2. **Непрямі**: шкідливий текст ховається у зовнішньому контенті.
3. **Мультимодальні**: інструкція у зображенні/аудіо/метаданих.
4. **Many-shot**: довгий префікс "демонстрацій", що зсуває поведінку моделі.

![MSJ trend illustration](images/msj_shots_success_table.png)

Примітка: таблиця з процентами в ілюстрації — концептуальна; це не універсальні константи для всіх моделей.

### 5.5 Приклад безпечного many-shot пояснення (нешкідливий)

Сценарій:
- асистент-репетитор має давати підказки, а не готові відповіді;
- attack-префікс містить багато фейкових пар "учень -> готова відповідь";
- модель починає наслідувати патерн.

Це показує, що in-context learning може працювати і як корисний інструмент, і як вектор атаки.

### 5.6 Захист (практичні шари)

1. Явно розділяти trusted і untrusted контент.
2. Принцип найменших привілеїв для інструментів.
3. Валідувати output схемою і policy-check'ами.
4. Обмежувати rate, робити аномалійний моніторинг.
5. Red teaming і регулярні атак-тести.
6. Human approval для критичних дій.
7. Kill-switch / аварійні прапори для agent-функцій.

### 5.7 Обмеження захисту

- LLM працює з токенами і не має "вродженого" secure provenance розуміння.
- Надмірно жорсткі фільтри псують UX (false positives).
- Атаки швидко еволюціонують (гонка озброєнь).
- Автоматизований brute-force промптів масштабно перебирає обхідні варіанти.

---

## ЧАСТИНА 6. РОЛІ ПОВІДОМЛЕНЬ У СУЧАСНИХ API (OPENAI / ANTHROPIC)

### 6.1 OpenAI: roles і hierarchy

Історично (`chat.completions`) використовували `system`, `user`, `assistant`.  
У новіших сценаріях є `developer` для інструкцій розробника.

Уточнення:
- поведінка `system`/`developer` залежить від конкретної моделі та API;
- `Responses API` публічно анонсований 11 березня 2025.

Джерела:
- [OpenAI changelog](https://developers.openai.com/api/docs/changelog)
- [OpenAI prompt engineering guide](https://developers.openai.com/api/docs/guides/prompt-engineering)
- [Lunary note on developer role](https://lunary.ai/blog/openai-developer-role)
- [PyPI openai](https://pypi.org/project/openai/)

### 6.2 Приклад через OpenRouter (OpenAI-compatible)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_KEY",
)

resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "developer",
            "content": "Ти перекладач з англійської українською. Перекладай точно."
        },
        {
            "role": "user",
            "content": "Translate to Ukrainian: The quick brown fox jumps over the lazy dog."
        }
    ],
)
print(resp.choices[0].message.content)
```

### 6.3 Anthropic Claude: `system` + `messages`

У Claude:
- рольову рамку задає `system`;
- діалог переважно як `user`/`assistant`;
- інструменти описують окремо в tool use механіці.

Практики Anthropic:
- role prompting;
- чітка структуризація (зокрема XML-теги);
- явна інструкція про цитування джерел.

Важливе уточнення:
- теги типу `<scratchpad>` не "ховають" текст автоматично; це залежить від логіки вашого застосунку.

Джерела:
- [Claude API overview](https://platform.claude.com/docs/en/api/overview#python)
- [Claude prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#give-claude-a-role)
- [Claude tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)

---

## ЧАСТИНА 7. ПІДСУМОК І ПРАКТИЧНИЙ ЧЕКЛІСТ

### 7.1 Вісім головних висновків

1. Контекст — ресурс, який потрібно проєктувати, а не "заповнювати до межі".
2. ICL працює через поточний контекст, а не через зміну ваг моделі.
3. Якість і порядок few-shot прикладів критичніші за грубу кількість.
4. Long context не скасовує потребу в структурі й релевантності.
5. CoT/ToT/ReAct/PoT/PS — різні режими керування міркуванням та інструментами.
6. Structured output зменшує форматні збої, але не гарантує істинності змісту.
7. Prompt injection — фундаментальний ризик LLM-систем, особливо агентних.
8. Найкращий результат дає багатошаровий дизайн: контекст + tools + guardrails + evals + людина для high-risk.

### 7.2 Чекліст для QA AI Engineer

- Чи є чіткий поділ trusted/untrusted даних?
- Чи є fallback "недостатньо даних"?
- Чи перевіряється і формат, і фактологія?
- Чи протестовано порядок/позицію критичних фактів у контексті?
- Чи є red-team сценарії на direct/indirect injection?
- Чи обмежені права інструментів мінімально необхідним?
- Чи є логи для forensic аналізу інцидентів?
- Чи є kill-switch для відключення ризикових дій?

### 7.3 Для Data Analyst / Data Scientist / AI Engineer

- **Data Analyst:** LLM допомагає формувати SQL/Python і пояснення, але верифікація формул і висновків обов'язкова.
- **Data Scientist:** промпти і retrieval стратегії оцінюються як гіпотези на репрезентативному eval-сеті.
- **AI Engineer:** мислити системою, а не одним prompt'ом: roles, memory, RAG, tools, policy layer, observability.

---

## ДЖЕРЕЛА ЛЕКЦІЇ (ЗА ТЕМАМИ)

### ICL, CoT, long context, context engineering

- [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
- [https://ludwig.ai/latest/user_guide/llms/in_context_learning/](https://ludwig.ai/latest/user_guide/llms/in_context_learning/)
- [https://www.prompthub.us/blog/the-few-shot-prompting-guide](https://www.prompthub.us/blog/the-few-shot-prompting-guide)
- [https://arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)
- [https://the-decoder.com/large-language-models-and-the-lost-middle-phenomenon/](https://the-decoder.com/large-language-models-and-the-lost-middle-phenomenon/)
- [https://thenewstack.io/context-engineering-going-beyond-prompt-engineering-and-rag/](https://thenewstack.io/context-engineering-going-beyond-prompt-engineering-and-rag/)
- [https://aclanthology.org/2025.findings-acl.28/](https://aclanthology.org/2025.findings-acl.28/)
- [https://arxiv.org/abs/2605.23170](https://arxiv.org/abs/2605.23170)

### Еволюція методів reasoning/prompting

- [https://arxiv.org/abs/2205.10625](https://arxiv.org/abs/2205.10625)
- [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- [https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)
- [https://arxiv.org/abs/2211.12588](https://arxiv.org/abs/2211.12588)
- [https://arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601)
- [https://arxiv.org/abs/2305.04091](https://arxiv.org/abs/2305.04091)
- [https://www.superannotate.com/blog/chain-of-thought-cot-prompting](https://www.superannotate.com/blog/chain-of-thought-cot-prompting)
- [https://arxiv.org/abs/2205.11916](https://arxiv.org/abs/2205.11916)
- [https://arxiv.org/abs/2204.02311](https://arxiv.org/abs/2204.02311)
- [https://github.com/openai/grade-school-math](https://github.com/openai/grade-school-math)
- [https://aclanthology.org/2021.eacl-main.20/](https://aclanthology.org/2021.eacl-main.20/)
- [https://research.google/pubs/finetuned-language-models-are-zero-shot-learners/](https://research.google/pubs/finetuned-language-models-are-zero-shot-learners/)
- [https://arxiv.org/abs/2308.09687](https://arxiv.org/abs/2308.09687)

### Prompt injection / jailbreak / security

- [https://genai.owasp.org/llmrisk/llm01-prompt-injection/](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
- [https://www.arthur.ai/blog/from-jailbreaks-to-gibberish-understanding-the-different-types-of-prompt-injections](https://www.arthur.ai/blog/from-jailbreaks-to-gibberish-understanding-the-different-types-of-prompt-injections)
- [https://www.anthropic.com/research/many-shot-jailbreaking](https://www.anthropic.com/research/many-shot-jailbreaking)
- [https://www.darknet.org.uk/2025/10/llm-black-markets-in-2025-prompt-injection-jailbreak-sales-model-leaks/](https://www.darknet.org.uk/2025/10/llm-black-markets-in-2025-prompt-injection-jailbreak-sales-model-leaks/)
- [https://www.hiddenlayer.com/research/prompt-injection-attacks-on-llms](https://www.hiddenlayer.com/research/prompt-injection-attacks-on-llms)
- [https://simonwillison.net/2022/Sep/12/prompt-injection/](https://simonwillison.net/2022/Sep/12/prompt-injection/)
- [https://llm-attacks.org/zou2023universal.pdf](https://llm-attacks.org/zou2023universal.pdf)
- [https://www.cve.org/CVERecord?id=CVE-2024-5184](https://www.cve.org/CVERecord?id=CVE-2024-5184)
- [https://www.theguardian.com/technology/2024/dec/24/chatgpt-search-tool-vulnerable-to-manipulation-and-deception-tests-show](https://www.theguardian.com/technology/2024/dec/24/chatgpt-search-tool-vulnerable-to-manipulation-and-deception-tests-show)

### API ролі, Model Spec, Claude/OpenAI

- [https://pypi.org/project/openai/](https://pypi.org/project/openai/)
- [https://lunary.ai/blog/openai-developer-role](https://lunary.ai/blog/openai-developer-role)
- [https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning)
- [https://platform.claude.com/docs/en/api/overview#python](https://platform.claude.com/docs/en/api/overview#python)
- [https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- [https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [https://developers.openai.com/api/docs/changelog](https://developers.openai.com/api/docs/changelog)
- [https://developers.openai.com/api/docs/guides/reasoning](https://developers.openai.com/api/docs/guides/reasoning)
- [https://developers.openai.com/api/docs/guides/reasoning-best-practices](https://developers.openai.com/api/docs/guides/reasoning-best-practices)

### Додатково (джерело зображення про Neptune)

- [https://openai.com/index/openai-to-acquire-neptune/](https://openai.com/index/openai-to-acquire-neptune/)
