#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Усовершенствованный модуль RAG с использованием модели DeepSeek-LLM-7B-Chat
с механизмом самопроверки и интеграцией с датасетом разработчика
УЛУЧШЕНО: Показывает конкретные файлы-источники вместо общего "RAG система"
"""
import logging
import torch
import re
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from ingest import client as qdrant, COLLECTION, embedder
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Настройка логирования
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Константы
MODEL_ID = "deepseek-ai/deepseek-llm-7b-chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 1024

# Загрузка модели и токенизатора
log.info(f"Загрузка модели {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
log.info("Модель загружена успешно")

class SmartDatasetManager:
    """Умный менеджер датасета разработчика с поиском по схожести."""
    
    def __init__(self, dataset_file: str = "developer_dataset.json"):
        self.dataset_file = Path(dataset_file)
        self.qa_pairs = self.load_qa_pairs()
        log.info(f"Загружено {len(self.qa_pairs)} пар Q&A из датасета разработчика")
    
    def load_qa_pairs(self) -> List[Dict]:
        """Загрузка пар вопрос-ответ из файла."""
        if self.dataset_file.exists():
            try:
                with open(self.dataset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    return []
            except Exception as e:
                log.error(f"Ошибка при загрузке датасета: {e}")
                return []
        return []
    
    def find_best_match(self, question: str) -> Tuple[Optional[str], float]:
        """
        Находит лучший ответ из датасета с оценкой уверенности.
        Возвращает (ответ, скор_уверенности) где скор от 0 до 1.
        """
        if not self.qa_pairs:
            return None, 0.0
        
        question_clean = self.clean_text(question)
        best_answer = None
        best_score = 0.0
        
        for pair in self.qa_pairs:
            pair_question = self.clean_text(pair["question"])
            
            # 1. Проверяем точное совпадение
            if question_clean == pair_question:
                log.info(f"Точное совпадение в датасете: {pair['question']}")
                return pair["answer"], 1.0
            
            # 2. Вычисляем схожесть
            similarity = self.calculate_similarity(question_clean, pair_question)
            
            if similarity > best_score:
                best_score = similarity
                best_answer = pair["answer"]
                best_question = pair["question"]
        
        if best_score > 0.3:  # Минимальный порог схожести
            log.info(f"Найдено похожее совпадение (скор: {best_score:.2f}): {best_question}")
        
        return best_answer, best_score
    
    def clean_text(self, text: str) -> str:
        """Очистка и нормализация текста для сравнения."""
        # Приводим к нижнему регистру
        text = text.lower().strip()
        # Удаляем знаки препинания
        text = re.sub(r'[^\w\s]', '', text)
        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Вычисление схожести текстов с учетом различных факторов."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # 1. Схожесть Жаккара
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # 2. Бонус за длинные совпадающие фразы
        phrase_bonus = self.calculate_phrase_bonus(text1, text2)
        
        # 3. Бонус за совпадающие ключевые слова
        keyword_bonus = self.calculate_keyword_bonus(words1, words2)
        
        # Итоговая оценка
        total_score = jaccard + phrase_bonus + keyword_bonus
        return min(total_score, 1.0)  # Максимум 1.0
    
    def calculate_phrase_bonus(self, text1: str, text2: str) -> float:
        """Бонус за совпадающие фразы."""
        words1 = text1.split()
        words2 = text2.split()
        max_phrase_len = 0
        
        for i in range(len(words1)):
            for j in range(len(words2)):
                if words1[i] == words2[j]:
                    phrase_len = 1
                    k, l = i + 1, j + 1
                    while (k < len(words1) and l < len(words2) and 
                           words1[k] == words2[l]):
                        phrase_len += 1
                        k += 1
                        l += 1
                    max_phrase_len = max(max_phrase_len, phrase_len)
        
        return min(max_phrase_len * 0.1, 0.2)  # Максимум 0.2 бонуса
    
    def calculate_keyword_bonus(self, words1: set, words2: set) -> float:
        """Бонус за важные ключевые слова."""
        # Список важных слов, которые дают дополнительный вес
        important_words = {
            'как', 'что', 'где', 'когда', 'почему', 'зачем', 'установить', 
            'настроить', 'создать', 'сделать', 'исправить', 'ошибка', 'проблема'
        }
        
        common_important = words1.intersection(words2).intersection(important_words)
        return min(len(common_important) * 0.05, 0.1)  # Максимум 0.1 бонуса
    
    def reload_dataset(self):
        """Перезагрузка датасета (для обновления в реальном времени)."""
        old_count = len(self.qa_pairs)
        self.qa_pairs = self.load_qa_pairs()
        new_count = len(self.qa_pairs)
        if new_count != old_count:
            log.info(f"Датасет перезагружен: {old_count} -> {new_count} записей")
        return new_count > old_count

# Глобальный экземпляр менеджера датасета
dataset_manager = SmartDatasetManager()

def analyze_answer_completeness(query: str, context: str) -> Tuple[bool, float]:
    """
    Анализирует, содержит ли контекст полный ответ на вопрос.
    Возвращает (is_complete, confidence_score)
    """
    query_lower = query.lower()
    context_lower = context.lower()
    
    # Определяем тип вопроса
    question_indicators = {
        'how': ['как', 'каким образом', 'способ'],
        'what': ['что', 'какой', 'какая', 'какое'],
        'where': ['где', 'куда', 'откуда'],
        'when': ['когда', 'во сколько'],
        'why': ['почему', 'зачем', 'для чего'],
        'who': ['кто', 'кого', 'кому']
    }
    
    question_type = None
    for q_type, indicators in question_indicators.items():
        if any(indicator in query_lower for indicator in indicators):
            question_type = q_type
            break
    
    # Базовая оценка на основе пересечения ключевых слов
    query_words = set(query_lower.split())
    context_words = set(context_lower.split())
    keyword_overlap = len(query_words.intersection(context_words)) / len(query_words) if query_words else 0
    
    # Дополнительные индикаторы полноты
    completeness_indicators = {
        'instructions': ['шаг', 'инструкция', 'следующий', 'сначала', 'затем', 'далее', 'после'],
        'definitions': ['это', 'означает', 'называется', 'представляет'],
        'locations': ['находится', 'расположен', 'адрес', 'место'],
        'procedures': ['процедура', 'процесс', 'алгоритм', 'метод', 'способ']
    }
    
    completeness_score = keyword_overlap
    
    # Бонусы за наличие структурированной информации
    if question_type == 'how':
        # Для вопросов "как" ищем пошаговые инструкции
        step_indicators = ['шаг', '1.', '2.', 'сначала', 'затем', 'после этого', 'далее']
        if any(indicator in context_lower for indicator in step_indicators):
            completeness_score += 0.2
            
    elif question_type == 'what':
        # Для вопросов "что" ищем определения
        definition_indicators = ['это', 'означает', 'представляет собой', 'является']
        if any(indicator in context_lower for indicator in definition_indicators):
            completeness_score += 0.15
    
    # Штрафы за незавершенность
    incomplete_indicators = ['см. также', 'подробнее', 'дополнительно', '...', 'продолжение']
    if any(indicator in context_lower for indicator in incomplete_indicators):
        completeness_score -= 0.1
    
    # Бонус за достаточную длину ответа
    if len(context) > 300:
        completeness_score += 0.1
    elif len(context) < 100:
        completeness_score -= 0.1
    
    # Ограничиваем оценку от 0 до 1
    completeness_score = max(0.0, min(1.0, completeness_score))
    
    is_complete = completeness_score > 0.6
    
    return is_complete, completeness_score

def clean_text(text: str) -> str:
    """Очистка и нормализация текста."""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def search_ctx(query: str, k: int = 8, threshold: float = 0.65) -> Tuple[str, List[str]]:
    """
    Улучшенный адаптивный поиск релевантного контекста в Qdrant.
    УЛУЧШЕНО: Более агрессивная фильтрация + анализ семантической связанности
    """
    log.info(f"Поиск контекста для запроса: '{query}'")
    try:
        collections = qdrant.get_collections().collections
        if COLLECTION not in {c.name for c in collections}:
            log.error(f"Коллекция {COLLECTION} не существует")
            return "База знаний пуста. Пожалуйста, загрузите документы и постройте индекс.", []

        # Векторизация запроса
        query_vector = list(embedder.embed([query]))[0]

        # Поиск с базовым порогом релевантности
        hits = qdrant.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
            score_threshold=threshold,
        )

        log.info(f"Найдено {len(hits)} релевантных результатов")
        if not hits:
            return "Не найдено релевантной информации по вашему запросу.", []

        # Сортировка результатов по релевантности
        hits = sorted(hits, key=lambda x: x.score, reverse=True)

        # Адаптивный выбор количества документов
        scores = [hit.score for hit in hits]
        log.info(f"Оценки релевантности: {scores}")

        # НОВОЕ: Анализ семантической связанности между фрагментами
        def analyze_semantic_relevance(query_text: str, fragments: list) -> list:
            """Анализирует семантическую связанность фрагментов с запросом."""
            query_words = set(query_text.lower().split())
            
            # Извлекаем ключевые слова из запроса
            important_query_words = set()
            stopwords = {'что', 'как', 'где', 'когда', 'почему', 'зачем', 'для', 'при', 'если', 'то', 'на', 'в', 'с', 'и', 'или', 'а', 'но'}
            for word in query_words:
                if len(word) > 2 and word not in stopwords:
                    important_query_words.add(word)
            
            relevant_fragments = []
            for hit in fragments:
                text = hit.payload.get("text", "").lower()
                text_words = set(text.split())
                
                # Вычисляем пересечение ключевых слов
                keyword_overlap = len(important_query_words.intersection(text_words)) / len(important_query_words) if important_query_words else 0
                
                # Проверяем наличие прямых ответов на вопрос
                direct_answer_indicators = ['решение', 'способ', 'инструкция', 'если', 'необходимо', 'следует', 'нужно']
                has_direct_answer = any(indicator in text for indicator in direct_answer_indicators)
                
                # Бонус за структурированность (списки, пункты)
                structure_bonus = 0.1 if any(marker in text for marker in ['1.', '2.', '•', '-', 'пункт']) else 0
                
                # Итоговая оценка семантической релевантности
                semantic_score = keyword_overlap + (0.1 if has_direct_answer else 0) + structure_bonus
                
                if semantic_score > 0.3:  # Порог семантической релевантности
                    relevant_fragments.append((hit, semantic_score))
                    log.info(f"Фрагмент принят: релевантность={hit.score:.3f}, семантика={semantic_score:.3f}")
                else:
                    log.info(f"Фрагмент отклонен: релевантность={hit.score:.3f}, семантика={semantic_score:.3f}")
            
            # Сортируем по комбинированной оценке
            relevant_fragments.sort(key=lambda x: x[0].score * 0.7 + x[1] * 0.3, reverse=True)
            return [item[0] for item in relevant_fragments]

        # Применяем семантический анализ
        semantically_relevant = analyze_semantic_relevance(query, hits)
        
        if not semantically_relevant:
            # Если семантический анализ ничего не нашел, берем лучший по базовой релевантности
            semantically_relevant = hits[:1]
            log.info("Семантический анализ не нашел релевантных фрагментов, используем лучший по базовой релевантности")

        # УЛУЧШЕННАЯ ЛОГИКА: Более агрессивная фильтрация с учетом семантики
        used_hits = semantically_relevant
        selection_reason = "семантический анализ"

        # Работаем только с семантически релевантными фрагментами
        semantic_scores = [hit.score for hit in semantically_relevant]
        
        if len(semantic_scores) > 1:
            # 1. Если есть явный лидер по релевантности (разрыв > 0.06) - используем только его
            if (semantic_scores[0] - semantic_scores[1]) > 0.06:
                used_hits = semantically_relevant[:1]
                selection_reason = f"явный семантический лидер (разрыв {semantic_scores[0] - semantic_scores[1]:.3f})"
                log.info(f"Найден явный семантический лидер: {semantic_scores[0]:.3f} vs {semantic_scores[1]:.3f}")

            # 2. Если топ документ очень релевантен (>0.8) и есть заметный разрыв (>0.04) - используем только его
            elif semantic_scores[0] > 0.8 and (semantic_scores[0] - semantic_scores[1]) > 0.04:
                used_hits = semantically_relevant[:1]
                selection_reason = f"высокая семантическая релевантность лидера ({semantic_scores[0]:.3f})"

            # 3. Проверяем, относятся ли фрагменты к одной теме
            else:
                # Анализируем тематическое единство
                first_text = semantically_relevant[0].payload.get("text", "").lower()
                
                related_fragments = [semantically_relevant[0]]  # Всегда включаем лучший
                
                for hit in semantically_relevant[1:3]:  # Проверяем до 2 следующих
                    other_text = hit.payload.get("text", "").lower()
                    
                    # Проверяем тематическую связанность
                    first_words = set(first_text.split())
                    other_words = set(other_text.split())
                    
                    # Ключевые слова из первого фрагмента
                    first_keywords = {word for word in first_words if len(word) > 4}
                    other_keywords = {word for word in other_words if len(word) > 4}
                    
                    # Если есть значительное пересечение ключевых слов - фрагменты связаны
                    if first_keywords and other_keywords:
                        keyword_similarity = len(first_keywords.intersection(other_keywords)) / len(first_keywords.union(other_keywords))
                        
                        if keyword_similarity > 0.2:  # Порог тематической связанности
                            related_fragments.append(hit)
                            log.info(f"Фрагмент тематически связан (similarity: {keyword_similarity:.3f})")
                        else:
                            log.info(f"Фрагмент тематически НЕ связан (similarity: {keyword_similarity:.3f}), исключаем")
                            break  # Прерываем, если нашли несвязанный фрагмент
                    else:
                        break  # Если нет ключевых слов для сравнения
                
                used_hits = related_fragments
                selection_reason = f"тематически связанные фрагменты ({len(used_hits)} шт.)"

        # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: Анализ полноты ответа в лучшем документе
        if len(used_hits) > 1:
            best_hit = used_hits[0]
            best_text = best_hit.payload.get("text", "").strip()
            
            # Проверяем, содержит ли лучший фрагмент структурированный ответ
            structured_answer_indicators = [
                'решение', 'способ решения', 'инструкция', 'алгоритм',
                '1.', '2.', '3.', 'шаг', 'пункт', 'необходимо', 'следует'
            ]
            
            has_structured_answer = sum(1 for indicator in structured_answer_indicators if indicator in best_text.lower())
            
            # Простая эвристика: если в лучшем документе есть ключевые слова вопроса
            # и структурированный ответ, возможно, этого достаточно
            query_words = set(query.lower().split())
            text_words = set(best_text.lower().split())
            overlap = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
            
            # Если пересечение слов высокое И есть структурированный ответ - используем только лучший
            if ((overlap > 0.3 and has_structured_answer >= 2) or 
                (overlap > 0.5 and len(best_text) > 150) or
                best_hit.score > 0.85):
                used_hits = [best_hit]
                selection_reason += " + полный структурированный ответ"
                log.info(f"Определено, что лучший фрагмент содержит полный ответ (пересечение: {overlap:.2f}, структура: {has_structured_answer})")

        log.info(f"Используется {len(used_hits)} из {len(hits)} документов. Стратегия: {selection_reason}")

        # Формирование контекста и сбор источников
        context_parts = []
        source_files = set()  # Используем set для уникальных файлов
        
        for i, hit in enumerate(used_hits):
            text = hit.payload.get("text", "").strip()
            source = hit.payload.get("source", hit.payload.get("file", "Неизвестный источник"))
            source_files.add(source)  # Добавляем файл в набор источников
            
            if text:
                text = clean_text(text)
                context_parts.append(
                    f"[Источник {i+1}: {source}, релевантность: {hit.score:.2f}]\n{text}"
                )

        if not context_parts:
            return "Найденные документы не содержат текстовой информации.", []

        search_info = (
            f"[Инфо: найдено {len(hits)} док., семантически релевантных {len(semantically_relevant)}, использовано {len(used_hits)}, стратегия: {selection_reason}]"
        )

        context = search_info + "\n\n" + "\n\n".join(context_parts)
        sources = list(source_files)  # Преобразуем set обратно в список
        
        return context, sources

    except Exception as e:
        log.error(f"Ошибка при поиске контекста: {e}")
        return f"Произошла ошибка при поиске: {e}", []

def search_in_specific_file(query: str, filename: str, k: int = 5) -> str:
    """Поиск только в конкретном файле."""
    log.info(f"Поиск в файле '{filename}' по запросу: '{query}'")
    try:
        collections = qdrant.get_collections().collections
        if COLLECTION not in {c.name for c in collections}:
            return "База знаний пуста."

        # Векторизация запроса
        query_vector = list(embedder.embed([query]))[0]

        # Поиск с фильтрацией по файлу (упрощенный способ)
        # Сначала получаем все релевантные результаты
        all_hits = qdrant.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=k * 3,  # Получаем больше результатов для фильтрации
            with_payload=True
        )
        
        # Фильтруем результаты по файлу в Python
        hits = [hit for hit in all_hits if hit.payload.get("file") == filename][:k]

        log.info(f"Найдено {len(hits)} результатов в файле {filename}")
        if not hits:
            return f"В файле '{filename}' не найдено релевантной информации по запросу."

        # Формирование контекста
        context_parts = []
        for i, hit in enumerate(hits):
            text = hit.payload.get("text", "").strip()
            if text:
                text = clean_text(text)
                context_parts.append(
                    f"[Фрагмент {i+1}, релевантность: {hit.score:.2f}]\n{text}"
                )

        return f"[Поиск в файле: {filename}]\n\n" + "\n\n".join(context_parts)

    except Exception as e:
        log.error(f"Ошибка при поиске в файле: {e}")
        return f"Произошла ошибка при поиске в файле: {e}"

def format_sources_citation(sources: List[str]) -> str:
    """Форматирует список источников в красивую строку цитирования."""
    if not sources:
        return "*[Источник: Неизвестно]*"
    
    if len(sources) == 1:
        return f"*[Источник: {sources[0]}]*"
    elif len(sources) <= 3:
        return f"*[Источники: {', '.join(sources)}]*"
    else:
        # Если источников много, показываем первые 3 + количество остальных
        first_three = ', '.join(sources[:3])
        remaining = len(sources) - 3
        return f"*[Источники: {first_three} и еще {remaining} файл(ов)]*"

def generate_base_prompt(query: str, context: str) -> str:
    """Создание базового промпта для генерации ответа."""
    return (
        f"Human: Вот запрос пользователя: \"{query}\"\n\n"
        "Используй только информацию из приведенного ниже контекста для формирования точного ответа.\n"
        "НЕ ПРИДУМЫВАЙ информацию, которой нет в контексте.\n"
        "Если в контексте содержатся разные процедуры или инструкции, не смешивай их.\n"
        "Если контекст противоречив, укажи на это явно.\n"
        "Ответ должен быть структурированным, конкретным и содержать ТОЛЬКО факты из контекста.\n"
        "Пиши на литературном русском языке без ошибок.\n\n"
        "Контекст:\n"
        f"{context}\n\n"
        f"Вопрос: {query}\n"
        "Assistant:"
    )

def generate_verification_prompt(query: str, context: str, initial_response: str) -> str:
    """Создание промпта для самопроверки и улучшения ответа."""
    return (
        "Human: Я дам тебе запрос, контекст и сгенерированный ответ.\n"
        "Проверь ответ и если нужно - ИСПРАВЬ его.\n"
        "ВАЖНО: В своём ответе дай ТОЛЬКО ИСПРАВЛЕННУЮ ВЕРСИЮ ответа без комментариев о проверке.\n\n"
        f"Запрос: \"{query}\"\n\n"
        f"Контекст:\n{context}\n\n"
        f"Первоначальный ответ:\n{initial_response}\n\n"
        "Дай улучшенный ответ (или тот же, если он хорош):\n"
        "Assistant:"
    )

def clean_verification_parts(response: str) -> str:
    """Удаляет части ответа, связанные с верификацией."""
    # Удаляем весь блок с проверкой
    patterns_to_remove = [
        r"Проверь:[\s\S]*$",
        r"\d+\.\s*Нет ли выдуманной информации[\s\S]*$",
        r"\d+\.\s*Полнота ответа[\s\S]*$", 
        r"\d+\.\s*Нет ли смешения тем[\s\S]*$",
        r"\d+\.\s*Правильный порядок шагов[\s\S]*$",
        r"^Нет,?\s*ответ не содержит выдуманной информации[\s\S]*$",
        r"^Ответ[\s\S]*соответствует[\s\S]*порядку шагов[\s\S]*$",
        r"В ответе указано[\s\S]*$"
    ]
    
    cleaned = response.strip()
    
    # Применяем паттерны для удаления
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
    
    # Удаляем лишние пробелы и переносы
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

@torch.inference_mode()
def generate_response(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    """Генерация ответа по промпту."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.3,
        num_beams=4,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id  # Добавляем pad_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Более надёжное извлечение ответа
    if "Assistant:" in text:
        response = text.split("Assistant:")[-1].strip()
    else:
        response = text.replace(prompt, "").strip()
    
    return response

def generate(query: str) -> str:
    """Основная функция генерации с улучшенной логикой верификации и отображением источников."""
    log.info(f"Генерация ответа на запрос: '{query}'")
    
    try:
        # Перезагружаем датасет для получения свежих данных
        dataset_manager.reload_dataset()
        
        # Проверяем датасет разработчика
        dataset_answer, confidence_score = dataset_manager.find_best_match(query)
        
        # Логика принятия решения на основе уверенности
        if dataset_answer and confidence_score >= 0.8:
            log.info(f"Высокая уверенность в датасете ({confidence_score:.2f}), используем ответ разработчика")
            return f"{dataset_answer}\n\n*[Источник: База знаний разработчика, уверенность: {confidence_score:.0%}]*"
        
        elif dataset_answer and confidence_score >= 0.5:
            log.info(f"Средняя уверенность в датасете ({confidence_score:.2f}), комбинируем с RAG")
            context, sources = search_ctx(query)
            
            if context.startswith(("Произошла ошибка", "Не найдено", "База знаний")):
                return f"{dataset_answer}\n\n*[Источник: База знаний разработчика, уверенность: {confidence_score:.0%}]*"
            
            # Упрощённая комбинация без верификации
            combined_prompt = (
                f"Human: Ответь на вопрос: \"{query}\"\n\n"
                f"Информация из базы знаний:\n{dataset_answer}\n\n"
                f"Дополнительная информация:\n{context}\n\n"
                "Дай полный ответ, используя всю доступную информацию.\n"
                "Assistant:"
            )
            
            combined_response = generate_response(combined_prompt)
            
            # Форматируем источники для комбинированного ответа
            if sources:
                sources_citation = format_sources_citation(sources)
                sources_info = f"База знаний разработчика + {sources_citation.replace('*[Источник', '').replace(']*', '').replace('и:', '')}"
            else:
                sources_info = "База знаний разработчика"
            
            return f"{combined_response}\n\n*[Источник: {sources_info}]*"
        
        else:
            # Используем только RAG
            log.info("Используем RAG систему")
            context, sources = search_ctx(query)
            
            if context.startswith(("Произошла ошибка", "Не найдено", "База знаний")):
                if dataset_answer and confidence_score > 0.3:
                    return f"{dataset_answer}\n\n*[Источник: База знаний разработчика (слабое совпадение), уверенность: {confidence_score:.0%}]*"
                return context

            # НОВОЕ: Анализируем полноту контекста
            is_complete, completeness_score = analyze_answer_completeness(query, context)
            log.info(f"Анализ полноты контекста: {is_complete} (оценка: {completeness_score:.2f})")

            # Упрощённый промпт без двойной верификации
            if is_complete and completeness_score > 0.7:
                # Если контекст полный, используем более уверенный промпт
                simple_prompt = (
                    f"Human: Ответь на вопрос полно и точно: \"{query}\"\n\n"
                    "В контексте ниже содержится вся необходимая информация для ответа:\n"
                    f"{context}\n\n"
                    "Дай структурированный и полный ответ на основе этой информации.\n"
                    "Assistant:"
                )
            else:
                # Если контекст может быть неполным, делаем промпт более осторожным
                simple_prompt = (
                    f"Human: Ответь на вопрос на основе доступной информации: \"{query}\"\n\n"
                    "Используй информацию из контекста ниже:\n"
                    f"{context}\n\n"
                    "Если информация неполная, укажи это в ответе.\n"
                    "Assistant:"
                )
            
            response = generate_response(simple_prompt)
            
            # Простая очистка
            if query in response:
                response = response.replace(query, "").strip()
            
            if not response or len(response) < 10:
                return "Не удалось сгенерировать ответ на основе контекста."
            
            # УЛУЧШЕНИЕ: Показываем конкретные файлы-источники + информацию о полноте
            sources_citation = format_sources_citation(sources)
            
            # Добавляем индикатор качества ответа
            if is_complete and completeness_score > 0.8:
                quality_indicator = " ✓ полный ответ"
            elif completeness_score > 0.6:
                quality_indicator = " ⚡ хороший ответ"  
            elif completeness_score > 0.4:
                quality_indicator = " ⚠ частичный ответ"
            else:
                quality_indicator = " ⚡ базовый ответ"
            
            return f"{response}\n\n{sources_citation.replace(']*', quality_indicator + ']*')}"

    except Exception as e:
        log.error(f"Ошибка при генерации ответа: {e}")
        return f"Произошла ошибка при генерации: {e}"

def test_rag(test_queries: list) -> dict:
    """Тестирование RAG-системы на наборе запросов."""
    results = {}
    for i, q in enumerate(test_queries, 1):
        log.info(f"Тест {i}/{len(test_queries)}: {q}")
        try:
            resp = generate(q)
            success = not resp.startswith(("Произошла ошибка", "Не удалось", "Не найдено"))
            results[q] = {"response": resp, "success": success}
        except Exception as e:
            log.error(f"Ошибка тестирования '{q}': {e}")
            results[q] = {"response": f"Ошибка: {e}", "success": False}
    return results