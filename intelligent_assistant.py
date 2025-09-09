#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Истинно проактивный ИИ-ассистент, который анализирует ВСЕ источники знаний:
- Документы в Qdrant
- Датасет Q&A разработчика
И генерирует персонализированные уточняющие вопросы
"""

import logging
import torch
import re
import json
from typing import Dict, List, Optional, Tuple, Set
from transformers import AutoModelForCausalLM, AutoTokenizer
from ingest import client as qdrant, COLLECTION, embedder
import pathlib

log = logging.getLogger(__name__)

class EnhancedIntelligentProactiveAssistant:
    """Умный проактивный ассистент с анализом всех источников знаний"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.knowledge_cache = {}
        self.qa_dataset = self._load_qa_dataset()
        
    def _load_qa_dataset(self) -> List[Dict]:
        """Загружает датасет Q&A разработчика"""
        try:
            dataset_path = pathlib.Path("developer_dataset.json")
            if dataset_path.exists():
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                    log.info(f"✅ Загружен датасет Q&A: {len(dataset)} пар")
                    return dataset
            else:
                log.info("ℹ️  Датасет Q&A не найден")
                return []
        except Exception as e:
            log.error(f"❌ Ошибка загрузки датасета Q&A: {e}")
            return []
    
    def reload_qa_dataset(self):
        """Перезагружает датасет Q&A (для обновлений в реальном времени)"""
        old_count = len(self.qa_dataset)
        self.qa_dataset = self._load_qa_dataset()
        new_count = len(self.qa_dataset)
        if new_count != old_count:
            log.info(f"🔄 Датасет Q&A обновлен: {old_count} -> {new_count} записей")

    def analyze_comprehensive_knowledge(self, user_query: str) -> Dict:
        """
        Комплексный анализ запроса по ВСЕМ источникам знаний:
        1. Документы в Qdrant
        2. Датасет Q&A разработчика
        """
        log.info(f"🔍 Комплексный анализ запроса: {user_query}")
        
        try:
            # Перезагружаем датасет для актуальности
            self.reload_qa_dataset()
            
            # 1. Поиск в документах Qdrant
            qdrant_results = self._search_qdrant_knowledge(user_query)
            
            # 2. Поиск в датасете Q&A
            qa_results = self._search_qa_dataset(user_query)
            
            # 3. Комбинированный анализ
            combined_analysis = self._combine_knowledge_sources(
                user_query, qdrant_results, qa_results
            )
            
            # 4. Определяем недостающую информацию
            missing_info = self._identify_comprehensive_missing_info(
                combined_analysis, user_query
            )
            
            # 5. Генерируем умные вопросы на основе всех источников
            smart_questions = self._generate_comprehensive_questions(
                user_query, combined_analysis, missing_info
            )
            
            return {
                "status": "comprehensive_analysis_complete",
                "questions": smart_questions,
                "qdrant_context": qdrant_results.get("context", ""),
                "qa_context": qa_results.get("context", ""),
                "combined_context": combined_analysis["full_context"],
                "missing_info": missing_info,
                "confidence": combined_analysis["confidence"],
                "sources": {
                    "qdrant_sources": qdrant_results.get("sources_count", 0),
                    "qa_matches": qa_results.get("matches_count", 0),
                    "total_sources": qdrant_results.get("sources_count", 0) + qa_results.get("matches_count", 0)
                }
            }
            
        except Exception as e:
            log.error(f"❌ Ошибка комплексного анализа: {e}")
            return {"status": "error", "questions": [], "context": str(e)}
    
    def _search_qdrant_knowledge(self, query: str) -> Dict:
        """Поиск в документах Qdrant"""
        try:
            # Проверяем наличие коллекции
            collections = qdrant.get_collections().collections
            if COLLECTION not in {c.name for c in collections}:
                return {"context": "", "sources_count": 0, "confidence": 0}
            
            # Векторизация и поиск
            query_vector = list(embedder.embed([query]))[0]
            hits = qdrant.search(
                collection_name=COLLECTION,
                query_vector=query_vector,
                limit=10,  # Ищем больше для анализа
                with_payload=True,
                score_threshold=0.3,
            )
            
            if not hits:
                return {"context": "", "sources_count": 0, "confidence": 0}
            
            # Анализ найденных документов
            documents_context = []
            sources = set()
            procedures = []
            required_fields = set()
            
            for hit in hits[:5]:  # Берем топ-5
                text = hit.payload.get("text", "")
                source = hit.payload.get("file", "unknown")
                
                documents_context.append(f"[{source}] {text}")
                sources.add(source)
                
                # Извлекаем процедурную информацию
                self._extract_procedure_info(text, procedures, required_fields)
            
            context = "\n\n".join(documents_context)
            confidence = max(hit.score for hit in hits) if hits else 0
            
            return {
                "context": context,
                "sources_count": len(sources),
                "confidence": confidence,
                "procedures": procedures[:5],
                "required_fields": list(required_fields)[:10],
                "hit_scores": [hit.score for hit in hits]
            }
            
        except Exception as e:
            log.error(f"❌ Ошибка поиска в Qdrant: {e}")
            return {"context": "", "sources_count": 0, "confidence": 0}
    
    def _search_qa_dataset(self, query: str) -> Dict:
        """Поиск в датасете Q&A разработчика"""
        if not self.qa_dataset:
            return {"context": "", "matches_count": 0, "confidence": 0}
        
        try:
            query_clean = self._clean_text_for_matching(query)
            matches = []
            
            for qa_pair in self.qa_dataset:
                question = qa_pair.get("question", "")
                answer = qa_pair.get("answer", "")
                
                # Вычисляем релевантность
                question_similarity = self._calculate_text_similarity(
                    query_clean, self._clean_text_for_matching(question)
                )
                
                answer_similarity = self._calculate_text_similarity(
                    query_clean, self._clean_text_for_matching(answer)
                )
                
                # Берем максимальную релевантность
                max_similarity = max(question_similarity, answer_similarity)
                
                if max_similarity > 0.2:  # Порог релевантности
                    matches.append({
                        "question": question,
                        "answer": answer,
                        "similarity": max_similarity,
                        "source": "developer_qa"
                    })
            
            # Сортируем по релевантности
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Формируем контекст из лучших совпадений
            context_parts = []
            for match in matches[:3]:  # Топ-3 совпадения
                context_parts.append(
                    f"[Q&A] Вопрос: {match['question']}\nОтвет: {match['answer']}"
                )
            
            context = "\n\n".join(context_parts)
            confidence = matches[0]["similarity"] if matches else 0
            
            return {
                "context": context,
                "matches_count": len(matches),
                "confidence": confidence,
                "best_matches": matches[:3],
                "all_similarities": [m["similarity"] for m in matches]
            }
            
        except Exception as e:
            log.error(f"❌ Ошибка поиска в датасете Q&A: {e}")
            return {"context": "", "matches_count": 0, "confidence": 0}
    
    def _clean_text_for_matching(self, text: str) -> str:
        """Очистка текста для сопоставления"""
        # Приводим к нижнему регистру
        text = text.lower().strip()
        # Удаляем знаки препинания, кроме важных
        text = re.sub(r'[^\w\s№\-]', '', text)
        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Расширенное вычисление схожести текстов"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # 1. Схожесть Жаккара
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # 2. Бонус за точные совпадения ключевых терминов
        key_terms = {"vlife", "бонус", "азс", "оплата", "карта", "топливо", "проблема"}
        key_matches = words1.intersection(words2).intersection(key_terms)
        key_bonus = len(key_matches) * 0.1
        
        # 3. Бонус за длинные совпадающие фразы
        phrase_bonus = self._calculate_phrase_similarity(text1, text2)
        
        # 4. Бонус за числовые совпадения (номера АЗС и т.д.)
        number_bonus = self._calculate_number_similarity(text1, text2)
        
        total_score = jaccard + key_bonus + phrase_bonus + number_bonus
        return min(total_score, 1.0)
    
    def _calculate_phrase_similarity(self, text1: str, text2: str) -> float:
        """Бонус за совпадающие фразы"""
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
        
        return min(max_phrase_len * 0.05, 0.15)
    
    def _calculate_number_similarity(self, text1: str, text2: str) -> float:
        """Бонус за совпадающие номера"""
        numbers1 = set(re.findall(r'\b\d+\b', text1))
        numbers2 = set(re.findall(r'\b\d+\b', text2))
        
        if numbers1 and numbers2:
            common_numbers = numbers1.intersection(numbers2)
            return min(len(common_numbers) * 0.1, 0.2)
        return 0
    
    def _combine_knowledge_sources(self, query: str, qdrant_results: Dict, qa_results: Dict) -> Dict:
        """Комбинирует информацию из всех источников"""
        
        # Формируем полный контекст
        contexts = []
        
        if qdrant_results.get("context"):
            contexts.append(f"=== ДОКУМЕНТЫ ===\n{qdrant_results['context']}")
        
        if qa_results.get("context"):
            contexts.append(f"=== БАЗА ЗНАНИЙ Q&A ===\n{qa_results['context']}")
        
        full_context = "\n\n".join(contexts)
        
        # Вычисляем общую уверенность
        qdrant_confidence = qdrant_results.get("confidence", 0)
        qa_confidence = qa_results.get("confidence", 0)
        
        # Взвешенная уверенность
        if qdrant_confidence > 0 and qa_confidence > 0:
            combined_confidence = (qdrant_confidence + qa_confidence) / 2
        else:
            combined_confidence = max(qdrant_confidence, qa_confidence)
        
        # Определяем доминирующий источник
        dominant_source = "hybrid"
        if qdrant_confidence > qa_confidence * 1.5:
            dominant_source = "documents"
        elif qa_confidence > qdrant_confidence * 1.5:
            dominant_source = "qa_dataset"
        
        # Собираем все процедуры и поля
        all_procedures = qdrant_results.get("procedures", [])
        all_required_fields = qdrant_results.get("required_fields", [])
        
        # Извлекаем дополнительную информацию из Q&A
        qa_procedures, qa_fields = self._extract_qa_procedures(qa_results)
        all_procedures.extend(qa_procedures)
        all_required_fields.extend(qa_fields)
        
        return {
            "full_context": full_context,
            "confidence": combined_confidence,
            "dominant_source": dominant_source,
            "all_procedures": list(set(all_procedures))[:7],
            "all_required_fields": list(set(all_required_fields))[:10],
            "source_breakdown": {
                "qdrant_confidence": qdrant_confidence,
                "qa_confidence": qa_confidence,
                "qdrant_sources": qdrant_results.get("sources_count", 0),
                "qa_matches": qa_results.get("matches_count", 0)
            }
        }
    
    def _extract_qa_procedures(self, qa_results: Dict) -> Tuple[List[str], List[str]]:
        """Извлекает процедуры и поля из результатов Q&A"""
        procedures = []
        fields = []
        
        for match in qa_results.get("best_matches", []):
            answer = match.get("answer", "")
            
            # Ищем пошаговые инструкции в ответах
            steps = re.findall(r'\d+\.\s+([^.]+)', answer)
            procedures.extend(steps[:2])
            
            # Ищем обязательные поля
            field_patterns = [
                r'[Нн]омер\s+([^.]+)',
                r'[Дд]ата\s+([^.]+)',
                r'[Сс]умма\s+([^.]+)',
                r'[Тт]елефон\s+([^.]+)',
            ]
            
            for pattern in field_patterns:
                matches = re.findall(pattern, answer)
                fields.extend(matches[:2])
        
        return procedures, fields
    
    def _extract_procedure_info(self, text: str, procedures: List, fields: Set):
        """Извлекает информацию о процедурах и обязательных полях"""
        
        # Ищем пошаговые инструкции
        step_patterns = [
            r'\d+\.\s+([^.]+)',
            r'[Шш]аг\s+\d+[:\.]?\s+([^.]+)',
            r'[Сс]начала\s+([^.]+)',
            r'[Зз]атем\s+([^.]+)',
            r'[Дд]алее\s+([^.]+)'
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, text)
            procedures.extend(matches[:3])
        
        # Ищем обязательные поля/требования
        field_patterns = [
            r'[Нн]ужно|[Тт]ребуется|[Уу]кажите|[Пп]редоставьте]\s+([^.]+)',
            r'[Нн]еобходимо\s+([^.]+)',
            r'№\s*(\w+)',
            r'[Дд]ата\s+([^.]+)',
            r'[Вв]ремя\s+([^.]+)',
            r'[Сс]умма\s+([^.]+)'
        ]
        
        for pattern in field_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:5]:
                if len(match.strip()) > 3 and len(match.strip()) < 50:
                    fields.add(match.strip())
    
    def _identify_comprehensive_missing_info(self, analysis: Dict, user_query: str) -> List[str]:
        """Определяет недостающую информацию на основе всех источников"""
        
        missing = []
        query_lower = user_query.lower()
        full_context = analysis["full_context"].lower()
        
        # Расширенные категории информации
        info_categories = {
            "location": {
                "keywords": ["азс", "адрес", "номер", "где", "заправка"],
                "importance": "высокая"
            },
            "time": {
                "keywords": ["когда", "дата", "время", "час", "день"],
                "importance": "высокая"
            },
            "amount": {
                "keywords": ["сумма", "объем", "литр", "рубл", "стоимость"],
                "importance": "средняя"
            },
            "contact": {
                "keywords": ["телефон", "номер", "связь", "контакт"],
                "importance": "средняя"
            },
            "document": {
                "keywords": ["документ", "справка", "номер", "чек"],
                "importance": "низкая"
            },
            "person": {
                "keywords": ["кто", "фамилия", "имя", "сотрудник"],
                "importance": "низкая"
            },
            "details": {
                "keywords": ["как именно", "подробно", "детали", "что конкретно"],
                "importance": "высокая"
            },
            "payment_method": {
                "keywords": ["карта", "наличные", "приложение", "терминал"],
                "importance": "средняя"
            },
            "vehicle": {
                "keywords": ["машина", "автомобиль", "марка", "номер авто"],
                "importance": "низкая"
            }
        }
        
        # Определяем тип запроса для приоритизации
        query_type = self._determine_query_type(query_lower)
        
        # Проверяем наличие информации
        mentioned_categories = set()
        for category, info in info_categories.items():
            if any(keyword in query_lower or keyword in full_context 
                   for keyword in info["keywords"]):
                mentioned_categories.add(category)
        
        # Определяем необходимые категории по типу запроса
        required_categories = self._get_required_categories_by_type(query_type)
        
        # Находим недостающие категории
        missing_categories = required_categories - mentioned_categories
        
        # Приоритизируем по важности
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for category in missing_categories:
            importance = info_categories.get(category, {}).get("importance", "низкая")
            if importance == "высокая":
                high_priority.append(category)
            elif importance == "средняя":
                medium_priority.append(category)
            else:
                low_priority.append(category)
        
        # Возвращаем в порядке приоритета
        missing.extend(high_priority)
        missing.extend(medium_priority[:2])  # Ограничиваем средний приоритет
        missing.extend(low_priority[:1])     # Только один низкий приоритет
        
        return missing[:4]  # Максимум 4 категории
    
    def _determine_query_type(self, query: str) -> str:
        """Определяет тип запроса для приоритизации вопросов"""
        
        if any(word in query for word in ["проблема", "ошибка", "не работает", "сломалось"]):
            return "problem"
        elif any(word in query for word in ["бонус", "баллы", "vlife", "лояльность"]):
            return "bonus"
        elif any(word in query for word in ["оплата", "карта", "деньги", "платеж"]):
            return "payment"
        elif any(word in query for word in ["топливо", "бензин", "дизель", "качество"]):
            return "fuel"
        elif any(word in query for word in ["как", "что", "где", "инструкция"]):
            return "instruction"
        else:
            return "general"
    
    def _get_required_categories_by_type(self, query_type: str) -> Set[str]:
        """Возвращает необходимые категории информации по типу запроса"""
        
        type_requirements = {
            "problem": {"location", "time", "details"},
            "bonus": {"location", "time", "contact", "payment_method"},
            "payment": {"amount", "time", "location", "payment_method"},
            "fuel": {"location", "time", "amount", "vehicle"},
            "instruction": {"details"},
            "general": {"details"}
        }
        
        return type_requirements.get(query_type, {"details"})
    
    @torch.inference_mode()
    def _generate_comprehensive_questions(self, user_query: str, analysis: Dict, missing_info: List[str]) -> List[str]:
        """Генерирует умные вопросы на основе всех источников знаний"""
        
        # Подготавливаем контекст
        context_summary = analysis["full_context"][:1000]  # Ограничиваем длину
        procedures = analysis.get("all_procedures", [])[:3]
        required_fields = analysis.get("all_required_fields", [])[:5]
        
        # Информация об источниках
        source_info = analysis.get("source_breakdown", {})
        dominant_source = analysis.get("dominant_source", "hybrid")
        
        # Создаем расширенный промпт
        prompt = f"""Human: Анализирую запрос пользователя: "{user_query}"

НАЙДЕННАЯ ИНФОРМАЦИЯ ИЗ ВСЕХ ИСТОЧНИКОВ:
{context_summary}

Ключевые процедуры: {'; '.join(procedures) if procedures else 'Не найдены'}
Обязательные поля: {'; '.join(required_fields) if required_fields else 'Не найдены'}

АНАЛИЗ ИСТОЧНИКОВ:
- Документы Qdrant: {source_info.get('qdrant_sources', 0)} источников (уверенность: {source_info.get('qdrant_confidence', 0):.2f})
- База знаний Q&A: {source_info.get('qa_matches', 0)} совпадений (уверенность: {source_info.get('qa_confidence', 0):.2f})
- Доминирующий источник: {dominant_source}

Недостающие категории информации: {', '.join(missing_info)}

ЗАДАЧА: Сгенерируй 2-3 КОНКРЕТНЫХ уточняющих вопроса, которые помогут получить недостающую информацию для решения проблемы пользователя.

Вопросы должны быть:
- Основаны на РЕАЛЬНОЙ информации из найденных источников
- Конкретными и практичными
- Учитывающими процедуры из документов И опыт из базы Q&A
- Помогающими определить точное решение

Формат ответа:
1. [конкретный вопрос]
2. [конкретный вопрос]  
3. [конкретный вопрос]

Assistant:"""

        try:
            # Генерируем ответ
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,  # Увеличиваем для более детальных вопросов
                do_sample=True,
                temperature=0.2,    # Снижаем для более фокусированных вопросов
                repetition_penalty=1.3,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем сгенерированные вопросы
            if "Assistant:" in response:
                questions_text = response.split("Assistant:")[-1].strip()
            else:
                questions_text = response.replace(prompt, "").strip()
            
            # Парсим вопросы
            questions = self._parse_generated_questions(questions_text)
            
            # Если LLM не сгенерировал вопросы, используем fallback
            if not questions:
                questions = self._generate_comprehensive_fallback_questions(
                    user_query, missing_info, analysis
                )
            
            return questions
            
        except Exception as e:
            log.error(f"❌ Ошибка генерации вопросов: {e}")
            return self._generate_comprehensive_fallback_questions(
                user_query, missing_info, analysis
            )
    
    def _parse_generated_questions(self, text: str) -> List[str]:
        """Парсит сгенерированные вопросы из ответа LLM"""
        questions = []
        
        # Ищем нумерованные вопросы
        patterns = [
            r'\d+\.\s*([^?\n]+\?)',  # "1. Вопрос?"
            r'\d+\)\s*([^?\n]+\?)',  # "1) Вопрос?"
            r'[•-]\s*([^?\n]+\?)',   # "• Вопрос?" или "- Вопрос?"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                clean_question = match.strip()
                if len(clean_question) > 15 and clean_question not in questions:
                    questions.append(clean_question)
        
        # Если паттерны не сработали, ищем любые вопросы
        if not questions:
            question_sentences = re.findall(r'[^.!?\n]+\?', text)
            for q in question_sentences:
                clean_q = q.strip()
                if len(clean_q) > 15 and clean_q not in questions:
                    questions.append(clean_q)
        
        return questions[:3]  # Максимум 3 вопроса
    
    def _generate_comprehensive_fallback_questions(self, user_query: str, missing_info: List[str], analysis: Dict) -> List[str]:
        """Генерирует fallback вопросы на основе комплексного анализа"""
        
        query_type = self._determine_query_type(user_query.lower())
        dominant_source = analysis.get("dominant_source", "hybrid")
        
        # Специализированные вопросы по типу запроса и источнику
        specialized_questions = {
            "problem": {
                "qdrant": [
                    "На какой конкретно АЗС возникла проблема? Согласно нашим процедурам, номер станции необходим для диагностики.",
                    "Когда именно это произошло? Это поможет найти операцию в системе и определить причину сбоя.",
                    "Опишите подробнее симптомы проблемы - это поможет применить правильную процедуру решения."
                ],
                "qa_dataset": [
                    "Судя по похожим случаям, важно знать: на какой АЗС это случилось и когда?",
                    "В аналогичных ситуациях помогало уточнение: какие именно действия вы предпринимали?",
                    "Исходя из опыта решения подобных проблем: есть ли у вас номер операции или чека?"
                ],
                "hybrid": [
                    "Для точной диагностики нужны детали: номер АЗС, дата/время и описание проблемы.",
                    "Чтобы применить правильную процедуру решения: опишите, что именно не работает?",
                    "На основе нашего опыта: какие действия вы уже предпринимали для решения?"
                ]
            },
            "bonus": {
                "qdrant": [
                    "Согласно регламенту Vlife: на какой АЗС и когда происходила заправка?",
                    "Для проверки начисления бонусов: какая была сумма покупки и способ оплаты?",
                    "По процедуре ОККС нужно знать: использовали ли вы карту лояльности или приложение?"
                ],
                "qa_dataset": [
                    "В похожих случаях помогало: номер АЗС, дата заправки и сумма операции.",
                    "Опыт показывает важность: как именно вы идентифицировались в системе Vlife?",
                    "Для быстрого решения: есть ли у вас чек или данные операции?"
                ],
                "hybrid": [
                    "Для решения проблемы с бонусами Vlife: номер АЗС, дата/время и сумма заправки?",
                    "Как происходила идентификация: через приложение, карту или номер телефона?",
                    "Проверялся ли баланс бонусов сразу после заправки?"
                ]
            },
            "payment": {
                "qdrant": [
                    "По процедуре возврата средств: номер АЗС, точное время и сумма операции?",
                    "Для проверки в системе: последние 4 цифры карты и тип операции?",
                    "Согласно регламенту: есть ли номер авторизации или чека?"
                ],
                "qa_dataset": [
                    "В аналогичных случаях важно: какой именно был тип ошибки платежа?",
                    "Опыт показывает: помогает номер терминала и точное время операции.",
                    "Для быстрого возврата: какой способ оплаты использовался?"
                ],
                "hybrid": [
                    "Для решения проблемы с оплатой: номер АЗС, время и сумма операции?",
                    "Что именно произошло: карта не прошла, деньги списались без топлива, или ошибка терминала?",
                    "Есть ли у вас SMS о списании или номер операции в банковском приложении?"
                ]
            }
        }
        
        # Базовые fallback вопросы
        fallback_mapping = {
            "location": "На какой АЗС это произошло? Укажите номер станции или адрес.",
            "time": "Когда именно это случилось? Нужны дата и примерное время.",
            "amount": "Какая была сумма операции или объем заправки?",
            "contact": "Какой номер телефона используете для связи с нами?",
            "details": "Опишите подробнее, что именно произошло и как это проявляется.",
            "payment_method": "Каким способом производилась оплата: карта, наличные или приложение?",
            "document": "Есть ли у вас номер чека, операции или другие документы?",
            "vehicle": "Какой автомобиль: марка, модель, гос. номер?"
        }
        
        questions = []
        
        # Сначала пытаемся использовать специализированные вопросы
        if query_type in specialized_questions:
            source_questions = specialized_questions[query_type].get(dominant_source, [])
            if source_questions:
                questions.extend(source_questions[:2])
        
        # Дополняем базовыми вопросами на основе недостающей информации
        for info_type in missing_info:
            if info_type in fallback_mapping and len(questions) < 3:
                questions.append(fallback_mapping[info_type])
        
        # Если все еще мало вопросов, добавляем универсальные
        if len(questions) < 2:
            universal_questions = [
                "Расскажите подробнее о вашей ситуации для точного решения проблемы.",
                "Какие действия вы уже предпринимали для решения этого вопроса?",
                "Есть ли дополнительные детали, которые могут помочь в решении?"
            ]
            for q in universal_questions:
                if len(questions) < 3:
                    questions.append(q)
        
        return questions[:3]

    def should_ask_comprehensive_questions(self, analysis_result: Dict) -> bool:
        """Определяет, нужно ли задавать уточняющие вопросы на основе комплексного анализа"""
        
        if analysis_result["status"] != "comprehensive_analysis_complete":
            return False
        
        # Факторы для принятия решения
        has_missing_info = len(analysis_result.get("missing_info", [])) > 0
        low_confidence = analysis_result.get("confidence", 0) < 0.7
        multiple_sources = analysis_result.get("sources", {}).get("total_sources", 0) > 1
        has_procedures = len(analysis_result.get("combined_context", "")) > 100
        
        # Анализ доминирующего источника
        dominant_source = analysis_result.get("sources", {})
        qdrant_strong = dominant_source.get("qdrant_sources", 0) >= 3
        qa_strong = dominant_source.get("qa_matches", 0) >= 2
        
        # Логика принятия решения
        should_ask = (
            has_missing_info or                           # Есть недостающая информация
            (low_confidence and has_procedures) or       # Низкая уверенность при наличии процедур
            (multiple_sources and not (qdrant_strong or qa_strong))  # Множественные слабые источники
        )
        
        log.info(f"🤔 Анализ необходимости вопросов: missing_info={has_missing_info}, "
                f"confidence={analysis_result.get('confidence', 0):.2f}, "
                f"should_ask={should_ask}")
        
        return should_ask

    def enhance_query_with_comprehensive_answers(self, original_query: str, user_answers: str) -> str:
        """Объединяет исходный запрос с ответами пользователя для финального решения"""
        
        enhanced = f"""Исходный запрос пользователя: {original_query}

Дополнительная информация от пользователя: {user_answers}

ОБЪЕДИНЕННЫЙ ЗАПРОС: {original_query}

КОНТЕКСТ ДИАЛОГА:
Пользователь предоставил дополнительные детали: {user_answers}

Используй всю эту информацию для формирования точного и полного ответа."""
        
        return enhanced

# Расширенная интеграция с основной системой RAG
class EnhancedIntelligentRAGSystem:
    """Интеграция улучшенного умного ассистента с существующей RAG системой"""
    
    def __init__(self, original_generate_func, model, tokenizer):
        self.original_generate = original_generate_func
        self.assistant = EnhancedIntelligentProactiveAssistant(model, tokenizer)
        self.active_sessions = {}  # Хранение активных сессий диалога
    
    def enhanced_generate(self, query: str, session_id: str = "default") -> Dict:
        """Улучшенная генерация с комплексным анализом всех источников"""
        
        # Проверяем, есть ли активная сессия
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Это ответ на уточняющие вопросы
            if session.get("awaiting_response"):
                enhanced_query = self.assistant.enhance_query_with_comprehensive_answers(
                    session["original_query"], query
                )
                
                # Генерируем финальный ответ
                final_response = self.original_generate(enhanced_query)
                
                # Очищаем сессию
                del self.active_sessions[session_id]
                
                return {
                    "type": "final_answer",
                    "content": final_response,
                    "session_complete": True,
                    "analysis_used": session.get("analysis", {})
                }
        
        # Новый запрос - комплексный анализ
        analysis = self.assistant.analyze_comprehensive_knowledge(query)
        
        if self.assistant.should_ask_comprehensive_questions(analysis):
            # Нужны уточняющие вопросы
            questions = analysis.get("questions", [])
            
            if questions:
                # Создаем новую сессию
                self.active_sessions[session_id] = {
                    "original_query": query,
                    "analysis": analysis,
                    "awaiting_response": True,
                    "questions": questions
                }
                
                # Форматируем вопросы для пользователя
                formatted_questions = self._format_comprehensive_questions(questions, analysis)
                
                return {
                    "type": "clarifying_questions",
                    "content": formatted_questions,
                    "session_active": True,
                    "questions": questions,
                    "analysis": analysis
                }
        
        # Не нужны уточнения - отвечаем сразу
        response = self.original_generate(query)
        return {
            "type": "direct_answer",
            "content": response,
            "session_complete": True,
            "analysis": analysis
        }
    
    def _format_comprehensive_questions(self, questions: List[str], analysis: Dict) -> str:
        """Форматирует вопросы с информацией о комплексном анализе"""
        
        confidence = analysis.get("confidence", 0)
        sources = analysis.get("sources", {})
        total_sources = sources.get("total_sources", 0)
        qdrant_sources = sources.get("qdrant_sources", 0)
        qa_matches = sources.get("qa_matches", 0)
        
        # Определяем иконки для источников
        sources_info = []
        if qdrant_sources > 0:
            sources_info.append(f"📚 {qdrant_sources} документов")
        if qa_matches > 0:
            sources_info.append(f"💡 {qa_matches} Q&A совпадений")
        
        sources_text = " + ".join(sources_info) if sources_info else "общая база"
        
        intro = f"""🔍 **Анализирую вашу ситуацию по всем источникам...**

Найдено релевантной информации: **{sources_text}** (уверенность: {confidence:.0%})

🎯 **Для точного решения уточните несколько деталей:**"""
        
        formatted_questions = intro + "\n\n"
        
        for i, question in enumerate(questions, 1):
            formatted_questions += f"**{i}.** {question}\n\n"
        
        # Добавляем информацию об источниках
        if qdrant_sources > 0 and qa_matches > 0:
            source_note = "💬 *Ответ будет основан на официальных документах и накопленном опыте решения аналогичных вопросов*"
        elif qdrant_sources > 0:
            source_note = "📋 *Ответ будет основан на официальных процедурах и регламентах*"
        elif qa_matches > 0:
            source_note = "🎯 *Ответ будет основан на опыте решения аналогичных ситуаций*"
        else:
            source_note = ""
        
        if source_note:
            formatted_questions += f"\n{source_note}\n\n"
        
        formatted_questions += "💬 **Ответьте на вопросы выше, и я дам максимально точное решение!**"
        
        return formatted_questions

# Функции для интеграции с существующей системой
def create_enhanced_intelligent_rag_system(rag_module):
    """Создает улучшенную умную RAG систему на основе существующей"""
    
    try:
        # Импортируем модель и токенизатор из rag_chat
        from rag_chat import model, tokenizer
        
        # Создаем улучшенную умную систему
        intelligent_system = EnhancedIntelligentRAGSystem(
            rag_module.generate, model, tokenizer
        )
        
        return intelligent_system
        
    except Exception as e:
        log.error(f"❌ Ошибка создания улучшенной умной системы: {e}")
        return None

# Глобальная переменная для умной системы
intelligent_rag = None

def init_intelligent_system(rag_module):
    """Инициализация улучшенной умной системы"""
    global intelligent_rag
    
    if intelligent_rag is None:
        intelligent_rag = create_enhanced_intelligent_rag_system(rag_module)
        if intelligent_rag:
            log.info("✅ Улучшенная умная RAG система инициализирована")
            log.info(f"📊 Загружено Q&A записей: {len(intelligent_rag.assistant.qa_dataset)}")
            return True
        else:
            log.error("❌ Ошибка инициализации улучшенной умной системы")
            return False
    return True

def intelligent_generate(query: str, session_id: str = "default") -> Dict:
    """Основная функция для улучшенной умной генерации"""
    global intelligent_rag
    
    if intelligent_rag is None:
        # Fallback к обычной генерации
        import rag_chat
        return {
            "type": "direct_answer", 
            "content": rag_chat.generate(query),
            "session_complete": True
        }
    
    return intelligent_rag.enhanced_generate(query, session_id)