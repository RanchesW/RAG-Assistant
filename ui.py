#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Красивый продакшен интерфейс для ИИ Ассистента
"""

# САМЫЙ ПЕРВЫЙ импорт и команда Streamlit!
import streamlit as st

st.set_page_config(
    page_title="ИИ Ассистент", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Только ПОСЛЕ set_page_config импортируем остальное
import pathlib, tempfile, logging, json, hashlib, traceback
from datetime import datetime
import pandas as pd

import ingest
from ingest import (
    build_vector_store,
    add_documents_to_existing_collection,
    client as qdrant,
    COLLECTION,
    SUPPORT,
    READERS,
    read_pdf,
    read_docx,
    read_text,
    read_image,
)
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue

# Настраиваем логгирование
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Данные для авторизации разработчика
DEV_CREDENTIALS = {
    "RanchesW": "308111"
}

def init_session_state():
    """Инициализация состояний для подтверждения удаления."""
    if 'confirm_delete_all' not in st.session_state:
        st.session_state.confirm_delete_all = False
    if 'files_to_delete' not in st.session_state:
        st.session_state.files_to_delete = set()
    if 'ai_initialized' not in st.session_state:
        st.session_state.ai_initialized = False

def reset_delete_states():
    """Сброс всех состояний удаления."""
    st.session_state.confirm_delete_all = False
    st.session_state.files_to_delete = set()

def lazy_import_rag() -> bool:
    """Однократный импорт RAG-функций и сохранение их в session_state."""
    if 'generate' in st.session_state:
        return True                # уже загружено

    try:
        import torch, types
        if not hasattr(torch.classes, "__path__"):
            torch.classes.__path__ = []    # фикс streamlit-watcher

        from rag_chat import (
            generate   as _gen,
            search_ctx as _sctx,
            search_in_specific_file as _sfile,
        )

        # 🟢 Сохраняем ссылки, чтобы переживали перезапуск скрипта
        st.session_state.generate = _gen
        st.session_state.search_ctx = _sctx
        st.session_state.search_in_specific_file = _sfile
        return True
    except Exception as e:
        st.error(f"Ошибка загрузки ИИ-модуля: {e}")
        return False


# CSS стили для темного минималистичного интерфейса в стиле ChatGPT
# Замените весь блок CSS стилей (строки примерно 45-350) на этот улучшенный код:

st.markdown("""
<style>
    /* Основная темная тема с принудительным применением */
    .main, .main .block-container, .stApp {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
        background-color: #1a1a1a !important;
    }
    
    /* Принудительно скрываем элементы Streamlit */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    
    /* Улучшенные стили для чата с принудительным применением */
    .stChatMessage, .stChatMessage > div {
        background-color: #2d2d2d !important;
        border: 2px solid #404040 !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    .stChatMessage[data-testid="user"], .stChatMessage[data-testid="user"] > div {
        background-color: #333333 !important;
        border-color: #555555 !important;
    }
    
    .stChatMessage[data-testid="assistant"], .stChatMessage[data-testid="assistant"] > div {
        background-color: #2a2a2a !important;
        border-color: #444444 !important;
    }
    
    /* Принудительные стили для заголовков */
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Улучшенные кнопки с высоким контрастом */
    .stButton > button, button[kind="primary"], button[kind="secondary"] {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 2px solid #555555 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        min-height: 40px !important;
    }
    
    .stButton > button:hover, button[kind="primary"]:hover, button[kind="secondary"]:hover {
        background-color: #404040 !important;
        border-color: #777777 !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Специальные стили для primary кнопок */
    button[kind="primary"], .stButton > button[type="submit"] {
        background-color: #10a37f !important;
        border-color: #10a37f !important;
        color: #ffffff !important;
    }
    
    button[kind="primary"]:hover, .stButton > button[type="submit"]:hover {
        background-color: #0d8c6b !important;
        border-color: #0d8c6b !important;
    }
    
    /* Улучшенные поля ввода */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select,
    input, textarea, select {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 2px solid #555555 !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        font-size: 14px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    input:focus, textarea:focus {
        border-color: #10a37f !important;
        box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2) !important;
        outline: none !important;
    }
    
    /* Улучшенный чат инпут как в ChatGPT */
    .stChatInputContainer, .stChatInput {
        background-color: transparent !important;
        padding: 1rem 0 !important;
    }
    
    .stChatInput > div, .stChatInput input {
        background-color: #2d2d2d !important;
        border: 2px solid #555555 !important;
        border-radius: 25px !important;
        padding: 1rem 1.5rem !important;
        color: #ffffff !important;
        font-size: 16px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stChatInput input::placeholder {
        color: #888888 !important;
        font-size: 16px !important;
    }
    
    /* Кнопка отправки сообщения */
    .stChatInput button {
        background-color: #10a37f !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin-left: 8px !important;
    }
    
    .stChatInput button:hover {
        background-color: #0d8c6b !important;
        transform: scale(1.05) !important;
    }
    
    /* Улучшенные табы с высоким контрастом */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a1a !important;
        gap: 8px !important;
        padding: 0.5rem !important;
        border-radius: 12px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d2d !important;
        color: #cccccc !important;
        border: 2px solid #404040 !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #10a37f !important;
        color: #ffffff !important;
        border-color: #10a37f !important;
        font-weight: 600 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #404040 !important;
        border-color: #555555 !important;
    }
    
    /* Компактные метрики с улучшенным контрастом */
    [data-testid="metric-container"] {
        background-color: #2d2d2d !important;
        border: 2px solid #404040 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="metric-container"]:hover {
        border-color: #555555 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #ffffff !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        color: #cccccc !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-delta"] {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }
    
    /* Улучшенные алерты */
    .stAlert, .stAlert > div {
        background-color: #2d2d2d !important;
        border: 2px solid #555555 !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        padding: 1rem !important;
    }
    
    .stAlert[data-baseweb="notification"] [data-testid="alertContent"] {
        color: #ffffff !important;
    }
    
    /* Улучшенные экспандеры */
    .streamlit-expanderHeader, .stExpanderHeader {
        background-color: #2d2d2d !important;
        border: 2px solid #404040 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        padding: 0.75rem 1rem !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderContent, .stExpanderContent {
        background-color: #1e1e1e !important;
        border: 2px solid #404040 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        color: #ffffff !important;
        padding: 1rem !important;
    }
    
    /* Улучшенные селектбоксы */
    .stSelectbox > div > div, .stSelectbox select {
        background-color: #2d2d2d !important;
        border: 2px solid #555555 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
    
    /* Улучшенный файловый загрузчик */
    .stFileUploader > div, .stFileUploader label {
        background-color: #2d2d2d !important;
        border: 3px dashed #555555 !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        padding: 2rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader > div:hover {
        border-color: #10a37f !important;
        background-color: #333333 !important;
    }
    
    /* Улучшенные формы */
    .stForm, form {
        background-color: #2d2d2d !important;
        border: 2px solid #404040 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
    }
    
    /* Принудительное скрытие спиннеров */
    input[type=number]::-webkit-outer-spin-button,
    input[type=number]::-webkit-inner-spin-button {
        -webkit-appearance: none !important;
        margin: 0 !important;
    }
    
    /* Центрирование контента */
    .main .block-container {
        max-width: 900px !important;
        margin: 0 auto !important;
        padding: 1rem !important;
    }
    
    /* Улучшенные стили для приветственного сообщения */
    .welcome-message {
        text-align: center !important;
        color: #cccccc !important;
        font-size: 1.2rem !important;
        margin: 4rem 0 !important;
        padding: 2rem !important;
        background-color: #2d2d2d !important;
        border-radius: 12px !important;
        border: 2px solid #404040 !important;
    }
    
    .welcome-message h3 {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        margin-bottom: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Улучшенные стили для заголовка */
    .main-title {
        text-align: center !important;
        margin-bottom: 3rem !important;
        padding: 1rem !important;
    }
    
    .main-title h1 {
        color: #ffffff !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5) !important;
    }
    
    .main-title p {
        color: #cccccc !important;
        font-size: 1.1rem !important;
        margin: 0 !important;
        font-weight: 400 !important;
    }
    
    /* Убираем лишние отступы */
    .main > div {
        padding-top: 1rem !important;
    }
    
    /* Улучшенные download кнопки */
    .stDownloadButton > button {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 2px solid #555555 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
    }
    
    .stDownloadButton > button:hover {
        background-color: #404040 !important;
        border-color: #777777 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Принудительные стили для всех текстовых элементов */
    p, span, div, label, .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Улучшенные чекбоксы */
    .stCheckbox > label {
        color: #ffffff !important;
    }
    
    /* Улучшенная прогресс-бар */
    .stProgress > div > div > div {
        background-color: #10a37f !important;
    }
    
    /* Скрываем логотип Streamlit */
    .css-1v0mbdj > img, [data-testid="stLogo"] {
        display: none !important;
    }
    
    /* Дополнительная защита от белого фона */
    body, html, .stApp, .main, .block-container, .element-container {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Улучшенные стили для sidebar (если используется) */
    .css-1d391kg, .stSidebar {
        background-color: #1e1e1e !important;
        border-right: 2px solid #404040 !important;
    }
    
    /* Стили для спиннера загрузки */
    .stSpinner {
        color: #10a37f !important;
    }
    
    /* Принудительные стили для таблиц */
    .stDataFrame, table {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #404040 !important;
    }
    
    .stDataFrame th, table th {
        background-color: #404040 !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
    }
    
    .stDataFrame td, table td {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #404040 !important;
    }
</style>
""", unsafe_allow_html=True)

class DeveloperDataset:
    """Простой класс для работы с датасетом разработчика."""
    
    def __init__(self, dataset_file: str = "developer_dataset.json"):
        self.dataset_file = pathlib.Path(dataset_file)
        self.dataset = self.load_dataset()
    
    def load_dataset(self):
        """Загрузка датасета."""
        if self.dataset_file.exists():
            try:
                with open(self.dataset_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_dataset(self):
        """Сохранение датасета."""
        with open(self.dataset_file, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)
    
    def add_qa_pair(self, question: str, answer: str) -> bool:
        """Добавление пары вопрос-ответ."""
        if not question.strip() or not answer.strip():
            return False
        
        qa_pair = {
            "id": hashlib.md5(question.encode('utf-8')).hexdigest()[:12],
            "question": question.strip(),
            "answer": answer.strip(),
            "created_at": datetime.now().isoformat(),
            "source": "developer_manual"
        }
        
        # Проверяем, нет ли уже такого вопроса
        existing_ids = {item.get("id") for item in self.dataset}
        if qa_pair["id"] not in existing_ids:
            self.dataset.append(qa_pair)
            self.save_dataset()
            return True
        return False
    
    def get_stats(self):
        """Получение статистики датасета."""
        return {
            "total_pairs": len(self.dataset),
            "last_updated": max([item.get("created_at", "") for item in self.dataset]) if self.dataset else "Нет данных"
        }

def check_authentication():
    """Проверка авторизации разработчика."""
    if 'dev_authenticated' not in st.session_state:
        st.session_state.dev_authenticated = False
    
    if not st.session_state.dev_authenticated:
        st.markdown("""
        <div class='auth-container'>
            <h2>🔐 Панель разработчика</h2>
            <p>Введите учетные данные для доступа к расширенным функциям</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("dev_login"):
                login = st.text_input("👤 Логин:", placeholder="Введите логин")
                password = st.text_input("🔒 Пароль:", type="password", placeholder="Введите пароль")
                
                submitted = st.form_submit_button("🚀 Войти", use_container_width=True)
                
                if submitted:
                    if login in DEV_CREDENTIALS and DEV_CREDENTIALS[login] == password:
                        st.session_state.dev_authenticated = True
                        st.success("✅ Добро пожаловать!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Неверные учетные данные")
        
        st.info("💡 Панель разработчика содержит инструменты для управления базой знаний и извлечения текста из документов")
        return False
    
    return True

# Инициализируем состояния сразу
init_session_state()

# Заголовок (простой, без блока)
st.markdown("""
<div class="main-title">
    <h1>🤖 ИИ Ассистент</h1>
    <p>Умный помощник на основе искусственного интеллекта</p>
</div>
""", unsafe_allow_html=True)

# Основные вкладки - только 2!
tab1, tab2 = st.tabs(["💬 Чат с ИИ", "👨‍💻 Панель разработчика"])

with tab1:
    # Чат с ИИ - главная функция (без белого контейнера)
    
    # Компактная информационная панель (в углу)
    with st.expander("ℹ️ Статус системы", expanded=False):
        col1, col2, col3, col4 = st.columns(4, gap="small")
        with col1:
            # Проверяем наличие коллекции
            try:
                collections = qdrant.get_collections().collections
                if COLLECTION in {c.name for c in collections}:
                    collection_info = qdrant.get_collection(collection_name=COLLECTION)
                    docs_count = collection_info.points_count
                    status = "🟢 Активна"
                else:
                    docs_count = 0
                    status = "🔴 Пуста"
            except:
                docs_count = 0
                status = "⚠️ Ошибка"
            
            st.metric("База знаний", status, f"{docs_count} док.")
        
        with col2:
            # Проверяем датасет разработчика
            try:
                if pathlib.Path("developer_dataset.json").exists():
                    with open("developer_dataset.json", 'r', encoding='utf-8') as f:
                        dataset = json.load(f)
                        qa_count = len(dataset)
                        dataset_status = "🟢 Активен"
                else:
                    qa_count = 0
                    dataset_status = "⚪ Пуст"
            except:
                qa_count = 0
                dataset_status = "⚠️ Ошибка"
            
            st.metric("Датасет Q&A", dataset_status, f"{qa_count} пар")
        
        with col3:
            # ИИ Модель
            ai_status = "🟢 Готова" if st.session_state.ai_initialized else "🔴 Не загружена"
            st.metric("ИИ Модель", ai_status, "DeepSeek 7B")
        
        with col4:
            # Статус системы
            overall_status = "🟢 Готова" if (docs_count > 0 or qa_count > 0) else "⚠️ Настройка"
            st.metric("Система", overall_status, "Онлайн")
    
    # Проверяем, есть ли данные для работы
    docs_available = False
    qa_available = False
    
    try:
        collections = qdrant.get_collections().collections
        if COLLECTION in {c.name for c in collections}:
            collection_info = qdrant.get_collection(collection_name=COLLECTION)
            docs_available = collection_info.points_count > 0
    except:
        pass
    
    try:
        if pathlib.Path("developer_dataset.json").exists():
            with open("developer_dataset.json", 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                qa_available = len(dataset) > 0
    except:
        pass
    
    # Показываем предупреждение только если совсем нет данных
    if not docs_available and not qa_available:
        st.warning("""
        **📝 Система пока не обучена!**
        
        Чтобы ИИ мог отвечать на ваши вопросы, администратор должен настроить систему.
        Обратитесь к администратору для загрузки данных.
        """)
    
    # Основной чат
    # Создаем или получаем историю чата
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Приветственное сообщение только если есть данные
        if docs_available or qa_available:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "👋 Привет! Я ваш ИИ ассистент. Задавайте любые вопросы, и я постараюсь помочь!"
            })
    
    # Отображаем историю чата
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        # Показываем приветствие если нет истории и есть данные
        if docs_available or qa_available:
            st.markdown("""
            <div class="welcome-message">
                <h3>Как я могу помочь?</h3>
                <p>Задавайте любые вопросы - я готов ответить!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="welcome-message">
                <h3>Система настраивается...</h3>
                <p>Пожалуйста, обратитесь к администратору для завершения настройки</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Ввод нового сообщения
    if prompt := st.chat_input("Напишите ваш вопрос..."):
        # Ленивый импорт ИИ модулей только при первом сообщении
        if not st.session_state.ai_initialized:
            with st.spinner("🔄 Загружаю ИИ систему..."):
                if lazy_import_rag():
                    st.session_state.ai_initialized = True
                else:
                    st.stop()
        
        # Добавляем сообщение пользователя в историю
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Отображаем сообщение пользователя
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Генерируем ответ
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("🤔 Обдумываю ответ..."):
                try:
                    response = st.session_state.generate(prompt)
                except Exception as e:
                    response = f"⚠️ Произошла ошибка при генерации ответа: {str(e)}"
            message_placeholder.markdown(response)
        
        # Добавляем ответ в историю
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Кнопка очистки истории (компактная)
    if len(st.session_state.messages) > 1:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Очистить историю", help="Удалить все сообщения"):
                st.session_state.messages = []
                st.rerun()

with tab2:
    # Панель разработчика
    if not check_authentication():
        st.stop()
    
    # Успешная авторизация
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("""
        <div class='welcome-container'>
            <h3>🎉 Добро пожаловать в панель разработчика!</h3>
            <p>Здесь вы можете управлять базой знаний ИИ</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🚪 Выйти", use_container_width=True):
            st.session_state.dev_authenticated = False
            st.rerun()
    
    # Вкладки панели разработчика
    dev_tab1, dev_tab2, dev_tab3, dev_tab4 = st.tabs([
        "📄 Извлечение текста", 
        "📚 Управление документами", 
        "🤖 Датасет Q&A",
        "🔧 Отладка системы"
    ])
    
    with dev_tab1:
        st.markdown("### 📄 Извлечение текста из документов")
        
        st.info("""
        **🎯 Назначение:** Извлечение чистого текста из любых документов для последующего использования
        
        **📋 Поддерживаемые форматы:**
        - 📄 PDF (включая сканированные с OCR)
        - 📝 DOCX (Microsoft Word)
        - 📰 TXT, MD (текстовые файлы)
        - 🖼️ PNG, JPG, JPEG, WEBP (изображения с текстом)
        """)
        
        uploaded_file = st.file_uploader(
            "🔍 Выберите файл для извлечения текста:",
            type=[ext.strip(".") for ext in SUPPORT],
            key="dev_file_upload",
            help="Поддерживаются PDF, DOCX, изображения и текстовые файлы"
        )
        
        if uploaded_file:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📁 Файл", uploaded_file.name)
            with col2:
                st.metric("📏 Размер", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                file_ext = pathlib.Path(uploaded_file.name).suffix.lower()
                st.metric("🏷️ Тип", file_ext)
            
            if st.button("🚀 Извлечь текст", type="primary", use_container_width=True):
                with st.spinner("⚙️ Обрабатываю документ..."):
                    try:
                        reader_func = READERS.get(file_ext)
                        if not reader_func:
                            st.error(f"❌ Формат {file_ext} не поддерживается")
                            st.stop()
                        
                        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                            tmp_path = pathlib.Path(tmp.name)
                            tmp_path.write_bytes(uploaded_file.getvalue())
                        
                        extracted_text = reader_func(tmp_path)
                        tmp_path.unlink(missing_ok=True)
                        
                        if extracted_text and not extracted_text.startswith("Ошибка"):
                            char_count = len(extracted_text)
                            word_count = len(extracted_text.split())
                            line_count = len(extracted_text.splitlines())
                            
                            st.success("✅ Текст успешно извлечен!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📝 Символов", f"{char_count:,}")
                            with col2:
                                st.metric("🔤 Слов", f"{word_count:,}")
                            with col3:
                                st.metric("📄 Строк", f"{line_count:,}")
                            
                            with st.expander("👀 Предварительный просмотр (первые 1000 символов)"):
                                preview_text = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
                                st.text_area("", preview_text, height=200, disabled=True)
                            
                            filename = f"extracted_text_{uploaded_file.name.split('.')[0]}.txt"
                            st.download_button(
                                label="💾 Скачать извлеченный текст",
                                data=extracted_text,
                                file_name=filename,
                                mime="text/plain",
                                use_container_width=True
                            )
                        else:
                            st.error(f"❌ Ошибка извлечения: {extracted_text}")
                    except Exception as e:
                        st.error(f"💥 Произошла ошибка: {str(e)}")
    
    with dev_tab2:
        st.markdown("### 📚 Управление базой документов")
        
        # Информация о текущей коллекции
        try:
            collections = qdrant.get_collections().collections
            if COLLECTION in {c.name for c in collections}:
                collection_info = qdrant.get_collection(collection_name=COLLECTION)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Статус базы", "🟢 Активна")
                with col2:
                    st.metric("📄 Фрагментов", f"{collection_info.points_count:,}")
                with col3:
                    st.metric("🧠 Размерность", f"{collection_info.config.params.vectors.size}D")
                
                # Получаем информацию о файлах в базе
                try:
                    sample_points = qdrant.scroll(
                        collection_name=COLLECTION,
                        limit=1000,
                        with_payload=True
                    )
                    
                    # Подсчитываем уникальные файлы
                    files_in_db = {}
                    for point in sample_points[0]:
                        filename = point.payload.get("file", "Неизвестно")
                        if filename in files_in_db:
                            files_in_db[filename] += 1
                        else:
                            files_in_db[filename] = 1
                    
                    if files_in_db:
                        st.success(f"✅ База содержит {len(files_in_db)} файлов")
                        
                        with st.expander("📋 Файлы в базе данных", expanded=False):
                            for filename, chunks in files_in_db.items():
                                col1, col2, col3 = st.columns([3, 1, 1])
                                with col1:
                                    st.write(f"📄 {filename}")
                                with col2:
                                    st.write(f"{chunks} фрагментов")
                                with col3:
                                    file_key = f"file_{filename.replace('.', '_').replace(' ', '_').replace('/', '_')}"
                                    
                                    if filename not in st.session_state.files_to_delete:
                                        if st.button(f"🗑️", key=f"delete_{file_key}", help=f"Удалить {filename}"):
                                            st.session_state.files_to_delete.add(filename)
                                            st.rerun()
                                    else:
                                        st.write("⚠️ Удалить?")
                                        col_yes, col_no = st.columns(2)
                                        
                                        with col_yes:
                                            if st.button("✅", key=f"confirm_{file_key}", help="Подтвердить"):
                                                try:
                                                    all_points = qdrant.scroll(
                                                        collection_name=COLLECTION,
                                                        with_payload=True,
                                                        limit=10000
                                                    )
                                                    
                                                    points_to_delete = []
                                                    for point in all_points[0]:
                                                        if point.payload.get("file") == filename:
                                                            points_to_delete.append(point.id)
                                                    
                                                    if points_to_delete:
                                                        qdrant.delete(
                                                            collection_name=COLLECTION,
                                                            points_selector=points_to_delete
                                                        )
                                                        st.success(f"✅ Файл {filename} удален ({len(points_to_delete)} фрагментов)")
                                                        st.session_state.files_to_delete.discard(filename)
                                                        st.rerun()
                                                    else:
                                                        st.warning(f"⚠️ Файл {filename} не найден")
                                                        st.session_state.files_to_delete.discard(filename)
                                                except Exception as e:
                                                    st.error(f"❌ Ошибка удаления: {str(e)}")
                                                    st.session_state.files_to_delete.discard(filename)
                                        
                                        with col_no:
                                            if st.button("❌", key=f"cancel_{file_key}", help="Отмена"):
                                                st.session_state.files_to_delete.discard(filename)
                                                st.rerun()
                    else:
                        st.info("ℹ️ Не удалось получить список файлов")
                        
                except Exception as e:
                    st.warning(f"⚠️ Не удалось получить информацию о файлах: {str(e)}")
            else:
                st.warning("⚠️ База знаний пуста")
                st.metric("📊 Статус базы", "🔴 Пуста")
        except Exception as e:
            st.error(f"❌ Ошибка подключения к базе: {str(e)}")
        
        st.divider()
        
        # Загрузка новых документов
        st.markdown("#### ➕ Добавить новые документы")
        st.info("💡 Новые файлы будут добавлены к существующим в базе знаний")
        
        uploaded = st.file_uploader(
            "Перетащите файлы для добавления в базу знаний:",
            accept_multiple_files=True,
            type=[ext.strip(".") for ext in SUPPORT],
            key="add_files"
        )
        
        if uploaded:
            st.write(f"📋 Выбрано файлов: **{len(uploaded)}**")
            
            for file in uploaded:
                st.write(f"• {file.name} ({file.size / 1024:.1f} KB)")
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("➕ Добавить в базу", type="primary", use_container_width=True):
                    with st.spinner("🔄 Обрабатываю новые документы..."):
                        collections = qdrant.get_collections().collections
                        if COLLECTION not in {c.name for c in collections}:
                            st.info("🆕 Создаю новую базу знаний...")
                            qdrant.create_collection(
                                COLLECTION,
                                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
                            )
                        
                        tmpdir = pathlib.Path(tempfile.mkdtemp())
                        paths = []
                        
                        for f in uploaded:
                            path = tmpdir / f.name
                            path.write_bytes(f.read())
                            paths.append(path)
                        
                        st.info(f"📚 Добавляю {len(paths)} файлов...")
                        
                        try:
                            add_documents_to_existing_collection(paths)
                            collection_info = qdrant.get_collection(collection_name=COLLECTION)
                            st.success(f"🎉 Файлы добавлены! Всего фрагментов в базе: {collection_info.points_count}")
                            st.balloons()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Ошибка добавления файлов: {str(e)}")
        
        st.divider()
        st.markdown("#### 🔧 Управление базой")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Тестировать поиск", use_container_width=True, key="test_search_btn"):
                st.session_state['show_test_search'] = True
            
            if st.session_state.get('show_test_search', False):
                test_query = st.text_input("Тестовый запрос:", placeholder="Введите вопрос для тестирования", key="test_search_query")
                if test_query:
                    if not st.session_state.ai_initialized:
                        if lazy_import_rag():
                            st.session_state.ai_initialized = True
                    
                    with st.spinner("🔍 Ищу..."):
                        try:
                            context = st.session_state.search_ctx(test_query, k=3)
                            st.text_area("Результат поиска:", context, height=200, key="test_search_result")
                        except Exception as e:
                            st.error(f"Ошибка поиска: {str(e)}")
        
        with col2:
            if not st.session_state.confirm_delete_all:
                if st.button("🗑️ Очистить всю базу", use_container_width=True, type="secondary"):
                    st.session_state.confirm_delete_all = True
                    st.rerun()
            else:
                st.warning("⚠️ Вы уверены, что хотите удалить ВСЮ базу данных?")
                col_confirm, col_cancel = st.columns(2)
                
                with col_confirm:
                    if st.button("✅ Да, удалить", type="primary", use_container_width=True, key="confirm_delete_all_btn"):
                        try:
                            collections = qdrant.get_collections().collections
                            if COLLECTION in {c.name for c in collections}:
                                qdrant.delete_collection(collection_name=COLLECTION)
                                st.success("✅ База полностью очищена")
                                reset_delete_states()
                                st.rerun()
                            else:
                                st.info("ℹ️ База уже пуста")
                                reset_delete_states()
                        except Exception as e:
                            st.error(f"❌ Ошибка: {str(e)}")
                            reset_delete_states()
                
                with col_cancel:
                    if st.button("❌ Отмена", use_container_width=True, key="cancel_delete_all_btn"):
                        reset_delete_states()
                        st.rerun()
        
        with col3:
            if st.button("📊 Поиск по файлу", use_container_width=True, key="file_search_btn"):
                st.session_state['show_file_search'] = True
            
            if st.session_state.get('show_file_search', False):
                try:
                    sample_points = qdrant.scroll(collection_name=COLLECTION, limit=100, with_payload=True)
                    files_list = list(set(point.payload.get("file", "Неизвестно") for point in sample_points[0]))
                    
                    if files_list:
                        selected_file = st.selectbox("Выберите файл:", files_list, key="file_search_select")
                        search_query = st.text_input("Поиск в файле:", placeholder="Введите запрос для поиска в конкретном файле", key="file_search_query")
                        
                        if search_query and selected_file and st.button("🔍 Искать в файле", key="search_in_file_btn"):
                            if not st.session_state.ai_initialized:
                                if lazy_import_rag():
                                    st.session_state.ai_initialized = True
                            
                            with st.spinner("🔍 Ищу в файле..."):
                                try:
                                    context = st.session_state.search_in_specific_file(search_query, selected_file)
                                    st.text_area(f"Результат поиска в {selected_file}:", context, height=200, key="file_search_result")
                                except Exception as e:
                                    st.error(f"Ошибка поиска в файле: {str(e)}")
                    else:
                        st.info("📁 Файлы в базе не найдены")
                        
                except Exception as e:
                    st.error(f"❌ Ошибка получения списка файлов: {str(e)}")
    
    with dev_tab3:
        st.markdown("### 🤖 Управление датасетом Q&A")
        
        if 'dev_dataset' not in st.session_state:
            st.session_state.dev_dataset = DeveloperDataset()
        
        dataset = st.session_state.dev_dataset
        stats = dataset.get_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💬 Всего пар Q&A", stats["total_pairs"])
        with col2:
            st.metric("📅 Последнее обновление", 
                     stats["last_updated"][:10] if stats["last_updated"] != "Нет данных" else "Нет данных")
        with col3:
            avg_length = sum(len(item["answer"]) for item in dataset.dataset) // max(1, len(dataset.dataset))
            st.metric("📏 Средняя длина ответа", f"{avg_length} символов")
        
        st.divider()
        
        st.markdown("#### ➕ Добавить новую пару вопрос-ответ")
        
        with st.form("add_qa_pair", clear_on_submit=True):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                question = st.text_area(
                    "❓ Вопрос пользователя:",
                    height=120,
                    placeholder="Например: Как установить Python на Windows?",
                    help="Введите вопрос, который может задать пользователь"
                )
            
            with col2:
                answer = st.text_area(
                    "💡 Ответ ИИ:",
                    height=120,
                    placeholder="Например: Для установки Python на Windows перейдите на python.org...",
                    help="Введите правильный ответ, который должен дать ИИ"
                )
            
            submitted = st.form_submit_button("💾 Добавить в датасет", type="primary", use_container_width=True)
            
            if submitted:
                if question.strip() and answer.strip():
                    if dataset.add_qa_pair(question, answer):
                        st.success("✅ Пара добавлена в датасет!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.warning("⚠️ Такой вопрос уже существует")
                else:
                    st.error("❌ Заполните оба поля")
        
        if dataset.dataset:
            st.divider()
            st.markdown("#### 📋 Управление существующими парами")
            
            search_query = st.text_input("🔍 Поиск по вопросам:", placeholder="Введите ключевые слова...")
            
            filtered_dataset = dataset.dataset
            if search_query:
                filtered_dataset = [
                    item for item in dataset.dataset 
                    if search_query.lower() in item["question"].lower()
                ]
            
            st.write(f"📊 Показано: **{len(filtered_dataset)}** из **{len(dataset.dataset)}** пар")
            
            for i, item in enumerate(reversed(filtered_dataset[-10:])):
                with st.expander(f"❓ {item['question'][:80]}...", expanded=False):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**Вопрос:** {item['question']}")
                        st.markdown(f"**Ответ:** {item['answer']}")
                        st.caption(f"📅 Создано: {item.get('created_at', 'N/A')[:16]}")
                    
                    with col2:
                        if st.button("🗑️ Удалить", key=f"delete_qa_{item['id']}", help="Удалить эту пару"):
                            dataset.dataset = [d for d in dataset.dataset if d["id"] != item["id"]]
                            dataset.save_dataset()
                            st.rerun()
            
            st.divider()
            st.markdown("#### 💾 Экспорт датасета")
            
            col1, col2 = st.columns(2)
            
            with col1:
                json_data = json.dumps(dataset.dataset, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📄 Скачать JSON",
                    data=json_data,
                    file_name=f"qa_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                txt_data = ""
                for item in dataset.dataset:
                    txt_data += f"ВОПРОС: {item['question']}\n"
                    txt_data += f"ОТВЕТ: {item['answer']}\n"
                    txt_data += f"ДАТА: {item.get('created_at', 'N/A')}\n"
                    txt_data += "-" * 50 + "\n\n"
                
                st.download_button(
                    label="📝 Скачать TXT",
                    data=txt_data,
                    file_name=f"qa_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("📝 Датасет пуст. Добавьте первую пару вопрос-ответ!")
    
    with dev_tab4:
        st.markdown("### 🔧 Отладка и диагностика системы")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Состояние компонентов")
            
            try:
                if st.session_state.ai_initialized:
                    st.success("✅ ИИ модуль загружен")
                else:
                    st.error("❌ ИИ модуль не загружен")
            except:
                st.error("❌ Ошибка проверки ИИ модуля")
            
            try:
                collections = qdrant.get_collections()
                st.success("✅ Qdrant подключен")
                st.info(f"🗄️ Коллекций: {len(collections.collections)}")
            except Exception as e:
                st.error(f"❌ Ошибка Qdrant: {str(e)}")
            
            try:
                from ingest import embedder
                test_embedding = list(embedder.embed(["тест"]))[0]
                st.success("✅ Эмбеддинг модель работает")
                st.info(f"🔢 Размерность: {len(test_embedding)}")
            except Exception as e:
                st.error(f"❌ Ошибка эмбеддингов: {str(e)}")
        
        with col2:
            st.markdown("#### 🧪 Тестирование системы")
            
            test_query = st.text_input("🔍 Тестовый запрос:", placeholder="Введите вопрос для тестирования", key="debug_test_query")
            
            if test_query and st.button("🚀 Запустить полный тест", use_container_width=True, key="debug_test_btn"):
                if not st.session_state.ai_initialized:
                    if lazy_import_rag():
                        st.session_state.ai_initialized = True
                
                with st.spinner("🔬 Тестирую систему..."):
                    
                    st.markdown("**📚 Тест RAG системы:**")
                    try:
                        context = st.session_state.search_ctx(test_query, k=3)
                        if context.startswith(("Произошла ошибка", "Не найдено", "База знаний")):
                            st.warning("⚠️ RAG не нашел релевантных документов")
                        else:
                            st.success("✅ RAG нашел релевантный контекст")
                            st.text_area("📄 Контекст:", context[:300] + "...", height=100, key="debug_context_result")
                    except Exception as e:
                        st.error(f"❌ Ошибка RAG: {str(e)}")
                    
                    st.markdown("**🎯 Тест полной генерации:**")
                    try:
                        full_response = st.session_state.generate(test_query)
                        st.success("✅ Ответ сгенерирован успешно")
                        st.text_area("🤖 Ответ ИИ:", full_response, height=150, key="debug_full_result")
                    except Exception as e:
                        st.error(f"❌ Ошибка генерации: {str(e)}")
        
        st.divider()
        st.markdown("#### 📋 Системные логи")
        
        if st.button("🔄 Обновить системную информацию", key="debug_system_info"):
            try:
                import torch
                import psutil
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("💾 Использование RAM", f"{psutil.virtual_memory().percent:.1f}%")
                    st.metric("🔥 CUDA доступна", "✅" if torch.cuda.is_available() else "❌")
                
                with col2:
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        st.metric("🎮 GPU память", f"{gpu_memory:.1f} GB")
                        st.metric("🚀 GPU использование", f"{torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
                    else:
                        st.metric("🎮 GPU", "Недоступна")
                        st.metric("💻 CPU режим", "✅")
                
                with col3:
                    import sys
                    st.metric("🐍 Python", f"{sys.version.split()[0]}")
                    st.metric("⏰ Статус ИИ", "✅ Готово" if st.session_state.ai_initialized else "❌ Ошибка")
            except ImportError:
                st.error("Ошибка импорта модулей")
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

st.markdown("""
<script>
// Принудительное применение темной темы
document.addEventListener('DOMContentLoaded', function() {
    document.body.style.backgroundColor = '#1a1a1a';
    document.body.style.color = '#ffffff';
    
    // Дополнительная проверка через 500мс
    setTimeout(function() {
        document.body.style.backgroundColor = '#1a1a1a';
        document.body.style.color = '#ffffff';
        
        // Принудительно применяем к main элементу
        const mainElements = document.querySelectorAll('.main, .stApp, .block-container');
        mainElements.forEach(el => {
            el.style.backgroundColor = '#1a1a1a';
            el.style.color = '#ffffff';
        });
    }, 500);
});
</script>
""", unsafe_allow_html=True)