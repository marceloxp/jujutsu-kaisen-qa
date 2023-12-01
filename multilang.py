translations = {
    'en': {
        'hi': 'Hi, ask me any questions about Jujutsu Kaisen and I\'ll be happy to answer them.',
        'description': 'This is a Streamlit app with language support.',
        # Adicione mais chaves e valores conforme necessário
    },
    'pt': {
        'hi': 'Olá, faça qualquer pergunta sobre Jujutsu Kaisen e terei prazer em respondê-la.',
        'description': 'Este é um aplicativo Streamlit com suporte a vários idiomas.',
        # Adicione mais chaves e valores conforme necessário
    }
}

def get_translation(lang, key):
    return translations[lang][key]