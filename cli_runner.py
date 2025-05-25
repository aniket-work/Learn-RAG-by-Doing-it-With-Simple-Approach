import colorama
from colorama import Fore, Style
from config import load_config
from core_rag import CoreRAG
from doc_processing import load_sample_documents

def main():
    colorama.init(autoreset=True)
    config = load_config()
    rag = CoreRAG(config)
    sample_docs = [
        {
            'text': "A Next.js and Tailwind CSS blogging starter template, forked from timlrx/tailwind-nextjs-starter-blog. Easily configurable and customizable for technical writing.",
            'metadata': {'topic': 'web_development', 'framework': 'Next.js', 'css': 'Tailwind CSS'},
            'id': 'tailwind-nextjs-blog-template'
        },
        {
            'text': "Personal portfolio website built with Gatsby, showcasing projects and skills.",
            'metadata': {'topic': 'portfolio', 'framework': 'Gatsby', 'language': 'JavaScript'},
            'id': 'personal-portfolio'
        },
        {
            'text': "A video conferencing application demonstrating real-time communication features.",
            'metadata': {'topic': 'real_time_communication', 'language': 'JavaScript'},
            'id': 'video-conferencing-app'
        },
        {
            'text': "Demo Angular project utilizing NgRx for state management.",
            'metadata': {'topic': 'web_development', 'framework': 'Angular', 'state_management': 'NgRx'},
            'id': 'angular-with-ngrx'
        },
        {
            'text': "Personal portfolio website built using React.js and TypeScript.",
            'metadata': {'topic': 'portfolio', 'framework': 'React', 'language': 'TypeScript'},
            'id': 'personal-portfolio-react'
        },
        {
            'text': "React web application for typing practice, enhancing typing speed and accuracy.",
            'metadata': {'topic': 'education', 'framework': 'React', 'language': 'JavaScript'},
            'id': 'typingpracticecenter.com'
        },
        {
            'text': "A simple to-do list application built with React and Redux for state management.",
            'metadata': {'topic': 'productivity', 'framework': 'React', 'state_management': 'Redux'},
            'id': 'todo-list-app'
        },
        {
            'text': "Weather forecasting application using OpenWeatherMap API, built with React.",
            'metadata': {'topic': 'weather', 'framework': 'React', 'api': 'OpenWeatherMap'},
            'id': 'weather-app'
        },
        {
            'text': "Chat application demonstrating real-time messaging using Socket.io and Node.js.",
            'metadata': {'topic': 'real_time_communication', 'framework': 'Node.js', 'library': 'Socket.io'},
            'id': 'chat-app'
        },
        {
            'text': "E-commerce website built with MERN stack, featuring product listings and shopping cart functionality.",
            'metadata': {'topic': 'e_commerce', 'stack': 'MERN', 'language': 'JavaScript'},
            'id': 'ecommerce-website'
        }
    ]

    info = rag.get_collection_info()
    if info['document_count'] != 0:
        print(f"{Fore.YELLOW}Clearing existing collection...{Style.RESET_ALL}")
        rag.vector_store.clear()
        info = rag.get_collection_info()
    print(f"{Fore.GREEN}Adding sample documents...{Style.RESET_ALL}")
    rag.add_documents(sample_docs)
    print(f"\n{Fore.CYAN}=== RAG System Ready ==={Style.RESET_ALL}")
    print(f"{Fore.CYAN}Ask questions about the documents (type 'quit' to exit){Style.RESET_ALL}")
    while True:
        question = input(f"\n{Fore.MAGENTA}Your question: {Style.RESET_ALL}").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if question:
            try:
                result = rag.query(question)
                print(f"\n{Fore.GREEN}Answer: {result['answer']}{Style.RESET_ALL}")
                print(f"\n{Fore.YELLOW}Sources used: {len(result['sources'])}{Style.RESET_ALL}")
                for i, source in enumerate(result['sources']):
                    print(f"  {Fore.BLUE}{i+1}. {source['id']} (distance: {source['distance']:.3f}){Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()

