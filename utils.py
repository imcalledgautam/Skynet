from dotenv import load_dotenv, find_dotenv

def load_env_vars():
    _ = load_dotenv(find_dotenv())
    OPENAI_API_KEY = getpass()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY