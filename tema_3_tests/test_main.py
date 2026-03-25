import requests
import httpx
import sys
import pytest

# foloseste UTF-8 pentru stdout ca sa evite erori de codare
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"

#ToDo: Adăugați un test pentru endpoint-ul root 

#ToDo: Adăugați un scenariu de testare pentru endpoint-ul /chat/ care să fie evaluat de LLM as a Judge

#ToDo: Adăugațu un test negativ pentru endpoint-ul /chat/ care să fie evaluat de LLM as a Judge 