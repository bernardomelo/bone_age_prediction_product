
## ⚙️ Configurando o Ambiente

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```
   
2. **Crie e ative um ambiente virtual:**
     ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Crie a imagem Docker e rode:**
     ```bash
   docker build -t bone-age-api .
   docker run -d -p 8001:8001 -v ${PWD}/image-requests:/app/image-requests --name bone-age-container bone-age-api
   ```

4. **(Alternativamente) Rode local:**
     ```bash
   pip install -r requirements.txt
   cd src/api
   python main.py
   ```