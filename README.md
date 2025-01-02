# README.md

**Setting up the Environment**

1. **Create a Virtual Environment:**
   - Use the `venv` module to create an isolated environment:

     ```bash
     python3 -m venv .venv 
     ```

2. **Activate the Virtual Environment:**
   - **On Linux/macOS:**

     ```bash
     source .venv/bin/activate
     ```

   - **On Windows:**

     ```bash
     .\.venv\Scripts\activate
     ```

3. **Install Dependencies:**
   - Install the required packages from `requirements.txt`:

     ```bash
     pip install -r requirements.txt
     ```

**Running the Application**

- Once the dependencies are installed, you can run your application using the following command:

   ```bash
   python your_app_file.py 
