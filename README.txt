# AI Item-Entwicklungs-Tool

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up secrets:
   - Copy `.streamlit/secrets.toml.template` to `.streamlit/secrets.toml`
   - Add your Anthropic API key to the new `secrets.toml` file
   ```toml
   ANTHROPIC_API_KEY = "your-actual-api-key"
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Environment Variables
This application requires the following secret:
- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude

## Deployment
For deployment (e.g., on Streamlit Cloud):
1. Add the secret in the deployment platform's settings
2. For Streamlit Cloud: Add it under "Advanced Settings" -> "Secrets"