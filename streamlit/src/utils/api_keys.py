import os
import streamlit as st
from dotenv import load_dotenv

# --- API Key Validation ---
def load_api_keys():
    """Load API keys from environment variables"""
    load_dotenv()
    return {
        'openai': os.getenv("OPENAI_API_KEY"),
        'groq': os.getenv("GROQ_API_KEY")
    }

def validate_api_keys_precheck():
    """Initial non-Streamlit validation"""
    keys = load_api_keys()
    missing = [name for name, key in keys.items() if not key]
    
    # Critical failure if both keys missing
    if len(missing) == 2:
        print("ERROR: Missing both API keys. Add them to .env and restart.")
        sys.exit(1)
        
    return keys

def validate_api_keys(required_for=None):
    """
    Validate API keys and show Streamlit errors
    Args:
        required_for: None (check all) or specific provider ('openai'/'groq')
    """
    keys = load_api_keys()
    messages = []
    stop_execution = False

    # Specific provider check
    if required_for:
        if not keys.get(required_for):
            messages.append(f"🔐 **{required_for.upper()} API Key Required**")
            stop_execution = True
    # Full validation
    else:
        missing = [name for name, key in keys.items() if not key]
        if missing:
            if len(missing) == 2:
                messages.append("🔐 **Both API Keys Required**")
                stop_execution = True
            else:
                messages.append(f"⚠️ **Missing Keys:** {', '.join(missing)}")
    
    if messages:
        st.error("\n\n".join(messages))
        if stop_execution:
            st.markdown("### ℹ️ How to fix:")
            st.markdown("""
            ### **Step-by-Step Terminal Commands:**
            1. Navigate to your project root:
                        
            `cd /path/to/project-root` 

            2. Create .env file:
                        
            `touch .env`

            3. Add API keys (replace values with your actual keys):
                        
            `echo "OPENAI_API_KEY=sk-your-openai-key-here" >> .env`
            `echo "GROQ_API_KEY=gsk-your-groq-key-here" >> .env`

            4. Verify file contents:
                        
            `cat .env  # Should show both keys`

            5. Reload Docker containers:
                        
            `docker-compose --file docker/docker-compose.yml down`
            `docker-compose --file docker/docker-compose.yml up --build`

            6. Verify keys in container (optional):
                        
            `docker exec -it tractian-document-api env | grep API_KEY`
            """)
            st.stop()
    
    return keys