import os
import sys

# Add backend directory to Python path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.gmail import get_gmail_service

def main():
    tokens_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tokens"))
    credentials_path = os.path.join(tokens_dir, "credentials.json")
    token_path = os.path.join(tokens_dir, "token.json")

    if not os.path.exists(credentials_path):
        print(f"❌ Error: credentials.json not found at {credentials_path}")
        print("Please download it from Google Cloud Console and place it in the tokens/ folder.")
        sys.exit(1)

    print("🚀 Starting Gmail authentication flow...")
    print("A browser window should open. Please log in to your Gmail account and authorize the app.")
    
    # This will trigger the browser flow and create token.json
    try:
        service = get_gmail_service(credentials_path, token_path)
        print(f"✅ Success! token.json has been created at {token_path}")
        print("You can now start your Docker container.")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
