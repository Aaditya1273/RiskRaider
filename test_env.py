import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test if variables are loaded
token = os.getenv("TOKEN")
prefix = os.getenv("PREFIX")

print(f"TOKEN loaded: {token is not None}")
print(f"TOKEN value: {token[:10] if token else 'None'}...")  # Show first 10 chars for security
print(f"PREFIX loaded: {prefix is not None}")
print(f"PREFIX value: {prefix}")

# Check if .env file exists
if os.path.exists(".env"):
    print(".env file exists")
    with open(".env", "r") as f:
        content = f.read()
        print(f".env file size: {len(content)} characters")
        print("First few lines:")
        for i, line in enumerate(content.split('\n')[:3]):
            print(f"  Line {i+1}: '{line}'")
else:
    print(".env file NOT found")
