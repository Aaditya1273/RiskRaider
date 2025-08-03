import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test each variable
def debug_env():
    print("🔍 Environment Variables Debug:")
    print("=" * 40)
    
    # Check if .env file exists
    env_file_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file_path):
        print(f"✅ .env file found at: {env_file_path}")
    else:
        print("❌ .env file not found!")
    
    # Required environment variables
    required_vars = ['DISCORD_TOKEN', 'BITSCRUNCH_API_KEY', 'OPENROUTER_API_KEY']
    # Optional environment variables
    optional_vars = ['PREFIX', 'DEBUG', 'COMMAND_COOLDOWN', 'OWNER_IDS']
    
    all_vars = required_vars + optional_vars
    
    for key in all_vars:
        val = os.getenv(key)
        print(f"{key} loaded: {val is not None}")
        if val:
            # Show first 20 and last 10 characters for tokens (but not for prefix)
            if key != 'PREFIX' and len(val) > 30:
                display_val = f"{val[:20]}...{val[-10:]}"
                print(f"{key} value: {display_val}")
                # Validate API key length
                if key == 'BITSCRUNCH_API_KEY':
                    if len(val) >= 32:
                        print(f"✅ {key} appears to be valid length ({len(val)} characters)")
                    else:
                        print(f"⚠️  Warning: {key} might be invalid (shorter than expected: {len(val)} characters)")
                        print("💡 bitsCrunch API keys are typically 32+ characters long")
                elif key == 'OPENROUTER_API_KEY' and len(val) < 20:
                    print(f"⚠️  Warning: {key} appears too short to be valid")
            else:
                display_val = val
                print(f"{key} value: {display_val}")
        else:
            if key in required_vars:
                print(f"❌ {key} value: None (REQUIRED)")
            else:
                print(f"{key} value: None")
        print()
    
    # Additional check for required variables
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
    else:
        print("✅ All required environment variables present")
        
    # Validate API key formats
    bitscrunch_key = os.getenv('BITSCRUNCH_API_KEY')
    if bitscrunch_key:
        if len(bitscrunch_key) < 32:
            print("⚠️  BITSCRUNCH_API_KEY may be invalid (too short)")
            print("💡 bitsCrunch API keys are typically 32+ characters long")
        else:
            print("✅ BITSCRUNCH_API_KEY appears to be valid length")
    else:
        print("❌ BITSCRUNCH_API_KEY not found!")

debug_env()

# Final status
token = os.getenv("DISCORD_TOKEN")
if token:
    print("✅ SUCCESS: DISCORD_TOKEN loaded successfully!")
    print(f"Token length: {len(token)} characters")
else:
    print("❌ FAILURE: DISCORD_TOKEN not found!")
    print("Please ensure your .env file exists and contains DISCORD_TOKEN=your_token_here")
    
# Check bitsCrunch API key
bitscrunch_key = os.getenv("BITSCRUNCH_API_KEY")
if bitscrunch_key:
    print("✅ SUCCESS: BITSCRUNCH_API_KEY loaded!")
    print(f"Key length: {len(bitscrunch_key)} characters")
    if len(bitscrunch_key) < 32:
        print("⚠️  WARNING: API key seems shorter than expected for bitsCrunch")
        print("💡 Troubleshooting tips:")
        print("   - Verify your key is correct and active")
        print("   - Check if it has been approved for mainnet access")
        print("   - Try regenerating a new API key from bitsCrunch dashboard")
else:
    print("❌ FAILURE: BITSCRUNCH_API_KEY not found!")
    print("Please ensure your .env file contains BITSCRUNCH_API_KEY=your_bitscrunch_key_here")
