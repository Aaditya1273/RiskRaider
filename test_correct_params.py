import os
import asyncio
import aiohttp
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_correct_parameters():
    """Test with the correct 'address' parameter"""
    api_key = os.getenv('BITSCRUNCH_API_KEY')
    
    if not api_key:
        print("❌ BITSCRUNCH_API_KEY not found!")
        return
    
    print("🔍 Testing UnleashNFTs API with Correct Parameters")
    print("=" * 60)
    print(f"✅ API Key: {api_key[:8]}...{api_key[-5:]}")
    
    headers = {
        "x-api-key": api_key,
        "accept": "application/json"
    }
    
    base_url = "https://api.unleashnfts.com/api/v1"
    test_wallet = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"  # Vitalik
    
    # Test with 'address' parameter (from error message hint)
    test_endpoints = {
        "nfts_with_address": f"{base_url}/nfts?address={test_wallet}&chain_id=1&offset=0&limit=10",
        "wallet_balance_with_address": f"{base_url}/wallet/balance/token?address={test_wallet}&chain_id=1",
        "wallet_portfolio_with_address": f"{base_url}/wallet/portfolio?address={test_wallet}&chain_id=1",
        
        # Also test some working endpoints we know
        "blockchains": f"{base_url}/blockchains?sort_by=blockchain_name&offset=0&limit=5",
        "collections": f"{base_url}/collections?chain_id=1&sort_by=market_cap&offset=0&limit=5",
    }
    
    async with aiohttp.ClientSession() as session:
        successful_endpoints = []
        
        for endpoint_name, url in test_endpoints.items():
            print(f"\n🔍 Testing {endpoint_name}...")
            print(f"   URL: {url}")
            
            try:
                async with session.get(url, headers=headers) as response:
                    print(f"   Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ SUCCESS! Data keys: {list(data.keys()) if isinstance(data, dict) else 'List data'}")
                        
                        # Show sample data structure
                        if isinstance(data, dict):
                            for key, value in list(data.items())[:3]:  # Show first 3 keys
                                if isinstance(value, list) and value:
                                    print(f"   📊 {key}: List with {len(value)} items")
                                    if value and isinstance(value[0], dict):
                                        print(f"       Sample item keys: {list(value[0].keys())[:5]}")
                                else:
                                    print(f"   📊 {key}: {type(value).__name__}")
                        
                        successful_endpoints.append({
                            "name": endpoint_name,
                            "url": url,
                            "data": data
                        })
                        
                    elif response.status == 422:
                        error_data = await response.json()
                        print(f"   ❌ Validation Error: {error_data.get('message', 'Unknown')}")
                    else:
                        error_text = await response.text()
                        print(f"   ❌ Error {response.status}: {error_text[:150]}")
                        
            except Exception as e:
                print(f"   💥 Exception: {e}")
            
            await asyncio.sleep(0.5)
        
        print("\n" + "=" * 60)
        print("🎉 RESULTS SUMMARY")
        print("=" * 60)
        
        if successful_endpoints:
            print(f"✅ Found {len(successful_endpoints)} working endpoints:")
            for endpoint in successful_endpoints:
                print(f"   • {endpoint['name']}")
            
            # Show the correct API structure for the bot
            wallet_endpoints = [e for e in successful_endpoints if 'wallet' in e['name'] or 'nfts' in e['name']]
            if wallet_endpoints:
                print(f"\n🎯 CORRECT API STRUCTURE FOR BOT:")
                for endpoint in wallet_endpoints:
                    print(f"   {endpoint['name']}: {endpoint['url']}")
        else:
            print("❌ No working wallet endpoints found")

if __name__ == "__main__":
    asyncio.run(test_correct_parameters())
