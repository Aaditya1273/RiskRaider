import discord
from discord.ext import commands
from discord import app_commands
import requests
import os
import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file (if present)
from dotenv import load_dotenv
load_dotenv()

# Log API key presence for early debugging
logger.info(f"bitsCrunch key loaded: {bool(os.getenv('BITSCRUNCH_API_KEY'))}")
logger.info(f"OpenRouter key loaded: {bool(os.getenv('OPENROUTER_API_KEY'))}")

class RiskLevel(Enum):
    LOW = "🟢 LOW"
    MEDIUM = "🟡 MEDIUM" 
    HIGH = "🟠 HIGH"
    CRITICAL = "🔴 CRITICAL"

@dataclass
class WalletAnalysis:
    wallet: str
    risk_score: int
    risk_level: RiskLevel
    risky_nfts: List[Dict]
    transaction_count: int
    total_value: float
    suspicious_activity: List[str]
    recommendations: List[str]
    last_activity: Optional[str]
    connected_wallets: List[str]

class NFTRiskCache:
    def __init__(self, ttl_minutes: int = 30):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def get(self, key: str) -> Optional[WalletAnalysis]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: WalletAnalysis):
        self.cache[key] = (value, datetime.now())
    
    def clear_expired(self):
        now = datetime.now()
        expired_keys = [k for k, (_, ts) in self.cache.items() if now - ts >= self.ttl]
        for key in expired_keys:
            del self.cache[key]

class NFTCheck(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.cache = NFTRiskCache(ttl_minutes=30)
        self.rate_limits = {}
        self.session = None
        
        # API configurations
        self.bitscrunch_base = "https://api.unleashnfts.com/api/v1"
        self.openrouter_base = "https://openrouter.ai/api/v1"
        
        # Enhanced risk thresholds
        self.risk_thresholds = {
            RiskLevel.LOW: (0, 25),
            RiskLevel.MEDIUM: (26, 50),
            RiskLevel.HIGH: (51, 75),
            RiskLevel.CRITICAL: (76, 100)
        }

    async def cog_load(self):
        """Initialize aiohttp session when cog loads"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
        )
        
        # Debug API key loading
        bitscrunch_key = os.getenv("BITSCRUNCH_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        if not bitscrunch_key:
            logger.error("❌ bitsCrunch API key not found!")
        else:
            logger.info(f"✅ bitsCrunch API key loaded: {bitscrunch_key[:8]}...")
            
        if not openrouter_key:
            logger.error("❌ OpenRouter API key not found!")
        else:
            logger.info(f"✅ OpenRouter API key loaded: {openrouter_key[:8]}...")

    async def cog_unload(self):
        """Clean up aiohttp session when cog unloads"""
        if self.session:
            await self.session.close()

    def _get_risk_level(self, score: int) -> RiskLevel:
        """Determine risk level based on score"""
        for level, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= score <= max_score:
                return level
        return RiskLevel.CRITICAL

    def _validate_wallet(self, wallet: str) -> Tuple[bool, str]:
        """Validate wallet address format"""
        wallet = wallet.strip()
        
        # Check for common contract addresses (not wallets)
        known_contracts = {
            "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d": "Bored Ape Yacht Club (BAYC)",
            "0x60e4d786628fea6478f785a6d7e704777c86a7c6": "Mutant Ape Yacht Club (MAYC)",
            "0x34d85c9cdeb23fa97cb08333b511ac86e1c4e258": "Otherdeed for Otherside",
            "0x57f1887a8bf19b14fc0df6fd9b2acc9af147ea85": "Ethereum Name Service (ENS)"
        }
        
        wallet_lower = wallet.lower()
        if wallet_lower in known_contracts:
            return False, f"⚠️ This is a contract address ({known_contracts[wallet_lower]}), not a wallet. Please enter a valid user wallet address."
        
        # Ethereum address validation
        if wallet.startswith('0x') and len(wallet) == 42:
            try:
                int(wallet[2:], 16)  # Check if hex
                return True, wallet
            except ValueError:
                return False, "Invalid Ethereum address format"
        
        # ENS domain validation
        if wallet.endswith('.eth') and len(wallet) > 4:
            return True, wallet
            
        return False, "Unsupported wallet format. Use Ethereum address (0x...) or ENS domain (.eth)"

    async def _check_rate_limit(self, user_id: int) -> bool:
        """Check if user is rate limited (max 5 requests per 10 minutes)"""
        now = time.time()
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
        
        # Clean old requests
        self.rate_limits[user_id] = [req_time for req_time in self.rate_limits[user_id] 
                                   if now - req_time < 600]  # 10 minutes
        
        if len(self.rate_limits[user_id]) >= 5:
            return False
        
        self.rate_limits[user_id].append(now)
        return True

    async def _fetch_bitscrunch_data(self, wallet: str) -> Dict:
        """Fetch comprehensive data from bitsCrunch API"""
        api_key = os.getenv('BITSCRUNCH_API_KEY')
        if not api_key:
            logger.error("BITSCRUNCH_API_KEY not found in environment variables.")
            return {}

        # Validate API key format (should be longer than 10 characters)
        if len(api_key) < 10:
            logger.error(f"BITSCRUNCH_API_KEY appears to be invalid (too short: {len(api_key)} characters)")
            return {}

        headers = {
            "x-api-key": api_key,
            "accept": "application/json"  # Added accept header as per API documentation
        }
        
        endpoints = {
            "nfts": f"{self.bitscrunch_base}/nfts?address={wallet}&offset=0&limit=50",
            "collections": f"{self.bitscrunch_base}/collections?limit=10",
            "market_data": f"{self.bitscrunch_base}/nft_market_report?limit=5",
            "blockchains": f"{self.bitscrunch_base}/blockchains?limit=5"
        }
        
        results = {}
        
        # Log the API key validation status (without exposing the actual key)
        logger.info(f"🔑 bitsCrunch API key validation: {'✅ Valid' if len(api_key) > 20 else '⚠️ Potentially Invalid'}")
        
        session = self.session
        tasks = []
        for key, url in endpoints.items():
            tasks.append(self._safe_request(session, url, headers, key))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for key, response in zip(endpoints.keys(), responses):
            if isinstance(response, Exception):
                logger.error(f"Error fetching {key}: {response}")
                results[key] = {}
            else:
                results[key] = response

        return results

    async def _safe_request(self, session: aiohttp.ClientSession, url: str, 
                          headers: Dict, key: str) -> Dict:
        """Make a safe HTTP request with error handling"""
        try:
            logger.info(f"📡 Making {key} API request to: {url}")
            async with session.get(url, headers=headers) as response:
                logger.info(f"📡 {key} API Status Code: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ {key} API success - received {len(data)} fields")
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"❌ {key} API failed: {response.status} - {error_text[:200]}")
                    
                    # Specific error messages for common issues
                    if response.status == 400:
                        if "invalid request id" in error_text.lower():
                            logger.error(f"🔑 {key}: Invalid request ID - this usually indicates an API key issue")
                            logger.error("💡 Please verify your BITSCRUNCH_API_KEY is correct and active")
                        else:
                            logger.error(f"🔑 {key}: Bad request - check request format")
                    elif response.status == 401:
                        logger.error(f"🔑 {key}: Invalid API key or unauthorized")
                    elif response.status == 403:
                        logger.error(f"🚫 {key}: Forbidden - key not approved for mainnet")
                    elif response.status == 429:
                        logger.error(f"⏱️ {key}: Rate limited - too many requests")
                    elif response.status == 500:
                        logger.error(f"💥 {key}: Server error")
                    
                    return {}
        except Exception as e:
            logger.error(f"💥 Request error for {key}: {e}")
            return {}

    async def _analyze_wallet_comprehensive(self, wallet: str, data: Dict) -> WalletAnalysis:
        """Perform comprehensive wallet analysis using UnleashNFTs API data"""
        nfts_data = data.get("nfts", {})
        collections_data = data.get("collections", {})
        market_data = data.get("market_data", {})
        blockchains_data = data.get("blockchains", {})
        
        # Extract NFT data for the specific wallet
        nfts = nfts_data.get("nfts", []) if isinstance(nfts_data, dict) else []
        nft_count = len(nfts)
        
        # Get market context from collections data
        collections = collections_data.get("collections", []) if isinstance(collections_data, dict) else []
        market_report = market_data.get("reports", []) if isinstance(market_data, dict) else []
        
        # Calculate risk score based on available data
        risk_score = self._calculate_risk_score_simple(nft_count, nfts, collections)
        
        # Analyze NFTs for patterns
        risky_nfts = []
        suspicious_activity = []
        recommendations = []
        
        # Basic NFT analysis
        if nft_count == 0:
            suspicious_activity.append("ℹ️ No NFTs found - new wallet or inactive")
            recommendations.append("🔍 Verify wallet activity through other means")
        elif nft_count > 500:
            suspicious_activity.append(f"📈 Very high NFT count ({nft_count}) - possible bot activity")
            recommendations.append("⚠️ Investigate transaction patterns")
        
        # Analyze based on available NFT data
        collection_names = set()
        for nft in nfts:
            if isinstance(nft, dict):
                collection_name = nft.get("collection_name", nft.get("name", "Unknown"))
                collection_names.add(collection_name)
                
                # Check for potential red flags in NFT metadata
                if any(flag in str(nft).lower() for flag in ['suspicious', 'flagged', 'reported']):
                    risky_nfts.append({
                        "name": nft.get("name", "Unknown NFT"),
                        "collection": collection_name,
                        "reason": "Metadata indicates potential issues"
                    })
        
        # Collection diversity analysis
        unique_collections = len(collection_names)
        if nft_count > 0 and unique_collections > 0:
            diversity_ratio = unique_collections / nft_count
            if diversity_ratio < 0.1:  # Less than 10% diversity
                suspicious_activity.append(f"⚠️ Low collection diversity ({unique_collections} collections for {nft_count} NFTs)")
        
        # Generate risk-based recommendations
        if risk_score > 70:
            recommendations.extend([
                "🛑 HIGH RISK: Avoid interacting with this wallet",
                "🔍 Conduct additional due diligence",
                "📊 Verify wallet authenticity through multiple sources"
            ])
        elif risk_score > 40:
            recommendations.extend([
                "⚠️ CAUTION: Proceed with extra verification",
                "💰 Consider smaller transaction amounts initially",
                "🔍 Monitor wallet activity patterns"
            ])
        else:
            recommendations.extend([
                "✅ Generally appears safe for interaction",
                "🔄 Regular monitoring recommended",
                "📈 Standard wallet activity detected"
            ])
        
        # Add NFT-specific insights
        if nft_count > 100:
            recommendations.append("🎨 Active NFT collector detected")
        elif nft_count > 0:
            recommendations.append("🖼️ Moderate NFT activity")
        else:
            recommendations.append("🆕 No NFT activity - new or non-NFT wallet")
        
        # Estimate portfolio value (simplified)
        estimated_value = nft_count * 100  # Very rough estimate
        
        return WalletAnalysis(
            wallet=wallet,
            risk_score=risk_score,
            risk_level=self._get_risk_level(risk_score),
            risky_nfts=risky_nfts,
            transaction_count=nft_count,  # Use NFT count as proxy
            total_value=estimated_value,
            suspicious_activity=suspicious_activity,
            recommendations=recommendations,
            last_activity=self._get_last_activity_simple(nfts),
            connected_wallets=[]  # Not available in current API
        )

    def _is_recent_transaction(self, tx: Dict) -> bool:
        """Check if transaction is from last 7 days"""
        try:
            tx_time = datetime.fromtimestamp(tx.get("timestamp", 0))
            return datetime.now() - tx_time < timedelta(days=7)
        except:
            return False

    def _get_last_activity(self, transactions: List[Dict]) -> Optional[str]:
        """Get formatted last activity time"""
        if not transactions:
            return "No transactions found"
        
        try:
            # Filter out transactions without valid timestamps
            valid_transactions = [tx for tx in transactions if tx.get("timestamp") and isinstance(tx.get("timestamp"), (int, float))]
            
            if not valid_transactions:
                return "No valid timestamps found"
                
            latest_tx = max(valid_transactions, key=lambda x: x.get("timestamp", 0))
            timestamp = latest_tx.get("timestamp", 0)
            
            # Handle different timestamp formats
            if isinstance(timestamp, str):
                try:
                    timestamp = int(timestamp)
                except ValueError:
                    return "Unknown"
            
            # Convert timestamp to datetime
            if timestamp > 10000000000:  # If timestamp is in milliseconds
                timestamp = timestamp / 1000
                
            last_time = datetime.fromtimestamp(timestamp)
            
            # Calculate time difference
            time_diff = datetime.now() - last_time
            
            if time_diff.days > 365:
                return f"{time_diff.days // 365} years ago"
            elif time_diff.days > 30:
                return f"{time_diff.days // 30} months ago"
            elif time_diff.days > 0:
                return f"{time_diff.days} days ago"
            elif time_diff.seconds > 3600:
                return f"{time_diff.seconds // 3600} hours ago"
            elif time_diff.seconds > 60:
                return f"{time_diff.seconds // 60} minutes ago"
            else:
                return "Just now"
                
        except Exception as e:
            logger.error(f"Error calculating last activity: {e}")
            return "Unknown"

    def _calculate_risk_score(self, nfts: List[Dict], portfolio_data: Dict, balance_data: Dict) -> int:
        """Calculate risk score based on NFT and portfolio data"""
        risk_score = 0
        
        # Base risk factors
        nft_count = len(nfts)
        
        # NFT count analysis
        if nft_count == 0:
            risk_score += 20  # No NFTs might indicate new/inactive wallet
        elif nft_count > 1000:
            risk_score += 30  # Extremely high NFT count could indicate bot activity
        elif nft_count > 500:
            risk_score += 15  # High but reasonable for active collectors
        
        # Collection diversity analysis
        collections = {}
        suspicious_count = 0
        
        for nft in nfts:
            collection_name = nft.get("collection_name", "Unknown")
            collections[collection_name] = collections.get(collection_name, 0) + 1
            
            # Check for suspicious indicators
            if nft.get("is_suspicious", False):
                suspicious_count += 1
        
        # Collection concentration risk
        if collections:
            max_collection_ratio = max(collections.values()) / nft_count
            if max_collection_ratio > 0.9:  # 90%+ in one collection
                risk_score += 25
            elif max_collection_ratio > 0.7:  # 70%+ in one collection
                risk_score += 15
        
        # Suspicious NFT ratio
        if nft_count > 0:
            suspicious_ratio = suspicious_count / nft_count
            risk_score += int(suspicious_ratio * 40)  # Up to 40 points for suspicious NFTs
        
        # Portfolio value analysis
        portfolio_value = portfolio_data.get("total_value_usd", 0)
        if portfolio_value > 10000000:  # $10M+ could indicate institutional or high-risk activity
            risk_score += 20
        elif portfolio_value > 1000000:  # $1M+
            risk_score += 10
        
        # Ensure score is within bounds
        return min(max(risk_score, 0), 100)
    
    def _calculate_risk_score_simple(self, nft_count: int, nfts: List[Dict], collections: List[Dict]) -> int:
        """Calculate simplified risk score based on available data"""
        risk_score = 0
        
        # NFT count analysis
        if nft_count == 0:
            risk_score += 30  # No NFTs might indicate new/inactive wallet
        elif nft_count > 1000:
            risk_score += 40  # Extremely high NFT count could indicate bot activity
        elif nft_count > 500:
            risk_score += 25  # High but might be legitimate collector
        elif nft_count > 100:
            risk_score += 10  # Active collector
        else:
            risk_score += 5   # Normal activity
        
        # Collection analysis (if available)
        if nfts:
            collection_names = set()
            suspicious_indicators = 0
            
            for nft in nfts:
                if isinstance(nft, dict):
                    collection_name = nft.get("collection_name", nft.get("name", "Unknown"))
                    collection_names.add(collection_name)
                    
                    # Check for suspicious indicators in metadata
                    nft_str = str(nft).lower()
                    if any(flag in nft_str for flag in ['suspicious', 'flagged', 'reported', 'fake']):
                        suspicious_indicators += 1
            
            # Collection diversity risk
            if nft_count > 0:
                diversity_ratio = len(collection_names) / nft_count
                if diversity_ratio < 0.1:  # Very low diversity
                    risk_score += 20
                elif diversity_ratio < 0.3:  # Low diversity
                    risk_score += 10
            
            # Suspicious indicators
            if suspicious_indicators > 0:
                suspicious_ratio = suspicious_indicators / nft_count
                risk_score += int(suspicious_ratio * 30)  # Up to 30 points
        
        # Market context (if collections data available)
        if collections and isinstance(collections, list):
            # If we have market data, we can make more informed decisions
            risk_score -= 5  # Slight reduction for having market context
        
        return min(max(risk_score, 0), 100)
    
    def _get_last_activity_simple(self, nfts: List[Dict]) -> Optional[str]:
        """Get simplified last activity information"""
        if not nfts:
            return "No NFT activity found"
        
        # For now, just return a generic message based on NFT count
        nft_count = len(nfts)
        if nft_count > 100:
            return "High activity - many NFTs detected"
        elif nft_count > 10:
            return "Moderate activity - several NFTs detected"
        elif nft_count > 0:
            return "Low activity - few NFTs detected"
        else:
            return "No NFT activity detected"
    
    def _get_last_activity_from_nfts(self, nfts: List[Dict]) -> Optional[str]:
        """Get last activity time from NFT data"""
        if not nfts:
            return "No NFT activity found"
        
        try:
            # Look for timestamp fields in NFT data
            timestamps = []
            for nft in nfts:
                # Check various possible timestamp fields
                for field in ['last_transfer_timestamp', 'created_at', 'updated_at', 'timestamp']:
                    if field in nft and nft[field]:
                        try:
                            if isinstance(nft[field], str):
                                # Try to parse ISO format or convert to int
                                if 'T' in nft[field]:  # ISO format
                                    timestamp = datetime.fromisoformat(nft[field].replace('Z', '+00:00')).timestamp()
                                else:
                                    timestamp = int(nft[field])
                            else:
                                timestamp = float(nft[field])
                            
                            if timestamp > 1000000000:  # Valid timestamp
                                timestamps.append(timestamp)
                        except (ValueError, TypeError):
                            continue
            
            if not timestamps:
                return "No timestamp data available"
            
            # Get the most recent timestamp
            latest_timestamp = max(timestamps)
            
            # Handle milliseconds
            if latest_timestamp > 10000000000:
                latest_timestamp = latest_timestamp / 1000
            
            last_time = datetime.fromtimestamp(latest_timestamp)
            time_diff = datetime.now() - last_time
            
            if time_diff.days > 365:
                return f"{time_diff.days // 365} years ago"
            elif time_diff.days > 30:
                return f"{time_diff.days // 30} months ago"
            elif time_diff.days > 0:
                return f"{time_diff.days} days ago"
            elif time_diff.seconds > 3600:
                return f"{time_diff.seconds // 3600} hours ago"
            elif time_diff.seconds > 60:
                return f"{time_diff.seconds // 60} minutes ago"
            else:
                return "Recently active"
                
        except Exception as e:
            logger.error(f"Error calculating NFT activity: {e}")
            return "Activity data unavailable"

    async def _generate_ai_summary(self, analysis: WalletAnalysis) -> str:
        """Generate AI-powered risk assessment summary"""
        try:
            prompt = f"""
            Analyze this NFT wallet risk assessment:
            
            Wallet: {analysis.wallet}
            Risk Score: {analysis.risk_score}/100
            Risk Level: {analysis.risk_level.value}
            Risky NFTs: {len(analysis.risky_nfts)}
            Transaction Count: {analysis.transaction_count}
            Total Portfolio Value: ${analysis.total_value:,.2f}
            Suspicious Activities: {len(analysis.suspicious_activity)}
            Last Activity: {analysis.last_activity}
            
            Provide a concise, professional risk assessment in 2-3 sentences. 
            Focus on actionable insights and clear recommendations.
            Use emojis appropriately but sparingly.
            """

            headers = {
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("BOT_REFERER_URL", "riskraider-bot"),
                "X-Title": "RiskRaider NFT Risk Analyzer"
            }

            payload = {
                "model": "anthropic/claude-3-haiku",  # More reliable than GPT-3.5
                "messages": [
                    {"role": "system", "content": "You are an expert NFT risk analyst. Provide clear, actionable risk assessments."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.3
            }

            async with self.session.post(f"{self.openrouter_base}/chat/completions", 
                                       headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content'].strip()
                
        except Exception as e:
            logger.error(f"AI summary generation failed: {e}")
        
        # Fallback summary
        risk_emoji = "🔴" if analysis.risk_score > 70 else "🟡" if analysis.risk_score > 40 else "🟢"
        return f"{risk_emoji} This wallet has a {analysis.risk_level.value.split()[1]} risk profile with {len(analysis.risky_nfts)} flagged NFTs. {'Proceed with extreme caution.' if analysis.risk_score > 70 else 'Standard security practices recommended.' if analysis.risk_score > 40 else 'Generally safe for interaction.'}"

    def _create_detailed_embed(self, analysis: WalletAnalysis, ai_summary: str) -> discord.Embed:
        """Create comprehensive Discord embed"""
        # Color based on risk level
        color_map = {
            RiskLevel.LOW: discord.Color.green(),
            RiskLevel.MEDIUM: discord.Color.yellow(), 
            RiskLevel.HIGH: discord.Color.orange(),
            RiskLevel.CRITICAL: discord.Color.red()
        }
        
        embed = discord.Embed(
            title=f"🔍 NFT Wallet Analysis",
            description=ai_summary,
            color=color_map[analysis.risk_level],
            timestamp=datetime.now(timezone.utc)
        )
        
        # Wallet info
        wallet_display = analysis.wallet[:10] + "..." + analysis.wallet[-8:] if len(analysis.wallet) > 20 else analysis.wallet
        embed.add_field(
            name="📋 Wallet Address", 
            value=f"`{wallet_display}`", 
            inline=False
        )
        
        # Risk metrics
        risk_bar = self._create_risk_bar(analysis.risk_score)
        embed.add_field(
            name="⚠️ Risk Assessment",
            value=f"{analysis.risk_level.value}\n`{risk_bar}` **{analysis.risk_score}/100**",
            inline=True
        )
        
        # Portfolio info
        embed.add_field(
            name="💼 Portfolio Overview",
            value=f"🏷️ **Risky NFTs:** {len(analysis.risky_nfts)}\n"
                  f"📊 **Transactions:** {analysis.transaction_count:,}\n"
                  f"💰 **Est. Value:** ${analysis.total_value:,.2f}",
            inline=True
        )
        
        # Activity info
        embed.add_field(
            name="📈 Activity Status", 
            value=f"🕒 **Last Activity:** {analysis.last_activity or 'Unknown'}\n"
                  f"🔗 **Connected Wallets:** {len(analysis.connected_wallets)}",
            inline=True
        )
        
        # Suspicious activity
        if analysis.suspicious_activity:
            embed.add_field(
                name="🚨 Suspicious Activity",
                value="\n".join(analysis.suspicious_activity[:3]),  # Limit to 3
                inline=False
            )
        
        # Recommendations
        if analysis.recommendations:
            embed.add_field(
                name="💡 Recommendations",
                value="\n".join(analysis.recommendations[:3]),  # Limit to 3
                inline=False
            )
        
        embed.set_footer(
            text="Powered by bitsCrunch • Data cached for 30 minutes",
            icon_url="https://cdn.discordapp.com/emojis/1234567890123456789.png"  # Optional: Add your bot's icon
        )
        
        return embed

    def _create_risk_bar(self, score: int) -> str:
        """Create visual risk score bar"""
        filled = int(score / 10)
        empty = 10 - filled
        return "█" * filled + "░" * empty

    @commands.command(name="nftcheck", help="🔍 Comprehensive NFT wallet risk analysis with AI insights")
    async def nftcheck(self, ctx: commands.Context, *, wallet: str):
        """Analyzes an NFT wallet for risk factors."""
        logger.info(f"🔍 nftcheck called by {ctx.author} with wallet: {wallet}")
        
        # Defer the response to prevent timeout
        async with ctx.typing():
            # Rate limiting
            if not await self._check_rate_limit(ctx.author.id):
                embed = discord.Embed(
                    title="⏱️ Rate Limited",
                    description="You can only check 5 wallets per 10 minutes. Please try again later.",
                    color=discord.Color.red()
                )
                await ctx.send(embed=embed)
                return

            # Validate wallet
            is_valid, result = self._validate_wallet(wallet)
            if not is_valid:
                embed = discord.Embed(
                    title="❌ Invalid Wallet",
                    description=result,
                    color=discord.Color.red()
                )
                await ctx.send(embed=embed)
                return

            wallet = result
            logger.info(f"✅ Wallet validated: {wallet}")
            
            # Check cache first
            cache_key = hashlib.md5(f"{wallet}".encode()).hexdigest()
            cached_analysis = self.cache.get(cache_key)
            
            if cached_analysis:
                ai_summary = await self._generate_ai_summary(cached_analysis)
                embed = self._create_detailed_embed(cached_analysis, ai_summary)
                embed.set_footer(text="Powered by bitsCrunch • Cached data")
                await ctx.send(embed=embed)
                return
            
            # Fetch fresh data
            status_embed = discord.Embed(
                title="🔄 Analyzing Wallet...",
                description=f"Fetching risk data for `{wallet[:10]}...{wallet[-8:]}`\n"
                           f"⏳ This may take up to 30 seconds...",
                color=discord.Color.blue()
            )
            status_message = await ctx.send(embed=status_embed)
            
            # Get data from bitsCrunch
            logger.info(f"📡 Calling bitsCrunch API for wallet: {wallet}")
            bitscrunch_data = await self._fetch_bitscrunch_data(wallet)
            logger.info(f"📊 bitsCrunch data received: {list(bitscrunch_data.keys())}")
            logger.info(f"📊 Data contents: {bitscrunch_data}")
            
            if not any(bitscrunch_data.values()):
                embed = discord.Embed(
                    title="❌ Data Unavailable",
                    description="Unable to fetch wallet data. This could be due to:\n"
                               "• API rate limits\n"
                               "• Invalid wallet address\n" 
                               "• Temporary service issues\n\n"
                               "Please try again in a few minutes.",
                    color=discord.Color.red()
                )
                await status_message.edit(embed=embed)
                return
            
            # Analyze the data
            analysis = await self._analyze_wallet_comprehensive(wallet, bitscrunch_data)
            
            # Cache the result
            self.cache.set(cache_key, analysis)
            
            # Generate AI summary
            ai_summary = await self._generate_ai_summary(analysis)
            
            # Create and send final embed
            embed = self._create_detailed_embed(analysis, ai_summary)
            
            await status_message.edit(embed=embed)
            
            # Log successful analysis
            logger.info(f"NFT analysis completed for {wallet} (Risk: {analysis.risk_score})")
            
    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        """Global error handler for debugging"""
        if isinstance(error, commands.CommandInvokeError):
            original_error = error.original
            logger.error(f"💥 Command {ctx.command} failed: {original_error}")
            
            embed = discord.Embed(
                title="❌ Command Error",
                description=f"An error occurred: `{str(original_error)[:500]}`",
                color=discord.Color.red()
            )
            await ctx.send(embed=embed)
        else:
            logger.error(f"💥 Unhandled error in {ctx.command}: {error}")
            await ctx.send(f"❌ Error: `{error}`")
            

    @commands.command(name="nftstats", help="📊 View your NFT checking statistics")
    async def nftstats(self, ctx: commands.Context):
        """Show user's usage statistics"""
        user_id = ctx.author.id
        recent_requests = len(self.rate_limits.get(user_id, []))
        
        embed = discord.Embed(
            title="📊 Your NFT Check Statistics",
            color=discord.Color.blue()
        )
        embed.add_field(
            name="⏱️ Recent Usage",
            value=f"**{recent_requests}/5** requests used in the last 10 minutes",
            inline=False
        )
        embed.add_field(
            name="💾 Cache Status", 
            value=f"**{len(self.cache.cache)}** analyses cached",
            inline=True
        )
        embed.add_field(
            name="🔄 Reset Time",
            value=f"<t:{int(time.time() + 600)}:R>",  # 10 minutes from now
            inline=True
        )
        
        await ctx.send(embed=embed)

    @commands.command(
        name="clearcache", 
        description="(Admin) Clears the NFT analysis cache.",
        hidden=True
    )
    @commands.has_permissions(administrator=True)
    async def clear_cache(self, ctx):
        """Admin command to clear the analysis cache"""
        self.cache.cache.clear()
        await ctx.send("🧹 Analysis cache cleared successfully!")

async def setup(bot):
    await bot.add_cog(NFTCheck(bot))