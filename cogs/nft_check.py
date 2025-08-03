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
        self.bitscrunch_base = "https://api.bitscrunch.com/v1"
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
        wallet = wallet.strip().lower()
        
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
        headers = {
            "Authorization": f"Bearer {os.getenv('BITSCRUNCH_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        endpoints = {
            "risk": f"{self.bitscrunch_base}/wallet/{wallet}/risk",
            "profile": f"{self.bitscrunch_base}/wallet/{wallet}/profile", 
            "transactions": f"{self.bitscrunch_base}/wallet/{wallet}/transactions?limit=100",
            "connections": f"{self.bitscrunch_base}/wallet/{wallet}/connections"
        }
        
        results = {}
        
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
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"API request failed for {key}: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Request error for {key}: {e}")
            return {}

    async def _analyze_wallet_comprehensive(self, wallet: str, data: Dict) -> WalletAnalysis:
        """Perform comprehensive wallet analysis"""
        risk_data = data.get("risk", {})
        profile_data = data.get("profile", {})
        tx_data = data.get("transactions", {})
        connection_data = data.get("connections", {})
        
        # Extract basic metrics
        risk_score = risk_data.get("riskScore", 0)
        risky_nfts = risk_data.get("riskyNFTs", [])
        
        # Advanced analysis
        suspicious_activity = []
        recommendations = []
        
        # Analyze transaction patterns
        transactions = tx_data.get("transactions", [])
        tx_count = len(transactions)
        
        # Check for suspicious patterns
        if risk_score > 80:
            suspicious_activity.append("🚨 Extremely high risk score detected")
        
        if len(risky_nfts) > 10:
            suspicious_activity.append(f"⚠️ Large number of risky NFTs ({len(risky_nfts)})")
        
        # Check transaction frequency
        recent_txs = [tx for tx in transactions if self._is_recent_transaction(tx)]
        if len(recent_txs) > 50:
            suspicious_activity.append("📈 Unusually high transaction frequency")
        
        # Generate recommendations
        if risk_score > 70:
            recommendations.append("🛑 HIGH RISK: Avoid interacting with this wallet")
            recommendations.append("🔍 Conduct additional due diligence")
        elif risk_score > 40:
            recommendations.append("⚠️ CAUTION: Proceed with extra verification")
            recommendations.append("💰 Consider smaller transaction amounts initially")
        else:
            recommendations.append("✅ Generally safe for interaction")
            recommendations.append("🔄 Regular monitoring recommended")
        
        # Get wallet connections
        connected_wallets = connection_data.get("connectedWallets", [])[:5]  # Limit to 5
        
        return WalletAnalysis(
            wallet=wallet,
            risk_score=risk_score,
            risk_level=self._get_risk_level(risk_score),
            risky_nfts=risky_nfts,
            transaction_count=tx_count,
            total_value=profile_data.get("totalValue", 0),
            suspicious_activity=suspicious_activity,
            recommendations=recommendations,
            last_activity=self._get_last_activity(transactions),
            connected_wallets=connected_wallets
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
            return None
        
        try:
            latest_tx = max(transactions, key=lambda x: x.get("timestamp", 0))
            last_time = datetime.fromtimestamp(latest_tx.get("timestamp", 0))
            
            time_diff = datetime.now() - last_time
            if time_diff.days > 0:
                return f"{time_diff.days} days ago"
            elif time_diff.seconds > 3600:
                return f"{time_diff.seconds // 3600} hours ago"
            else:
                return f"{time_diff.seconds // 60} minutes ago"
        except:
            return "Unknown"

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
            bitscrunch_data = await self._fetch_bitscrunch_data(wallet)
            
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