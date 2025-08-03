import asyncio
import json
import logging
import os
import platform
import random
import sys
import traceback
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import signal

import aiosqlite
import discord
from discord.ext import commands, tasks
from discord.ext.commands import Context
from dotenv import load_dotenv
import aiohttp

from database import DatabaseManager, create_database_manager

# Load environment variables
load_dotenv()

class ColoredFormatter(logging.Formatter):
    """Enhanced logging formatter with colors and better formatting"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{self.BOLD}{record.levelname:<8}{self.RESET}"
        record.name = f"\033[94m{self.BOLD}{record.name}{self.RESET}"
        record.msg = f"{record.msg}"
        
        formatter = logging.Formatter(
            fmt="[{asctime}] {levelname} {name}: {message}",
            datefmt="%Y-%m-%d %H:%M:%S",
            style="{"
        )
        return formatter.format(record)

def setup_logging() -> logging.Logger:
    """Setup enhanced logging configuration"""
    logger = logging.getLogger("discord_bot")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    console_handler.setLevel(logging.INFO)
    
    # File handler for persistent logging
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(
        filename=f"logs/discord_{time.strftime('%Y%m%d')}.log",
        encoding="utf-8",
        mode="a"
    )
    file_formatter = logging.Formatter(
        fmt="[{asctime}] [{levelname:<8}] {name}: {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Error file handler for errors only
    error_handler = logging.FileHandler(
        filename=f"logs/errors_{time.strftime('%Y%m%d')}.log",
        encoding="utf-8",
        mode="a"
    )
    error_handler.setFormatter(file_formatter)
    error_handler.setLevel(logging.ERROR)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    return logger

class EnhancedBot(commands.Bot):
    """Enhanced Discord bot with advanced features and better error handling"""
    
    def __init__(self, **kwargs) -> None:
        # Setup intents inline to avoid method reference issue
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True  # For member management features
        intents.guilds = True
        intents.guild_messages = True
        intents.guild_reactions = True
        
        super().__init__(
            command_prefix=self._get_prefix,
            intents=intents,
            case_insensitive=True,
            strip_after_prefix=True,
            help_command=None,
            **kwargs
        )
        
        # Configuration
        self.config = self._load_config()
        
        # Bot state
        self.start_time = datetime.now(timezone.utc)
        self.session: Optional[aiohttp.ClientSession] = None
        self.database: Optional[DatabaseManager] = None
        self.command_usage: Dict[str, int] = {}
        self.blacklist_cache: set = set()
        self.logger = setup_logging()
        
        # Performance tracking
        self.error_count = 0
        self.last_activity = time.time()
        
        # Guild-specific prefixes (cached)
        self.guild_prefixes: Dict[int, str] = {}
        
        # Graceful shutdown handling
        self._shutdown_event = asyncio.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.close())
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables with validation"""
        required_vars = ["DISCORD_TOKEN"]
        config = {}
        
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                raise ValueError(f"Required environment variable {var} is not set")
            config[var.lower()] = value
        
        # Optional configuration with defaults
        config.update({
            "prefix": os.getenv("PREFIX", "!"),
            "invite_link": os.getenv("INVITE_LINK", ""),
            "owner_ids": [int(id.strip()) for id in os.getenv("OWNER_IDS", "").split(",") if id.strip()],
            "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
            "command_cooldown": float(os.getenv("COMMAND_COOLDOWN", "1.0")),
        })
        
        return config
    
    async def _get_prefix(self, bot, message: discord.Message) -> List[str]:
        """Dynamic prefix system with guild-specific prefixes"""
        # Default prefixes are the bot's mention and the configured default prefix
        return commands.when_mentioned_or(self.config["prefix"])(bot, message)
    
    async def setup_hook(self) -> None:
        """Enhanced setup with better error handling and performance optimizations"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("🚀 Enhanced Discord Bot Starting Up...")
            self.logger.info("=" * 60)
            
            # System information
            self.logger.info(f"Bot: {self.user}")
            self.logger.info(f"Discord.py: {discord.__version__}")
            self.logger.info(f"Python: {platform.python_version()}")
            self.logger.info(f"Platform: {platform.system()} {platform.release()}")
            self.logger.info(f"Process ID: {os.getpid()}")
            
            # Setup HTTP session for external requests
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(limit=100, limit_per_host=30)
            )
            
            # Initialize database
            await self._init_database()
            
            # Load extensions
            await self._load_extensions()
            
            # Sync slash commands to Discord
            synced = await self.tree.sync()
            self.logger.info(f"🔄 Synced {len(synced)} application commands.")
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("✅ Bot setup completed successfully!")
            
        except Exception as e:
            self.logger.critical(f"Failed to setup bot: {e}")
            self.logger.critical(traceback.format_exc())
            await self.close()
            sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.close())
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables with validation"""
        required_vars = ["DISCORD_TOKEN"]
        config = {}
        
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                raise ValueError(f"Required environment variable {var} is not set")
            config[var.lower()] = value
        
        # Optional configuration with defaults
        config.update({
            "prefix": os.getenv("PREFIX", "!"),
            "invite_link": os.getenv("INVITE_LINK", ""),
            "owner_ids": [int(id.strip()) for id in os.getenv("OWNER_IDS", "").split(",") if id.strip()],
            "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
            "command_cooldown": float(os.getenv("COMMAND_COOLDOWN", "1.0")),
        })
        
        return config
    
    async def _init_database(self) -> None:
        """
        Initialize database connection.
        """
        try:
            db_path = os.path.join(os.path.dirname(__file__), "database", "database.db")
            
            # Ensure database directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Initialize database manager using the factory function
            self.database = await create_database_manager(db_path)
            
            self.logger.info("️ Database connection established")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _load_extensions(self) -> None:
        """Load cogs with better error handling and performance tracking"""
        cogs_dir = os.path.join(os.path.dirname(__file__), "cogs")
        if not os.path.exists(cogs_dir):
            self.logger.warning("Cogs directory not found, creating...")
            os.makedirs(cogs_dir, exist_ok=True)
            return
        
        loaded_count = 0
        failed_count = 0
        
        for filename in os.listdir(cogs_dir):
            if filename.endswith(".py") and not filename.startswith("_") and filename != "template.py":
                extension = filename[:-3]
                try:
                    start_time = time.time()
                    await self.load_extension(f"cogs.{extension}")
                    load_time = (time.time() - start_time) * 1000
                    self.logger.info(f"📦 Loaded extension '{extension}' in {load_time:.2f}ms")
                    loaded_count += 1
                except Exception as e:
                    self.logger.error(f"❌ Failed to load extension '{extension}': {e}")
                    if self.config["debug_mode"]:
                        self.logger.error(traceback.format_exc())
                    failed_count += 1
        
        self.logger.info(f"📦 Extensions loaded: {loaded_count} successful, {failed_count} failed")
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks for maintenance and monitoring"""
        self.status_rotation.start()
        self.performance_monitor.start()
        self.cache_cleanup.start()
        self.logger.info("⚙️ Background tasks started")
    
    @tasks.loop(minutes=5)
    async def status_rotation(self) -> None:
        """Rotate bot status with dynamic information"""
        try:
            statuses = [
                f"with {len(self.guilds)} servers!",
                f"with {len(self.users)} users!",
                f"since {time.strftime('%H:%M', time.localtime(self.start_time.timestamp()))}",
                "and learning new tricks!",
                f"with {len(self.commands)} commands!",
            ]
            
            status = random.choice(statuses)
            activity = discord.Activity(
                type=discord.ActivityType.playing,
                name=status
            )
            await self.change_presence(activity=activity, status=discord.Status.online)
            
        except Exception as e:
            self.logger.error(f"Status rotation error: {e}")
    
    @tasks.loop(minutes=30)
    async def performance_monitor(self) -> None:
        """Monitor bot performance and log statistics"""
        try:
            uptime = time.time() - self.start_time.timestamp()
            memory_usage = sys.getsizeof(self) / 1024 / 1024  # MB
            
            stats = {
                "uptime": f"{uptime/3600:.1f}h",
                "guilds": len(self.guilds),
                "users": len(self.users),
                "memory": f"{memory_usage:.1f}MB",
                "errors": self.error_count,
                "latency": f"{self.latency*1000:.0f}ms"
            }
            
            self.logger.info(f"📊 Performance Stats: {stats}")
            
        except Exception as e:
            self.logger.error(f"Performance monitoring error: {e}")
    
    @tasks.loop(hours=1)
    async def cache_cleanup(self) -> None:
        """Clean up caches and perform maintenance"""
        try:
            # Clear old guild prefix cache entries
            current_guild_ids = {guild.id for guild in self.guilds}
            self.guild_prefixes = {
                gid: prefix for gid, prefix in self.guild_prefixes.items()
                if gid in current_guild_ids
            }
            
            # Reset command usage stats daily
            if time.time() - self.start_time.timestamp() > 86400:  # 24 hours
                self.command_usage.clear()
                self.error_count = 0
            
            self.logger.debug("🧹 Cache cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cache cleanup error: {e}")
    
    @status_rotation.before_loop
    @performance_monitor.before_loop
    @cache_cleanup.before_loop
    async def wait_until_ready_wrapper(self) -> None:
        """Wait until bot is ready before starting tasks"""
        await self.wait_until_ready()
    
    async def on_ready(self) -> None:
        """Enhanced ready event with comprehensive information"""
        self.logger.info("🎉 Bot is now online and ready!")
        self.logger.info(f"📊 Connected to {len(self.guilds)} guilds with {len(self.users)} users")
        
        if self.config["invite_link"]:
            self.logger.info(f"🔗 Invite link: {self.config['invite_link']}")
    
    async def on_guild_join(self, guild: discord.Guild) -> None:
        """Log when bot joins a new guild"""
        self.logger.info(f"📥 Joined guild: {guild.name} (ID: {guild.id}) with {guild.member_count} members")
        
        # Clear cached prefix for this guild
        self.guild_prefixes.pop(guild.id, None)
    
    async def on_guild_remove(self, guild: discord.Guild) -> None:
        """Log when bot leaves a guild"""
        self.logger.info(f"📤 Left guild: {guild.name} (ID: {guild.id})")
        
        # Clear cached data for this guild
        self.guild_prefixes.pop(guild.id, None)
    
    async def on_message(self, message: discord.Message) -> None:
        """Enhanced message processing with better performance"""
        if message.author.bot:
            return
        
        self.last_activity = time.time()
        
        # Process commands
        await self.process_commands(message)
    
    async def on_command(self, ctx: Context) -> None:
        """Track command usage for analytics"""
        command_name = ctx.command.qualified_name
        self.command_usage[command_name] = self.command_usage.get(command_name, 0) + 1
    
    async def on_command_completion(self, ctx: Context) -> None:
        """Enhanced command completion logging"""
        execution_time = time.time() - ctx.message.created_at.timestamp()
        location = f"{ctx.guild.name} (ID: {ctx.guild.id})" if ctx.guild else "DM"
        
        self.logger.info(
            f"✅ Command '{ctx.command.qualified_name}' executed by "
            f"{ctx.author} (ID: {ctx.author.id}) in {location} "
            f"[{execution_time*1000:.0f}ms]"
        )
    
    async def on_command_error(self, ctx: Context, error: Exception) -> None:
        """Comprehensive error handling with user-friendly messages"""
        self.error_count += 1
        
        # Cooldown errors
        if isinstance(error, commands.CommandOnCooldown):
            retry_after = error.retry_after
            if retry_after < 60:
                time_left = f"{retry_after:.1f} seconds"
            elif retry_after < 3600:
                time_left = f"{retry_after/60:.1f} minutes"
            else:
                time_left = f"{retry_after/3600:.1f} hours"
            
            embed = discord.Embed(
                title="⏰ Command on Cooldown",
                description=f"Please wait **{time_left}** before using this command again.",
                color=0xFF6B6B
            )
            await ctx.send(embed=embed, delete_after=10)
        
        # Permission errors
        elif isinstance(error, (commands.MissingPermissions, commands.BotMissingPermissions)):
            missing_perms = error.missing_permissions
            perms_list = ", ".join([perm.replace("_", " ").title() for perm in missing_perms])
            
            if isinstance(error, commands.MissingPermissions):
                title = "🚫 Missing Permissions"
                description = f"You need the following permissions: **{perms_list}**"
            else:
                title = "⚠️ Bot Missing Permissions"
                description = f"I need the following permissions: **{perms_list}**"
            
            embed = discord.Embed(title=title, description=description, color=0xFFB84D)
            await ctx.send(embed=embed)
        
        # Owner only commands
        elif isinstance(error, commands.NotOwner):
            embed = discord.Embed(
                title="👑 Owner Only",
                description="This command can only be used by bot owners.",
                color=0x9B59B6
            )
            await ctx.send(embed=embed, delete_after=5)
        
        # Missing arguments
        elif isinstance(error, commands.MissingRequiredArgument):
            embed = discord.Embed(
                title="❓ Missing Argument",
                description=f"Missing required argument: **{error.param.name}**\n"
                           f"Use `{ctx.prefix}help {ctx.command}` for usage information.",
                color=0x3498DB
            )
            await ctx.send(embed=embed)
        
        # Command not found (silently ignore)
        elif isinstance(error, commands.CommandNotFound):
            return
        
        # Unexpected errors
        else:
            error_id = hash(str(error)) % 10000
            
            self.logger.error(
                f"❌ Unexpected error (ID: {error_id}) in command '{ctx.command}': {error}"
            )
            self.logger.error(traceback.format_exc())
            
            embed = discord.Embed(
                title="💥 Unexpected Error",
                description=f"An unexpected error occurred (Error ID: `{error_id}`).\n"
                           f"This has been logged and will be investigated.",
                color=0xE74C3C
            )
            
            if self.config.get("debug_mode", False):
                traceback_str = traceback.format_exc()
                embed.add_field(name="Traceback", value=f"```py\n{traceback_str[:1000]}\n```", inline=False)
            
            await ctx.send(embed=embed)
    
    async def close(self) -> None:
        """Graceful shutdown with cleanup"""
        self.logger.info("🔄 Initiating graceful shutdown...")
        
        try:
            # Stop background tasks
            self.status_rotation.cancel()
            self.performance_monitor.cancel()
            self.cache_cleanup.cancel()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
            
            # Close database connections
            if self.database:
                await self.database.close()
            
            # Close bot connection
            await super().close()
            
            self.logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        
        finally:
            # Close logging handlers
            for handler in self.logger.handlers:
                handler.close()

def main():
    """Main entry point with error handling"""
    try:
        bot = EnhancedBot()
        bot.run(bot.config["discord_token"])
    except KeyboardInterrupt:
        print("\n🔄 Shutdown initiated by user")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()