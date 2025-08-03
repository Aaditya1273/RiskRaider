import asyncio
import logging
from typing import Optional, Union, Any, Dict, List
from datetime import datetime, timedelta

import discord
from discord.ext import commands, tasks
from discord.ext.commands import Context, Cog, Bot
from discord import app_commands


class RiskRaiderTemplate(commands.Cog, name="riskraider_template"):
    """
    🏆 RiskRaider Professional Discord Bot Cog Template
    
    A comprehensive, production-ready cog template featuring:
    • Advanced error handling and logging
    • Hybrid commands (slash + prefix)
    • Permission management and security
    • Rate limiting and cooldowns
    • Database-ready structure
    • Auto-completion and validation
    • Professional documentation
    • Performance monitoring
    """
    
    def __init__(self, bot: Bot) -> None:
        self.bot = bot
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configuration and state management
        self.config: Dict[str, Any] = {
            "max_retries": 3,
            "timeout": 30.0,
            "cache_size": 1000,
            "rate_limit": 5  # commands per minute
        }
        
        # Runtime statistics
        self.stats: Dict[str, Any] = {
            "commands_executed": 0,
            "errors_handled": 0,
            "uptime_start": datetime.utcnow(),
            "last_maintenance": None
        }
        
        # Cache and temporary storage
        self.cache: Dict[int, Dict[str, Any]] = {}
        self.user_cooldowns: Dict[int, datetime] = {}
        
        # Start background tasks
        self.maintenance_task.start()
        self.stats_updater.start()
        
        self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")

    def cog_unload(self) -> None:
        """Cleanup when cog is unloaded"""
        self.maintenance_task.cancel()
        self.stats_updater.cancel()
        self.logger.info(f"🔄 {self.__class__.__name__} unloaded")

    async def cog_before_invoke(self, ctx: Context) -> None:
        """Pre-execution hook for all commands"""
        self.stats["commands_executed"] += 1
        self.logger.debug(f"🎯 Command '{ctx.command}' invoked by {ctx.author} in {ctx.guild}")

    async def cog_after_invoke(self, ctx: Context) -> None:
        """Post-execution hook for all commands"""
        # Update user activity cache
        if ctx.guild:
            self._update_user_cache(ctx.author.id, ctx.guild.id)

    async def cog_command_error(self, ctx: Context, error: commands.CommandError) -> None:
        """Centralized error handling for all commands in this cog"""
        self.stats["errors_handled"] += 1
        
        # Create professional error embed
        embed = discord.Embed(
            title="⚠️ Command Error",
            color=discord.Color.red(),
            timestamp=datetime.utcnow()
        )
        
        if isinstance(error, commands.CommandOnCooldown):
            embed.description = f"⏰ Please wait {error.retry_after:.1f} seconds before using this command again."
            embed.color = discord.Color.orange()
        elif isinstance(error, commands.MissingPermissions):
            embed.description = f"🚫 You need the following permissions: {', '.join(error.missing_permissions)}"
        elif isinstance(error, commands.BotMissingPermissions):
            embed.description = f"🤖 I need the following permissions: {', '.join(error.missing_permissions)}"
        elif isinstance(error, commands.CommandNotFound):
            return  # Ignore command not found errors
        else:
            embed.description = f"❌ An unexpected error occurred: `{str(error)}`"
            self.logger.error(f"Unhandled error in {ctx.command}: {error}", exc_info=error)
        
        embed.set_footer(text=f"Error ID: {hash(str(error)) % 10000}")
        
        try:
            await ctx.send(embed=embed, ephemeral=True)
        except discord.HTTPException:
            pass

    def _update_user_cache(self, user_id: int, guild_id: int) -> None:
        """Update user activity cache"""
        if user_id not in self.cache:
            self.cache[user_id] = {}
        
        self.cache[user_id].update({
            "last_seen": datetime.utcnow(),
            "guild_id": guild_id,
            "command_count": self.cache[user_id].get("command_count", 0) + 1
        })

    def _check_rate_limit(self, user_id: int) -> bool:
        """Check if user is rate limited"""
        now = datetime.utcnow()
        if user_id in self.user_cooldowns:
            if now - self.user_cooldowns[user_id] < timedelta(minutes=1):
                return False
        
        self.user_cooldowns[user_id] = now
        return True

    # ==================== HYBRID COMMANDS ====================

    @commands.hybrid_command(
        name="status",
        description="📊 Display comprehensive bot status and statistics",
        aliases=["stats", "info", "health"]
    )
    @app_commands.describe(
        detailed="Show detailed statistics and performance metrics"
    )
    async def status_command(self, ctx: Context, detailed: bool = False) -> None:
        """
        Display comprehensive bot status information
        
        Args:
            ctx: Command context
            detailed: Whether to show detailed metrics
        """
        try:
            # Calculate uptime
            uptime = datetime.utcnow() - self.stats["uptime_start"]
            uptime_str = f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m"
            
            # Create status embed
            embed = discord.Embed(
                title="🤖 Bot Status Dashboard",
                description="Real-time system status and performance metrics",
                color=discord.Color.green(),
                timestamp=datetime.utcnow()
            )
            
            # Basic stats
            embed.add_field(
                name="⏱️ Uptime",
                value=f"`{uptime_str}`",
                inline=True
            )
            
            embed.add_field(
                name="🎯 Commands Executed",
                value=f"`{self.stats['commands_executed']:,}`",
                inline=True
            )
            
            embed.add_field(
                name="🛡️ Errors Handled",
                value=f"`{self.stats['errors_handled']:,}`",
                inline=True
            )
            
            # Bot info
            embed.add_field(
                name="🌐 Servers",
                value=f"`{len(self.bot.guilds):,}`",
                inline=True
            )
            
            embed.add_field(
                name="👥 Users",
                value=f"`{len(self.bot.users):,}`",
                inline=True
            )
            
            embed.add_field(
                name="📡 Latency",
                value=f"`{self.bot.latency * 1000:.1f}ms`",
                inline=True
            )
            
            if detailed:
                # Additional detailed metrics
                embed.add_field(
                    name="💾 Cache Size",
                    value=f"`{len(self.cache)} users`",
                    inline=True
                )
                
                embed.add_field(
                    name="🔧 Cogs Loaded",
                    value=f"`{len(self.bot.cogs)}`",
                    inline=True
                )
                
                embed.add_field(
                    name="⚡ Last Maintenance",
                    value=f"`{self.stats.get('last_maintenance', 'Never')}`",
                    inline=True
                )
            
            embed.set_thumbnail(url=self.bot.user.avatar.url if self.bot.user.avatar else None)
            embed.set_footer(
                text=f"Requested by {ctx.author.display_name}",
                icon_url=ctx.author.avatar.url if ctx.author.avatar else None
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Error in status command: {e}")
            await ctx.send("❌ Failed to retrieve status information.", ephemeral=True)

    @commands.hybrid_command(
        name="ping",
        description="🏓 Check bot responsiveness and latency",
        aliases=["latency", "pong"]
    )
    @commands.cooldown(1, 5, commands.BucketType.user)
    async def ping_command(self, ctx: Context) -> None:
        """Check bot latency and responsiveness"""
        try:
            start_time = datetime.utcnow()
            
            # Send initial message
            message = await ctx.send("🏓 Pinging...")
            
            # Calculate response time
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            # Create response embed
            embed = discord.Embed(
                title="🏓 Pong!",
                color=discord.Color.green(),
                timestamp=datetime.utcnow()
            )
            
            embed.add_field(
                name="📡 WebSocket Latency",
                value=f"`{self.bot.latency * 1000:.1f}ms`",
                inline=True
            )
            
            embed.add_field(
                name="⚡ Response Time",
                value=f"`{response_time:.1f}ms`",
                inline=True
            )
            
            # Determine status color based on latency
            avg_latency = (self.bot.latency * 1000 + response_time) / 2
            if avg_latency < 100:
                embed.color = discord.Color.green()
                status = "🟢 Excellent"
            elif avg_latency < 200:
                embed.color = discord.Color.orange()
                status = "🟡 Good"
            else:
                embed.color = discord.Color.red()
                status = "🔴 Poor"
            
            embed.add_field(
                name="📊 Status",
                value=status,
                inline=True
            )
            
            await message.edit(content=None, embed=embed)
            
        except Exception as e:
            self.logger.error(f"Error in ping command: {e}")
            await ctx.send("❌ Failed to measure latency.", ephemeral=True)

    @commands.hybrid_command(
        name="userinfo",
        description="👤 Get detailed information about a user",
        aliases=["user", "whois", "profile"]
    )
    @app_commands.describe(
        user="The user to get information about (defaults to yourself)"
    )
    async def userinfo_command(self, ctx: Context, user: Optional[discord.Member] = None) -> None:
        """Get comprehensive user information"""
        try:
            target_user = user or ctx.author
            
            # Create user info embed
            embed = discord.Embed(
                title=f"👤 User Information: {target_user.display_name}",
                color=target_user.color if target_user.color != discord.Color.default() else discord.Color.blue(),
                timestamp=datetime.utcnow()
            )
            
            # Basic user info
            embed.add_field(
                name="🏷️ Username",
                value=f"`{target_user.name}`",
                inline=True
            )
            
            embed.add_field(
                name="🆔 User ID",
                value=f"`{target_user.id}`",
                inline=True
            )
            
            embed.add_field(
                name="🤖 Bot",
                value="✅ Yes" if target_user.bot else "❌ No",
                inline=True
            )
            
            # Dates
            embed.add_field(
                name="📅 Account Created",
                value=f"<t:{int(target_user.created_at.timestamp())}:F>\n<t:{int(target_user.created_at.timestamp())}:R>",
                inline=False
            )
            
            if isinstance(target_user, discord.Member):
                embed.add_field(
                    name="📥 Joined Server",
                    value=f"<t:{int(target_user.joined_at.timestamp())}:F>\n<t:{int(target_user.joined_at.timestamp())}:R>",
                    inline=False
                )
                
                # Roles (limit to prevent embed overflow)
                if target_user.roles[1:]:  # Exclude @everyone
                    roles = [role.mention for role in sorted(target_user.roles[1:], reverse=True)]
                    roles_text = " ".join(roles[:10])  # Limit to 10 roles
                    if len(target_user.roles) > 11:
                        roles_text += f" ... and {len(target_user.roles) - 11} more"
                    
                    embed.add_field(
                        name=f"🎭 Roles ({len(target_user.roles) - 1})",
                        value=roles_text,
                        inline=False
                    )
            
            # Set thumbnail
            if target_user.avatar:
                embed.set_thumbnail(url=target_user.avatar.url)
            
            embed.set_footer(
                text=f"Requested by {ctx.author.display_name}",
                icon_url=ctx.author.avatar.url if ctx.author.avatar else None
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Error in userinfo command: {e}")
            await ctx.send("❌ Failed to retrieve user information.", ephemeral=True)

    # ==================== ADMIN COMMANDS ====================

    @commands.hybrid_command(
        name="maintenance",
        description="🔧 Perform maintenance operations (Admin only)",
        aliases=["maint", "cleanup"]
    )
    @commands.has_permissions(administrator=True)
    @app_commands.describe(
        operation="Type of maintenance to perform"
    )
    async def maintenance_command(
        self, 
        ctx: Context, 
        operation: str = "cache"
    ) -> None:
        """Perform various maintenance operations"""
        try:
            embed = discord.Embed(
                title="🔧 Maintenance Operation",
                color=discord.Color.blue(),
                timestamp=datetime.utcnow()
            )
            
            if operation.lower() in ["cache", "clear"]:
                # Clear cache
                cache_size = len(self.cache)
                self.cache.clear()
                self.user_cooldowns.clear()
                
                embed.description = f"✅ Cache cleared successfully!\n📊 Removed {cache_size} cached entries"
                
            elif operation.lower() in ["stats", "reset"]:
                # Reset statistics
                old_stats = self.stats.copy()
                self.stats.update({
                    "commands_executed": 0,
                    "errors_handled": 0,
                    "last_maintenance": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                })
                
                embed.description = f"✅ Statistics reset successfully!\n📊 Previous stats: {old_stats['commands_executed']} commands, {old_stats['errors_handled']} errors"
                
            else:
                embed.color = discord.Color.red()
                embed.description = f"❌ Unknown operation: `{operation}`\n💡 Available: `cache`, `stats`"
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Error in maintenance command: {e}")
            await ctx.send("❌ Maintenance operation failed.", ephemeral=True)

    # ==================== BACKGROUND TASKS ====================

    @tasks.loop(hours=1)
    async def maintenance_task(self) -> None:
        """Automated maintenance task"""
        try:
            # Clean up old cache entries (older than 24 hours)
            now = datetime.utcnow()
            expired_users = []
            
            for user_id, data in self.cache.items():
                if "last_seen" in data:
                    if (now - data["last_seen"]).total_seconds() > 86400:  # 24 hours
                        expired_users.append(user_id)
            
            for user_id in expired_users:
                del self.cache[user_id]
            
            # Clean up old cooldowns
            expired_cooldowns = []
            for user_id, timestamp in self.user_cooldowns.items():
                if (now - timestamp).total_seconds() > 3600:  # 1 hour
                    expired_cooldowns.append(user_id)
            
            for user_id in expired_cooldowns:
                del self.user_cooldowns[user_id]
            
            self.stats["last_maintenance"] = now.strftime("%Y-%m-%d %H:%M:%S UTC")
            self.logger.info(f"🧹 Maintenance completed: removed {len(expired_users)} cache entries, {len(expired_cooldowns)} cooldowns")
            
        except Exception as e:
            self.logger.error(f"Error in maintenance task: {e}")

    @tasks.loop(minutes=5)
    async def stats_updater(self) -> None:
        """Update runtime statistics"""
        try:
            # This could be used to update database statistics, send metrics to monitoring services, etc.
            self.logger.debug(f"📊 Stats update: {self.stats['commands_executed']} commands executed")
        except Exception as e:
            self.logger.error(f"Error in stats updater: {e}")

    @maintenance_task.before_loop
    async def before_maintenance(self) -> None:
        """Wait for bot to be ready before starting maintenance"""
        await self.bot.wait_until_ready()

    @stats_updater.before_loop
    async def before_stats_updater(self) -> None:
        """Wait for bot to be ready before starting stats updater"""
        await self.bot.wait_until_ready()

    # ==================== EVENT LISTENERS ====================

    @commands.Cog.listener()
    async def on_guild_join(self, guild: discord.Guild) -> None:
        """Handle bot joining a new guild"""
        self.logger.info(f"🎉 Joined new guild: {guild.name} (ID: {guild.id})")
        
        # Find a suitable channel to send welcome message
        for channel in guild.text_channels:
            if channel.permissions_for(guild.me).send_messages:
                embed = discord.Embed(
                    title="👋 Hello there!",
                    description=f"Thanks for adding me to **{guild.name}**!\n\nUse `/status` to see what I can do.",
                    color=discord.Color.green()
                )
                try:
                    await channel.send(embed=embed)
                    break
                except discord.HTTPException:
                    continue

    @commands.Cog.listener()
    async def on_command_completion(self, ctx: Context) -> None:
        """Log successful command completions"""
        self.logger.info(f"✅ Command '{ctx.command}' completed successfully for {ctx.author} in {ctx.guild}")

    # ==================== AUTO-COMPLETION ====================

    @maintenance_command.autocomplete('operation')
    async def maintenance_autocomplete(
        self, 
        interaction: discord.Interaction, 
        current: str
    ) -> List[app_commands.Choice[str]]:
        """Auto-completion for maintenance operations"""
        operations = [
            ("Clear Cache", "cache"),
            ("Reset Statistics", "stats"),
            ("Full Cleanup", "full")
        ]
        
        return [
            app_commands.Choice(name=name, value=value)
            for name, value in operations
            if current.lower() in name.lower()
        ][:25]  # Discord limit


# ==================== COG SETUP ====================

async def setup(bot: Bot) -> None:
    """
    Load the RiskRaider Template cog
    
    This function is called by Discord.py when loading the cog.
    It performs any necessary setup and adds the cog to the bot.
    """
    try:
        # Ensure logging is configured
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add the cog to the bot
        await bot.add_cog(RiskRaiderTemplate(bot))
        
        # Log successful loading
        logger = logging.getLogger(__name__)
        logger.info("🚀 RiskRaider Template cog loaded successfully")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Failed to load RiskRaider Template cog: {e}")
        raise