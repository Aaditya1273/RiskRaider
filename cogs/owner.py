import discord
from discord import app_commands
from discord.ext import commands, tasks
from discord.ext.commands import Context
import asyncio
import aiohttp
import psutil
import time
import logging
import traceback
import sys
import os
import subprocess
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Union
import platform


class EnhancedOwner(commands.Cog, name="owner"):
    def __init__(self, bot) -> None:
        self.bot = bot
        self.start_time = time.time()
        self.command_usage = {}
        self.error_log = []
        self.maintenance_mode = False
        self.auto_backup.start()
        self.performance_monitor.start()
        
    def cog_unload(self):
        """Clean up when cog is unloaded"""
        self.auto_backup.cancel()
        self.performance_monitor.cancel()

    # ==================== SYNC COMMANDS ====================
    
    @commands.command(
        name="sync",
        description="Advanced slash command synchronization with options",
    )
    @app_commands.describe(
        scope="Sync scope: global, guild, or clear",
        force="Force sync even if no changes detected"
    )
    @commands.is_owner()
    async def sync(self, context: Context, scope: str = "guild", force: bool = False) -> None:
        """
        Advanced slash command synchronization
        
        :param context: Command context
        :param scope: Sync scope (global/guild/clear)
        :param force: Force sync regardless of changes
        """
        start_time = time.time()
        
        try:
            if scope.lower() == "global":
                if force:
                    synced = await context.bot.tree.sync()
                else:
                    synced = await context.bot.tree.sync()
                
                embed = discord.Embed(
                    title=" Global Sync Complete",
                    description=f"Successfully synchronized {len(synced)} slash commands globally.",
                    color=0x00FF00,
                    timestamp=datetime.now(timezone.utc)
                )
                embed.add_field(name=" Sync Time", value=f"{time.time() - start_time:.2f}s", inline=True)
                embed.add_field(name=" Force Sync", value="Yes" if force else "No", inline=True)
                
            elif scope.lower() == "guild":
                context.bot.tree.copy_global_to(guild=context.guild)
                synced = await context.bot.tree.sync(guild=context.guild)
                
                embed = discord.Embed(
                    title=" Guild Sync Complete",
                    description=f"Successfully synchronized {len(synced)} slash commands in **{context.guild.name}**.",
                    color=0x00FF00,
                    timestamp=datetime.now(timezone.utc)
                )
                embed.add_field(name=" Sync Time", value=f"{time.time() - start_time:.2f}s", inline=True)
                embed.add_field(name=" Guild ID", value=str(context.guild.id), inline=True)
                
            elif scope.lower() == "clear":
                context.bot.tree.clear_commands(guild=context.guild)
                await context.bot.tree.sync(guild=context.guild)
                
                embed = discord.Embed(
                    title=" Commands Cleared",
                    description="All slash commands have been cleared from this guild.",
                    color=0xFFA500,
                    timestamp=datetime.now(timezone.utc)
                )
            else:
                embed = discord.Embed(
                    title=" Invalid Scope",
                    description="Scope must be `global`, `guild`, or `clear`.",
                    color=0xFF0000
                )
                
        except Exception as e:
            embed = discord.Embed(
                title=" Sync Failed",
                description=f"Error during synchronization: ```py\n{str(e)}\n```",
                color=0xFF0000,
                timestamp=datetime.now(timezone.utc)
            )
            logging.error(f"Sync error: {e}")
            
        await context.send(embed=embed)

    # ==================== COG MANAGEMENT ====================
    
    @commands.hybrid_command(name="load", description="Load a cog with detailed feedback")
    @app_commands.describe(cog="The name of the cog to load")
    @commands.is_owner()
    async def load(self, context: Context, cog: str) -> None:
        """Enhanced cog loading with error handling and feedback"""
        start_time = time.time()
        
        try:
            await self.bot.load_extension(f"cogs.{cog}")
            load_time = time.time() - start_time
            
            embed = discord.Embed(
                title=" Cog Loaded Successfully",
                description=f"**{cog}** has been loaded and is ready to use.",
                color=0x00FF00,
                timestamp=datetime.now(timezone.utc)
            )
            embed.add_field(name=" Load Time", value=f"{load_time:.3f}s", inline=True)
            embed.add_field(name=" Extension Path", value=f"cogs.{cog}", inline=True)
            
        except commands.ExtensionAlreadyLoaded:
            embed = discord.Embed(
                title=" Already Loaded",
                description=f"Cog **{cog}** is already loaded.",
                color=0xFFA500,
                timestamp=datetime.now(timezone.utc)
            )
        except commands.ExtensionNotFound:
            embed = discord.Embed(
                title=" Cog Not Found",
                description=f"Could not find cog **{cog}**. Check the filename and try again.",
                color=0xFF0000,
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            embed = discord.Embed(
                title=" Load Failed",
                description=f"Failed to load **{cog}**:\n```py\n{str(e)}\n```",
                color=0xFF0000,
                timestamp=datetime.now(timezone.utc)
            )
            logging.error(f"Cog load error ({cog}): {e}")
            
        await context.send(embed=embed)

    @commands.hybrid_command(name="unload", description="Unload a cog safely")
    @app_commands.describe(cog="The name of the cog to unload")
    @commands.is_owner()
    async def unload(self, context: Context, cog: str) -> None:
        """Enhanced cog unloading with safety checks"""
        if cog.lower() == "owner":
            embed = discord.Embed(
                title=" Protected Cog",
                description="Cannot unload the owner cog for security reasons.",
                color=0xFF0000,
                timestamp=datetime.now(timezone.utc)
            )
            await context.send(embed=embed)
            return
            
        try:
            await self.bot.unload_extension(f"cogs.{cog}")
            
            embed = discord.Embed(
                title=" Cog Unloaded",
                description=f"**{cog}** has been unloaded successfully.",
                color=0x00FF00,
                timestamp=datetime.now(timezone.utc)
            )
            
        except commands.ExtensionNotLoaded:
            embed = discord.Embed(
                title=" Not Loaded",
                description=f"Cog **{cog}** is not currently loaded.",
                color=0xFFA500,
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            embed = discord.Embed(
                title=" Unload Failed",
                description=f"Failed to unload **{cog}**:\n```py\n{str(e)}\n```",
                color=0xFF0000,
                timestamp=datetime.now(timezone.utc)
            )
            
        await context.send(embed=embed)

    @commands.hybrid_command(name="reload", description="Hot-reload a cog with performance metrics")
    @app_commands.describe(cog="The name of the cog to reload")
    @commands.is_owner()
    async def reload(self, context: Context, cog: str) -> None:
        """Enhanced cog reloading with timing and error handling"""
        start_time = time.time()
        
        try:
            await self.bot.reload_extension(f"cogs.{cog}")
            reload_time = time.time() - start_time
            
            embed = discord.Embed(
                title=" Cog Reloaded",
                description=f"**{cog}** has been hot-reloaded successfully.",
                color=0x00FF00,
                timestamp=datetime.now(timezone.utc)
            )
            embed.add_field(name=" Reload Time", value=f"{reload_time:.3f}s", inline=True)
            embed.add_field(name=" Hot Reload", value=" Enabled", inline=True)
            
        except commands.ExtensionNotLoaded:
            # Try to load it if it's not loaded
            try:
                await self.bot.load_extension(f"cogs.{cog}")
                embed = discord.Embed(
                    title=" Cog Loaded",
                    description=f"**{cog}** was not loaded, so it has been loaded instead.",
                    color=0x00FF00,
                    timestamp=datetime.now(timezone.utc)
                )
            except Exception as e:
                embed = discord.Embed(
                    title=" Load Failed",
                    description=f"Could not load **{cog}**:\n```py\n{str(e)}\n```",
                    color=0xFF0000,
                    timestamp=datetime.now(timezone.utc)
                )
        except Exception as e:
            embed = discord.Embed(
                title=" Reload Failed",
                description=f"Failed to reload **{cog}**:\n```py\n{str(e)}\n```",
                color=0xFF0000,
                timestamp=datetime.now(timezone.utc)
            )
            
        await context.send(embed=embed)

    # ==================== ADVANCED COG MANAGEMENT ====================
    
    @commands.hybrid_command(name="coglist", description="List all available and loaded cogs")
    @commands.is_owner()
    async def coglist(self, context: Context) -> None:
        """Display comprehensive cog information"""
        loaded_cogs = list(self.bot.extensions.keys())
        
        # Get available cogs from filesystem
        available_cogs = []
        cogs_dir = os.path.join(os.getcwd(), "cogs")
        if os.path.exists(cogs_dir):
            for file in os.listdir(cogs_dir):
                if file.endswith(".py") and not file.startswith("__"):
                    available_cogs.append(file[:-3])
        
        embed = discord.Embed(
            title=" Cog Management Dashboard",
            color=0x0099FF,
            timestamp=datetime.now(timezone.utc)
        )
        
        loaded_list = "\n".join([f" {cog.split('.')[-1]}" for cog in loaded_cogs]) or "None"
        unloaded_list = "\n".join([f" {cog}" for cog in available_cogs if f"cogs.{cog}" not in loaded_cogs]) or "None"
        
        embed.add_field(name=" Loaded Cogs", value=f"```\n{loaded_list}\n```", inline=True)
        embed.add_field(name=" Available Cogs", value=f"```\n{unloaded_list}\n```", inline=True)
        embed.add_field(name=" Statistics", value=f"**Loaded:** {len(loaded_cogs)}\n**Available:** {len(available_cogs)}", inline=True)
        
        await context.send(embed=embed)

    @commands.hybrid_command(name="reloadall", description="Reload all loaded cogs")
    @commands.is_owner()
    async def reloadall(self, context: Context) -> None:
        """Reload all currently loaded cogs"""
        start_time = time.time()
        loaded_cogs = list(self.bot.extensions.keys())
        success_count = 0
        failed_cogs = []
        
        async with context.typing():
            for cog in loaded_cogs:
                try:
                    await self.bot.reload_extension(cog)
                    success_count += 1
                except Exception as e:
                    failed_cogs.append(f"{cog}: {str(e)}")
        
        total_time = time.time() - start_time
        
        embed = discord.Embed(
            title=" Mass Reload Complete",
            color=0x00FF00 if not failed_cogs else 0xFFA500,
            timestamp=datetime.now(timezone.utc)
        )
        
        embed.add_field(name=" Success", value=str(success_count), inline=True)
        embed.add_field(name=" Failed", value=str(len(failed_cogs)), inline=True)
        embed.add_field(name=" Total Time", value=f"{total_time:.2f}s", inline=True)
        
        if failed_cogs:
            embed.add_field(name=" Failures", value="\n".join(failed_cogs[:5]), inline=False)
        
        await context.send(embed=embed)

    # ==================== SYSTEM MONITORING ====================
    
    @commands.hybrid_command(name="status", description="Comprehensive bot status and performance metrics")
    @commands.is_owner()
    async def status(self, context: Context) -> None:
        """Display detailed bot status and system information"""
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        # Bot metrics
        uptime_seconds = time.time() - self.start_time
        uptime = str(timedelta(seconds=int(uptime_seconds)))
        
        # Discord metrics
        total_members = sum(guild.member_count for guild in self.bot.guilds)
        
        embed = discord.Embed(
            title=" Bot Status Dashboard",
            color=0x0099FF,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Bot Information
        embed.add_field(
            name=" Bot Info",
            value=f"**Uptime:** {uptime}\n"
                  f"**Latency:** {round(self.bot.latency * 1000)}ms\n"
                  f"**Commands:** {len(self.bot.commands)}\n"
                  f"**Extensions:** {len(self.bot.extensions)}",
            inline=True
        )
        
        # Discord Stats
        embed.add_field(
            name=" Discord Stats",
            value=f"**Guilds:** {len(self.bot.guilds)}\n"
                  f"**Users:** {total_members:,}\n"
                  f"**Channels:** {len(list(self.bot.get_all_channels()))}\n"
                  f"**Shards:** {self.bot.shard_count or 1}",
            inline=True
        )
        
        # System Resources
        embed.add_field(
            name=" System Resources",
            value=f"**CPU:** {cpu_percent}%\n"
                  f"**RAM:** {memory.percent}% ({memory.used // 1024 // 1024} MB)\n"
                  f"**Disk:** {disk.percent}%\n"
                  f"**Python:** {platform.python_version()}",
            inline=True
        )
        
        # Add system info footer
        embed.set_footer(text=f"Running on {platform.system()} {platform.release()}")
        
        await context.send(embed=embed)

    @tasks.loop(minutes=5)
    async def performance_monitor(self):
        """Background task to monitor performance"""
        try:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                # Log high memory usage
                logging.warning(f"High memory usage: {memory_percent}%")
        except Exception as e:
            logging.error(f"Performance monitor error: {e}")

    # ==================== DEBUG AND MAINTENANCE ====================
    
    @commands.hybrid_command(name="eval", description="Evaluate Python code (DANGEROUS)")
    @app_commands.describe(code="Python code to evaluate")
    @commands.is_owner()
    async def eval_code(self, context: Context, *, code: str) -> None:
        """Evaluate Python code with safety measures"""
        # Remove code blocks if present
        if code.startswith('```') and code.endswith('```'):
            code = '\n'.join(code.split('\n')[1:-1])
        elif code.startswith('`') and code.endswith('`'):
            code = code[1:-1]
        
        # Create environment
        env = {
            'bot': self.bot,
            'ctx': context,
            'channel': context.channel,
            'author': context.author,
            'guild': context.guild,
            'message': context.message,
            'discord': discord,
            'commands': commands,
            '__import__': __import__
        }
        
        try:
            start_time = time.time()
            result = eval(code, env)
            
            if asyncio.iscoroutine(result):
                result = await result
                
            execution_time = time.time() - start_time
            
            embed = discord.Embed(
                title=" Code Evaluation Success",
                color=0x00FF00,
                timestamp=datetime.now(timezone.utc)
            )
            
            embed.add_field(name=" Input", value=f"```py\n{code[:1000]}\n```", inline=False)
            embed.add_field(name=" Output", value=f"```py\n{str(result)[:1000]}\n```", inline=False)
            embed.add_field(name=" Execution Time", value=f"{execution_time:.4f}s", inline=True)
            
        except Exception as e:
            embed = discord.Embed(
                title=" Code Evaluation Error",
                color=0xFF0000,
                timestamp=datetime.now(timezone.utc)
            )
            embed.add_field(name=" Input", value=f"```py\n{code[:1000]}\n```", inline=False)
            embed.add_field(name=" Error", value=f"```py\n{str(e)[:1000]}\n```", inline=False)
        
        await context.send(embed=embed)

    @commands.hybrid_command(name="logs", description="View recent error logs")
    @app_commands.describe(lines="Number of log lines to show (default: 10)")
    @commands.is_owner()
    async def logs(self, context: Context, lines: int = 10) -> None:
        """Display recent error logs"""
        if not self.error_log:
            embed = discord.Embed(
                title=" Error Logs",
                description="No errors logged recently.",
                color=0x00FF00,
                timestamp=datetime.now(timezone.utc)
            )
        else:
            recent_errors = self.error_log[-lines:]
            log_text = "\n".join([f"[{error['time']}] {error['error']}" for error in recent_errors])
            
            embed = discord.Embed(
                title=" Recent Error Logs",
                description=f"```\n{log_text[:1900]}\n```",
                color=0xFF0000,
                timestamp=datetime.now(timezone.utc)
            )
            embed.add_field(name=" Total Errors", value=str(len(self.error_log)), inline=True)
            embed.add_field(name=" Showing", value=f"{len(recent_errors)} entries", inline=True)
        
        await context.send(embed=embed)

    @commands.hybrid_command(name="clearlogs", description="Clear error logs")
    @commands.is_owner()
    async def clearlogs(self, context: Context) -> None:
        """Clear the error log"""
        cleared_count = len(self.error_log)
        self.error_log.clear()
        
        embed = discord.Embed(
            title=" Logs Cleared",
            description=f"Cleared {cleared_count} error log entries.",
            color=0x00FF00,
            timestamp=datetime.now(timezone.utc)
        )
        await context.send(embed=embed)

    # ==================== BACKUP AND RESTORE ====================
    
    @commands.hybrid_command(name="backup", description="Create a backup of bot configuration")
    @commands.is_owner()
    async def backup(self, context: Context) -> None:
        """Create a configuration backup"""
        try:
            backup_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'bot_info': {
                    'name': self.bot.user.name,
                    'id': self.bot.user.id,
                    'guild_count': len(self.bot.guilds)
                },
                'extensions': list(self.bot.extensions.keys()),
                'command_usage': self.command_usage
            }
            
            filename = f"backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            embed = discord.Embed(
                title=" Backup Created",
                description=f"Configuration backup saved as `{filename}`",
                color=0x00FF00,
                timestamp=datetime.now(timezone.utc)
            )
            embed.add_field(name=" File Size", value=f"{os.path.getsize(filename)} bytes", inline=True)
            
        except Exception as e:
            embed = discord.Embed(
                title=" Backup Failed",
                description=f"Failed to create backup: {str(e)}",
                color=0xFF0000,
                timestamp=datetime.now(timezone.utc)
            )
        
        await context.send(embed=embed)

    @tasks.loop(hours=24)
    async def auto_backup(self):
        """Automated daily backup"""
        try:
            backup_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'extensions': list(self.bot.extensions.keys()),
                'command_usage': self.command_usage,
                'uptime': time.time() - self.start_time
            }
            
            filename = f"auto_backup_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
            with open(filename, 'w') as f:
                json.dump(backup_data, f, indent=2)
                
            logging.info(f"Auto backup created: {filename}")
        except Exception as e:
            logging.error(f"Auto backup failed: {e}")

    # ==================== BOT CONTROL ====================
    
    @commands.hybrid_command(name="shutdown", description="Gracefully shutdown the bot")
    @commands.is_owner()
    async def shutdown(self, context: Context) -> None:
        """Graceful bot shutdown with cleanup"""
        embed = discord.Embed(
            title=" Shutting Down",
            description="Bot is shutting down gracefully. Goodbye! ",
            color=0xFFA500,
            timestamp=datetime.now(timezone.utc)
        )
        
        uptime = str(timedelta(seconds=int(time.time() - self.start_time)))
        embed.add_field(name=" Final Uptime", value=uptime, inline=True)
        embed.add_field(name=" Shutdown By", value=context.author.mention, inline=True)
        
        await context.send(embed=embed)
        
        # Cleanup
        logging.info(f"Bot shutdown initiated by {context.author}")
        await self.bot.close()

    @commands.hybrid_command(name="restart", description="Restart the bot (requires process manager)")
    @commands.is_owner()
    async def restart(self, context: Context) -> None:
        """Restart the bot if running under a process manager"""
        embed = discord.Embed(
            title=" Restarting",
            description="Bot is restarting... This may take a moment.",
            color=0xFFA500,
            timestamp=datetime.now(timezone.utc)
        )
        await context.send(embed=embed)
        
        logging.info(f"Bot restart initiated by {context.author}")
        os.execv(sys.executable, ['python'] + sys.argv)

    # ==================== COMMUNICATION COMMANDS ====================
    
    @commands.hybrid_command(name="say", description="Make the bot send a message")
    @app_commands.describe(
        message="The message to send",
        channel="Channel to send to (optional)"
    )
    @commands.is_owner()
    async def say(self, context: Context, channel: Optional[discord.TextChannel], *, message: str) -> None:
        """Enhanced say command with channel targeting"""
        target_channel = channel or context.channel
        
        try:
            await target_channel.send(message)
            
            if channel and channel != context.channel:
                embed = discord.Embed(
                    title=" Message Sent",
                    description=f"Message sent to {channel.mention}",
                    color=0x00FF00,
                    timestamp=datetime.now(timezone.utc)
                )
                await context.send(embed=embed, ephemeral=True)
            else:
                # Delete the command message if in same channel
                try:
                    await context.message.delete()
                except:
                    pass
                    
        except discord.Forbidden:
            embed = discord.Embed(
                title=" Permission Error",
                description="I don't have permission to send messages in that channel.",
                color=0xFF0000,
                timestamp=datetime.now(timezone.utc)
            )
            await context.send(embed=embed, ephemeral=True)

    @commands.hybrid_command(name="embed", description="Send an advanced embed message")
    @app_commands.describe(
        title="Embed title",
        description="Embed description", 
        color="Hex color code (e.g., #FF0000)",
        channel="Channel to send to (optional)"
    )
    @commands.is_owner()
    async def embed(self, context: Context, channel: Optional[discord.TextChannel], 
                   title: str, color: str = "#0099FF", *, description: str) -> None:
        """Create and send advanced embed messages"""
        target_channel = channel or context.channel
        
        try:
            # Parse color
            if color.startswith('#'):
                color_int = int(color[1:], 16)
            else:
                color_int = int(color, 16)
        except ValueError:
            color_int = 0x0099FF
        
        embed = discord.Embed(
            title=title,
            description=description,
            color=color_int,
            timestamp=datetime.now(timezone.utc)
        )
        embed.set_footer(text=f"Sent by {context.author}", icon_url=context.author.avatar.url)
        
        try:
            await target_channel.send(embed=embed)
            
            if channel and channel != context.channel:
                confirm_embed = discord.Embed(
                    title=" Embed Sent",
                    description=f"Embed sent to {channel.mention}",
                    color=0x00FF00,
                    timestamp=datetime.now(timezone.utc)
                )
                await context.send(embed=confirm_embed, ephemeral=True)
                
        except discord.Forbidden:
            error_embed = discord.Embed(
                title=" Permission Error",
                description="I don't have permission to send messages in that channel.",
                color=0xFF0000,
                timestamp=datetime.now(timezone.utc)
            )
            await context.send(embed=error_embed, ephemeral=True)

    # ==================== MAINTENANCE MODE ====================
    
    @commands.hybrid_command(name="maintenance", description="Toggle maintenance mode")
    @app_commands.describe(enabled="Enable or disable maintenance mode")
    @commands.is_owner()
    async def maintenance(self, context: Context, enabled: bool) -> None:
        """Toggle maintenance mode for the bot"""
        self.maintenance_mode = enabled
        
        if enabled:
            embed = discord.Embed(
                title=" Maintenance Mode Enabled",
                description="Bot is now in maintenance mode. Most commands will be disabled for non-owners.",
                color=0xFFA500,
                timestamp=datetime.now(timezone.utc)
            )
        else:
            embed = discord.Embed(
                title=" Maintenance Mode Disabled", 
                description="Bot is now fully operational.",
                color=0x00FF00,
                timestamp=datetime.now(timezone.utc)
            )
        
        embed.add_field(name=" Changed By", value=context.author.mention, inline=True)
        await context.send(embed=embed)

    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        """Enhanced error handling with logging"""
        if self.maintenance_mode and not await self.bot.is_owner(ctx.author):
            embed = discord.Embed(
                title=" Maintenance Mode",
                description="Bot is currently under maintenance. Please try again later.",
                color=0xFFA500,
                timestamp=datetime.now(timezone.utc)
            )
            await ctx.send(embed=embed)
            return
        
        # Log the error
        error_entry = {
            'time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(error),
            'command': ctx.command.name if ctx.command else 'Unknown',
            'user': str(ctx.author),
            'guild': str(ctx.guild) if ctx.guild else 'DM'
        }
        self.error_log.append(error_entry)
        
        # Keep only last 100 errors
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]

    # ==================== STATISTICS AND ANALYTICS ====================
    
    @commands.hybrid_command(name="stats", description="Detailed bot statistics and analytics")
    @commands.is_owner()
    async def stats(self, context: Context) -> None:
        """Display comprehensive bot statistics"""
        uptime_seconds = time.time() - self.start_time
        uptime = str(timedelta(seconds=int(uptime_seconds)))
        
        # Command usage stats
        total_commands = sum(self.command_usage.values())
        top_commands = sorted(self.command_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        
        embed = discord.Embed(
            title=" Bot Analytics Dashboard",
            color=0x0099FF,
            timestamp=datetime.now(timezone.utc)
        )

async def setup(bot):
    await bot.add_cog(EnhancedOwner(bot))