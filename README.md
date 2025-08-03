# RiskRaider - Advanced NFT Risk Analysis Discord Bot


<p align="center">
  <a href="https://discord.gg/xj6y5ZaTMr"><img src="https://img.shields.io/discord/1358456011316396295?logo=discord"></a>
  <a href="https://github.com/Aaditya1273/RiskRaider/releases"><img src="https://img.shields.io/github/v/release/Aaditya1273/RiskRaider"></a>
  <a href="https://github.com/Aaditya1273/RiskRaider/commits/main"><img src="https://img.shields.io/github/last-commit/Aaditya1273/RiskRaider"></a>
  <a href="https://github.com/Aaditya1273/RiskRaider/blob/main/LICENSE.md"><img src="https://img.shields.io/github/license/Aaditya1273/RiskRaider"></a>
  <a href="https://github.com/Aaditya1273/RiskRaider"><img src="https://img.shields.io/github/languages/code-size/Aaditya1273/RiskRaider"></a>
  <a href="https://conventionalcommits.org/en/v1.0.0/"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

**RiskRaider** is an advanced Discord bot designed for comprehensive NFT wallet risk analysis and community engagement. Built with cutting-edge technology and professional-grade architecture, RiskRaider helps Discord communities make informed decisions about NFT wallet interactions while providing entertainment and moderation features.

## 🚀 Key Features

### 🔍 **Advanced NFT Risk Analysis**
- **Comprehensive Wallet Assessment** - Deep analysis using bitsCrunch API integration
- **AI-Powered Risk Scoring** - Intelligent risk assessment with 0-100 scoring system
- **Real-time Analysis** - Live wallet risk evaluation with caching for performance
- **Visual Risk Reports** - Beautiful, color-coded embeds with detailed insights
- **Connected Wallet Detection** - Identify related wallets and suspicious networks
- **Transaction Pattern Analysis** - Advanced behavioral analysis and anomaly detection

### 🎮 **Interactive Entertainment**
- **Coin Flip Game** - Interactive betting with suspenseful animations
- **Rock Paper Scissors** - Classic game with enhanced UI and emojis
- **Tic-Tac-Toe** - Multiplayer game with button-based interface
- **Number Guessing** - Customizable range with hint system
- **Trivia Quiz** - Multiple categories with timed responses
- **Magic 8-Ball** - Fortune telling with animated responses

### 🛡️ **Professional Moderation**
- **Advanced Kick/Ban System** - With reason logging and permission checks
- **Timeout Management** - Temporary restrictions with duration control
- **Message Purging** - Bulk deletion with smart filtering
- **Warning System** - Progressive discipline with tracking
- **Admin Controls** - Permission-based command access

### 📊 **Performance Monitoring**
- **Real-time Statistics** - Bot performance and system metrics
- **Command Usage Analytics** - Track popular features and usage patterns
- **Error Logging** - Comprehensive error tracking and reporting
- **Uptime Monitoring** - System health and availability tracking
- **Resource Usage** - CPU, memory, and disk utilization monitoring

### 🎨 **Beautiful UI Design**
- **Glassmorphism Effects** - Modern, translucent design elements
- **Colorful Charts** - Dynamic visualizations with Chart.js integration
- **Animated Interactions** - Smooth 60fps animations and transitions
- **Professional Embeds** - Consistent, branded message formatting
- **Mobile Responsive** - Optimized for all device types

## 🏗️ Architecture

RiskRaider is built with a modular, scalable architecture:

- **Cog-based System** - Organized, maintainable code structure
- **Async/Await** - High-performance asynchronous operations
- **Database Integration** - SQLite with migration support
- **API Integration** - bitsCrunch, OpenRouter, and external services
- **Error Handling** - Comprehensive exception management
- **Logging System** - Detailed activity and error logging
- **Rate Limiting** - Built-in protection against abuse
- **Caching** - Intelligent data caching for performance

If you use this code, please:

- Keep the credits and link to this repository
- Maintain the same license for unchanged code

See [the license file](https://github.com/Aaditya1273/RiskRaider/blob/master/LICENSE.md) for more
information, I reserve the right to take down any repository that does not meet these requirements.

## About RiskRaider

RiskRaider is a sophisticated Discord bot that combines cutting-edge NFT risk analysis with entertainment features. Built with security and user experience in mind, RiskRaider helps Discord communities make informed decisions about NFT wallet interactions.

### Key Features
- 🔍 **Advanced NFT Risk Analysis** - Comprehensive wallet risk assessment using bitsCrunch API
- 🎮 **Interactive Games** - Coin flip, rock-paper-scissors, tic-tac-toe, number guessing, and trivia
- 🛡️ **Server Moderation** - Complete moderation toolkit for Discord servers
- 📊 **Performance Monitoring** - Real-time bot statistics and system metrics
- 🤖 **AI-Powered Insights** - Smart risk summaries and recommendations
- 🎯 **Professional UI** - Beautiful embeds and interactive components

## Support

Before requesting support, you should know that this template requires you to have at least a **basic knowledge** of
Python and the library is made for **advanced users**. Do not use this template if you don't know the
basics or some advanced topics such as OOP or async. [Here's](https://pythondiscord.com/pages/resources) a link for resources to learn python.

If you need some help for something, do not hesitate to create an issue over [here](https://github.com/Aaditya1273/RiskRaider/issues), but don't forget the read the [frequently asked questions](https://github.com/Aaditya1273/RiskRaider/wiki/Frequently-Asked-Questions) before.

All the updates of the template are available [here](UPDATES.md).

## Disclaimer

Slash commands can take some time to get registered globally, so if you want to test a command you should use
the `@app_commands.guilds()` decorator so that it gets registered instantly. Example:

```py
@commands.hybrid_command(
  name="command",
  description="Command description",
)
@app_commands.guilds(discord.Object(id=GUILD_ID)) # Place your guild ID here
```

When using the template you confirm that you have read the [license](LICENSE.md) and comprehend that I can take down
your repository if you do not meet these requirements.

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+ (recommended: Python 3.11)
- Discord Developer Account
- bitsCrunch API Key (for NFT analysis)
- OpenRouter API Key (optional, for AI features)

### 🚀 Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Aaditya1273/RiskRaider.git
   cd RiskRaider
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   - Copy `.env.example` to `.env`
   - Fill in your configuration values:
   ```env
   DISCORD_TOKEN=your_discord_bot_token
   BITSCRUNCH_API_KEY=your_bitscrunch_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key  # Optional
   PREFIX=!
   INVITE_LINK=your_bot_invite_link
   ```

4. **Create Discord Application**
   - Visit [Discord Developer Portal](https://discord.com/developers/applications)
   - Create a new application
   - Go to "Bot" section and create a bot
   - Copy the token to your `.env` file
   - Enable necessary intents (Message Content Intent recommended)

5. **Get API Keys**
   - **bitsCrunch API**: Visit [bitsCrunch](https://bitscrunch.com) for NFT analysis
   - **OpenRouter API**: Visit [OpenRouter](https://openrouter.ai) for AI features (optional)

6. **Run RiskRaider**
   ```bash
   python bot.py
   ```

## 📚 Usage Guide

### Basic Commands

#### NFT Risk Analysis
```
/nft_check <wallet_address>  # Analyze NFT wallet risk
/risk_report <address>       # Detailed risk assessment
```

#### Entertainment Commands
```
/coinflip <heads|tails>      # Coin flip game
/rps <rock|paper|scissors>   # Rock Paper Scissors
/tictactoe @user             # Tic-tac-toe game
/guess <min> <max>           # Number guessing game
/trivia                      # Trivia quiz
/8ball <question>            # Magic 8-ball
```

#### Moderation Commands
```
/kick @user [reason]         # Kick user from server
/ban @user [reason]          # Ban user from server
/timeout @user <duration>    # Timeout user
/purge <amount>              # Delete messages
/warn @user <reason>         # Warn user
```

#### Utility Commands
```
/ping                        # Check bot latency
/info                        # Bot information
/help                        # Command help
/stats                       # Bot statistics
```

### 🐳 Docker Deployment

For production deployment, use Docker:

```bash
# Build and run with Docker Compose
docker compose up -d --build

# View logs
docker compose logs -f

# Stop the bot
docker compose down
```

> **Note**: The `-d` flag runs the container in detached mode (background).

### 🔧 Development Setup

For development and testing:

```bash
# Install development dependencies
pip install -r requirements.txt

# Run in development mode
python bot.py

# Enable debug logging (optional)
export LOG_LEVEL=DEBUG
```

> **Note**: You may need to use `py`, `python3`, or `python3.11` depending on your Python installation.

## 🔥 Advanced Features:

### NFT Risk Analysis Engine
- **Multi-layered Risk Assessment** - Combines transaction patterns, wallet behavior, and network analysis
- **Real-time Data Processing** - Live updates from bitsCrunch API with intelligent caching
- **AI-Powered Insights** - OpenRouter integration for natural language risk summaries
- **Visual Risk Indicators** - Color-coded embeds with risk scores and recommendations

### Performance & Reliability
- **Rate Limiting** - Built-in protection against API abuse and spam
- **Error Recovery** - Graceful handling of API failures and network issues
- **Monitoring Dashboard** - Real-time bot statistics and performance metrics
- **Automatic Backups** - Database backup and recovery systems

### Security Features
- **Permission Checks** - Role-based access control for sensitive commands
- **Audit Logging** - Comprehensive logging of all bot activities
- **Safe Evaluation** - Sandboxed code execution for owner commands
- **Data Protection** - Secure handling of user data and API keys

## 🐛 Troubleshooting

### Common Issues

**Bot not responding to commands:**
- Verify bot has proper permissions in your server
- Check if Message Content Intent is enabled
- Ensure bot is online and connected

**NFT analysis not working:**
- Verify bitsCrunch API key is valid
- Check API rate limits and quotas
- Ensure wallet address format is correct

**Permission errors:**
- Bot needs Administrator permission for moderation commands
- Check role hierarchy (bot role should be higher than target users)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for changes
- Use conventional commit messages

### Feature Requests
Have an idea for RiskRaider? [Open an issue](https://github.com/Aaditya1273/RiskRaider/issues) with the `enhancement` label.

## 📞 Support & Community

- **Discord Server**: [Join our community](https://discord.gg/xj6y5ZaTMr)
- **GitHub Issues**: [Report bugs or request features](https://github.com/Aaditya1273/RiskRaider/issues)
- **Documentation**: [Wiki & Guides](https://github.com/Aaditya1273/RiskRaider/wiki)

## 📈 Versioning

We use [SemVer](http://semver.org) for versioning. For available versions, see the [releases page](https://github.com/Aaditya1273/RiskRaider/releases).

**Current Version**: 7.0.0 - RiskRaider Edition

## 🔧 Built With

- **[Python 3.12.9](https://www.python.org/)** - Core programming language
- **[discord.py 2.x](https://discordpy.readthedocs.io/)** - Discord API wrapper
- **[bitsCrunch API](https://bitscrunch.com)** - NFT data and analytics
- **[OpenRouter API](https://openrouter.ai)** - AI-powered insights
- **[SQLite](https://sqlite.org/)** - Database for persistent storage
- **[Docker](https://docker.com)** - Containerization and deployment

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details.

## 🚀 Acknowledgments

- Thanks to the Discord.py community for excellent documentation
- bitsCrunch team for providing comprehensive NFT analytics
- All contributors who help improve RiskRaider

---

<p align="center">
  <strong>Made with ❤️ by <a href="https://github.com/Aaditya1273">Aaditya Sharma</a></strong><br>
  <em>Securing NFT communities, one analysis at a time.</em>
</p>
