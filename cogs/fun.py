import random
import asyncio
import time
from typing import Optional

import aiohttp
import discord
from discord.ext import commands
from discord.ext.commands import Context


class Choice(discord.ui.View):
    def __init__(self) -> None:
        super().__init__(timeout=60.0)
        self.value = None

    @discord.ui.button(label="Heads", style=discord.ButtonStyle.blurple, emoji="🪙")
    async def heads_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        self.value = "heads"
        await interaction.response.defer()
        self.stop()

    @discord.ui.button(label="Tails", style=discord.ButtonStyle.blurple, emoji="🔄")
    async def tails_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        self.value = "tails"
        await interaction.response.defer()
        self.stop()

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True


class NumberGuessingView(discord.ui.View):
    def __init__(self, target_number: int, max_attempts: int = 5) -> None:
        super().__init__(timeout=120.0)
        self.target_number = target_number
        self.attempts = 0
        self.max_attempts = max_attempts
        self.game_over = False

    @discord.ui.button(label="Guess a Number (1-100)", style=discord.ButtonStyle.green, emoji="🎯")
    async def guess_number(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        if self.game_over:
            return
            
        modal = NumberGuessModal(self)
        await interaction.response.send_modal(modal)

    async def on_timeout(self) -> None:
        self.game_over = True
        for item in self.children:
            item.disabled = True


class NumberGuessModal(discord.ui.Modal, title="Number Guessing Game"):
    def __init__(self, view: NumberGuessingView) -> None:
        super().__init__()
        self.view = view

    number_input = discord.ui.TextInput(
        label="Enter your guess (1-100)",
        placeholder="Type a number between 1 and 100...",
        min_length=1,
        max_length=3,
    )

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            guess = int(self.number_input.value)
            if guess < 1 or guess > 100:
                await interaction.response.send_message("Please enter a number between 1 and 100!", ephemeral=True)
                return
        except ValueError:
            await interaction.response.send_message("Please enter a valid number!", ephemeral=True)
            return

        self.view.attempts += 1
        embed = discord.Embed(color=0xBEBEFE)
        embed.set_author(name=f"{interaction.user.name}'s Number Game", icon_url=interaction.user.display_avatar.url)

        if guess == self.view.target_number:
            embed.description = f"🎉 **Congratulations!** You guessed the number {self.view.target_number} in {self.view.attempts} attempts!"
            embed.color = 0x57F287
            self.view.game_over = True
            self.view.clear_items()
        elif self.view.attempts >= self.view.max_attempts:
            embed.description = f"💀 **Game Over!** The number was {self.view.target_number}. Better luck next time!"
            embed.color = 0xE02B2B
            self.view.game_over = True
            self.view.clear_items()
        else:
            hint = "📈 Too high!" if guess > self.view.target_number else "📉 Too low!"
            remaining = self.view.max_attempts - self.view.attempts
            embed.description = f"{hint} You have {remaining} attempts remaining."
            embed.color = 0xF59E42

        await interaction.response.edit_message(embed=embed, view=self.view)


class TicTacToeButton(discord.ui.Button):
    def __init__(self, x: int, y: int) -> None:
        super().__init__(style=discord.ButtonStyle.secondary, label="\u200b", row=y)
        self.x = x
        self.y = y

    async def callback(self, interaction: discord.Interaction) -> None:
        view: TicTacToeView = self.view
        if view.current_player != interaction.user:
            await interaction.response.send_message("It's not your turn!", ephemeral=True)
            return

        if view.board[self.y][self.x] != 0:
            await interaction.response.send_message("That position is already taken!", ephemeral=True)
            return

        view.board[self.y][self.x] = view.current_player_symbol
        self.label = "X" if view.current_player_symbol == 1 else "O"
        self.style = discord.ButtonStyle.danger if view.current_player_symbol == 1 else discord.ButtonStyle.primary
        self.disabled = True

        winner = view.check_winner()
        if winner:
            embed = discord.Embed(
                title="🎉 Game Over!",
                description=f"{view.current_player.mention} wins!",
                color=0x57F287
            )
            view.disable_all_buttons()
        elif view.is_board_full():
            embed = discord.Embed(
                title="🤝 Game Over!",
                description="It's a tie!",
                color=0xF59E42
            )
            view.disable_all_buttons()
        else:
            view.switch_player()
            embed = discord.Embed(
                title="Tic Tac Toe",
                description=f"Current turn: {view.current_player.mention} ({'X' if view.current_player_symbol == 1 else 'O'})",
                color=0xBEBEFE
            )

        await interaction.response.edit_message(embed=embed, view=view)


class TicTacToeView(discord.ui.View):
    def __init__(self, player1: discord.Member, player2: discord.Member) -> None:
        super().__init__(timeout=300.0)
        self.player1 = player1
        self.player2 = player2
        self.current_player = player1
        self.current_player_symbol = 1  # 1 for X, 2 for O
        self.board = [[0 for _ in range(3)] for _ in range(3)]

        for y in range(3):
            for x in range(3):
                self.add_item(TicTacToeButton(x, y))

    def switch_player(self) -> None:
        self.current_player = self.player2 if self.current_player == self.player1 else self.player1
        self.current_player_symbol = 2 if self.current_player_symbol == 1 else 1

    def check_winner(self) -> bool:
        # Check rows, columns, and diagonals
        for i in range(3):
            if all(self.board[i][j] == self.current_player_symbol for j in range(3)):
                return True
            if all(self.board[j][i] == self.current_player_symbol for j in range(3)):
                return True
        
        if all(self.board[i][i] == self.current_player_symbol for i in range(3)):
            return True
        if all(self.board[i][2-i] == self.current_player_symbol for i in range(3)):
            return True
        
        return False

    def is_board_full(self) -> bool:
        return all(self.board[i][j] != 0 for i in range(3) for j in range(3))

    def disable_all_buttons(self) -> None:
        for item in self.children:
            item.disabled = True

    async def on_timeout(self) -> None:
        self.disable_all_buttons()


class RockPaperScissors(discord.ui.Select):
    def __init__(self) -> None:
        options = [
            discord.SelectOption(
                label="Rock", description="Crushes scissors.", emoji="🪨"
            ),
            discord.SelectOption(
                label="Paper", description="Covers rock.", emoji="📄"
            ),
            discord.SelectOption(
                label="Scissors", description="Cuts paper.", emoji="✂️"
            ),
        ]
        super().__init__(
            placeholder="Choose your weapon...",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        choices = {
            "rock": 0,
            "paper": 1,
            "scissors": 2,
        }
        emojis = {"rock": "🪨", "paper": "📄", "scissors": "✂️"}
        
        user_choice = self.values[0].lower()
        user_choice_index = choices[user_choice]

        bot_choice = random.choice(list(choices.keys()))
        bot_choice_index = choices[bot_choice]

        result_embed = discord.Embed(color=0xBEBEFE)
        result_embed.set_author(
            name=f"{interaction.user.name} vs Bot", 
            icon_url=interaction.user.display_avatar.url
        )

        winner = (3 + user_choice_index - bot_choice_index) % 3
        user_emoji = emojis[user_choice]
        bot_emoji = emojis[bot_choice]
        
        if winner == 0:
            result_embed.description = f"🤝 **Draw!**\n{user_emoji} You: {user_choice.title()}\n{bot_emoji} Bot: {bot_choice.title()}"
            result_embed.colour = 0xF59E42
        elif winner == 1:
            result_embed.description = f"🎉 **You Won!**\n{user_emoji} You: {user_choice.title()}\n{bot_emoji} Bot: {bot_choice.title()}"
            result_embed.colour = 0x57F287
        else:
            result_embed.description = f"💀 **You Lost!**\n{user_emoji} You: {user_choice.title()}\n{bot_emoji} Bot: {bot_choice.title()}"
            result_embed.colour = 0xE02B2B

        await interaction.response.edit_message(
            embed=result_embed, content=None, view=None
        )


class RockPaperScissorsView(discord.ui.View):
    def __init__(self) -> None:
        super().__init__(timeout=60.0)
        self.add_item(RockPaperScissors())

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True


class QuizView(discord.ui.View):
    def __init__(self, question_data: dict) -> None:
        super().__init__(timeout=30.0)
        self.question_data = question_data
        self.answered = False
        
        # Add answer buttons
        answers = question_data["incorrect_answers"] + [question_data["correct_answer"]]
        random.shuffle(answers)
        
        for i, answer in enumerate(answers):
            button = discord.ui.Button(
                label=f"{chr(65+i)}) {answer[:50]}{'...' if len(answer) > 50 else ''}",
                style=discord.ButtonStyle.secondary,
                custom_id=str(i)
            )
            button.callback = self.answer_callback
            self.add_item(button)

    async def answer_callback(self, interaction: discord.Interaction) -> None:
        if self.answered:
            return
            
        self.answered = True
        button_clicked = interaction.data["custom_id"]
        selected_answer = None
        
        for i, item in enumerate(self.children):
            if item.custom_id == button_clicked:
                selected_answer = item.label[3:]  # Remove "A) " prefix
                if selected_answer.endswith("..."):
                    # Find the full answer
                    answers = self.question_data["incorrect_answers"] + [self.question_data["correct_answer"]]
                    for answer in answers:
                        if answer.startswith(selected_answer[:-3]):
                            selected_answer = answer
                            break
                break

        correct = selected_answer == self.question_data["correct_answer"]
        
        embed = discord.Embed(
            title="🧠 Quiz Result",
            color=0x57F287 if correct else 0xE02B2B
        )
        
        if correct:
            embed.description = f"🎉 **Correct!**\n\n**Question:** {self.question_data['question']}\n**Answer:** {self.question_data['correct_answer']}"
        else:
            embed.description = f"❌ **Incorrect!**\n\n**Question:** {self.question_data['question']}\n**Your Answer:** {selected_answer}\n**Correct Answer:** {self.question_data['correct_answer']}"
        
        # Disable all buttons
        for item in self.children:
            item.disabled = True
            if hasattr(item, 'label') and self.question_data['correct_answer'] in item.label:
                item.style = discord.ButtonStyle.success
            elif hasattr(item, 'label') and selected_answer in item.label and not correct:
                item.style = discord.ButtonStyle.danger

        await interaction.response.edit_message(embed=embed, view=self)

    async def on_timeout(self) -> None:
        self.answered = True
        for item in self.children:
            item.disabled = True


class Fun(commands.Cog, name="fun"):
    def __init__(self, bot) -> None:
        self.bot = bot

    @commands.hybrid_command(name="randomfact", description="Get a random interesting fact.")
    async def randomfact(self, context: Context) -> None:
        """
        Get a random interesting fact from multiple sources.

        :param context: The hybrid command context.
        """
        facts_apis = [
            "https://uselessfacts.jsph.pl/random.json?language=en",
            "https://api.api-ninjas.com/v1/facts",
        ]
        
        async with aiohttp.ClientSession() as session:
            for api_url in facts_apis:
                try:
                    headers = {}
                    if "api-ninjas" in api_url:
                        headers = {"X-Api-Key": "YOUR_API_KEY"}  # Replace with actual key if available
                    
                    async with session.get(api_url, headers=headers, timeout=5) as request:
                        if request.status == 200:
                            data = await request.json()
                            
                            if "uselessfacts" in api_url:
                                fact_text = data["text"]
                            elif "api-ninjas" in api_url and isinstance(data, list) and len(data) > 0:
                                fact_text = data[0]["fact"]
                            else:
                                continue
                                
                            embed = discord.Embed(
                                title="🧠 Random Fact",
                                description=fact_text,
                                color=0xD75BF4
                            )
                            embed.set_footer(text="💡 Learn something new every day!")
                            await context.send(embed=embed)
                            return
                except:
                    continue
            
            # Fallback facts if APIs fail
            fallback_facts = [
                "Octopuses have three hearts and blue blood!",
                "Honey never spoils - archaeologists have found edible honey in ancient Egyptian tombs!",
                "A group of flamingos is called a 'flamboyance'!",
                "Bananas are berries, but strawberries aren't!",
                "The shortest war in history lasted only 38-45 minutes!",
            ]
            
            embed = discord.Embed(
                title="🧠 Random Fact",
                description=random.choice(fallback_facts),
                color=0xD75BF4
            )
            embed.set_footer(text="💡 Backup fact - APIs unavailable!")
            await context.send(embed=embed)

    @commands.hybrid_command(name="coinflip", description="Flip a coin and test your luck!")
    async def coinflip(self, context: Context) -> None:
        """
        Make a coin flip with an interactive betting system.

        :param context: The hybrid command context.
        """
        buttons = Choice()
        embed = discord.Embed(
            title="🪙 Coin Flip Game",
            description="Place your bet! Will it be heads or tails?",
            color=0xBEBEFE
        )
        embed.set_footer(text="You have 60 seconds to decide!")
        
        message = await context.send(embed=embed, view=buttons)
        await buttons.wait()
        
        if buttons.value is None:
            timeout_embed = discord.Embed(
                title="⏰ Time's Up!",
                description="You took too long to decide. Try again!",
                color=0xE02B2B
            )
            await message.edit(embed=timeout_embed, view=None)
            return
        
        # Add suspense
        thinking_embed = discord.Embed(
            title="🪙 Flipping coin...",
            description="The coin is spinning through the air...",
            color=0xF59E42
        )
        await message.edit(embed=thinking_embed, view=None)
        await asyncio.sleep(2)
        
        result = random.choice(["heads", "tails"])
        result_emoji = "🪙" if result == "heads" else "🔄"
        
        if buttons.value == result:
            embed = discord.Embed(
                title="🎉 You Won!",
                description=f"{result_emoji} The coin landed on **{result}**!\nYou guessed **{buttons.value}** - Congratulations!",
                color=0x57F287,
            )
        else:
            embed = discord.Embed(
                title="💀 You Lost!",
                description=f"{result_emoji} The coin landed on **{result}**!\nYou guessed **{buttons.value}** - Better luck next time!",
                color=0xE02B2B,
            )
        
        await message.edit(embed=embed, view=None)

    @commands.hybrid_command(name="rps", description="Play rock, paper, scissors against the bot!")
    async def rock_paper_scissors(self, context: Context) -> None:
        """
        Play the classic rock paper scissors game against the bot.

        :param context: The hybrid command context.
        """
        embed = discord.Embed(
            title="🎮 Rock Paper Scissors",
            description="Choose your weapon wisely! The bot is ready to battle.",
            color=0xBEBEFE
        )
        embed.set_footer(text="Rock crushes Scissors • Paper covers Rock • Scissors cuts Paper")
        
        view = RockPaperScissorsView()
        await context.send(embed=embed, view=view)

    @commands.hybrid_command(name="guess", description="Play a number guessing game!")
    async def number_guessing_game(self, context: Context, max_number: Optional[int] = 100) -> None:
        """
        Play a number guessing game with the bot.

        :param context: The hybrid command context.
        :param max_number: Maximum number to guess (default: 100)
        """
        if max_number < 10 or max_number > 1000:
            max_number = 100
            
        target_number = random.randint(1, max_number)
        max_attempts = min(7, max(3, max_number // 20))
        
        embed = discord.Embed(
            title="🎯 Number Guessing Game",
            description=f"I'm thinking of a number between 1 and {max_number}!\nYou have {max_attempts} attempts to guess it.",
            color=0xBEBEFE
        )
        embed.set_footer(text="Click the button below to make your guess!")
        
        view = NumberGuessingView(target_number, max_attempts)
        await context.send(embed=embed, view=view)

    @commands.hybrid_command(name="tictactoe", description="Play tic-tac-toe with another user!")
    async def tic_tac_toe(self, context: Context, opponent: discord.Member) -> None:
        """
        Start a tic-tac-toe game with another user.

        :param context: The hybrid command context.
        :param opponent: The user to play against.
        """
        if opponent == context.author:
            await context.send("❌ You can't play against yourself!")
            return
        
        if opponent.bot:
            await context.send("❌ You can't play against a bot!")
            return

        embed = discord.Embed(
            title="⭕ Tic Tac Toe",
            description=f"{context.author.mention} (X) vs {opponent.mention} (O)\n\nCurrent turn: {context.author.mention}",
            color=0xBEBEFE
        )
        embed.set_footer(text="First to get 3 in a row wins!")
        
        view = TicTacToeView(context.author, opponent)
        await context.send(embed=embed, view=view)

    @commands.hybrid_command(name="quiz", description="Take a random trivia quiz!")
    async def quiz(self, context: Context, category: Optional[str] = None) -> None:
        """
        Get a random trivia question to test your knowledge.

        :param context: The hybrid command context.
        :param category: Quiz category (optional)
        """
        categories = {
            "general": 9, "books": 10, "film": 11, "music": 12, "tv": 14,
            "games": 15, "science": 17, "computers": 18, "math": 19,
            "nature": 17, "sports": 21, "geography": 22, "history": 23,
            "art": 25, "celebrities": 26, "animals": 27
        }
        
        api_url = "https://opentdb.com/api.php?amount=1&type=multiple"
        if category and category.lower() in categories:
            api_url += f"&category={categories[category.lower()]}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(api_url, timeout=10) as request:
                    if request.status == 200:
                        data = await request.json()
                        if data["results"]:
                            question_data = data["results"][0]
                            
                            # Decode HTML entities
                            import html
                            question_data["question"] = html.unescape(question_data["question"])
                            question_data["correct_answer"] = html.unescape(question_data["correct_answer"])
                            question_data["incorrect_answers"] = [html.unescape(ans) for ans in question_data["incorrect_answers"]]
                            
                            embed = discord.Embed(
                                title=f"🧠 {question_data['category']} Quiz",
                                description=f"**Difficulty:** {question_data['difficulty'].title()}\n\n**Question:**\n{question_data['question']}",
                                color=0x3498db
                            )
                            embed.set_footer(text="You have 30 seconds to answer!")
                            
                            view = QuizView(question_data)
                            await context.send(embed=embed, view=view)
                            return
            except:
                pass
        
        # Fallback quiz questions
        fallback_questions = [
            {
                "category": "General Knowledge",
                "difficulty": "medium",
                "question": "What is the largest planet in our solar system?",
                "correct_answer": "Jupiter",
                "incorrect_answers": ["Saturn", "Neptune", "Uranus"]
            },
            {
                "category": "Science",
                "difficulty": "easy",
                "question": "What gas do plants absorb from the atmosphere?",
                "correct_answer": "Carbon Dioxide",
                "incorrect_answers": ["Oxygen", "Nitrogen", "Hydrogen"]
            }
        ]
        
        question_data = random.choice(fallback_questions)
        embed = discord.Embed(
            title=f"🧠 {question_data['category']} Quiz",
            description=f"**Difficulty:** {question_data['difficulty'].title()}\n\n**Question:**\n{question_data['question']}",
            color=0x3498db
        )
        embed.set_footer(text="Fallback question - API unavailable!")
        
        view = QuizView(question_data)
        await context.send(embed=embed, view=view)

    @commands.hybrid_command(name="8ball", description="Ask the magic 8-ball a question!")
    async def magic_8ball(self, context: Context, *, question: str) -> None:
        """
        Ask the magic 8-ball a yes/no question.

        :param context: The hybrid command context.
        :param question: Your question for the 8-ball.
        """
        responses = [
            "🎱 It is certain.", "🎱 It is decidedly so.", "🎱 Without a doubt.",
            "🎱 Yes definitely.", "🎱 You may rely on it.", "🎱 As I see it, yes.",
            "🎱 Most likely.", "🎱 Outlook good.", "🎱 Yes.", "🎱 Signs point to yes.",
            "🎱 Reply hazy, try again.", "🎱 Ask again later.", "🎱 Better not tell you now.",
            "🎱 Cannot predict now.", "🎱 Concentrate and ask again.",
            "🎱 Don't count on it.", "🎱 My reply is no.", "🎱 My sources say no.",
            "🎱 Outlook not so good.", "🎱 Very doubtful."
        ]
        
        embed = discord.Embed(
            title="🔮 Magic 8-Ball",
            color=0x800080
        )
        embed.add_field(name="❓ Question", value=question, inline=False)
        embed.add_field(name="🎱 Answer", value="*Shaking the magic 8-ball...*", inline=False)
        
        message = await context.send(embed=embed)
        await asyncio.sleep(2)
        
        answer = random.choice(responses)
        embed.set_field_at(1, name="🎱 Answer", value=answer, inline=False)
        
        # Color based on answer type
        if any(word in answer.lower() for word in ["yes", "certain", "definitely", "good"]):
            embed.color = 0x57F287
        elif any(word in answer.lower() for word in ["no", "don't", "doubtful", "not"]):
            embed.color = 0xE02B2B
        else:
            embed.color = 0xF59E42
            
        await message.edit(embed=embed)

    @commands.hybrid_command(name="funhelp", description="Get help with all fun commands!")
    async def fun_help(self, context: Context) -> None:
        """
        Display all available fun commands with descriptions.

        :param context: The hybrid command context.
        """
        embed = discord.Embed(
            title="🎮 Fun Commands Help",
            description="Here are all the fun games and activities you can enjoy!",
            color=0x9932cc
        )
        
        commands_info = [
            ("🪙 `/coinflip`", "Interactive coin flip game"),
            ("🎯 `/guess [max_number]`", "Number guessing game (1-100 by default)"),
            ("🎮 `/rps`", "Rock Paper Scissors against the bot"),
            ("⭕ `/tictactoe @user`", "Play Tic-Tac-Toe with another user"),
            ("🧠 `/quiz [category]`", "Take a trivia quiz"),
            ("🔮 `/8ball <question>`", "Ask the magic 8-ball"),
            ("💡 `/randomfact`", "Get an interesting random fact"),
            ("❓ `/funhelp`", "Show this help message")
        ]
        
        for cmd, desc in commands_info:
            embed.add_field(name=cmd, value=desc, inline=False)
        
        embed.set_footer(text="🎉 Have fun playing! All games have timeouts for better performance.")
        await context.send(embed=embed)


async def setup(bot) -> None:
    await bot.add_cog(Fun(bot))