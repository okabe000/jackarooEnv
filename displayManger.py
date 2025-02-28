from rich.console import Console
from rich.layout import Layout
from rich.table import Table
import numpy as np

from loggerHandler import logger

class displayManger:
    def __init__(self, game):
        self.game = game
        self.consoleHandler = Console()
        self.available_sections = {
            'Board': self._create_board_section,
            'Obstacles': self._create_obstacles_section,
            'Balls': self._create_balls_section,
            'SafeCells': self._create_safe_cells_section,
            'TinyObMap': self._create_tiny_ob_map_section,
            'Players': self._create_players_section
        }

    def display_game(self, game, view_sections=None):
        if view_sections is None:
            view_sections = list(self.available_sections.keys())
        
        # Create base layout
        layout = Layout()
        
        # Create main layout with fixed sections
        game_area = Layout(name="game_area", size=45)
        player_area = Layout(name="player_area", size=60)
        
        # Split main layout
        if 'Players' in view_sections:
            layout.split_row(
                game_area,
                player_area
            )
        else:
            layout.update(game_area)

        # Create game area sections
        board_section = Layout(name="board_section",size=8)
        obstacles_section = Layout(name="obstacles_section", size=8)
        BallsView = Layout(name="Balls", size=80)
        info_section = Layout(name="info_section",size=10)
        # Split game area into sections
        sections_to_show = []
        if 'Board' in view_sections:
            sections_to_show.append(board_section)
        if 'Obstacles' in view_sections:
            sections_to_show.append(obstacles_section)
        if any(s in view_sections for s in [ 'SafeCells', 'TinyObMap']):
            sections_to_show.append(info_section)
            sections_to_show.append(BallsView)
            
        if sections_to_show:
            game_area.split_column(*sections_to_show)
            
            # Configure info section if needed
            if any(s in view_sections for s in ['Balls', 'SafeCells', 'TinyObMap']):
                # balls_area = Layout(name="balls_area")
                safe_cells_area = Layout(name="safe_cells_area")
                tiny_map_area = Layout(name="tiny_map_area")
                
                info_sections = []
                # if 'Balls' in view_sections:
                #     info_sections.append(balls_area)
                if 'SafeCells' in view_sections:
                    info_sections.append(safe_cells_area)
                if 'TinyObMap' in view_sections:
                    info_sections.append(tiny_map_area)
                
                if info_sections:
                    info_section.split_row(*info_sections)

        # Update game board
        game.place_balls_on_board()

        # Create and update content
        for section in view_sections:
            if section in self.available_sections:
                content = self.available_sections[section](game)
                if section == 'Board':
                    board_section.update(content)
                elif section == 'Obstacles':
                    obstacles_section.update(content)
                elif section == 'Balls':
                    BallsView.update(content)
                elif section == 'SafeCells':
                    safe_cells_area.update(content)
                elif section == 'TinyObMap':
                    tiny_map_area.update(content)
                elif section == 'Players':
                    player_area.update(content)

        # Render Layout
        self.consoleHandler.print(layout)

    def _update_layout_section(self, layout, section, content):
        try:
            if section == 'Board' and "game_info" in layout and "Board" in layout["game_info"]:
                layout["game_info"]["Board"].update(content)
            elif section == 'Obstacles' and "game_info" in layout and "Obstacles" in layout["game_info"]:
                layout["game_info"]["Obstacles"].update(content)
            elif section == 'SafeCells' and "game_info" in layout and "Balls_safeCells" in layout["game_info"]:
                layout["game_info"]["Balls_safeCells"]["safeCells"].update(content)
            elif section == 'TinyObMap' and "game_info" in layout and "Balls_safeCells" in layout["game_info"]:
                layout["game_info"]["Balls_safeCells"]["tiny_OB_map"].update(content)
            elif section == 'Players' and "player_info" in layout:
                layout["player_info"].update(content)
            elif section == 'Balls' and "game_info" in layout and "Balls_safeCells" in layout["game_info"]:
                layout["Balls"].update(content)
        except KeyError:
            logger.warning(f"Unable to update section {section}: layout section not found")

    def _create_board_section(self, game):
        return self.create_matrix_table(game.Board, title="Board")

    def _create_obstacles_section(self, game):
        return self.create_matrix_table(game.obsticles_map, title="Obstacles")

    def _create_balls_section(self, game):
        balls_table = Table(title="Balls", show_lines=True)
        balls_table.add_column("Player", width=3)
        balls_table.add_column("Ball Info", width=30)
        
        for player_id, player_balls in enumerate(game.Balls):
            ball_info = [
                f"P:{ball.pos:03d}" +
                f" OB:{ball.distance_to_obstacle:03d}" +
                f" WG:{ball.distance_to_win_gate:03d}"
                for ball in player_balls
            ]
            balls_table.add_row(f"P{player_id + 1:1d}", "\n".join(ball_info))
        return balls_table

    def _create_safe_cells_section(self, game):
        return self.create_matrix_table(
            np.array(game.safeCells).reshape(4, 4), title="Safe Cells"
        )

    def _create_tiny_ob_map_section(self, game):
        return self.create_matrix_table(
            np.array(game.tinyObMap).reshape(4, 5), title="Tiny OB Map"
        )

    def _create_players_section(self, game):
        player_table = Table(title="Players", show_lines=True)
        player_table.add_column("ID", width=2)
        player_table.add_column("Team", width=4)
        player_table.add_column("Hand", width=15)
        player_table.add_column("Action", width=25)

        for player in game.players:
            # Format hand to fixed width
            hand_str = str(player.hand)[:15].ljust(15)
            
            # Format actions to fixed width
            action_summaries = []
            if player.actions:
                for action in player.actions[:2]:  # Limit to 2 actions
                    if action:
                        verb = action.get('verb', '')[:8]  # Limit verb length
                        card = action.get('card_value', '')
                        action_summary = f"{verb:8s}({card:2})"
                    else:
                        action_summary = "None    (--)"
                    action_summaries.append(action_summary)
            else:
                action_summaries.append("None    (--)")

            player_table.add_row(
                f"{player.id:2d}",
                f"{player.team:2d}",  # Changed from :4s to :2d since team is an integer
                hand_str,
                " | ".join(action_summaries)[:25]
            )
        return player_table

    def clearConole(self):
        """
        Clear the console without disrupting the layout structure.
        """
        self.consoleHandler.clear()

    def create_matrix_table(self, matrix, title):
        """
        Create a formatted matrix table with fixed-width cells.
        """
        if isinstance(matrix, list):
            # Safeguard: Convert list to NumPy array for consistent operations
            matrix = np.array(matrix)

        # Ensure table retains its shape (no dynamic resizing)
        if matrix.ndim == 1:
            if len(matrix) == 76:
                matrix = matrix.reshape(4, 19)
            elif len(matrix) == 52:
                matrix = matrix.reshape(4, 13)

        table = Table(title=title, box=None, padding=0)
        
        # Add column labels with fixed width
        col_labels = [""] + [f"{i:2d}" for i in range(matrix.shape[1])]
        table.add_row(*col_labels)
        
        # Add rows with fixed-width cells
        for idx, row in enumerate(matrix):
            colored_row = [f"{idx:2d}"]
            for cell in row:
                cell_str = f"{cell:2}"
                if cell_str.strip() == '1':
                    colored_row.append(f"[red]{cell_str:2}[/red]")
                elif cell_str.strip() == '2':
                    colored_row.append(f"[green]{cell_str:2}[/green]")
                elif cell_str.strip() == '3':
                    colored_row.append(f"[blue]{cell_str:2}[/blue]")
                elif cell_str.strip() == '4':
                    colored_row.append(f"[yellow]{cell_str:2}[/yellow]")
                else:
                    colored_row.append(f"{cell_str:2}")
            table.add_row(*colored_row)
        
        return table
